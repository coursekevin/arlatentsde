import argparse
import os
import sys
import pickle as pkl
import pathlib
from typing import Any
from functools import partial
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


torch.set_default_dtype(torch.float64)
CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(CURR_DIR)
from generate_data import DATA_DIR, ScaleTsfm
from forward_model import LorenzModel

class TorchDEQCompat(LorenzModel):
    def forward(self, t, x):
        return super().forward(x, t)

class NeuralODE(pl.LightningModule):
    _zero: Tensor

    def __init__(self, tol: float, dt: float, drift: nn.Module, x0: Tensor) -> None:
        super().__init__()
        self.drift = drift
        self.dt = dt
        self.tol = tol
        self.ode_kwargs = dict(method="dopri5", rtol=tol, atol=tol)
        # self.ode_kwargs = dict(method="rk4")
        self.register_buffer("x0", x0)
        self.register_buffer("_zero", torch.tensor([0.0]))

    def training_step(self, batch, batch_idx):
        t, y = batch
        sort_idx = torch.argsort(t)
        t = t[sort_idx]
        y = y[sort_idx]
        t = torch.cat([self._zero, t])
        pred_y = odeint(self.drift, self.x0, t, **self.ode_kwargs)[1:]  # type: ignore
        loss = (pred_y - y).pow(2).mean()
        return loss

def get_dataloaders(dpath: str):
    with open(dpath, "rb") as f:
        data = pkl.load(f)
    train_loader = DataLoader(
        TensorDataset(data["train_t"], data["train_y"]), batch_size=len(data["train_t"])
    )
    valid_loader = DataLoader(
        TensorDataset(data["test_t"], data["test_y"]), batch_size=len(data["test_t"])
    )
    dt = data["train_t"][1] - data["train_t"][0]
    return train_loader, valid_loader, dt, data["x0"]

def get_gradients(tolerance, tf_seed_dpath):
    tf, seed, dpath = tf_seed_dpath
    train_loader, valid_loader, dt, x0 = get_dataloaders(dpath)
    drift = TorchDEQCompat(seed)
    model = NeuralODE(tolerance, dt, drift, x0)
    for batch in train_loader:
        loss = model.training_step(batch, 0)
        loss.backward()
    return torch.norm(model.drift.params.grad).item()


def main():
    pl.seed_everything(23)
    dpath_list = [dpath for dpath in os.listdir(DATA_DIR) if dpath.endswith(".pkl")]
    dpath_list = [os.path.join(DATA_DIR, dpath) for dpath in dpath_list]
    experiments = []
    for dpath in dpath_list:
        seed = int(dpath.split("_")[-1].split(".")[0])
        tf = float(dpath.split("_")[-2])
        experiments.append((tf, seed, dpath))
    experiments = sorted(experiments)
    tolerance = 1e-5
    grads = process_map(partial(get_gradients, tolerance), experiments, max_workers=20)
    results = {"times": [tf for tf, _, _ in experiments], "grads": grads}
    with open(os.path.join(CURR_DIR, "adjoint_grads.pkl"), "wb") as f:
        pkl.dump(results, f)


if __name__ == "__main__":
    main()
