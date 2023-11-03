import argparse
import os
import sys
import pickle as pkl
import pathlib

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

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(CURR_DIR)

from forward_model import LoggedFCLatentSDE


class TorchDiffeqCompat(LoggedFCLatentSDE):
    def forward(self, t, x):
        return super().forward(x, t)


class NeuralODE(pl.LightningModule):
    def __init__(self, tol: float, dt: float, drift: nn.Module) -> None:
        super().__init__()
        self.drift = drift
        self.dt = dt
        self.tol = tol
        self.ode_kwargs = dict(method="dopri5", rtol=self.tol, atol=self.tol)

    def training_step(self, batch, batch_idx):
        y0, y = batch
        y = y.transpose(1, 0)
        t = torch.arange(y.shape[0]) * self.dt
        pred_y = odeint(self.drift, y0, t, **self.ode_kwargs)
        loss = (pred_y - y).pow(2).mean()
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        t, y = batch
        with torch.no_grad():
            x0 = y[0]
            ode_kwargs = dict(method="dopri5", rtol=self.tol, atol=self.tol)
            sol = odeint(self.drift, x0, t, **ode_kwargs)
        self.log("batchfe", self.drift.batchfe)  # type: ignore
        self.log("totalfe", self.drift.totalfe)  # type: ignore
        self.log("hp/val_rmse", (sol - y).pow(2).mean().sqrt())


class BatchedTimeSeries(Dataset):
    def __init__(self, t: Tensor, y: Tensor, int_win: int) -> None:
        super().__init__()
        self.int_win = int_win
        self.t, self.y = t, y

    def __len__(self):
        return self.t.shape[0] - self.int_win

    def __getitem__(self, idx):
        y0 = self.y[idx]
        return y0, self.y[idx : idx + self.int_win]


def get_dataloaders(batch_size: int, int_win: int):
    with open(os.path.join(CURR_DIR, "data.pkl"), "rb") as f:
        data = pkl.load(f)
    dataset = BatchedTimeSeries(data["train_t"], data["train_y"], int_win)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(
        TensorDataset(data["valid_t"], data["valid_y"]), batch_size=len(data["valid_t"])
    )
    dt = data["train_t"][1] - data["train_t"][0]
    return train_loader, valid_loader, dt


def get_loggers(tolerance: float):
    os.makedirs(os.path.join(CURR_DIR, "ckpts", "odeint"), exist_ok=True)
    ckpt_dir = os.path.join("ckpts", f"odeint_{tolerance}")
    tensorboard = pl_loggers.TensorBoardLogger(
        str(CURR_DIR), name=ckpt_dir, default_hp_metric=False
    )
    save_dir = os.path.join(
        CURR_DIR, "ckpts", f"odeint_{tolerance}", f"version_{tensorboard.version}"
    )
    os.makedirs(save_dir, exist_ok=True)
    return tensorboard, save_dir


def main(args):
    pl.seed_everything(args.random_seed)
    tensorboard, save_dir = get_loggers(args.tolerance)
    train_loader, valid_loader, dt = get_dataloaders(10, 8)
    drift = TorchDiffeqCompat(2, [64, 64, 64], nn.ReLU())
    model = NeuralODE(args.tolerance, dt, drift)
    device = torch.device("cuda")
    trainer = pl.Trainer(
        accelerator=device.type,
        log_every_n_steps=1,
        val_check_interval=0.1,
        max_epochs=5,
        logger=tensorboard,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="odeint train utility")
    parser.add_argument(
        "-rs", "--random_seed", type=int, help="random seed", required=True
    )
    parser.add_argument(
        "-tol", "--tolerance", type=float, help="tolerance", required=True
    )
    main(parser.parse_args())
