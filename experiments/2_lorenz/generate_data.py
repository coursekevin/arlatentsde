import argparse
import os
import pathlib
import numpy as np
import numpy.random as npr
from scipy.integrate import solve_ivp
from functools import partial
import pickle as pkl
import torch
from torch import nn
from torch import Tensor
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
import pytorch_lightning as pl

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
DATA_DIR = os.path.join(CURR_DIR, "data")

class LorenzModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(10.0))
        self.beta = nn.Parameter(torch.tensor(8 / 3))
        self.rho = nn.Parameter(torch.tensor(28.0))

    def forward(self, t, x):
        return torch.stack(lorenz(self.sigma, self.beta, self.rho, t, x), dim=-1)

    def grad_norm(self):
        return torch.sqrt(self.rho.grad.abs())


def lorenz(sigma, beta, rho, t, x):
    x_dot = sigma * (x[..., 1] - x[..., 0])
    y_dot = x[..., 0] * (rho - x[..., 2]) - x[..., 1]
    z_dot = x[..., 0] * x[..., 1] - beta * x[..., 2]

    return [x_dot, y_dot, z_dot]


def get_data(n_data, tf, seed):
    params = (10, 8 / 3, 28)
    ode = partial(lorenz, *params)
    # x0 = np.array([-8.0, 7.0, 27.0])
    x0 = np.array([8.0, -2.0, 36.05])
    ivp_kwargs = dict(method="RK45", rtol=1e-6, atol=1e-8)
    sol = solve_ivp(ode, (0, tf), x0, t_eval=np.linspace(0, tf, n_data), **ivp_kwargs)
    rng = npr.default_rng(seed)
    unoise = rng.normal(0, 1, (n_data, 3))
    y = sol.y.T + unoise * 1.0
    y = torch.as_tensor(y)
    t = torch.as_tensor(sol.t)
    x0 = torch.as_tensor(x0)
    return t, y, x0

class ScaleTsfm(nn.Module):
    def __init__(self, train_x: Tensor):
        super().__init__()
        self.register_buffer("mu", train_x.mean(0))
        self.register_buffer("scale", train_x.std(0))

    def forward(self, x: Tensor):
        return (x - self.mu) / self.scale

    def unscale(self, x: Tensor):
        return x * self.scale + self.mu


def main(rs, tf):
    # creating some data
    seed = rs
    pl.seed_everything(seed)
    os.makedirs(DATA_DIR, exist_ok=True)
    tf = tf # 120.0
    train_end = tf * 100.0 / 120.0 # 100.0
    n_data = int(tf / 0.005)  # 0.005 should work
    t, y_data, x0 = get_data(n_data, tf, seed)
    train_ind = (0 < t) & (t < train_end)
    test_ind = t >= train_end
    data = dict(
        train_t=t[train_ind],
        train_y=y_data[train_ind],
        scale_tsfm=ScaleTsfm(y_data[train_ind]),
        test_t=t[test_ind],
        test_y=y_data[test_ind],
        x0=torch.as_tensor(x0).float(),
    )
    fname = f"lorenz_data_{int(tf)}_{seed}.pkl"
    with open(os.path.join(DATA_DIR, fname), "wb") as f:
        pkl.dump(data, f)
    print(
        f"Created dataset with {train_ind.sum()} training points and {test_ind.sum()} test points "
    )


if __name__ == "__main__":
    rs_list = range(5)
    tf_list = [1.0, 10.0, 50.0, 100.0]
    for rs in rs_list:
        for tf in tf_list:
            main(rs, tf)
