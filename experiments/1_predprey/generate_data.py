import os
import pathlib
import pickle as pkl
from functools import partial
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl
from jaxtyping import Float
from scipy.integrate import solve_ivp
from torchsde import sdeint

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())


def lotka_volterra(alpha, beta, delta, gamma, t, x):
    dx = alpha * x[..., 0] - beta * x[..., 0] * x[..., 1]
    dy = delta * x[..., 0] * x[..., 1] - gamma * x[..., 1]
    return torch.stack([dx, dy], dim=-1)


class LotkaVolterra:
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self) -> None:
        self.ode = partial(lotka_volterra, 2 / 3, 4 / 3, 1.0, 1.0)
        self.diff = torch.ones(1, 2) * 1e-3

    def f(self, t, y):
        return self.ode(t, y)

    def g(self, t, y):
        return self.diff


def generate_data(hz: int, tend) -> None:
    """Generate frames for the dataset."""
    pl.seed_everything(23)
    x0 = [0.9, 0.2]
    t_span = [0, tend + 15]
    t_eval = torch.linspace(t_span[0], t_span[1], (t_span[1] - t_span[0]) * hz)
    x0 = torch.as_tensor(x0, dtype=torch.float32).unsqueeze(0)
    sde_kwargs = dict(dt=1e-1, atol=1e-5, rtol=1e-5, adaptive=True)
    sol = sdeint(LotkaVolterra(), x0, t_eval, **sde_kwargs).squeeze(1)
    train_ind = t_eval <= tend
    data = dict(t=t_eval, y=sol)
    plt.plot(data["t"], data["y"], "k-")
    data["y"] += torch.randn_like(data["y"]) * 1e-2
    plt.plot(data["t"], data["y"], "ro", alpha=0.1)
    plt.show()
    data["train_t"] = data["t"][train_ind]
    data["train_y"] = data["y"][train_ind]
    data["valid_t"] = data["t"][~train_ind]
    data["valid_y"] = data["y"][~train_ind]
    with open(os.path.join(CURR_DIR, "data.pkl"), "wb") as f:
        pkl.dump(data, f)


def main():
    hz = 10
    generate_data(hz, 50)


if __name__ == "__main__":
    main()
