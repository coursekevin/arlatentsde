import os
import pathlib
import pickle as pkl
from functools import partial
import matplotlib.pyplot as plt
from dataclasses import dataclass

import torch
from torch import Tensor
import pytorch_lightning as pl
from torchsde import sdeint

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())


@dataclass
class PredPreySetup:
    # birth rate of prey
    alpha: tuple[float, float]
    # predation rate by predator 1
    beta: tuple[float, float]
    # predation rate by predator 2
    gamma: tuple[float, float]
    # death rates of predators
    delta: tuple[float, float]
    # conversion rate of prey1 into predator
    epsilon: tuple[float, float]
    # conversion rate of prey2 in predator
    psi: tuple[float, float]
    # interation between predators
    nu: tuple[float, float]
    # carrying capacity
    k: tuple[float, float]


def four_dim_pred_prey(setup: PredPreySetup, t, state):
    """
    four dimensional predator-prey model
    """
    x1, x2, y1, y2 = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
    dx1 = x1 * (setup.alpha[0] - setup.beta[0] * y1 - setup.gamma[0] * y2)
    dx1 = dx1 * (1 - x1 / setup.k[0])
    dx2 = x2 * (setup.alpha[1] - setup.beta[1] * y1 - setup.gamma[1] * y2)
    dx2 = dx2 * (1 - x2 / setup.k[1])
    dy1 = y1 * (
        -setup.delta[0] + setup.epsilon[0] * x1 + setup.psi[0] * x2 - setup.nu[0] * y2
    )
    dy2 = y2 * (
        -setup.delta[1] + setup.epsilon[1] * x1 + setup.psi[1] * x2 + setup.nu[1] * y1
    )
    return torch.stack([dx1, dx2, dy1, dy2], dim=-1)


class FourDimPredPrey:
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self) -> None:
        setup = PredPreySetup(
            alpha=(0.3, 0.3),
            beta=(0.6, 0.3),
            gamma=(0.3, 0.6),
            delta=(1.0, 1.0),
            epsilon=(0.6, 0.3),
            psi=(0.3, 0.6),
            nu=(0.05, 0.025),
            k=(1.5, 1.5),
        )
        self.ode = partial(four_dim_pred_prey, setup)
        self.diff = torch.ones(1, 4) * 1e-3

    def f(self, t, y):
        return self.ode(t, y)

    def g(self, t, y):
        return self.diff


def generate_data(hz: int, tend) -> None:
    """Generate frames for the dataset."""
    pl.seed_everything(23)
    x0 = [0.9, 0.8, 0.2, 0.1]
    # x0 = [0.5203, 1.2369, 0.3395, 0.4709]
    t_span = [0, tend + 50]
    t_eval = torch.linspace(t_span[0], t_span[1], (t_span[1] - t_span[0]) * hz)
    x0 = torch.as_tensor(x0, dtype=torch.float32).unsqueeze(0)
    sde_kwargs = dict(dt=1e-1, atol=1e-5, rtol=1e-5, adaptive=True, method="srk")
    sol = sdeint(FourDimPredPrey(), x0, t_eval, **sde_kwargs).squeeze(1)  # type: ignore
    train_ind = t_eval <= tend
    data = dict(t=t_eval, y=sol)
    data["y"] += torch.randn_like(data["y"]) * 1e-2
    data["train_t"] = data["t"][train_ind]
    data["train_y"] = data["y"][train_ind]
    data["valid_t"] = data["t"][~train_ind]
    data["valid_y"] = data["y"][~train_ind]
    with open(os.path.join(CURR_DIR, "data.pkl"), "wb") as f:
        pkl.dump(data, f)


def main():
    hz = 10
    generate_data(hz, 300)


if __name__ == "__main__":
    main()
