import os
import pathlib
import pickle as pkl
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from scipy.integrate import solve_ivp
from tqdm.contrib.concurrent import process_map

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
FRAME_DIR = os.path.join(CURR_DIR, "frames")


def nonlinear_pendulum(t: float, y: Float[np.ndarray, "2"]) -> Float[np.ndarray, "2"]:
    """
    Nonlinear pendulum ODE.
    """
    return np.stack([y[1], -np.sin(y[0])])


def plot_frame(
    xlim: list[float], ylim: list[float], idx_theta: tuple[int, float, float]
) -> None:
    """Plot a single frame of the dataset."""
    idx, theta, dtheta = idx_theta
    fig, ax = plt.subplots(1, 1, squeeze=False)
    ax = ax[0, 0]
    x, y = np.sin(theta), -np.cos(theta)
    ax.plot([0, x], [0, y], "-ko", linewidth=20, markersize=1)
    ax.axis("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis("off")
    fig.savefig(os.path.join(FRAME_DIR, f"{idx:04d}.png"))  # type: ignore
    plt.close(fig)


def generate_frames() -> None:
    """Generate frames for the dataset."""
    x0 = [np.pi / 2, 0]
    t_span = [0, 30]
    fps = 15
    t_eval = np.linspace(t_span[0], t_span[1], (t_span[1] - t_span[0]) * fps)
    sol = solve_ivp(nonlinear_pendulum, t_span, x0, t_eval=t_eval, method="RK45")
    x_lim = [-1.5, 1.5]
    y_lim = [-1.5, 0.1]
    theta = sol.y.T[:, 0]
    dtheta = sol.y.T[:, 1]
    with open(os.path.join(FRAME_DIR, "t.pkl"), "wb") as f:
        pkl.dump({"t": torch.as_tensor(t_eval)}, f)
    with open(os.path.join(FRAME_DIR, "y.pkl"), "wb") as f:
        pkl.dump({"y": torch.as_tensor(sol.y.T)}, f)
    pmap_args = [(j, thj, dthj) for j, (thj, dthj) in enumerate(zip(theta, dtheta))]
    process_map(partial(plot_frame, x_lim, y_lim), pmap_args, max_workers=4)


def main():
    os.makedirs(FRAME_DIR, exist_ok=True)
    generate_frames()


if __name__ == "__main__":
    main()
