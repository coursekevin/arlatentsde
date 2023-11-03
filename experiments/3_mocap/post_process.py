import os
import math
import pathlib
import pickle as pkl
import sys
from scipy import stats
from functools import partial

import pytorch_lightning as pl
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rc("text", usetex=True)
torch.set_float32_matmul_precision("high")

import arlatentsde

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(os.path.join(CURR_DIR, ".."))

from plotting_config import PlotConfig  # type: ignore

from train_ctvae import HParams, setup_model, load_mocap_data, get_dataloaders


def get_latest_checkpoint(ckpt_dir: str) -> arlatentsde.CTVAE:
    with open(os.path.join(ckpt_dir, "init_state.pkl"), "rb") as f:
        init_state = pkl.load(f)

    model = setup_model(*init_state)
    # ckpt_dir = os.path.join(ckpt_dir, "checkpoints")
    last_ckpt = [f for f in os.listdir(ckpt_dir) if "ckpt" in f][-1]
    path = os.path.join(ckpt_dir, last_ckpt)
    model.load_state_dict(
        torch.load(path, map_location=torch.device("cpu"))["state_dict"]
    )
    return model


def convert_mse_to_rmse(mse, err):
    rmse = np.sqrt(mse)
    newerr = err / (2 * rmse) if mse != 0 else 0
    return rmse, newerr


def main():
    pl.seed_everything(23)
    ckpt_path = os.path.join(CURR_DIR, "ckpts", "arcta", "version_0")
    top_dir = os.path.join(CURR_DIR, "ckpts", "arcta")
    ckpt_list = [
        os.path.join(top_dir, dir) for dir in os.listdir(top_dir) if "version" in dir
    ]
    err_list = []
    ckpt_list = sorted(ckpt_list)#[::-1]
    soln_dict = {}
    for ckpt_path in ckpt_list:
        model = get_latest_checkpoint(ckpt_path)
        model.eval()

        local_filename = os.path.join(CURR_DIR, "mocap35.mat")
        data = load_mocap_data(local_filename)
        hparams = HParams.load_from_toml()
        # load dataset
        _, _, _, _, test_loader, _, _ = get_dataloaders(
            data, hparams.batch_size, hparams.M
        )
        mse = partial(model.compute_mse, 128)  # type: ignore
        for batch in test_loader:
            _, err, soln_set = mse(batch)
        err = err.mean(1)
        err_list.append(err.ravel()) # type: ignore
        soln_dict[err.mean().sqrt()] = soln_set
    err = torch.cat(err_list)
    N = len(err)
    t_val = stats.t.ppf(0.5 + 0.95 / 2, N - 1)
    pm = t_val / math.sqrt(N) * err.std()
    err, pm = convert_mse_to_rmse(err.mean(), pm)
    print(f"RMSE: {err.mean()} +/- {pm}")
    # ------------
    # plotting...
    # ------------
    test_ind = 1
    ys, _ = soln_set[test_ind]  # type: ignore
    t, y = batch  # type: ignore
    t, y = t[test_ind], y[test_ind]
    n_plots = 3
    # some cool looking trajectories
    traj_inds = y.std(0).argsort(descending=True)[:n_plots]

    # plot predictions
    fill_color = "#9ebcda"
    sample_colors = ("#8c96c6", "#8c6bb1", "#810f7c")
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    alphas = [a + 0.1 for a in alphas]
    percentiles = [0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5]

    PlotConfig.setup()
    fig, ax = plt.subplots(n_plots, figsize=PlotConfig.convert_width((6, 4)), sharex=True)

    for j in range(n_plots):
        ax[j].plot(t.cpu(), y[:, traj_inds[j]], "k-.", alpha=0.5)

    for quant, alpha in zip(percentiles, alphas):
        lb = torch.quantile(ys, 1 - quant, dim=0)
        ub = torch.quantile(ys, quant, dim=0)
        # for lb_, ub_ in zip(lb.T, ub.T):
        for j in range(n_plots):
            lb_, ub_ = lb.T[traj_inds[j]], ub.T[traj_inds[j]]
            ax[j].fill_between(t, lb_, ub_, alpha=alpha, color=fill_color)  # type: ignore

    for k in range(n_plots):
        for j, color in enumerate(sample_colors):
            ax[k].plot(t, ys[j, :, traj_inds[k]], color=color, linewidth=1.0)

    for j in range(n_plots):
        ax[j].set_ylabel(f"$x_{j}$")
        ax[j].spines["top"].set_visible(False)
        ax[j].spines["right"].set_visible(False)
    ax[-1].set_xlabel("$t (s)$")
    PlotConfig.save_fig(fig, os.path.join(CURR_DIR, "mocap_pred"))


if __name__ == "__main__":
    main()
