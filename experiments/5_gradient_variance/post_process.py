import os
import sys
import pathlib
import pickle as pkl
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import arlatentsde
from arlatentsde import elbo
from arlatentsde.integrate import solve_sde
from arlatentsde.autoencoders import squeeze_wrap

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(CURR_DIR)
sys.path.append(os.path.join(CURR_DIR, ".."))

from plotting_config import PlotConfig

from train_ctvae import setup_model, HParams

def get_metrics_data(logdir: str, model: str, metrics: List[str]) -> pd.DataFrame:
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()

    data = []
    for metric in metrics:
        metric_data = event_acc.Scalars(metric)
        for entry in metric_data:
            data.append(
                {
                    "iter": entry.step,
                    "model": model,
                    "metric": metric,
                    "value": entry.value,
                    "wall_time": entry.wall_time,
                }
            )

    return pd.DataFrame(data)


def collect_metrics(chkpts_dir: str, model_name: str) -> pd.DataFrame:
    model_path = os.path.join(chkpts_dir, model_name)
    data = []

    for version in os.listdir(model_path):
        version_path = os.path.join(model_path, version)
        if os.path.isdir(version_path):
            metrics_data = get_metrics_data(
                version_path, model_name, ["hp/val_rmse", "totalfe", "batchfe"]
            )

            # Add version column to the metrics_data
            metrics_data["version"] = version

            # Filter rows with hp/val_rmse metric and set the wall_time values for totalfe rows
            val_rmse_data = metrics_data[metrics_data["metric"] == "hp/val_rmse"]
            totalfe_data = metrics_data[metrics_data["metric"] == "totalfe"].copy()
            totalfe_data.loc[:, "wall_time"] = val_rmse_data["wall_time"].values

            # Merge the data
            combined_data = pd.concat([val_rmse_data, totalfe_data])

            # Pivot the DataFrame to have metrics as columns
            combined_data = combined_data.pivot_table(
                values="value",
                index=["model", "version", "iter", "wall_time"],
                columns="metric",
            ).reset_index()

            # Compute the minimum log_val_rmse after a certain iteration
            combined_data["hp/val_rmse"] = combined_data.groupby(["model", "version"])[
                "hp/val_rmse"
            ].cummin()

            data.append(combined_data)

    model_df = (
        pd.concat(data)
        .sort_values(by=["model", "version", "iter"])
        .reset_index(drop=True)
    )
    # Get seconds elapsed since first iteration
    model_df["wall_time"] -= model_df.groupby(["model", "version"])[
        "wall_time"
    ].transform("min")

    return model_df


# The rest of the script remains the same


@torch.no_grad()
def make_fevals_plot():
    all_data = []
    for model_name in ["arcta", "odeint_0.0001", "odeint_0.01", "odeint_1e-06"]:
        model_df = collect_metrics(os.path.join(CURR_DIR, "ckpts"), model_name)
        model_df["log_totalfe"] = np.log10(model_df.totalfe)
        model_df["log_val_rmse"] = np.log10(model_df["hp/val_rmse"])
        all_data.append(model_df)
    all_data = pd.concat(all_data)
    mapping = {
        "arcta": "ARCTA",
        "odeint_0.01": "NODE, tol=$10^{-2}$",
        "odeint_0.0001": "NODE, tol=$10^{-4}$",
        "odeint_1e-06": "NODE, tol=$10^{-6}$",
    }

    sns.set(style="ticks")
    PlotConfig.setup()
    fig = plt.figure(figsize=PlotConfig.convert_width((1, 1)))
    for j, (key, val) in enumerate(mapping.items()):
        sns.kdeplot(
            data=all_data[all_data["model"].str.startswith(key)],
            x="log_totalfe",
            y="log_val_rmse",
            color=sns.color_palette()[j],
            fill=True,
            alpha=0.3,
        )
        sns.scatterplot(
            data=all_data[all_data["model"].str.startswith(key)],
            x="log_totalfe",
            y="log_val_rmse",
            label=val,
            # marker="o",
            color=sns.color_palette()[j],
            alpha=0.6,
            s=5,
        )

    plt.xlabel("$\log_{{10}}$ NFE")
    plt.ylabel("$\log_{{10}}$ Validation RMSE")

    lgnd = plt.legend(loc="best", scatterpoints=1)

    # change the marker size manually for both lines
    for j in range(len(lgnd.legend_handles)):
        lgnd.legend_handles[j]._sizes = [50]
    # plt.savefig(
    #     os.path.join(CURR_DIR, "rmse-vs-totalfe.png"), dpi=300, bbox_inches="tight"
    # )
    PlotConfig.save_fig(fig, os.path.join(CURR_DIR, "rmse-vs-totalfe"))


def get_latest_checkpoint(ckpt_dir: str) -> arlatentsde.CTVAE:
    with open(os.path.join(ckpt_dir, "init_state.pkl"), "rb") as f:
        init_state = pkl.load(f)

    model = setup_model(*init_state)
    # ckpt_dir = os.path.join(ckpt_dir, "checkpoints")
    last_ckpt = [f for f in os.listdir(ckpt_dir) if "ckpt" in f][-1]
    path = os.path.join(ckpt_dir, last_ckpt)
    model.load_state_dict(torch.load(path)["state_dict"])
    return model


def is_matching_int(dir: str, list_num: list[int]) -> bool:
    return any(num == int(dir.split("_")[1]) for num in list_num)


@torch.no_grad()
def collect_summary_dict(batch_size: list[int]) -> pd.DataFrame:
    ckpt_dirs = [
        ckpt
        for ckpt in os.listdir(os.path.join(CURR_DIR, "ckpts"))
        if is_matching_int(ckpt, batch_size)
    ]
    df_combined = []
    for ckpt in ckpt_dirs:
        ckpt_path = os.path.join(CURR_DIR, "ckpts", ckpt)
        vdirs = [vdir for vdir in os.listdir(ckpt_path) if "version" in vdir]
        for vdir in vdirs:
            summary_path = os.path.join(ckpt_path, vdir, "summary_dict.pkl")
            with open(summary_path, "rb") as f:
                summary_dict = pkl.load(f)
            df = pd.DataFrame(summary_dict)
            df["version"] = int(vdir.split("_")[1])
            df["batch_size"] = int(ckpt.split("_")[1])
            df_combined.append(df)

    df_combined = (
        pd.concat(df_combined)
        .sort_values(by=["batch_size", "version", "iter_count"])
        .reset_index(drop=True)
    )
    df_combined["val_rmse"] = df_combined.groupby(["batch_size", "version"])[
        "val_rmse"
    ].cummin()
    df_combined["log10valrmse"] = np.log10(df_combined["val_rmse"])
    df_combined["log10totalfe"] = np.log10(df_combined["totalfe"])
    return df_combined


def make_ablation_plot(
    df: pd.DataFrame, x_val: tuple[str, str], y_val: tuple[str, str]
):
    PlotConfig.setup()
    fig = plt.figure(figsize=PlotConfig.convert_width((1, 1)))
    for j, batch_size in enumerate(df.batch_size.unique()):
        label = f"R=1, S = {batch_size}"
        sns.lineplot(
            data=df[df["batch_size"] == batch_size],
            x=x_val[0],
            y=y_val[0],
            color=sns.color_palette()[j],
            # fill=True,
            alpha=0.3,
        )
        sns.scatterplot(
            data=df[df["batch_size"] == batch_size],
            x=x_val[0],
            y=y_val[0],
            label=label,
            # marker="o",
            color=sns.color_palette()[j],
            alpha=0.6,
            s=5,
        )
        plt.xlabel(x_val[1])
        plt.ylabel(y_val[1])

        lgnd = plt.legend(loc="best", scatterpoints=1)
    name = f"{y_val[0]}vs{x_val[0]}_ablation"
    PlotConfig.save_fig(fig, os.path.join(CURR_DIR, name))


@torch.no_grad()
def make_prediction_plot():
    PlotConfig.setup()
    with open(os.path.join(CURR_DIR, "data.pkl"), "rb") as f:
        data = pkl.load(f)
    plt.plot(data["train_t"], data["train_y"].numpy(), "o", label="Train data")
    plt.plot(data["valid_t"], data["valid_y"].numpy(), "-", label="Valid data")
    ns = 128
    # colors from: https://github.com/google-research/torchsde/blob/master/examples/latent_sde.py
    fill_color = "#9ebcda"
    sample_colors = ("#8c96c6", "#8c6bb1", "#810f7c")
    y0 = data["valid_y"][0].repeat(ns, 1)
    y0 += torch.randn_like(y0) * 1e-3
    sde_kwargs = dict(dt=1e-1, adaptive=True, rtol=1e-4, atol=1e-6, method="srk")
    t = data["valid_t"]
    ckpt_path = os.path.join(CURR_DIR, "ckpts", "arcta_10", "version_0")
    # load model
    model = get_latest_checkpoint(ckpt_path)
    model.eval()
    soln = solve_sde(model.drift, model.brownian, y0, t, **sde_kwargs)
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    alphas = [a + 0.1 for a in alphas]
    percentiles = [0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5]
    fig, ax = plt.subplots(squeeze=False, figsize=PlotConfig.convert_width((1, 1)))
    ax = ax[0, 0]  # for type checking
    for quant, alpha in zip(percentiles, alphas):
        lb = torch.quantile(soln, 1 - quant, dim=0)
        ub = torch.quantile(soln, quant, dim=0)
        for lb_, ub_ in zip(lb.T, ub.T):
            ax.fill_between(t, lb_, ub_, alpha=alpha, color=fill_color)  # type: ignore
    for j, color in enumerate(sample_colors):
        ax.plot(t, soln[j].numpy(), color=color, linewidth=1.0)
    datacolor = "k"
    ax.plot(
        t,
        data["valid_y"].numpy(),
        "o",
        alpha=0.7,
        markersize=1,
        label="Valid data",
        color=datacolor,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    PlotConfig.save_fig(fig, os.path.join(CURR_DIR, "four-dim-pred-prey"))


def main():
    pl.seed_everything(23)
    # make_fevals_plot()
    make_prediction_plot()
    df = collect_summary_dict([10, 50, 100])
    make_ablation_plot(df, ("iter_count", "Iteration"), ("totalfe", "NFE"))
    make_ablation_plot(df, ("totalfe", "NFE"), ("log10valrmse", "$\log_{{10}}$ Validation RMSE"))
    make_ablation_plot(df, ("iter_count", "Iteration"), ("log10valrmse", "$\log_{{10}}$ Validation RMSE"))


if __name__ == "__main__":
    main()
