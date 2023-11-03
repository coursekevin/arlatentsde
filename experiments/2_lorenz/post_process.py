import os
import pathlib
import sys
import numpy as np
import pickle as pkl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(os.path.join(CURR_DIR, ".."))

from plotting_config import PlotConfig  # type: ignore

def get_adjoint_grads():
    with open(os.path.join(CURR_DIR, "adjoint_grads.pkl"), "rb") as handle:
        adjoint_results = pd.DataFrame.from_dict(pkl.load(handle))

    adjoint_results["method"] = "Adjoints"
    return adjoint_results


def get_arcta_grads():
    ckpt_dir = os.path.join(CURR_DIR, "ckpts", "arcta")
    version_list = [
        folder for folder in os.listdir(ckpt_dir) if folder.startswith("version")
    ]
    arcta_grads = []
    test_curves = []
    for version in version_list:
        with open(
            os.path.join(ckpt_dir, version, "experiment_log.pkl"), "rb"
        ) as handle:
            grad_data = pkl.load(handle)
            df = pd.DataFrame(grad_data, columns=["grad_norm", "tf"])
            df = df.rename(columns={"grad_norm": "grads", "tf": "times"})
            arcta_grads.append(df)
            df = pd.DataFrame(grad_data, columns=["param_norm", "tf"])
            df["iteration"] = np.linspace(0, 2000, len(df))
            test_curves.append(df)
    arcta_grads, test_curves = pd.concat(arcta_grads), pd.concat(test_curves)
    arcta_grads["method"] = "ARCTA"
    return arcta_grads, test_curves


def get_all_data():
    adj_grads = get_adjoint_grads()
    arcta_grads, test_curves = get_arcta_grads()
    combined_data = pd.concat([arcta_grads, adj_grads]).reindex()
    combined_data["log10grads"] = np.log10(combined_data["grads"])
    return combined_data, test_curves


def plot_grads(grads):
    PlotConfig.setup()
    colors = ["#8c96c6", "#8c6bb1", "#810f7c"]
    sns.set_palette(sns.color_palette(colors))
    fig = plt.figure(figsize=PlotConfig.convert_width((1, 1)))
    barplot = sns.barplot(
        data=grads, x="times", y="log10grads", hue="method", errorbar=("sd", 1.0)
    )
    # # Get the existing y-ticks
    yticks = barplot.get_yticks()
    # # Calculate the new labels
    new_labels = [f"$10^{{{int(tick)}}}$" for tick in yticks]
    # # Set the new labels
    barplot.set_yticks(yticks)
    barplot.set_yticklabels(new_labels)

    plt.xlabel("$T (s)$", fontsize=PlotConfig.MAJOR_FONT_SIZE)
    plt.ylabel("$\\log_{10} ||\\nabla_{\\theta} \\mathcal{L}||$", fontsize=PlotConfig.MAJOR_FONT_SIZE)
    # locations, labels = plt.yticks()

    lgnd = plt.legend(loc="best", scatterpoints=1, fontsize=PlotConfig.MINOR_FONT_SIZE)
    # barplot.tick_params(labelsize=15)
    # plt.savefig(os.path.join(CURR_DIR, "grads.png"), bbox_inches="tight", dpi=300)
    PlotConfig.save_fig(fig, os.path.join(CURR_DIR, "grads"))


def main():
    grads, test_curves = get_all_data()
    plot_grads(grads)


if __name__ == "__main__":
    main()
