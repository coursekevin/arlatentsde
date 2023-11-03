import os
import math
import pathlib
import pickle as pkl
import sys
from scipy import stats
from functools import partial

import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import seaborn as sns

import arlatentsde
from arlatentsde.integrate import solve_sde

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(os.path.join(CURR_DIR, ".."))

from plotting_config import PlotConfig  # type: ignore

from train_ctvae import HParams, setup_model, get_dataloaders
import numpy as np


def get_latest_checkpoint(ckpt_dir: str) -> arlatentsde.CTVAE:
    with open(os.path.join(ckpt_dir, "init_state.pkl"), "rb") as f:
        init_state = pkl.load(f)

    model = setup_model(*init_state)
    ckpt_dir = os.path.join(ckpt_dir, "checkpoints")
    last_ckpt = [f for f in os.listdir(ckpt_dir) if "ckpt" in f][-1]
    path = os.path.join(ckpt_dir, last_ckpt)
    model.load_state_dict(
        torch.load(path, map_location=torch.device("cpu"))["state_dict"]
    )
    return model

def create_rgba_image(img):
    # Create an RGB image with the light blue color
    rgb_img = np.ones((*img.shape, 3))

    # Modulate the color intensity with the original image
    rgb_img = rgb_img * img[..., np.newaxis]

    # Create an alpha channel where white pixels (value=1) are fully transparent
    alpha_channel = 1 - img

    # Stack RGB and alpha channels together
    rgba_img = np.concatenate([rgb_img, alpha_channel[..., np.newaxis]], axis=-1)

    return rgba_img




def main():
    with torch.no_grad():
        pl.seed_everything(42)

        # loading most recent model
        ckpt_path = os.path.join(CURR_DIR, "ckpts", "version_0")
        model = get_latest_checkpoint(ckpt_path)
        model.eval()

        # loading data
        hparams = HParams.load_from_toml()
        dataset, _, dt, n_data = get_dataloaders(hparams.batch_size, hparams.res, hparams.M)
        skip = 10
        n_frames = 5
        n_samples = 128
        plot_samples = 20

        PlotConfig.setup()
        gs = gridspec.GridSpec(2, n_frames)
        fig = plt.figure(figsize=PlotConfig.convert_width((n_frames * 3, 6), page_scale=1.0))
        t_eval = torch.linspace(0, 30, 300)
        tf = float(dataset[(n_frames - 1) * skip][0])
        t_test = torch.linspace(0, tf, n_frames)
        for j in range(n_frames):
            i = j * skip
            t, img_win, img = dataset[i]
            ax = fig.add_subplot(gs[0, j])
            ax.axis("off")
            if i == 0:
                # plot the latent state and get some predictions
                t, fn, _ = model.encoder(t.unsqueeze(0),img_win.unsqueeze(0))
                z_eval = fn(t)
                mu, logvar = z_eval.chunk(2, dim=-1)
                z0 = arlatentsde.elbo.rsample_latent(n_samples, mu[0], logvar[0])
                sde_kwargs = dict(dt=1e-2, adaptive=True, rtol=1e-4, atol=1e-6)
                soln = solve_sde(model.drift, model.brownian, z0, t_eval, **sde_kwargs)
                soln_test = solve_sde(model.drift, model.brownian, z0, t_test, **sde_kwargs)
                dec_test = model.decoder(soln_test)

                fill_color = "#9ebcda"
                sample_colors = ("#8c96c6", "#8c6bb1", "#810f7c")
                alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
                alphas = [a + 0.1 for a in alphas]
                percentiles = [0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5]

                ax_long = fig.add_subplot(gs[1, :])
                for quant, alpha in zip(percentiles, alphas):
                    lb = torch.quantile(soln, 1 - quant, dim=0)
                    ub = torch.quantile(soln, quant, dim=0)
                    for lb_, ub_ in zip(lb.T, ub.T):
                        ax_long.fill_between(t_eval, lb_, ub_, alpha=alpha, color=fill_color)  # type: ignore

                for k, color in enumerate(sample_colors):
                    ax_long.plot(t_eval, soln[k, :, :].numpy(), color=color, linewidth=1.0)

                ax_long.set_ylabel("$z(t)$")
                ax_long.spines['top'].set_visible(False)
                ax_long.spines['right'].set_visible(False)
                ax_long.set_xlabel("$t (s)$")
            img_plot = np.zeros_like(img.numpy())
            for k in range(plot_samples):
                img_plot +=  torch.sigmoid(dec_test[k, j]).numpy() # type: ignore
            ax.imshow(img_plot, cmap=sns.color_palette("flare", as_cmap=True), interpolation='nearest', aspect='auto') # type: ignore
            ax.imshow(create_rgba_image(1 - img.numpy()), interpolation='nearest', aspect='auto')
            # ax.imshow(create_rgba_image(1 - img_plot), cmap="Blues")
        plt.tight_layout()
        # fig.savefig(os.path.join(CURR_DIR, "sdefromvideo.png"), dpi=300, bbox_inches="tight" )
        PlotConfig.save_fig(fig, os.path.join(CURR_DIR, "sdefromvideo"))
            

        


if __name__ == "__main__":
    main()
