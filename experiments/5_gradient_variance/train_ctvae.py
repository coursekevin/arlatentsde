import argparse
import os
import sys
import math
import pathlib
import pickle as pkl
from dataclasses import dataclass, asdict
import warnings
from functools import partial

import torch
from torch import nn
from torch import Tensor
from jaxtyping import Float
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import tomli
from arlatentsde.lit_model import warmup, rsample_latent, normed_residual, vmap_safe_stratified_mc

import matplotlib


matplotlib.use("Agg")  # Use the 'Agg' renderer


import arlatentsde
from arlatentsde.types import Device
from arlatentsde import plotting
from arlatentsde.integrate import solve_sde

torch.set_float32_matmul_precision("high")
warnings.simplefilter("always", UserWarning)
CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(CURR_DIR)

from forward_model import LoggedFCLatentSDE


class DirectEncFCNN(nn.Module):
    def __init__(self, in_dim: int, layer_description: list[int], nonlinearity) -> None:
        super().__init__()
        layer_description.insert(0, in_dim)
        layers = []
        layers += [
            (nn.Linear(layer_description[i], layer_description[i + 1]), nonlinearity)
            for i in range(len(layer_description) - 1)
        ]
        layers = [mod for layer in layers for mod in layer]
        layers.append(nn.Linear(layer_description[-1], in_dim * 2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Float[Tensor, "batch c ..."]) -> Float[Tensor, "batch ..."]:
        # y = torch.cat([x[:, 0], torch.zeros_like(x[:, 0])], dim=-1)
        mu, logvar = self.layers(x[:, 0]).chunk(2, dim=-1)
        return torch.cat([mu + x[:, 0], logvar], dim=-1)


class Identity(nn.Module):
    zero: Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("zero", torch.tensor(0.0))

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return x

    def kl_divergence(self) -> Float[Tensor, ""]:
        return self.zero


def get_dataloaders(
    batch_size: int, M: int
) -> tuple[
    arlatentsde.data.EvenlySpacedTensors, DataLoader, DataLoader, Float[Tensor, ""], int
]:
    """Get the dataloader for the dataset."""
    with open(os.path.join(CURR_DIR, "data.pkl"), "rb") as f:
        data = pkl.load(f)
    dataset = arlatentsde.data.EvenlySpacedTensors(data["train_t"], data["train_y"], M)
    sampler = arlatentsde.data.TemporalSampler(dataset, batch_size, n_repeats=6)
    train_loader = DataLoader(
        dataset,
        num_workers=6,
        persistent_workers=True,
        batch_sampler=sampler,
    )
    valid_loader = DataLoader(
        TensorDataset(data["valid_t"], data["valid_y"]),
        batch_size=len(data["valid_t"]),
    )
    # todo: is this the correct num_data
    n_data = dataset.N
    dt = dataset.dt
    return dataset, train_loader, valid_loader, dt, n_data


@dataclass(frozen=True, kw_only=True)
class HParams:
    """HParams for the training script."""

    batch_size: int
    n_mc_quad: int
    M: int
    n_warmup: int
    latent_dim: int
    lr: float
    lr_sched_freq: int
    kernel_len_init: float
    kernel_var_init: float
    brownian_mu_prior: float
    brownian_sigma_prior: float
    brownian_mu_init: float
    brownian_sigma_init: float

    @classmethod
    def load_from_toml(cls, n_mc_quad: int):
        with open(os.path.join(CURR_DIR, "hparams.toml"), "rb") as fp:
            hparams = tomli.load(fp)
        hparams["n_mc_quad"] = n_mc_quad
        return cls(**hparams)


def parse_hparams(hparams: HParams, n_data: int, device: Device) -> arlatentsde.CTVAEConfig:
    quad_settings = arlatentsde.MCQuadSettings(n_samples=hparams.n_mc_quad, device=device)
    config = arlatentsde.CTVAEConfig(
        n_data=n_data,
        n_warmup=hparams.n_warmup,
        n_samples=1,
        latent_dim=hparams.latent_dim,
        lr=hparams.lr,
        lr_sched_freq=hparams.lr_sched_freq,
        quad_settings=quad_settings,
    )
    return config


def get_loggers(n_mc_quad: int):
    os.makedirs(os.path.join(CURR_DIR, "ckpts", f"arcta_{n_mc_quad}"), exist_ok=True)
    ckpt_dir = os.path.join("ckpts", f"arcta_{n_mc_quad}")
    tensorboard = pl_loggers.TensorBoardLogger(
        str(CURR_DIR), name=ckpt_dir, default_hp_metric=False
    )
    save_dir = os.path.join(
        CURR_DIR, "ckpts", f"arcta_{n_mc_quad}", f"version_{tensorboard.version}"
    )
    os.makedirs(save_dir, exist_ok=True)
    return tensorboard, save_dir


summary_dict = dict(iter_count=[], val_rmse=[], totalfe=[], batchfe=[])

class CTVAEwithLogging(arlatentsde.CTVAE):
    totalfe = 0
    batchfe = 0

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        return loss

    def compute_elbo_components(self, batch):
        t, x_win, x = batch
        _, z, dz = self.encoder(t, x_win)
        self.log("hp/kern_var", self.encoder.kernel.var.item())

        # ----------------------------------------------------------------------------
        # kl-divergence
        def sample_normed_residual(rsamples):
            eps, brownian = rsamples
            return partial(normed_residual, eps, brownian, dz, self.drift)

        n_samples, latent_dim = self.config.n_samples, self.config.latent_dim
        self.totalfe += self.config.n_samples * self.config.quad_settings.n_samples
        self.batchfe += 1
        rsamples = [
            torch.randn(n_samples, latent_dim, device=self.device),
            self.brownian(n_samples),
        ]
        rsample_mean = [ri.mean(0) for ri in rsamples]
        self.log("hp/brownian_mean", rsample_mean[1].mean())
        quad = vmap_safe_stratified_mc(
            sample_normed_residual,
            rsamples,
            rsample_mean,
            (t[0], t[-1]),
            settings=self.config.quad_settings,
        )
        N = self.config.n_data
        M = len(t)
        residual = 0.5 * quad.mean(0) / M
        kl_div = self.brownian.kl_divergence() + self.decoder.kl_divergence()
        kl_div /= N * M
        self.log("hp/residual", residual.item())
        self.log("hp/kl_div", kl_div.item())
        # ----------------------------------------------------------------------------
        # expected log-likelihood
        mu, logvar = z(t).chunk(2, dim=-1)
        z_samples = rsample_latent(n_samples, mu, logvar, self.device)
        logits = self.decoder(z_samples)
        expected_log_like = self.loglikelihood(x, logits, self.device).mean()
        self.log("hp/log_like", expected_log_like.item())
        # ----------------------------------------------------------------------------
        # elbo
        beta = warmup(self.config.n_warmup, self._iter_count)
        elbo = expected_log_like - beta * (residual + kl_div)
        return elbo, expected_log_like, kl_div, residual

    def validation_step(self, batch, batch_idx):
        t, y = batch
        ns = 128
        y0 = y[0].repeat(ns, 1)
        y0 += torch.randn_like(y0) * 1e-3
        sde_kwargs = dict(dt=1e-1, adaptive=True, rtol=1e-4, atol=1e-4, method="srk")
        soln = solve_sde(self.drift, self.brownian, y0, t, **sde_kwargs)
        mu = soln.mean(0)
        val_rmse = (mu - y).pow(2).mean().sqrt()
        self.log("batchfe", self.drift.batchfe)  # type: ignore
        self.log("totalfe", self.drift.totalfe)  # type: ignore
        self.log("hp/val_rmse", val_rmse)
        summary_dict["iter_count"].append(self.global_step)
        summary_dict["val_rmse"].append(val_rmse.item())
        summary_dict["totalfe"].append(self.totalfe)
        summary_dict["batchfe"].append(self.batchfe)


def setup_model(
    config: arlatentsde.CTVAEConfig,
    hparams: HParams,
    dt: float,
    device: Device,
) -> arlatentsde.CTVAE:
    """Using config and hparams, sets up a continuous time VAE pl model."""
    print("setup from grad variance")
    offset, scale = arlatentsde.DeepKernel.get_rescale_params(hparams.batch_size, dt)
    # encoder setup
    kernel = arlatentsde.DeepKernel(
        [1, 32, 32, 1],
        torch.nn.ReLU(),
        offset,
        scale,
        hparams.kernel_len_init,
        hparams.kernel_var_init,
    )
    enc = DirectEncFCNN(hparams.latent_dim, [32, 32], nn.ReLU())
    encoder = arlatentsde.Encoder(dt, kernel, enc)
    # latent sde
    drift = LoggedFCLatentSDE(hparams.latent_dim, [64, 64], nn.ReLU())
    ones = torch.ones(hparams.latent_dim)
    brownian = arlatentsde.priors.LogNormalPriorSetup(
        mu_prior=ones * math.log(hparams.brownian_mu_prior),
        log_sigma_prior=ones * math.log(hparams.brownian_sigma_prior),
        mu_init=ones * math.log(hparams.brownian_mu_init),
        log_sigma_init=ones * math.log(hparams.brownian_sigma_init),
    ).get_var_distn()
    # pl model
    var = torch.tensor(1e-2**2, device=device)
    loglikelihood = partial(arlatentsde.elbo.gaussian_log_density, var)
    return CTVAEwithLogging(
        config=config,
        encoder=encoder,
        decoder=Identity(),
        drift=drift,
        brownian=brownian,
        loglikelihood=loglikelihood,  # type: ignore
    )


def freeze_init_state(save_dir, args):
    """save the initial state the initial state
    when postprocessing, we should be able to call: model = setup_model(**args)
    """
    with open(os.path.join(save_dir, "init_state.pkl"), "wb") as fp:
        pkl.dump(args, fp)


def get_summarizer_callback(
    model: arlatentsde.CTVAE, dset: arlatentsde.data.EvenlySpacedTensors, save_dir: str
):
    fig_dir = os.path.join(save_dir, "progress_figs")
    os.makedirs(fig_dir, exist_ok=True)
    summarizer = plotting.ProgressSummarizer(
        50,  # 50 frames in latent space
        fig_dir,
        dset,
        model.encoder,
        model.decoder,
        model.drift,
        model.brownian,
        [
            partial(plotting.plot_latent_sde, False),  # type: ignore
        ],
        max_process=12,
    )
    return plotting.ProgressSummarizerCallback(summary_freq=200, summarizer=summarizer)


def main(args):
    # load hparams from hparams.toml
    hparams = HParams.load_from_toml(args.S)
    pl.seed_everything(args.random_seed)
    tensorboard, save_dir = get_loggers(hparams.n_mc_quad)
    # load dataset
    _, dataloader, valid_loader, dt, n_data = get_dataloaders(
        hparams.batch_size, hparams.M
    )
    # get device
    device = torch.device("cpu")
    # parse hparams into config
    config = parse_hparams(hparams, n_data, device)
    init_state = config, hparams, float(dt), device
    # save the init state for post_processing
    freeze_init_state(save_dir, init_state)
    model = setup_model(*init_state)
    # log hparams for tensorboard
    tensorboard.log_hyperparams(asdict(hparams))
    # setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir, save_top_k=1, monitor="hp/val_rmse"
    )
    num_epochs = 30 #max(int(5000 // len(dataloader)), 1)
    print(f"Training for {num_epochs} epochs")
    trainer = pl.Trainer(
        accelerator=device.type,
        max_epochs=num_epochs,
        logger=tensorboard,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, dataloader, valid_loader)
    with open(os.path.join(save_dir, "summary_dict.pkl"), "wb") as fp:
        pkl.dump(summary_dict, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arcta train utility")
    parser.add_argument("-rs", "--random_seed", type=int, help="random seed", required=True)
    parser.add_argument("-S", type=int, help="number of random samples", required=True)
    main(parser.parse_args())
