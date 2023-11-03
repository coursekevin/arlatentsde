import argparse
import os
import math
from scipy.io import loadmat
import pathlib
import requests
from typing import List, Dict
from dataclasses import dataclass, asdict
from functools import partial
import pickle as pkl
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import Tensor
import numpy as np
from jaxtyping import Float
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import tomli

import arlatentsde
from arlatentsde.types import Device
from arlatentsde.integrate import solve_sde
from arlatentsde import plotting

torch.set_float32_matmul_precision("high")

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
DATA_LINK = "https://www.dropbox.com/s/p75wc1j53itonuo/mocap35.mat?dl=1"


# ---- data loading utils -------------------------------------------------------------
class ScaleTsfm(nn.Module):
    def __init__(self, x: Tensor, eps: float = 1e-8) -> None:
        super().__init__()
        dim = x.shape[-1]
        self.mu = x.view(-1, dim).mean(0)
        self.scale = x.view(-1, dim).std(0) + eps

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mu).div(self.scale)

    def inverse(self, x: Tensor) -> Tensor:
        return x.mul(self.scale) + self.mu


def download_file(url: str, local_filename: str):
    """
    Dowloads the file at the given url to the given local_filename.

    Args:
        url (str): url of file
        local_filename (str): file to save to
    """
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def load_mocap_data(local_filename: str, dt: float = 0.1) -> Dict[str, torch.Tensor]:
    """Loads the mocap data from file"""
    mocap_data = loadmat(local_filename)

    # -----------------------------------------------------------------------------------
    # from: https://github.com/cagatayyildiz/ODE2VAE/blob/master/model/data/mocap_many.py
    # making sure we get the same train/test split as the original paper
    x_test = mocap_data["Xtest"]
    t_test = dt * np.arange(0, x_test.shape[1], dtype=np.float32)
    t_test = np.tile(t_test, [x_test.shape[0], 1])
    x_val = mocap_data["Xval"]
    t_val = dt * np.arange(0, x_val.shape[1], dtype=np.float32)
    t_val = np.tile(t_val, [x_val.shape[0], 1])
    x_tr = mocap_data["Xtr"]
    t_tr = dt * np.arange(0, x_tr.shape[1], dtype=np.float32)
    t_tr = np.tile(t_tr, [x_tr.shape[0], 1])
    # -----------------------------------------------------------------------------------
    data = dict(
        test_x=x_test,
        test_t=t_test,
        valid_x=x_val,
        valid_t=t_val,
        train_x=x_tr,
        train_t=t_tr,
    )
    return to_tensors(data)


def to_tensors(data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Converts all data in the given dictionary to torch tensors."""
    return_data = {}
    for key, val in data.items():
        return_data[key] = torch.from_numpy(val).float()
    return return_data


# ---- model setup utils -------------------------------------------------------------
class EncodingFCNN(nn.Module):
    def __init__(
        self,
        full_dim: int,
        n_snapshots: int,
        latent_dim: int,
        layer_description: list[int],
        nonlinearity,
    ) -> None:
        super().__init__()
        self.M = n_snapshots
        layer_description.insert(0, full_dim * n_snapshots)
        layers = []
        layers += [
            (nn.Linear(layer_description[i], layer_description[i + 1]), nonlinearity)
            for i in range(len(layer_description) - 1)
        ]
        layers = [mod for layer in layers for mod in layer]
        layers.append(nn.Linear(layer_description[-1], latent_dim * 2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Float[Tensor, "batch c ..."]) -> Float[Tensor, "batch ..."]:
        batch_size = x.shape[:-2]
        x = self.layers(x.view(*batch_size, -1))
        return x


class DecodingFCNN(nn.Module):
    _bias: torch.Tensor

    def __init__(
        self,
        full_dim: int,
        latent_dim: int,
        layer_description: list[int],
        nonlinearity,
    ) -> None:
        super().__init__()
        self.full_dim = full_dim
        layer_description.insert(0, latent_dim)
        layers = []
        layers += [
            (nn.Linear(layer_description[i], layer_description[i + 1]), nonlinearity)
            for i in range(len(layer_description) - 1)
        ]
        layers = [mod for layer in layers for mod in layer]
        layers.append(nn.Linear(layer_description[-1], full_dim))
        self.layers = nn.Sequential(*layers)
        # variance
        ones = torch.ones(full_dim)
        self.variance = arlatentsde.priors.LogNormalPriorSetup(
            mu_prior=ones * math.log(1.0),
            log_sigma_prior=ones * math.log(1.0),
            mu_init=ones * math.log(1e-2),
            log_sigma_init=ones * math.log(1e-2),
        ).get_var_distn()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[:-1]
        x = self.layers(x)
        x = x.view(*batch_size, -1)  # reshape back to the original size
        var = self.variance(x.shape[0]).unsqueeze(1)
        return x, var

    def kl_divergence(self) -> Float[Tensor, ""]:
        return self.variance.kl_divergence()


def get_dataloaders(
    data: Dict[str, torch.Tensor], batch_size: int, M: int
) -> tuple[
    arlatentsde.data.MultiEvenlySpacedTensors,
    DataLoader,
    DataLoader,
    DataLoader,
    DataLoader,
    Float[Tensor, ""],
    int,
]:
    """Get the dataloader for the dataset."""
    dataset = arlatentsde.data.MultiEvenlySpacedTensors(data["train_t"], data["train_x"], M)
    sampler = arlatentsde.data.MultiTemporalSampler(dataset, batch_size, n_repeats=1)
    train_loader = DataLoader(
        dataset,
        num_workers=6,
        persistent_workers=True,
        batch_sampler=sampler,
    )
    vdset = arlatentsde.data.MultiEvenlySpacedTensors(data["valid_t"], data["valid_x"], M)
    valid_loader = DataLoader(
        vdset,
        num_workers=6,
        batch_sampler=arlatentsde.data.MultiTemporalSampler(vdset, batch_size),
    )
    test_loader = DataLoader(
        TensorDataset(data["test_t"], data["test_x"]),
        batch_size=len(data["test_t"]),
    )
    vpred_loader = DataLoader(
        TensorDataset(data["valid_t"], data["valid_x"]),
        batch_size=len(data["valid_t"]),
    )
    n_data = dataset.total_data
    dt = dataset.dt
    return dataset, train_loader, valid_loader, vpred_loader, test_loader, dt, n_data


@dataclass(frozen=True, kw_only=True)
class HParams:
    """HParams for the training script."""

    batch_size: int
    M: int
    n_warmup: int
    n_samples: int
    latent_dim: int
    lr: float
    lr_sched_freq: int
    n_mc_quad: int
    kernel_len_init: float
    kernel_var_init: float
    brownian_mu_prior: float
    brownian_sigma_prior: float
    brownian_mu_init: float
    brownian_sigma_init: float

    @classmethod
    def load_from_toml(cls):
        with open(os.path.join(CURR_DIR, "hparams.toml"), "rb") as fp:
            hparams = tomli.load(fp)
        return cls(**hparams)


def parse_hparams(hparams: HParams, n_data: int, device: Device) -> arlatentsde.CTVAEConfig:
    quad_settings = arlatentsde.MCQuadSettings(n_samples=hparams.n_mc_quad, device=device)
    config = arlatentsde.CTVAEConfig(
        n_data=n_data,
        n_warmup=hparams.n_warmup,
        n_samples=hparams.n_samples,
        latent_dim=hparams.latent_dim,
        lr=hparams.lr,
        lr_sched_freq=hparams.lr_sched_freq,
        quad_settings=quad_settings,
    )
    return config


def get_loggers():
    os.makedirs(os.path.join(CURR_DIR, "ckpts", "arcta"), exist_ok=True)
    ckpt_dir = os.path.join("ckpts", "arcta")
    tensorboard = pl_loggers.TensorBoardLogger(
        str(CURR_DIR), name=ckpt_dir, default_hp_metric=False
    )
    save_dir = os.path.join(
        CURR_DIR, "ckpts", "arcta", f"version_{tensorboard.version}"
    )
    os.makedirs(save_dir, exist_ok=True)
    return tensorboard, save_dir


class CTVAEwithMSE(arlatentsde.CTVAE):
    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        return loss

    def compute_mse(self, ns, batch):
        err = []
        soln_set = []
        n_snaps = self.encoder.encode.M  # type: ignore
        with torch.no_grad():
            for t, y in zip(*batch):  # type: ignore
                t0, z, _ = self.encoder(t[[0]], y[:n_snaps].unsqueeze(0))
                y0 = arlatentsde.rsample_latent(
                    ns, *z(t0).chunk(2, dim=-1), device=self.device
                ).squeeze(1)
                sde_kwargs = dict(dt=1e-2, adaptive=True, rtol=1e-4, atol=1e-6)
                soln = solve_sde(self.drift, self.brownian, y0, t, **sde_kwargs)
                mu, var = self.decoder(soln)
                logvar = var.log()
                ys = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
                soln_set.append((ys, soln))
                err.append((mu.mean(0) - y).pow(2))
        err = torch.stack(err)
        return torch.mean(err), err, soln_set

    def validation_step(self, batch, batch_idx):
        # elbo, expected_log_like, kl_div, residual = self.compute_elbo_components(batch)
        # self.log("hp/val_loss", -elbo.item())
        # return -elbo
        ns = 128
        mse, _, _ = self.compute_mse(ns, batch)
        self.log("hp/val_loss", mse.item())
        return mse

    def test_step(self, batch, batch_idx):
        ns = 128
        mse, _, _ = self.compute_mse(ns, batch)
        self.log("hp/test_mse", mse.item())
        return mse


def setup_model(
    config: arlatentsde.CTVAEConfig, hparams: HParams, dt: float, device: Device
) -> arlatentsde.CTVAE:
    """Using config and hparams, sets up a continuous time VAE pl model."""
    offset, scale = arlatentsde.DeepKernel.get_rescale_params(hparams.batch_size, dt)
    # encoder setup
    kernel = arlatentsde.DeepKernel(
        [1, 32, 1],
        torch.nn.Tanh(),
        offset,
        scale,
        hparams.kernel_len_init,
        hparams.kernel_var_init,
    )
    enc = EncodingFCNN(50, hparams.M, hparams.latent_dim, [30, 30, 15], nn.Tanh())
    encoder = arlatentsde.Encoder(dt, kernel, enc)
    decoder = DecodingFCNN(50, hparams.latent_dim, [15, 30, 30], nn.Tanh())
    # latent sde
    drift = arlatentsde.FCLatentSDE(hparams.latent_dim, [30], nn.Tanh())
    ones = torch.ones(hparams.latent_dim)
    brownian = arlatentsde.priors.LogNormalPriorSetup(
        mu_prior=ones * math.log(hparams.brownian_mu_prior),
        log_sigma_prior=ones * math.log(hparams.brownian_sigma_prior),
        mu_init=ones * math.log(hparams.brownian_mu_init),
        log_sigma_init=ones * math.log(hparams.brownian_sigma_init),
    ).get_var_distn()
    # pl model
    loglikelihood = arlatentsde.elbo.generative_gaussian_log_density
    return CTVAEwithMSE(
        config=config,
        encoder=encoder,
        decoder=decoder,
        drift=drift,
        brownian=brownian,
        loglikelihood=loglikelihood,
    )


def freeze_init_state(save_dir, args):
    """save the initial state the initial state
    when postprocessing, we should be able to call: model = setup_model(**args)
    """
    with open(os.path.join(save_dir, "init_state.pkl"), "wb") as fp:
        pkl.dump(args, fp)


def get_summarizer_callback(
    model: arlatentsde.CTVAE, data: Dict[str, torch.Tensor], hparams: HParams, save_dir: str
):
    fig_dir = os.path.join(save_dir, "progress_figs")
    dset = arlatentsde.data.EvenlySpacedTensors(
        data["valid_t"][0], data["valid_x"][0], hparams.M
    )
    os.makedirs(fig_dir, exist_ok=True)
    summarizer = plotting.ProgressSummarizer(
        50,  # 50 frames in latent space
        fig_dir,
        dset,
        model.encoder,
        model.decoder,
        model.drift,
        model.brownian,
        [partial(plotting.plot_latent_sde, True)],  # type: ignore
        max_process=5,
    )
    return plotting.ProgressSummarizerCallback(summary_freq=160, summarizer=summarizer)


def main(args):
    pl.seed_everything(args.random_seed)
    # ---------------------- loading data ---------------------- #
    local_filename = os.path.join(CURR_DIR, "mocap35.mat")
    if not os.path.exists(local_filename):
        download_file(DATA_LINK, local_filename)
    data = load_mocap_data(local_filename)
    hparams = HParams.load_from_toml()
    tensorboard, save_dir = get_loggers()
    # load dataset
    (
        _,
        dataloader,
        _,
        vpred_loader,
        _,
        dt,
        n_data,
    ) = get_dataloaders(data, hparams.batch_size, hparams.M)
    # get device
    device = torch.device("cuda")
    # ---------------------- setting up model ---------------------- #
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
        dirpath=save_dir, save_top_k=1, monitor="hp/val_loss"
    )
    summarizer_callback = get_summarizer_callback(model, data, hparams, save_dir)
    # callbacks =
    trainer = pl.Trainer(
        accelerator=device.type,
        log_every_n_steps=1,
        max_epochs=100,
        logger=tensorboard,
        callbacks=[checkpoint_callback, summarizer_callback],
        check_val_every_n_epoch=10,
        num_sanity_val_steps=0,
    )
    # ---------------------- training ---------------------- #
    trainer.fit(model, dataloader, vpred_loader)
    print("Final validation MSE:")
    mse = partial(model.compute_mse, 128)  # type: ignore
    for batch in vpred_loader:
        _, err, _ = mse(batch)  # type: ignore
    err = err.mean(1).mean().sqrt()  # type: ignore
    print(f"Final validation RMSE: {err.item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARCTA train utility")
    parser.add_argument(
        "-rs", "--random_seed", type=int, help="random seed", default=23
    )
    main(parser.parse_args())
