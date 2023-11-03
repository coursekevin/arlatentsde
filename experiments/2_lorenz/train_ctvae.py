import argparse
import os
import sys
import math
import pathlib
import pickle as pkl
from dataclasses import dataclass, asdict
from typing import Any, Optional, Dict
import warnings
from functools import partial
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
from torch import nn
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from jaxtyping import Float
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from scipy.integrate import solve_ivp
import tomli

import matplotlib

from arlatentsde.autoencoders import Decoder, Encoder
from arlatentsde.latent_sdes import LatentSDE
from arlatentsde.lit_model import CTVAEConfig
from arlatentsde.priors import Prior

matplotlib.use("Agg")  # Use the 'Agg' renderer

import arlatentsde
from arlatentsde.types import Device, LoglikeFn
from arlatentsde import plotting
from arlatentsde.integrate import solve_sde

torch.set_default_dtype(torch.float64)
CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(CURR_DIR)

from generate_data import DATA_DIR, ScaleTsfm
from forward_model import LorenzModel


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
    _zero: Float[Tensor, ""]

    def __init__(self):
        super().__init__()
        self.register_buffer("_zero", torch.tensor(0.0))

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return x

    def kl_divergence(self) -> Float[Tensor, ""]:
        return self._zero


def get_dataloaders(
    batch_size: int, M: int, dpath: str
) -> tuple[
    arlatentsde.data.EvenlySpacedTensors, DataLoader, DataLoader, Float[Tensor, ""], int
]:
    """Get the dataloader for the dataset."""
    with open(dpath, "rb") as f:
        data = pkl.load(f)
    dataset = arlatentsde.data.EvenlySpacedTensors(data["train_t"], data["train_y"], M)
    sampler = arlatentsde.data.TemporalSampler(dataset, batch_size, n_repeats=1)
    train_loader = DataLoader(
        dataset,
        num_workers=6,
        persistent_workers=True,
        batch_sampler=sampler,
    )
    valid_loader = DataLoader(
        TensorDataset(data["test_t"], data["test_y"]),
        batch_size=len(data["test_t"]),
    )
    n_data = dataset.N
    dt = dataset.dt
    return dataset, train_loader, valid_loader, dt, n_data


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


class CTVAEGradLogging(arlatentsde.CTVAE):
    experiment_log: Dict[str, list[float]]
    def __init__(
        self,
        config: CTVAEConfig,
        encoder: Encoder,
        decoder: Decoder,
        drift: LatentSDE,
        brownian: Prior,
        loglikelihood: LoglikeFn,
    ):
        super().__init__(config, encoder, decoder, drift, brownian, loglikelihood)
        # extra log just for the experiment
        self.experiment_log = {
            "grad_norm": [],
            "param_norm": [torch.norm(drift.params - drift.raw_params).item()],
        }

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        return loss

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        gnorm = torch.norm(self.drift.params.grad).item()
        self.log("grad_norm", gnorm)
        self.experiment_log["grad_norm"].append(gnorm)

    def validation_step(self, batch, batch_idx):
        loss = torch.norm(self.drift.params - self.drift.raw_params)
        self.log("param_norm", loss.item())
        self.experiment_log["param_norm"].append(loss.item())
        return loss


def setup_model(
    rs, config: arlatentsde.CTVAEConfig, hparams: HParams, dt: float, device: Device
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
    enc = DirectEncFCNN(hparams.latent_dim, [32, 32], nn.ReLU())
    encoder = arlatentsde.Encoder(dt, kernel, enc)
    # decoder setup
    # latent sde
    drift = LorenzModel(rs)
    ones = torch.ones(hparams.latent_dim)
    brownian = arlatentsde.priors.LogNormalPriorSetup(
        mu_prior=ones * math.log(hparams.brownian_mu_prior),
        log_sigma_prior=ones * math.log(hparams.brownian_sigma_prior),
        mu_init=ones * math.log(hparams.brownian_mu_init),
        log_sigma_init=ones * math.log(hparams.brownian_sigma_init),
    ).get_var_distn()
    # pl model
    var = torch.tensor(1.0, device=device)
    loglikelihood = partial(arlatentsde.elbo.gaussian_log_density, var)
    return CTVAEGradLogging(
        config=config,
        encoder=encoder,
        decoder=Identity(),
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
        [partial(plotting.plot_latent_sde, False)],
        max_process=12,
    )
    return plotting.ProgressSummarizerCallback(summary_freq=100, summarizer=summarizer)


def main(args):
    seed = args.random_seed
    # load hparams from hparams.toml
    hparams = HParams.load_from_toml()
    pl.seed_everything(seed)
    tensorboard, save_dir = get_loggers()
    # load dataset
    dataset, dataloader, valid_loader, dt, n_data = get_dataloaders(
        hparams.batch_size, hparams.M, args.dpath
    )
    # get device
    device = torch.device("cuda")
    # parse hparams into config
    config = parse_hparams(hparams, n_data, device)
    init_state = config, hparams, float(dt), device
    # save the init state for post_processing
    freeze_init_state(save_dir, init_state)
    model = setup_model(seed, *init_state)
    # log hparams for tensorboard
    tensorboard.log_hyperparams(asdict(hparams))
    # setup callbacks
    # summarizer_callback = get_summarizer_callback(model, dataset, save_dir)
    trainer = pl.Trainer(
        accelerator=device.type,
        log_every_n_steps=1,
        max_steps=2000,
        logger=tensorboard,
        # callbacks=[summarizer_callback],
        val_check_interval=0.1,
        # callbacks=callbacks,  # type: ignore
        num_sanity_val_steps=0,
    )
    trainer.fit(model, dataloader, valid_loader)
    experiment_log = model.experiment_log
    experiment_log["tf"] = args.final_time # type: ignore
    experiment_log["rs"] = seed # type: ignore
    with open(os.path.join(save_dir, "experiment_log.pkl"), "wb") as fp:
        pkl.dump(experiment_log, fp)
    print("Finished with final error of: ", experiment_log["param_norm"][-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arcta train utility")
    parser.add_argument(
        "-rs", "--random_seed", type=int, help="random seed", required=True
    )
    parser.add_argument("--dpath", type=str, help="path to data", required=True)
    parser.add_argument("-tf", "--final_time", type=float, help="final time", required=True)
    main(parser.parse_args())
