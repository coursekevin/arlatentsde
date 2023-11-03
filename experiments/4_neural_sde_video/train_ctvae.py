import os
import math
import pathlib
import pickle as pkl
from dataclasses import dataclass, asdict
from typing import List
import warnings
from functools import partial

import torch
from torch import nn
from torch.func import vmap, jacfwd
from torch import Tensor
from torchvision import transforms
from jaxtyping import Float
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import tomli
from PIL import Image

import arlatentsde
from arlatentsde.types import Img, ImgWin, Device
from arlatentsde import plotting

torch.set_float32_matmul_precision("high")
warnings.simplefilter("always", UserWarning)
CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
FRAME_DIR = os.path.join(CURR_DIR, "frames")


class EncodingCNN(nn.Module):
    def __init__(self, resolution: tuple[int, int], channels: int, hidden_dim: int):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        self.flattened_size = (resolution[0] // 4) * (resolution[1] // 4) * 32

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, hidden_dim)

        self.nonlin = nn.ReLU()

    # @torch.compile
    def forward(self, x: ImgWin) -> Float[Tensor, "batch hidden_dim"]:
        x = self.nonlin(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.nonlin(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = x.reshape(-1, self.flattened_size)
        x = self.nonlin(self.bn4(self.fc1(x)))
        x = self.fc2(x)

        return x


class Decoder(nn.Module):
    zero: Tensor
    def __init__(self, resolution: tuple[int, int], latent_dim: int):
        super().__init__()

        self.initial_resolution = (resolution[0] // 4, resolution[1] // 4)
        resize_dim = 64 * self.initial_resolution[0] * self.initial_resolution[1]
        self.fc1 = nn.Linear(latent_dim, resize_dim)
        self.fc2 = nn.Linear(resize_dim, resize_dim)

        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
        self.nonlin = nn.ReLU()
        self.register_buffer("zero", torch.tensor(0.)) 

    # @torch.compile
    def forward(
        self, z: Float[Tensor, "... latent_dim"]
    ) -> Float[Tensor, "... res1 res2"]:
        """returns logits"""
        batch_size = z.shape[:-1]
        x = self.nonlin(self.fc1(z))
        x = self.nonlin(self.fc2(x))
        x = x.view(-1, 64, *self.initial_resolution)
        x = self.nonlin(self.bn1(self.deconv1(x)))
        x = self.deconv2(x)
        return x.view(*batch_size, *x.shape[1:]).squeeze(-3)

    def kl_divergence(self):
        return self.zero


def binarize(x: Img) -> Img:
    """Binarize the frames."""
    return (x < 0.5).float().squeeze(0)


def get_dataloaders(
    batch_size: int, res: int, M: int
) -> tuple[arlatentsde.data.EvenlySpacedFrames, DataLoader, Float[Tensor, ""], int]:
    """Get the dataloader for the dataset."""
    transform = transforms.Compose(
        [
            transforms.Resize((res, res), interpolation=Image.BICUBIC),  # type: ignore
            transforms.ToTensor(),
            transforms.Lambda(binarize),  # Binarize the frames
        ]
    )
    with open(os.path.join(FRAME_DIR, "t.pkl"), "rb") as f:
        t = pkl.load(f)["t"].float()
    torch.allclose(t[1:] - t[:-1], t[1] - t[0])
    dt = t[1] - t[0]
    dataset = arlatentsde.data.EvenlySpacedFrames(FRAME_DIR, M, dt, transform)
    sampler = arlatentsde.data.TemporalSampler(dataset, batch_size, n_repeats=50)
    dataloader = DataLoader(
        dataset,
        num_workers=24,
        persistent_workers=True,
        batch_sampler=sampler,
    )
    n_data = dataset.N
    return dataset, dataloader, dt, n_data


@dataclass(frozen=True, kw_only=True)
class HParams:
    """HParams for the training script."""

    seed: int
    batch_size: int
    res: int
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


def get_loggers(random_seed: int):
    os.makedirs(os.path.join(CURR_DIR, "ckpts"), exist_ok=True)
    tensorboard = pl_loggers.TensorBoardLogger(
        str(CURR_DIR), name="ckpts", default_hp_metric=False
    )
    save_dir = os.path.join(CURR_DIR, "ckpts", f"version_{tensorboard.version}")
    os.makedirs(save_dir, exist_ok=True)
    return tensorboard, save_dir


def setup_model(config: arlatentsde.CTVAEConfig, hparams: HParams, dt: float) -> arlatentsde.CTVAE:
    """Using config and hparams, sets up a continuous time VAE pl model."""
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
    encoding_cnn = EncodingCNN(
        (hparams.res, hparams.res), hparams.M, hparams.latent_dim * 2
    )
    encoder = arlatentsde.Encoder(dt, kernel, encoding_cnn)
    # decoder setup
    decoder = Decoder((hparams.res, hparams.res), hparams.latent_dim)
    # latent sde
    drift = arlatentsde.FCLatentSDE(hparams.latent_dim, [64, 64], nn.Tanh())
    ones = torch.ones(hparams.latent_dim)
    brownian = arlatentsde.priors.LogNormalPriorSetup(
        mu_prior=ones * math.log(hparams.brownian_mu_prior),
        log_sigma_prior=ones * math.log(hparams.brownian_sigma_prior),
        mu_init=ones * math.log(hparams.brownian_mu_init),
        log_sigma_init=ones * math.log(hparams.brownian_sigma_init),
    ).get_var_distn()
    # pl model
    return arlatentsde.CTVAE(
        config=config,
        encoder=encoder,
        decoder=decoder,
        drift=drift,
        brownian=brownian,
        loglikelihood=arlatentsde.elbo.bernoulli_log_density,
    )


def freeze_init_state(save_dir, args):
    """save the initial state the initial state
    when postprocessing, we should be able to call: model = setup_model(**args)
    """
    with open(os.path.join(save_dir, "init_state.pkl"), "wb") as fp:
        pkl.dump(args, fp)


def get_summarizer_callback(
    model: arlatentsde.CTVAE, dset: arlatentsde.EvenlySpacedFrames, save_dir: str
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
        [partial(plotting.plot_latent_sde, False), plotting.plot_reconstruction], # type: ignore
        max_process=12,
    )
    return plotting.ProgressSummarizerCallback(summary_freq=500, summarizer=summarizer)


def main():
    # load hparams from hparams.toml
    hparams = HParams.load_from_toml()
    pl.seed_everything(hparams.seed)
    tensorboard, save_dir = get_loggers(hparams.seed)
    # load dataset
    dataset, dataloader, dt, n_data = get_dataloaders(
        hparams.batch_size, hparams.res, hparams.M
    )
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        warnings.warn("CUDA not available, running on CPU")
    # parse hparams into config
    config = parse_hparams(hparams, n_data, device)
    init_state = config, hparams, float(dt)
    # save the init state for post_processing
    freeze_init_state(save_dir, init_state)
    model = setup_model(*init_state)
    # log hparams for tensorboard
    tensorboard.log_hyperparams(asdict(hparams))
    # setup callbacks
    callbacks = [get_summarizer_callback(model, dataset, save_dir)]
    # note: we use duplicate sampling so max_epochs is actually n_epochs * n_repeats
    trainer = pl.Trainer(
        accelerator=device.type,
        log_every_n_steps=1,
        max_epochs=50,
        logger=tensorboard,
        callbacks=callbacks,  # type: ignore
    )
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
