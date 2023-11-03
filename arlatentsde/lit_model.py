import math
from dataclasses import dataclass
from functools import partial

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.optim import lr_scheduler

from .autoencoders import Decoder, Encoder, EncodingFCNN, DecodingFCNN
from .autoencoders import ConvLayer, EncodingCNN, DecodingCNN
from .elbo import normed_residual, rsample_latent, generative_gaussian_log_density
from .kernels import DeepKernel
from .latent_sdes import FCLatentSDE, LatentSDE
from .monte_carlo import MCQuadSettings, vmap_safe_stratified_mc
from .priors import Prior, LogNormalPriorSetup
from .types import LoglikeFn


def warmup(num_warmup: int, iter_count: int) -> float:
    return min(iter_count / num_warmup, 1.0)


@dataclass(frozen=True)
class CTVAEConfig:
    """Configuration for continuous time variational autoencoders."""

    n_data: int
    n_warmup: int
    n_samples: int
    latent_dim: int
    lr: float
    lr_sched_freq: int
    quad_settings: MCQuadSettings


class CTVAE(pl.LightningModule):
    """Maximum likelihood estimation of continuous time variational autoencoder."""

    _empty_tensor: Tensor  # empty tensor to get device
    _iter_count: int  # iteration count
    config: CTVAEConfig
    encoder: Encoder
    decoder: Decoder
    drift: LatentSDE
    brownian: Prior
    loglikelihood: LoglikeFn

    def __init__(
        self,
        config: CTVAEConfig,
        encoder: Encoder,
        decoder: Decoder,
        drift: LatentSDE,
        brownian: Prior,
        loglikelihood: LoglikeFn,
    ):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.drift = drift
        self.brownian = brownian
        self.loglikelihood = loglikelihood

        self.register_buffer("_empty_tensor", torch.empty(0))
        self._iter_count = 0

    @property
    def device(self) -> torch.device:
        return self._empty_tensor.device

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

    def training_step(self, batch, batch_idx):
        """perform one training step."""
        self._iter_count += 1
        # compute losses
        elbo, expected_log_like, kl_div, residual = self.compute_elbo_components(batch)
        self.log("hp/train_loss", -elbo.item())
        # train loss without beta scaling
        self.log("hp/raw_train_loss", -(expected_log_like - kl_div - residual).item())
        return -elbo

    def configure_optimizers(self):
        # ------------------------------------------------------------------------------
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        gamma = math.exp(math.log(0.9) / self.config.lr_sched_freq)
        scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "hp/train_loss",
                "interval": "step",  # call scheduler after every train step
                "frequency": 1,
            },
        }


@dataclass(frozen=True)
class FCHParams:
    """Configuration for latent SDE with fully-connected encoder-decoder"""

    full_dim: int
    dt: float
    batch_size: int
    M: int
    config: CTVAEConfig
    kernel_layer_description: list[int]
    encoder_layer_description: list[int]
    decoder_layer_description: list[int]
    drift_layer_description: list[int]
    autonomous: bool = True
    kernel_nonlinearity: nn.Module = nn.Tanh()
    encoder_nonlinearity: nn.Module = nn.Tanh()
    decoder_nonlinearity: nn.Module = nn.Tanh()
    drift_nonlinearity: nn.Module = nn.Tanh()
    kernel_len_init: float = 1e-2
    kernel_var_init: float = 1e-5
    brownian_mu_prior: float = 1.0
    brownian_sigma_prior: float = 1.0
    brownian_mu_init: float = 1e-5
    brownian_sigma_init: float = 1e-5

    def init_fc_latent_sde(self):
        config = self.config
        offset, scale = DeepKernel.get_rescale_params(self.batch_size, self.dt)
        # encoder setup
        kernel = DeepKernel(
            self.kernel_layer_description,
            self.kernel_nonlinearity,
            offset,
            scale,
            self.kernel_len_init,
            self.kernel_var_init,
        )
        enc = EncodingFCNN(
            self.full_dim,
            self.M,
            config.latent_dim,
            self.encoder_layer_description,
            self.encoder_nonlinearity,
        )
        encoder = Encoder(self.dt, kernel, enc)
        decoder = DecodingFCNN(
            self.full_dim,
            config.latent_dim,
            self.decoder_layer_description,
            self.decoder_nonlinearity,
        )
        # latent sde
        drift = FCLatentSDE(
            config.latent_dim,
            self.drift_layer_description,
            self.drift_nonlinearity,
            autonomous=self.autonomous,
        )
        ones = torch.ones(config.latent_dim)
        brownian = LogNormalPriorSetup(
            mu_prior=ones * math.log(self.brownian_mu_prior),
            log_sigma_prior=ones * math.log(self.brownian_sigma_prior),
            mu_init=ones * math.log(self.brownian_mu_init),
            log_sigma_init=ones * math.log(self.brownian_sigma_init),
        ).get_var_distn()
        # pl model
        loglikelihood = generative_gaussian_log_density
        return CTVAE(
            config=config,
            encoder=encoder,
            decoder=decoder,
            drift=drift,
            brownian=brownian,
            loglikelihood=loglikelihood,
        )


@dataclass(frozen=True)
class ConvHParams:
    """Configuration for latent SDE with fully-connected encoder-decoder"""

    in_channels: int
    in_dims: tuple[int, int]
    dt: float
    batch_size: int
    M: int
    config: CTVAEConfig
    conv_arch: list[ConvLayer]
    fc_arch: list[int]
    kernel_layer_description: list[int]
    drift_layer_description: list[int]
    autonomous: bool = True
    kernel_nonlinearity: nn.Module = nn.Tanh()
    encoder_nonlinearity: nn.Module = nn.Tanh()
    decoder_nonlinearity: nn.Module = nn.Tanh()
    drift_nonlinearity: nn.Module = nn.Tanh()
    kernel_len_init: float = 1e-2
    kernel_var_init: float = 1e-5
    brownian_mu_prior: float = 1.0
    brownian_sigma_prior: float = 1.0
    brownian_mu_init: float = 1e-5
    brownian_sigma_init: float = 1e-5

    def init_fc_latent_sde(self):
        config = self.config
        offset, scale = DeepKernel.get_rescale_params(self.batch_size, self.dt)
        # encoder setup
        kernel = DeepKernel(
            self.kernel_layer_description,
            self.kernel_nonlinearity,
            offset,
            scale,
            self.kernel_len_init,
            self.kernel_var_init,
        )
        enc = EncodingCNN(
            self.in_channels * self.M,
            self.in_dims,
            self.conv_arch,
            self.fc_arch,
            self.config.latent_dim,
            self.encoder_nonlinearity,
        )
        encoder = Encoder(self.dt, kernel, enc)
        decoder = DecodingCNN(
            self.in_channels,
            self.in_dims,
            list(reversed(self.conv_arch)),
            list(reversed(self.fc_arch)),
            self.config.latent_dim,
            self.decoder_nonlinearity,
        )
        # latent sde
        drift = FCLatentSDE(
            config.latent_dim,
            self.drift_layer_description,
            self.drift_nonlinearity,
            autonomous=self.autonomous,
        )
        ones = torch.ones(config.latent_dim)
        brownian = LogNormalPriorSetup(
            mu_prior=ones * math.log(self.brownian_mu_prior),
            log_sigma_prior=ones * math.log(self.brownian_sigma_prior),
            mu_init=ones * math.log(self.brownian_mu_init),
            log_sigma_init=ones * math.log(self.brownian_sigma_init),
        ).get_var_distn()
        # pl model
        loglikelihood = generative_gaussian_log_density
        return CTVAE(
            config=config,
            encoder=encoder,
            decoder=decoder,
            drift=drift,
            brownian=brownian,
            loglikelihood=loglikelihood,
        )
