from functools import partial, wraps
from dataclasses import dataclass
from typing import Protocol, Callable, Optional

import torch
from torch.func import vmap, jacfwd  # type: ignore
import numpy as np
from jaxtyping import Float
from torch import Tensor, nn
from .types import (
    ImgWin,
    FrozenKernelFn,
    LatentStateFn,
    LatentStateDerivativeFn,
)
from .kernels import interpolate, KernelFn


class EncodingFn(Protocol):
    def __call__(self, x: ImgWin) -> Float[Tensor, "batch 2*latent_dim"]:
        ...


class Decoder(Protocol):
    def __call__(
        self, z: Float[Tensor, "... latent_dim"]
    ) -> tuple[Float[Tensor, "..."], ...]:
        ...

    def kl_divergence(self) -> Float[Tensor, ""]:
        ...


def squeeze_wrap(
    fn: Callable[..., tuple[Tensor, ...]]
) -> Callable[..., tuple[Tensor, ...]]:
    """Squeeze the output of a function that returns a tuple of tensors."""

    @wraps(fn)
    def wrapper(*args, **kwargs) -> tuple[Tensor, ...]:
        out_tuple = fn(*args, **kwargs)
        return tuple(out.squeeze() for out in out_tuple)

    return wrapper


def return_first(fn: Callable[..., tuple[Tensor, ...]]) -> Callable[..., Tensor]:
    """Return the first element of a function that returns a tuple of tensors."""

    @wraps(fn)
    def wrapper(*args, **kwargs) -> Tensor:
        return fn(*args, **kwargs)[0]

    return wrapper


class Encoder(nn.Module):
    """
    Encoder for the latent state of a dynamical system.

    Args:
        dt: the time step size
        channels: the number of channels in the input image
        kernel: the kernel function
        encoding_cnn: the encoding cnn
    """

    dt: Float[Tensor, ""]

    def __init__(
        self,
        dt: float,
        kernel: KernelFn,
        encoding_fn: EncodingFn,
    ):
        super().__init__()
        self.register_buffer("dt", torch.tensor(dt))
        self.kernel = kernel
        self.encode = encoding_fn

    def interp_latent(
        self,
        t0: Float[Tensor, ""],
        hidden: Float[Tensor, "batch 2*latent_dim"],
        frozen_kern: FrozenKernelFn,
        kern_chol: Float[Tensor, "nodes nodes"],
        t: Float[Tensor, " query"],
    ) -> tuple[
        Float[Tensor, "batch 2*latent_dim"], Float[Tensor, "batch 2*latent_dim"]
    ]:
        interp = interpolate(t - t0, hidden, frozen_kern, kern_chol)
        return interp, interp

    def forward(
        self, t: Float[Tensor, " n_steps"], x: ImgWin
    ) -> tuple[Float[Tensor, " n_steps"], LatentStateFn, LatentStateDerivativeFn]:
        hidden = self.encode(x)
        n_steps = x.shape[0]
        t_node = t - t[0]
        eye = torch.eye(n_steps, device=self.dt.device)
        kern_chol = torch.linalg.cholesky(
            self.kernel(t_node, t_node) + self.kernel.var * eye
        )
        frozen_kern = partial(self.kernel, t_node)
        interp_fn = partial(self.interp_latent, t[0].clone(), hidden, frozen_kern, kern_chol)
        z = return_first(interp_fn)
        # dz_eval, z_eval = dz(t)
        dz = squeeze_wrap(vmap(jacfwd(interp_fn, has_aux=True)))
        return t_node, z, dz  # type: ignore


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
    _empty_tensor: torch.Tensor

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
        layers.append(nn.Linear(layer_description[-1], full_dim * 2))
        self.layers = nn.Sequential(*layers)
        self.register_buffer("_empty_tensor", torch.empty(0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.layers(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        return mu, logvar.exp()

    def kl_divergence(self) -> Float[Tensor, ""]:
        return torch.tensor(0.0, device=self._empty_tensor.device)


@dataclass
class ConvLayer:
    channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    maxpool: Optional[dict] = None


def comp_output_dims(in_dims: tuple[int, int], layer: ConvLayer) -> tuple[int, int]:
    # Compute output dimensions
    in_dims = (
        (in_dims[0] - layer.kernel_size + 2 * layer.padding) // layer.stride + 1,
        (in_dims[1] - layer.kernel_size + 2 * layer.padding) // layer.stride + 1,
    )
    if layer.maxpool:
        in_dims = (
            (in_dims[0] - layer.maxpool["kernel_size"]) // layer.maxpool["stride"] + 1,
            (in_dims[1] - layer.maxpool["kernel_size"]) // layer.maxpool["stride"] + 1,
        )
    return in_dims


class EncodingCNN(nn.Module):
    conv: nn.Module
    fc: nn.Module

    def __init__(
        self,
        in_channels: int,
        in_dims: tuple[int, int],
        conv_arch: list[ConvLayer],
        fc_arch: list[int],
        latent_dim: int,
        nonlinearity,
    ) -> None:
        super().__init__()
        self.nonlinearity = nonlinearity

        conv_layers = []
        for i, layer in enumerate(conv_arch):
            in_ch = conv_arch[i - 1].channels if i > 0 else in_channels
            conv_layers.append(
                nn.Conv2d(
                    in_ch,
                    layer.channels,
                    layer.kernel_size,
                    layer.stride,
                    layer.padding,
                )
            )
            conv_layers.append(nonlinearity)
            if layer.maxpool:
                conv_layers.append(nn.MaxPool2d(**layer.maxpool))

            in_dims = comp_output_dims(in_dims, layer)
        self.conv = nn.Sequential(*conv_layers)
        fc_layers = []
        for i, out_features in enumerate(fc_arch):
            in_features = (
                conv_arch[-1].channels * in_dims[0] * in_dims[1]
                if i == 0
                else fc_arch[i - 1]
            )
            fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(nonlinearity)
        fc_layers.append(nn.Linear(fc_arch[-1], latent_dim * 2))
        self.fc = nn.Sequential(*fc_layers)

    def forward(
        self, x: Float[Tensor, "... c d1 d2"]
    ) -> Float[Tensor, "... 2*latent_dim"]:
        batch_size = x.shape[:-3]
        x = self.conv(x.view(-1, *x.shape[-3:]))
        x = x.view(*batch_size, -1)
        x = self.fc(x)
        return x


def required_padding(
    in_dims: tuple[int, int], out_dims: tuple[int, int], layer: ConvLayer
) -> tuple[int, int]:
    """Compute padding required by CONVT to match equivalent CONV"""
    padding = (
        out_dims[0]
        - ((in_dims[0] - 1) * layer.stride - 2 * layer.padding + layer.kernel_size),
        out_dims[1]
        - ((in_dims[1] - 1) * layer.stride - 2 * layer.padding + layer.kernel_size),
    )
    return padding


def compute_conv_out_dims(
    out_dims: tuple[int, int], layer: ConvLayer
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Compute output dimensions, padding and usampling size required by CONVT to
    match equivalent CONV with pooling for a single layer"""
    conv_out_dims = (
        (out_dims[0] - layer.kernel_size + 2 * layer.padding) // layer.stride + 1,
        (out_dims[1] - layer.kernel_size + 2 * layer.padding) // layer.stride + 1,
    )
    padding = required_padding(conv_out_dims, out_dims, layer)
    us_size = conv_out_dims
    if layer.maxpool:
        conv_out_dims = (
            (conv_out_dims[0] - layer.maxpool["kernel_size"]) // layer.maxpool["stride"]
            + 1,
            (conv_out_dims[1] - layer.maxpool["kernel_size"]) // layer.maxpool["stride"]
            + 1,
        )
    return conv_out_dims, padding, us_size


def get_deconv_parameters(
    conv_arch: list[ConvLayer], in_dims: tuple[int, int]
) -> tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    """Compute output dimensions, padding and usampling size required by CONVT to
    match equivalent CONV with pooling for all layers"""
    conv_out_dims = in_dims
    padding_arch = []
    us_arch = []
    for i, layer in enumerate(conv_arch):
        conv_out_dims, padding, ussize = compute_conv_out_dims(conv_out_dims, layer)
        padding_arch.append(padding)
        us_arch.append(ussize)
    return conv_out_dims, padding_arch, us_arch


class DecodingCNN(nn.Module):
    fc: nn.Module
    deconv: nn.Module
    nonlinearity: nn.Module
    in_channels: int
    in_dims: tuple[int, int]
    # dimension of output before starting deconvolution
    conv_out_dims: tuple[int, int]
    conv_out_channels: int
    _empty_tensor: Float[Tensor, ""]

    def __init__(
        self,
        in_channels: int,
        in_dims: tuple[int, int],
        rev_conv_arch: list[ConvLayer],
        fc_arch: list[int],
        latent_dim: int,
        nonlinearity: nn.Module,
    ) -> None:
        super().__init__()
        self.nonlinearity = nonlinearity
        self.in_channels = in_channels
        self.in_dims = in_dims
        conv_arch = list(reversed(rev_conv_arch))
        self.conv_out_channels = conv_arch[-1].channels
        conv_out_dims, padding_arch, us_arch = get_deconv_parameters(conv_arch, in_dims)
        self.conv_out_dims = conv_out_dims
        # building fc layers
        fc_layers = []
        for i, out_features in enumerate(fc_arch):
            in_features = fc_arch[i - 1] if i > 0 else latent_dim
            fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(nonlinearity)
        fc_out_features = conv_out_dims[0] * conv_out_dims[1] * conv_arch[-1].channels
        fc_layers.append(nn.Linear(fc_arch[-1], fc_out_features))
        fc_layers.append(nonlinearity)
        self.fc = nn.Sequential(*fc_layers)
        # building deconv layers
        deconv_layers = []
        for i, layer in reversed(list(enumerate(conv_arch))):
            out_ch = conv_arch[i - 1].channels if i > 0 else in_channels * 2
            if layer.maxpool:
                deconv_layers.append(nn.Upsample(size=us_arch[i]))
            deconv_layers.append(
                nn.ConvTranspose2d(
                    layer.channels,
                    out_ch,
                    layer.kernel_size,
                    layer.stride,
                    layer.padding,
                    output_padding=padding_arch[i],
                )
            )
            if i > 0:
                deconv_layers.append(nonlinearity)
        self.deconv = nn.Sequential(*deconv_layers)
        self.register_buffer("_empty_tensor", torch.empty(0))

    def forward(
        self, x: Float[Tensor, "... latent_dim"]
    ) -> tuple[Float[Tensor, "... c d1 d2"], Float[Tensor, "... c d1 d2"]]:
        batch_size = x.shape[:-1]
        x = self.fc(x.view(-1, x.shape[-1]))
        x = x.view(-1, self.conv_out_channels, *self.conv_out_dims)
        x = self.deconv(x)
        x = x.view(*batch_size, self.in_channels * 2, *self.in_dims)
        mu, logvar = torch.chunk(x, 2, dim=-3)
        return mu, logvar.exp()

    def kl_divergence(self) -> Float[Tensor, ""]:
        return torch.tensor(0.0, device=self._empty_tensor.device)
