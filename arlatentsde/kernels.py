import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn.functional import softplus
from typing import Protocol

from .types import FrozenKernelFn


class KernelFn(Protocol):
    def __call__(
        self, x1: Float[Tensor, " n"], x2: Float[Tensor, " m"]
    ) -> Float[Tensor, "n m"]:
        ...

    @property
    def var(self) -> Float[Tensor, ""]:
        ...


def inverse_softplus(x: Tensor) -> Tensor:
    return torch.log(torch.exp(x) - 1)


def interpolate(
    t: Float[Tensor, " query"],
    y: Float[Tensor, "nodes dim"],
    kern_fn: FrozenKernelFn,
    kern_chol: Float[Tensor, "nodes nodes"],
) -> Float[Tensor, "query dim"]:
    """Interpolate y at t using the kernel function kern_fn and the precomputed
    cholesky decompositions kern_chol"""
    return kern_fn(t).T @ torch.cholesky_solve(y, kern_chol)


class DeepKernel(nn.Module):
    layer_description: list[int]
    rawlens: Float[Tensor, ""]
    offset: Float[Tensor, ""]
    scale: Float[Tensor, ""]

    def __init__(
        self,
        layer_description: list[int],
        nonlinearity: nn.Module,
        offset: Float[Tensor, ""],
        scale: Float[Tensor, ""],
        len_init: float = 1e-2,
        var_init: float = 1e-5,
    ) -> None:
        super().__init__()
        self.layer_description = layer_description
        layers = [
            (nn.Linear(layer_description[i], layer_description[i + 1]), nonlinearity)
            for i in range(len(layer_description) - 2)
        ]
        layers = [mod for layer in layers for mod in layer]
        layers.append(nn.Linear(layer_description[-2], layer_description[-1]))
        self.fc = nn.Sequential(*layers)
        assert layer_description[-1] == 1
        rawlens = inverse_softplus(len_init * torch.ones(layer_description[-1]))
        self.rawlens = nn.Parameter(rawlens)
        self.rawvar = nn.Parameter(inverse_softplus(torch.tensor(var_init)))
        self.rawsigma = nn.Parameter(inverse_softplus(torch.tensor(1.0)))
        self.register_buffer("offset", offset)
        self.register_buffer("scale", scale)

    @property
    def lens(self) -> Float[Tensor, ""]:
        return softplus(self.rawlens)

    @property
    def var(self) -> Float[Tensor, ""]:
        return softplus(self.rawvar)

    @property
    def sigma(self) -> Float[Tensor, ""]:
        return softplus(self.rawsigma)

    def rescale(self, x: Float[Tensor, " n"]) -> Float[Tensor, " n"]:
        return (x - self.offset) / self.scale

    def forward(
        self, x1: Float[Tensor, " n"], x2: Float[Tensor, " m"]
    ) -> Float[Tensor, "m n"]:
        x1, x2 = self.rescale(x1.view(-1, 1)), self.rescale(x2.view(-1, 1))
        x1 = self.fc(x1) + x1
        x2 = self.fc(x2) + x2
        mean = x1.mean(0)
        x1, x2 = x1-mean, x2-mean
        x1 = x1.view(x1.shape[0], 1, x1.shape[1])
        x2 = x2.view(1, x2.shape[0], x2.shape[1])
        return torch.exp(-(x1 - x2).pow(2).div(self.lens).sum(-1)).mul(self.sigma)

    @classmethod
    def get_rescale_params(cls, batch_size: int, dt: float) -> tuple[Tensor, Tensor]:
        """returns rescale params assuming the snapshots are equally spaced"""
        offset = 0.5 * (batch_size - 1) * dt
        scale = 0.5 * (batch_size - 1) * dt
        return torch.tensor(offset), torch.tensor(scale)
