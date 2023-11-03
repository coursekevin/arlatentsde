from arlatentsde.latent_sdes import FCLatentSDE
import torch
from torch import Tensor
import numpy as np
from torch import nn
from jaxtyping import Float
from functools import partial, wraps
from typing import Callable, Dict
from dataclasses import dataclass


@dataclass
class NFELog:
    # total number of function evaluations
    # number of batchwise fevals.
    # (e.g. f(z, t) with z.shape = (batch_size, latent_dim) => 1 batchfe
    totalfe: int
    batchfe: int


def log_nfes(log: Dict[str, NFELog], fn: Callable) -> Callable:
    """logs the number of function evaluations of a method. Uses the first
    argument (not counting self) to compute the number of function evaluations."""

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        name = self.__class__.__name__ + "." + fn.__name__
        if name not in log:
            log[name] = NFELog(0, 0)
        log[name].totalfe += int(np.prod(args[0].shape[:-1]))
        log[name].batchfe += 1
        return fn(self, *args, **kwargs)

    return wrapper


nfe_log = {}


class LoggedFCLatentSDE(FCLatentSDE):
    totalfe: int
    batchfe: int

    def __init__(
        self, latent_dim: int, layer_description: list[int], nonlinearity: nn.Module
    ):
        self.totalfe = 0
        self.batchfe = 0
        super().__init__(latent_dim, layer_description, nonlinearity)

    def forward(
        self, z: Float[Tensor, "... latent_dim"], t: Float[Tensor, "... 1"]
    ) -> Float[Tensor, "... latent_dim"]:
        if self.training:
            self.totalfe += int(np.prod(z.shape[:-1]))
            self.batchfe += 1
        return super().forward(z, t)


if __name__ == "__main__":
    model = LoggedFCLatentSDE(2, [8, 16], nn.Tanh())
    # example usage: 2 batchfe, 30 totalfe
    model(torch.randn(100, 2), torch.randn(10, 1))
    model(torch.randn(20, 2), torch.randn(20, 1))
    print(model.totalfe, model.batchfe)
