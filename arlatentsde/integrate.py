import torch
from torch import nn
from torch import Tensor
from jaxtyping import Float
from .types import DriftFn
from .priors import Prior
from torchsde import sdeint


class TorchSDE(nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, n_samples: int, drift: DriftFn, brownian: Prior) -> None:
        super().__init__()
        self.drift = drift
        self._bsamples = brownian(n_samples).sqrt()

    def f(self, t, y):
        return self.drift(y, t)

    def g(self, t, y):
        return self._bsamples


def solve_sde(
    drift: DriftFn,
    brownian: Prior,
    y0: Float[Tensor, "ns dim"],
    ts: Float[Tensor, " nt"],
    **sdeint_kwargs
) -> Float[Tensor, "ns nt dim"]:
    sde = TorchSDE(n_samples=y0.shape[0], drift=drift, brownian=brownian)
    with torch.no_grad():
        sde_soln = sdeint(sde=sde, y0=y0, ts=ts, **sdeint_kwargs)
    assert isinstance(sde_soln, Tensor), "sde_soln is not a torch.Tensor"
    return sde_soln.transpose(1, 0)
