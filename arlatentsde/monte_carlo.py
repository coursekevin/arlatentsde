from dataclasses import dataclass

import torch
from torch import Tensor
from torch.func import vmap  # type: ignore
from jaxtyping import Float

from typing import Callable
from .types import BndType, IntegrableFn

STensor = Float[Tensor, " n_samples ..."]
IntegrableFnInput = list[Float[Tensor, " ..."]]
VmapTensors = list[Float[Tensor, "r ..."]]


@dataclass(kw_only=True)
class MCQuadSettings:
    device: str | torch.device = "cpu"
    n_samples: int = 120


def vmap_safe_stratified_mc(
    f: Callable[[IntegrableFnInput], IntegrableFn],
    epsilon: VmapTensors,
    eps_candidate: list[Float[Tensor, " ..."]],
    bounds: BndType,
    settings: MCQuadSettings,
) -> Float[Tensor, "n_samples dim"] | Float[Tensor, " n_samples"]:
    """Vectorized version of stratified monte-carlo quad. Vmaps over epsilon
    and computes quadrature estimate of f(eps[i]) for each i. 

    eps_candidate is a legacy argument required by the old version of the 
    integration code. It is not used in this function.
    """

    t_lb, t_ub = float(bounds[0]), float(bounds[1])
    t = torch.linspace(t_lb, t_ub, settings.n_samples + 1, device=settings.device)
    w = t[1] - t[0]
    t_unif = torch.rand(len(epsilon[0]), settings.n_samples, device=settings.device)
    samples = t_unif * w + t[:-1]
    bnd_len = t_ub - t_lb

    def wrapped_stratified_mc(zs, t_batch):
        return f(zs)(t_batch).mean() * bnd_len

    quad = vmap(wrapped_stratified_mc)(epsilon, samples)
    return quad
