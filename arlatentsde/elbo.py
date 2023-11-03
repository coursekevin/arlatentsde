from functools import partial

import torch
from torch import Tensor
from torch.func import vmap  # type: ignore
from jaxtyping import Float

from .types import TimeFn, DriftFn


def reparam_tsfm(
    mu: Float[Tensor, "... latent_dim"],
    logvar: Float[Tensor, "... latent_dim"],
    epsilon: Float[Tensor, "... latent_dim"],
) -> Float[Tensor, "... latent_dim"]:
    """Generate reparameterized samplees from a normal distribution"""
    return mu + epsilon * torch.exp(0.5 * logvar)


def rsample_latent(
    n_samples: int,
    mu: Float[Tensor, "... latent_dim"],
    logvar: Float[Tensor, "... latent_dim"],
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "... n_samples latent_dim"]:
    """Use reparameterization trick to sample from latent distribution"""
    epsilon = torch.randn(n_samples, *mu.shape, device=device)
    return vmap(partial(reparam_tsfm, mu, logvar))(epsilon)


def solve_lyapunov(
    logvar: Float[Tensor, "nquad latent_dim"],
    dlogvar: Float[Tensor, "nquad latent_dim"],
    brownian: Float[Tensor, " latent_dim"],
) -> Float[Tensor, "nquad latent_dim"]:
    """
    Solve the lyapunov equation

    Args:
        logvar: logvariance of the variational distribution
        dlogvar: time derivative of the logvariance
        brownian: log of the brownian motion matrix (L Q L^T)

    Returns:
        Tensor: solution to the Lyapunov equations
    """
    return 0.5 * (brownian / (logvar.exp()) - dlogvar)


def residual(
    epsilon: Float[Tensor, " latent_dim"],
    brownian: Float[Tensor, " latent_dim"],
    dz: TimeFn,
    f: DriftFn,
    t: Float[Tensor, " nquad"],
) -> Float[Tensor, "nquad latent_dim"]:
    t = t.unsqueeze(-1)
    dz_eval, z_eval = dz(t)
    mu, logvar = z_eval.chunk(2, dim=-1)
    dmu, dlogvar = dz_eval.chunk(2, dim=-1)
    a_mat = solve_lyapunov(logvar, dlogvar, brownian)
    zs = reparam_tsfm(mu, logvar, epsilon)
    return a_mat * (mu - zs) + (dmu - f(zs, t))


def normed_residual(
    epsilon: Float[Tensor, " latent_dim"],
    brownian: Float[Tensor, " latent_dim"],
    dz: TimeFn,
    f: DriftFn,
    t: Float[Tensor, " nquad"],
) -> Float[Tensor, " nquad"]:
    res = residual(epsilon, brownian, dz, f, t)
    return res.pow(2).div(brownian).sum(-1)


@torch.compile
def bernoulli_log_density(
    b: Float[Tensor, "..."],
    unnormalized_logprob: Float[Tensor, "..."],
    device: torch.device | str,
) -> Float[Tensor, "..."]:
    """Compute the log density of a bernoulli distribution element wise"""
    s = b * 2 - 1
    zero = torch.tensor(0.0, device=device)
    return -torch.logaddexp(zero, -s * unnormalized_logprob).sum((-1, -2))

# @torch.compile
def generative_gaussian_log_density(
    y_true: Float[Tensor, "... dim"],
    params: tuple[Float[Tensor, "... dim"], Float[Tensor, "... dim"]],
    device: torch.device | str,
) -> Float[Tensor, "..."]:
    """Compute the log likelihood of a gaussian distribution element wise. Useful
    for learning mean and variance."""
    mu, var = params
    dim = mu.shape[-1]
    klog2pi = torch.tensor(2 * torch.pi, device=device).log() * dim
    prob = -0.5 * (klog2pi + var.log().sum(-1) + (y_true - mu).pow(2).div(var).sum(-1))
    return prob / dim

def gaussian_log_density(
    var: Float[Tensor, ""] | Float[Tensor, " dim"],
    y_true: Float[Tensor, "... dim"],
    mu: Float[Tensor, "... dim"],
    device: torch.device | str,
) -> Float[Tensor, "..."]:
    """Compute the log likelihood of a gaussian distribution element wise. Useful
    if the variance is fixed and known. To use it as a likelihood for training 
    CTVAE models, use likelihood = partial(gaussian_log_density, var).
    """
    log2pi = torch.tensor(2 * torch.pi, device=device).log()
    logvar = var.log()
    return -0.5 * (log2pi + logvar + (y_true - mu).pow(2).div(var))
