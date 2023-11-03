from typing import Union, Protocol, Optional
from dataclasses import dataclass

import torch
from torch import Tensor
from torch import nn
from jaxtyping import Float


from .variationalsparsebayes.svi_half_cauchy import LogNormalMeanFieldVariational
from .variationalsparsebayes.svi_half_cauchy import (
    SVIHalfCauchyPrior as HalfCauchyPrior,
)


class Prior(Protocol):
    def __call__(self, n_samples: int) -> Float[Tensor, "n_samplees n_params"]:
        ...

    def kl_divergence(self) -> Float[Tensor, ""]:
        ...


@dataclass(frozen=True)
class LogNormalPriorSetup:
    mu_prior: Tensor
    log_sigma_prior: Tensor
    mu_init: Tensor
    log_sigma_init: Tensor

    def get_var_distn(self):
        return LogNormalPrior(
            mu_prior=self.mu_prior,
            log_sigma_prior=self.log_sigma_prior,
            mu_init=self.mu_init,
            log_sigma_init=self.log_sigma_init,
        )


class LogNormalPrior(nn.Module):
    """
    Lognormal variational distribution with a log normal prior.

    Args:
        mu_prior (Tensor): prior on log normal mean
        log_sigma_prior (Tensor): prior on log normal stdev
        mu_init (Tensor): initial posterior log normal mean
        log_sigma_init (Tensor): initial posterior log normal stdev
    """

    mu_prior: Tensor
    log_sigma_prior: Tensor

    def __init__(
        self,
        mu_prior: Float[Tensor, " n_params"],
        log_sigma_prior: Float[Tensor, " n_params"],
        mu_init: Optional[Float[Tensor, " n_params"]] = None,
        log_sigma_init: Optional[Float[Tensor, " n_params"]] = None,
    ) -> None:
        super().__init__()
        n_params = len(mu_prior)
        if mu_init is None:
            mu_init = torch.zeros(n_params)
        if log_sigma_init is None:
            log_sigma_init = torch.ones(n_params).log()
        self.var_distn = LogNormalMeanFieldVariational(
            mu_init=mu_init, log_sigma_init=log_sigma_init
        )
        self.register_buffer("log_sigma_prior", log_sigma_prior)
        self.register_buffer("mu_prior", mu_prior)
        self._check_params()

    def _check_params(self):
        err_msg = "prior and init params must have the same shape."
        assert len(self.mu_prior) == len(self.log_sigma_prior), err_msg
        assert len(self.var_distn.mu) == len(self.var_distn.log_sigma), err_msg
        assert len(self.mu_prior) == len(self.var_distn.mu), err_msg

    def get_reparam_weights(
        self, n_samples: int
    ) -> Float[Tensor, "n_samples n_params"]:
        """
        Generates n_samples from the variational distribution
        """
        return self.var_distn(n_samples)

    def forward(self, n_samples: int) -> Float[Tensor, "n_samples n_params"]:
        """Generates n_samples form the variational distribution"""
        return self.get_reparam_weights(n_samples)

    def kl_divergence(self) -> Float[Tensor, ""]:
        """
        Computes the KL divergence for the approximating posteriors

        Returns:
            Tensor: kl divergence
        """
        sigma_prior = torch.exp(self.log_sigma_prior)
        return (
            (self.log_sigma_prior.sum() - self.var_distn.log_sigma.sum())
            - self.var_distn.d / 2
            + 0.5
            * (self.var_distn.log_sigma.exp().square().div(sigma_prior.square())).sum()
            + 0.5 * ((self.var_distn.mu - self.mu_prior).div(sigma_prior)).pow(2).sum()
        )
