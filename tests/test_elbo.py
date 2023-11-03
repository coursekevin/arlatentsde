"""
Testing for elbo.py utilities.
"""
import torch
from torch.func import vmap, jacfwd # type: ignore
from functools import wraps
import pytest

from arlatentsde import elbo


def test_reparam_tsfm():
    mu = torch.randn(10, 5) + 1.0
    logvar = torch.randn(10, 5)
    epsilon = torch.randn(10, 5)
    z = elbo.reparam_tsfm(mu, logvar, epsilon)
    assert torch.allclose(z, mu + torch.exp(0.5 * logvar) * epsilon)


def test_rsample_latent():
    torch.manual_seed(0)
    mu = torch.randn(5)
    logvar = torch.randn(5)
    nsamples = 1024
    zs = elbo.rsample_latent(nsamples, mu, logvar)
    assert torch.allclose(zs.mean(0), mu, atol=0.1)


def test_solve_lyapunov():
    def logvar(t):
        return torch.stack([torch.sin(t), torch.cos(t)], dim=1)

    def var(t):
        return logvar(t).exp()

    def dlogvar(t):
        return torch.stack([torch.cos(t), -torch.sin(t)], dim=1)

    brownian = torch.randn(2).pow(2)
    tquad = torch.rand(10)
    lyap = elbo.solve_lyapunov(logvar(tquad), dlogvar(tquad), brownian)
    dS = vmap(jacfwd(var))(tquad.unsqueeze(1)).squeeze()
    lyap_man = 0.5 * (brownian - dS) / var(tquad)
    assert torch.allclose(lyap, lyap_man)


def squeeze_wrap(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        return fn(*args, **kwargs).squeeze()

    return wrapped


@pytest.fixture
def residual_setup():
    epsilon = torch.randn(2)

    def mu(t):
        return torch.stack([t + torch.sin(t), t.pow(2) + torch.cos(t)], dim=1)

    def logvar(t):
        return torch.stack([torch.sin(t), torch.cos(t)], dim=1)

    def dlogvar(t):
        return torch.stack([torch.cos(t), -torch.sin(t)], dim=1)

    def z(t):
        output = torch.cat([mu(t), logvar(t)], dim=1).squeeze()
        return output, output

    def dz(t):
        z_eval = z(t)[0]
        dz_eval = torch.cat(
            [1 + torch.cos(t), 2 * t - torch.sin(t), torch.cos(t), -torch.sin(t)], dim=1
        )
        return dz_eval, z_eval

    def f(z, t):
        return torch.stack([torch.cos(t), -torch.sin(t)], dim=1).squeeze()

    t = torch.rand(10)
    brownian = torch.randn(2).pow(2)
    return epsilon, mu, logvar, dlogvar, dz, f, t, brownian


def test_residual(residual_setup):
    epsilon, mu, logvar, dlogvar, dz, f, t, brownian = residual_setup
    # library function
    res = elbo.residual(epsilon, brownian, dz, f, t)
    # computing it manually using already tested functions
    a = elbo.solve_lyapunov(logvar(t), dlogvar(t), brownian)
    zs = elbo.reparam_tsfm(mu(t), logvar(t), epsilon)
    vmap(jacfwd(mu))(t.unsqueeze(1)).squeeze()
    dmu = vmap(jacfwd(mu))(t.unsqueeze(1)).squeeze()
    res_man = a * (mu(t) - zs) + (dmu - f(zs, t))
    assert torch.allclose(res, res_man)


def test_normed_residual(residual_setup):
    epsilon, _, _, _, dz, f, t, brownian = residual_setup
    # computing it manually using already tested functions
    res = elbo.residual(epsilon, brownian, dz, f, t)
    nres_man = torch.sum(res.pow(2) / brownian, dim=-1)
    # library function
    nres = elbo.normed_residual(epsilon, brownian, dz, f, t)
    assert torch.allclose(nres, nres_man)


def test_bernoulli_log_density():
    bcetorch = torch.nn.BCEWithLogitsLoss(reduction="none")
    b_true = torch.tensor([0.0, 1.0, 1.0, 0.0])
    logits = torch.randn(4)
    bce_loss = bcetorch(logits, b_true)
    assert torch.allclose(bce_loss, -elbo.bernoulli_log_density(b_true, logits, "cpu"))


def test_generative_gaussian_log_density():
    mu, var = torch.randn(10), torch.randn(10).exp()
    y_true = torch.randn(10)
    log_density = elbo.generative_gaussian_log_density(y_true, [mu, var], "cpu")
    torch_log_norm = torch.distributions.Normal(mu, var.sqrt()).log_prob(y_true).mean()
    assert torch.allclose(log_density, torch_log_norm)


def test_gaussian_log_density():
    mu, var = torch.randn(10), torch.randn(10).exp()
    y_true = torch.randn(10)
    log_density = elbo.gaussian_log_density(var, y_true, mu, "cpu")
    torch_log_norm = torch.distributions.Normal(mu, var.sqrt()).log_prob(y_true)
    assert torch.allclose(log_density, torch_log_norm)


