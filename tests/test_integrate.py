import torch
from torch.func import vmap, jacfwd # type: ignore

from arlatentsde import integrate


class ExactPrior:
    _params = torch.tensor([0.1, 0.1])

    def __call__(self, n_samples: int):
        return self._params.repeat(n_samples, 1)

    def kl_divergence(self):
        return torch.tensor(0.0)


def test_solve_sde():
    torch.manual_seed(0)
    n_samples = 512

    brownian = ExactPrior()

    # exact covariance
    def S(t):
        return torch.stack(
            [torch.sin(t * 2 * torch.pi) + 1.1, torch.cos(t * 2 * torch.pi) + 1.1]
        )

    # exact mean
    def m(t):
        return torch.stack([t**1.2 + 1, t + torch.sin(t)])

    def A(t):
        return 0.5 * (brownian._params - jacfwd(S)(t)) / S(t)

    def b(t):
        return jacfwd(m)(t) + A(t) * m(t)

    def drift(z, t):
        return -A(t) * z + b(t)

    t_eval = torch.linspace(0, torch.pi, 100)
    # n_samples = 100
    z0 = m(t_eval[0]) + S(t_eval[0]).sqrt() * torch.randn(n_samples, 2)
    sde_kwargs = dict(dt=0.5e-1, atol=1e-6, rtol=1e-4)
    z = integrate.solve_sde(drift, brownian, z0, t_eval, **sde_kwargs)
    z = z.float()
    assert z.shape == (n_samples, len(t_eval), 2)
    assert (z.mean(0) - vmap(m)(t_eval)).pow(2).mean().sqrt().item() < 1e-1
    assert (z.std(0) - vmap(S)(t_eval).sqrt()).pow(2).mean().sqrt().item() < 1e-1
