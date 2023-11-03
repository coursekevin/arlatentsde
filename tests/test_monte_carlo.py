import math
from functools import partial

import torch

from arlatentsde import monte_carlo


def test_vmap_safe_stratified_mc():
    torch.manual_seed(0)
    eps = [
        torch.linspace(0, 1, 10).view(-1, 1),
    ]

    def f(eps, t):
        scale = eps[0]
        return scale + torch.sqrt(1 - t.pow(2))

    def integrable_fn(eps):
        return partial(f, eps)

    t_bounds = (-1.0, 1.0)
    settings = monte_carlo.MCQuadSettings(n_samples=1000)
    quad = monte_carlo.vmap_safe_stratified_mc(
        integrable_fn, eps, [eps[0]], t_bounds, settings
    )
    assert torch.allclose(quad, (math.pi / 2 + 2 * eps[0].view(-1)), atol=1e-2)


if __name__ == "__main__":
    test_vmap_safe_stratified_mc()
