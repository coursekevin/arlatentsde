import torch
from torch import nn

def lorenz(sigma, beta, rho, t, x):
    x_dot = sigma * (x[..., 1] - x[..., 0])
    y_dot = x[..., 0] * (rho - x[..., 2]) - x[..., 1]
    z_dot = x[..., 0] * x[..., 1] - beta * x[..., 2]

    return [x_dot, y_dot, z_dot]


class LorenzModel(nn.Module):
    def __init__(self, rs):
        super().__init__()
        raw_params = torch.tensor([10.0, 8 / 3, 28.0])
        self.register_buffer("raw_params", raw_params)
        self.params = nn.Parameter(
            raw_params
            + torch.randn(*raw_params.shape, generator=torch.manual_seed(rs))
            * (0.2 * raw_params)
        )

    def forward(self, x, t):
        return torch.stack(lorenz(*self.params, t, x), dim=-1)