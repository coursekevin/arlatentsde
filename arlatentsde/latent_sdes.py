import torch
from torch import nn
from torch import Tensor
from jaxtyping import Float
from typing import Protocol


class LatentSDE(Protocol):
    def __call__(
        self, z: Float[Tensor, "... latent_dim"], t: Float[Tensor, "... 1"]
    ) -> Float[Tensor, "... latent_dim"]:
        ...


class FCLatentSDE(nn.Module):
    latent_dim: int
    layer_description: list[int]
    nonlinearity: nn.Module
    autonomous: bool

    def __init__(
        self,
        latent_dim: int,
        layer_description: list[int],
        nonlinearity: nn.Module,
        autonomous: bool = True
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.autonomous = autonomous
        if autonomous:
            first_dim = latent_dim
        else:
            first_dim = latent_dim + 2
        layers = [(nn.Linear(first_dim, layer_description[0]), nonlinearity)]
        layers += [
            (nn.Linear(layer_description[i], layer_description[i + 1]), nonlinearity)
            for i in range(len(layer_description) - 1)
        ]
        layers = [mod for layer in layers for mod in layer]
        layers.append(nn.Linear(layer_description[-1], latent_dim))
        self.layers = nn.Sequential(*layers)

    def forward(
        self, z: Float[Tensor, "... latent_dim"], t: Float[Tensor, "... 1"]
    ) -> Float[Tensor, "... latent_dim"]:
        if self.autonomous:
            return self.layers(z)
        else:
            # suggested here: https://github.com/google-research/torchsde/blob/master/examples/latent_sde.py
            # Positional encoding in transformers for time-inhomogeneous posterior.
            t = torch.ones(*z.shape[:-1], 1, device=z.device).mul(t)
            return self.layers(torch.cat((torch.cos(t), torch.sin(t), z),dim=-1))