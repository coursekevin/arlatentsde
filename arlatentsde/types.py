from jaxtyping import Float
import torch
from torch import Tensor
from typing import Callable, Union, List, Tuple, ClassVar, Dict, Protocol


class GenericDataclass(Protocol):
    """
    type hinting for generic dataclasses: https://stackoverflow.com/a/55240861/12184561
    """

    __dataclass_fields__: ClassVar[Dict]


# window of images for autoencoder
ImgWin = Float[Tensor, "batch c res1 res2"]
# single image
Img = Float[Tensor, "batch res1 res2"]
# a vector of time stamps
TimeVec = Float[Tensor, "batch 1"]
# a function that acts on a time vector
TimeFn = Callable[
    [TimeVec],
    Tuple[Float[Tensor, "batch 2*latent_dim"], Float[Tensor, "batch 2*latent_dim"]],
]
# a variational function (i.e. mean and logvar)
VariationalFn = Callable[[TimeVec], Float[Tensor, "batch latent_dim"]]


# drift function definition
class DriftFn(Protocol):
    def __call__(
        self,
        z: Float[Tensor, "... latent_dim"],
        t: Float[Tensor, "... 1"],
    ) -> Float[Tensor, "... latent_dim"]:
        ...


# kernel functions
FrozenKernelFn = Callable[[Float[Tensor, " n"]], Float[Tensor, "m n"]]

# latent state estimate
LatentStateFn = Callable[[Float[Tensor, " query"]], Float[Tensor, "query 2*latent_dim"]]
# first is the derivative, second is the latent state
LatentStateDerivativeFn = Tuple[LatentStateFn, LatentStateFn]

# log likelihood function
Device = Union[torch.device, str]


class LoglikeFn(Protocol):
    def __call__(
        self,
        y_true: Float[Tensor, "..."],
        params: tuple[Float[Tensor, "..."], ...],
        device: Device,
    ) -> Float[Tensor, "..."]:
        ...


# integrable function
IntegrableFn = Callable[[Float[Tensor, " nquad"]], Float[Tensor, " nquad"]]
# quadrature
Scalar = float | Tensor | int
BndType = Union[tuple[Scalar, Scalar], List[Scalar]]
STensor = Float[Tensor, " n_samples ..."]
IntegrableFnInput = Float[Tensor, " ..."] | list[Float[Tensor, " ..."]]
# vmap quad rule
VmapQuadFn = Callable[
    [
        Callable[[IntegrableFnInput], IntegrableFn],
        STensor | list[STensor],
        IntegrableFnInput,
        BndType,
        GenericDataclass,
    ],
    Float[Tensor, "n_samples dim"] | Float[Tensor, " n_samples"],
]
