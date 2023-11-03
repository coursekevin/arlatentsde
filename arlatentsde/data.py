"""
Datasets and samplers for amortized reparametrization. All datasets
assume that observations are evenly spaced but this should be easy 
to extend if necessary.
"""
import os
from glob import glob
from typing import Sized
from functools import cached_property

import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, RandomSampler, Sampler
from torchvision import transforms

from .types import Img, ImgWin


class EvenlySpacedFrames(Dataset):
    """Dataset for video frames that are evenly spaced.

    Args:
        frame_dir (str): Path to directory containing frames.
        num_window (int): Number of frames to use as input.
        dt (float): Spacing between frames.
        transform (transforms.Compose): Transformation to apply to frames.
    """

    def __init__(
        self, frame_dir: str, num_window: int, dt: float, transform: transforms.Compose
    ):
        self.transform = transform
        self.image_filenames = sorted(glob(os.path.join(frame_dir, "*.png")))
        N = len(self.image_filenames)
        self.N = N
        self.M = num_window

        self.dt = dt
        self.t = torch.arange(self.N) * self.dt

    def __len__(self) -> int:
        return self.N - self.M

    def __getitem__(self, idx: int) -> tuple[Float[Tensor, ""], ImgWin, Img]:
        """Returns a timestamp, input image window, and input image.

        Note:
            the input image = input image window[0]
        """
        input_imgs = torch.stack(
            [  # type: ignore
                self.transform(Image.open(img).convert("L"))
                for img in self.image_filenames[idx : idx + self.M]
            ],
            dim=0,
        )
        image = input_imgs[0]
        return (self.t[idx], input_imgs, image)


class EvenlySpacedTensors(Dataset):
    """Dataset for tensors that are evenly spaced.
    Returns data in the same format as EvenlySpacedFrames"""

    t: Float[Tensor, " n_data"]
    y: Float[Tensor, "n_data dim"]
    dt: Float[Tensor, ""]

    def __init__(
        self,
        t: Float[Tensor, " n_data"],
        y: Float[Tensor, "n_data dim"],
        num_window: int,
    ) -> None:
        super().__init__()
        self.N = len(t)
        self.M = num_window
        assert torch.allclose(t[1:] - t[:-1], t[1] - t[0], atol=1e-4)
        self.dt = t[1] - t[0]
        # storing data
        self.t = t
        self.y = y

    def __len__(self) -> int:
        return self.N - self.M

    def __getitem__(self, idx: int) -> tuple[Float[Tensor, ""], ImgWin, Img]:
        """Returns a timestamp, input image window, and input image.

        Note:
            the input image = input image window[0]
        """
        input_tensors = torch.stack(
            [y_i for y_i in self.y[idx : idx + self.M]],
            dim=0,
        )
        tensor = input_tensors[0]
        return (self.t[idx], input_tensors, tensor)


class MultiEvenlySpacedTensors(Dataset):
    t: Float[Tensor, "n_traj n_data"]
    y: Float[Tensor, "n_traj n_data dim"]
    dt: Float[Tensor, ""]

    def __init__(
        self,
        t: Float[Tensor, "n_traj n_data"],
        y: Float[Tensor, "n_traj n_data dim"],
        num_window: int,
    ) -> None:
        super().__init__()
        dt = t[0][1] - t[0][0]
        self.M = num_window
        assert torch.allclose(dt, t[:, 1:] - t[:, :-1], atol=1e-5)
        self.dt = dt
        self.t = t
        self.y = y
        self.n_traj = len(t)
        self.n_data = len(t[0])

    @property
    def total_data(self) -> int:
        return self.n_traj * self.n_data

    def __len__(self) -> int:
        return self.n_traj * (self.n_data - self.M)

    def __getitem__(self, idx: int):
        n_cols = self.n_data - self.M + 1
        traj_id, data_id = idx // n_cols, idx % n_cols
        input_tensors = self.y[traj_id, data_id : data_id + self.M]
        tensor = input_tensors[0]
        return (self.t[traj_id, data_id], input_tensors, tensor)


def break_indices(inds: list[int], M: int) -> list[list[int]]:
    """breaks up a list of ints into a list of lists of length M
    i.e. break_indices([1,2,3,4,5], 2) -> [[1,2], [3,4], [5]]"""
    lp = 0
    rp = lp + M
    broken_list = []
    while rp < len(inds):
        broken_list.append(inds[lp:rp])
        lp += M
        rp += M
    broken_list.append(inds[lp:])
    return broken_list


def nested_indices(n_data, n_traj, M):
    indices = []
    for i in range(n_traj):
        inds = list(range(i * n_data, (i + 1) * n_data))
        indices.append(break_indices(inds, M))
    return indices


class MultiTemporalSampler(Sampler[list[int]]):
    def __init__(
        self,
        data_source: MultiEvenlySpacedTensors,
        time_window: int,
        generator=None,
        n_repeats: int = 1,
    ) -> None:
        self.data_source = data_source
        self.time_window = time_window
        self.generator = generator
        self.n_repeats = n_repeats
        self.sampler = RandomSampler(
            self.indices,
            replacement=False,
            num_samples=len(self.indices) * n_repeats,
            generator=generator,
        )

    @cached_property
    def indices(self) -> list[list[int]]:
        inds = nested_indices(
            self.data_source.n_data - self.data_source.M + 1,
            self.data_source.n_traj,
            self.time_window,
        )
        return [j for i in inds for j in i]

    def __iter__(self):
        yield from (self.indices[i] for i in self.sampler)

    def __len__(self) -> int:
        return len(self.indices) * self.n_repeats


class TemporalSampler(Sampler[list[int]]):
    """Dataset sampler that samples windows of data randomly from a dataset. For
    example, consider the dataset [1, 2, 3, 4, 5] with a time_window = 2. Then,
    the sampler will first break the dataset as [[1, 2], [3, 4], [5]]. Then sampler
    will randomly sample from the broken dataset.

    For smaller datasets, significant time is wasted between epochs. Setting
    n_repeats > 1 will sample from the broken dataset multiple times per epoch. When
    sampling more than once, it will sample the entire broken dataset before repeating.

    Args:
        data_source (Sized): Dataset to sample from.
        time_window (int): Number of frames to use as input.
        generator (torch.Generator, optional): Generator used for sampling.
        n_repeats (int, optional): Number of times to repeat the dataset. Defaults to 1.

    """

    def __init__(
        self, data_source: Sized, time_window: int, generator=None, n_repeats: int = 1
    ) -> None:
        self.data_source = data_source
        self.time_window = time_window
        self.generator = generator
        self.n_repeats = n_repeats
        self.sampler = RandomSampler(
            self.indices,
            replacement=False,
            num_samples=len(self.indices) * n_repeats,
            generator=generator,
        )

    @property
    def indices(self) -> list[list[int]]:
        return break_indices(list(range(len(self.data_source))), self.time_window)

    def __iter__(self):
        yield from (self.indices[i] for i in self.sampler)

    def __len__(self) -> int:
        return len(self.indices) * self.n_repeats
