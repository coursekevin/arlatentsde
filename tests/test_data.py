"""
Some basic testing for the data utils created by GPT4.
This is definitely not an exhaustive test suite but should be enough
to catch major bugs.
"""
import pytest
import tempfile
from glob import glob
import shutil
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from arlatentsde import data


@pytest.fixture
def frame_dir():
    dirpath = tempfile.mkdtemp()
    for i in range(10):
        img = Image.fromarray(np.random.randint(0, 256, (10, 10), np.uint8))
        img.save(f"{dirpath}/{i}.png")
    yield dirpath
    shutil.rmtree(dirpath)


def test_EvenlySpacedFrames(frame_dir):
    transform = transforms.Compose([transforms.ToTensor()])
    dt = 0.1
    dataset = data.EvenlySpacedFrames(frame_dir, 5, dt, transform)
    assert len(dataset) == 5
    timestamp, img_window, image = dataset[0]
    img_files = sorted(glob(frame_dir + "/*.png"))
    img_exp = torch.stack(
        [transform(Image.open(img)) for img in img_files[:5]]  # type:ignore
    )
    assert timestamp == 0
    assert len(img_window) == 5
    assert torch.all(img_window[0] == image)
    assert torch.all(img_exp == img_window)


def test_EvenlySpacedTensors():
    t = torch.arange(10).float()
    y = torch.rand(10, 50)
    dataset = data.EvenlySpacedTensors(t, y, 5)

    assert len(dataset) == 5
    _, tensor_window, tensor = dataset[0]
    assert len(tensor_window) == 5
    assert torch.all(tensor_window[0] == tensor)
    assert torch.all(tensor_window == y[:5])


def test_MultiEvenlySpacedTensors():
    t = torch.arange(10).reshape(2, 5).float()
    y = torch.rand(2, 5, 3, 64, 64)
    dataset = data.MultiEvenlySpacedTensors(t, y, 2)

    assert len(dataset) == 6
    _, tensor_window, tensor = dataset[0]
    assert len(tensor_window) == 2
    assert torch.all(tensor_window[0] == tensor)
    assert torch.all(y[0, :2] == tensor_window)


def test_break_indices():
    indices = list(range(10))
    broken_list = data.break_indices(indices, 3)
    assert broken_list == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert len(broken_list) == 4
    assert broken_list[0] == [0, 1, 2]
    assert broken_list[-1] == [9]


def test_nested_indices():
    nested_inds = data.nested_indices(10, 2, 3)
    expected_nested_inds = [
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18], [19]],
    ]
    assert nested_inds == expected_nested_inds
    assert len(nested_inds) == 2
    assert len(nested_inds[0]) == 4
    assert len(nested_inds[1]) == 4


def test_TemporalSampler():
    source_data = torch.tensor(list(range(20)))
    sampler = data.TemporalSampler(source_data, time_window=5, n_repeats=1)
    batches = list(iter(sampler))

    # check that each integer appears exactly once in the samples
    flattened_samples = [idx for batch in batches for idx in batch]
    assert set(flattened_samples) == set(source_data.tolist())

    # check that each batch is in increasing order
    for batch in batches:
        assert all(i < j for i, j in zip(batch, batch[1:]))


def test_MultiTemporalSampler():
    t = torch.arange(5).unsqueeze(0).repeat(5, 1)  # time points for 5 trajectories
    y = torch.randn(5, 5, 3, 64, 64)  # 5 trajectories, each with 5 frames

    dataset = data.MultiEvenlySpacedTensors(t, y, num_window=3)
    sampler = data.MultiTemporalSampler(dataset, time_window=2)

    assert len(sampler) == 10  # Should match the length of dataset

    for idx in sampler:
        print(idx)
        # assert 0 <= idx < len(dataset)  # Each index should be valid for the dataset
