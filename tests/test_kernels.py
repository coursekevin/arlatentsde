from functools import partial

import torch
from torch import nn
from torch.nn.functional import softplus

from arlatentsde import kernels
from arlatentsde.kernels import DeepKernel


def test_inverse_softplus():
    torch.manual_seed(23)
    x = torch.linspace(-10, 10, 100)
    assert torch.allclose(kernels.inverse_softplus(softplus(x)), x, atol=1e-3)


def squared_exp_kernel(t1, t2):
    diff = t1.view(-1, 1) - t2.view(1, -1)
    diff.pow_(2).div_(1.0).mul_(-1.0).exp_()
    return diff


def test_interpolate():
    tnode = torch.linspace(0, 1, 20, dtype=torch.float64)
    ynode = torch.sin(tnode * 2 * torch.pi).unsqueeze(1)
    tq = torch.linspace(0, 1, 100, dtype=torch.float64)
    kern_fn = partial(squared_exp_kernel, tnode)
    assert kern_fn(tq).shape == (len(tnode), len(tq))
    kern_chol = torch.linalg.cholesky(kern_fn(tnode) + 1e-6 * torch.eye(len(tnode)))
    yq = kernels.interpolate(tq, ynode, kern_fn, kern_chol)
    assert torch.allclose(yq.ravel(), torch.sin(tq * 2 * torch.pi), atol=1e-2)


def test_deep_kernel_init():
    layer_description = [10, 5, 1]
    nonlinearity = nn.ReLU()
    offset, scale = DeepKernel.get_rescale_params(32, 0.1)

    dk = DeepKernel(layer_description, nonlinearity, offset, scale)
    assert dk.layer_description == layer_description
    assert torch.allclose(
        dk.lens, softplus(kernels.inverse_softplus(torch.tensor(1e-2)))
    )
    assert torch.allclose(
        dk.var, softplus(kernels.inverse_softplus(torch.tensor(1e-5)))
    )
    assert torch.allclose(
        dk.sigma, softplus(kernels.inverse_softplus(torch.tensor(1.0)))
    )


def test_deep_kernel_rescale():
    layer_description = [10, 5, 1]
    nonlinearity = nn.ReLU()
    offset, scale = DeepKernel.get_rescale_params(32, 0.1)
    dk = DeepKernel(layer_description, nonlinearity, offset, scale)

    input_tensor = torch.rand(10)
    rescaled_tensor = dk.rescale(input_tensor)
    assert torch.allclose(rescaled_tensor, (input_tensor - offset) / scale, atol=1e-4)


def test_deep_kernel_forward():
    layer_description = [1, 10, 5, 1]
    nonlinearity = nn.ReLU()
    offset, scale = DeepKernel.get_rescale_params(32, 0.1)
    dk = DeepKernel(layer_description, nonlinearity, offset, scale)

    x1 = torch.rand(10)
    x2 = torch.rand(5)
    result = dk.forward(x1, x2)
    assert result.shape == (10, 5)


def test_deep_kernel_get_rescale_params():
    batch_size = 32
    dt = 0.1
    offset, scale = DeepKernel.get_rescale_params(batch_size, dt)
    assert torch.allclose(offset, torch.tensor(0.5 * (batch_size - 1) * dt))
    assert torch.allclose(scale, torch.tensor(0.5 * (batch_size - 1) * dt))
