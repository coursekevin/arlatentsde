import math
from functools import partial

import torch
from torch.func import jacfwd, vmap  # type: ignore
from torch import Tensor

from arlatentsde import autoencoders


def test_squeeze_wrap():
    @autoencoders.squeeze_wrap
    def fn():
        return (torch.randn(1, 2, 1, 4), torch.randn(3, 1))

    x, y = fn()
    assert x.shape == (2, 4)
    assert y.shape == (3,)


def test_return_first():
    @autoencoders.return_first
    def fn():
        return (torch.randn(1, 2, 1, 4), torch.randn(3, 1))

    x = fn()
    assert x.shape == (1, 2, 1, 4)


class SquaredExpKernel:
    def __call__(self, x1: Tensor, x2: Tensor) -> Tensor:
        diff = x1.view(-1, 1) - x2.view(1, -1)
        diff.pow_(2).div_(1e-2).mul_(-1.0).exp_()
        return diff

    @property
    def var(self):
        return torch.tensor(1e-5)


def test_encoder():
    def encoding_fn(x):
        return torch.cat(
            [torch.sin(x * 2 * math.pi), torch.cos(x * 2 * math.pi)], dim=-1
        )

    kernel = SquaredExpKernel()

    n_data = 500
    x = torch.linspace(0, 1, n_data).view(-1, 1)

    encoder = autoencoders.Encoder(float(x[1] - x[0]), kernel, encoding_fn)
    # testing forward
    t = torch.linspace(0, 1, n_data)
    t_node, z, dz = encoder(t, x)
    assert torch.allclose(t_node, t - t[0])
    assert t_node.shape == (n_data,)
    assert z(t_node).shape == (n_data, 2)
    assert torch.allclose(z(t), encoding_fn(x), atol=1e-1)
    dz_eval, _ = dz(t)
    dz_man = vmap(jacfwd(z))(t).squeeze()
    assert torch.allclose(dz_eval, dz_man)

    # test interp
    hidden = encoding_fn(x)
    frozen_kernel = partial(kernel, t_node)
    eye = torch.eye(n_data)
    kern_chol = torch.linalg.cholesky(frozen_kernel(t_node) + kernel.var * eye)
    tq = torch.rand(10).view(-1, 1)
    interp = encoder.interp_latent(t[0], hidden, frozen_kernel, kern_chol, tq)
    assert interp[0].shape == (10, 2)
    assert interp[1].shape == (10, 2)
    torch.allclose(interp[0], encoding_fn(tq))


def test_encoding_fcnn():
    enc = autoencoders.EncodingFCNN(50, 3, 2, [10, 10], torch.nn.ReLU())
    x = torch.randn(32, 3, 50)
    mu, logvar = enc(x).chunk(2, dim=-1)
    assert mu.shape == (32, 2)
    assert logvar.shape == (32, 2)
    n_params = 50 * 3 * 10 + 10 + 10 * 10 + 10 + 10 * 4 + 4
    assert sum(p.numel() for p in enc.parameters()) == n_params


def test_decoding_fcnn():
    dec = autoencoders.DecodingFCNN(50, 2, [10, 10], torch.nn.ReLU())
    z = torch.randn(20, 32, 2)
    mu, var = dec(z)
    assert mu.shape == (20, 32, 50)
    assert var.shape == (20, 32, 50)
    n_params = 2 * 10 + 10 + 10 * 10 + 10 + 10 * 50 * 2 + 50 * 2
    assert sum(p.numel() for p in dec.parameters()) == n_params


def test_encoding_cnn():
    mpool_desc = dict(kernel_size=2, stride=2)
    conv_arch = [
        autoencoders.ConvLayer(16, 3, maxpool=mpool_desc),
        autoencoders.ConvLayer(32, 3, maxpool=mpool_desc),
    ]
    fc_arch = [64, 32]
    in_dims = (50, 30)
    enc = autoencoders.EncodingCNN(3, in_dims, conv_arch, fc_arch, 10, torch.nn.ReLU())
    x = torch.randn(64, 32, 3, *in_dims)
    assert enc(x).shape == (64, 32, 10 * 2)


def test_decoding_cnn():
    mpool_desc = dict(kernel_size=2, stride=2)
    conv_arch = [
        autoencoders.ConvLayer(16, 5, stride=2, maxpool=mpool_desc),
        autoencoders.ConvLayer(32, 5, stride=2, maxpool=mpool_desc),
    ]
    in_dims = (256, 512)
    enc = autoencoders.EncodingCNN(3, in_dims, conv_arch, [64, 32], 10, torch.nn.ReLU())
    print(enc)
    dec = autoencoders.DecodingCNN(
        3, in_dims, list(reversed(conv_arch)), [32, 64], 10, torch.nn.ReLU()
    )
    print(dec)
    x = torch.randn(8, 6, 3, *in_dims)
    z = enc(x)
    mu, logvar = z.chunk(2, dim=-1)
    assert dec(mu)[0].shape == x.shape


if __name__ == "__main__":
    test_encoder()
