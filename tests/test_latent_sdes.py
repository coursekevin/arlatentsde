import torch

from arlatentsde import latent_sdes

def test_fclatentsde():
    fc = latent_sdes.FCLatentSDE(2, [16, 32, 64], torch.nn.Tanh(), autonomous=True)
    z = torch.randn(10, 20, 30, 2)
    t = torch.randn(10, 20, 30, 1)
    assert fc(z, t).shape == (10, 20, 30, 2)
    # check that the sde is autonomous
    def get_grad(fc):
        t.requires_grad_(True)
        t.grad = torch.zeros_like(t)
        z.requires_grad_(True)
        fc(z,t).sum().backward()
        return t.grad.sum().item() 
    assert get_grad(fc) == 0.0 

    fc = latent_sdes.FCLatentSDE(2, [16, 32, 64], torch.nn.Tanh(), autonomous=False)
    assert fc(z, t).shape == (10, 20, 30, 2)
    assert get_grad(fc) != 0.0

