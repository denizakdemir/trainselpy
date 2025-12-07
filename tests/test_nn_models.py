import pytest
import torch
import torch.nn as nn
from trainselpy.nn_models import VAE, Generator, Discriminator, DecisionStructure, compute_gradient_penalty

@pytest.fixture
def structure():
    # Example structure:
    # 5 binary
    # 2 permutations of 3 items (3*3*2 = 18 ? No. n=5, k=3? Spec: (n, k))
    # Let's say we have ONE permutation set of n=4 items, selecting k=2 items. Encoding size = 4*2 = 8.
    # 3 continuous
    return DecisionStructure(
        binary_dim=5,
        permutation_dims=[(4, 2)], 
        continuous_dim=3
    )

def test_decision_structure(structure):
    # Total dim = 5 + (4*2) + 3 = 16
    assert structure.total_dim == 16
    
    # Create fake raw output
    batch_size = 2
    raw = torch.randn(batch_size, 16)
    decoded = structure.decode_raw_output(raw)
    
    assert decoded.shape == (batch_size, 16)
    
    # Check binary part is in [0, 1] (sigmoid)
    assert (decoded[:, :5] >= 0).all() and (decoded[:, :5] <= 1).all()
    
    # Check permutation part sums to 1 per slot?
    # structure.permutation_dims = [(4, 2)]
    # perm_start = 5, perm_end = 5+8=13
    # Reshaped to (batch, 2, 4), softmax on dim 2.
    # So sum over dim 2 should be 1.
    perm_part = decoded[:, 5:13].view(batch_size, 2, 4)
    sums = perm_part.sum(dim=2)
    assert torch.allclose(sums, torch.ones_like(sums))

def test_vae_forward(structure):
    vae = VAE(structure, latent_dim=8)
    x = torch.randn(10, 16)
    # Ensure inputs are somewhat valid (e.g. probs for permutations) but VAE takes raw input or processed?
    # Actually VAE reconstruction loss expects `x` to be the target.
    # If using BCE, target should be 0-1.
    # If using CrossEntropy, target should be one-hot or indices?
    # My implementation assumes x is correct representation (one-hot for perm, 0/1 for binary).
    
    # Make a valid-ish input x
    x_bin = (torch.rand(10, 5) > 0.5).float()
    
    # x_perm: random one-hot
    # for each of the 2 slots, pick one of 4
    x_perm_list = []
    for _ in range(10):
        # slot 1
        s1 = torch.zeros(4)
        s1[torch.randint(0, 4, (1,))] = 1
        # slot 2
        s2 = torch.zeros(4)
        s2[torch.randint(0, 4, (1,))] = 1
        x_perm_list.append(torch.cat([s1, s2]))
    x_perm = torch.stack(x_perm_list)
    
    x_cont = torch.randn(10, 3)
    
    x_in = torch.cat([x_bin, x_perm, x_cont], dim=1)
    
    recon_x, mu, logvar = vae(x_in)
    
    assert recon_x.shape == x_in.shape
    assert mu.shape == (10, 8)
    assert logvar.shape == (10, 8)
    
    loss, recon, kld = vae.loss_function(recon_x, x_in, mu, logvar)
    assert loss.item() > 0
    
    # Test backward
    loss.backward()

def test_gan_shapes(structure):
    cond_dim = 4
    gen = Generator(structure, cond_dim, latent_dim=32)
    disc = Discriminator(structure, cond_dim)
    
    batch_size = 5
    noise = torch.randn(batch_size, 32)
    cond = torch.randn(batch_size, cond_dim)
    
    fake_data = gen(noise, cond)
    assert fake_data.shape == (batch_size, structure.total_dim)
    
    score = disc(fake_data, cond)
    assert score.shape == (batch_size, 1)

def test_gradient_penalty(structure):
    cond_dim = 4
    disc = Discriminator(structure, cond_dim)
    batch_size = 5
    
    real_data = torch.randn(batch_size, structure.total_dim)
    fake_data = torch.randn(batch_size, structure.total_dim)
    cond = torch.randn(batch_size, cond_dim)
    
    # Forward pass through disc inside the function implies we need to pass labels?
    # compute_gradient_penalty(discriminator, real_samples, fake_samples, labels, device='cpu')
    gp = compute_gradient_penalty(disc, real_data, fake_data, cond)
    
    assert gp.item() >= 0
    
    # Gradient penalty should allow backward
    loss = gp
    loss.backward()
