"""Sanity checks that need no dataset: equivariance, masking, one train step.

    python test_smoke.py
"""

from __future__ import annotations

import torch

from MolEDM.diffusion import EquivariantDiffusion, remove_mean
from MolEDM.egnn import EGNNDynamics

torch.manual_seed(0)
K, B, N = 5, 4, 9


def fake_batch():
    sizes = torch.tensor([9, 7, 5, 3])
    mask = (torch.arange(N)[None, :] < sizes[:, None]).float()
    x = remove_mean(torch.randn(B, N, 3) * 1.5, mask)
    h = torch.nn.functional.one_hot(torch.randint(0, K, (B, N)), K).float()
    return x, h * mask[..., None], mask


def random_rotation():
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:  # keep it a proper rotation
        q[:, 0] *= -1
    return q


def test_equivariance():
    dyn = EGNNDynamics(K, hidden=32, n_layers=3)
    x, h, mask = fake_batch()
    t = torch.rand(B)
    eps_x, eps_h = dyn(t, x, h, mask)

    R = random_rotation()
    eps_x_r, eps_h_r = dyn(t, x @ R, h, mask)
    rot_err = (eps_x_r - eps_x @ R).abs().max().item()
    inv_err = (eps_h_r - eps_h).abs().max().item()
    print(f"  rotation equivariance err {rot_err:.2e}   type invariance err {inv_err:.2e}")
    assert rot_err < 1e-4 and inv_err < 1e-4

    # translating the input must not change anything (zero-CoM subspace)
    shift = torch.randn(1, 1, 3)
    eps_x_t, _ = dyn(t, (x + shift) * mask[..., None], h, mask)
    trans_err = (eps_x_t - eps_x).abs().max().item()
    print(f"  translation invariance err {trans_err:.2e}")
    assert trans_err < 1e-4


def test_masking_and_com():
    dyn = EGNNDynamics(K, hidden=32, n_layers=3)
    x, h, mask = fake_batch()
    eps_x, eps_h = dyn(torch.rand(B), x, h, mask)
    pad = (1 - mask)[..., None]
    assert (eps_x * pad).abs().max() < 1e-6, "padding leaked into coord output"
    assert (eps_h * pad).abs().max() < 1e-6, "padding leaked into type output"
    com = (eps_x * mask[..., None]).sum(1).abs().max().item()
    print(f"  max |centre of mass| of output {com:.2e}")
    assert com < 1e-4


def test_train_step_and_sample():
    model = EquivariantDiffusion(EGNNDynamics(K, hidden=32, n_layers=3), K, timesteps=20)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, h, mask = fake_batch()
    first = last = None
    for i in range(30):
        loss = model.loss(x, h, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    print(f"  overfit 4 molecules: loss {first:.3f} -> {last:.3f}")
    assert last < first

    xs, one_hot, _ = model.sample(mask)
    assert xs.shape == (B, N, 3) and one_hot.shape == (B, N, K)
    assert torch.isfinite(xs).all(), "sampling produced NaN/Inf"
    assert (one_hot.sum(-1) == mask).all(), "atom types decoded outside the mask"
    print(f"  sampled ok, coord std {xs[mask.bool()].std():.2f}")


def test_sampler_calibration():
    """Validate the reverse process independently of the network.

    For data x ~ N(0, I) the optimal denoiser is known exactly:
    E[eps | z_t] = sigma_t * z_t.  Plugging it in, the sampler must return
    samples with unit variance -- if the schedule or the ancestral step has its
    variance bookkeeping wrong, the scale drifts away from 1.
    """
    model = EquivariantDiffusion(EGNNDynamics(K, hidden=8, n_layers=1), K, timesteps=500)

    class OptimalGaussianDenoiser(torch.nn.Module):
        def forward(self, t, z_x, z_h, node_mask):
            t_int = (t[0] * model.T).round().long()
            sigma = model.sigmas[t_int]
            return remove_mean(sigma * z_x, node_mask), sigma * z_h * node_mask[..., None]

    model.dynamics = OptimalGaussianDenoiser()
    mask = torch.ones(512, 6)
    x, _, _ = model.sample(mask)

    std = x[mask.bool()].std().item()
    print(f"  recovered std {std:.3f} (target 1.0, minus the zero-CoM d.o.f.)")
    assert 0.85 < std < 1.15, f"sampler variance is off: std={std}"


if __name__ == "__main__":
    for fn in (test_equivariance, test_masking_and_com, test_sampler_calibration,
               test_train_step_and_sample):
        print(fn.__name__)
        fn()
    print("\nall smoke tests passed")
