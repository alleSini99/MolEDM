"""Equivariant denoising diffusion over molecules (coordinates + atom types).

Follows Hoogeboom et al., "Equivariant Diffusion for Molecule Generation in 3D"
(EDM, 2022), trained with the simple eps-prediction objective:

    z_t = alpha_t * [x, h] + sigma_t * eps,      L = || eps - eps_hat(z_t, t) ||^2

Two things make this different from an image diffusion model:

1. Coordinates live in the zero-centre-of-mass subspace.  A distribution over R^3n
   cannot be invariant to translation, so all noise -- and the model's coordinate
   output -- is projected to have zero mean.
2. Atom types are diffused as a continuous relaxation of their one-hot vectors
   (scaled down, since a one-hot is "louder" than a coordinate) and decoded by an
   argmax at the last step.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Noise Schedules
# --------------------------------------------------------------------------- #
def _clip_noise_schedule(alphas2: np.ndarray, clip_value: float = 0.001) -> np.ndarray:
    """Bound alpha_t/alpha_{t-1} from below to avoid destroying the signal from one step."""
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
    step = np.clip(alphas2[1:] / alphas2[:-1], a_min=clip_value, a_max=1.0)
    step = np.sqrt(step)
    return np.cumprod(step, axis=0)

def polynomial_schedule(timesteps: int, s: float = 1e-5, power: float = 2.0) -> np.ndarray:
    """The schedule EDM uses for molecules"""
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - (x / steps) ** power) ** 2
    alphas2 = _clip_noise_schedule(alphas2, clip_value=0.001)
    return ((1 - 2 * s) * alphas2 + s)**2

# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #
def remove_mean(x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Project coordinates onto the zero-centre-of-mass subspace."""
    nm = node_mask.unsqueeze(-1)
    n = node_mask.sum(1).view(-1, 1, 1).clamp(min=1.0)
    mean = (x * nm).sum(1, keepdim=True) / n
    return (x - mean) * nm

def sample_zero_com_noise(shape, node_mask, device) -> torch.Tensor:
    return remove_mean(torch.randn(shape, device=device), node_mask)

# --------------------------------------------------------------------------- #
# Diffusion Model
# --------------------------------------------------------------------------- #
class EquivariantDiffusion(nn.Module):
    def __init__(
        self,
        dynamics: nn.Module,
        num_types: int,
        timesteps: int = 1000,
        h_scale: float = 0.25,
    ):
        super().__init__()
        self.dynamics = dynamics
        self.num_types = num_types
        self.timesteps = timesteps
        self.h_scale = h_scale

        alphas2 = (
            polynomial_schedule(timesteps)
        )
        self.register_buffer("alphas", torch.tensor(np.sqrt(alphas2), dtype=torch.float32))
        self.register_buffer(
            "sigmas", torch.tensor(np.sqrt(1.0 - alphas2), dtype=torch.float32)
        )

    # ----------------------------- Training ------------------------------- #
    def normalize(self, x, h):
        return x, h * self.h_scale

    def unnormalize(self, x, h):
        return x, h / self.h_scale

    def loss(self, x, h, node_mask):
        """Loss computation for a batch of molecules.  x: [B,N,3], h: [B,N,K], node_mask: [B,N]."""

        # Prepare the batch and normalize the coordinates and atom types
        b = x.shape[0]
        x, h = self.normalize(x, h) #(B, N, 3) and (B, N, K)
        x = remove_mean(x, node_mask) #(B, N, 3)

        # Extract alpha and sigma from random timestep t_int for each molecule in the batch
        t_int = torch.randint(0, self.timesteps + 1, (b,), device=x.device) #(B,)
        alpha = self.alphas[t_int].view(b, 1, 1) #(B, 1, 1)
        sigma = self.sigmas[t_int].view(b, 1, 1) #(B, 1, 1)

        # Sample the noise
        eps_x = sample_zero_com_noise(x.shape, node_mask, x.device) #(B, N, 3)
        eps_h = torch.randn_like(h) * node_mask.unsqueeze(-1) #(B, N, K)

        # Compute the noisy input z_t = alpha * [x, h] + sigma * eps
        z_x = alpha * x + sigma * eps_x  #(B, N, 3)
        z_h = alpha * h + sigma * eps_h #(B, N, K)

        # Predict the noise with the model and compute the loss
        pred_x, pred_h = self.dynamics(t_int.float() / self.timesteps, z_x, z_h, node_mask)  #(B, N, 3), (B, N, K)

        # Compute the mean squared error, normalized by the degrees of freedom (dof)
        err = ((pred_x - eps_x) ** 2).sum() + ((pred_h - eps_h) ** 2).sum() #(1, )
        n_atoms = node_mask.sum()   #(1, )
        dof = 3 * (n_atoms - b) + self.num_types * n_atoms

        return err / dof

    # ----------------------------- Sampling ------------------------------- #
    def _p_step(self, z_x, z_h, t_int, node_mask):
        """One sampling step"""

        # Extract alpha and sigma for the current timestep t_int and the previous timestep s_int
        b = z_x.shape[0]
        s_int = t_int - 1
        a_t, s_t = self.alphas[t_int], self.sigmas[t_int] #(1,)
        a_s, s_s = self.alphas[s_int], self.sigmas[s_int] #(1,)

        # Build the coefficients for the reverse process
        a_ts = a_t / a_s #(B, 1, 1)
        var_ts = (s_t**2 - a_ts**2 * s_s**2).clamp(min=1e-12) #(1,)
        std = (var_ts.sqrt() * s_s / s_t).view(1, 1, 1) #(1,)

        # Build the times
        t = torch.full((b,), t_int / self.timesteps, device=z_x.device), #(B,)

        # Predict the noise with the model
        pred_x, pred_h = self.dynamics(t, z_x, z_h, node_mask)  #(B, N, 3), (B, N, K)

        # Build the mean of the reverse process
        coef = (var_ts / (a_ts * s_t)).view(1, 1, 1)  #(B, N, 3), (B, N, K)
        mu_x = z_x / a_ts - coef * pred_x  #(B, N, 3)
        mu_h = z_h / a_ts - coef * pred_h  #(B, N, K)

        # Sample the next step in the reverse process
        z_x = mu_x + std * sample_zero_com_noise(z_x.shape, node_mask, z_x.device)  #(B, N, 3)
        z_h = mu_h + std * torch.randn_like(z_h) * node_mask.unsqueeze(-1)  #(B, N, K)

        return remove_mean(z_x, node_mask), z_h * node_mask.unsqueeze(-1)

    @torch.no_grad()
    def sample(self, node_mask, keep_frames: int = 0):
        """Generate molecules for the given padding mask"""

        # Extact dimensions
        device = node_mask.device
        b, n = node_mask.shape
        nm = node_mask.unsqueeze(-1)

        # Sample noise
        z_x = sample_zero_com_noise((b, n, 3), node_mask, device)  #(B, N, 3)
        z_h = torch.randn(b, n, self.num_types, device=device) * nm  #(B, N, K)

        # Run the reverse process for timesteps -> 0
        frames = []
        for t_int in range(self.timesteps, 0, -1):
            z_x, z_h = self._p_step(z_x, z_h, t_int, node_mask)
            if keep_frames and (t_int - 1) % max(1, self.timesteps // keep_frames) == 0:
                frames.append((z_x.cpu(), z_h.cpu()))

        # Final denoise at t=0
        t = torch.zeros(b, device=device)  #(B,)
        pred_x, pred_h = self.dynamics(t, z_x, z_h, node_mask)  #(B, N, 3), (B, N, K)
        a0, s0 = self.alphas[0], self.sigmas[0]  #(B, N, 3),
        x = remove_mean((z_x - s0 * pred_x) / a0, node_mask)
        h = (z_h - s0 * pred_h) / a0    #(B, N, K)

        # Unnormalize the coordinates and extract atom types as one-hot vectors
        x, h = self.unnormalize(x, h)  #(B, N, 3), (B, N, K)
        types = h.argmax(-1) #(B, N)
        one_hot = torch.nn.functional.one_hot(types, self.num_types).float() * nm #(B, N, K)

        return x * nm, one_hot, frames
