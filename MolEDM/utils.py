"""Small shared utilities: EMA, device selection, XYZ output."""

from __future__ import annotations

import copy
import os
from typing import List, Sequence

import numpy as np
import torch


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EMA:
    """Exponential moving average of weights -- diffusion samples are much better
    from the averaged model than from the raw one."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for s, p in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
            else:
                s.copy_(p)


def write_xyz(path: str, positions: np.ndarray, symbols: Sequence[str], comment: str = "") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{len(symbols)}\n{comment}\n")
        for s, p in zip(symbols, positions):
            f.write(f"{s} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")


def to_molecules(x: torch.Tensor, one_hot: torch.Tensor, node_mask: torch.Tensor,
                 atom_types: Sequence[str]) -> List:
    """Dense batch -> list of (positions [n,3], symbols)."""
    out = []
    types = one_hot.argmax(-1).cpu().numpy()
    x = x.cpu().numpy()
    mask = node_mask.cpu().numpy().astype(bool)
    for i in range(len(x)):
        m = mask[i]
        out.append((x[i][m], [atom_types[t] for t in types[i][m]]))
    return out
