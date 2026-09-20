"""Sample molecules from a trained checkpoint.

    python sample.py --ckpt runs/qm9/best.pt --n 100 --out samples/
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from MolEDM.data import ATOM_TYPES
from MolEDM.diffusion import EquivariantDiffusion
from MolEDM.egnn import EGNNDynamics
from MolEDM.stability import evaluate, molecule_stability
from MolEDM.utils import get_device, to_molecules, write_xyz


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/qm9/best.pt")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--out", default="samples")
    p.add_argument("--n-atoms", type=int, default=None,
                   help="fix the molecule size instead of sampling p(N)")
    p.add_argument("--raw", action="store_true", help="use raw weights, not the EMA copy")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    symbols = ATOM_TYPES[1:] if cfg["remove_h"] else ATOM_TYPES

    dyn = EGNNDynamics(cfg["num_types"], hidden=cfg["hidden"], n_layers=cfg["layers"])
    model = EquivariantDiffusion(dyn, cfg["num_types"], cfg["timesteps"])
    model.load_state_dict(ckpt["model" if args.raw else "ema"])
    model.to(device).eval()

    size_hist = ckpt["size_hist"].to(device)
    max_n = cfg["max_n"]
    molecules = []
    while len(molecules) < args.n:
        b = min(args.batch_size, args.n - len(molecules))
        if args.n_atoms:
            sizes = torch.full((b,), args.n_atoms, device=device)
        else:
            sizes = torch.multinomial(size_hist, b, replacement=True)
        mask = (torch.arange(max_n, device=device)[None, :] < sizes[:, None]).float()
        x, one_hot, _ = model.sample(mask)
        molecules += to_molecules(x, one_hot, mask, symbols)
        print(f"  sampled {len(molecules)}/{args.n}", flush=True)

    metrics = evaluate(molecules, implicit_h=cfg["remove_h"])
    print(json.dumps(metrics, indent=2))

    os.makedirs(args.out, exist_ok=True)
    for i, (pos, syms) in enumerate(molecules):
        stable, n_ok, n = molecule_stability(pos, syms, cfg["remove_h"])
        write_xyz(os.path.join(args.out, f"mol_{i:04d}.xyz"), pos, syms,
                  comment=f"stable={stable} stable_atoms={n_ok}/{n}")
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote {len(molecules)} .xyz files to {args.out}/")

if __name__ == "__main__":
    main()
