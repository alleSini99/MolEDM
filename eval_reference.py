"""Stability metrics on *real* QM9 molecules.

Sanity check for the bond-inference metric and an upper bound on what a
generative model can score.  Expect ~0.99 atom stability / ~0.95 molecule
stability with hydrogens.

    python eval_reference.py --n 2000
"""

from __future__ import annotations

import argparse
import json

from MolEDM.data import ATOM_TYPES, QM9Dataset
from MolEDM.stability import evaluate

p = argparse.ArgumentParser()
p.add_argument("--data-root", default="data/qm9")
p.add_argument("--split", default="test")
p.add_argument("--n", type=int, default=2000)
p.add_argument("--remove-h", action="store_true")
args = p.parse_args()

ds = QM9Dataset(args.data_root, args.split, args.remove_h, limit=args.n)
symbols = ATOM_TYPES[1:] if args.remove_h else ATOM_TYPES
mols = []
for i in range(len(ds)):
    n = int(ds.num_atoms[i])
    mols.append((ds.positions[i][:n].numpy(), [symbols[t] for t in ds.atom_types[i][:n]]))
print(json.dumps(evaluate(mols, implicit_h=args.remove_h), indent=2))
