# Equivariant diffusion for 3D molecule generation on QM9

A small, self-contained (PyTorch-only) implementation of an **E(3)-equivariant
denoising diffusion model** that generates small molecules as 3D point clouds —
atom types *and* coordinates jointly. It's a compact reimplementation of
[EDM (Hoogeboom et al., 2022)](https://arxiv.org/abs/2203.17003) with an
[EGNN](https://arxiv.org/abs/2102.09844) denoiser.

No PyTorch Geometric, no RDKit, no DGL. `torch` + `numpy` is the whole stack.

## Why this shape of model

Generating a molecule in 3D means generating a point cloud where the *labels*
are discrete (atom types) and the *positions* are continuous, and where physics
doesn't care about the global pose. Three design consequences:

| Problem | What the code does |
| --- | --- |
| No distribution over `R^{3n}` can be translation invariant | Coordinates live in the **zero-centre-of-mass subspace**: the data, the noise, and the model's coordinate output all have their mean removed (`diffusion.remove_mean`) |
| The model should not have to learn rotational symmetry from data | The denoiser is **E(3)-equivariant** by construction: messages see only squared interatomic distances, coordinates are updated along relative-position vectors (`egnn.EquivariantConv`) |
| Atom types are categorical, diffusion is Gaussian | One-hot vectors are diffused as a **continuous relaxation**, scaled down by `h_scale=0.25` so they don't dominate the loss compared to the positions, and decoded with an `argmax` at `t=0` |

Training objective is the simple eps-prediction loss with `w(t)=1`:

```
z_t = alpha_t * [x, h] + sigma_t * eps      L = || eps - eps_hat(z_t, t) ||^2
```

with a `polynomial_2` noise schedule.

## Documentation

[`docs/code_walkthrough.tex`](docs/code_walkthrough.tex) is a 15-page walkthrough
of the whole implementation — the EGNN, the diffusion process, training and
sampling — annotating every tensor with its shape and tracing how each step
changes it. Build with:

```bash
cd docs && latexmk -pdf code_walkthrough.tex
```

## Layout

```
MolEDM/
  data.py        QM9 download, parsing, and dataset creation
  egnn.py        EGNN denoiser -> (eps_x, eps_h)
  diffusion.py   noise schedules, training loss, ancestral sampler
  stability.py   bond inference from distances, atom/molecule stability metrics
  utils.py       EMA, device pick, XYZ writer
train.py         training loop
sample.py        generate molecules -> .xyz + metrics
eval_reference.py  metrics on real QM9 (sanity check / upper bound)
test_smoke.py    equivariance, masking, overfit-4-molecules, sampling
docs/code_walkthrough.tex   annotated walkthrough with full shape traces
```

## Data

`data.py` downloads and caches QM9 on first use (~45 MB download → ~50 MB
`.npz`, ~1 min). It parses `gdb9.sdf` from MoleculeNet's DeepChem S3 mirror;
the original figshare release is kept as a fallback but that endpoint currently
answers `202`-forever instead of serving the file. Result: 133,885 molecules,
max 29 atoms — the expected QM9 counts.

One caveat: the list of 3,054 "uncharacterized" molecules (relaxed geometry
inconsistent with the SMILES) is normally excluded, but both known URLs for
`uncharacterized.txt` are dead (403 / 202), so they are currently **kept in** —
2.3% of slightly noisier data. If you have the file, drop it at
`data/qm9/raw/uncharacterized.txt` and delete `data/qm9/processed/` to re-cache;
the filter picks it up automatically.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Quick start

Verify the maths before spending GPU hours — checks rotation equivariance,
translation invariance, that padding never leaks, and that the model can
overfit four molecules:

```bash
.venv/bin/python test_smoke.py
```

Two-minute end-to-end run (downloads and caches QM9 on the first call):

```bash
.venv/bin/python train.py --limit-train 2000 --epochs 2 --timesteps 100 --hidden 64 --layers 3 --eval-every 1 --n-eval-samples 16
```

Full run with hydrogens (needs a CUDA GPU to be practical — see the timing
table below):

```bash
.venv/bin/python train.py --epochs 300 --batch-size 64 --hidden 192 --layers 6
```

Heavy-atoms-only run, which does train to something reasonable on a laptop:

```bash
.venv/bin/python train.py --remove-h --hidden 128 --layers 4 --epochs 300 --eval-every 10
```

Sample from a checkpoint (writes one `.xyz` per molecule, viewable in PyMOL /
VMD / Avogadro / ASE):

```bash
.venv/bin/python sample.py --ckpt runs/qm9/best.pt --n 100 --out samples
```

## Evaluation

The standard QM9 3D metrics, implemented in `stability.py`: bonds are inferred
from interatomic distances against a table of single/double/triple bond lengths,
then an atom is "stable" if its inferred valence matches its expected one
(H:1, C:4, N:3, O:2, F:1); a molecule is stable if all of its atoms are.

Measured here on 3,000 real QM9 molecules (`.venv/bin/python eval_reference.py --n 3000`),
next to the published figures:

| | atom stable | molecule stable |
| --- | --- | --- |
| Real QM9, with H (this code) | **0.994** | **0.951** |
| Real QM9, with H (EDM paper) | 0.99 | 0.95 |
| EDM paper, generated, with H | 0.98 | 0.82 |
| Real QM9, heavy atoms only (this code) | 0.996 | 0.978 |

The with-H row reproducing the paper's data numbers is the check that the metric
itself is implemented correctly.

**Heavy-atom-only metrics are weaker and not comparable.** With hydrogens
stripped, exact valence is unsatisfiable (a carbon from methane has no heavy
neighbours), so `--remove-h` mode only checks that the heavy-atom valence does
not *exceed* what the element allows, treating the remainder as implicit H.

Molecule stability is the metric that actually hurts — it needs *every* atom in
the molecule right, so it falls off a cliff for undertrained models. Watch
`ema_atom_stable` in `runs/qm9/log.jsonl` first; it moves long before
`ema_mol_stable` does.

## Notes on getting good samples

- **EMA weights are the default** (`--raw` to compare). The averaged weights are worth
  more as runs get longer, which is when diffusion sampling gets noticeably
  cleaner from them.
- **Budget matters.** The EDM paper trains ~1000–1700 epochs on 100k molecules
  with a 9-layer, 256-hidden EGNN.
- **`--remove-h` is the cheap mode.** Heavy atoms only means ~9 atoms instead of
  ~18, which is much faster and a good place to iterate; H-free stability
  numbers are not comparable to with-H ones.
- **Pick the config to fit your hardware.** Measured on an M-series Mac (MPS),
  batch size 64, one epoch = 100k molecules. The last column times the
  training step alone; add ~50% for data loading and the per-epoch val pass:

  | config | params | ms/iter | min/epoch |
  | --- | --- | --- | --- |
  | `--hidden 256 --layers 9` (EDM paper) | 4.2M | 613 | 16.0 |
  | `--hidden 192 --layers 6` (default) | 1.6M | 294 | 7.7 |
  | `--hidden 128 --layers 4` | 0.5M | 123 | 3.2 |
  | `--hidden 128 --layers 4 --remove-h` | 0.5M | 21 | **0.5** (0.75 end-to-end) |

  The paper's config for its ~1100 epochs would be ~12 days on MPS. Use a CUDA
  GPU for with-H runs; `--remove-h` is the only setting that trains to
  convergence locally in a couple of hours.
- **Memory scales as `B * N^2 * hidden`.** The dense `[B, N, N, hidden]` edge
  tensor is the bottleneck — drop `--hidden` or `--batch-size` if you hit MPS
  limits.

## Known simplifications vs. the EDM paper

- Trains the `L_simple` L2 objective only — no variational lower bound / NLL
  reporting, no `L_0` reconstruction term.
- No atom-charge feature (EDM diffuses a scalar nuclear-charge channel too).
- No conditional generation on QM9 properties (alpha, mu, Cv, ...).
- Dense fully-connected graphs rather than sparse edge lists — simpler, and
  equivalent at QM9 sizes.
- Validity/uniqueness via RDKit SMILES is not computed; only the geometric
  stability metrics.
