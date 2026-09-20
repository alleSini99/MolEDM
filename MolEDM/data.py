"""QM9 dataset: download, parse, cache and batch.

QM9 (Ramakrishnan et al. 2014) is 133,885 small organic molecules built from
{H, C, N, O, F}, each with a DFT-relaxed 3D geometry.  We only need atom types
and coordinates, so we parse the MoleculeNet SDF release, drop everything else,
and cache a single padded ``.npz``.  No PyTorch Geometric / RDKit needed.

Sources, in the order we try them:

1. ``gdb9.sdf`` inside MoleculeNet's ``qm9.zip`` (DeepChem S3 mirror). Currently
   the only reliably reachable copy.
2. ``dsgdb9nsd.xyz.tar.bz2``, the original figshare release.  Kept as a fallback
   because figshare's ``ndownloader`` endpoint has been answering 202-forever
   rather than serving the file.

The 3,054 molecules flagged "uncharacterized" (relaxed geometry inconsistent
with the SMILES) are excluded when ``uncharacterized.txt`` can be fetched; both
known URLs are currently 403/202, so by default they stay in.  That is 2.3% of
slightly noisier data -- fine for training, worth knowing when comparing
numbers to published results.  Drop the file at
``<root>/raw/uncharacterized.txt`` yourself if you have it.
"""

from __future__ import annotations

import bz2
import os
import tarfile
import urllib.error
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

SDF_ZIP_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
XYZ_TAR_URL = "https://springernature.figshare.com/ndownloader/files/3195389"
EXCLUDE_URLS = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/uncharacterized.txt",
    "https://springernature.figshare.com/ndownloader/files/3195404",
)

# QM9 contains only these five elements.
ATOM_TYPES: Tuple[str, ...] = ("H", "C", "N", "O", "F")
ATOM_ENCODER: Dict[str, int] = {a: i for i, a in enumerate(ATOM_TYPES)}
ATOMIC_NUMBERS = np.array([1, 6, 7, 8, 9])
MAX_N_ATOMS = 29  # largest molecule in QM9 (with hydrogens)


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _download(url: str, dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"downloading {url}\n        -> {dest}")
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "MolEDM/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status} for {url}")
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="")
        print()
    if os.path.getsize(tmp) == 0:
        os.remove(tmp)
        raise RuntimeError(f"empty response from {url}")
    os.rename(tmp, dest)


def _uncharacterized_ids(root: str) -> set:
    """1-based indices of molecules whose relaxed geometry failed QM9's checks."""
    path = os.path.join(root, "raw", "uncharacterized.txt")
    if not os.path.exists(path):
        for url in EXCLUDE_URLS:
            try:
                _download(url, path)
                break
            except Exception as e:  # noqa: BLE001 - any failure just means "no list"
                print(f"  (could not fetch exclusion list: {type(e).__name__}: {e})")
    if not os.path.exists(path):
        print("proceeding without the uncharacterized-molecule filter")
        return set()

    ids = set()
    with open(path) as f:
        for line in f:
            parts = line.split()
            # the list is a header + rows whose first field is the molecule index
            if len(parts) >= 1 and parts[0].isdigit():
                ids.add(int(parts[0]))
    print(f"skipping {len(ids)} uncharacterized molecules")
    return ids


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _parse_sdf(path: str, skip: set) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Stream an SDF (V2000) and pull out atom types + coordinates."""
    types_out: List[np.ndarray] = []
    pos_out: List[np.ndarray] = []
    with open(path, "r", errors="replace") as f:
        while True:
            name = f.readline()
            if not name:
                break
            name = name.strip()
            f.readline()  # program line
            f.readline()  # comment line
            counts = f.readline()
            if not counts:
                break
            n_atoms = int(counts[0:3])
            n_bonds = int(counts[3:6])

            types = np.empty(n_atoms, dtype=np.int8)
            pos = np.empty((n_atoms, 3), dtype=np.float32)
            ok = True
            for i in range(n_atoms):
                line = f.readline()
                sym = line[31:34].strip()
                if sym not in ATOM_ENCODER:  # QM9 should not contain anything else
                    ok = False
                    break
                types[i] = ATOM_ENCODER[sym]
                pos[i] = (float(line[0:10]), float(line[10:20]), float(line[20:30]))

            for _ in range(n_bonds):  # bond block: unused, we generate geometry
                f.readline()
            while True:  # skip to the end of the record
                line = f.readline()
                if not line or line.startswith("$$$$"):
                    break

            mol_id = int(name.split("_")[-1]) if "_" in name else len(types_out) + 1
            if ok and mol_id not in skip:
                types_out.append(types)
                pos_out.append(pos)
            if len(types_out) % 20000 == 0 and types_out:
                print(f"\r  parsed {len(types_out)}", end="")
    return types_out, pos_out


def _to_float_fortran(token: str) -> float:
    # the XYZ release uses Fortran-style exponents, e.g. "1.2*^-6"
    return float(token.replace("*^", "e"))


def _parse_xyz_tar(path: str, skip: set) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    types_out: List[np.ndarray] = []
    pos_out: List[np.ndarray] = []
    with bz2.open(path) as fobj, tarfile.open(fileobj=fobj, mode="r|") as tar:
        for member in tar:
            if not member.name.endswith(".xyz"):
                continue
            mol_id = int(member.name.split("_")[-1].split(".")[0])
            if mol_id in skip:
                continue
            lines = tar.extractfile(member).read().decode("utf-8").splitlines()
            n = int(lines[0])
            types = np.empty(n, dtype=np.int8)
            pos = np.empty((n, 3), dtype=np.float32)
            for i, line in enumerate(lines[2 : 2 + n]):
                parts = line.split()
                types[i] = ATOM_ENCODER[parts[0]]
                pos[i] = [_to_float_fortran(p) for p in parts[1:4]]
            types_out.append(types)
            pos_out.append(pos)
            if len(types_out) % 20000 == 0:
                print(f"\r  parsed {len(types_out)}", end="")
    return types_out, pos_out


def process_qm9(root: str) -> str:
    """Build (or reuse) the cached ``.npz``. Returns its path."""
    cache = os.path.join(root, "processed", "qm9.npz")
    if os.path.exists(cache):
        return cache

    skip = _uncharacterized_ids(root)
    sdf_path = os.path.join(root, "raw", "gdb9.sdf")
    all_types: Optional[List[np.ndarray]] = None

    if not os.path.exists(sdf_path):
        zip_path = os.path.join(root, "raw", "qm9.zip")
        try:
            _download(SDF_ZIP_URL, zip_path)
            with zipfile.ZipFile(zip_path) as z:
                member = next(n for n in z.namelist() if n.endswith("gdb9.sdf"))
                print(f"extracting {member}")
                with z.open(member) as src, open(sdf_path, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
        except Exception as e:  # noqa: BLE001
            print(f"SDF source failed ({type(e).__name__}: {e}); trying the XYZ release")

    if os.path.exists(sdf_path):
        all_types, all_pos = _parse_sdf(sdf_path, skip)
    else:
        tar_path = os.path.join(root, "raw", "dsgdb9nsd.xyz.tar.bz2")
        _download(XYZ_TAR_URL, tar_path)
        all_types, all_pos = _parse_xyz_tar(tar_path, skip)

    print(f"\r  parsed {len(all_types)} molecules")
    if len(all_types) < 1000:
        raise RuntimeError(f"only parsed {len(all_types)} molecules -- source looks wrong")

    m = len(all_types)
    max_n = max(len(t) for t in all_types)
    num_atoms = np.array([len(t) for t in all_types], dtype=np.int16)
    types_pad = np.zeros((m, max_n), dtype=np.int8)
    pos_pad = np.zeros((m, max_n, 3), dtype=np.float32)
    for i, (t, p) in enumerate(zip(all_types, all_pos)):
        types_pad[i, : len(t)] = t
        pos_pad[i, : len(t)] = p

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(
        cache, atom_types=types_pad, positions=pos_pad, num_atoms=num_atoms
    )
    print(f"cached {m} molecules (max {max_n} atoms) -> {cache}")
    return cache


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class QM9Dataset(Dataset):
    """Padded dense QM9. Each item: coords [N,3], one-hot types [N,K], mask [N]."""

    def __init__(
        self,
        root: str = "data/qm9",
        split: str = "train",
        remove_h: bool = False,
        limit: Optional[int] = None,
    ):
        d = np.load(process_qm9(root))
        types, pos, n_atoms = d["atom_types"], d["positions"], d["num_atoms"]

        if remove_h:
            new_t = np.zeros_like(types)
            new_p = np.zeros_like(pos)
            new_n = np.zeros_like(n_atoms)
            for i in range(len(types)):
                n = n_atoms[i]
                keep = types[i, :n] != ATOM_ENCODER["H"]
                k = int(keep.sum())
                new_t[i, :k] = types[i, :n][keep] - 1  # shift down, H slot is gone
                new_p[i, :k] = pos[i, :n][keep]
                new_n[i] = k
            types, pos, n_atoms = new_t, new_p, new_n

        # Canonical split: fixed permutation, 100k train / 10% test / rest val.
        perm = np.random.RandomState(0).permutation(len(types))
        n_test = int(0.1 * len(types))
        n_train = min(100_000, len(types) - n_test - 1)
        idx = {
            "train": perm[:n_train],
            "val": perm[n_train : len(types) - n_test],
            "test": perm[len(types) - n_test :],
        }[split]
        if limit is not None:
            idx = idx[:limit]

        self.remove_h = remove_h
        self.num_types = len(ATOM_TYPES) - int(remove_h)
        self.max_n = int(n_atoms[idx].max())
        self.atom_types = torch.from_numpy(types[idx][:, : self.max_n].astype(np.int64))
        self.positions = torch.from_numpy(pos[idx][:, : self.max_n])
        self.num_atoms = torch.from_numpy(n_atoms[idx].astype(np.int64))

    def __len__(self) -> int:
        return len(self.num_atoms)

    def __getitem__(self, i: int):
        n = int(self.num_atoms[i])
        mask = torch.zeros(self.max_n)
        mask[:n] = 1.0
        one_hot = torch.nn.functional.one_hot(self.atom_types[i], self.num_types).float()
        return {
            "x": self.positions[i] * mask[:, None],
            "h": one_hot * mask[:, None],
            "mask": mask,
        }

    def size_histogram(self) -> torch.Tensor:
        """p(number of atoms), used to sample molecule sizes at generation time."""
        hist = torch.bincount(self.num_atoms, minlength=self.max_n + 1).float()
        return hist / hist.sum()
