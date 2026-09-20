"""Geometry-only evaluation of generated molecules.

The standard QM9 3D-generation metrics (atom stability / molecule stability)
infer bonds from interatomic distances using tabulated bond lengths, then check
that every atom has its expected valence.  No RDKit required.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# Single / double / triple bond lengths measured in picometres
BONDS_1: Dict[str, Dict[str, int]] = {
    "H": {"H": 74, "C": 109, "N": 101, "O": 96, "F": 92},
    "C": {"H": 109, "C": 154, "N": 147, "O": 143, "F": 135},
    "N": {"H": 101, "C": 147, "N": 145, "O": 140, "F": 136},
    "O": {"H": 96, "C": 143, "N": 140, "O": 148, "F": 142},
    "F": {"H": 92, "C": 135, "N": 136, "O": 142, "F": 142},
}
BONDS_2: Dict[str, Dict[str, int]] = {
    "C": {"C": 134, "N": 129, "O": 120},
    "N": {"C": 129, "N": 125, "O": 121},
    "O": {"C": 120, "N": 121, "O": 121},
}
BONDS_3: Dict[str, Dict[str, int]] = {
    "C": {"C": 120, "N": 116, "O": 113},
    "N": {"C": 116, "N": 110},
    "O": {"C": 113},
}
MARGINS = (10, 5, 3)  # pm tolerance for single / double / triple
ALLOWED_VALENCE = {"H": 1, "C": 4, "N": 3, "O": 2, "F": 1}


def bond_order(a: str, b: str, dist_angstrom: float) -> int:
    """Infer the type of bonds from distance alone: 0 = no bond, 1/2/3 = single/double/triple"""
    d = 100.0 * dist_angstrom
    if b not in BONDS_1.get(a, {}) or d >= BONDS_1[a][b] + MARGINS[0]:
        return 0
    if b in BONDS_2.get(a, {}) and d < BONDS_2[a][b] + MARGINS[1]:
        if b in BONDS_3.get(a, {}) and d < BONDS_3[a][b] + MARGINS[2]:
            return 3
        return 2
    return 1


def molecule_stability(
    positions: np.ndarray, symbols: Sequence[str], implicit_h: bool = False
) -> Tuple[bool, int, int]:
    """Comput the stability of a molecule.

    With explicit hydrogens an atom is stable iff its inferred valence *equals*
    the expected one.  When hydrogens were stripped from the data
    (``implicit_h=True``) exact equality is unsatisfiable -- a carbon in methane
    has no heavy-atom neighbours -- so we only require that the heavy-atom
    valence does not *exceed* what the element allows, the remainder being
    implicit H.  That is a much weaker check; no-H numbers are not comparable to
    with-H ones.
    """
    n = len(symbols)
    valence = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(positions[i] - positions[j]))
            o = bond_order(symbols[i], symbols[j], d)
            valence[i] += o
            valence[j] += o
    if implicit_h:
        ok = np.array([valence[i] <= ALLOWED_VALENCE[symbols[i]] for i in range(n)])
    else:
        ok = np.array([valence[i] == ALLOWED_VALENCE[symbols[i]] for i in range(n)])
    return bool(ok.all()), int(ok.sum()), n


def evaluate(
    molecules: List[Tuple[np.ndarray, List[str]]], implicit_h: bool = False
) -> Dict[str, float]:
    """Compute the standard QM9 stability metrics on a list of molecules"""
    n_stable_mol = 0
    n_stable_atom = 0
    n_atom = 0
    for pos, syms in molecules:
        stable, s_atoms, n = molecule_stability(pos, syms, implicit_h)
        n_stable_mol += int(stable)
        n_stable_atom += s_atoms
        n_atom += n
    return {
        "mol_stable": n_stable_mol / max(1, len(molecules)),
        "atom_stable": n_stable_atom / max(1, n_atom),
        "n_molecules": len(molecules),
    }
