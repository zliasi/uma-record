"""Helpers shared by the test modules."""

from __future__ import annotations

from ase import Atoms
from ase.build import fcc111
from ase.constraints import FixAtoms

REAL_CHECKPOINT_VARIABLE = "UMA_RECORD_TEST_CHECKPOINT"
FIXED_ATOMS = [0, 1, 2, 3]
VACUUM_ANG = 10.0
UNAVAILABLE = "unavailable"
SETTINGS = "Inference settings, effective when written"


def make_slab(calc: object) -> Atoms:
    """Return a two layer Cu slab with its bottom layer fixed."""
    # ase.build is unannotated
    atoms: Atoms = fcc111("Cu", size=(2, 2, 2), vacuum=VACUUM_ANG)
    atoms.set_constraint(FixAtoms(indices=FIXED_ATOMS))
    atoms.calc = calc
    return atoms


def unavailable_rows(fields: dict[str, dict[str, str]]) -> set[str]:
    """Return the keys of all sections whose value could not be read."""
    return {
        key
        for rows in fields.values()
        for key, value in rows.items()
        if value == UNAVAILABLE
    }
