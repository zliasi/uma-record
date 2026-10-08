#!/usr/bin/env python
"""
Single point UMA calculations on a Cu(111) slab, the slab with H, and H2.

    python run.py
    python run.py --device cuda

The settings of each system are the constants below, the command line takes
the device and a checkpoint override. Each record goes to 'output/cpu/' or
'output/gpu/' and is read back through ASE.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ase import Atoms
from ase.build import add_adsorbate, fcc111, molecule
from ase.constraints import FixAtoms
from ase.io import read
from fairchem.core import FAIRChemCalculator

from uma_record import write_record

# a fairchem model name, downloaded from Hugging Face on first use, or a
# checkpoint file. '--checkpoint' overrides it
CHECKPOINT = "uma-s-1p2p1"
# one UMA task for all three systems, H2 is the adsorbate reference of the slabs
TASK_NAME = "oc20"
SLAB_SIZE = (2, 2, 3)
SLAB_VACUUM_ANG = 10.0
MOLECULE_VACUUM_ANG = 5.0

HERE = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--checkpoint", default=CHECKPOINT, help="model name or checkpoint file"
    )
    return parser.parse_args()


def slab() -> Atoms:
    """Return a Cu(111) slab with the two bottom layers fixed."""
    atoms = fcc111("Cu", size=SLAB_SIZE, vacuum=SLAB_VACUUM_ANG)
    # periodic along the surface normal too, as the OC20 slabs are
    atoms.pbc = True
    # fcc111 tags the layers from the top
    atoms.set_constraint(FixAtoms(mask=atoms.get_tags() >= 2))
    return atoms


def slab_h() -> Atoms:
    """Return the Cu(111) slab with H in an fcc hollow site."""
    atoms = slab()
    add_adsorbate(atoms, "H", height=1.0, position="fcc")
    return atoms


def h2() -> Atoms:
    """Return H2 in a box."""
    return molecule("H2", vacuum=MOLECULE_VACUUM_ANG)


SYSTEMS = {"slab": slab, "slab-h": slab_h, "h2": h2}


def main() -> None:
    args = parse_args()
    outdir = HERE / "output" / ("gpu" if args.device == "cuda" else "cpu")
    outdir.mkdir(parents=True, exist_ok=True)
    # a model name or a checkpoint file, anything else raises. One calculator
    # serves all three systems, the model is loaded once. The seed warning in
    # the log comes from fairchem itself, from_model_checkpoint passes the
    # deprecated argument on
    calc = FAIRChemCalculator.from_model_checkpoint(
        args.checkpoint, task_name=TASK_NAME, device=args.device
    )

    for name, build in SYSTEMS.items():
        atoms = build()
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        record = write_record(atoms, filename=outdir / f"{name}.uma.txt")

        recorded = read(record)
        print(f"{record}: {recorded.get_chemical_formula()}, {energy:.6f} eV")


if __name__ == "__main__":
    main()
