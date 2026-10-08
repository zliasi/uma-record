"""
Scenario tests over a real UMA predict unit.

They run when 'UMA_RECORD_TEST_CHECKPOINT' names a checkpoint and are skipped
otherwise. The checkpoint loads once per session on the CPU. Each scenario
takes a path the stand-in of 'test_uma_record' cannot reach, the batcher,
the parallel unit, the fast path fallback, the omol defaults, and a wrapper
from fairchem, and checks that nothing beyond the expected rows is
unavailable. The batcher and the parallel unit start ray on the node, the
parallel unit loads the checkpoint once more per worker and must run last,
its test shuts ray down. The first run writes the md5 and sha256 listings
beside the checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from ase.build import add_adsorbate, molecule
from ase.mep import NEB
from ase.optimize import BFGS
from fairchem.core import FAIRChemCalculator
from fairchem.core.calculate import FormationEnergyCalculator, InferenceBatcher
from fairchem.core.common.distutils import cleanup_gp_ray
from fairchem.core.units.mlip_unit import load_predict_unit

from tests.support import (
    REAL_CHECKPOINT_VARIABLE,
    SETTINGS,
    UNAVAILABLE,
    make_slab,
    unavailable_rows,
)
from uma_record import read_fields, read_record, write_record

pytestmark = pytest.mark.skipif(
    not os.environ.get(REAL_CHECKPOINT_VARIABLE),
    reason=f"set {REAL_CHECKPOINT_VARIABLE} to a UMA checkpoint to run",
)

# rows a batch server cannot serve, the model lives in its process
SERVED_WITHOUT = {"model_id", "direct_forces"}


@pytest.fixture(scope="session")
def checkpoint() -> Path:
    return Path(os.environ[REAL_CHECKPOINT_VARIABLE]).absolute()


@pytest.fixture(scope="session")
def predict_unit(checkpoint: Path) -> Any:
    return load_predict_unit(checkpoint, device="cpu")


def test_batcher_serves_the_private_attributes(
    tmp_path: Path, checkpoint: Path, predict_unit: Any
) -> None:
    with InferenceBatcher(predict_unit, num_replicas=1) as batcher:
        calc = FAIRChemCalculator(batcher.batch_predict_unit, task_name="oc20")
        atoms = make_slab(calc)
        energy = atoms.get_potential_energy()
        record = write_record(atoms, filename=tmp_path / "slab.uma.txt")

    fields = read_fields(record)
    calculator = fields["Calculator"]
    assert calculator["predict_unit"] == "BatchServerPredictUnit"
    assert calculator["checkpoint"] == str(checkpoint)
    assert calculator["checkpoint md5"] != UNAVAILABLE
    assert calculator["seed"] == "41"
    assert "merge_mole" in fields[SETTINGS]
    assert fields["References"]["model"] == "wood2025uma"
    assert unavailable_rows(fields) == SERVED_WITHOUT
    assert read_record(record).get_potential_energy() == pytest.approx(energy, abs=1e-6)


def test_omol_defaults_and_given_charge_and_spin(
    tmp_path: Path, predict_unit: Any
) -> None:
    calc = FAIRChemCalculator(predict_unit, task_name="omol")
    water = molecule("H2O")
    water.calc = calc
    cation = molecule("H2O")
    cation.info.update(charge=1, spin=2)
    cation.calc = calc

    defaults = read_fields(write_record(water, filename=tmp_path / "water.uma.txt"))
    given = read_fields(write_record(cation, filename=tmp_path / "cation.uma.txt"))

    assert (defaults["Atoms"]["charge"], defaults["Atoms"]["spin"]) == ("0", "1")
    assert (given["Atoms"]["charge"], given["Atoms"]["spin"]) == ("1", "2")
    assert defaults["References"]["training data"] == "levine2025omol25"
    assert float(given["Results"]["energy / eV"]) != pytest.approx(
        float(defaults["Results"]["energy / eV"])
    )
    assert not unavailable_rows(given)


def test_composition_change_leaves_the_fast_path(
    tmp_path: Path, predict_unit: Any
) -> None:
    # the default settings merge the model for one composition and fall back
    # when another one arrives, the record shows the settings after that
    calc = FAIRChemCalculator(predict_unit, task_name="oc20")
    make_slab(calc).get_potential_energy()
    slab_h = make_slab(calc)
    add_adsorbate(slab_h, "H", height=1.0, position="fcc")

    fields = read_fields(write_record(slab_h, filename=tmp_path / "slab-h.uma.txt"))

    settings = fields[SETTINGS]
    assert fields["Atoms"]["formula"] == "HCu8"
    assert settings["merge_mole"] == "False"
    assert settings["compile"] == "False"
    assert not unavailable_rows(fields)


def test_formation_energy_wrapper_is_recorded(
    tmp_path: Path, checkpoint: Path, predict_unit: Any
) -> None:
    calc = FAIRChemCalculator(predict_unit, task_name="oc20")
    wrapper = FormationEnergyCalculator(
        calc, element_references={"Cu": 0.0}, apply_corrections=False
    )
    atoms = make_slab(wrapper)
    energy = atoms.get_potential_energy()

    fields = read_fields(write_record(atoms, filename=tmp_path / "slab.uma.txt"))

    assert fields["Calculator"]["calculator"] == "FAIRChemCalculator"
    assert fields["Calculator"]["wrapped by"] == "FormationEnergyCalculator"
    assert fields["Calculator"]["checkpoint"] == str(checkpoint)
    assert float(fields["Results"]["energy / eV"]) == pytest.approx(energy, abs=1e-6)


def test_band_with_a_shared_calculator_records_each_image(
    tmp_path: Path, predict_unit: Any
) -> None:
    calc = FAIRChemCalculator(predict_unit, task_name="oc20")
    initial = make_slab(calc)
    add_adsorbate(initial, "H", height=1.0, position="fcc")
    final = initial.copy()
    final.positions[-1, 0] += 1.0
    images = [initial, initial.copy(), initial.copy(), final]
    for image in images:
        image.calc = calc
    band = NEB(images, method="improvedtangent", allow_shared_calculator=True)
    band.interpolate()
    # ASE annotates the optimizer argument as Atoms, a band works the same
    opt = BFGS(band, logfile=None)  # type: ignore[arg-type]
    opt.run(fmax=5.0, steps=1)

    records = [
        write_record(image, opt, filename=tmp_path / f"image-{number}.uma.txt")
        for number, image in enumerate(images[1:-1], start=1)
    ]

    for image, record in zip(images[1:-1], records, strict=True):
        fields = read_fields(record)
        assert fields["Dynamics"]["optimizer"] == "BFGS"
        assert fields["Atoms"]["formula"] == "HCu8"
        assert read_record(record).positions == pytest.approx(image.positions)


def test_parallel_unit_needs_the_checkpoint_argument(
    tmp_path: Path, checkpoint: Path
) -> None:
    # the parallel unit runs its rank 0 in this process and leaves the torch
    # distributed group and the graph parallel state behind. A plain predict
    # unit made afterwards in the same process then hangs on a collective, so
    # the state is cleared before the next test
    parallel = load_predict_unit(checkpoint, device="cpu", workers=2)
    calc = FAIRChemCalculator(parallel, task_name="oc20")
    try:
        without = read_fields(
            write_record(make_slab(calc), filename=tmp_path / "without.uma.txt")
        )
        given = read_fields(
            write_record(
                make_slab(calc),
                filename=tmp_path / "given.uma.txt",
                checkpoint=checkpoint,
            )
        )
    finally:
        cleanup_gp_ray()

    assert without["Calculator"]["predict_unit"] == "ParallelMLIPPredictUnit"
    assert without["Calculator"]["checkpoint"] == UNAVAILABLE
    assert given["Calculator"]["checkpoint"] == str(checkpoint)
    assert given["Calculator"]["checkpoint md5"] != UNAVAILABLE
    assert "merge_mole" in given[SETTINGS]
    # the parallel unit keeps no model and no seed of its own
    assert unavailable_rows(given) <= SERVED_WITHOUT | {"seed", "overrides", "device"}
