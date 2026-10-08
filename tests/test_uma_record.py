"""
Contract tests for 'write_record', 'read_record', and 'read_fields'.

A real 'FAIRChemCalculator' runs over a stand-in predict unit, so the tests
need fairchem installed, but no checkpoint and no GPU. One test loads a real
predict unit when 'UMA_RECORD_TEST_CHECKPOINT' names a checkpoint, and is
skipped otherwise.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import platform
import stat
import sys
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from ase import Atoms, units
from ase.build import fcc111, molecule
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.calculators.mixing import SumCalculator
from ase.constraints import FixAtoms, FixBondLength
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    validate_uma_atoms_data,
)

import uma_record
from tests.support import (
    FIXED_ATOMS,
    REAL_CHECKPOINT_VARIABLE,
    SETTINGS,
    UNAVAILABLE,
    VACUUM_ANG,
    make_slab,
)
from uma_record import read_fields, read_record, write_record

CHECKPOINT_BYTES = b"stand-in checkpoint"
SPRING_EV_ANG2 = 0.5
TASK_PROPERTIES = ("energy", "forces", "stress")
STRESS_KEY = "stress xx yy zz yz xz xy / (eV/Ang**3)"
GAP_KEY = "max gap along a b c / Ang"
VACUUM_KEY = "vacuum below, above along c / Ang"


@dataclasses.dataclass(frozen=True)
class StandInTask:
    property: str


class BarePredictUnit:
    """
    Harmonic model with the interface 'FAIRChemCalculator' needs.

    Each atom is bound to the centroid by a spring. The private attributes of
    a real predict unit are left out.
    """

    def __init__(self) -> None:
        tasks = [StandInTask(name) for name in TASK_PROPERTIES]
        self.dataset_to_tasks = {"oc20": tasks, "omol": tasks}
        self.inference_settings = InferenceSettings(
            predict_untrained_stress={"omol", "oc20"}
        )

    def validate_atoms_data(self, atoms: Atoms, task_name: str) -> None:
        validate_uma_atoms_data(atoms, task_name)

    def predict(self, data: Any) -> dict[str, torch.Tensor]:
        displacement = data.pos - data.pos.mean(dim=0)
        return {
            "energy": 0.5 * SPRING_EV_ANG2 * (displacement**2).sum().reshape(1),
            "forces": -SPRING_EV_ANG2 * displacement,
            "stress": torch.zeros(1, 9),
        }


class StandInPredictUnit(BarePredictUnit):
    """Stand-in with the private attributes 'write_record' reads."""

    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        self._inference_model_path = str(checkpoint)
        self._seed = 41
        self._overrides = None
        self.device = "cpu"


class DelegatingCalculator(Calculator):
    """Wrapper that keeps its inner calculator in 'calculator', as fairchem does."""

    implemented_properties = ["energy", "forces"]

    # the ASE base class is unannotated, and the exclusion in pyproject.toml
    # does not reach calls made through super()
    def __init__(self, calculator: Calculator) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.calculator = calculator

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] = ("energy",),
        system_changes: Sequence[str] = tuple(all_changes),
    ) -> None:
        super().calculate(  # type: ignore[no-untyped-call]
            atoms, list(properties), list(system_changes)
        )
        assert atoms is not None
        self.results = {
            "energy": self.calculator.get_potential_energy(atoms),
            "forces": self.calculator.get_forces(atoms),
        }


class EnergyOnlyCalculator(DelegatingCalculator):
    """Wrapper whose results hold no forces."""

    implemented_properties = ["energy"]

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] = ("energy",),
        system_changes: Sequence[str] = tuple(all_changes),
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": self.results["energy"]}


@pytest.fixture(autouse=True)
def no_configured_filename(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UMA_RECORD_PATH", raising=False)


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "uma-s-test.pt"
    path.write_bytes(CHECKPOINT_BYTES)
    return path


@pytest.fixture
def calc(checkpoint: Path) -> FAIRChemCalculator:
    # the stand-in follows the protocol, not the nominal type
    predict_unit: Any = StandInPredictUnit(checkpoint)
    return FAIRChemCalculator(predict_unit, task_name="oc20")


def test_record_holds_calculator_and_results(
    tmp_path: Path, checkpoint: Path, calc: FAIRChemCalculator
) -> None:
    atoms = make_slab(calc)
    energy = atoms.get_potential_energy()

    record = write_record(atoms, filename=tmp_path / "slab.uma.txt")

    fields = read_fields(record)
    assert record == tmp_path / "slab.uma.txt"
    calculator = fields["Calculator"]
    assert calculator["calculator"] == "FAIRChemCalculator"
    assert calculator["predict_unit"] == "StandInPredictUnit"
    assert calculator["task_name"] == "oc20"
    assert calculator["implemented_properties"] == "energy, forces, stress, free_energy"
    assert calculator["checkpoint"] == str(checkpoint)
    assert calculator["checkpoint md5"] == hashlib.md5(CHECKPOINT_BYTES).hexdigest()
    assert (
        calculator["checkpoint sha256"] == hashlib.sha256(CHECKPOINT_BYTES).hexdigest()
    )
    assert calculator["checkpoint size / bytes"] == str(len(CHECKPOINT_BYTES))
    assert calculator["device"] == "cpu"
    assert calculator["seed"] == "41"
    assert calculator["overrides"] == "none"
    assert fields[SETTINGS]["tf32"] == "False"
    assert fields[SETTINGS]["torch_num_threads"] == "none"
    assert fields[SETTINGS]["predict_untrained_stress"] == "[oc20, omol]"
    assert fields["Results"]["source"] == "last calculation"
    assert float(fields["Results"]["energy / eV"]) == pytest.approx(energy, abs=1e-6)
    assert fields["Atoms"]["formula"] == "Cu8"
    assert fields["Atoms"]["natoms"] == "8"
    assert fields["References"] == {
        "software": "fairchem",
        "model": "wood2025uma",
        "training data": "chanussot2021oc20",
    }
    text = record.read_text()
    assert text.startswith("UMA calculation record (uma-record")
    assert "@software{fairchem," in text
    assert "@inproceedings{wood2025uma," in text
    assert "@article{chanussot2021oc20," in text
    fairchem_version = metadata.version("fairchem-core")
    assert fields["Dependencies"]["fairchem.core"].startswith(f"{fairchem_version}, ")


def test_structure_block_reads_back(tmp_path: Path, calc: FAIRChemCalculator) -> None:
    atoms = make_slab(calc)
    atoms.rattle(stdev=0.05, seed=1)
    energy = atoms.get_potential_energy()
    raw_forces = atoms.get_forces(apply_constraint=False)

    recorded = read_record(write_record(atoms, filename=tmp_path / "slab.uma.txt"))

    assert recorded.get_chemical_formula() == "Cu8"
    assert recorded.positions == pytest.approx(atoms.positions, abs=1e-6)
    assert np.asarray(recorded.cell) == pytest.approx(np.asarray(atoms.cell), abs=1e-6)
    assert list(recorded.pbc) == list(atoms.pbc)
    assert recorded.get_potential_energy() == pytest.approx(energy, abs=1e-6)
    assert recorded.get_forces(apply_constraint=False) == pytest.approx(
        raw_forces, abs=1e-6
    )
    assert np.abs(raw_forces[FIXED_ATOMS]).max() > 1e-3
    assert sorted(recorded.constraints[0].index) == FIXED_ATOMS


def test_fmax_has_constraints_applied(tmp_path: Path, calc: FAIRChemCalculator) -> None:
    atoms = make_slab(calc)
    atoms.rattle(stdev=0.05, seed=1)
    fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())

    fields = read_fields(write_record(atoms, filename=tmp_path / "slab.uma.txt"))

    assert float(fields["Results"]["fmax / (eV/Ang)"]) == pytest.approx(fmax, abs=1e-6)
    assert fields["Atoms"]["constraints"] == "FixAtoms on 4 atoms (indices 0-3)"


def test_fmax_is_omitted_without_forces_and_nothing_is_recalculated(
    tmp_path: Path, calc: FAIRChemCalculator
) -> None:
    atoms = make_slab(EnergyOnlyCalculator(calc))
    energy = atoms.get_potential_energy()

    fields = read_fields(write_record(atoms, filename=tmp_path / "slab.uma.txt"))

    assert fields["Results"]["source"] == "last calculation"
    assert float(fields["Results"]["energy / eV"]) == pytest.approx(energy, abs=1e-6)
    assert "fmax / (eV/Ang)" not in fields["Results"]


def test_changed_structure_is_calculated_again(
    tmp_path: Path, calc: FAIRChemCalculator
) -> None:
    atoms = make_slab(calc)
    atoms.get_potential_energy()
    atoms.positions[-1, 2] += 0.2

    fields = read_fields(write_record(atoms, filename=tmp_path / "slab.uma.txt"))

    assert fields["Results"]["source"] == "calculated for this record"
    assert float(fields["Results"]["energy / eV"]) == pytest.approx(
        atoms.get_potential_energy(), abs=1e-6
    )


def test_largest_gap_is_the_slab_vacuum(
    tmp_path: Path, calc: FAIRChemCalculator
) -> None:
    atoms = make_slab(calc)

    fields = read_fields(write_record(atoms, filename=tmp_path / "slab.uma.txt"))

    gaps = [float(cell) for cell in fields["Atoms"][GAP_KEY].split()]
    assert gaps[2] == pytest.approx(2 * VACUUM_ANG, abs=1e-4)
    vacuum = fields["Atoms"][VACUUM_KEY].split()
    assert [float(v) for v in vacuum] == pytest.approx([VACUUM_ANG] * 2, abs=1e-4)


def test_vacuum_across_the_cell_boundary_has_no_split(
    tmp_path: Path, calc: FAIRChemCalculator
) -> None:
    atoms = make_slab(calc)
    atoms.pbc = True
    atoms.positions[:, 2] += atoms.cell[2, 2] / 2
    atoms.wrap()

    fields = read_fields(write_record(atoms, filename=tmp_path / "slab.uma.txt"))

    gaps = [float(cell) for cell in fields["Atoms"][GAP_KEY].split()]
    assert gaps[2] == pytest.approx(2 * VACUUM_ANG, abs=1e-4)
    assert VACUUM_KEY not in fields["Atoms"]


def test_molecule_in_a_box_has_no_open_axis(
    tmp_path: Path, calc: FAIRChemCalculator
) -> None:
    atoms = molecule("H2", vacuum=VACUUM_ANG)
    atoms.calc = calc

    fields = read_fields(write_record(atoms, filename=tmp_path / "h2.uma.txt"))

    gaps = [float(cell) for cell in fields["Atoms"][GAP_KEY].split()]
    assert min(gaps) > VACUUM_ANG
    assert not [key for key in fields["Atoms"] if key.startswith("vacuum")]


def test_gaps_are_plane_spacings_in_a_skewed_cell() -> None:
    # the gap along an axis is measured normal to the plane of the other two
    atoms = Atoms("H", cell=[[4.0, 0.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 10.0]])
    volume = atoms.cell.volume
    cross = np.cross(atoms.cell[1], atoms.cell[2])

    gaps = uma_record._largest_gaps(atoms)

    assert gaps is not None
    assert gaps == pytest.approx([volume / np.linalg.norm(cross), 4.0, 10.0])


def test_stress_is_reported_for_periodic_cells_only(
    tmp_path: Path, calc: FAIRChemCalculator
) -> None:
    atoms = make_slab(calc)
    assert not atoms.pbc.all()

    fields = read_fields(write_record(atoms, filename=tmp_path / "slab.uma.txt"))
    assert STRESS_KEY not in fields["Results"]

    atoms.pbc = True
    fields = read_fields(write_record(atoms, filename=tmp_path / "bulk.uma.txt"))
    assert STRESS_KEY in fields["Results"]


def test_environment_is_described(tmp_path: Path, calc: FAIRChemCalculator) -> None:
    fields = read_fields(
        write_record(make_slab(calc), filename=tmp_path / "slab.uma.txt")
    )

    environment = fields["Environment"]
    assert environment["type"] and environment["name"] and environment["version"]
    assert environment["path"] == sys.prefix
    assert fields["Dependencies"]["python"] == platform.python_version()


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"conda-meta/pixi": '{"pixi_version": "0.40.0"}'}, ("pixi", "0.40.0")),
        ({"conda-meta/history": "# conda version: 26.1.1\n"}, ("conda", "26.1.1")),
        ({"conda-meta/history": "==> 2026-01-01 <==\n"}, ("conda", None)),
        ({"pyvenv.cfg": "home = /usr/bin\nuv = 0.5.0\n"}, ("uv", "0.5.0")),
        ({"pyvenv.cfg": "home = /usr/bin\n"}, ("venv", None)),
        ({}, ("system python", None)),
    ],
)
def test_environment_tool_is_read_from_the_prefix(
    tmp_path: Path, files: dict[str, str], expected: tuple[str, str | None]
) -> None:
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    assert uma_record._environment_tool(tmp_path) == expected


def test_slurm_array_task_is_described(
    tmp_path: Path, calc: FAIRChemCalculator, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "5")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "short")
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "4")
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "1024")

    fields = read_fields(
        write_record(make_slab(calc), filename=tmp_path / "slab.uma.txt")
    )

    run = fields["Run"]
    assert run["job id"] == "5_3"
    assert run["partition"] == "short"
    assert run["cpus"] == "4"
    assert run["memory per cpu / MB"] == "1073.7"
    assert "memory / MB" not in run


def test_slurm_job_is_described(
    tmp_path: Path, calc: FAIRChemCalculator, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "4")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "2048")
    monkeypatch.delenv("SLURM_MEM_PER_CPU", raising=False)

    fields = read_fields(
        write_record(make_slab(calc), filename=tmp_path / "slab.uma.txt")
    )

    run = fields["Run"]
    assert run["job id"] == "7"
    assert run["cpus"] == "2"
    assert run["memory / MB"] == "2147.5"


def test_no_slurm_rows_outside_a_job(
    tmp_path: Path, calc: FAIRChemCalculator, monkeypatch: pytest.MonkeyPatch
) -> None:
    for variable in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        monkeypatch.delenv(variable, raising=False)

    fields = read_fields(
        write_record(make_slab(calc), filename=tmp_path / "slab.uma.txt")
    )

    assert "job id" not in fields["Run"]


def test_process_start_is_recorded(
    tmp_path: Path, calc: FAIRChemCalculator, monkeypatch: pytest.MonkeyPatch
) -> None:
    if sys.platform != "linux":
        pytest.skip("the process start is read from /proc")
    fields = read_fields(
        write_record(make_slab(calc), filename=tmp_path / "slab.uma.txt")
    )
    assert fields["Run"]["process started"] <= fields["Run"]["written"]

    # elsewhere the import time stands in, under its own name
    monkeypatch.setattr(uma_record, "_PROCESS_START", None)
    fields = read_fields(
        write_record(make_slab(calc), filename=tmp_path / "slab.uma.txt")
    )
    assert "process started" not in fields["Run"]
    assert fields["Run"]["module imported"] <= fields["Run"]["written"]


def test_duration_is_worded() -> None:
    assert uma_record._duration(90061.5) == "1 days 1 hours 1 minutes 1.5 seconds"


def test_dynamics_are_recorded(tmp_path: Path, calc: FAIRChemCalculator) -> None:
    atoms = make_slab(calc)
    trajectory = tmp_path / "slab.traj"
    optimizer = BFGS(atoms, logfile=None, trajectory=trajectory)
    optimizer.run(fmax=1e-6, steps=2)

    fields = read_fields(
        write_record(atoms, dyn=optimizer, filename=tmp_path / "slab.uma.txt")
    )

    dynamics = fields["Dynamics"]
    assert dynamics["optimizer"] == "BFGS"
    assert float(dynamics["fmax"]) == pytest.approx(1e-6)
    assert dynamics["nsteps"] == "2"
    assert dynamics["converged"] == "False"
    assert dynamics["trajectory"] == str(trajectory)
    assert dynamics["logfile"] == "none"


def test_dyn_given_a_path_is_rejected(tmp_path: Path, calc: FAIRChemCalculator) -> None:
    record = tmp_path / "slab.uma.txt"

    with pytest.raises(TypeError, match="filename="):
        write_record(make_slab(calc), dyn=record)
    assert not record.exists()


def test_host_is_described(tmp_path: Path, calc: FAIRChemCalculator) -> None:
    fields = read_fields(
        write_record(make_slab(calc), filename=tmp_path / "slab.uma.txt")
    )

    assert fields["Run"]["os"] == platform.platform()
    assert fields["Calculator"]["torch threads"] == str(torch.get_num_threads())
    assert "cudnn" in fields["Dependencies"]


def test_packages_are_listed_on_request(
    tmp_path: Path, calc: FAIRChemCalculator
) -> None:
    without = read_fields(
        write_record(make_slab(calc), filename=tmp_path / "slab.uma.txt")
    )
    given = read_fields(
        write_record(make_slab(calc), filename=tmp_path / "slab.uma.txt", packages=True)
    )

    assert "Installed packages" not in without
    assert given["Installed packages"]["ase"] == metadata.version("ase")


def test_molecular_dynamics_are_recorded(
    tmp_path: Path, calc: FAIRChemCalculator
) -> None:
    atoms = make_slab(calc)
    dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs, logfile=None)
    dyn.run(2)

    fields = read_fields(write_record(atoms, dyn=dyn, filename=tmp_path / "md.uma.txt"))

    assert fields["Dynamics"]["md-type"] == "VelocityVerlet"
    assert fields["Dynamics"]["nsteps"] == "2"
    assert "converged" not in fields["Dynamics"]


def test_thermostat_settings_are_recorded_when_ase_reports_them(
    tmp_path: Path, calc: FAIRChemCalculator
) -> None:
    atoms = make_slab(calc)
    dyn = Langevin(
        atoms,
        timestep=1.0 * units.fs,
        temperature_K=300.0,
        friction=0.01 / units.fs,
        fixcm=False,
        logfile=None,
    )
    dyn.run(2)

    fields = read_fields(write_record(atoms, dyn=dyn, filename=tmp_path / "md.uma.txt"))

    assert fields["Dynamics"]["md-type"] == "Langevin"
    assert float(fields["Dynamics"]["temperature_K"]) == pytest.approx(300.0)
    assert "friction" in fields["Dynamics"]


def test_constraints_are_described(tmp_path: Path, calc: FAIRChemCalculator) -> None:
    atoms = make_slab(calc)
    atoms.set_constraint([FixAtoms(indices=[0, 2, 3, 6]), FixBondLength(4, 5)])

    fields = read_fields(write_record(atoms, filename=tmp_path / "slab.uma.txt"))

    constraints = fields["Atoms"]["constraints"]
    assert constraints.startswith(
        "FixAtoms on 4 atoms (indices 0, 2-3, 6), FixBondLengths("
    )
    assert " object at " not in constraints


def test_index_runs_are_collapsed() -> None:
    assert uma_record._index_ranges([5, 0, 1, 2, 7]) == "0-2, 5, 7"
    assert uma_record._index_ranges([]) == ""


def test_structure_block_keeps_results_without_info(
    tmp_path: Path, calc: FAIRChemCalculator, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ASE rejects some atoms.info entries when writing extended xyz. The
    # fallback drops the info, not the energy
    def write_rejecting_info(buffer: Any, frame: Atoms, **kwargs: Any) -> None:
        if frame.info:
            raise ValueError("stand-in for an entry extended xyz cannot store")
        write(buffer, frame, **kwargs)

    # the name uma_record imports from ase.io, patched by its path since it
    # is not part of the public interface
    monkeypatch.setattr("uma_record.write", write_rejecting_info)
    atoms = make_slab(calc)
    atoms.info["label"] = "slab"
    energy = atoms.get_potential_energy()

    recorded = read_record(write_record(atoms, filename=tmp_path / "slab.uma.txt"))

    assert "label" not in recorded.info
    assert recorded.get_potential_energy() == pytest.approx(energy, abs=1e-6)


def test_info_entries_extended_xyz_cannot_hold_are_dropped() -> None:
    info = {
        "label": "slab",
        "count": 2,
        "array": np.arange(2),
        "lines": "two\nlines",
        "opaque": object(),
    }

    kept = uma_record._storable_info(info)

    assert set(kept) == {"label", "count", "array"}


def test_charge_and_spin_are_recorded_for_omol(
    tmp_path: Path, checkpoint: Path
) -> None:
    predict_unit: Any = StandInPredictUnit(checkpoint)
    atoms = molecule("H2O")
    atoms.info.update(charge=1, spin=2)
    atoms.calc = FAIRChemCalculator(predict_unit, task_name="omol")

    record = write_record(atoms, filename=tmp_path / "water.uma.txt")

    fields = read_fields(record)
    assert fields["Atoms"]["charge"] == "1"
    assert fields["Atoms"]["spin"] == "2"
    assert GAP_KEY not in fields["Atoms"]
    assert read_record(record).get_chemical_formula() == "H2O"


def test_listed_checksum_is_used(
    tmp_path: Path, checkpoint: Path, calc: FAIRChemCalculator
) -> None:
    listed = "a" * 32
    checkpoint.with_name(checkpoint.name + ".md5").write_text(
        f"{listed}  {checkpoint.name}\n"
    )
    atoms = make_slab(calc)

    fields = read_fields(write_record(atoms, filename=tmp_path / "slab.uma.txt"))

    assert fields["Calculator"]["checkpoint md5"] == listed


def test_computed_checksum_is_listed(
    tmp_path: Path, checkpoint: Path, calc: FAIRChemCalculator
) -> None:
    atoms = make_slab(calc)

    write_record(atoms, filename=tmp_path / "slab.uma.txt")

    for algorithm in ("md5", "sha256"):
        digest = hashlib.new(algorithm, CHECKPOINT_BYTES).hexdigest()
        listing = checkpoint.with_name(f"{checkpoint.name}.{algorithm}").read_text()
        assert listing == f"{digest}  {checkpoint.name}\n"
    leftovers = [entry.name for entry in tmp_path.iterdir()]
    assert not [name for name in leftovers if name.startswith(".")]


def test_download_sha256_is_read_from_the_blob_name(tmp_path: Path) -> None:
    # the Hugging Face cache links snapshots/<commit>/<file> to blobs/<sha256>
    digest = hashlib.sha256(CHECKPOINT_BYTES).hexdigest()
    blob = tmp_path / "blobs" / digest
    blob.parent.mkdir()
    blob.write_bytes(CHECKPOINT_BYTES)
    checkpoint = tmp_path / "snapshots" / "abc123" / "checkpoints" / "uma-s-test.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.symlink_to(blob)
    predict_unit: Any = StandInPredictUnit(checkpoint)
    atoms = make_slab(FAIRChemCalculator(predict_unit, task_name="oc20"))

    fields = read_fields(write_record(atoms, filename=tmp_path / "slab.uma.txt"))

    assert fields["Calculator"]["checkpoint sha256"] == digest
    assert (
        fields["Calculator"]["checkpoint md5"]
        == hashlib.md5(CHECKPOINT_BYTES).hexdigest()
    )
    assert not checkpoint.with_name("uma-s-test.pt.sha256").exists()


def test_listing_older_than_the_checkpoint_is_replaced(
    tmp_path: Path, checkpoint: Path, calc: FAIRChemCalculator
) -> None:
    listing = checkpoint.with_name(checkpoint.name + ".md5")
    listing.write_text(f"{'a' * 32}  {checkpoint.name}\n")
    # a checkpoint replaced after the listing was written is newer than it
    replaced_ns = listing.stat().st_mtime_ns + 10**9
    os.utime(checkpoint, ns=(replaced_ns, replaced_ns))
    atoms = make_slab(calc)

    fields = read_fields(write_record(atoms, filename=tmp_path / "slab.uma.txt"))

    digest = hashlib.md5(CHECKPOINT_BYTES).hexdigest()
    assert fields["Calculator"]["checkpoint md5"] == digest
    assert listing.read_text() == f"{digest}  {checkpoint.name}\n"


def test_unreadable_details_are_unavailable(tmp_path: Path) -> None:
    predict_unit: Any = BarePredictUnit()
    atoms = make_slab(FAIRChemCalculator(predict_unit, task_name="oc20"))

    fields = read_fields(write_record(atoms, filename=tmp_path / "slab.uma.txt"))

    calculator = fields["Calculator"]
    assert calculator["checkpoint"] == UNAVAILABLE
    assert calculator["device"] == UNAVAILABLE
    assert calculator["seed"] == UNAVAILABLE
    assert calculator["model_id"] == UNAVAILABLE
    assert "model" not in fields["References"]
    assert fields["References"]["software"] == "fairchem"


def test_explicit_checkpoint_is_used(tmp_path: Path, checkpoint: Path) -> None:
    predict_unit: Any = BarePredictUnit()
    atoms = make_slab(FAIRChemCalculator(predict_unit, task_name="oc20"))

    fields = read_fields(
        write_record(atoms, filename=tmp_path / "slab.uma.txt", checkpoint=checkpoint)
    )

    assert fields["Calculator"]["checkpoint"] == str(checkpoint)
    assert (
        fields["Calculator"]["checkpoint md5"]
        == hashlib.md5(CHECKPOINT_BYTES).hexdigest()
    )


def test_explicit_checkpoint_wins_over_the_reported_one(
    tmp_path: Path, calc: FAIRChemCalculator
) -> None:
    other = tmp_path / "other.pt"
    other.write_bytes(b"other checkpoint")

    fields = read_fields(
        write_record(make_slab(calc), filename=tmp_path / "s.uma.txt", checkpoint=other)
    )

    assert fields["Calculator"]["checkpoint"] == str(other)
    assert (
        fields["Calculator"]["checkpoint md5"]
        == hashlib.md5(b"other checkpoint").hexdigest()
    )


def test_explicit_checkpoint_must_be_a_file(tmp_path: Path) -> None:
    predict_unit: Any = BarePredictUnit()
    atoms = make_slab(FAIRChemCalculator(predict_unit, task_name="oc20"))

    with pytest.raises(ValueError, match="not a file"):
        write_record(atoms, filename=tmp_path / "slab.uma.txt", checkpoint="uma-s-1p2")
    assert not (tmp_path / "slab.uma.txt").exists()


@pytest.mark.skipif(
    not os.environ.get(REAL_CHECKPOINT_VARIABLE),
    reason=f"set {REAL_CHECKPOINT_VARIABLE} to a UMA checkpoint to run",
)
def test_real_predict_unit_leaves_nothing_unavailable(tmp_path: Path) -> None:
    # The stand-in defines the private fairchem attributes itself, so only a
    # real predict unit shows whether a fairchem release renamed them
    predict_unit = load_predict_unit(os.environ[REAL_CHECKPOINT_VARIABLE], device="cpu")
    atoms = make_slab(FAIRChemCalculator(predict_unit, task_name="oc20"))

    record = write_record(atoms, filename=tmp_path / "slab.uma.txt")

    fields = read_fields(record)
    assert UNAVAILABLE not in record.read_text()
    assert fields["Calculator"]["model_id"].lower().startswith("uma")
    assert fields["References"]["model"] == "wood2025uma"


def test_wrapped_calculator_is_found(
    tmp_path: Path, checkpoint: Path, calc: FAIRChemCalculator
) -> None:
    atoms = make_slab(SumCalculator([calc]))

    fields = read_fields(write_record(atoms, filename=tmp_path / "slab.uma.txt"))

    assert fields["Calculator"]["calculator"] == "FAIRChemCalculator"
    assert fields["Calculator"]["wrapped by"] == "SumCalculator"
    assert fields["Calculator"]["checkpoint"] == str(checkpoint)


def test_calculator_attribute_of_a_wrapper_is_followed(
    tmp_path: Path, calc: FAIRChemCalculator
) -> None:
    atoms = make_slab(DelegatingCalculator(calc))
    energy = atoms.get_potential_energy()

    fields = read_fields(write_record(atoms, filename=tmp_path / "slab.uma.txt"))

    assert fields["Calculator"]["wrapped by"] == "DelegatingCalculator"
    assert float(fields["Results"]["energy / eV"]) == pytest.approx(energy, abs=1e-6)


def test_filename_defaults_to_the_environment_variable(
    tmp_path: Path, calc: FAIRChemCalculator, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "results" / "job.uma.txt"
    monkeypatch.setenv("UMA_RECORD_PATH", str(target))

    assert write_record(make_slab(calc)) == target
    assert target.is_file()


def test_record_named_by_the_environment_variable_is_not_replaced(
    tmp_path: Path, calc: FAIRChemCalculator, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "job.uma.txt"
    monkeypatch.setenv("UMA_RECORD_PATH", str(target))
    first = write_record(make_slab(calc))

    second = write_record(make_slab(calc))

    assert first == target
    assert second != target
    assert second.parent == tmp_path
    assert second.name.startswith("job-") and second.name.endswith(".uma.txt")
    assert first.is_file() and second.is_file()


def test_time_stamp_goes_before_any_suffix(
    tmp_path: Path, calc: FAIRChemCalculator, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "job.log"
    monkeypatch.setenv("UMA_RECORD_PATH", str(target))
    write_record(make_slab(calc))

    second = write_record(make_slab(calc))

    assert second.name.startswith("job-") and second.name.endswith(".log")
    assert ".uma" not in second.name


def test_filename_defaults_to_a_timestamped_name(
    tmp_path: Path, calc: FAIRChemCalculator, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    record = write_record(make_slab(calc))

    assert record.parent == Path(".")
    assert record.name.startswith("Cu8-")
    assert record.name.endswith(".uma.txt")
    assert (tmp_path / record.name).is_file()


@pytest.mark.skipif(os.name != "posix", reason="file modes follow the umask on posix")
def test_record_and_listings_follow_the_umask(
    tmp_path: Path, checkpoint: Path, calc: FAIRChemCalculator
) -> None:
    # the temporary file is opened for the owner only, the written files
    # must not stay that way
    previous = os.umask(0o027)
    try:
        record = write_record(make_slab(calc), filename=tmp_path / "slab.uma.txt")
    finally:
        os.umask(previous)

    listing = checkpoint.with_name(checkpoint.name + ".md5")
    assert stat.S_IMODE(record.stat().st_mode) == 0o640
    assert stat.S_IMODE(listing.stat().st_mode) == 0o640


def test_cudnn_version_is_dotted() -> None:
    assert uma_record._cudnn_text(92000) == "9.20.0"
    assert uma_record._cudnn_text(8902) == "8.9.2"
    assert uma_record._cudnn_text(None) == "none"
    assert uma_record._cudnn_text(uma_record._MISSING) == "unavailable"


def test_no_temporary_file_is_left(tmp_path: Path, calc: FAIRChemCalculator) -> None:
    output = tmp_path / "records"

    write_record(make_slab(calc), filename=output / "slab.uma.txt")

    assert [entry.name for entry in output.iterdir()] == ["slab.uma.txt"]


def test_atoms_without_calculator_are_rejected(tmp_path: Path) -> None:
    atoms = fcc111("Cu", size=(2, 2, 2), vacuum=VACUUM_ANG)

    with pytest.raises(ValueError, match="no calculator"):
        write_record(atoms, filename=tmp_path / "slab.uma.txt")
    assert not (tmp_path / "slab.uma.txt").exists()


def test_other_calculators_are_rejected(tmp_path: Path) -> None:
    atoms = make_slab(EMT())

    with pytest.raises(ValueError, match="no FAIRChemCalculator"):
        write_record(atoms, filename=tmp_path / "slab.uma.txt")


def test_non_atoms_input_is_rejected(tmp_path: Path) -> None:
    record = tmp_path / "slab.uma.txt"

    with pytest.raises(TypeError, match="ase.Atoms"):
        write_record("slab.traj", filename=record)  # type: ignore[arg-type]


def test_ase_reads_the_record_by_its_first_line(
    tmp_path: Path, calc: FAIRChemCalculator
) -> None:
    atoms = make_slab(calc)
    energy = atoms.get_potential_energy()
    record = write_record(atoms, filename=tmp_path / "slab.uma.txt")

    recorded = read(record)

    assert isinstance(recorded, Atoms)
    assert recorded.get_chemical_formula() == "Cu8"
    assert recorded.get_potential_energy() == pytest.approx(energy, abs=1e-6)


def test_record_without_structure_block_is_rejected(tmp_path: Path) -> None:
    record = tmp_path / "empty.uma.txt"
    record.write_text("UMA calculation record (uma-record v0)\n")

    with pytest.raises(ValueError, match="no structure block"):
        read_record(record)


def test_fields_are_read_by_section(tmp_path: Path, calc: FAIRChemCalculator) -> None:
    atoms = make_slab(calc)
    optimizer = BFGS(atoms, logfile=None)
    optimizer.run(fmax=1e-6, steps=1)
    record = write_record(atoms, dyn=optimizer, filename=tmp_path / "slab.uma.txt")

    fields = read_fields(record)

    # the same key in two sections, and the bibtex lines are not rows
    assert fields["Dynamics"]["type"] == "optimization"
    assert fields["Environment"]["type"] != "optimization"
    assert "title" not in fields["References"]
    assert "Structure, extended xyz" not in fields
    assert list(fields)[:2] == ["Run", "Calculator"]


def test_other_files_are_not_records(tmp_path: Path) -> None:
    other = tmp_path / "notes.txt"
    other.write_text("not a record\n")

    with pytest.raises(ValueError, match="not a UMA calculation record"):
        read_fields(other)
