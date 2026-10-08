"""
Write a human readable record of a UMA calculation.

'write_record' records the run, the fairchem calculator and its inference
settings, the final structure with energy and forces, and the package
versions in one text file. Everything is read from the 'FAIRChemCalculator'
attached to the atoms, so the calling script needs no extra bookkeeping.
'read_record' returns the structure stored in such a file. The file is also
registered with ASE as the 'uma-record' format, see 'uma_record_ase', so
'ase.io.read' and 'ase gui' open it as well.
"""

from __future__ import annotations

import dataclasses
import functools
import getpass
import hashlib
import io
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.dependencies import format_dependency
from ase.io import write
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit.api.inference import UMATask
from numpy.typing import NDArray

from uma_record_ase import (
    MAGIC,
    RECORD_SUFFIX,
    XYZ_TITLE,
    read_fields,
    read_uma_record,
)

__all__ = ["read_fields", "read_record", "read_uma_record", "write_record"]

try:
    __version__ = metadata.version("uma-record")
except metadata.PackageNotFoundError:
    __version__ = ""

_Row = tuple[str, str]

_FILENAME_VARIABLE: Final = "UMA_RECORD_PATH"
_UNAVAILABLE: Final = "unavailable"
_MISSING: Final = object()
# md5 as the fairchem model card lists it, sha256 as Hugging Face and git
# LFS list it, with the hex digest each one gives
_DIGEST_PATTERNS: Final = {
    "md5": re.compile(r"[0-9a-f]{32}"),
    "sha256": re.compile(r"[0-9a-f]{64}"),
}
# a checkpoint is a few GB, hashed in chunks
_HASH_CHUNK_BYTES: Final = 1 << 22
# only when the batch server predict unit holds no timeout of its own
_SERVER_TIMEOUT_S: Final = 10.0
# torch does not expose the driver version, nvidia-smi is asked once
_NVIDIA_SMI_TIMEOUT_S: Final = 5.0
# a gap this wide along a cell axis is vacuum rather than a layer spacing
_VACUUM_MIN_ANG: Final = 5.0
# wrappers nest a few levels at most, the cap only guards against cycles
_MAX_CALC_SEARCH: Final = 16
# the import is the closest point to the process start that every platform
# can give, Linux reports the start itself in /proc
_IMPORT_TIME: Final = time.time()
# import name and distribution name
_VERSION_MODULES: Final = (
    ("fairchem.core", "fairchem-core"),
    ("torch", "torch"),
    ("ase", "ase"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
)
_SLURM_VARIABLES: Final = (
    ("job name", "SLURM_JOB_NAME"),
    ("partition", "SLURM_JOB_PARTITION"),
    # the first is only set when the job asked for cpus per task
    ("cpus", "SLURM_CPUS_PER_TASK"),
    ("cpus", "SLURM_CPUS_ON_NODE"),
)
# Slurm reports memory in MiB, the record uses MB throughout. The second
# variable is set instead of the first when the job asked for memory per cpu
_SLURM_MEMORY_VARIABLES: Final = (
    ("memory / MB", "SLURM_MEM_PER_NODE"),
    ("memory per cpu / MB", "SLURM_MEM_PER_CPU"),
)
_MODEL_REFERENCE: Final = "wood2025uma"
_SOFTWARE_REFERENCE: Final = "fairchem"
# keyed by task_name. plain strings, UMATask does not hold every task of
# every fairchem release. The fairchem changelog names the dataset behind
# each task, odac is trained on ODAC23
_DATASET_REFERENCES: Final = {
    "oc20": "chanussot2021oc20",
    "oc22": "tran2023oc22",
    "oc25": "sahoo2025oc25",
    "omat": "barrosoluque2026omat24",
    "omol": "levine2025omol25",
    "odac": "sriram2024odac23",
    "omc": "gharakhanyan2026omc25",
}
# Preprints from the arXiv bibtex export of each paper, with the year of the
# first version. Journal and proceedings entries from the publisher metadata
# in Crossref, which fixes the title, author order, volume, and pages. The
# software entry holds the Zenodo concept DOI from the fairchem CITATION.cff
_BIBTEX: Final = {
    "fairchem": """\
@software{fairchem,
  title = {{FAIRChem}},
  author = {{FAIR Chemistry}},
  doi = {10.5281/zenodo.15587497},
  url = {https://github.com/facebookresearch/fairchem},
}""",
    "wood2025uma": """\
@inproceedings{wood2025uma,
  title = {{UMA}: A Family of Universal Models for Atoms},
  author = {Brandon M. Wood and Misko Dzamba and Xiang Fu and Meng Gao and Muhammed Shuaibi and Luis Barroso-Luque and Kareem Abdelmaqsoud and Vahe Gharakhanyan and John R. Kitchin and Daniel S. Levine and Kyle Michel and Anuroop Sriram and Taco Cohen and Abhishek Das and Sushree Jagriti Sahoo and Ammar Rizvi and Zachary W. Ulissi and C. Lawrence Zitnick},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {38},
  pages = {143528--143564},
  year = {2025},
  doi = {10.52202/085713-4310},
  eprint = {2506.23971},
  archivePrefix = {arXiv},
}""",
    "chanussot2021oc20": """\
@article{chanussot2021oc20,
  title = {Open Catalyst 2020 ({OC20}) Dataset and Community Challenges},
  author = {Lowik Chanussot and Abhishek Das and Siddharth Goyal and Thibaut Lavril and Muhammed Shuaibi and Morgane Riviere and Kevin Tran and Javier Heras-Domingo and Caleb Ho and Weihua Hu and Aini Palizhati and Anuroop Sriram and Brandon Wood and Junwoong Yoon and Devi Parikh and C. Lawrence Zitnick and Zachary Ulissi},
  journal = {ACS Catalysis},
  volume = {11},
  number = {10},
  pages = {6059--6072},
  year = {2021},
  doi = {10.1021/acscatal.0c04525},
  eprint = {2010.09990},
  archivePrefix = {arXiv},
}""",
    "tran2023oc22": """\
@article{tran2023oc22,
  title = {The Open Catalyst 2022 ({OC22}) Dataset and Challenges for Oxide Electrocatalysts},
  author = {Richard Tran and Janice Lan and Muhammed Shuaibi and Brandon M. Wood and Siddharth Goyal and Abhishek Das and Javier Heras-Domingo and Adeesh Kolluru and Ammar Rizvi and Nima Shoghi and Anuroop Sriram and F\\'elix Therrien and Jehad Abed and Oleksandr Voznyy and Edward H. Sargent and Zachary Ulissi and C. Lawrence Zitnick},
  journal = {ACS Catalysis},
  volume = {13},
  number = {5},
  pages = {3066--3084},
  year = {2023},
  doi = {10.1021/acscatal.2c05426},
  eprint = {2206.08917},
  archivePrefix = {arXiv},
}""",
    "sahoo2025oc25": """\
@misc{sahoo2025oc25,
  title = {The Open Catalyst 2025 ({OC25}) Dataset and Models for Solid-Liquid Interfaces},
  author = {Sushree Jagriti Sahoo and Mikael Maraschin and Daniel S. Levine and Zachary Ulissi and C. Lawrence Zitnick and Joel B Varley and Joseph A. Gauthier and Nitish Govindarajan and Muhammed Shuaibi},
  year = {2025},
  eprint = {2509.17862v1},
  archivePrefix = {arXiv},
  primaryClass = {cond-mat.mtrl-sci},
  note = {Version 1 describes the dataset, version 2 of the preprint carries another title},
}""",
    "barrosoluque2026omat24": """\
@article{barrosoluque2026omat24,
  title = {The Open Materials 2024 ({OMat24}) Inorganic Materials Dataset and Models},
  author = {Luis Barroso-Luque and Muhammed Shuaibi and Xiang Fu and Brandon M. Wood and Misko Dzamba and Meng Gao and Ammar Rizvi and Matt Uyttendaele and C. Lawrence Zitnick and Zachary W. Ulissi},
  journal = {Nature Computational Science},
  volume = {6},
  number = {6},
  pages = {642--652},
  year = {2026},
  doi = {10.1038/s43588-026-00996-w},
  eprint = {2410.12771},
  archivePrefix = {arXiv},
}""",
    "levine2025omol25": """\
@misc{levine2025omol25,
  title = {The Open Molecules 2025 ({OMol25}) Dataset, Evaluations, and Models},
  author = {Daniel S. Levine and Muhammed Shuaibi and Evan Walter Clark Spotte-Smith and Michael G. Taylor and Muhammad R. Hasyim and Kyle Michel and Ilyes Batatia and G\\'abor Cs\\'anyi and Misko Dzamba and Peter Eastman and Nathan C. Frey and Xiang Fu and Vahe Gharakhanyan and Aditi S. Krishnapriyan and Joshua A. Rackers and Sanjeev Raja and Ammar Rizvi and Andrew S. Rosen and Zachary Ulissi and Santiago Vargas and C. Lawrence Zitnick and Samuel M. Blau and Brandon M. Wood},
  year = {2025},
  eprint = {2505.08762},
  archivePrefix = {arXiv},
  primaryClass = {physics.chem-ph},
}""",
    "sriram2024odac23": """\
@article{sriram2024odac23,
  title = {The Open {DAC} 2023 Dataset and Challenges for Sorbent Discovery in Direct Air Capture},
  author = {Anuroop Sriram and Sihoon Choi and Xiaohan Yu and Logan M. Brabson and Abhishek Das and Zachary Ulissi and Matt Uyttendaele and Andrew J. Medford and David S. Sholl},
  journal = {ACS Central Science},
  volume = {10},
  number = {5},
  pages = {923--941},
  year = {2024},
  doi = {10.1021/acscentsci.3c01629},
  eprint = {2311.00341},
  archivePrefix = {arXiv},
}""",
    "gharakhanyan2026omc25": """\
@article{gharakhanyan2026omc25,
  title = {Open Molecular Crystals 2025 ({OMC25}) Dataset and Models},
  author = {Vahe Gharakhanyan and Luis Barroso-Luque and Yi Yang and Muhammed Shuaibi and Kyle Michel and Daniel S. Levine and Misko Dzamba and Xiang Fu and Meng Gao and Xingyu Liu and Haoran Ni and Keian Noori and Brandon M. Wood and Matt Uyttendaele and Arman Boromand and C. Lawrence Zitnick and Noa Marom and Zachary W. Ulissi and Anuroop Sriram},
  journal = {Scientific Data},
  volume = {13},
  pages = {354},
  year = {2026},
  doi = {10.1038/s41597-026-06628-2},
  eprint = {2508.02651},
  archivePrefix = {arXiv},
}""",
}


def write_record(
    atoms: Atoms,
    *,
    dyn: object | None = None,
    filename: str | os.PathLike[str] | None = None,
    checkpoint: str | os.PathLike[str] | None = None,
    packages: bool = False,
) -> Path:
    """
    Write the record for the current state of 'atoms' and return its path.

    'atoms' must carry a 'FAIRChemCalculator', directly or inside a wrapper.
    'dyn' is the ASE optimizer or molecular dynamics object that produced the
    structure. 'filename' defaults to the 'UMA_RECORD_PATH' environment variable,
    then to '<formula>-<timestamp>.uma.txt' in the working directory. A file
    named by 'filename' is replaced, a file named by the variable is kept and
    the new record gets the time stamp before its suffix. 'checkpoint' is only
    needed when the calculator cannot report its own, and must be an existing
    file. 'packages' adds every installed package with its version, the
    record lists only the packages that decide the result otherwise.

    Energy and forces come from the last calculation when it matches the
    structure. Otherwise they are calculated again and the record says so.
    Details that cannot be read are written as "unavailable".

    Side effects beyond the record. Missing directories of 'filename' are
    created. The md5 and sha256 of the checkpoint are written to
    '<checkpoint>.md5' and '<checkpoint>.sha256' beside it when they are not
    listed there yet and the directory is writable.

    Raise 'TypeError' when 'atoms' is not an 'ase.Atoms' or 'dyn' is a path,
    'ValueError' when no 'FAIRChemCalculator' is attached or 'checkpoint' is
    not a file, and 'OSError' when the file cannot be written.
    """
    if not isinstance(atoms, Atoms):
        raise TypeError(f"atoms must be an ase.Atoms, got {type(atoms).__name__}")
    if isinstance(dyn, (str, bytes, os.PathLike)):
        # the mistake of the earlier positional signature, a filename as dyn
        raise TypeError(
            f"dyn must be an ASE optimizer or molecular dynamics object, got "
            f"{type(dyn).__name__}, pass the record path as filename="
        )
    calc = _find_calc(atoms.calc)
    predict_unit = calc.predictor
    task_name = calc.task_name
    model_checkpoint = _find_checkpoint(predict_unit, checkpoint)
    model_id = _text(_attempt(lambda: _model(predict_unit).model_id))
    # Order matters from here. Asking 'dyn' for convergence can run the
    # calculator on other structures, as in a band with a shared calculator,
    # so it goes first. The results are read right after the evaluation, and
    # the settings after that, since a calculation can change them.
    dyn_rows = _dyn_rows(dyn) if dyn is not None else []
    recalculated = _evaluate(atoms)
    results_rows = _results_rows(atoms, recalculated)
    xyz_lines = _xyz_lines(atoms)

    lines = [_header(), ""]
    lines += _section("Run", _run_rows())
    lines += _section(
        "Calculator", _calc_rows(calc, atoms.calc, model_id, model_checkpoint)
    )
    lines += _section(
        "Inference settings, effective when written", _settings_rows(predict_unit)
    )
    if dyn_rows:
        lines += _section("Dynamics", dyn_rows)
    lines += _section("Atoms", _atoms_rows(atoms, task_name))
    lines += _section("Results", results_rows)
    references = _reference_rows(model_id, model_checkpoint, task_name)
    lines += _section("References", references)
    lines += _bibtex_lines(references)
    lines += _section("Environment", _environment_rows())
    lines += _section("Dependencies", _dependency_rows())
    if packages:
        lines += _section("Installed packages", _package_rows())
    # last, so that everything below the title is one extended xyz frame
    lines += xyz_lines

    text = "\n".join(lines).rstrip("\n") + "\n"
    assert text.startswith(MAGIC.decode()), "ASE recognises a record by its first line"
    target = Path(filename) if filename is not None else _default_filename(atoms)
    _write_atomically(target, text)
    return target


def read_record(filename: str | os.PathLike[str]) -> Atoms:
    """
    Return the structure stored in a record, with its energy and forces.

    The results sit on a single point calculator, as for any structure that
    ASE reads from an extended xyz file. Raise 'ValueError' when the file
    holds no structure block.
    """
    with Path(filename).open(encoding="utf-8") as fileobj:
        return read_uma_record(fileobj)


def _header() -> str:
    # no version when the module runs from a checkout that is not installed
    writer = f"uma-record v{__version__}" if __version__ else "uma-record"
    writer = _with_commit(writer, "uma-record")
    return f"{MAGIC.decode()} ({writer})"


def _attempt(function: Callable[[], object]) -> object:
    """Return 'function()', or '_MISSING' when it fails."""
    # Used only on fairchem and ASE internals that differ between versions. A
    # detail that cannot be read must not stop the record at the end of a long
    # job, so every failure maps to the same marker.
    try:
        return function()
    except Exception:
        return _MISSING


def _getattr(source: object, name: str) -> object:
    return _attempt(lambda: getattr(source, name))


def _text(value: object) -> str:
    if value is _MISSING:
        return _UNAVAILABLE
    if value is None:
        return "none"
    if isinstance(value, (set, frozenset)):
        # sets have no stable order
        return "[" + ", ".join(sorted(str(item) for item in value)) + "]"
    return " ".join(str(value).splitlines())


def _number(value: float) -> str:
    return f"{float(value):.6f}"


def _vector(values: Sequence[float] | NDArray[np.float64]) -> str:
    """Return numbers in columns, a space stands in for each plus sign."""
    return " ".join(f"{float(value): .6f}" for value in values).lstrip()


def _duration(seconds: float) -> str:
    """Return a duration in words, as ORCA prints its run time."""
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, rest = divmod(rest, 60)
    return (
        f"{int(days)} days {int(hours)} hours {int(minutes)} minutes "
        f"{rest:.1f} seconds"
    )


def _timestamp(seconds: float) -> str:
    """Return local time in ISO 8601 form with its UTC offset."""
    return datetime.fromtimestamp(seconds).astimezone().isoformat(timespec="seconds")


def _section(title: str, rows: Sequence[_Row]) -> list[str]:
    """Return a titled block of aligned 'key: value' lines and a blank line."""
    if not rows:
        return [title, "  none", ""]
    width = max(len(key) for key, _ in rows) + 1
    lines = [title]
    for key, value in rows:
        # the first colon separates key from value when the record is parsed
        label = key.replace(":", " ") + ":"
        # a minus sign hangs into the gap, so digits align with text, and at
        # least one space separates it from the label
        gap = " " if value.startswith("-") else "  "
        lines.append(f"  {label:<{width}}{gap}{value}".rstrip())
    lines.append("")
    return lines


def _find_calc(calc: object) -> FAIRChemCalculator:
    """
    Return the 'FAIRChemCalculator' behind 'calc'.

    Search through wrappers that keep their inner calculators in 'calculator',
    as 'FormationEnergyCalculator' does, or in 'calcs' or 'mixer.calcs', as
    the ASE mixing calculators do. Raise 'ValueError' when none is found.
    """
    if calc is None:
        raise ValueError(
            "atoms has no calculator, attach the FAIRChemCalculator before "
            "calling write_record"
        )
    pending = [calc]
    for _ in range(_MAX_CALC_SEARCH):
        if not pending:
            break
        candidate = pending.pop(0)
        if isinstance(candidate, FAIRChemCalculator):
            return candidate
        pending.extend(_wrapped_calcs(candidate))
    raise ValueError(
        f"atoms has no FAIRChemCalculator, found {type(calc).__name__} instead"
    )


def _wrapped_calcs(calc: object) -> list[object]:
    # a wrapper with its own __getattr__ can raise on any name
    wrapped: list[object] = []
    inner = _getattr(calc, "calculator")
    if inner is not _MISSING and inner is not None:
        wrapped.append(inner)
    mixer = _getattr(calc, "mixer")
    calcs = _getattr(calc if mixer is _MISSING else mixer, "calcs")
    if isinstance(calcs, (list, tuple)):
        wrapped.extend(calcs)
    return wrapped


def _evaluate(atoms: Atoms) -> bool:
    """
    Make the results of 'atoms.calc' match 'atoms'.

    Return True when that took a new calculation.
    """
    calc = atoms.calc
    recalculated = "energy" not in calc.results or bool(calc.check_state(atoms))
    atoms.get_potential_energy()
    return recalculated


def _run_rows() -> list[_Row]:
    written = time.time()
    rows = [("command", shlex.join(sys.argv))]
    rows.extend(_script_rows())
    rows.extend(
        [
            ("user", _user()),
            ("node", platform.node() or _UNAVAILABLE),
            ("os", platform.platform()),
            # the working directory can have been removed under the process
            ("working directory", _text(_attempt(os.getcwd))),
        ]
    )
    rows.extend(_slurm_rows())
    # the process rows describe the whole process, not this structure, a script
    # that records several structures repeats them in every record
    started = _PROCESS_START
    if started is None:
        started = _IMPORT_TIME
        rows.append(("module imported", _timestamp(started)))
    else:
        rows.append(("process started", _timestamp(started)))
    rows.extend(
        [
            ("written", _timestamp(written)),
            ("process elapsed", _duration(written - started)),
            ("peak process memory / MB", _peak_memory()),
        ]
    )
    return rows


def _process_start_time() -> float | None:
    """Return the process start as a unix time, None outside Linux."""
    # /proc/self/stat field 22 is the start in clock ticks since boot, the
    # fields after the command name in parentheses start at field 3
    try:
        stat = Path("/proc/self/stat").read_text(encoding="ascii")
        ticks = int(stat.rpartition(")")[2].split()[19])
        boot = next(
            int(line.split()[1])
            for line in Path("/proc/stat").read_text(encoding="ascii").splitlines()
            if line.startswith("btime ")
        )
        return boot + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, StopIteration):
        return None


_PROCESS_START: Final = _process_start_time()


def _user() -> str:
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        # containers can run under a user id without a passwd entry
        return _UNAVAILABLE


# The cached functions below describe the process, not the structure. A script
# that records thousands of structures must not hash and scan again for each one.
@functools.cache
def _script_rows() -> tuple[_Row, ...]:
    if not sys.argv or not sys.argv[0]:
        return ()
    script = Path(sys.argv[0])
    if not script.is_file():
        return ()
    return (
        ("script", str(script.resolve())),
        ("script sha256", _file_digest(script, "sha256") or _UNAVAILABLE),
    )


def _slurm_rows() -> list[_Row]:
    """Return what Slurm reports about the job, nothing outside a job."""
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_job and array_task:
        job_id = f"{array_job}_{array_task}"
    else:
        job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id:
        return []
    rows = [("job id", job_id)]
    for key, variable in _SLURM_VARIABLES:
        value = os.environ.get(variable)
        if value and key not in dict(rows):
            rows.append((key, value))
    for key, variable in _SLURM_MEMORY_VARIABLES:
        memory = os.environ.get(variable, "")
        if memory.isdigit():
            rows.append((key, f"{int(memory) * 1048576 / 1e6:.1f}"))
            break
    return rows


def _peak_memory() -> str:
    try:
        # not available on windows
        import resource
    except ImportError:
        return _UNAVAILABLE
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in kilobytes on linux and in bytes on macos
    in_bytes = maxrss if sys.platform == "darwin" else maxrss * 1024
    return f"{in_bytes / 1e6:.1f}"


def _calc_rows(
    calc: FAIRChemCalculator, wrapper: object, model_id: str, checkpoint: Path | None
) -> list[_Row]:
    predict_unit = calc.predictor
    rows = [("calculator", type(calc).__name__)]
    if wrapper is not calc:
        rows.append(("wrapped by", type(wrapper).__name__))
    rows.append(("predict_unit", type(predict_unit).__name__))
    rows.append(("task_name", _text(calc.task_name)))
    rows.append(("implemented_properties", ", ".join(calc.implemented_properties)))
    rows.append(("model_id", model_id))
    rows.append(("direct_forces", _text(_attempt(lambda: _direct_forces(calc)))))
    rows.extend(_checkpoint_rows(checkpoint))
    device = _predict_unit_value(predict_unit, "device")
    rows.append(("device", _text(device)))
    rows.extend(_gpu_rows(device))
    # what torch runs with, the inference settings only hold what fairchem set
    rows.append(("torch threads", str(torch.get_num_threads())))
    rows.append(("seed", _text(_predict_unit_value(predict_unit, "_seed"))))
    rows.append(("overrides", _text(_predict_unit_value(predict_unit, "_overrides"))))
    return rows


def _model(predict_unit: object) -> Any:
    # the model is a torch module tree without a usable static type. it is
    # not asked from a batch server, that would copy the whole model over
    model: Any = getattr(predict_unit, "model")
    return model.module


def _direct_forces(calc: FAIRChemCalculator) -> bool:
    """Return whether forces come from a direct head or from the gradient."""
    return bool(_model(calc.predictor).backbone.regress_config.direct_forces)


def _predict_unit_value(predict_unit: object, name: str) -> object:
    """Return a predict unit attribute, asking the batch server when needed."""
    value = _getattr(predict_unit, name)
    if value is _MISSING and hasattr(predict_unit, "server_handle"):
        value = _attempt(lambda: _served_attribute(predict_unit, name))
    return value


def _served_attribute(predict_unit: object, name: str) -> object:
    """
    Read an attribute of the predict unit behind a fairchem batch server.

    A 'BatchServerPredictUnit' holds only a server handle, the model lives in
    the server process. The call needs the server to be running.
    """
    # the handle is a ray object without a usable static type
    server_handle: Any = getattr(predict_unit, "server_handle")
    # fairchem sizes its own timeout for the server, None means no limit
    timeout = _getattr(predict_unit, "_request_timeout_s")
    if timeout is _MISSING:
        timeout = _SERVER_TIMEOUT_S
    response = server_handle.get_predict_unit_attribute.remote(name)
    result: object = response.result(timeout_s=timeout)
    return result


def _find_checkpoint(
    predict_unit: object, checkpoint: str | os.PathLike[str] | None
) -> Path | None:
    if checkpoint is not None:
        given = Path(checkpoint)
        if not given.is_file():
            raise ValueError(f"checkpoint is not a file: {given}")
        return given.absolute()
    found = _predict_unit_value(predict_unit, "_inference_model_path")
    if isinstance(found, (str, os.PathLike)):
        return Path(found).absolute()
    return None


def _checkpoint_rows(checkpoint: Path | None) -> list[_Row]:
    if checkpoint is None:
        return [("checkpoint", _UNAVAILABLE)]
    rows = [("checkpoint", str(checkpoint))]
    try:
        status = checkpoint.stat()
    except OSError:
        rows.extend(
            (f"checkpoint {algorithm}", _UNAVAILABLE) for algorithm in _DIGEST_PATTERNS
        )
        return rows
    digests = _checkpoint_digests(checkpoint, status.st_size, status.st_mtime_ns)
    for algorithm in _DIGEST_PATTERNS:
        rows.append((f"checkpoint {algorithm}", digests[algorithm] or _UNAVAILABLE))
    rows.append(("checkpoint size / bytes", str(status.st_size)))
    rows.append(("checkpoint modified", _timestamp(status.st_mtime)))
    return rows


@functools.lru_cache(maxsize=32)
def _checkpoint_digests(
    checkpoint: Path, size: int, mtime_ns: int
) -> dict[str, str | None]:
    """
    Return the md5 and sha256 of the checkpoint, None where unreadable.

    A '<checkpoint>.<algorithm>' beside the checkpoint is read when it is
    newer than the checkpoint. A Hugging Face download carries its sha256 in
    the blob name. The rest is hashed in one pass over the file and written
    beside it, so only the first run after a download pays for it. 'size' and
    'mtime_ns' only key the cache, so a replaced file is hashed again within
    one process.
    """
    digests: dict[str, str | None] = {}
    for algorithm in _DIGEST_PATTERNS:
        listing = checkpoint.with_name(f"{checkpoint.name}.{algorithm}")
        digest = _listed_digest(listing, checkpoint.name, algorithm, mtime_ns)
        if digest is None and algorithm == "sha256":
            digest = _download_sha256(checkpoint)
        digests[algorithm] = digest
    missing = [algorithm for algorithm, digest in digests.items() if digest is None]
    for algorithm, digest in _file_digests(checkpoint, missing).items():
        digests[algorithm] = digest
        # in md5sum and sha256sum output format, so the file also serves
        # their -c option. A directory that cannot be written, a shared
        # install say, is left alone and hashed every run
        listing = checkpoint.with_name(f"{checkpoint.name}.{algorithm}")
        try:
            _write_atomically(listing, f"{digest}  {checkpoint.name}\n")
        except OSError:
            pass
    return digests


def _listed_digest(
    listing: Path, name: str, algorithm: str, mtime_ns: int
) -> str | None:
    """
    Return the digest that 'listing' holds for 'name', in 'md5sum' format.

    A listing older than the checkpoint, modified at 'mtime_ns', describes a
    file that has since been replaced and is ignored.
    """
    try:
        if listing.stat().st_mtime_ns < mtime_ns:
            return None
        text = listing.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    for line in text.splitlines():
        digest, _, listed = line.partition(" ")
        # md5sum separates with two spaces, or a space and * in binary mode
        if listed.lstrip(" *") == name and _DIGEST_PATTERNS[algorithm].fullmatch(
            digest
        ):
            return digest
    return None


def _download_sha256(checkpoint: Path) -> str | None:
    """Return the sha256 a Hugging Face download carries in its blob name."""
    # the hub cache links each file in a snapshot to blobs/<sha256>, which
    # is the digest the model repository lists, so nothing needs hashing
    if not checkpoint.is_symlink():
        return None
    blob = Path(os.path.realpath(checkpoint)).name
    return blob if _DIGEST_PATTERNS["sha256"].fullmatch(blob) else None


def _file_digest(path: Path, algorithm: str) -> str | None:
    return _file_digests(path, [algorithm]).get(algorithm)


def _file_digests(path: Path, algorithms: Sequence[str]) -> dict[str, str]:
    """Return the digests of a file from one pass, empty when unreadable."""
    if not algorithms:
        return {}
    hashers = {algorithm: hashlib.new(algorithm) for algorithm in algorithms}
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(_HASH_CHUNK_BYTES):
                for hasher in hashers.values():
                    hasher.update(chunk)
    except OSError:
        return {}
    return {algorithm: hasher.hexdigest() for algorithm, hasher in hashers.items()}


def _gpu_rows(device: object) -> list[_Row]:
    if not str(device).startswith("cuda"):
        return []
    name = _attempt(lambda: torch.cuda.get_device_name(str(device)))
    # zero with a batch server, the model then lives in another process
    peak = _attempt(lambda: torch.cuda.max_memory_allocated(str(device)) / 1e6)
    rows = [("gpu", _text(name)), ("gpu driver", _gpu_driver_version())]
    if isinstance(peak, float) and peak > 0.0:
        rows.append(("peak device memory / MB", f"{peak:.1f}"))
    return rows


@functools.cache
def _gpu_driver_version() -> str:
    """Return the NVIDIA driver version, one per distinct version on the node."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return _UNAVAILABLE
    versions = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    return ", ".join(sorted(versions)) or _UNAVAILABLE


def _settings_rows(predict_unit: object) -> list[_Row]:
    inference_settings = _getattr(predict_unit, "inference_settings")
    if isinstance(inference_settings, type) or not dataclasses.is_dataclass(
        inference_settings
    ):
        return [("inference_settings", _UNAVAILABLE)]
    return [
        (field.name, _text(_getattr(inference_settings, field.name)))
        for field in dataclasses.fields(inference_settings)
    ]


def _dyn_rows(dyn: object) -> list[_Row]:
    rows: list[_Row] = []
    # ASE names the optimizer or integrator itself in here
    description = _attempt(lambda: getattr(dyn, "todict")())
    if isinstance(description, dict):
        rows.extend((str(key), _text(value)) for key, value in description.items())
    else:
        rows.append(("class", type(dyn).__name__))
    # a resumed run only counts the steps taken by this process
    rows.append(("nsteps", _text(_getattr(dyn, "nsteps"))))
    # only optimizers have a force criterion to be converged against
    if getattr(dyn, "fmax", None) is not None:
        converged = _attempt(lambda: getattr(dyn, "converged")())
        rows.append(("converged", _text(converged)))
    # the files that hold the full history. ASE wraps the logfile in a Log
    # object that keeps the given value in a private attribute
    rows.append(("trajectory", _file_text(_getattr(dyn, "trajectory"))))
    rows.append(("logfile", _file_text(_getattr(_getattr(dyn, "logfile"), "_logfile"))))
    return rows


def _file_text(value: object) -> str:
    """Return the name of a file given as a path, an open file, or a writer."""
    if value is _MISSING:
        return _UNAVAILABLE
    if value is None:
        return "none"
    if value == "-":
        # ASE writes to standard output for this name
        return "stdout"
    if isinstance(value, (str, os.PathLike)):
        return os.fspath(value)
    # ASE trajectory writers hold filename, open files hold name
    name = _getattr(value, "filename")
    if name is _MISSING:
        name = _getattr(value, "name")
    return _text(name)


def _atoms_rows(atoms: Atoms, task_name: str) -> list[_Row]:
    rows = [
        ("formula", atoms.get_chemical_formula()),
        ("natoms", str(len(atoms))),
        ("pbc", " ".join("T" if periodic else "F" for periodic in atoms.pbc)),
    ]
    for axis, vector in zip("abc", np.asarray(atoms.cell), strict=True):
        rows.append((f"cell {axis} / Ang", _vector(vector)))
    if atoms.cell.rank == 3:
        rows.append(("cell parameters / Ang, deg", _vector(atoms.cell.cellpar())))
        rows.append(("cell volume / Ang**3", _number(atoms.cell.volume)))
    gaps = _largest_gaps(atoms)
    if gaps is not None:
        rows.append(("max gap along a b c / Ang", _vector(gaps)))
        rows.extend(_vacuum_rows(atoms, gaps))
    rows.append(("constraints", _constraints_text(atoms)))
    # fairchem fills in charge and spin for every task, only omol uses them
    if task_name == UMATask.OMOL.value:
        rows.append(("charge", _text(atoms.info.get("charge", _MISSING))))
        rows.append(("spin", _text(atoms.info.get("spin", _MISSING))))
    return rows


def _largest_gaps(atoms: Atoms) -> NDArray[np.float64] | None:
    """
    Return the largest empty gap along each cell axis in Angstrom.

    The gap is the widest atom free slice parallel to the plane of the other
    two cell vectors, counted across the cell boundary. For a slab this is
    the vacuum. Return None when the cell does not span three dimensions.
    """
    if atoms.cell.rank < 3 or len(atoms) == 0:
        return None
    scaled = np.sort(atoms.get_scaled_positions(wrap=True), axis=0)
    inside = np.diff(scaled, axis=0).max(axis=0, initial=0.0)
    across_boundary = 1.0 - (scaled[-1] - scaled[0])
    # rows of the reciprocal cell have length one over the plane spacing
    heights = 1.0 / np.linalg.norm(np.asarray(atoms.cell.reciprocal()), axis=1)
    gaps: NDArray[np.float64] = np.maximum(inside, across_boundary) * heights
    return gaps


def _vacuum_rows(atoms: Atoms, gaps: NDArray[np.float64]) -> list[_Row]:
    """
    Return the empty space below and above the atoms along the open axis.

    The open axis is the one axis with a gap of at least '_VACUUM_MIN_ANG',
    a narrower gap is a layer spacing. A slab has one such axis, a molecule
    in a box has three and no open axis, and both give no row then.
    """
    open_axes = np.flatnonzero(gaps >= _VACUUM_MIN_ANG)
    if len(open_axes) != 1:
        return []
    axis = int(open_axes[0])
    scaled = atoms.get_scaled_positions(wrap=True)[:, axis]
    height = 1.0 / np.linalg.norm(np.asarray(atoms.cell.reciprocal())[axis])
    below = float(scaled.min() * height)
    above = float((1.0 - scaled.max()) * height)
    # atoms across the cell boundary put the vacuum inside the cell instead,
    # atoms outside a cell that is open along the axis leave nothing to split
    if min(below, above) < 0.0 or abs(below + above - gaps[axis]) > 1e-6:
        return []
    name = "abc"[axis]
    return [(f"vacuum below, above along {name} / Ang", _vector([below, above]))]


def _constraints_text(atoms: Atoms) -> str:
    descriptions = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            # ASE indices, the row order of the structure block, which also
            # marks fixed atoms in its move_mask column
            indices = _index_ranges(constraint.index)
            descriptions.append(
                f"FixAtoms on {len(constraint.index)} atoms (indices {indices})"
            )
        else:
            descriptions.append(_constraint_text(constraint))
    return ", ".join(descriptions) if descriptions else "none"


def _constraint_text(constraint: object) -> str:
    """Return an ASE constraint as 'Name(arg=value, ...)' from its 'todict()'."""
    # str(constraint) is the object address for constraints without a repr
    description = _attempt(lambda: getattr(constraint, "todict")())
    if not isinstance(description, dict) or "name" not in description:
        return type(constraint).__name__
    kwargs = description.get("kwargs")
    if not isinstance(kwargs, dict):
        return str(description["name"])
    arguments = ", ".join(f"{key}={_text(value)}" for key, value in kwargs.items())
    return f"{description['name']}({arguments})"


def _index_ranges(indices: Iterable[int]) -> str:
    """Return sorted indices with runs collapsed, '0-7, 12, 14-15'."""
    runs: list[list[int]] = []
    for index in sorted(int(i) for i in indices):
        if runs and index == runs[-1][1] + 1:
            runs[-1][1] = index
        else:
            runs.append([index, index])
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in runs)


def _results_rows(atoms: Atoms, recalculated: bool) -> list[_Row]:
    """Return the scalar results, the forces are in the structure block."""
    results = atoms.calc.results
    source = "calculated for this record" if recalculated else "last calculation"
    rows = [("source", source)]
    for name in ("energy", "free_energy"):
        if name in results:
            rows.append((f"{name} / eV", _number(results[name])))
    fmax = _fmax(atoms)
    if fmax is not None:
        rows.append(("fmax / (eV/Ang)", _number(fmax)))
    stress = np.asarray(results.get("stress", ()), dtype=np.float64)
    # fairchem returns a stress for any cell, it only means something when
    # the cell is periodic. The structure block keeps it either way
    if stress.shape == (6,) and bool(atoms.pbc.all()):
        rows.append(("stress xx yy zz yz xz xy / (eV/Ang**3)", _vector(stress)))
    return rows


def _fmax(atoms: Atoms) -> float | None:
    """Return the largest force as ASE optimizers measure it, or None."""
    # read from the results rather than through atoms.get_forces(), which
    # would run the calculator again when the last calculation held no forces
    stored = atoms.calc.results.get("forces")
    if stored is None:
        return None
    forces = np.array(stored, dtype=np.float64)
    if forces.shape != (len(atoms), 3):
        return None
    # constraints applied, as in the fmax column of an ASE optimizer
    for constraint in atoms.constraints:
        if hasattr(constraint, "adjust_forces"):
            constraint.adjust_forces(atoms, forces)
    return float(np.linalg.norm(forces, axis=1).max())


def _xyz_lines(atoms: Atoms) -> list[str]:
    """
    Return the structure block, one extended xyz frame written by ASE.

    The frame holds cell, pbc, positions, the forces before constraints,
    fixed atoms as 'move_mask', and energy, stress, and 'atoms.info'.
    """
    calc = atoms.calc
    frame = atoms.copy()
    frame.info = _storable_info(frame.info)
    # ASE refuses to write results over entries of the same name
    for name in calc.results:
        frame.info.pop(name, None)
        frame.arrays.pop(name, None)
    frame.calc = calc
    buffer = io.StringIO()
    try:
        write(buffer, frame, format="extxyz")
    except (TypeError, ValueError):
        # atoms.info can hold entries the format cannot store, so the frame
        # is written without them. write_info=False is not an option, ASE
        # keeps the calculator results in the same dict and would drop them
        frame.info = {}
        buffer = io.StringIO()
        write(buffer, frame, format="extxyz")
    return [XYZ_TITLE, *buffer.getvalue().splitlines()]


def _storable_info(info: dict[str, object]) -> dict[str, object]:
    """
    Return the entries of 'atoms.info' that extended xyz can hold.

    ASE drops the others itself with a warning per structure. Structures from
    'ase.build' carry such an entry, 'adsorbate_info', so they are dropped
    here first. A string with a line break would break the frame, ASE does
    not escape it, so it is dropped too.
    """
    kept = {}
    for key, value in info.items():
        if isinstance(value, str) and "\n" in value:
            continue
        if not isinstance(value, (np.ndarray, np.generic)):
            try:
                json.dumps(value)
            except TypeError:
                continue
        kept[key] = value
    return kept


def _reference_rows(
    model_id: str, checkpoint: Path | None, task_name: str
) -> list[_Row]:
    rows = [("software", _SOFTWARE_REFERENCE)]
    # the model names itself from fairchem 1.2 on, older checkpoints and a
    # batch server leave only the file name
    names = [model_id, checkpoint.name if checkpoint is not None else ""]
    if any(name.lower().startswith("uma") for name in names):
        rows.append(("model", _MODEL_REFERENCE))
    dataset = _DATASET_REFERENCES.get(str(task_name))
    if dataset is not None:
        rows.append(("training data", dataset))
    return rows


def _bibtex_lines(references: Sequence[_Row]) -> list[str]:
    lines: list[str] = []
    for _, key in references:
        lines += [_BIBTEX[key], ""]
    return lines


@functools.cache
def _dependency_rows() -> tuple[_Row, ...]:
    """
    Return the packages that decide the result, with their locations.

    Versions and paths come from ASE's 'format_dependency', as the 'ase info'
    command prints them.
    """
    rows = [("python", platform.python_version())]
    for modname, distribution_name in _VERSION_MODULES:
        # ASE only catches ImportError, a broken package must not stop the record
        dependency = _attempt(lambda: format_dependency(modname))
        if not isinstance(dependency, tuple):
            rows.append((modname, _UNAVAILABLE))
            continue
        name, info = dependency
        version = name.removeprefix(f"{modname}-")
        # a module that is not installed comes back as its bare name
        text = info if version == modname else f"{version}, {info}"
        rows.append((modname, _with_commit(text, distribution_name)))
    rows.append(("cuda", _text(torch.version.cuda)))
    rows.append(("cudnn", _cudnn_text(_attempt(torch.backends.cudnn.version))))
    return tuple(rows)


def _cudnn_text(version: object) -> str:
    """Return the cudnn version torch reports as an integer in dotted form."""
    if not isinstance(version, int):
        return _text(version)
    # cudnn 9 encodes major * 10000 + minor * 100 + patch, cudnn 8 and older
    # major * 1000 + minor * 100 + patch
    major, rest = divmod(version, 10000 if version >= 90000 else 1000)
    minor, patch = divmod(rest, 100)
    return f"{major}.{minor}.{patch}"


@functools.cache
def _environment_rows() -> tuple[_Row, ...]:
    prefix = Path(sys.prefix)
    name = ""
    if os.environ.get("CONDA_PREFIX") == sys.prefix:
        name = os.environ.get("CONDA_DEFAULT_ENV", "")
    kind, version = _environment_tool(prefix)
    rows = [
        ("type", kind),
        ("version", version or "N/A"),
        ("name", name or prefix.name),
        ("path", str(prefix)),
    ]
    container = os.environ.get("APPTAINER_CONTAINER") or os.environ.get(
        "SINGULARITY_CONTAINER"
    )
    if container:
        rows.append(("container", container))
    return tuple(rows)


def _environment_tool(prefix: Path) -> tuple[str, str | None]:
    """Return the tool that made the environment and its version, if stored."""
    pixi = prefix / "conda-meta" / "pixi"
    if pixi.is_file():
        # pixi notes its own version in a json file beside the conda metadata
        try:
            version = json.loads(pixi.read_text(encoding="utf-8")).get("pixi_version")
        except (OSError, ValueError, AttributeError):
            version = None
        return "pixi", version if isinstance(version, str) else None
    history = prefix / "conda-meta" / "history"
    if history.is_file():
        # conda notes its version at each change, the last one is current
        versions = [
            line.removeprefix("# conda version:").strip()
            for line in _lines(history)
            if line.startswith("# conda version:")
        ]
        return "conda", versions[-1] if versions else None
    if (prefix / "conda-meta").is_dir():
        return "conda", None
    config = _lines(prefix / "pyvenv.cfg")
    if not config:
        return "system python", None
    # uv and virtualenv each note their version in pyvenv.cfg, venv does not
    for line in config:
        key, _, value = line.partition("=")
        if key.strip() in ("uv", "virtualenv"):
            return key.strip(), value.strip()
    return "venv", None


def _lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []


def _with_commit(text: str, distribution_name: str) -> str:
    """Return 'text', with the git commit pip installed the package from."""
    commit = _source_commit(distribution_name)
    return text if commit is None else f"{text}, git {commit}"


def _source_commit(distribution_name: str) -> str | None:
    try:
        text = metadata.distribution(distribution_name).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if text is None:
        return None
    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        return None
    vcs_info = record.get("vcs_info") if isinstance(record, dict) else None
    commit = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    return commit if isinstance(commit, str) else None


@functools.cache
def _package_rows() -> tuple[_Row, ...]:
    packages = set()
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            # the version is None for broken metadata, str keeps the sort
            packages.add((str(name), str(distribution.version)))
    ordered = sorted(packages, key=lambda package: (package[0].lower(), package[1]))
    return tuple(ordered)


def _default_filename(atoms: Atoms) -> Path:
    # microseconds keep several records from one script apart
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    configured = os.environ.get(_FILENAME_VARIABLE)
    if not configured:
        return Path(f"{atoms.get_chemical_formula()}-{stamp}{RECORD_SUFFIX}")
    path = Path(configured)
    if not path.exists():
        return path
    # a job script sets the variable once, so a script that records several
    # structures must not replace the first. The stamp goes before the suffix,
    # job.uma.txt becomes job-<stamp>.uma.txt
    suffix = RECORD_SUFFIX if path.name.endswith(RECORD_SUFFIX) else path.suffix
    stem = path.name[: len(path.name) - len(suffix)]
    return path.with_name(f"{stem}-{stamp}{suffix}")


def _write_atomically(path: Path, text: str) -> None:
    """Write 'text' to 'path' so that a failure never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # a unique name from mkstemp, a pid is shared between nodes of a cluster
    # writing the same file on a shared file system
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        # mkstemp opens the file for its owner only, the record and the
        # checksum listings follow the umask like any other file so a group
        # can read them
        os.chmod(temporary, 0o666 & ~_umask())
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _umask() -> int:
    # there is no read without a set, the reset follows at once
    mask = os.umask(0)
    os.umask(mask)
    return mask
