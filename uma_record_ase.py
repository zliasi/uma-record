"""
The ASE file format for uma-record files.

Kept apart from 'uma_record'. ASE loads this module through the 'ase.ioformats'
entry point while 'ase.io' is still being imported, so it must not import
'ase.io' or fairchem at module level. That also keeps 'ase gui' free of the
torch import.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import IO, TYPE_CHECKING, Final

from ase.utils.plugins import ExternalIOFormat

if TYPE_CHECKING:
    from ase import Atoms

# the first line of every record, ASE recognises the format by it
MAGIC: Final = b"UMA calculation record"
# the extension a record gets by default, program tag plus plain text, so
# that it is told apart from other text records beside it
RECORD_SUFFIX: Final = ".uma.txt"
# the title line above the extended xyz frame at the end of a record
XYZ_TITLE: Final = "Structure, extended xyz"

ASE_IO_FORMAT: Final = ExternalIOFormat(
    desc="uma-record of a UMA calculation",
    code="1F",
    module="uma_record_ase",
    magic=MAGIC,
)


def read_fields(filename: str | os.PathLike[str]) -> dict[str, dict[str, str]]:
    """
    Return the 'key: value' rows of a record by section.

    The result maps a section title, 'Run' or 'Results' say, to its rows.
    The first colon of a row separates key from value, the bibtex entries and
    the structure block are left out. A key repeated in one section keeps its
    first value. Raise 'ValueError' when the file is not a record.
    """
    text = Path(filename).read_text(encoding="utf-8")
    if not text.startswith(MAGIC.decode()):
        raise ValueError(f"{filename} is not a UMA calculation record")
    sections: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for line in text.splitlines()[1:]:
        if line == XYZ_TITLE:
            break
        if not line:
            continue
        if line.startswith("  "):
            # a bibtex field line is indented too, current is None there
            if current is None:
                continue
            key, found, value = line.strip().partition(":")
            if found:
                current.setdefault(key, value.strip())
        elif line.startswith("@") or line == "}":
            current = None
        else:
            current = sections.setdefault(line, {})
    return sections


def read_uma_record(fileobj: IO[str]) -> Atoms:
    """
    Return the structure of a record, with energy, forces, and stress.

    The results sit on a single point calculator, as for any structure that
    ASE reads from an extended xyz file. Raise 'ValueError' when the file
    holds no structure block.
    """
    # imported here, see the module docstring
    from ase import Atoms
    from ase.io import read

    text = fileobj.read()
    # the block is the last thing in a record, so the last title line counts
    _, found, xyz = text.rpartition(f"\n{XYZ_TITLE}\n")
    if not found:
        name = getattr(fileobj, "name", "the record")
        raise ValueError(f"{name} holds no structure block")
    atoms = read(io.StringIO(xyz), format="extxyz")
    assert isinstance(atoms, Atoms), "one frame is written, one frame is read"
    return atoms
