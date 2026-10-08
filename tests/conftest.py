"""Register the ASE format for a checkout that is not installed."""

from __future__ import annotations

from ase.io.formats import define_io_format, ioformats

from uma_record_ase import ASE_IO_FORMAT


def pytest_configure(config: object) -> None:
    # the entry point does this on installation, a plain checkout has none
    if "uma-record" in ioformats:
        return
    define_io_format(
        "uma-record",
        ASE_IO_FORMAT.desc,
        ASE_IO_FORMAT.code,
        module=ASE_IO_FORMAT.module,
        magic=ASE_IO_FORMAT.magic,
        external=True,
    )
