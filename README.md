# uma-record

Human readable record files for UMA calculations.

One call writes a text file with the model, settings, environment, final
structure, energy, and forces of a calculation run with the fairchem
`FAIRChemCalculator` in ASE, comparable to the text output of a DFT code.
The file is meant for data management and for the archive published with a
paper. The structure within the record file is ASE compatible.

Requires Python 3.11 or newer and fairchem-core.

## Installation

Install into the environment that holds fairchem. The dependencies carry
lower bounds only, so the environment stays as it is.

```
pip install git+https://github.com/zliasi/uma-record
```

Or in the `pip` section of a conda `environment.yml`, pinned to a commit:

```
  - pip:
    - git+https://github.com/zliasi/uma-record@<commit>
```

Tested with Python 3.13, fairchem-core 2.22.0, ASE 3.29.0, torch 2.13.0.

## Usage

Set up the calculator as usual and call `write_record` after the calculation.

```python
from ase.optimize import LBFGS
from fairchem.core import FAIRChemCalculator
from uma_record import write_record

atoms.calc = FAIRChemCalculator.from_model_checkpoint(
    "uma-s-1p2p1", task_name="oc20", device="cuda"
)

opt = LBFGS(atoms)
opt.run(fmax=0.05)

write_record(atoms, dyn=opt)
```

For a single point, `write_record(atoms)`. Call it once per structure, right
after that structure is done. Everything after `atoms` is a keyword.

```
atoms        the structure, with the FAIRChemCalculator attached
dyn          optional, the ASE optimizer or molecular dynamics object
filename     optional, where to write (default: UMA_RECORD_PATH, then
             <formula>-<timestamp>.uma.txt in the working directory)
checkpoint   optional, the checkpoint file, for a calculator that cannot
             report its own
packages     optional, True adds every installed package with its version
```

A file named by `filename` is replaced. A job script can set the path once
instead:

```
export UMA_RECORD_PATH="$output_dir/$job_name.uma.txt"
```

A file named by the variable is kept. When it exists, the next record gets
the time stamp before its suffix, `job-20261008-101500-123456.uma.txt`, so a
script that records several structures keeps them all. The tasks of a job
array need the task id in the path, two tasks that see the same missing file
both write it.

## Reading a record

The record is an ASE file format, recognised by its first line whatever the
file is called.

```python
from ase.io import read

atoms = read("slab.uma.txt")    # structure with energy, forces, stress
```

```
ase gui slab.uma.txt
```

`uma_record.read_record` does the same without ASE's format detection. The
format is registered by the install, so an editable install made before the
format existed needs `pip install -e .` again.

The `key: value` rows come back by section through `read_fields`:

```python
from uma_record import read_fields

fields = read_fields("slab.uma.txt")
fields["Results"]["energy / eV"]        # '-39.492197'
fields["Calculator"]["checkpoint md5"]
```

## The record

```
UMA calculation record (uma-record v0.1.0)

Run
  command:                   run.py --device cpu
  script:                    /home/user/examples/relax/run.py
  user:                      user
  node:                      node024.cluster
  job id:                    65895142
  process started:           2026-10-07T17:28:59+02:00
  written:                   2026-10-07T17:29:51+02:00
  ...

Calculator
  calculator:               FAIRChemCalculator
  task_name:                oc20
  model_id:                 UMA-S-1.2.1
  checkpoint:               /path/to/uma-s-1p2p1.pt
  checkpoint md5:           ...
  checkpoint sha256:        ...
  device:                   cpu
  ...

Results
  source:                                  last calculation
  energy / eV:                            -39.492197
  fmax / (eV/Ang):                         0.045059
  ...
```

Full records are in [`examples/`](examples/). The sections are `Run`,
`Calculator`, `Inference settings`, `Dynamics`, `Atoms`, `Results`,
`References` with the bibtex entries of fairchem, the model, and its
training data, `Environment`, `Dependencies`, `Installed packages` when
asked for, and the structure as one extended xyz frame. Values that cannot
be read are `unavailable`. The layout is tied to the version in the first
line, a layout change raises the minor version.

The record identifies the run. It carries the user name, the node, absolute
paths, and the full command line. A token passed on the command line ends up
in the record, and in any archive the record is published in.

## Details

- Energy and forces are the cached results of the last calculation when they
  match the structure, else they are calculated again and `source` says so.
  fairchem counts a changed `atoms.info` as a changed structure, so an entry
  added after the run makes the record calculate again.
- The structure block holds the forces before constraints, `fmax` has the
  constraints applied, as in an ASE optimizer log.
- The stress row is written for a periodic cell only. fairchem returns one
  for any cell, the structure block keeps it either way.
- `Dependencies` lists the packages that decide the result. `packages=True`
  adds `Installed packages`, the Python distributions the interpreter sees,
  not the conda packages beneath them. It is a long section, so it is off
  for a script that records many structures.
- `process started` comes from `/proc` on Linux. Elsewhere the row is
  `module imported`, the time `uma_record` was imported. `process elapsed`
  counts from that row.
- `Dynamics` names the trajectory and logfile the ASE object was given, the
  files that hold the full history.
- `gpu driver` comes from `nvidia-smi`, torch does not expose it.
- The checkpoint MD5 matches the fairchem model card, the sha256 matches
  Hugging Face and git LFS. Each is read from a `<checkpoint>.md5` or
  `.sha256` beside the checkpoint when that file is newer than the checkpoint.
  A Hugging Face download carries its sha256 in the cache blob name. Else the
  checkpoint is hashed in one pass and both files are written beside it, also
  inside the Hugging Face cache. A directory that cannot be written is hashed
  on every run.
- A `FAIRChemCalculator` inside a wrapper such as `SumCalculator` is found,
  the energy is then the one the wrapper returns.
- The file is written to a temporary name and renamed, so a failure never
  leaves a partial record.

## Limits

- A job that is killed before the call leaves no record.
- The checkpoint path, `seed`, and `overrides` come from private fairchem
  attributes. If a release renames them they are `unavailable`, and
  `checkpoint` can supply the path.
- With the fairchem `InferenceBatcher` the model lives in a server process.
  Call `write_record` before the batcher is shut down. `model_id` and
  `direct_forces` are then `unavailable`.
- With `workers > 1`, the fairchem `ParallelMLIPPredictUnit`, the checkpoint,
  `model_id`, and `direct_forces` are `unavailable`. `checkpoint` supplies
  the path. The parallel unit also runs its rank 0 in the calling process
  and leaves the torch distributed group behind, so a plain predict unit
  made afterwards in the same process hangs on a collective. That is
  fairchem behaviour, `fairchem.core.common.distutils.cleanup_gp_ray()`
  clears it.
- ASE molecular dynamics objects report little about themselves, so
  thermostat settings are not in the record. The `logfile` row reads a
  private ASE attribute and is `unavailable` if a release renames it.
- A nudged elastic band needs one call per image.
- Any model loaded through `FAIRChemCalculator` is recorded, but the first
  line and the model reference are written for UMA.
- A checksum file is stale when the checkpoint is newer than it. A
  checkpoint restored with an older modification time, `cp -p` or `rsync -a`,
  keeps the old checksum. Delete the `.md5` and `.sha256` files then.

## Development

```
pip install -e ".[dev]"
pytest
black --check .
mypy uma_record.py uma_record_ase.py tests
```

`tests/test_uma_record.py` runs a real `FAIRChemCalculator` over a stand-in
predict unit, no checkpoint or GPU needed. `tests/test_real_model.py` loads a
checkpoint on the CPU and runs the scenarios the stand-in cannot reach, the
`InferenceBatcher`, `workers > 1`, the fast path fallback, omol defaults, and
the fairchem wrappers. It starts ray on the node and takes some minutes, and
the first run writes the `.md5` and `.sha256` files beside the checkpoint.
Both real model parts are skipped unless `UMA_RECORD_TEST_CHECKPOINT` names a
checkpoint:

```
UMA_RECORD_TEST_CHECKPOINT=/path/to/uma-s-1p2p1.pt pytest
```

## License

MIT, see LICENSE. The UMA checkpoints are released under the FAIR Chemistry
License, which asks for an acknowledgement in publications, and the datasets
have their own licenses. The record holds the citations.
