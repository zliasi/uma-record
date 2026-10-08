# Examples

Two scripts, each running a Cu(111) slab, the slab with H, and H2 with
`uma-s-1p2p1` and the `oc20` task. One record per system goes to
`output/cpu/` or `output/gpu/` and is read back through ASE.

```
single-point/   one energy and force evaluation per system
relax/          LBFGS to fmax 0.05, the record holds the optimizer too
site/           the Slurm submit script of the author's cluster
```

```
python relax/run.py                                  # CPU
python relax/run.py --device cuda                    # GPU
python relax/run.py --checkpoint /path/to/uma-s-1p2p1.pt
```

Without `--checkpoint` the model is downloaded from Hugging Face on first
use, which needs a login with access to the UMA models.

The records in `output/` come from these runs. `relax/output/cpu/slab-h.uma.txt`
shows constraints, a slab vacuum, and a `Dynamics` section, `h2.uma.txt` a
molecule in a box.

```
ase gui relax/output/cpu/slab-h.uma.txt
```

`site/submit.sh` submits an example as one CPU job and one GPU job with
Slurm. The partitions, conda environment, and checkpoint path at its top
belong to the author's cluster, change them for yours.

```
site/submit.sh relax
site/submit.sh single-point --device gpu
```
