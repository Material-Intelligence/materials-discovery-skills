---
name: structure-generation
description: Use when you need candidate crystal structures for a chemical system from the MatterGen generative model, or when reading a finished MatterGen run back into pymatgen structures.
---

# Structure generation

Drives MatterGen through its `mattergen-generate` console script: this skill builds the
command line, runs it once per chemical system, and reads the structures the run wrote back
into `pymatgen.core.Structure` objects. MatterGen is never imported as a Python package, so
the only requirement is that its console script is on `PATH`.

Sampling can be conditioned on a chemical system alone or on a chemical system plus a target
energy above the hull, with a classifier-free guidance factor controlling how hard the model
is pushed towards that target.

## Prerequisites

- The package itself: `pip install -e .` from the repository root.
- MatterGen, installed from its own repository (<https://github.com/microsoft/mattergen>). It
  is not a declared dependency of this package and not an extra, because it is never imported
  and because its own pins (a CUDA build of torch) do not resolve from PyPI. The only thing
  this package needs from it is the `mattergen-generate` command on `PATH`. When it is missing
  the run stops with a message naming the executable and the repository.
- A MatterGen checkpoint directory, passed as `--model-path` or read from
  `$MATTERGEN_MODEL_PATH`:

  ```bash
  export MATTERGEN_MODEL_PATH=/path/to/mattergen/checkpoint
  ```

  Without it the run stops with `No MatterGen checkpoint given.`
- A GPU, in practice. Sampling on CPU is possible but slow enough not to be worth it.

## Usage

```bash
matdisc generate --help
```

Generate 24 candidates for one system:

```bash
matdisc generate --chemsys Ba-Cd-P -n 24 -o runs/generation
```

`--chemsys` repeats, so several systems can be sampled in one call; each gets its own
subdirectory. Other switches: `--batch-size` (MatterGen samples in batches, so
`ceil(n / batch_size)` batches are requested), `--energy-above-hull` (the property-guidance
target in eV/atom, 0.1 by default), `--guidance-factor`, and `--no-skip-existing` to resample
over an output directory that already holds a run. The CLI, `generate()` and
`generation.energy_above_hull` in a pipeline configuration all read the same defaults from
`matdisc.generation.mattergen`, so the entry point does not change the distribution sampled.

From Python:

```python
from matdisc.generation.mattergen import generate, generate_many, read_generated_structures

structures = generate("Ba-Cd-P", n=24, output_dir="runs/generation/mattergen/Ba-Cd-P")
```

`generate_many` takes a list of systems and records a per-system failure in its result
instead of stopping the batch. `read_generated_structures` reads an existing run directory
without sampling anything - useful when MatterGen was run elsewhere, for instance on a
cluster.

## Inputs and outputs

Input: one or more hyphen-separated chemical systems. Every token is validated as an element
symbol, so a typo is reported before MatterGen starts.
`matdisc.generation.mattergen.read_chemical_systems` reads a list from a text file (one per
line, `#` comments ignored) and `element_combinations` enumerates systems from a pool of
elements.

Output under `<outdir>`:

- `mattergen/<chemsys>/` - the raw MatterGen run: `generated_crystals_cif.zip` (one CIF per
  structure), `generated_crystals.extxyz` (all frames) and `generated_trajectories.zip`.
- `structures/<chemsys>_<i>.vasp` - one file per returned structure, which is what the next
  stage reads.

Structures are read from the CIF archive when it exists and from the extxyz file otherwise.

## Pitfalls

- **MatterGen writes no loose `*.cif` files.** A directory listing filtered on `.cif` reports
  zero on a perfectly successful run; read the archive or the extxyz file instead, which is
  what `read_generated_structures` does.
- **`n` is a request, not a promise.** Whole batches are sampled and the first `n` structures
  are returned; a short run is logged as a warning rather than an error.
- **An existing output directory is reused by default.** That makes a repeated call cheap,
  but it also means a changed `--energy-above-hull` has no effect until you pass
  `--no-skip-existing` or point `-o` somewhere new.
- **Generated structures are unrelaxed and unscreened.** They are candidates, nothing more.
  Every downstream stage - clustering, relaxation, hull, phonons - exists because most of
  them will not survive.
- A CIF member that cannot be parsed is logged and skipped, so a returned count below the
  file count in the archive is worth checking in the log.

## What comes next

`descriptor-clustering` reduces the generated set to the structures worth a DFT calculation.
For a small set you can skip it and go straight to `dft-calculation`.
