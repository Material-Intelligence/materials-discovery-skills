---
name: unified-workflow
description: Use when running several stages of the discovery workflow from a single YAML configuration - generation through phonons - or when resuming a run after DFT or training jobs have finished elsewhere.
---

# Unified workflow

One YAML file describes a run: what to call it, where to put it, which stages to execute, and
a section per stage. The stages run in a fixed order and each writes into its own numbered
directory:

| # | Stage | Writes into | Needs |
|---|---|---|---|
| 1 | `generation` | `01_generation/structures/` | MatterGen, `$MATTERGEN_MODEL_PATH` |
| 2 | `clustering` | `02_clustering/selected/` | `mlip` + `clustering` extras, `$DPA3_MODEL_PATH` |
| 3 | `dft` | `03_dft/calculations/` | pymatgen; `$PMG_VASP_PSP_DIR` for real POTCARs |
| 4 | `finetune` | `04_finetune/` | `mlip` extra, `$DPA3_MODEL_PATH` |
| 5 | `competing` | `05_competing/` | `$MP_API_KEY`, network |
| 6 | `screening` | `06_screening/` | an ASE calculator |
| 7 | `phonons` | `07_phonons/` | an ASE calculator |

Each stage calls the same public functions you would call directly, so the pipeline is
wiring, not a second implementation of anything.

**How a stage gets its input.** A path written in the configuration always wins. When it is
absent, the stage takes the output of the stage that produced it earlier in the same run.
When neither exists the run stops with a message naming the configuration key and the stage
that would have supplied it.

## Prerequisites

The package itself (`pip install -e .` from the repository root) plus whatever the stages you
list need - see the table above and the corresponding skill. A run that lists only
`competing` and `screening` needs no MatterGen and no `maml`. Every optional backend is
imported inside the function that uses it, so `--help`, `--write-config`, `--dry-run` and
`--validate` all work in an environment with none of them installed.

## Usage

```bash
matdisc pipeline --help
```

Write a configuration holding every key at its default value:

```bash
matdisc pipeline --write-config bacdp.yaml --chemsys Ba-Cd-P
```

Check what a configuration would do before it does it:

```bash
matdisc pipeline -c bacdp.yaml --dry-run
matdisc pipeline -c bacdp.yaml --stages competing screening --dry-run
matdisc pipeline -c bacdp.yaml --resume-from screening --dry-run
```

Run one, and report afterwards what each stage directory holds:

```bash
matdisc pipeline -c dft_only.yaml
matdisc pipeline --validate runs/Ba-Cd-P
```

`matdisc` and `python -m matdisc.cli` are the same entry point. From Python:

```python
from matdisc.pipeline import run_from_config

result = run_from_config("bacdp.yaml", stages=["competing", "screening"])
```

## Configuration

`--write-config` emits every key with its default value. The `dft_only.yaml` used above is a
short one that runs with no backend, no credentials and no network, preparing VASP inputs for
structures you already have:

```yaml
name: Ba-Cd-P
work_dir: ./runs/Ba-Cd-P
stages: [dft]

dft:
  structure_dir: ./structs
  kind: relax
  write_slurm: true
  slurm:
    ntasks: 96
    walltime_minutes: 600
```

A longer one, for the stages that need credentials and a calculator:

```yaml
name: Ba-Cd-P
work_dir: ./runs/Ba-Cd-P
stages: [competing, screening, phonons]

competing:
  chemsys: Ba-Cd-P
  only_icsd: true
  download_structures: true

screening:
  structure_dir: ./candidates
  model_path: $DPA3_MODEL_PATH
  tolerance: 1.0e-06
  competing_energy_source: relax

phonons:
  threshold: -0.1
  max_structures: 5
```

Credentials are never part of a configuration file: `MP_API_KEY` is read from the environment
at call time. Path-valued entries (`work_dir`, `model_path`, `pretrained_model`, the `*_dir`
inputs and `csv_file`) expand `$VAR` and `~` when the file is loaded; no other string is
expanded, so a `$VASP_CMD` meant for a generated job script survives literally.

## Outputs

Every stage writes its own numbered directory under `work_dir`, and the run writes
`pipeline_result.json` there: the status of each stage, the counts it reported, the paths it
produced, and the full configuration that produced them. A run reports `completed`,
`stopped` (it ended deliberately, with work left to do elsewhere) or `failed`.

## Pitfalls

- **An unknown key in a section is an error**, not a warning. A silently ignored typo would
  change the science without saying so.
- **The `dft` stage stops the run** when it has only written inputs and `finetune` is still
  ahead. Run the calculations, then resume with `--resume-from finetune` and
  `finetune.dft_dir` pointing at them.
- **`finetune.run` defaults to `false`**, so a full run prepares the training job and does not
  produce a checkpoint. The screening stage then relaxes with the checkpoint named in its own
  section. Set `finetune.run: true` only if training in-process is really what you want.
- **`screening.competing_energy_source` defaults to `relax`**, which relaxes the downloaded
  competing structures with the same calculator as the candidates so that both sides of the
  hull are on one energy scale. Setting it to `table` reads Materials Project energies
  instead, which is only correct when the candidate energies come from the same kind of
  calculation; the run warns when you do.
- **Listing fewer stages is how you resume**, but the first listed stage then needs its input
  supplied through its own `*_dir` or `csv_file` key.
- `--dry-run` and `--validate` cost nothing. Use them before a run that will occupy a cluster.
