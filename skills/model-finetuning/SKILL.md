---
name: model-finetuning
description: Use when turning finished VASP relaxations into a DeePMD-kit training set and fine-tuning a pretrained DPA-3 checkpoint on it, so that later screening runs on a potential that has seen the chemistry in question.
---

# Model fine-tuning

A general-purpose machine-learning potential is accurate enough to rank candidates but not
always to separate the close ones. This skill closes that gap: the DFT relaxations that were
run on the selected structures become training labels, and a pretrained DPA-3 checkpoint is
fine-tuned on them.

The steps are: read every ionic step of every relaxation with `dpdata`, write one
`deepmd/npy` system per structure, split systems into training and validation sets, build the
DPA-3 training configuration as JSON, and either run `dp --pt train <config> --finetune
<checkpoint> --model-branch <head>` or write a job script that does.

## Prerequisites

- The package itself: `pip install -e .` from the repository root.
- `dpdata` for the conversion and DeePMD-kit for the training command: the `mlip` extra in
  `pyproject.toml`. DeePMD-kit is never imported here, only invoked as the `dp` console
  script, and its absence is reported with the name of the extra:
  `dpdata is required to convert DFT output into a training set but is not installed.`
- A pretrained checkpoint, passed as `--pretrained-model` or read from `$DPA3_MODEL_PATH`:

  ```bash
  export DPA3_MODEL_PATH=/path/to/pretrained/checkpoint.pt
  ```

- A GPU for any real training run. Preparing the dataset and the configuration is cheap and
  needs neither a GPU nor the `dp` command.

## Expected input layout

The DFT tree is the one `dft-calculation` writes and VASP then runs in: one directory per
calculation, with the OUTCAR beside the inputs.

```
<dft_dir>/                 # e.g. runs/dft/calculations
  <system>/
    INCAR POSCAR POTCAR OUTCAR ...
  <other system>/ ...
```

Each immediate subdirectory is one system. A single OUTCAR already carries every ionic step of
the relaxation, so that one file is the whole trajectory.

Trees that put each ionic step in its own directory are also supported, but have to be asked
for: `output_subdir` names a directory between `<dft_dir>` and the systems, `relax_subdir` one
between a system and its step directories, and both default to `""`.

```python
find_system_outcars("runs/dft_from_elsewhere", output_subdir="Output", relax_subdir="relax")
```

`find_system_outcars` is the directory walk on its own. It needs neither `dpdata` nor
DeePMD-kit, so a tree can be checked before any conversion is attempted:

```python
from matdisc.finetune.dataset import find_system_outcars

find_system_outcars("runs/dft/calculations")
# {'candidate_a': [PosixPath('runs/dft/calculations/candidate_a/OUTCAR')], ...}
```

A system with no usable OUTCAR maps to an empty list rather than disappearing, so a
calculation that has not finished is visible rather than silently missing.

## Usage

```bash
matdisc finetune --help
```

Prepare the dataset, the configuration and a job script, without training:

```bash
matdisc finetune -d runs/dft/calculations -o runs/finetune
```

Train here instead of submitting:

```bash
matdisc finetune -d runs/dft/calculations -o runs/finetune --run --numb-steps 100000
```

From Python, when you want to inspect the configuration before it is written:

```python
from matdisc.finetune.dataset import convert_dft_to_deepmd, split_train_val, get_type_map
from matdisc.finetune.train import build_dpa3_config, write_dpa3_config, finetune

_, stats = convert_dft_to_deepmd("runs/dft/calculations", "runs/finetune/deepmd_data")
train, val = split_train_val("runs/finetune/deepmd_data",
                             "runs/finetune/train_data", "runs/finetune/val_data")
config = build_dpa3_config(get_type_map("runs/finetune/deepmd_data"), train, val)
result = finetune(write_dpa3_config(config, "runs/finetune/finetune_config.json"))
```

`check_training_progress(work_dir)` reads `lcurve.out` and reports the step count and the
last energy and force RMSE; `find_latest_checkpoint(work_dir)` returns the newest
`model.ckpt-*.pt`.

## Inputs and outputs

Output under `<outdir>`:

- `deepmd_data/<system>/` - the converted labelled frames.
- `train_data/`, `val_data/` - the split, by system.
- `finetune_config.json` - the DPA-3 training configuration.
- `run_finetune.sh` - the job script, written when `--run` is not passed.
- `model.ckpt-*.pt`, `lcurve.out` - written by a training run.

## Pitfalls

- **Without `--run`, nothing is trained and no checkpoint exists.** The stage says so
  explicitly and names the script to submit. A later screening stage then uses the checkpoint
  named in its own configuration, not a fine-tuned one. Training belongs in a batch job, not
  inside a pipeline call.
- **`--exclude` matches the directory holding an OUTCAR, not the frames inside it.** With the
  flat layout the DFT stage writes there is one such directory per system, so `--exclude`
  drops whole systems. It is the nested, step-per-directory layout where `--exclude S0` drops
  the unrelaxed starting geometry; with a single OUTCAR per relaxation that first frame is
  inside the file and is kept.
- **The split is by system, not by frame**, and at least one system goes to validation
  whenever `val_ratio` is above zero. Frames from one relaxation are correlated, so splitting
  them across the two sets would flatter the validation error. A single converted system is
  therefore an error rather than an empty training set: convert more calculations, or pass
  `val_ratio=0` to train on the one you have with no validation set.
- **The type map is read from the converted data** unless you pass one. An element that
  appears in only a few systems still enters the map, which is usually right but worth
  checking against the chemistry you intend to screen.
- **The generated job script is deliberately incomplete** for a specific site: it activates a
  conda environment and runs the training command. Add your own account, partition and module
  lines.
- An OUTCAR that cannot be read is logged and skipped; the conversion statistics report
  `total_frames`, and the stage stops when that is zero rather than training on nothing. A
  system whose calculation has not finished shows up with no OUTCAR rather than vanishing.

## What comes next

The fine-tuned checkpoint is what `stability-screening` should relax with. In a pipeline run,
the screening stage picks it up automatically when the fine-tuning stage produced one.
