---
name: descriptor-clustering
description: Use when a set of candidate structures is too large to calculate in full and you need a representative subset, chosen by clustering the per-atom descriptors of a machine-learning interatomic potential.
---

# Descriptor clustering

Picks the structures worth an expensive calculation. Every atom of every candidate becomes
one row of a feature matrix - the per-atom descriptor a DeePMD-kit model produces
(`DeepPot.eval_descriptor`) - and DIRECT sampling (PCA, then BIRCH clustering, then a fixed
number of rows per cluster, as implemented in `maml.sampling.direct`) selects the rows that
cover that space. The structures those rows belong to are the output.

Coordinates, cells and atom types come straight from `pymatgen.core.Structure`, and the
element order is read from the loaded model, so any element the model knows about is
supported.

## Prerequisites

- The package itself: `pip install -e .` from the repository root.
- DeePMD-kit, for the descriptors: the `mlip` extra in `pyproject.toml`. It ships wheels per
  CUDA build, so pick the one that matches your machine.
- `maml` and scikit-learn, for DIRECT sampling: the `clustering` extra.
- A checkpoint, passed as `--model-path` or read from `$DPA3_MODEL_PATH`:

  ```bash
  export DPA3_MODEL_PATH=/path/to/checkpoint.pt
  ```

  Without one the command stops with
  `The clustering stage needs a machine-learning potential checkpoint.`
- `--head` names the branch of a multi-task checkpoint (default `Omat24`). A single-task
  checkpoint has no branches; load it through the Python API with `head=None`.

Each optional backend is imported inside the function that needs it, so importing this
package without them works and `--help` always answers.

## Usage

```bash
matdisc cluster --help
```

Select about ten representatives from a directory of candidates:

```bash
matdisc cluster -s runs/generation/structures -n 10 -o runs/clustering
```

`--threshold` fixes the BIRCH threshold and skips the tuning search; `--k-per-cluster` sets
how many rows are taken from each cluster.

From Python, where the intermediate objects are worth keeping:

```python
from matdisc.clustering.descriptors import load_descriptor_model, compute_descriptor_set
from matdisc.clustering.sampling import select_representative_structures, write_selected_structures

model = load_descriptor_model(head="Omat24")
descriptors = compute_descriptor_set(structures, model, names=names)
selections = select_representative_structures(descriptors, n=10)
write_selected_structures(structures, selections, "runs/clustering/selected")
```

`DescriptorSet.save` and `.load` keep a computed feature matrix on disk as `.npz`, which is
worth doing before experimenting with thresholds. `plot_pca_coverage` draws the selected rows
against all rows in the first two PCA components.

## Inputs and outputs

Input: a structure file or a directory of them (`*.vasp`, `*.cif`, `POSCAR*`, `CONTCAR*`,
`*.xyz`, `*.extxyz`).

Output under `<outdir>`:

- `selected/<name>.vasp` - one file per selected structure.
- `selected.csv` - `id`, `index`, `n_atoms`, `n_selected_atoms`.

`compute_descriptors(structures, calculator)` returns the raw
`(total_atoms, descriptor_dim)` array when you only want the features;
`compute_descriptor_set` returns the same array together with the atom-to-structure mapping,
which is what makes the selection invertible.

## Pitfalls

- **Sampling happens on atoms, not on structures.** A structure is selected as soon as one of
  its atoms is, so asking for `n` representatives yields at most `n` structures and usually
  fewer. The command logs both counts.
- **Threshold tuning costs a full DIRECT run per bisection step** (ten by default). For a
  large feature matrix, tune once, note the threshold, and pass `--threshold` afterwards.
- **An element outside the model's type map raises** a message naming the element and the
  size of the type map. Use a model whose type map covers your chemistry rather than editing
  a list by hand.
- **Descriptors are model-specific.** A selection made with one checkpoint is not transferable
  to another, and neither is a tuned threshold.
- Descriptor extraction needs the model object itself, not an ASE calculator. Use
  `load_descriptor_model` here; `matdisc.common.calculators.load_calculator` is for
  relaxation and phonons. An ASE calculator that wraps a model as `.dp` is accepted too.

## What comes next

`dft-calculation` turns the selected structures into VASP calculation directories.
