---
name: stability-screening
description: Use when deciding whether candidate structures are stable - relax them with a machine-learning potential, place their energies on a convex hull to get an energy above the hull, and check the survivors for imaginary phonon frequencies.
---

# Stability screening

Three steps, each usable on its own:

1. **Relax** candidates with an ASE calculator (`matdisc.screening.relax`).
2. **Convex hull** - place the relaxed energies on a `pymatgen` `PhaseDiagram` built from the
   competing phases and report an energy above the hull per candidate
   (`matdisc.screening.hull`).
3. **Phonons** - displace atoms in a supercell with `ase.phonons` and report the lowest
   frequency along the band path (`matdisc.screening.phonons`).

## Criteria

| Test | Criterion | Default |
|---|---|---|
| Thermodynamic | stable when `e_above_hull <= tolerance` | `tolerance = 1e-6` eV/atom (`--tolerance`) |
| Dynamical | stable when the lowest frequency is above `threshold` | `threshold = -0.1` THz (`--threshold`) |

The hull tolerance admits a phase sitting exactly on the hull while absorbing the rounding of
the hull solve; a strict `< 0` test would call an on-hull phase unstable. The phonon
threshold tolerates the small imaginary frequencies that finite displacements leave at the
zone centre.

**Both sides of the hull must be on one energy scale.** Candidate and competing energies have
to come from the same kind of calculation. The pipeline's default is to relax the downloaded
competing structures with the same calculator as the candidates. The alternative is to use
formation energies on both sides; `matdisc.screening.hull.formation_energy_per_atom` converts
a per-atom energy given elemental reference energies. The same applies to DFT candidate
energies from `matdisc.dft.outputs.collect_results`: the column names match the competing table
`matdisc competing` harvests, the energy scales do not.

**The hull is only as complete as the reference set.** `matdisc competing` keeps ICSD-backed
phases by default, which is a subset of the Materials Project's own reference set. A phase it
leaves out that belongs on the true hull is simply missing, so the hull sits lower and every
`e_above_hull` measured against it is a lower bound — candidates can read stable, or land below
the hull, that would not against the full set. Harvest with `--no-icsd` when the question is
thermodynamic stability rather than "stable among the compounds already known".

## Prerequisites

- The package itself: `pip install -e .` from the repository root. The hull needs nothing
  beyond the required dependencies - no credentials, no network, no calculator.
- Relaxation and phonons need an ASE calculator:
  - `--calculator dp` loads a DeePMD-kit checkpoint (`mlip` extra in `pyproject.toml`) from
    `--model-path` or `$DPA3_MODEL_PATH`, with `--head` naming the branch of a multi-task
    checkpoint. Without one the command stops with
    `The phonons stage needs a machine-learning potential checkpoint.`
  - `--calculator emt` is ASE's built-in effective-medium potential. It is parameterised for
    a handful of metals and is there for smoke tests, not for screening.

## Usage

```bash
matdisc hull --help
matdisc phonons --help
```

The hull runs offline with no inputs at all:

```bash
matdisc hull --demo
```

which prints three Ba-Cd-P candidates - one below the hull, one above it and one exactly on a
tie-line - and reports `2 of 3 candidate(s) at or below the hull`.

With your own tables:

```bash
matdisc hull --candidates candidates.csv --competing competing_phases.csv -o hull.csv
```

Both tables are read on the same per-atom energy column (`--energy-column`, default
`energy_per_atom`). The phonon check takes a structure file or a directory of them:

```bash
matdisc phonons -s runs/screening/stable -o runs/phonons --model-path "$DPA3_MODEL_PATH"
```

EMT needs no checkpoint, so it gives a quick end-to-end smoke test of the phonon machinery on
a simple metal - not a screening result:

```bash
matdisc phonons -s structures/Cu.vasp -o runs/phonon_smoke --calculator emt --supercell 2 2 2
```

Relaxation and the hull together are one command, which relaxes the candidates and the
competing structures with the same calculator and writes `hull.csv` plus `stable/`:

```bash
matdisc screen -s runs/candidates --competing-structures runs/competing/structures \
    -o runs/screening --model-path "$DPA3_MODEL_PATH"
```

It stops rather than report a hull that is missing a competing phase whose relaxation failed
(`--allow-incomplete-hull` overrides that), and it does not report a candidate as stable when
that candidate's own relaxation hit the step cap. The same steps from the library:

```python
from matdisc.common.calculators import load_calculator
from matdisc.screening.relax import relax_many
from matdisc.screening.hull import compute_e_above_hull

calculator = load_calculator("dp", model_path="/path/to/checkpoint")
_, candidates = relax_many(structures, calculator, output_dir="runs/screening/relaxed")
hull = compute_e_above_hull(candidates, competing, tolerance=1e-6)
```

`relax_many` returns a table carrying the `id`, `composition` and `energy_per_atom` columns
`compute_e_above_hull` reads, so no adapter is needed between the two.

## Inputs and outputs

Candidates table: `id`, `composition`, `energy_per_atom`. Competing table: the schema
`competing-phases` writes. `compute_e_above_hull` returns a copy of the candidates with four
columns added: `e_above_hull` (eV/atom), `hull_energy_per_atom`, `is_stable`, and
`decomposition` (the hull decomposition, or the reason a candidate was skipped).

The phonon command writes per structure: `band_structure.dat` (distance and frequencies in
THz), `dos.dat`, `band_path_info.txt`, `phonon_spectrum.png`, and - when relaxing first -
`relaxed.vasp` and `relaxation.log`. Alongside them: `phonons.csv`
(`id, success, min_frequency, is_stable, relaxation_converged, output_dir, error`) and a
`stable/` directory holding the structures that passed, as the relaxed geometries the spectra
were computed on.

`matdisc screen` writes `candidates.csv`, `competing_relaxed.csv`, `relaxed/<id>/relaxed.vasp`,
`hull.csv` (the candidate columns above plus `calculator`, `calculator_head`,
`n_competing_used` and `n_competing_dropped`) and `stable/` holding the relaxed cells of the
candidates at or below the hull.

## Pitfalls

- **Every element needs an elemental reference phase** in the competing table, or the phase
  diagram cannot be built and the command says so.
- **A candidate outside the competing elements** gets `NaN` and a reason in `decomposition`
  rather than a silently wrong number.
- **Phonons are expensive.** The supercell chosen from `--min-cell-length` (10 A by default)
  can hold a few hundred atoms, and every displacement is a force evaluation. Use
  `--supercell` and `--max-structures` when exploring.
- **A failed phonon run is reported, not raised**, so a batch always finishes; check the
  `success` and `error` columns of `phonons.csv`.
- **An unconverged pre-relaxation is a failed check, not an unstable material.** Displacing a
  geometry that is still carrying forces manufactures imaginary modes, so the check stops
  there and says so in `error`; `relaxation_converged` records it either way.
- **A head belongs to the energies it produced.** With a multi-task DeePMD checkpoint, a head
  that the checkpoint rejects is an error rather than a quiet reload without one, and the head
  actually used is written into `hull.csv` as `calculator_head`.
- The demo energies are round synthetic numbers chosen to exercise the three outcomes of the
  stability test. They are not calculated values.
