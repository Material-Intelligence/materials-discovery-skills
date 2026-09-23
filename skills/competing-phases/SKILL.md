---
name: competing-phases
description: Use when you need the competing phases of a chemical system - every subsystem, ICSD-backed by default - harvested from the Materials Project as a table plus structure files, ready to build a convex hull from.
---

# Competing phases

A convex hull for a candidate rests on every phase the candidate competes with, including
the phases of each subsystem: a Ba-Cd-P hull needs Ba, Cd and P, then Ba-Cd, Ba-P and Cd-P,
then the Ba-Cd-P phases themselves. This skill enumerates those subsystems, queries the
Materials Project for each one, filters the result, and downloads the relaxed structure
behind every entry.

Formulas are parsed by `pymatgen.core.Composition`, so a nested formula such as `Ba(CdP)2`
keeps its multiplier and reduces to the same canonical string every time.

## Prerequisites

- The package itself: `pip install -e .` from the repository root. `mp-api` and `pymatgen`
  are required dependencies, so no extra is needed for this skill.
- `MP_API_KEY` must be set. The key is read from the environment at call time:

  ```bash
  export MP_API_KEY=your_key_here     # keys are issued at https://materialsproject.org/api
  ```

  Without it the command stops with:
  `No Materials Project API key found. Set the MP_API_KEY environment variable or pass
  api_key=... explicitly.`
- Network access. Everything in this skill talks to the Materials Project; only the saved
  CSV can be reused offline.

## Usage

```bash
matdisc competing --help
```

Harvest a system and download the structures:

```bash
matdisc competing -c Ba-Cd-P -o runs/competing
```

Useful switches: `--no-icsd` keeps materials without an ICSD cross-reference (a wider but
less experimentally grounded set), `--only-stable` keeps only phases the Materials Project
puts on its own hull, `--max-energy-above-hull` drops phases further above that hull than the
value given, `--all-polymorphs` keeps every polymorph instead of one per formula,
`--allow-partial` accepts a reference set in which a subsystem query failed, `--no-download`
writes the table alone, and `--no-unary`, `--no-binary`, `--no-higher-order` drop whole
subsystem orders.

From Python:

```python
from matdisc.competing.search import find_competing_phases
from matdisc.competing.download import download_structures

phases = find_competing_phases("Ba-Cd-P", only_icsd=True)
download_structures(phases, "runs/competing/structures")
```

The saved table is the hull's input and needs no network afterwards:

```bash
matdisc hull --candidates candidates.csv --competing runs/competing/competing_phases.csv
```

## Inputs and outputs

Input: a chemical system string (`Ba-Cd-P`), or a formula (`BaCdP`), which is reduced to its
elements.

`<outdir>/competing_phases.csv`, one row per phase:

| Column | Meaning |
|---|---|
| `material_id` | Materials Project id, e.g. `mp-8279` |
| `formula_pretty` | Formula as the Materials Project reports it |
| `composition` | Canonical reduced formula from pymatgen |
| `natoms` | Atoms in one reduced formula unit |
| `formation_energy_per_atom` | eV/atom, as reported |
| `energy_per_atom` | eV/atom, as reported |
| `is_stable_mp` | On the Materials Project hull |
| `icsd_ids` | ICSD cross-references, e.g. `icsd-260668\|icsd-58643`; empty when none |

`<outdir>/structures/` holds one file per phase, named `mp-527_icsd-58642.vasp` when an ICSD
code is known and `mp-527.vasp` otherwise.

## Pitfalls

- **Both energy columns are already per atom.** Nothing downstream divides them by `natoms`
  again; `matdisc hull` multiplies by the atom count of the reduced formula to build its
  phase diagram entries.
- **One entry per reduced formula** is kept by default - the one with the lowest *Materials
  Project* formation energy per atom, which is not necessarily the polymorph the calculator
  used for screening would prefer. Pass `--all-polymorphs` (`unique_formula=False`) to keep
  them all.
- **The default ICSD filter is a scientific choice**, not a performance one: it restricts the
  reference set to compounds that have been made and characterised. It also **lowers the hull**.
  An ICSD-only set is a subset of the Materials Project's own reference set, so any phase it
  drops that would have been on the true hull is simply absent, and every `e_above_hull` measured
  against it is a lower bound on the real one. A candidate can come out stable, or even below
  the hull, that is not: `examples/02_hull_from_csv.py` ships exactly that case, with `mp-8279`
  landing at `e_above_hull = -0.0815` eV/atom against the shipped ICSD-only table. Use
  `--no-icsd` when the question is thermodynamic stability rather than "stable among the
  compounds already known".
- **A missing elemental reference breaks the hull.** If you drop the unary subsystems, the
  phase diagram cannot be built and `matdisc hull` reports it rather than guessing.
- **Downloads are paced** at 0.5 s apiece to stay inside the rate limit, and existing files
  are skipped, so a repeated run is cheap.
- **Materials Project data is CC BY 4.0.** If you redistribute the downloaded structures or
  the table, carry the attribution in `NOTICE.md` with them.
- **A failing subsystem query stops the search.** Its phases would otherwise be missing from
  the hull with nothing on the table to say so. `--allow-partial` (`allow_partial=True`)
  continues anyway and names the missing subsystems in the stage result and in
  `frame.attrs["missing_subsystems"]`.

## What comes next

The table and structures feed `stability-screening`: the screening stage relaxes the
downloaded competing structures with the same calculator as the candidates so that both
sides of the hull are on one energy scale.
