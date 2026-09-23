# Bundled Materials Project data

Two things in this repository are Materials Project data rather than this project's own
output. Both are redistributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
which requires the attribution below. `NOTICE.md` at the repository root carries the same
notice alongside the third-party software list.

## Attribution

> A. Jain, S. P. Ong, G. Hautier, W. Chen, W. D. Richards, S. Dacek, S. Cholia, D. Gunter,
> D. Skinner, G. Ceder and K. A. Persson, "The Materials Project: A materials genome approach
> to accelerating materials innovation", *APL Materials* **1**, 011002 (2013).
> doi:[10.1063/1.4812323](https://doi.org/10.1063/1.4812323)

Retrieved from the Materials Project API. The retrieval date and database version were not
recorded. Treat the values as a frozen snapshot for demonstration, not as the Materials
Project's current numbers — re-run `examples/01_competing_phases.py` with your own key for
those.

ICSD collection codes appear in the file names and in the `icsd_ids` column. They are
Materials Project cross-references to the Inorganic Crystal Structure Database; no ICSD data
is included in this repository.

## What is bundled

### `examples/data/BaCdP_competing_phases.csv`

Twenty-one Ba-Cd-P phases: the three elements, the Ba-Cd, Ba-P and Cd-P binaries, and the
ternary, each carrying an ICSD cross-reference. The columns are
`matdisc.competing.search.COMPETING_PHASE_COLUMNS`:

| Column | Meaning |
|---|---|
| `material_id` | Materials Project id |
| `formula_pretty` | the formula as Materials Project reports it |
| `composition` | the canonical reduced formula, from `pymatgen.core.Composition` |
| `natoms` | atoms in one reduced formula unit |
| `formation_energy_per_atom` | eV/atom, as Materials Project reports it |
| `energy_per_atom` | **empty in this file** — the harvest that produced it did not record it |
| `is_stable_mp` | whether Materials Project puts the phase on its own hull |
| `icsd_ids` | ICSD cross-references, `\|`-separated |

Because `energy_per_atom` is empty, read this table with
`--energy-column formation_energy_per_atom` (or `energy_column="formation_energy_per_atom"`).
Asking for the empty column fails with a message saying no competing phase carried a usable
energy, rather than silently producing a wrong hull.

The `composition` and `natoms` columns were re-derived with `pymatgen.core.Composition` from
`formula_pretty`. One row changes as a result: `mp-8279`, whose Materials Project formula is
`Ba(CdP)2`. It is now recorded as BaCd2P2 with five atoms per formula unit; it had been
recorded as `BaCdP` with three.

### `examples/data/structures/`

Three relaxed structures in VASP POSCAR format, downloaded by `material_id` and named
`<material_id>_<first ICSD code>.vasp`:

| File | Material | Composition |
|---|---|---|
| `mp-122_icsd-77367.vasp` | mp-122 | Ba |
| `mp-527_icsd-58642.vasp` | mp-527 | BaCd |
| `mp-11266_icsd-260668.vasp` | mp-11266 | BaCd2 |

They are example inputs — enough to run the VASP input writer or the structure reader without
credentials — not a complete set of competing-phase structures for the Ba-Cd-P hull. Download
the full set with `python examples/01_competing_phases.py --download`.

## Terms

Materials Project data is subject to the terms at <https://materialsproject.org/about/terms>.
Anyone redistributing these files further carries the same attribution requirement.
