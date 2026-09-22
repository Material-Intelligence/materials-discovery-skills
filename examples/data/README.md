# Example data

Everything in this directory is **Materials Project data**, redistributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

- `BaCdP_competing_phases.csv` — 21 ICSD-backed Ba-Cd-P phases, read by
  `../02_hull_from_csv.py`. Its only energy column is `formation_energy_per_atom`;
  `energy_per_atom` is present but empty.
- `structures/*.vasp` — three relaxed structures (Ba, BaCd, BaCd2) in POSCAR format.

Cite: A. Jain *et al.*, "The Materials Project: A materials genome approach to accelerating
materials innovation", *APL Materials* **1**, 011002 (2013),
doi:[10.1063/1.4812323](https://doi.org/10.1063/1.4812323).

Full provenance, the column meanings and the terms are in
[`../../docs/data/README.md`](../../docs/data/README.md); the repository-wide notice is in
[`../../NOTICE.md`](../../NOTICE.md).
