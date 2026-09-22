# Examples

Three scripts, in the order they are worth running. Only the second needs anything beyond the
package's own dependencies.

| Script | Needs | Runtime |
|---|---|---|
| `00_quickstart.py` | nothing | ~1 s |
| `01_competing_phases.py` | `MP_API_KEY` and network access | tens of seconds |
| `02_hull_from_csv.py` | nothing | ~1 s |

```bash
pip install -e .

python examples/00_quickstart.py            # offline: three candidates on a toy hull
python examples/02_hull_from_csv.py         # offline: a real Ba-Cd-P table, frozen in examples/data

export MP_API_KEY=your_key_here             # https://materialsproject.org/api
python examples/01_competing_phases.py --outdir example_output
```

`matdisc hull --demo` runs the same calculation as `00_quickstart.py` from the console script.

## What each one shows

**`00_quickstart.py`** places three candidates on the convex hull of a small synthetic Ba-Cd-P
system: one below the hull, one above it, and one sitting exactly on a tie-line between two
hull phases. The third is the point of the example — the criterion this package uses is
`e_above_hull <= tolerance`, so a phase on the hull reads stable. The energies are round
made-up numbers, not measured or calculated ones.

**`01_competing_phases.py`** harvests the real competing phases of a chemical system from the
Materials Project: one query per subsystem, only the six document fields the table uses, and
only materials carrying an ICSD cross-reference unless `--all` is passed. With `--download` it
also fetches a structure file per phase.

**`02_hull_from_csv.py`** runs the hull offline against the frozen table in `data/`, with the
Ba-Cd-P ternary as the candidate and the twenty other phases as the hull. It reads real
Materials Project formation energies and shows why the atom count matters: `Ba(CdP)2` is
BaCd2P2 with five atoms per formula unit, not `BaCdP` with three.

Data in `data/` is Materials Project data, licensed CC BY 4.0 — see
[`data/README.md`](data/README.md).
