#!/usr/bin/env python3
"""Offline convex hull from the frozen Ba-Cd-P competing-phase table.

Runs in about a second, from the CSV in ``examples/data/``. No Materials Project key and no
network: the table was harvested once and is shipped with the repository.

The question the script asks is the one the screening stage asks of every candidate: given
the phases a material could decompose into, is it stable? Here the candidate is the Ba-Cd-P
ternary Ba(CdP)2 (BaCd2P2, Materials Project id mp-8279) and the hull is built from the other
twenty phases in the table -- the elements and the Ba-Cd, Ba-P and Cd-P binaries. The
candidate is removed from the competing set first, because a phase compared against itself is
always exactly on its own hull and the answer means nothing.

Two details worth reading the code for:

* **The energy column.** This table's only energy is ``formation_energy_per_atom``, so that is
  what both sides are read on. ``energy_per_atom`` is present but empty -- the harvest that
  produced this file never recorded it -- and asking for it produces a clear failure rather
  than a wrong number. Candidates and competing phases must always be on one energy scale.
* **The atom count.** ``Ba(CdP)2`` is BaCd2P2 with five atoms per formula unit. Read with the
  parentheses ignored it becomes ``BaCdP`` with three, which is a different compound and the
  target ternary of this very system. Everything here goes through
  :class:`pymatgen.core.Composition`.

The hull the candidate is measured against holds only ICSD-backed phases, which is the
default of the harvest that produced the table. That is a smaller reference set than the full
Materials Project one, so a phase can come out below this hull, as this one does.

Usage::

    python examples/02_hull_from_csv.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from matdisc.screening.hull import DEFAULT_TOLERANCE, compute_e_above_hull

DATA = Path(__file__).resolve().parent / "data" / "BaCdP_competing_phases.csv"
ENERGY_COLUMN = "formation_energy_per_atom"
CANDIDATE_ID = "mp-8279"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments to parse, or ``None`` to read ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=str(DATA), help="competing-phase table to read")
    parser.add_argument("--candidate", default=CANDIDATE_ID, help="material_id to treat as the candidate")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="a candidate is stable when e_above_hull <= this, in eV/atom",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the hull and place the candidate on it.

    Args:
        argv: Arguments to parse, or ``None`` to read ``sys.argv``.

    Returns:
        The process exit status.
    """
    args = parse_args(argv)
    table = pd.read_csv(args.csv)

    candidates = table[table["material_id"] == args.candidate].rename(columns={"material_id": "id"})
    if candidates.empty:
        print(f"{args.candidate} is not in {args.csv}")
        return 1
    competing = table[table["material_id"] != args.candidate]

    print(f"Table: {args.csv}")
    print(f"{len(competing)} competing phase(s), candidate {args.candidate}, energies read from {ENERGY_COLUMN}.")
    print()
    print("Competing phases:")
    print(competing[["material_id", "composition", "natoms", ENERGY_COLUMN]].to_string(index=False))
    print()

    result = compute_e_above_hull(
        candidates,
        competing,
        tolerance=args.tolerance,
        energy_column=ENERGY_COLUMN,
    )

    columns = ["id", "formula_pretty", "composition", "natoms", ENERGY_COLUMN, "e_above_hull", "is_stable"]
    with pd.option_context("display.width", 200):
        print("Candidate:")
        print(result[columns].to_string(index=False))
    print()

    row = result.iloc[0]
    verdict = "at or below the hull" if row["is_stable"] else "above the hull"
    print(f"{row['id']} ({row['composition']}, {int(row['natoms'])} atoms per formula unit) is {verdict}:")
    print(f"  e_above_hull = {row['e_above_hull']:+.4f} eV/atom, criterion e_above_hull <= {args.tolerance:g}")
    print(f"  decomposition: {row['decomposition']}")
    print()
    print("The hull here is built from ICSD-backed phases only, which is a smaller reference set")
    print("than the full Materials Project hull; a Materials Project ground state can sit below it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
