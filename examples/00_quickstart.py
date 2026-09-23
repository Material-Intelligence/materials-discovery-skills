#!/usr/bin/env python3
"""Offline quickstart: place three candidates on a convex hull.

Runs in about a second. No Materials Project key, no network, no GPU, no VASP, no
machine-learning checkpoint -- only pymatgen, pandas and numpy, which the base install
(``pip install -e .``) already brings in.

The numbers come from :func:`matdisc.screening.hull.toy_system`: a small synthetic Ba-Cd-P
system chosen to exercise the three outcomes of the stability test. They are round made-up
values, not measured or calculated ones. For real Materials Project energies see
``02_hull_from_csv.py``.

The third candidate is the one worth watching. ``Ba2Cd2P3`` is one formula unit of BaP plus
one of BaCd2P2, so it sits exactly on the tie-line between two phases that are already on the
hull: its energy above the hull is zero to within the rounding of the hull solve. The
criterion this package uses -- stable when ``e_above_hull <= tolerance`` -- calls it stable.
A strict ``e_above_hull < 0`` test would call a phase sitting exactly on the hull unstable.

Usage::

    python examples/00_quickstart.py
"""

from __future__ import annotations

import pandas as pd

from matdisc.screening.hull import DEFAULT_TOLERANCE, compute_e_above_hull, toy_system


def main() -> int:
    """Run the toy hull and print the verdict table.

    Returns:
        The process exit status.
    """
    candidates, competing = toy_system()

    print("Competing phases (the hull is built from these):")
    print(competing[["material_id", "composition", "natoms", "energy_per_atom"]].to_string(index=False))
    print()

    result = compute_e_above_hull(candidates, competing, tolerance=DEFAULT_TOLERANCE)

    print("Candidates:")
    columns = ["id", "composition", "energy_per_atom", "e_above_hull", "is_stable", "decomposition"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print(result[columns].to_string(index=False))
    print()

    stable = int(result["is_stable"].sum())
    print(f"{stable} of {len(result)} candidate(s) are at or below the hull.")
    print(f"Criterion: stable when e_above_hull <= {DEFAULT_TOLERANCE:g} eV/atom.")

    on_hull = result.set_index("id").loc["toy-on-hull"]
    print(
        f"The candidate on the hull reads e_above_hull = {on_hull['e_above_hull']:.3e} eV/atom "
        f"-> is_stable = {bool(on_hull['is_stable'])}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
