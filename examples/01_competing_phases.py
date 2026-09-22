#!/usr/bin/env python3
"""Harvest the competing phases of Ba-Cd-P from the Materials Project.

Needs a Materials Project API key and network access; nothing else. Get a key from
https://materialsproject.org/api and export it::

    export MP_API_KEY=your_key_here
    python examples/01_competing_phases.py --outdir example_output

A convex hull for Ba-Cd-P rests on every subsystem: the elements Ba, Cd and P, the binaries
Ba-Cd, Ba-P and Cd-P, and the ternary itself. Materials Project matches ``chemsys`` exactly,
so the search below issues one query per subsystem -- seven for a ternary -- asking only for
the six document fields the table uses.

The default filter keeps materials that carry an ICSD cross-reference, which restricts the
reference set to compounds that have been made and characterised experimentally. That is a
choice, not a law: pass ``--all`` to keep everything Materials Project holds, and the hull
will sit lower.

Structures are downloaded only with ``--download``. Anything this script writes is Materials
Project data, licensed CC BY 4.0; see ``NOTICE.md`` before redistributing it.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from matdisc.common.logging import configure_logging
from matdisc.competing.download import download_structures
from matdisc.competing.search import find_competing_phases


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments to parse, or ``None`` to read ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--chemsys", default="Ba-Cd-P", help="chemical system to search")
    parser.add_argument("-o", "--outdir", default="example_output", help="directory to write into")
    parser.add_argument("--all", action="store_true", help="keep materials without an ICSD cross-reference too")
    parser.add_argument("--download", action="store_true", help="also download a structure file per phase")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the harvest.

    Args:
        argv: Arguments to parse, or ``None`` to read ``sys.argv``.

    Returns:
        The process exit status.
    """
    args = parse_args(argv)
    configure_logging(level=logging.INFO, stream=sys.stderr)

    if not os.environ.get("MP_API_KEY"):
        print(
            "MP_API_KEY is not set. Get a key from https://materialsproject.org/api and export it:\n"
            "    export MP_API_KEY=your_key_here",
            file=sys.stderr,
        )
        return 1

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    table = find_competing_phases(args.chemsys, only_icsd=not args.all)
    if table.empty:
        print(f"No competing phase found for {args.chemsys}. Try --all.", file=sys.stderr)
        return 1

    csv_path = outdir / "competing_phases.csv"
    table.to_csv(csv_path, index=False)

    print(f"{len(table)} competing phase(s) for {args.chemsys}:")
    print(table[["material_id", "composition", "natoms", "formation_energy_per_atom"]].to_string(index=False))
    print(f"\nWrote {csv_path}")

    if args.download:
        structures_dir = outdir / "structures"
        written = download_structures(table, structures_dir)
        succeeded = sum(1 for path in written.values() if path)
        print(f"Downloaded {succeeded} of {len(written)} structure(s) into {structures_dir}")
        print("These files are Materials Project data, licensed CC BY 4.0 -- see NOTICE.md.")

    print("\nNext: feed this table to the hull. See examples/02_hull_from_csv.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
