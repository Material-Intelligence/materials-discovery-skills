"""Shared helpers: compositions, Materials Project access, structure I/O, calculators and logging."""

from matdisc.common.calculators import load_calculator
from matdisc.common.composition import chemsys, natoms, parse, reduced_formula
from matdisc.common.io import from_ase, read_structure, to_ase, write_structure
from matdisc.common.logging import configure_logging, get_logger
from matdisc.common.mp import get_mprester

__all__ = [
    "chemsys",
    "configure_logging",
    "from_ase",
    "get_logger",
    "get_mprester",
    "load_calculator",
    "natoms",
    "parse",
    "read_structure",
    "reduced_formula",
    "to_ase",
    "write_structure",
]
