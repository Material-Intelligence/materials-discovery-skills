"""Composition helpers built on :class:`pymatgen.core.Composition`.

Every formula string in this package is parsed here, by pymatgen, so that nested formulas
such as ``Ba(CdP)2`` keep their multipliers. Nothing in this package parses a formula with a
regular expression.

The reduced formula returned by :func:`reduced_formula` is pymatgen's canonical one: two
strings that describe the same composition always reduce to the same string, which is what
makes it safe to group and join on.
"""

from __future__ import annotations

from pymatgen.core import Composition

__all__ = ["parse", "reduced_formula", "natoms", "chemsys", "elements"]


def parse(formula: str | Composition) -> Composition:
    """Parse a chemical formula into a :class:`~pymatgen.core.Composition`.

    Args:
        formula: A formula string such as ``"BaCdP"``, ``"Ba(CdP)2"`` or ``"Ba2 Cd1 P1"``,
            or an already-parsed :class:`~pymatgen.core.Composition`, which is returned
            unchanged.

    Returns:
        The parsed composition.

    Raises:
        ValueError: If the string is not a formula pymatgen can parse.
    """
    if isinstance(formula, Composition):
        return formula
    return Composition(formula)


def reduced_formula(formula: str | Composition) -> str:
    """Return the canonical reduced formula of a composition.

    Args:
        formula: Formula string or :class:`~pymatgen.core.Composition`.

    Returns:
        The reduced formula, for example ``"Ba(CdP)2"`` for ``"Ba2Cd4P4"``. Compositions
        that are equal up to a common factor give the same string.
    """
    return parse(formula).reduced_formula


def natoms(formula: str | Composition) -> int:
    """Return the number of atoms in one reduced formula unit.

    Args:
        formula: Formula string or :class:`~pymatgen.core.Composition`.

    Returns:
        Atoms per reduced formula unit -- 5 for ``"Ba(CdP)2"``, 3 for ``"BaCdP"``. The count
        is rounded to the nearest integer, which matters only for partially occupied
        compositions.
    """
    return int(round(parse(formula).reduced_composition.num_atoms))


def chemsys(formula: str | Composition) -> str:
    """Return the chemical system of a composition.

    Args:
        formula: Formula string or :class:`~pymatgen.core.Composition`.

    Returns:
        The elements joined by hyphens in alphabetical order, for example ``"Ba-Cd-P"``.
        This is the form the Materials Project ``chemsys`` filter expects.
    """
    return parse(formula).chemical_system


def elements(formula: str | Composition) -> list[str]:
    """Return the element symbols of a composition in alphabetical order.

    Args:
        formula: Formula string or :class:`~pymatgen.core.Composition`.

    Returns:
        Element symbols, for example ``["Ba", "Cd", "P"]``.
    """
    return sorted(el.symbol for el in parse(formula).elements)
