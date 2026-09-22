"""Formula parsing, reduction and atom counting.

Every formula in this package is parsed by :class:`pymatgen.core.Composition`. The tests
below pin the behaviour that the hand-written regular expression it replaced got wrong: a
parenthesised multiplier such as ``Ba(CdP)2`` was read as if the parentheses were not there,
which turned BaCd2P2 into ``BaCdP`` with three atoms instead of five. In a Ba-Cd-P hull that
mislabels a competing phase as the candidate itself.
"""

from __future__ import annotations

import pytest
from pymatgen.core import Composition

from matdisc.common import composition


class TestParse:
    """Parsing a formula string into a composition."""

    def test_parse_returns_composition(self) -> None:
        comp = composition.parse("BaCdP")
        assert isinstance(comp, Composition)
        assert comp.get_el_amt_dict() == {"Ba": 1.0, "Cd": 1.0, "P": 1.0}

    def test_parse_passes_a_composition_through(self) -> None:
        comp = Composition("BaCdP")
        assert composition.parse(comp) is comp

    def test_parse_accepts_spaced_formulas(self) -> None:
        assert composition.parse("Ba1 Cd2 P2").get_el_amt_dict() == {"Ba": 1.0, "Cd": 2.0, "P": 2.0}

    def test_parse_rejects_nonsense(self) -> None:
        with pytest.raises(ValueError):
            composition.parse("not a formula")


class TestParenthesisedFormulas:
    """The regression the regular-expression parser introduced (audit finding M1)."""

    def test_nested_multiplier_is_applied(self) -> None:
        comp = composition.parse("Ba(CdP)2")
        assert comp.get_el_amt_dict() == {"Ba": 1.0, "Cd": 2.0, "P": 2.0}

    def test_natoms_counts_five_not_three(self) -> None:
        assert composition.natoms("Ba(CdP)2") == 5
        assert composition.natoms("BaCd2P2") == 5

    def test_both_spellings_reduce_to_the_same_string(self) -> None:
        assert composition.reduced_formula("Ba(CdP)2") == composition.reduced_formula("BaCd2P2")

    def test_it_is_not_confused_with_the_ternary_bacdp(self) -> None:
        assert composition.reduced_formula("Ba(CdP)2") != composition.reduced_formula("BaCdP")
        assert composition.natoms("Ba(CdP)2") != composition.natoms("BaCdP")


class TestReducedFormula:
    """Reduction to the canonical formula and the matching atom count."""

    @pytest.mark.parametrize(
        ("formula", "expected_formula", "expected_natoms"),
        [
            ("BaCdP", "BaCdP", 3),
            ("Ba2P", "Ba2P", 3),
            ("Ba2Cd2P2", "BaCdP", 3),
            ("Ba", "Ba", 1),
            ("Ba2Cd4", "BaCd2", 3),
            ("Ba(CdP)2", "Ba(CdP)2", 5),
        ],
    )
    def test_reduction(self, formula: str, expected_formula: str, expected_natoms: int) -> None:
        assert composition.reduced_formula(formula) == expected_formula
        assert composition.natoms(formula) == expected_natoms

    def test_reduction_is_idempotent(self) -> None:
        once = composition.reduced_formula("Ba4Cd8P8")
        assert composition.reduced_formula(once) == once


class TestChemicalSystem:
    """Chemical-system and element listing."""

    @pytest.mark.parametrize("formula", ["BaCdP", "Ba(CdP)2", "Ba2Cd2P2", "P2Cd1Ba1"])
    def test_chemsys_is_alphabetical_and_spelling_independent(self, formula: str) -> None:
        assert composition.chemsys(formula) == "Ba-Cd-P"

    def test_elements_are_sorted_symbols(self) -> None:
        assert composition.elements("Ba(CdP)2") == ["Ba", "Cd", "P"]

    def test_unary_chemsys(self) -> None:
        assert composition.chemsys("Ba") == "Ba"
        assert composition.elements("Ba") == ["Ba"]
