"""Convex-hull placement of candidate energies.

The criterion under test is the one the package documents: a candidate is stable when its
energy above the hull is at or below the tolerance, which defaults to 1e-6 eV/atom. A strict
``e_above_hull < 0`` test would call a phase sitting exactly on the hull unstable, and that
is the first case below.

The toy system is a small A/B/C-style Ba-Cd-P table with round synthetic numbers. It is not
measured or calculated data, and no candidate composition appears in the competing-phase
table, so nothing is compared against itself.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from matdisc.screening.hull import (
    DEFAULT_TOLERANCE,
    build_phase_diagram,
    compute_e_above_hull,
    formation_energy_per_atom,
    load_element_references,
    split_usable_rows,
    toy_system,
)


def _candidates(*rows: tuple[str, str, float]) -> pd.DataFrame:
    """Build a candidates table.

    Args:
        *rows: ``(id, composition, energy_per_atom)`` triples.

    Returns:
        The table, with the columns :func:`compute_e_above_hull` expects.
    """
    return pd.DataFrame(list(rows), columns=["id", "composition", "energy_per_atom"])


class TestStabilityCriterion:
    """Where the line between stable and unstable falls."""

    def test_a_phase_exactly_on_the_hull_is_stable(self) -> None:
        """``BaP`` at the competing table's own energy sits on the hull: e_above_hull == 0."""
        _, competing = toy_system()
        result = compute_e_above_hull(_candidates(("on-hull", "BaP", -0.60)), competing)

        assert result.loc[0, "e_above_hull"] == pytest.approx(0.0, abs=1e-12)
        assert bool(result.loc[0, "is_stable"]) is True, "a strict '< 0' test would fail here"

    def test_a_phase_above_the_hull_is_unstable(self) -> None:
        _, competing = toy_system()
        result = compute_e_above_hull(_candidates(("above", "BaCdP", -0.10)), competing)

        assert result.loc[0, "e_above_hull"] > 0
        assert bool(result.loc[0, "is_stable"]) is False

    def test_a_phase_below_the_hull_is_stable(self) -> None:
        _, competing = toy_system()
        result = compute_e_above_hull(_candidates(("below", "BaCdP", -0.90)), competing)

        assert result.loc[0, "e_above_hull"] < 0
        assert bool(result.loc[0, "is_stable"]) is True

    def test_the_tolerance_is_applied(self) -> None:
        """A candidate 5 meV/atom above the hull: unstable by default, stable at 10 meV."""
        _, competing = toy_system()
        candidates = _candidates(("just-above", "BaP", -0.595))

        strict = compute_e_above_hull(candidates, competing)
        assert strict.loc[0, "e_above_hull"] == pytest.approx(0.005, abs=1e-9)
        assert bool(strict.loc[0, "is_stable"]) is False

        loose = compute_e_above_hull(candidates, competing, tolerance=0.01)
        assert bool(loose.loc[0, "is_stable"]) is True

    def test_the_default_tolerance_is_small_and_positive(self) -> None:
        assert 0 < DEFAULT_TOLERANCE <= 1e-4


class TestToySystem:
    """The offline example the quickstart and ``matdisc hull --demo`` print."""

    def test_the_three_verdicts(self) -> None:
        candidates, competing = toy_system()
        result = compute_e_above_hull(candidates, competing).set_index("id")

        assert bool(result.loc["toy-below-hull", "is_stable"]) is True
        assert bool(result.loc["toy-above-hull", "is_stable"]) is False
        assert bool(result.loc["toy-on-hull", "is_stable"]) is True
        assert result.loc["toy-on-hull", "e_above_hull"] == pytest.approx(0.0, abs=1e-9)

    def test_no_candidate_is_its_own_competing_phase(self) -> None:
        candidates, competing = toy_system()
        assert set(candidates["composition"]).isdisjoint(set(competing["composition"]))

    def test_both_energy_columns_agree(self) -> None:
        """The toy elemental references are zero, so the two columns are interchangeable."""
        candidates, competing = toy_system()
        by_total = compute_e_above_hull(candidates, competing, energy_column="energy_per_atom")
        by_formation = compute_e_above_hull(
            candidates.assign(formation_energy_per_atom=candidates["energy_per_atom"]),
            competing,
            energy_column="formation_energy_per_atom",
        )
        assert list(by_total["e_above_hull"]) == pytest.approx(list(by_formation["e_above_hull"]))


class TestEnergyConvention:
    """Per-atom energies in, per-atom energies out; nothing is divided twice."""

    def test_the_entry_energy_is_the_per_atom_energy_times_the_atom_count(self) -> None:
        competing = pd.DataFrame(
            [("x-Ba", "Ba", 0.0), ("x-P", "P", 0.0), ("x-BaP2", "BaP2", -1.0)],
            columns=["material_id", "composition", "energy_per_atom"],
        )
        diagram = build_phase_diagram(competing)
        entry = next(entry for entry in diagram.all_entries if entry.name == "x-BaP2")

        assert entry.energy == pytest.approx(-3.0), "BaP2 has 3 atoms per formula unit"
        assert entry.energy_per_atom == pytest.approx(-1.0)

    def test_a_candidates_hull_energy_is_reported_per_atom(self) -> None:
        _, competing = toy_system()
        result = compute_e_above_hull(_candidates(("c", "BaCdP", -0.40)), competing)
        row = result.iloc[0]
        assert row["hull_energy_per_atom"] == pytest.approx(row["energy_per_atom"] - row["e_above_hull"])

    def test_formation_energy_conversion(self) -> None:
        references = {"Ba": -1.9, "P": -5.4}
        # BaP2: one third Ba, two thirds P.
        reference = (-1.9 + 2 * -5.4) / 3
        assert formation_energy_per_atom(-5.0, "BaP2", references) == pytest.approx(-5.0 - reference)

    def test_formation_energy_needs_every_element(self) -> None:
        with pytest.raises(KeyError, match="Cd"):
            formation_energy_per_atom(-5.0, "BaCdP", {"Ba": -1.9, "P": -5.4})


class TestMissingCompetingPhases:
    """What happens when the hull cannot be built, or cannot hold the candidate.

    The failure that mattered in the original code was a handler that raised ``NameError``
    while reporting the real problem, which turned one bad candidate into an aborted batch.
    """

    def test_an_empty_competing_table_raises_a_readable_error(self) -> None:
        candidates, competing = toy_system()
        with pytest.raises(ValueError) as error:
            compute_e_above_hull(candidates, competing.iloc[0:0])
        assert "competing phase" in str(error.value)
        assert not isinstance(error.value, NameError)

    def test_competing_phases_without_energies_raise_a_readable_error(self) -> None:
        competing = pd.DataFrame(
            [("x-Ba", "Ba", float("nan")), ("x-P", "P", float("nan"))],
            columns=["material_id", "composition", "energy_per_atom"],
        )
        with pytest.raises(ValueError, match="usable energy"):
            compute_e_above_hull(_candidates(("c", "BaP", -0.5)), competing)

    def test_a_hull_that_cannot_be_built_names_the_reason(self) -> None:
        """No elemental references: pymatgen cannot span a phase diagram."""
        competing = pd.DataFrame(
            [("x-BaP", "BaP", -0.6), ("x-BaP2", "BaP2", -0.8)],
            columns=["material_id", "composition", "energy_per_atom"],
        )
        with pytest.raises(ValueError, match="elemental reference"):
            compute_e_above_hull(_candidates(("c", "BaP3", -0.5)), competing)

    def test_a_candidate_outside_the_diagram_is_reported_not_raised(self) -> None:
        """One uncoverable candidate must not cost the rest of the batch its results."""
        _, competing = toy_system()
        candidates = _candidates(("outside", "LiP", -0.5), ("inside", "BaCdP", -0.9))
        result = compute_e_above_hull(candidates, competing).set_index("id")

        assert math.isnan(result.loc["outside", "e_above_hull"])
        assert bool(result.loc["outside", "is_stable"]) is False
        assert "Li" in result.loc["outside", "decomposition"]
        assert bool(result.loc["inside", "is_stable"]) is True

    def test_a_candidate_without_an_energy_is_reported_not_raised(self) -> None:
        _, competing = toy_system()
        candidates = _candidates(("no-energy", "BaCdP", float("nan")), ("fine", "BaCdP", -0.9))
        result = compute_e_above_hull(candidates, competing).set_index("id")

        assert math.isnan(result.loc["no-energy", "e_above_hull"])
        assert bool(result.loc["fine", "is_stable"]) is True

    def test_a_missing_column_names_the_column(self) -> None:
        candidates, competing = toy_system()
        with pytest.raises(KeyError, match="energy_per_atom"):
            compute_e_above_hull(candidates.drop(columns=["energy_per_atom"]), competing)


class TestResultTable:
    """The shape of what comes back."""

    def test_the_input_columns_are_kept_and_four_are_added(self) -> None:
        candidates, competing = toy_system()
        result = compute_e_above_hull(candidates, competing)

        assert list(result.columns)[: len(candidates.columns)] == list(candidates.columns)
        assert list(result.columns)[len(candidates.columns) :] == [
            "e_above_hull",
            "hull_energy_per_atom",
            "is_stable",
            "decomposition",
        ]

    def test_the_decomposition_names_hull_phases(self) -> None:
        _, competing = toy_system()
        result = compute_e_above_hull(_candidates(("c", "BaCdP", -0.40)), competing)
        assert "*" in result.loc[0, "decomposition"]

    def test_the_input_table_is_not_modified(self) -> None:
        candidates, competing = toy_system()
        before = list(candidates.columns)
        compute_e_above_hull(candidates, competing)
        assert list(candidates.columns) == before


class TestElementReferences:
    """Reading elemental reference energies from a table."""

    def test_only_single_element_rows_are_used(self, tmp_path) -> None:
        path = tmp_path / "references.csv"
        pd.DataFrame(
            [("Ba", -1.9), ("P", -5.4), ("BaP2", -4.0)],
            columns=["composition", "energy_per_atom"],
        ).to_csv(path, index=False)

        assert load_element_references(path) == {"Ba": -1.9, "P": -5.4}

    def test_the_lowest_energy_polymorph_wins(self, tmp_path) -> None:
        path = tmp_path / "references.csv"
        pd.DataFrame(
            [("P", -5.4), ("P4", -5.1)],
            columns=["composition", "energy_per_atom"],
        ).to_csv(path, index=False)

        assert load_element_references(path) == {"P": -5.4}


class TestShippedExampleTable:
    """The Ba-Cd-P table under ``examples/data`` is a working hull input."""

    def test_the_ternary_sits_below_the_hull_of_its_binaries(self, example_competing_csv) -> None:
        table = pd.read_csv(example_competing_csv)
        candidate = table[table["material_id"] == "mp-8279"].rename(columns={"material_id": "id"})
        competing = table[table["material_id"] != "mp-8279"]

        result = compute_e_above_hull(candidate, competing, energy_column="formation_energy_per_atom")
        row = result.iloc[0]
        assert row["composition"] == "Ba(CdP)2"
        assert int(row["natoms"]) == 5
        assert bool(row["is_stable"]) is True

    def test_the_default_energy_column_fails_loudly_on_this_table(self, example_competing_csv) -> None:
        """``energy_per_atom`` is empty in this table; the failure says so rather than lying."""
        table = pd.read_csv(example_competing_csv)
        candidate = table.iloc[[0]].rename(columns={"material_id": "id"})
        with pytest.raises(ValueError, match="usable energy"):
            compute_e_above_hull(candidate, table)


class TestSplitUsableRows:
    """Which competing phases a hull may rest on, and which loss has to be reported."""

    def test_a_missing_energy_is_dropped(self) -> None:
        table = pd.DataFrame(
            [
                {"id": "good", "composition": "Cu", "energy_per_atom": -1.0, "converged": True},
                {"id": "failed", "composition": "Ag", "energy_per_atom": float("nan"), "converged": False},
            ]
        )
        usable, dropped = split_usable_rows(table)
        assert list(usable["id"]) == ["good"]
        assert list(dropped["id"]) == ["failed"]

    def test_an_unconverged_row_is_kept_unless_convergence_is_required(self) -> None:
        table = pd.DataFrame(
            [
                {"id": "converged", "composition": "Cu", "energy_per_atom": -1.0, "converged": True},
                {"id": "capped", "composition": "Ag", "energy_per_atom": -0.5, "converged": False},
            ]
        )
        assert list(split_usable_rows(table)[0]["id"]) == ["converged", "capped"]
        assert list(split_usable_rows(table, require_converged=True)[1]["id"]) == ["capped"]

    def test_a_table_without_a_convergence_column_is_left_alone(self) -> None:
        """A harvested Materials Project table carries no ``converged`` column."""
        table = pd.DataFrame([{"id": "mp-1", "composition": "Cu", "energy_per_atom": -1.0}])
        usable, dropped = split_usable_rows(table, require_converged=True)
        assert len(usable) == 1
        assert dropped.empty
