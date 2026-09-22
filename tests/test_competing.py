"""Competing-phase harvesting and structure download, with the Materials Project mocked.

This file re-implements, against the current API, the behaviours the original package's test
suite covered. That suite's source was lost; its 21 test ids survived in a stale pytest
cache, and the mapping from those ids to the classes below is:

===================================  =======================================================
Original test id                     Covered here by
===================================  =======================================================
``TestGetApiKey`` (4 tests)          :class:`TestApiKey`
``TestGetIcsdIdsFromDoc`` (5 tests)  :class:`TestIcsdIds`
``TestReduceFormula`` (5 tests)      :class:`TestReducedFormulaColumns` -- the same five
                                     formula shapes, checked where they now surface, in the
                                     ``composition`` and ``natoms`` columns of the table.
                                     The unit-level behaviour is in ``test_composition.py``.
``TestMockMPRester`` (1 test)        :class:`TestFindCompetingPhases`
``TestDownloadStructuresMock``       :class:`TestDownloadStructures`
``TestPrepareConvexHullInput`` (2)   :class:`TestHullHandoff` -- the old helper merged the
                                     competing and candidate CSVs into one file with a
                                     ``source`` column; the hull now takes the two tables
                                     directly, so what is worth testing is that handoff.
``TestGenerateCompetingPhasesFile``  :class:`TestTableOnDisk`
``TestIntegration`` (1 test)         :class:`TestAgainstTheRealApi`, marked ``network``
===================================  =======================================================
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from matdisc.common import mp
from matdisc.competing.download import download_structures
from matdisc.competing.search import COMPETING_PHASE_COLUMNS, SUMMARY_FIELDS, find_competing_phases
from matdisc.screening.hull import compute_e_above_hull
from tests.conftest import FakeMPRester, FakeSummaryDoc, ba_cd_p_docs, cubic_structure


class TestApiKey:
    """Resolving the Materials Project API key.

    No key is stored anywhere in this repository. The only sources are the argument and the
    ``MP_API_KEY`` environment variable, and an unset variable is an error rather than a
    silent fallback.
    """

    def test_key_from_argument(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MP_API_KEY", raising=False)
        assert mp.get_api_key("explicit_key") == "explicit_key"

    def test_key_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MP_API_KEY", "env_key")
        assert mp.get_api_key() == "env_key"

    def test_argument_wins_over_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MP_API_KEY", "env_key")
        assert mp.get_api_key("explicit_key") == "explicit_key"

    def test_missing_key_raises_with_instructions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MP_API_KEY", raising=False)
        with pytest.raises(RuntimeError) as error:
            mp.get_api_key()
        message = str(error.value)
        assert "MP_API_KEY" in message
        assert "materialsproject.org" in message

    def test_blank_key_is_treated_as_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MP_API_KEY", "   ")
        with pytest.raises(RuntimeError):
            mp.get_api_key()


class TestIcsdIds:
    """Reading ICSD cross-references out of a summary document."""

    def test_numeric_codes_are_prefixed_and_joined(self) -> None:
        doc = FakeSummaryDoc("mp-1", "BaCdP", icsd_ids=[201982, 23349])
        assert mp.icsd_ids_from_doc(doc) == "icsd-201982|icsd-23349"

    def test_already_prefixed_codes_are_left_alone(self) -> None:
        doc = FakeSummaryDoc("mp-1", "BaCdP", icsd_ids=["icsd-201982", "icsd-23349"])
        assert mp.icsd_ids_from_doc(doc) == "icsd-201982|icsd-23349"

    def test_other_databases_are_ignored(self) -> None:
        doc = FakeSummaryDoc("mp-1", "BaCdP", database_ids={"other": [123]})
        assert mp.icsd_ids_from_doc(doc) == ""

    def test_empty_database_ids(self) -> None:
        doc = FakeSummaryDoc("mp-1", "BaCdP", database_ids={})
        assert mp.icsd_ids_from_doc(doc) == ""

    def test_document_without_the_attribute(self) -> None:
        class Bare:
            """A document fetched without the ``database_IDs`` field."""

        assert mp.icsd_ids_from_doc(Bare()) == ""


class TestFindCompetingPhases:
    """The search itself, against a fake client."""

    def test_table_has_the_documented_columns(self, fake_mprester: FakeMPRester) -> None:
        table = find_competing_phases("Ba-Cd-P", client=fake_mprester)
        assert list(table.columns) == COMPETING_PHASE_COLUMNS
        assert not table.empty

    def test_every_subsystem_is_queried(self, fake_mprester: FakeMPRester) -> None:
        find_competing_phases("Ba-Cd-P", client=fake_mprester)
        queried = [search["chemsys"] for search in fake_mprester.searches]
        assert queried == ["Ba", "Cd", "P", "Ba-Cd", "Ba-P", "Cd-P", "Ba-Cd-P"]

    def test_a_formula_is_accepted_in_place_of_a_chemical_system(self, fake_mprester: FakeMPRester) -> None:
        find_competing_phases("BaCdP", client=fake_mprester)
        assert [search["chemsys"] for search in fake_mprester.searches][:3] == ["Ba", "Cd", "P"]

    def test_subsystem_orders_can_be_switched_off(self, fake_mprester: FakeMPRester) -> None:
        find_competing_phases(
            "Ba-Cd-P",
            include_binary=False,
            include_higher_order=False,
            client=fake_mprester,
        )
        assert [search["chemsys"] for search in fake_mprester.searches] == ["Ba", "Cd", "P"]

    def test_only_the_used_fields_are_requested(self, fake_mprester: FakeMPRester) -> None:
        find_competing_phases("Ba-Cd-P", client=fake_mprester)
        for search in fake_mprester.searches:
            assert search["fields"] == SUMMARY_FIELDS

    def test_only_icsd_filters_out_materials_without_a_cross_reference(self, fake_mprester: FakeMPRester) -> None:
        with_filter = find_competing_phases("Ba-Cd-P", client=fake_mprester)
        assert "mp-9999" not in set(with_filter["material_id"])

        fresh = FakeMPRester(docs_by_chemsys=fake_mprester.docs_by_chemsys)
        without_filter = find_competing_phases("Ba-Cd-P", only_icsd=False, client=fresh)
        assert "mp-9999" in set(without_filter["material_id"])

    def test_only_stable_filters_on_the_materials_project_hull(self) -> None:
        client = FakeMPRester(
            docs_by_chemsys={
                "Ba": [FakeSummaryDoc("mp-122", "Ba", 0.0, icsd_ids=[1])],
                "P": [
                    FakeSummaryDoc("mp-a", "P", 0.0, energy_above_hull=0.0, is_stable=True, icsd_ids=[2]),
                    FakeSummaryDoc("mp-b", "P", 0.1, energy_above_hull=0.1, is_stable=False, icsd_ids=[3]),
                ],
            }
        )
        table = find_competing_phases("Ba-P", only_stable=True, unique_formula=False, client=client)
        assert set(table["material_id"]) == {"mp-122", "mp-a"}

    def test_energy_above_hull_cutoff(self) -> None:
        client = FakeMPRester(
            docs_by_chemsys={
                "P": [
                    FakeSummaryDoc("mp-a", "P", 0.0, energy_above_hull=0.0, icsd_ids=[1]),
                    FakeSummaryDoc("mp-b", "P4", 0.1, energy_above_hull=0.05, icsd_ids=[2]),
                ]
            }
        )
        table = find_competing_phases("P", energy_above_hull_max=0.01, unique_formula=False, client=client)
        assert set(table["material_id"]) == {"mp-a"}

    def test_unique_formula_keeps_the_lowest_energy_entry(self) -> None:
        client = FakeMPRester(
            docs_by_chemsys={
                "Cd-P": [
                    FakeSummaryDoc("mp-high", "Cd3P2", -0.10, icsd_ids=[1]),
                    FakeSummaryDoc("mp-low", "Cd3P2", -0.20, icsd_ids=[2]),
                ]
            }
        )
        table = find_competing_phases("Cd-P", include_unary=False, client=client)
        assert list(table["material_id"]) == ["mp-low"]

    def test_a_failing_subsystem_stops_the_search(self, fake_mprester: FakeMPRester) -> None:
        """A missing subsystem lowers the hull, so it is an error rather than a warning."""
        fake_mprester.materials.summary.search = self._explode_on("Ba-P", fake_mprester)
        with pytest.raises(RuntimeError, match="Ba-P"):
            find_competing_phases("Ba-Cd-P", client=fake_mprester)

    def test_a_failing_subsystem_is_reported_when_partial_is_allowed(self, fake_mprester: FakeMPRester) -> None:
        fake_mprester.materials.summary.search = self._explode_on("Ba-P", fake_mprester)
        table = find_competing_phases("Ba-Cd-P", client=fake_mprester, allow_partial=True)
        assert "BaP2" not in set(table["composition"])
        assert "BaCd" in set(table["composition"])
        assert table.attrs["missing_subsystems"] == ["Ba-P"]

    @staticmethod
    def _explode_on(failing: str, client: FakeMPRester) -> object:
        def search(chemsys: str | None = None, **kwargs: object) -> list[object]:
            if chemsys == failing:
                raise RuntimeError("simulated Materials Project outage")
            return list(client.docs_by_chemsys.get(str(chemsys), []))

        return search

    def test_no_match_returns_an_empty_table_with_the_columns(self) -> None:
        table = find_competing_phases("Ba-Cd-P", client=FakeMPRester())
        assert table.empty
        assert list(table.columns) == COMPETING_PHASE_COLUMNS

    def test_an_unknown_element_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            find_competing_phases("Ba-Xx", client=FakeMPRester())

    def test_an_empty_chemical_system_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            find_competing_phases("   ", client=FakeMPRester())

    def test_the_client_can_also_come_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch, fake_mprester: FakeMPRester
    ) -> None:
        """No client passed: the key is read from the environment and a client is opened."""
        monkeypatch.setenv("MP_API_KEY", "env_key")
        monkeypatch.setattr(mp, "get_mprester", lambda api_key=None, **kwargs: fake_mprester)
        table = find_competing_phases("Ba-Cd-P")
        assert not table.empty
        assert fake_mprester.closed, "a client opened by the package must be closed again"

    def test_without_a_key_the_search_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MP_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="MP_API_KEY"):
            find_competing_phases("Ba-Cd-P")


class TestReducedFormulaColumns:
    """Formula reduction as it reaches the table.

    The five formula shapes of the original ``TestReduceFormula`` are checked here through
    the public search, and the parenthesised formula that the old regular-expression parser
    mislabelled is checked with them.
    """

    @pytest.mark.parametrize(
        ("formula_pretty", "expected_composition", "expected_natoms"),
        [
            ("BaCdP", "BaCdP", 3),
            ("Ba2P", "Ba2P", 3),
            ("Ba2Cd2P2", "BaCdP", 3),
            ("Ba", "Ba", 1),
            ("Ba2Cd4", "BaCd2", 3),
            ("Ba(CdP)2", "Ba(CdP)2", 5),
        ],
    )
    def test_composition_and_natoms(self, formula_pretty: str, expected_composition: str, expected_natoms: int) -> None:
        client = FakeMPRester(docs_by_chemsys={"Ba-Cd-P": [FakeSummaryDoc("mp-1", formula_pretty, icsd_ids=[1])]})
        table = find_competing_phases("Ba-Cd-P", include_unary=False, include_binary=False, client=client)
        assert table.loc[0, "formula_pretty"] == formula_pretty
        assert table.loc[0, "composition"] == expected_composition
        assert int(table.loc[0, "natoms"]) == expected_natoms

    def test_the_ternary_and_the_parenthesised_phase_stay_apart(self) -> None:
        """``Ba(CdP)2`` is BaCd2P2, not the ternary ``BaCdP`` the old parser recorded."""
        client = FakeMPRester(
            docs_by_chemsys={
                "Ba-Cd-P": [
                    FakeSummaryDoc("mp-8279", "Ba(CdP)2", -0.565, icsd_ids=[615814]),
                    FakeSummaryDoc("mp-other", "BaCdP", -0.40, icsd_ids=[1]),
                ]
            }
        )
        table = find_competing_phases("Ba-Cd-P", include_unary=False, include_binary=False, client=client)
        by_id = table.set_index("material_id")
        assert int(by_id.loc["mp-8279", "natoms"]) == 5
        assert int(by_id.loc["mp-other", "natoms"]) == 3
        assert by_id.loc["mp-8279", "composition"] != by_id.loc["mp-other", "composition"]


class TestDownloadStructures:
    """Downloading the structure behind each material id."""

    @staticmethod
    def _table() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "material_id": ["mp-122", "mp-527"],
                "composition": ["Ba", "BaCd"],
                "icsd_ids": ["icsd-77367|icsd-76156", "icsd-58642"],
            }
        )

    def test_files_are_written_and_named_after_the_material(self, tmp_path: Path) -> None:
        client = FakeMPRester(structures={"mp-122": cubic_structure("Ba"), "mp-527": cubic_structure("Ba", "Cd")})
        written = download_structures(self._table(), tmp_path, delay=0, client=client)

        assert set(written) == {"mp-122", "mp-527"}
        assert sorted(path.name for path in tmp_path.iterdir()) == ["mp-122_icsd-77367.vasp", "mp-527_icsd-58642.vasp"]
        assert all(Path(path).is_file() for path in written.values() if path)
        assert client.requested_ids == ["mp-122", "mp-527"]

    def test_the_written_file_is_a_readable_poscar(self, tmp_path: Path) -> None:
        from matdisc.common.io import read_structure

        client = FakeMPRester(structures={"mp-527": cubic_structure("Ba", "Cd")})
        written = download_structures(pd.DataFrame({"material_id": ["mp-527"]}), tmp_path, delay=0, client=client)
        structure = read_structure(str(written["mp-527"]))
        assert structure.composition.reduced_formula == "BaCd"

    def test_icsd_codes_can_be_left_out_of_the_name(self, tmp_path: Path) -> None:
        client = FakeMPRester()
        download_structures(self._table(), tmp_path, delay=0, include_icsd_in_name=False, client=client)
        assert sorted(path.name for path in tmp_path.iterdir()) == ["mp-122.vasp", "mp-527.vasp"]

    def test_existing_files_are_skipped(self, tmp_path: Path) -> None:
        client = FakeMPRester()
        download_structures(self._table(), tmp_path, delay=0, client=client)
        again = FakeMPRester()
        download_structures(self._table(), tmp_path, delay=0, client=again)
        assert again.requested_ids == []

    def test_a_failed_download_is_reported_as_none_and_the_rest_continue(self, tmp_path: Path) -> None:
        client = FakeMPRester(failing_ids=["mp-122"])
        written = download_structures(self._table(), tmp_path, delay=0, client=client)
        assert written["mp-122"] is None
        assert written["mp-527"] is not None

    def test_cif_output(self, tmp_path: Path) -> None:
        client = FakeMPRester()
        download_structures(self._table(), tmp_path, fmt="cif", delay=0, client=client)
        assert all(path.suffix == ".cif" for path in tmp_path.iterdir())

    def test_an_unknown_format_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown format"):
            download_structures(self._table(), tmp_path, fmt="xyzzy", delay=0, client=FakeMPRester())

    def test_a_table_without_material_id_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="material_id"):
            download_structures(pd.DataFrame({"composition": ["Ba"]}), tmp_path, delay=0, client=FakeMPRester())


class TestTableOnDisk:
    """Writing the harvested table out and reading it back.

    The original package wrote a second, differently shaped file for the hull stage. There is
    only one file now, and these tests pin that it survives a round trip unchanged and stays
    readable by the hull.
    """

    def test_csv_round_trip_preserves_the_schema(self, tmp_path: Path, fake_mprester: FakeMPRester) -> None:
        table = find_competing_phases("Ba-Cd-P", client=fake_mprester)
        path = tmp_path / "competing_phases.csv"
        table.to_csv(path, index=False)

        reloaded = pd.read_csv(path)
        assert list(reloaded.columns) == COMPETING_PHASE_COLUMNS
        assert list(reloaded["composition"]) == list(table["composition"])
        assert list(reloaded["natoms"]) == list(table["natoms"])

    def test_the_stage_writes_that_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from matdisc.pipeline import stages
        from matdisc.pipeline.config import CompetingConfig, PipelineConfig

        client = FakeMPRester(docs_by_chemsys=ba_cd_p_docs())
        monkeypatch.setattr(
            stages, "find_competing_phases", lambda **kwargs: find_competing_phases(client=client, **kwargs)
        )
        config = PipelineConfig(
            name="test",
            work_dir=str(tmp_path),
            competing=CompetingConfig(chemsys="Ba-Cd-P", download_structures=False),
        )
        result = stages.run_competing(config, output_dir=tmp_path)

        csv_path = Path(result.artifacts["csv"])
        assert csv_path.is_file()
        assert list(pd.read_csv(csv_path).columns) == COMPETING_PHASE_COLUMNS
        assert result.stats["n_phases"] > 0

    def test_a_supplied_table_is_reused_instead_of_searched(self, tmp_path: Path, example_competing_csv: Path) -> None:
        from matdisc.pipeline import stages
        from matdisc.pipeline.config import CompetingConfig, PipelineConfig

        config = PipelineConfig(
            name="test",
            work_dir=str(tmp_path),
            competing=CompetingConfig(
                chemsys="Ba-Cd-P", csv_file=str(example_competing_csv), download_structures=False
            ),
        )
        result = stages.run_competing(config, output_dir=tmp_path)
        assert result.status == "reused"
        assert result.stats["n_phases"] == 21


class TestHullHandoff:
    """The harvested table feeds the hull directly.

    The original package had to reshape the table before the hull could read it, and the
    reshaping renamed a column and divided an already per-atom energy a second time. The
    table now goes straight in.
    """

    def test_energies_are_not_divided_by_the_atom_count_again(self, fake_mprester: FakeMPRester) -> None:
        table = find_competing_phases("Ba-Cd-P", client=fake_mprester)
        ternary = table[table["material_id"] == "mp-8279"].iloc[0]
        assert ternary["natoms"] == 5
        assert ternary["formation_energy_per_atom"] == pytest.approx(-0.565)

    def test_the_table_goes_straight_into_the_hull(self, fake_mprester: FakeMPRester) -> None:
        competing = find_competing_phases("Ba-Cd-P", client=fake_mprester)
        candidates = pd.DataFrame(
            [("cand-1", "BaCdP", -0.90), ("cand-2", "BaCdP", -0.10)],
            columns=["id", "composition", "formation_energy_per_atom"],
        )
        result = compute_e_above_hull(candidates, competing, energy_column="formation_energy_per_atom")

        assert list(result["id"]) == ["cand-1", "cand-2"]
        assert bool(result.loc[0, "is_stable"]) is True
        assert bool(result.loc[1, "is_stable"]) is False
        assert result.loc[1, "e_above_hull"] > result.loc[0, "e_above_hull"]

    def test_candidates_can_be_merged_with_the_competing_phases_for_reporting(
        self, fake_mprester: FakeMPRester
    ) -> None:
        """The old helper's one job: one table carrying both sides, labelled by source."""
        competing = find_competing_phases("Ba-Cd-P", client=fake_mprester)
        candidates = pd.DataFrame([("cand-1", "BaCdP", -0.90)], columns=["id", "composition", "energy"])

        merged = pd.concat(
            [
                competing[["composition"]].assign(source="competing"),
                candidates[["composition"]].assign(source="candidate"),
            ],
            ignore_index=True,
        )
        assert set(merged["source"]) == {"competing", "candidate"}
        assert int((merged["source"] == "candidate").sum()) == 1


@pytest.mark.network
class TestAgainstTheRealApi:
    """The one test that needs a key and the network. Excluded by ``-m 'not network'``."""

    def test_small_real_search(self) -> None:
        import os

        if not os.environ.get("MP_API_KEY"):
            pytest.skip("MP_API_KEY is not set")

        table = find_competing_phases("Ba-Cd", include_higher_order=False)
        assert not table.empty
        assert list(table.columns) == COMPETING_PHASE_COLUMNS
        assert table["icsd_ids"].astype(bool).all()
