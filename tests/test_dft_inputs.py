"""VASP input generation.

No POTCAR file ships with this repository and none may be redistributed. pymatgen looks them
up in ``PMG_VASP_PSP_DIR``; when that directory is not configured, the package writes a
``POTCAR.spec`` file naming the pseudopotentials the run needs, so inputs can be prepared and
inspected on a machine without a VASP licence. Every test below forces that state, so the
suite behaves the same whether or not the machine running it has pseudopotentials.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pymatgen.core import SETTINGS, Lattice, Structure

from matdisc.dft.inputs import (
    DEFAULT_EDIFF,
    DEFAULT_EDIFFG,
    DEFAULT_KSPACING,
    INPUT_SET_KINDS,
    KINDS_NEEDING_CHGCAR,
    build_input_set,
    potcar_symbols,
    write_batch_inputs,
    write_vasp_inputs,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture
def structure() -> Structure:
    """Return a small cubic BaCd structure.

    Returns:
        The structure.
    """
    return Structure(Lattice.cubic(4.25), ["Ba", "Cd"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


@pytest.fixture(autouse=True)
def _no_pseudopotentials(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Point pymatgen at an empty pseudopotential directory for the whole module.

    Args:
        monkeypatch: pytest's monkeypatch fixture.
        tmp_path_factory: Used to create the empty directory.
    """
    empty = tmp_path_factory.mktemp("no_psp")
    monkeypatch.setitem(SETTINGS, "PMG_VASP_PSP_DIR", str(empty))
    monkeypatch.setenv("PMG_VASP_PSP_DIR", str(empty))


class TestWriteVaspInputs:
    """The files one calculation directory receives."""

    def test_incar_poscar_and_potcar_spec_are_written(self, structure: Structure, tmp_path: Path) -> None:
        result = write_vasp_inputs(structure, tmp_path / "relax", kind="relax")

        written = sorted(path.name for path in (tmp_path / "relax").iterdir())
        assert written == ["INCAR", "POSCAR", "POTCAR.spec"]
        assert result["files"] == written
        assert result["kind"] == "relax"
        assert Path(result["outdir"]) == (tmp_path / "relax").resolve()

    def test_the_potcar_spec_lists_the_required_symbols(self, structure: Structure, tmp_path: Path) -> None:
        result = write_vasp_inputs(structure, tmp_path / "relax")

        assert result["potcar"] == "POTCAR.spec"
        assert result["potcar_symbols"] == ["Ba_sv", "Cd"]
        spec = (tmp_path / "relax" / "POTCAR.spec").read_text().split()
        assert spec == ["Ba_sv", "Cd"]

    def test_no_potcar_file_is_produced(self, structure: Structure, tmp_path: Path) -> None:
        write_vasp_inputs(structure, tmp_path / "relax")
        assert not (tmp_path / "relax" / "POTCAR").exists()

    def test_kspacing_replaces_the_kpoints_file(self, structure: Structure, tmp_path: Path) -> None:
        result = write_vasp_inputs(structure, tmp_path / "relax", kspacing=DEFAULT_KSPACING)

        assert "KPOINTS" not in result["files"]
        assert result["incar"]["KSPACING"] == pytest.approx(DEFAULT_KSPACING)

    def test_a_kpoints_file_is_written_when_kspacing_is_off(self, structure: Structure, tmp_path: Path) -> None:
        result = write_vasp_inputs(structure, tmp_path / "relax", kspacing=None)

        assert "KPOINTS" in result["files"]
        assert "KSPACING" not in result["incar"]
        assert (tmp_path / "relax" / "KPOINTS").read_text().strip()

    def test_the_poscar_round_trips(self, structure: Structure, tmp_path: Path) -> None:
        from matdisc.common.io import read_structure

        write_vasp_inputs(structure, tmp_path / "relax")
        written = read_structure(tmp_path / "relax" / "POSCAR")
        assert written.composition.reduced_formula == structure.composition.reduced_formula
        assert len(written) == len(structure)

    def test_the_output_directory_is_created(self, structure: Structure, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "relax"
        write_vasp_inputs(structure, target)
        assert (target / "INCAR").is_file()


class TestIncarSettings:
    """The convergence settings carried over from the original templates."""

    def test_the_ported_settings_are_present(self, structure: Structure, tmp_path: Path) -> None:
        incar = write_vasp_inputs(structure, tmp_path / "relax", kind="relax")["incar"]

        assert incar["EDIFF"] == pytest.approx(DEFAULT_EDIFF)
        assert incar["EDIFFG"] == pytest.approx(DEFAULT_EDIFFG)
        assert incar["KSPACING"] == pytest.approx(DEFAULT_KSPACING)

    def test_runs_are_not_spin_polarised_by_default(self, structure: Structure, tmp_path: Path) -> None:
        incar = write_vasp_inputs(structure, tmp_path / "relax")["incar"]

        assert incar["ISPIN"] == 1
        assert "MAGMOM" not in incar

    def test_magnetic_keeps_the_input_sets_spin_settings(self, structure: Structure, tmp_path: Path) -> None:
        incar = write_vasp_inputs(structure, tmp_path / "relax", magnetic=True)["incar"]
        assert incar.get("ISPIN", 2) == 2

    def test_user_settings_win(self, structure: Structure, tmp_path: Path) -> None:
        incar = write_vasp_inputs(
            structure,
            tmp_path / "relax",
            user_incar_settings={"EDIFF": 1e-7, "NSW": 0, "ENCUT": 600},
        )["incar"]

        assert incar["EDIFF"] == pytest.approx(1e-7)
        assert incar["NSW"] == 0
        assert incar["ENCUT"] == pytest.approx(600)

    def test_a_user_setting_of_none_removes_the_tag(self, structure: Structure, tmp_path: Path) -> None:
        incar = write_vasp_inputs(structure, tmp_path / "relax", user_incar_settings={"LORBIT": None})["incar"]
        assert "LORBIT" not in incar


class TestInputSetKinds:
    """The five kinds of calculation directory."""

    @pytest.mark.parametrize("kind", INPUT_SET_KINDS)
    def test_every_kind_writes_an_incar_and_a_poscar(self, structure: Structure, tmp_path: Path, kind: str) -> None:
        result = write_vasp_inputs(structure, tmp_path / kind, kind=kind)
        assert "INCAR" in result["files"]
        assert "POSCAR" in result["files"]

    @pytest.mark.parametrize("kind", ["hse-relax", "hse-static"])
    def test_both_hybrid_kinds_switch_on_the_hybrid_functional(
        self, structure: Structure, tmp_path: Path, kind: str
    ) -> None:
        incar = write_vasp_inputs(structure, tmp_path / kind, kind=kind)["incar"]

        assert incar["LHFCALC"] is True
        assert incar["HFSCREEN"] == pytest.approx(0.2)
        assert incar["EDIFF"] == pytest.approx(DEFAULT_EDIFF)

    def test_hse_relax_moves_the_ions_and_hse_static_does_not(self, structure: Structure, tmp_path: Path) -> None:
        relaxation = write_vasp_inputs(structure, tmp_path / "hse-relax", kind="hse-relax")["incar"]
        single_point = write_vasp_inputs(structure, tmp_path / "hse-static", kind="hse-static")["incar"]

        assert relaxation["NSW"] > 0
        assert relaxation["EDIFFG"] == pytest.approx(DEFAULT_EDIFFG)

        assert single_point["NSW"] == 0
        # EDIFFG is an ionic criterion; a single point has no ionic steps to converge.
        assert "EDIFFG" not in single_point

    @pytest.mark.parametrize("kind", KINDS_NEEDING_CHGCAR)
    def test_a_continuation_run_reports_the_charge_density_it_needs(
        self, structure: Structure, tmp_path: Path, kind: str
    ) -> None:
        result = write_vasp_inputs(structure, tmp_path / kind, kind=kind)

        assert result["requires"] == ["CHGCAR"]
        assert result["incar"]["ICHARG"] in (1, 11)

    @pytest.mark.parametrize("kind", ["relax", "static"])
    def test_a_self_contained_run_requires_nothing(self, structure: Structure, tmp_path: Path, kind: str) -> None:
        result = write_vasp_inputs(structure, tmp_path / kind, kind=kind)

        assert result["requires"] == []
        assert "ICHARG" not in result["incar"]

    def test_every_kind_is_classified_as_continuation_or_not(self) -> None:
        # Nothing may be left out of the two lists, or a kind could quietly need a CHGCAR
        # without saying so.
        assert set(KINDS_NEEDING_CHGCAR) <= set(INPUT_SET_KINDS)
        assert set(INPUT_SET_KINDS) - set(KINDS_NEEDING_CHGCAR) == {"relax", "static"}

    @pytest.mark.parametrize("kind", ["band", "hse-static"])
    def test_an_explicit_k_point_list_replaces_the_kspacing_grid(
        self, structure: Structure, tmp_path: Path, kind: str
    ) -> None:
        result = write_vasp_inputs(structure, tmp_path / kind, kind=kind)

        assert "KSPACING" not in result["incar"]
        assert "KPOINTS" in result["files"]

    def test_a_band_run_follows_a_k_path(self, structure: Structure, tmp_path: Path) -> None:
        write_vasp_inputs(structure, tmp_path / "band", kind="band")
        assert "line" in (tmp_path / "band" / "KPOINTS").read_text().lower()

    def test_no_hubbard_u_is_written_for_a_phosphide(self, tmp_path: Path) -> None:
        # pymatgen's Materials Project sets apply a U only to oxides and fluorides, so the
        # original template's `pbe ldau` request has no INCAR equivalent for this chemistry.
        phosphide = Structure(
            Lattice.cubic(6.0),
            ["Ba", "Cd", "P", "P"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
        )
        incar = write_vasp_inputs(phosphide, tmp_path / "relax", kind="relax")["incar"]

        assert "LDAU" not in incar
        assert "LDAUU" not in incar
        assert "LDAUJ" not in incar

    def test_an_unknown_kind_is_rejected(self, structure: Structure, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown input set kind"):
            write_vasp_inputs(structure, tmp_path / "nope", kind="nope")

    def test_potcar_symbols_need_no_potcar_files(self, structure: Structure) -> None:
        assert potcar_symbols(build_input_set(structure, "relax")) == ["Ba_sv", "Cd"]


class TestWriteBatchInputs:
    """One calculation directory per structure file."""

    def test_one_directory_per_structure(self, structure: Structure, tmp_path: Path) -> None:
        source = tmp_path / "structures"
        source.mkdir()
        structure.to(filename=str(source / "BaCd.vasp"), fmt="poscar")
        structure.to(filename=str(source / "BaCd_copy.cif"), fmt="cif")

        results = write_batch_inputs(source, tmp_path / "calcs")

        assert sorted(results) == ["BaCd", "BaCd_copy"]
        for name in results:
            assert (tmp_path / "calcs" / name / "INCAR").is_file()

    def test_structures_already_in_hand_are_written_without_re_reading_files(
        self, structure: Structure, tmp_path: Path
    ) -> None:
        """The pipeline stage passes the structures it validated, so nothing is parsed twice."""
        results = write_batch_inputs({"cand-1": structure}, tmp_path / "calcs")

        assert sorted(results) == ["cand-1"]
        assert (tmp_path / "calcs" / "cand-1" / "POSCAR").is_file()

    def test_an_unreadable_file_is_skipped(self, structure: Structure, tmp_path: Path) -> None:
        source = tmp_path / "structures"
        source.mkdir()
        structure.to(filename=str(source / "good.vasp"), fmt="poscar")
        (source / "broken.vasp").write_text("this is not a POSCAR\n")

        results = write_batch_inputs(source, tmp_path / "calcs")
        assert sorted(results) == ["good"]

    def test_a_missing_path_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            write_batch_inputs(tmp_path / "absent", tmp_path / "calcs")

    def test_a_single_file_is_accepted(self, structure: Structure, tmp_path: Path) -> None:
        # ``matdisc dft-inputs -s`` is documented as "structure file or directory", the same as
        # ``matdisc cluster`` and ``matdisc phonons``. A single file used to raise here.
        source = tmp_path / "BaCd.vasp"
        structure.to(filename=str(source), fmt="poscar")

        results = write_batch_inputs(source, tmp_path / "calcs")

        assert sorted(results) == ["BaCd"]
        assert (tmp_path / "calcs" / "BaCd" / "INCAR").is_file()

    def test_a_single_file_is_not_filtered_by_the_glob_patterns(self, structure: Structure, tmp_path: Path) -> None:
        # Naming a file is explicit, so it is read even though no default pattern matches it.
        source = tmp_path / "relaxed.json"
        structure.to(filename=str(source), fmt="json")
        assert not any(source.match(pattern) for pattern in ("*.cif", "*.vasp", "POSCAR*", "CONTCAR*"))

        results = write_batch_inputs(source, tmp_path / "calcs")
        assert sorted(results) == ["relaxed"]


class TestWithRealPseudopotentials:
    """The POTCAR branch, which only a machine holding VASP pseudopotentials can run."""

    def test_a_real_potcar_is_written_when_available(
        self, structure: Structure, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        configured = os.environ.get("PMG_VASP_PSP_DIR_REAL")
        if not configured:
            pytest.skip("set PMG_VASP_PSP_DIR_REAL to a VASP pseudopotential directory to run this")
        monkeypatch.setitem(SETTINGS, "PMG_VASP_PSP_DIR", configured)

        result = write_vasp_inputs(structure, tmp_path / "relax")
        assert result["potcar"] == "POTCAR"
        assert (tmp_path / "relax" / "POTCAR").is_file()
