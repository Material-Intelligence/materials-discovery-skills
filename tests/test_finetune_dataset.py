"""Discovery of DFT systems for the fine-tuning dataset.

The conversion itself needs ``dpdata``, an optional extra. The directory walk that decides which
OUTCAR belongs to which system does not, and it is the part that has to agree with the layout the
DFT stage writes, so it is tested here on real `matdisc dft-inputs` output with stub OUTCARs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from matdisc.cli import main
from matdisc.finetune.dataset import find_system_outcars

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

STUB_OUTCAR = "this is not a real OUTCAR; only its presence is under test\n"


@pytest.fixture
def structure_dir(tmp_path: Path) -> Path:
    """Write two small structures for the DFT stage to consume.

    Args:
        tmp_path: pytest's temporary directory.

    Returns:
        The directory holding them.
    """
    source = tmp_path / "structures"
    source.mkdir()
    structure = Structure(Lattice.cubic(4.25), ["Ba", "Cd"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    structure.to(filename=str(source / "candidate_a.vasp"), fmt="poscar")
    structure.to(filename=str(source / "candidate_b.vasp"), fmt="poscar")
    return source


class TestTheLayoutTheDftStageWrites:
    """`matdisc dft-inputs` writes one flat directory per calculation; VASP drops OUTCAR in it."""

    def test_every_calculation_directory_is_found_as_one_system(self, structure_dir: Path, tmp_path: Path) -> None:
        outdir = tmp_path / "runs"
        assert main(["dft-inputs", "-s", str(structure_dir), "-o", str(outdir), "--kind", "relax"]) == 0

        calculations = outdir / "calculations"
        assert sorted(path.name for path in calculations.iterdir()) == ["candidate_a", "candidate_b"]

        # VASP writes OUTCAR into the directory it is run in, beside the inputs.
        for directory in sorted(calculations.iterdir()):
            (directory / "OUTCAR").write_text(STUB_OUTCAR, encoding="utf-8")

        found = find_system_outcars(calculations)

        assert sorted(found) == ["candidate_a", "candidate_b"]
        for name, outcars in found.items():
            assert outcars == [calculations / name / "OUTCAR"]

    def test_a_calculation_that_has_not_run_yet_is_reported_empty(self, structure_dir: Path, tmp_path: Path) -> None:
        outdir = tmp_path / "runs"
        assert main(["dft-inputs", "-s", str(structure_dir), "-o", str(outdir), "--kind", "relax"]) == 0

        calculations = outdir / "calculations"
        (calculations / "candidate_a" / "OUTCAR").write_text(STUB_OUTCAR, encoding="utf-8")

        found = find_system_outcars(calculations)

        assert found["candidate_a"]
        assert found["candidate_b"] == []

    def test_a_missing_root_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            find_system_outcars(tmp_path / "absent")


class TestNestedLayouts:
    """Step-per-directory trees are still supported, but only when asked for."""

    def test_step_directories_are_collected_in_order(self, tmp_path: Path) -> None:
        for step in ("S0", "S1", "S2"):
            directory = tmp_path / "Output" / "system_one.poscar" / "relax" / step
            directory.mkdir(parents=True)
            (directory / "OUTCAR").write_text(STUB_OUTCAR, encoding="utf-8")

        found = find_system_outcars(tmp_path, output_subdir="Output", relax_subdir="relax")

        assert list(found) == ["system_one"], "a trailing .poscar is stripped from the system name"
        assert [path.parent.name for path in found["system_one"]] == ["S0", "S1", "S2"]

    def test_exclude_patterns_drop_a_step(self, tmp_path: Path) -> None:
        for step in ("S0", "S1"):
            directory = tmp_path / "Output" / "system_one" / "relax" / step
            directory.mkdir(parents=True)
            (directory / "OUTCAR").write_text(STUB_OUTCAR, encoding="utf-8")

        found = find_system_outcars(tmp_path, output_subdir="Output", relax_subdir="relax", exclude_patterns=["S0"])

        assert [path.parent.name for path in found["system_one"]] == ["S1"]

    def test_the_nested_layout_is_not_the_default(self, tmp_path: Path) -> None:
        directory = tmp_path / "Output" / "system_one" / "relax" / "S0"
        directory.mkdir(parents=True)
        (directory / "OUTCAR").write_text(STUB_OUTCAR, encoding="utf-8")

        # Without the subdirectory names, "Output" is the only system and it holds no OUTCAR.
        assert find_system_outcars(tmp_path) == {"Output": []}
