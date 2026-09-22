"""The stage layer: the code that turns a configuration into files on disk.

These tests run the real stages -- relaxation, convex hull, phonon spectrum, stage
sequencing -- with ASE's effective-medium calculator. EMT needs no checkpoint, no GPU and no
network, and a four-atom copper cell relaxes in well under a second, so the stage that
produces this package's headline result is exercised on every run of the suite rather than
only on a cluster.

What is asserted here is what a reader of the output would check: that the structures written
into ``stable/`` are the geometries the reported energies belong to, that a hull is never
reported when a competing phase is missing from it, and that a phonon verdict carries whether
the geometry it was computed on was stationary.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from ase.calculators.emt import EMT
from pymatgen.core import SETTINGS, Lattice, Structure

from matdisc.cli import main
from matdisc.dft.outputs import check_calculation_status, collect_results
from matdisc.dft.slurm import SlurmSettings, render_slurm_script
from matdisc.pipeline.config import (
    DftConfig,
    GenerationConfig,
    PhononConfig,
    PipelineConfig,
    ScreeningConfig,
)
from matdisc.pipeline.data_interface import HULL_CSV, PHONON_CSV, write_structures
from matdisc.pipeline.run import run_pipeline
from matdisc.pipeline.stages import PHONON_COLUMNS, StageError, run_phonons, run_screening
from matdisc.screening.phonons import check_dynamical_stability
from matdisc.screening.relax import relax

# EMT is parameterised for copper; the hull below is therefore the (trivial) unary copper
# hull, which is enough to exercise every decision the screening stage makes.
LATTICE_PARAMETER = 3.61
"""Conventional fcc lattice parameter of copper in A, close to the EMT minimum."""

SCREENING_TOLERANCE = 1e-3
"""Stability tolerance for these tests, in eV/atom.

Two relaxations of the same phase stop at slightly different points, so the hull and the
candidate differ by a few times 1e-6 eV/atom; the package default of 1e-6 would read that
numerical difference as instability. 1e-3 eV/atom is the scale a screening run actually uses.
"""


def fcc_copper(a: float = LATTICE_PARAMETER) -> Structure:
    """Return a conventional four-atom fcc copper cell.

    Args:
        a: Cubic lattice parameter in A.

    Returns:
        The structure.
    """
    return Structure(
        Lattice.cubic(a),
        ["Cu"] * 4,
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
    )


def emt_config(work_dir: Path, **screening: object) -> PipelineConfig:
    """Return a configuration whose screening and phonon stages use EMT.

    Args:
        work_dir: Working directory of the run.
        **screening: Fields of :class:`~matdisc.pipeline.config.ScreeningConfig` to override.

    Returns:
        The configuration.
    """
    settings: dict[str, object] = {
        "calculator": "emt",
        "fmax": 0.05,
        "max_steps": 200,
        "tolerance": SCREENING_TOLERANCE,
    }
    settings.update(screening)
    return PipelineConfig(name="emt", work_dir=str(work_dir), screening=ScreeningConfig(**settings))


@pytest.fixture
def candidates(tmp_path: Path) -> Path:
    """Write one candidate whose cell is 2.5 % too large.

    Args:
        tmp_path: pytest's temporary directory.

    Returns:
        The directory holding it.
    """
    directory = tmp_path / "candidates"
    write_structures({"cand-1": fcc_copper(LATTICE_PARAMETER * 1.025)}, directory)
    return directory


@pytest.fixture
def competing(tmp_path: Path) -> Path:
    """Write the competing copper phase.

    Args:
        tmp_path: pytest's temporary directory.

    Returns:
        The directory holding it.
    """
    directory = tmp_path / "competing"
    write_structures({"Cu": fcc_copper()}, directory)
    return directory


class TestRunScreening:
    """What the screening stage writes, counts and refuses."""

    def test_the_stable_structure_is_the_relaxed_geometry(
        self, tmp_path: Path, candidates: Path, competing: Path
    ) -> None:
        """stable/ must hold the cell the reported energy belongs to, not the input cell."""
        config = emt_config(tmp_path / "work")
        result = run_screening(
            config,
            structure_dir=candidates,
            competing_structure_dir=competing,
            output_dir=tmp_path / "work",
        )

        assert result.stats["n_stable"] == 1
        written = Structure.from_file(tmp_path / "work" / "stable" / "cand-1.vasp")
        relaxed = Structure.from_file(tmp_path / "work" / "relaxed" / "cand-1" / "relaxed.vasp")
        assert written == relaxed
        # The input cell was strained by 2.5 %, so a copy of the input would be caught here.
        assert written.lattice.a == pytest.approx(relaxed.lattice.a)
        assert written.lattice.a != pytest.approx(LATTICE_PARAMETER * 1.025, abs=1e-3)

    def test_the_hull_table_records_what_the_hull_rests_on(
        self, tmp_path: Path, candidates: Path, competing: Path
    ) -> None:
        config = emt_config(tmp_path / "work")
        result = run_screening(
            config,
            structure_dir=candidates,
            competing_structure_dir=competing,
            output_dir=tmp_path / "work",
        )

        hull = pd.read_csv(tmp_path / "work" / HULL_CSV)
        assert set(hull["n_competing_used"]) == {1}
        assert set(hull["n_competing_dropped"]) == {0}
        assert set(hull["calculator"]) == {"emt"}
        assert result.stats["n_competing_requested"] == 1
        assert result.stats["calculator_head"] is None

    def test_a_competing_phase_that_cannot_be_relaxed_stops_the_run(
        self, tmp_path: Path, candidates: Path, competing: Path
    ) -> None:
        """A dropped competing phase lowers the hull, so the verdict must not be reported."""
        # EMT has no parameters for silicon, so this relaxation raises and its energy is NaN.
        write_structures({"Si": Structure(Lattice.cubic(5.43), ["Si"], [[0.0, 0.0, 0.0]])}, competing)

        config = emt_config(tmp_path / "work")
        with pytest.raises(StageError, match="Si"):
            run_screening(
                config,
                structure_dir=candidates,
                competing_structure_dir=competing,
                output_dir=tmp_path / "work",
            )

    def test_a_dropped_competing_phase_is_counted_when_it_is_allowed(
        self, tmp_path: Path, candidates: Path, competing: Path
    ) -> None:
        write_structures({"Si": Structure(Lattice.cubic(5.43), ["Si"], [[0.0, 0.0, 0.0]])}, competing)

        config = emt_config(tmp_path / "work", allow_incomplete_hull=True)
        result = run_screening(
            config,
            structure_dir=candidates,
            competing_structure_dir=competing,
            output_dir=tmp_path / "work",
        )

        assert result.stats["n_competing_requested"] == 2
        assert result.stats["n_competing_used"] == 1
        assert result.stats["n_competing_dropped"] == 1

    def test_an_unconverged_candidate_is_not_reported_as_stable(self, tmp_path: Path, candidates: Path) -> None:
        """A step-capped relaxation leaves an energy that belongs to no minimum."""
        table = tmp_path / "competing.csv"
        pd.DataFrame([{"material_id": "ref-Cu", "composition": "Cu", "energy_per_atom": 1.0}]).to_csv(
            table, index=False
        )

        config = emt_config(
            tmp_path / "work",
            competing_energy_source="table",
            fmax=1e-6,
            max_steps=1,
        )
        result = run_screening(
            config,
            structure_dir=candidates,
            competing_csv=table,
            output_dir=tmp_path / "work",
        )

        assert result.stats["n_on_hull"] == 1
        assert result.stats["n_unconverged"] == 1
        assert result.stats["n_stable"] == 0
        assert not list((tmp_path / "work" / "stable").glob("*.vasp"))

    def test_the_screen_subcommand_runs_the_stage(self, tmp_path: Path, candidates: Path, competing: Path) -> None:
        """The relax-and-hull step is reachable in one command, without a YAML file."""
        status = main(
            [
                "screen",
                "-s",
                str(candidates),
                "--competing-structures",
                str(competing),
                "-o",
                str(tmp_path / "work"),
                "--calculator",
                "emt",
                "--fmax",
                "0.05",
                "--tolerance",
                str(SCREENING_TOLERANCE),
            ]
        )
        assert status == 0
        assert (tmp_path / "work" / HULL_CSV).is_file()
        assert (tmp_path / "work" / "stable" / "cand-1.vasp").is_file()


class TestRunPhonons:
    """The dynamical-stability stage."""

    @pytest.fixture
    def relaxed_copper_dir(self, tmp_path: Path) -> Path:
        """Relax fcc copper with EMT and write it out, so the spectrum starts stationary.

        Args:
            tmp_path: pytest's temporary directory.

        Returns:
            The directory holding the relaxed structure.
        """
        structure, _ = relax(fcc_copper(), EMT(), fmax=1e-4, max_steps=200)
        directory = tmp_path / "survivors"
        write_structures({"Cu": structure}, directory)
        return directory

    def test_relaxed_copper_is_dynamically_stable(self, tmp_path: Path, relaxed_copper_dir: Path) -> None:
        config = PipelineConfig(
            name="phonons",
            work_dir=str(tmp_path / "work"),
            screening=ScreeningConfig(calculator="emt"),
            phonons=PhononConfig(supercell=[3, 3, 3], plot=False, fmax=1e-4, max_steps=200),
        )
        result = run_phonons(config, relaxed_copper_dir, output_dir=tmp_path / "work")

        summary = pd.read_csv(tmp_path / "work" / PHONON_CSV)
        assert list(summary.columns) == PHONON_COLUMNS
        assert bool(summary.loc[0, "success"])
        assert bool(summary.loc[0, "is_stable"])
        assert bool(summary.loc[0, "relaxation_converged"])
        assert result.stats["n_stable"] == 1

    def test_the_survivor_is_the_geometry_the_spectrum_was_computed_on(
        self, tmp_path: Path, relaxed_copper_dir: Path
    ) -> None:
        config = PipelineConfig(
            name="phonons",
            work_dir=str(tmp_path / "work"),
            screening=ScreeningConfig(calculator="emt"),
            phonons=PhononConfig(supercell=[3, 3, 3], plot=False, fmax=1e-4, max_steps=200),
        )
        run_phonons(config, relaxed_copper_dir, output_dir=tmp_path / "work")

        survivor = Structure.from_file(tmp_path / "work" / "stable" / "Cu.vasp")
        displaced = Structure.from_file(tmp_path / "work" / "Cu" / "relaxed.vasp")
        assert survivor == displaced


class TestCheckDynamicalStability:
    """The pre-relaxation gate in front of the spectrum."""

    def test_an_unconverged_pre_relaxation_refuses_the_spectrum(self, tmp_path: Path) -> None:
        """Imaginary modes of a non-stationary cell say nothing about the material."""
        strained = fcc_copper(LATTICE_PARAMETER * 1.08)
        strained.translate_sites([0], [0.03, 0.02, 0.0])

        result = check_dynamical_stability(
            strained,
            EMT(),
            output_dir=tmp_path / "phonon",
            supercell=(1, 1, 1),
            fmax=1e-8,
            max_steps=1,
            plot=False,
        )

        assert result["success"] is False
        assert result["relaxation_converged"] is False
        assert result["is_stable"] is False
        assert "pre-relaxation did not converge" in result["error_message"]
        assert not (tmp_path / "phonon" / "band_structure.dat").exists()

    def test_the_gate_can_be_opened_deliberately(self, tmp_path: Path) -> None:
        strained = fcc_copper(LATTICE_PARAMETER * 1.08)
        strained.translate_sites([0], [0.03, 0.02, 0.0])

        result = check_dynamical_stability(
            strained,
            EMT(),
            output_dir=tmp_path / "phonon",
            supercell=(1, 1, 1),
            fmax=1e-8,
            max_steps=1,
            require_relaxation_converged=False,
            plot=False,
        )

        assert result["success"] is True
        assert result["relaxation_converged"] is False


class TestRunPipeline:
    """Stage sequencing."""

    @pytest.fixture(autouse=True)
    def _no_pseudopotentials(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Make the DFT stage write a POTCAR.spec instead of looking for pseudopotentials.

        Args:
            monkeypatch: pytest's monkeypatch fixture.
            tmp_path: pytest's temporary directory.
        """
        empty = tmp_path / "no_psp"
        empty.mkdir()
        monkeypatch.setitem(SETTINGS, "PMG_VASP_PSP_DIR", str(empty))
        monkeypatch.setenv("PMG_VASP_PSP_DIR", str(empty))

    def test_the_run_stops_after_the_dft_stage(self, tmp_path: Path, candidates: Path) -> None:
        """Fine-tuning cannot start on calculations that have not run, so the run stops."""
        config = PipelineConfig(
            name="stop",
            work_dir=str(tmp_path / "work"),
            stages=["generation", "dft", "finetune"],
            generation=GenerationConfig(structure_dir=str(candidates)),
            dft=DftConfig(write_slurm=True, submit=False),
        )

        outcome = run_pipeline(config)

        assert outcome.status == "stopped"
        assert outcome.stopped_at == "dft"
        assert list(outcome.results) == ["generation", "dft"]
        assert "finetune" not in outcome.results
        assert (tmp_path / "work" / "03_dft" / "calculations" / "cand-1" / "INCAR").is_file()

    def test_the_stages_run_in_order(self, tmp_path: Path, candidates: Path, competing: Path) -> None:
        config = emt_config(tmp_path / "work")
        config.stages = ["generation", "screening"]
        config.generation = GenerationConfig(structure_dir=str(candidates))
        config.competing.structure_dir = str(competing)
        config.__post_init__()

        outcome = run_pipeline(config)

        assert outcome.status == "completed"
        assert list(outcome.results) == ["generation", "screening"]
        assert outcome.results["screening"].stats["n_stable"] == 1


class TestSlurmScript:
    """The submission script is text a cluster has to accept verbatim."""

    def test_the_script_reads_as_expected(self) -> None:
        script = render_slurm_script(
            SlurmSettings(
                job_name="bacdp",
                partition="regular",
                nodes=2,
                ntasks=128,
                walltime_minutes=90,
                account="m3342",
                modules=["vasp/6.4.3"],
                exports=["OMP_NUM_THREADS=1"],
            )
        )

        assert script.startswith("#!/bin/bash\n")
        assert "#SBATCH --job-name=bacdp" in script
        assert "#SBATCH --partition=regular" in script
        assert "#SBATCH --account=m3342" in script
        assert "#SBATCH --nodes=2" in script
        assert "#SBATCH --ntasks=128" in script
        assert "#SBATCH --time=01:30:00" in script
        assert "module load vasp/6.4.3" in script
        assert "export OMP_NUM_THREADS=1" in script
        assert 'if [ -z "${VASP_CMD:-}" ]; then' in script
        assert script.rstrip("\n").endswith('mpirun -np 128 "$VASP_CMD"')


class TestCalculationSweep:
    """Reading a tree of calculation directories that a cluster left behind."""

    @pytest.fixture
    def tree(self, tmp_path: Path) -> Path:
        """Build a stub tree of three calculation directories in three states.

        Args:
            tmp_path: pytest's temporary directory.

        Returns:
            The root of the tree.
        """
        root = tmp_path / "calculations"
        for name, tail in (
            ("done", " General timing and accounting informations for this job\n"),
            ("broken", " Error EDDDAV: Call to ZHEGV failed\n"),
            ("waiting", None),
        ):
            directory = root / name
            directory.mkdir(parents=True)
            (directory / "INCAR").write_text("ISMEAR = 0\n", encoding="utf-8")
            if tail is not None:
                (directory / "OUTCAR").write_text(tail, encoding="utf-8")
        return root

    def test_the_states_are_read_from_the_outcar(self, tree: Path) -> None:
        assert check_calculation_status(tree) == {
            "broken": "failed",
            "done": "completed",
            "waiting": "pending",
        }

    def test_a_tree_with_no_parseable_output_collects_nothing(self, tree: Path) -> None:
        table = collect_results(tree)
        assert table.empty
        assert "energy_per_atom" in table.columns
