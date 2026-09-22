"""The seven pipeline stages, each a function over the configuration.

Every stage takes the :class:`~matdisc.pipeline.config.PipelineConfig` and its input paths,
writes into its own directory under the working directory, and returns a :class:`StageResult`
recording what it produced. :mod:`matdisc.pipeline.run` sequences them and resolves each
stage's input from the previous stage's artifacts.

Failure policy
--------------
A stage that cannot do its work raises :class:`StageError`. That covers an optional backend
that is not installed, an input directory that is empty, and a checkpoint that was never
produced. No stage falls back to a reduced behaviour, writes placeholder files or logs a
fallback it does not implement: a stage either does what its name says or stops the run with
a message naming what is missing.

Energy scale
------------
The screening stage relaxes the candidates and, by default, the competing phases with the
same calculator, so the convex hull is built from one set of energies. Reading competing
energies from the harvested Materials Project table instead is possible but has to be asked
for; see :class:`~matdisc.pipeline.config.ScreeningConfig`.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from monty.json import MSONable
from pymatgen.core import Structure

from matdisc.clustering.descriptors import MODEL_PATH_ENV_VAR, compute_descriptor_set, load_descriptor_model
from matdisc.clustering.sampling import select_representative_structures, write_selected_structures
from matdisc.common.calculators import calculator_head, load_calculator
from matdisc.common.logging import get_logger
from matdisc.competing.download import download_structures
from matdisc.competing.search import find_competing_phases
from matdisc.dft.inputs import write_batch_inputs
from matdisc.dft.slurm import settings_from_mapping, submit_directory, write_slurm_script
from matdisc.finetune.dataset import convert_dft_to_deepmd, get_type_map, split_train_val
from matdisc.finetune.train import (
    build_dpa3_config,
    finetune,
    resolve_pretrained_model,
    write_dpa3_config,
    write_finetune_slurm_script,
)
from matdisc.generation.mattergen import generate_many
from matdisc.pipeline.config import PipelineConfig
from matdisc.pipeline.data_interface import (
    CANDIDATES_CSV,
    COMPETING_CSV,
    COMPETING_RELAXED_CSV,
    HULL_CSV,
    PHONON_CSV,
    SELECTED_SUBDIR,
    STABLE_SUBDIR,
    STRUCTURES_SUBDIR,
    load_structure_input,
    write_structures,
)
from matdisc.screening.hull import compute_e_above_hull, split_usable_rows
from matdisc.screening.phonons import check_dynamical_stability
from matdisc.screening.relax import relax_many

LOGGER = get_logger(__name__)

STAGE_STATUSES = ("completed", "prepared", "reused", "empty")
"""What a :class:`StageResult` reports.

``completed`` -- the stage did its work. ``prepared`` -- it wrote inputs for a job that must
be run elsewhere (a DFT submission, a training run). ``reused`` -- it used files the
configuration pointed at instead of producing them. ``empty`` -- it ran and there was
nothing to do, for example a phonon check with no surviving candidate.
"""

PHONON_COLUMNS = [
    "id",
    "success",
    "min_frequency",
    "is_stable",
    "relaxation_converged",
    "output_dir",
    "error",
]
"""Columns of the phonon summary table.

``relaxation_converged`` says whether the geometry the spectrum was computed on was
stationary; without it an imaginary mode cannot be told from leftover forces.
"""

__all__ = [
    "PHONON_COLUMNS",
    "STAGE_STATUSES",
    "StageError",
    "StageResult",
    "run_clustering",
    "run_competing",
    "run_dft",
    "run_finetune",
    "run_generation",
    "run_phonons",
    "run_screening",
]


class StageError(RuntimeError):
    """A stage could not do its work.

    Raised when a backend is missing, an input is absent or empty, or a step the stage
    promises failed. The message names what is missing and how to supply it.
    """


@dataclass
class StageResult(MSONable):
    """What one stage produced.

    Attributes:
        stage: The stage name.
        status: One of :data:`STAGE_STATUSES`.
        output_dir: Directory the stage wrote into.
        stats: Counts worth reporting, for example ``n_structures``.
        artifacts: Paths other stages read, keyed by role (``structure_dir``, ``csv``,
            ``checkpoint``, ...).
        message: One line for the log and the result file, for example the command a
            ``prepared`` stage left to be run.
    """

    stage: str
    status: str
    output_dir: str
    stats: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    message: str | None = None

    def __post_init__(self) -> None:
        """Validate the status.

        Raises:
            ValueError: If the status is not one of :data:`STAGE_STATUSES`.
        """
        if self.status not in STAGE_STATUSES:
            raise ValueError(f"Unknown stage status {self.status!r}. Known statuses: {', '.join(STAGE_STATUSES)}.")

    def as_dict(self) -> dict[str, Any]:
        """Serialise the result.

        Returns:
            A JSON-serialisable dictionary.
        """
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "stage": self.stage,
            "status": self.status,
            "output_dir": self.output_dir,
            "stats": dict(self.stats),
            "artifacts": dict(self.artifacts),
            "message": self.message,
        }


def _stage_dir(config: PipelineConfig, stage: str, output_dir: str | Path | None = None) -> Path:
    """Create and return the output directory of a stage.

    Args:
        config: The run configuration.
        stage: The stage name.
        output_dir: Directory to use instead of the one derived from the working directory.
            The single-stage command-line entry points pass the directory the user named.

    Returns:
        The created directory.
    """
    path = Path(output_dir).expanduser().resolve() if output_dir is not None else config.stage_dir(stage)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _require_structures(directory: str | Path, stage: str) -> dict[str, Structure]:
    """Read a structure file or a directory of them, failing when there is nothing to read.

    Args:
        directory: A structure file, or a directory holding some.
        stage: Stage name, for the error message.

    Returns:
        A mapping from name to structure.

    Raises:
        StageError: If the path is missing or holds no readable structure.
    """
    path = Path(directory).expanduser()
    if not path.exists():
        raise StageError(f"The {stage} stage needs structures, but {path} does not exist.")
    structures = load_structure_input(path)
    if not structures:
        raise StageError(f"The {stage} stage needs structures, but no readable structure file is in {path}.")
    return structures


def _row_labels(frame: pd.DataFrame, limit: int = 10) -> str:
    """Name the rows of a table for an error message.

    Args:
        frame: The rows to name.
        limit: How many names to spell out before summarising the rest.

    Returns:
        A comma-separated list of identifiers, taken from the first identifying column the
        table carries.
    """
    labels: list[str] = []
    for column in ("id", "material_id", "composition"):
        if column in frame.columns:
            labels = [str(value) for value in frame[column].tolist()]
            break
    if not labels:
        labels = [str(index) for index in frame.index.tolist()]
    if len(labels) > limit:
        return ", ".join(labels[:limit]) + f", ... (+{len(labels) - limit} more)"
    return ", ".join(labels)


def _resolve_model_path(model_path: str | None, stage: str, config_key: str) -> str:
    """Resolve a machine-learning potential checkpoint.

    Args:
        model_path: The configured path, or ``None`` to fall back to the environment.
        stage: Stage name, for the error message.
        config_key: Configuration key that also sets this path.

    Returns:
        The checkpoint path.

    Raises:
        StageError: If neither the configuration nor the environment supplies one.
    """
    resolved = model_path or os.environ.get(MODEL_PATH_ENV_VAR, "")
    if not resolved:
        raise StageError(
            f"The {stage} stage needs a machine-learning potential checkpoint. "
            f"Set {config_key} in the configuration or the {MODEL_PATH_ENV_VAR} environment variable."
        )
    return resolved


def _load_calculator(kind: str, model_path: str | None, head: str | None, stage: str, config_key: str) -> Any:
    """Load the ASE calculator a stage relaxes or displaces structures with.

    Args:
        kind: Calculator kind from the configuration.
        model_path: Checkpoint path, or ``None`` to fall back to the environment.
        head: Model branch of a multi-task checkpoint.
        stage: Stage name, for the error messages.
        config_key: Configuration key that sets the checkpoint path.

    Returns:
        An ASE calculator.

    Raises:
        StageError: If the backend is not installed or the checkpoint cannot be resolved.
    """
    resolved = None if kind == "emt" else _resolve_model_path(model_path, stage, config_key)
    try:
        return load_calculator(kind, model_path=resolved, head=head)
    except ImportError as error:
        raise StageError(f"The {stage} stage cannot load the {kind!r} calculator: {error}") from error
    except ValueError as error:
        raise StageError(f"The {stage} stage cannot load the {kind!r} calculator: {error}") from error


def run_generation(config: PipelineConfig, output_dir: str | Path | None = None) -> StageResult:
    """Generate candidate structures, or adopt a directory of existing ones.

    When ``generation.structure_dir`` is set the structures there are read and written into
    the stage directory unchanged, and MatterGen is never invoked. Otherwise MatterGen runs
    once per chemical system.

    Args:
        config: The run configuration.
        output_dir: Directory to write into instead of the one derived from the working
            directory.

    Returns:
        A :class:`StageResult` whose ``structure_dir`` artifact holds the candidates.

    Raises:
        StageError: If no chemical system is configured, MatterGen is unavailable, or a
            system produced no structure.
    """
    settings = config.generation
    stage_dir = _stage_dir(config, "generation", output_dir)
    out_dir = stage_dir / STRUCTURES_SUBDIR

    if settings.structure_dir:
        structures = _require_structures(settings.structure_dir, "generation")
        write_structures(structures, out_dir)
        return StageResult(
            stage="generation",
            status="reused",
            output_dir=str(stage_dir),
            stats={"n_structures": len(structures)},
            artifacts={"structure_dir": str(out_dir)},
            message=f"Used the {len(structures)} structure(s) in {settings.structure_dir}; MatterGen was not run.",
        )

    if not settings.chemical_systems:
        raise StageError(
            "The generation stage needs at least one chemical system. Set generation.chemical_systems, "
            "or point generation.structure_dir at structures you already have."
        )

    try:
        results = generate_many(
            chemical_systems=settings.chemical_systems,
            n=settings.n_structures,
            output_root=stage_dir / "mattergen",
            model_path=settings.model_path,
            energy_above_hull=settings.energy_above_hull,
            batch_size=settings.batch_size,
            guidance_factor=settings.guidance_factor,
            skip_existing=settings.skip_existing,
        )
    except (ImportError, FileNotFoundError, ValueError) as error:
        raise StageError(f"The generation stage could not run MatterGen: {error}") from error

    failures = [f"{result.chemical_system}: {result.error}" for result in results if result.error]
    if failures:
        raise StageError("MatterGen failed for " + "; ".join(failures))

    structures: dict[str, Structure] = {}
    per_system: dict[str, int] = {}
    for result in results:
        per_system[result.chemical_system] = len(result.structures)
        for index, structure in enumerate(result.structures):
            structures[f"{result.chemical_system}_{index}"] = structure

    if not structures:
        raise StageError(
            "MatterGen reported success but wrote no structures. Check the run directory "
            f"{stage_dir / 'mattergen'} before continuing."
        )

    write_structures(structures, out_dir)
    return StageResult(
        stage="generation",
        status="completed",
        output_dir=str(stage_dir),
        stats={"n_structures": len(structures), "per_system": per_system},
        artifacts={"structure_dir": str(out_dir)},
        message=f"Generated {len(structures)} structure(s) for {', '.join(settings.chemical_systems)}.",
    )


def run_clustering(
    config: PipelineConfig, structure_dir: str | Path, output_dir: str | Path | None = None
) -> StageResult:
    """Pick the representative structures that are worth a DFT calculation.

    Per-atom descriptors come from a DeePMD-kit model and DIRECT sampling selects the atoms
    that cover the descriptor space; the structures those atoms belong to are the output.

    Args:
        config: The run configuration.
        structure_dir: Directory of candidate structures.
        output_dir: Directory to write into instead of the one derived from the working
            directory.

    Returns:
        A :class:`StageResult` whose ``structure_dir`` artifact holds the selected
        structures.

    Raises:
        StageError: If the input is empty, DeePMD-kit is unavailable, or the checkpoint
            cannot be resolved.
    """
    settings = config.clustering
    stage_dir = _stage_dir(config, "clustering", output_dir)
    selected_dir = stage_dir / SELECTED_SUBDIR

    structures = _require_structures(structure_dir, "clustering")
    names = list(structures)
    values = [structures[name] for name in names]

    model_path = _resolve_model_path(settings.model_path, "clustering", "clustering.model_path")
    try:
        model = load_descriptor_model(model_path, head=settings.head)
        descriptors = compute_descriptor_set(values, model, names=names)
    except ImportError as error:
        raise StageError(f"The clustering stage cannot extract descriptors: {error}") from error
    except ValueError as error:
        raise StageError(f"The clustering stage could not describe the structures: {error}") from error

    selections = select_representative_structures(
        descriptors,
        n=settings.n_representatives,
        k_per_cluster=settings.k_per_cluster,
        threshold=settings.threshold,
    )
    write_selected_structures(values, selections, selected_dir)

    table = pd.DataFrame(
        [
            {
                "id": selection.name,
                "index": selection.index,
                "n_atoms": selection.n_atoms,
                "n_selected_atoms": selection.n_selected_atoms,
            }
            for selection in selections
        ]
    )
    table.to_csv(stage_dir / "selected.csv", index=False)

    return StageResult(
        stage="clustering",
        status="completed",
        output_dir=str(stage_dir),
        stats={"n_input": len(values), "n_selected": len(selections)},
        artifacts={"structure_dir": str(selected_dir), "csv": str(stage_dir / "selected.csv")},
        message=f"Selected {len(selections)} of {len(values)} structure(s) for DFT.",
    )


def run_dft(config: PipelineConfig, structure_dir: str | Path, output_dir: str | Path | None = None) -> StageResult:
    """Write one VASP calculation directory per structure, and optionally submit them.

    Args:
        config: The run configuration.
        structure_dir: Directory of structures to calculate.
        output_dir: Directory to write into instead of the one derived from the working
            directory.

    Returns:
        A :class:`StageResult`. The status is ``completed`` when the jobs were submitted and
        ``prepared`` when they were only written.

    Raises:
        StageError: If the input is empty, no inputs could be written, or submission failed.
    """
    settings = config.dft
    stage_dir = _stage_dir(config, "dft", output_dir)
    calculations = stage_dir / "calculations"

    # One file selection governs both the check and the writing: the structures that were
    # validated are the ones written, and nothing is parsed twice.
    structures = _require_structures(structure_dir, "dft")
    prepared = write_batch_inputs(
        structures,
        calculations,
        kind=settings.kind,
        user_incar_settings=settings.user_incar_settings or None,
        kspacing=settings.kspacing,
        encut_scale=settings.encut_scale,
        magnetic=settings.magnetic,
    )
    if not prepared:
        raise StageError(f"The dft stage wrote no calculation directories from {structure_dir}.")

    slurm_settings = settings_from_mapping(settings.slurm) if settings.slurm else None
    if settings.write_slurm:
        for name in prepared:
            write_slurm_script(calculations / name, slurm_settings)

    stats: dict[str, Any] = {"n_prepared": len(prepared), "kind": settings.kind}
    if settings.submit:
        if not settings.write_slurm:
            raise StageError("dft.submit is set but dft.write_slurm is not, so there is no script to submit.")
        try:
            submitted = submit_directory(calculations)
        except FileNotFoundError as error:
            raise StageError(f"The dft stage could not submit the calculations: {error}") from error
        job_ids = {name: job for name, job in submitted.items() if job}
        stats["n_submitted"] = len(job_ids)
        if not job_ids:
            raise StageError("sbatch accepted no job; see the log above for what it reported.")
        return StageResult(
            stage="dft",
            status="completed",
            output_dir=str(stage_dir),
            stats=stats,
            artifacts={"calculation_dir": str(calculations)},
            message=f"Submitted {len(job_ids)} of {len(prepared)} calculation(s).",
        )

    return StageResult(
        stage="dft",
        status="prepared",
        output_dir=str(stage_dir),
        stats=stats,
        artifacts={"calculation_dir": str(calculations)},
        message=(
            f"Wrote {len(prepared)} calculation director(ies) to {calculations}. "
            "Run them, then resume from the finetune stage."
        ),
    )


def run_finetune(config: PipelineConfig, dft_dir: str | Path, output_dir: str | Path | None = None) -> StageResult:
    """Turn finished DFT runs into a training set and fine-tune on it.

    With ``finetune.run`` unset the stage converts the data, writes the training
    configuration and a submission script, and stops: it produces no checkpoint and says so.
    With ``finetune.run`` set it runs the DeePMD-kit training command and returns the
    checkpoint.

    Args:
        config: The run configuration.
        dft_dir: Directory holding the finished DFT calculations.
        output_dir: Directory to write into instead of the one derived from the working
            directory.

    Returns:
        A :class:`StageResult`. The status is ``completed`` when training ran and
        ``prepared`` when it did not.

    Raises:
        StageError: If the conversion backend is missing, no frames could be converted, no
            checkpoint can be resolved, or the training command failed.
    """
    settings = config.finetune
    stage_dir = _stage_dir(config, "finetune", output_dir)
    data_dir = stage_dir / "deepmd_data"

    source = Path(dft_dir).expanduser()
    if not source.is_dir():
        raise StageError(f"The finetune stage needs DFT output, but {source} does not exist.")

    try:
        _, conversion_stats = convert_dft_to_deepmd(
            dft_base_dir=source,
            output_dir=data_dir,
            exclude_patterns=settings.exclude_patterns or None,
        )
    except ImportError as error:
        raise StageError(f"The finetune stage cannot convert the DFT data: {error}") from error

    n_frames = int(conversion_stats.get("total_frames", 0))
    if n_frames == 0:
        raise StageError(
            f"No frames were converted from {source}. Check that the calculations finished and that "
            "finetune.exclude_patterns is not excluding all of them."
        )

    type_map = settings.type_map or get_type_map(data_dir)
    train_systems, val_systems = split_train_val(
        deepmd_dir=data_dir,
        train_dir=stage_dir / "train_data",
        val_dir=stage_dir / "val_data",
        val_ratio=settings.val_ratio,
    )

    try:
        training_config = build_dpa3_config(
            type_map=type_map,
            train_systems=train_systems,
            val_systems=val_systems,
            numb_steps=settings.numb_steps,
            batch_size=settings.batch_size,
        )
    except ValueError as error:
        raise StageError(
            f"The finetune stage could not build a training configuration from the {len(train_systems)} training "
            f"and {len(val_systems)} validation system(s) converted from {source}: {error}"
        ) from error
    config_file = write_dpa3_config(training_config, stage_dir / "finetune_config.json")

    try:
        pretrained = resolve_pretrained_model(settings.pretrained_model)
    except ValueError as error:
        raise StageError(f"The finetune stage needs a pretrained checkpoint: {error}") from error

    stats = {
        "n_frames": n_frames,
        "n_train_systems": len(train_systems),
        "n_val_systems": len(val_systems),
        "type_map": list(type_map),
    }
    artifacts = {"config_file": str(config_file), "data_dir": str(data_dir)}

    if not settings.run:
        script = write_finetune_slurm_script(
            config_file=config_file,
            pretrained_model=pretrained,
            output_file=stage_dir / "run_finetune.sh",
            model_branch=settings.model_branch,
        )
        artifacts["slurm_script"] = str(script)
        return StageResult(
            stage="finetune",
            status="prepared",
            output_dir=str(stage_dir),
            stats=stats,
            artifacts=artifacts,
            message=(
                f"Converted {n_frames} frame(s) and wrote {config_file}. No training was run and no checkpoint "
                f"exists yet: submit {script}, or set finetune.run to train here. Until a checkpoint exists the "
                "screening stage uses the checkpoint named in its own configuration."
            ),
        )

    try:
        result = finetune(
            config_file=config_file,
            pretrained_model=pretrained,
            output_dir=stage_dir,
            model_branch=settings.model_branch,
        )
    except ImportError as error:
        raise StageError(f"The finetune stage cannot run the training command: {error}") from error

    if not result.success or not result.checkpoint:
        raise StageError(
            f"Fine-tuning failed (return code {result.return_code}): {result.error or 'no checkpoint was written'}"
        )

    artifacts["checkpoint"] = result.checkpoint
    return StageResult(
        stage="finetune",
        status="completed",
        output_dir=str(stage_dir),
        stats=stats,
        artifacts=artifacts,
        message=f"Fine-tuned on {n_frames} frame(s); checkpoint {result.checkpoint}.",
    )


def run_competing(config: PipelineConfig, output_dir: str | Path | None = None) -> StageResult:
    """Harvest the competing phases of the chemical system from the Materials Project.

    Args:
        config: The run configuration.
        output_dir: Directory to write into instead of the one derived from the working
            directory.

    Returns:
        A :class:`StageResult` whose ``csv`` artifact holds the competing-phase table and,
        when structures were downloaded, whose ``structure_dir`` artifact holds them.

    Raises:
        StageError: If no chemical system is configured, the Materials Project client or key
            is unavailable, or the search returned nothing.
    """
    settings = config.competing
    stage_dir = _stage_dir(config, "competing", output_dir)
    csv_path = stage_dir / COMPETING_CSV
    structures_dir = stage_dir / STRUCTURES_SUBDIR

    chemsys = settings.chemsys
    if not chemsys and config.generation.chemical_systems:
        chemsys = config.generation.chemical_systems[0]
    if not chemsys:
        raise StageError(
            "The competing stage needs a chemical system. Set competing.chemsys, or list one under "
            "generation.chemical_systems."
        )

    reused = False
    if settings.csv_file:
        source = Path(settings.csv_file).expanduser()
        if not source.is_file():
            raise StageError(f"competing.csv_file does not exist: {source}")
        shutil.copyfile(source, csv_path)
        table = pd.read_csv(csv_path)
        reused = True
        LOGGER.info("Read %d competing phase(s) from %s", len(table), source)
    else:
        try:
            table = find_competing_phases(
                chemsys=chemsys,
                include_unary=settings.include_unary,
                include_binary=settings.include_binary,
                include_higher_order=settings.include_higher_order,
                only_icsd=settings.only_icsd,
                only_stable=settings.only_stable,
                energy_above_hull_max=settings.energy_above_hull_max,
                unique_formula=settings.unique_formula,
                allow_partial=settings.allow_partial,
            )
        except (ImportError, RuntimeError, ValueError) as error:
            raise StageError(f"The competing stage could not search the Materials Project: {error}") from error
        if table.empty:
            raise StageError(
                f"No competing phase was found for {chemsys}. Relax the filters (competing.only_icsd, "
                "competing.only_stable) or supply competing.csv_file."
            )
        table.to_csv(csv_path, index=False)

    missing = list(table.attrs.get("missing_subsystems", []))
    stats: dict[str, Any] = {"chemsys": chemsys, "n_phases": int(len(table))}
    if missing:
        stats["missing_subsystems"] = missing
    artifacts = {"csv": str(csv_path)}

    if settings.structure_dir:
        existing = Path(settings.structure_dir).expanduser()
        if not existing.is_dir():
            raise StageError(f"competing.structure_dir does not exist: {existing}")
        artifacts["structure_dir"] = str(existing)
        stats["n_structures"] = len(load_structure_input(existing))
    elif settings.download_structures:
        try:
            downloaded = download_structures(table, structures_dir)
        except (ImportError, RuntimeError, ValueError) as error:
            raise StageError(f"The competing stage could not download the structures: {error}") from error
        succeeded = {key: value for key, value in downloaded.items() if value}
        if not succeeded:
            raise StageError(f"No competing-phase structure could be downloaded into {structures_dir}.")
        artifacts["structure_dir"] = str(structures_dir)
        stats["n_structures"] = len(succeeded)

    return StageResult(
        stage="competing",
        status="reused" if reused else "completed",
        output_dir=str(stage_dir),
        stats=stats,
        artifacts=artifacts,
        message=f"{len(table)} competing phase(s) for {chemsys}.",
    )


def _relax_set(
    structures: Mapping[str, Structure],
    calculator: Any,
    output_dir: Path,
    settings: Any,
) -> pd.DataFrame:
    """Relax a set of structures and return their energies.

    Args:
        structures: Mapping from name to structure.
        calculator: ASE calculator to relax with.
        output_dir: Directory for the relaxed structures.
        settings: The screening section, read for the optimizer settings.

    Returns:
        The summary table :func:`matdisc.screening.relax.relax_many` produces.
    """
    _, table = relax_many(
        structures,
        calculator,
        output_dir=output_dir,
        optimizer=settings.optimizer,
        fmax=settings.fmax,
        max_steps=settings.max_steps,
        relax_cell=settings.relax_cell,
    )
    return table


def run_screening(
    config: PipelineConfig,
    structure_dir: str | Path,
    competing_csv: str | Path | None = None,
    competing_structure_dir: str | Path | None = None,
    model_path: str | None = None,
    output_dir: str | Path | None = None,
) -> StageResult:
    """Relax the candidates and place them on the convex hull of the competing phases.

    By default the competing structures are relaxed with the same calculator as the
    candidates so that both sides of the hull are on one energy scale. Setting
    ``screening.competing_energy_source`` to ``"table"`` reads the competing energies from
    the harvested table instead, which the caller is then responsible for matching.

    Args:
        config: The run configuration.
        structure_dir: Directory of candidate structures.
        competing_csv: Competing-phase table from the competing stage. It is required when
            ``screening.competing_energy_source`` is ``"table"`` and unused otherwise, since
            the relaxed energies replace the harvested ones.
        competing_structure_dir: Directory of competing-phase structures, required unless
            ``screening.competing_energy_source`` is ``"table"``.
        model_path: Checkpoint to override ``screening.model_path`` with, for example the
            one the fine-tuning stage produced.
        output_dir: Directory to write into instead of the one derived from the working
            directory.

    Returns:
        A :class:`StageResult` whose ``csv`` artifact is the hull table and whose
        ``structure_dir`` artifact holds the relaxed structures at or below the hull -- the
        geometries the reported energies belong to, not the input cells.

    Raises:
        StageError: If an input is missing or empty, the calculator cannot be loaded, the
            competing energies are not on a usable scale, a competing phase carries no usable
            energy (unless ``screening.allow_incomplete_hull`` is set), or a surviving
            candidate has no relaxed structure on disk.
    """
    settings = config.screening
    stage_dir = _stage_dir(config, "screening", output_dir)
    stable_dir = stage_dir / STABLE_SUBDIR

    candidates = _require_structures(structure_dir, "screening")

    # Both sides of the hull are checked before anything expensive runs, so a missing input
    # is reported now rather than after the relaxations.
    competing_structures: dict[str, Structure] = {}
    table_energies: pd.DataFrame | None = None
    if settings.competing_energy_source == "relax":
        if not competing_structure_dir:
            raise StageError(
                "The screening stage relaxes the competing phases so that both sides of the hull come from the "
                "same calculator, but no competing-phase structures were given. Either let the competing stage "
                "download them (competing.download_structures) or set screening.competing_energy_source to "
                "'table' and make sure the table's energies match the candidate energies."
            )
        competing_structures = _require_structures(competing_structure_dir, "screening")
        if competing_csv:
            LOGGER.info("Ignoring the energies in %s; the competing phases are relaxed here instead", competing_csv)
    else:
        if not competing_csv:
            raise StageError(
                "screening.competing_energy_source is 'table', so the screening stage needs a competing-phase "
                "table. Run the competing stage or set competing.csv_file."
            )
        csv_path = Path(competing_csv).expanduser()
        if not csv_path.is_file():
            raise StageError(f"The screening stage needs a competing-phase table, but {csv_path} does not exist.")
        table_energies = pd.read_csv(csv_path)
        if table_energies.empty:
            raise StageError(f"The competing-phase table {csv_path} is empty.")
        if settings.energy_column not in table_energies.columns:
            raise StageError(
                f"screening.competing_energy_source is 'table' but {csv_path} has no "
                f"{settings.energy_column!r} column. Columns present: {', '.join(table_energies.columns)}."
            )
        LOGGER.warning(
            "Reading competing-phase energies from %s. Candidate and competing energies must come from the "
            "same kind of calculation for the hull to mean anything.",
            csv_path,
        )

    calculator = _load_calculator(
        settings.calculator, model_path or settings.model_path, settings.head, "screening", "screening.model_path"
    )
    head = calculator_head(calculator)

    LOGGER.info("Relaxing %d candidate(s)", len(candidates))
    relaxed_dir = stage_dir / "relaxed"
    candidate_table = _relax_set(candidates, calculator, relaxed_dir, settings)
    candidate_table.to_csv(stage_dir / CANDIDATES_CSV, index=False)

    if table_energies is not None:
        hull_input = table_energies
    else:
        LOGGER.info("Relaxing %d competing phase(s) with the same calculator", len(competing_structures))
        hull_input = _relax_set(competing_structures, calculator, stage_dir / "competing_relaxed", settings)
        hull_input.to_csv(stage_dir / COMPETING_RELAXED_CSV, index=False)

    competing_used, competing_dropped = split_usable_rows(
        hull_input, energy_column=settings.energy_column, require_converged=settings.require_converged
    )
    if not competing_dropped.empty:
        detail = _row_labels(competing_dropped)
        if not settings.allow_incomplete_hull:
            raise StageError(
                f"{len(competing_dropped)} of {len(hull_input)} competing phase(s) carry no usable "
                f"{settings.energy_column}: {detail}. A hull missing a competing phase can only sit lower, so "
                "every candidate would be reported closer to stability than it is. Fix those relaxations, or "
                "set screening.allow_incomplete_hull to build the hull without them knowingly."
            )
        LOGGER.warning(
            "Building the hull without %d competing phase(s) (screening.allow_incomplete_hull): %s",
            len(competing_dropped),
            detail,
        )

    try:
        hull = compute_e_above_hull(
            candidate_table,
            competing_used,
            tolerance=settings.tolerance,
            energy_column=settings.energy_column,
        )
    except (KeyError, ValueError) as error:
        raise StageError(f"The screening stage could not build the convex hull: {error}") from error

    # Provenance of the verdict travels with the table: which calculator branch produced the
    # energies, and how many competing phases the hull actually rests on.
    hull["calculator"] = settings.calculator
    hull["calculator_head"] = head or ""
    hull["n_competing_used"] = int(len(competing_used))
    hull["n_competing_dropped"] = int(len(competing_dropped))
    hull.to_csv(stage_dir / HULL_CSV, index=False)

    on_hull = hull["is_stable"].astype(bool)
    reported = on_hull
    n_unconverged = 0
    if settings.require_converged and "converged" in hull.columns:
        converged = hull["converged"].fillna(False).astype(bool)
        n_unconverged = int((on_hull & ~converged).sum())
        reported = on_hull & converged
        if n_unconverged:
            LOGGER.warning(
                "%d candidate(s) sit at or below the hull but their relaxation did not converge; "
                "they are not reported as stable (screening.require_converged)",
                n_unconverged,
            )

    stable_names = [str(name) for name in hull.loc[reported, "id"]]
    stable_dir.mkdir(parents=True, exist_ok=True)
    for name in stable_names:
        # relax() writes <output_dir>/<id>/relaxed.vasp, and that geometry is the one the
        # reported energy belongs to. Substituting the input cell would hand the phonon stage
        # an unrelaxed structure, so a missing file stops the run instead.
        source = relaxed_dir / name / "relaxed.vasp"
        if not source.is_file():
            raise StageError(
                f"Candidate {name} is at or below the hull but its relaxed structure is missing: {source}. "
                "The stable structures must be the geometries the reported energies belong to."
            )
        shutil.copyfile(source, stable_dir / f"{name}.vasp")

    return StageResult(
        stage="screening",
        status="completed",
        output_dir=str(stage_dir),
        stats={
            "n_candidates": int(len(hull)),
            "n_on_hull": int(on_hull.sum()),
            "n_stable": len(stable_names),
            "n_unconverged": n_unconverged,
            "n_competing_requested": int(len(hull_input)),
            "n_competing_used": int(len(competing_used)),
            "n_competing_dropped": int(len(competing_dropped)),
            "tolerance": settings.tolerance,
            "competing_energy_source": settings.competing_energy_source,
            "calculator": settings.calculator,
            "calculator_head": head,
        },
        artifacts={"csv": str(stage_dir / HULL_CSV), "structure_dir": str(stable_dir)},
        message=(
            f"{len(stable_names)} of {len(hull)} candidate(s) sit at or below the hull of "
            f"{len(competing_used)} competing phase(s) (tolerance {settings.tolerance:g} eV/atom"
            + (f", {n_unconverged} unconverged candidate(s) excluded" if n_unconverged else "")
            + ")."
        ),
    )


def _write_phonon_survivor(
    name: str,
    structure: Structure,
    run_dir: Path,
    settings: Any,
    stable_dir: Path,
) -> None:
    """Write the geometry a phonon verdict belongs to into the survivor directory.

    With ``relax_first`` the spectrum was computed on the relaxed positions, which
    :func:`matdisc.screening.phonons.check_dynamical_stability` wrote to ``relaxed.vasp``.
    That file, not the input cell, is the structure the verdict describes.

    Args:
        name: Structure identifier.
        structure: The structure as it was handed to the check.
        run_dir: Directory the check wrote into.
        settings: The phonons section, read for ``relax_first``.
        stable_dir: Directory the survivors are collected in.

    Raises:
        StageError: If the relaxed geometry the spectrum belongs to is missing.
    """
    if not settings.relax_first:
        write_structures({name: structure}, stable_dir)
        return
    source = run_dir / "relaxed.vasp"
    if not source.is_file():
        raise StageError(
            f"{name} passed the phonon check but its relaxed structure is missing: {source}. "
            "The surviving structures must be the geometries the spectra were computed on."
        )
    shutil.copyfile(source, stable_dir / f"{name}.vasp")


def run_phonons(
    config: PipelineConfig,
    structure_dir: str | Path,
    model_path: str | None = None,
    output_dir: str | Path | None = None,
) -> StageResult:
    """Check the surviving structures for imaginary phonon frequencies.

    Args:
        config: The run configuration.
        structure_dir: Directory of structures to check, normally the hull survivors.
        model_path: Checkpoint to override ``screening.model_path`` with.
        output_dir: Directory to write into instead of the one derived from the working
            directory.

    Returns:
        A :class:`StageResult` whose ``csv`` artifact is the summary table and whose
        ``structure_dir`` artifact holds the dynamically stable structures. The status is
        ``empty`` when there was nothing to check.

    Raises:
        StageError: If the calculator cannot be loaded, or the input directory does not
            exist.
    """
    settings = config.phonons
    stage_dir = _stage_dir(config, "phonons", output_dir)
    stable_dir = stage_dir / STABLE_SUBDIR
    summary_path = stage_dir / PHONON_CSV

    source = Path(structure_dir).expanduser()
    if not source.exists():
        raise StageError(f"The phonons stage needs structures, but {source} does not exist.")

    structures = load_structure_input(source)
    if not structures:
        pd.DataFrame(columns=PHONON_COLUMNS).to_csv(summary_path, index=False)
        return StageResult(
            stage="phonons",
            status="empty",
            output_dir=str(stage_dir),
            stats={"n_checked": 0, "n_stable": 0},
            artifacts={"csv": str(summary_path)},
            message=f"No structure to check in {source}.",
        )

    if settings.max_structures is not None:
        names = list(structures)[: settings.max_structures]
        if len(names) < len(structures):
            LOGGER.info("Checking %d of %d structure(s) (phonons.max_structures)", len(names), len(structures))
        structures = {name: structures[name] for name in names}

    calculator = _load_calculator(
        config.screening.calculator,
        model_path or config.screening.model_path,
        config.screening.head,
        "phonons",
        "screening.model_path",
    )

    supercell = tuple(settings.supercell) if settings.supercell else None
    rows: list[dict[str, Any]] = []
    stable_dir.mkdir(parents=True, exist_ok=True)
    for name, structure in structures.items():
        LOGGER.info("Phonon check for %s", name)
        result = check_dynamical_stability(
            structure,
            calculator,
            output_dir=stage_dir / name,
            supercell=supercell,
            min_cell_length=settings.min_cell_length,
            delta=settings.delta,
            relax_first=settings.relax_first,
            fmax=settings.fmax,
            max_steps=settings.max_steps,
            require_relaxation_converged=settings.require_relaxation_converged,
            threshold=settings.threshold,
            plot=settings.plot,
        )
        rows.append(
            {
                "id": name,
                "success": bool(result.get("success")),
                "min_frequency": result.get("min_frequency"),
                "is_stable": bool(result.get("is_stable")),
                "relaxation_converged": result.get("relaxation_converged"),
                "output_dir": result.get("output_dir"),
                "error": result.get("error_message"),
            }
        )
        if result.get("is_stable"):
            _write_phonon_survivor(name, structure, Path(str(result.get("output_dir"))), settings, stable_dir)

    summary = pd.DataFrame(rows, columns=PHONON_COLUMNS)
    summary.to_csv(summary_path, index=False)
    n_stable = int(summary["is_stable"].sum())
    n_failed = int((~summary["success"]).sum())

    return StageResult(
        stage="phonons",
        status="completed",
        output_dir=str(stage_dir),
        stats={"n_checked": int(len(summary)), "n_stable": n_stable, "n_failed": n_failed},
        artifacts={"csv": str(summary_path), "structure_dir": str(stable_dir)},
        message=(
            f"{n_stable} of {len(summary)} structure(s) have no frequency below "
            f"{settings.threshold:g} THz ({n_failed} check(s) failed)."
        ),
    )
