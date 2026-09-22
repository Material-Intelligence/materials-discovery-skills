"""Sequencing of the pipeline stages.

:func:`run_pipeline` executes the stages listed in the configuration, in
:data:`~matdisc.pipeline.config.STAGE_ORDER`, and writes a machine-readable summary of the
run into the working directory.

How a stage gets its input
--------------------------
An input path written in the configuration always wins. When it is absent the stage takes
the output of the stage that produced it earlier in the same run. When neither exists the
run stops with a message naming the configuration key and the stage that would have supplied
it. Nothing is guessed and no stage is silently skipped.

Resuming
--------
A run is resumed by listing fewer stages and pointing the first of them at the output of the
previous run, for example ``stages: [finetune, competing, screening]`` with
``finetune.dft_dir`` set to the calculations that have now finished. The DFT stage itself
stops the run when it has only written inputs and a later stage would need their results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from monty.json import MSONable

from matdisc.common.logging import get_logger
from matdisc.pipeline.config import STAGE_ORDER, PipelineConfig, load_config
from matdisc.pipeline.data_interface import RESULT_JSON
from matdisc.pipeline.stages import (
    StageError,
    StageResult,
    run_clustering,
    run_competing,
    run_dft,
    run_finetune,
    run_generation,
    run_phonons,
    run_screening,
)

LOGGER = get_logger(__name__)

RUN_STATUSES = ("completed", "stopped", "failed")
"""What a :class:`PipelineResult` reports.

``completed`` -- every listed stage ran. ``stopped`` -- the run ended deliberately because a
stage left work to be done elsewhere, such as DFT calculations that must finish before
fine-tuning. ``failed`` -- a stage raised :class:`~matdisc.pipeline.stages.StageError`.
"""

__all__ = ["RUN_STATUSES", "PipelineResult", "run_from_config", "run_pipeline", "stages_from"]


@dataclass
class PipelineResult(MSONable):
    """The outcome of a run.

    Attributes:
        name: The run name from the configuration.
        work_dir: The working directory.
        requested_stages: The stages the configuration asked for.
        results: The :class:`~matdisc.pipeline.stages.StageResult` of each stage that ran.
        status: One of :data:`RUN_STATUSES`.
        stopped_at: The stage the run stopped at, when it did not complete.
        message: Why the run stopped or failed.
        config: The configuration the run used.
    """

    name: str
    work_dir: str
    requested_stages: list[str] = field(default_factory=list)
    results: dict[str, StageResult] = field(default_factory=dict)
    status: str = "completed"
    stopped_at: str | None = None
    message: str | None = None
    config: PipelineConfig | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialise the result.

        Returns:
            A JSON-serialisable dictionary, including the configuration that produced it.
        """
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "name": self.name,
            "work_dir": self.work_dir,
            "requested_stages": list(self.requested_stages),
            "status": self.status,
            "stopped_at": self.stopped_at,
            "message": self.message,
            "results": {stage: result.as_dict() for stage, result in self.results.items()},
            "config": self.config.as_dict() if self.config is not None else None,
        }

    def write_json(self, path: str | Path | None = None) -> Path:
        """Write the result as JSON.

        Args:
            path: Destination file. Defaults to
                :data:`~matdisc.pipeline.data_interface.RESULT_JSON` in the working
                directory.

        Returns:
            The path written.
        """
        target = Path(path) if path is not None else Path(self.work_dir) / RESULT_JSON
        target = target.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        LOGGER.info("Wrote the run summary to %s", target)
        return target

    def summary_lines(self) -> list[str]:
        """Render one line per stage for a terminal report.

        Returns:
            The lines, in the order the stages ran.
        """
        lines = [f"{self.name}: {self.status}" + (f" at {self.stopped_at}" if self.stopped_at else "")]
        for stage, result in self.results.items():
            lines.append(f"  {stage:<11} {result.status:<9} {result.message or ''}".rstrip())
        if self.message:
            lines.append(f"  {self.message}")
        return lines


def _artifact(results: Mapping[str, StageResult], stage: str, key: str) -> str | None:
    """Read one artifact of a stage that already ran.

    Args:
        results: The stage results so far.
        stage: The stage that would have produced it.
        key: The artifact key.

    Returns:
        The path, or ``None`` when the stage did not run or did not produce it.
    """
    result = results.get(stage)
    if result is None:
        return None
    return result.artifacts.get(key)


def _resolve_input(
    configured: str | None,
    results: Mapping[str, StageResult],
    sources: Sequence[tuple[str, str]],
    stage: str,
    config_key: str,
) -> str:
    """Resolve a stage's input path.

    A configured path wins; otherwise the first available artifact of an earlier stage is
    used.

    Args:
        configured: The path written in the configuration, if any.
        results: The stage results so far.
        sources: ``(stage, artifact key)`` pairs to try, in order.
        stage: The stage needing the input, for the error message.
        config_key: The configuration key that supplies it, for the error message.

    Returns:
        The resolved path.

    Raises:
        StageError: If neither the configuration nor an earlier stage supplies the input.
    """
    if configured:
        return configured
    for source_stage, key in sources:
        path = _artifact(results, source_stage, key)
        if path:
            if source_stage != sources[0][0]:
                LOGGER.info("The %s stage takes its input from the %s stage: %s", stage, source_stage, path)
            return path
    producers = ", ".join(sorted({source for source, _ in sources}))
    raise StageError(
        f"The {stage} stage has no input. Set {config_key} in the configuration, or run the {producers} stage first."
    )


def _screening_model(config: PipelineConfig, results: Mapping[str, StageResult]) -> str | None:
    """Choose the checkpoint the screening and phonon stages relax with.

    Args:
        config: The run configuration.
        results: The stage results so far.

    Returns:
        The fine-tuned checkpoint when this run produced one and the configuration does not
        name a checkpoint of its own, otherwise ``None`` to let the configuration or the
        environment decide.
    """
    if config.screening.model_path:
        return None
    checkpoint = _artifact(results, "finetune", "checkpoint")
    if checkpoint:
        LOGGER.info("Screening with the checkpoint the fine-tuning stage produced: %s", checkpoint)
    return checkpoint


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run the stages a configuration lists.

    Args:
        config: The run configuration.

    Returns:
        The :class:`PipelineResult`, which is also written to
        :data:`~matdisc.pipeline.data_interface.RESULT_JSON` in the working directory.

    Raises:
        StageError: If a stage cannot do its work. The partial result is written before the
            error propagates.
    """
    work_dir = Path(config.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Run %s in %s", config.name, work_dir)
    LOGGER.info("Stages: %s", ", ".join(config.stages))

    outcome = PipelineResult(
        name=config.name,
        work_dir=str(work_dir),
        requested_stages=list(config.stages),
        config=config,
    )
    results = outcome.results
    stage = ""

    try:
        for position, stage in enumerate(config.stages):
            remaining = config.stages[position + 1 :]
            LOGGER.info("--- %s ---", stage)

            if stage == "generation":
                results[stage] = run_generation(config)

            elif stage == "clustering":
                structure_dir = _resolve_input(
                    config.clustering.structure_dir,
                    results,
                    [("generation", "structure_dir")],
                    "clustering",
                    "clustering.structure_dir",
                )
                results[stage] = run_clustering(config, structure_dir)

            elif stage == "dft":
                structure_dir = _resolve_input(
                    config.dft.structure_dir,
                    results,
                    [("clustering", "structure_dir"), ("generation", "structure_dir")],
                    "dft",
                    "dft.structure_dir",
                )
                results[stage] = run_dft(config, structure_dir)
                if results[stage].status == "prepared" and "finetune" in remaining:
                    outcome.status = "stopped"
                    outcome.stopped_at = "dft"
                    outcome.message = (
                        "The DFT inputs are written but the calculations have not run, so there is nothing to "
                        "fine-tune on yet. Run them, then resume with the remaining stages and "
                        "finetune.dft_dir set to "
                        f"{results[stage].artifacts.get('calculation_dir', 'the calculation directory')}."
                    )
                    break

            elif stage == "finetune":
                dft_dir = _resolve_input(
                    config.finetune.dft_dir,
                    results,
                    [("dft", "calculation_dir")],
                    "finetune",
                    "finetune.dft_dir",
                )
                results[stage] = run_finetune(config, dft_dir)

            elif stage == "competing":
                results[stage] = run_competing(config)

            elif stage == "screening":
                structure_dir = _resolve_input(
                    config.screening.structure_dir,
                    results,
                    [("generation", "structure_dir")],
                    "screening",
                    "screening.structure_dir",
                )
                competing_csv = config.competing.csv_file or _artifact(results, "competing", "csv")
                if config.screening.competing_energy_source == "table" and not competing_csv:
                    competing_csv = _resolve_input(
                        None, results, [("competing", "csv")], "screening", "competing.csv_file"
                    )
                competing_structures = config.competing.structure_dir or _artifact(
                    results, "competing", "structure_dir"
                )
                results[stage] = run_screening(
                    config,
                    structure_dir=structure_dir,
                    competing_csv=competing_csv,
                    competing_structure_dir=competing_structures,
                    model_path=_screening_model(config, results),
                )

            elif stage == "phonons":
                structure_dir = _resolve_input(
                    config.phonons.structure_dir,
                    results,
                    [("screening", "structure_dir")],
                    "phonons",
                    "phonons.structure_dir",
                )
                results[stage] = run_phonons(
                    config,
                    structure_dir=structure_dir,
                    model_path=_screening_model(config, results),
                )

            LOGGER.info("%s: %s", stage, results[stage].message or results[stage].status)

    except StageError as error:
        outcome.status = "failed"
        outcome.stopped_at = stage
        outcome.message = str(error)
        outcome.write_json()
        raise

    outcome.write_json()
    for line in outcome.summary_lines():
        LOGGER.info("%s", line)
    return outcome


def run_from_config(
    path: str | Path,
    stages: Sequence[str] | None = None,
    work_dir: str | None = None,
) -> PipelineResult:
    """Read a configuration file and run it.

    Args:
        path: Path to the YAML configuration.
        stages: Stage names that replace the file's ``stages`` list, or ``None``.
        work_dir: Working directory that replaces the file's ``work_dir``, or ``None``.

    Returns:
        The :class:`PipelineResult`.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration is invalid.
        StageError: If a stage cannot do its work.
    """
    config = load_config(path, stages=stages, work_dir=work_dir)
    return run_pipeline(config)


def stages_from(stage: str) -> list[str]:
    """List a stage and everything after it.

    Args:
        stage: The stage to start from.

    Returns:
        That stage and the stages that follow it in :data:`STAGE_ORDER`.

    Raises:
        ValueError: If the stage name is unknown.
    """
    if stage not in STAGE_ORDER:
        raise ValueError(f"Unknown stage {stage!r}. Known stages: {', '.join(STAGE_ORDER)}.")
    return list(STAGE_ORDER[STAGE_ORDER.index(stage) :])
