"""Configuration, stage wiring and the end-to-end run entry point.

A run is described by one YAML file (:mod:`matdisc.pipeline.config`), executed stage by stage
(:mod:`matdisc.pipeline.stages`) and sequenced by :func:`matdisc.pipeline.run.run_pipeline`.
:mod:`matdisc.pipeline.data_interface` holds the file conventions the stages hand each other
structures through, and the check that a stage produced what the next one reads.

Each stage calls the same public functions a user would call directly, so nothing here
duplicates the science: the pipeline is the wiring, not a second implementation.
"""

from matdisc.pipeline.config import STAGE_ORDER, PipelineConfig, load_config, write_config_template
from matdisc.pipeline.data_interface import StageCheck, validate_pipeline, validate_stage_output
from matdisc.pipeline.run import PipelineResult, run_from_config, run_pipeline, stages_from
from matdisc.pipeline.stages import StageError, StageResult

__all__ = [
    "STAGE_ORDER",
    "PipelineConfig",
    "PipelineResult",
    "StageCheck",
    "StageError",
    "StageResult",
    "load_config",
    "run_from_config",
    "run_pipeline",
    "stages_from",
    "validate_pipeline",
    "validate_stage_output",
    "write_config_template",
]
