"""Fine-tuning of a DPA-3 machine-learning interatomic potential on DFT labels.

The training input is a DeePMD-kit JSON configuration built by :func:`build_dpa3_config`; training
itself runs through the ``dp`` console script (``dp --pt train <config> --finetune <checkpoint>
--model-branch <head>``), either directly or from a SLURM script written by
:func:`write_finetune_slurm_script`.

DeePMD-kit is an optional extra: it is never imported, only invoked as a command, and its absence is
reported with a message naming the extra to install.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from monty.json import MSONable

from matdisc.common.logging import get_logger

logger = get_logger(__name__)

#: Console script installed by DeePMD-kit.
DEEPMD_EXECUTABLE = "dp"
#: Environment variable consulted when no pretrained checkpoint is passed explicitly.
MODEL_PATH_ENV_VAR = "DPA3_MODEL_PATH"
#: Default multi-task head to fine-tune from.
DEFAULT_MODEL_BRANCH = "Omat24"

_DEEPMD_HINT = (
    f"'{DEEPMD_EXECUTABLE}' was not found on PATH. DeePMD-kit is an optional backend: install the "
    "extra with pip install 'materials-discovery-skills[mlip]'"
)


@dataclass
class DPA3DescriptorConfig(MSONable):
    """DPA-3 descriptor block of a DeePMD-kit configuration."""

    type: str = "dpa3"
    activation_function: str = "silut:3.0"
    use_tebd_bias: bool = False
    precision: str = "float32"
    concat_output_tebd: bool = False
    repflow: dict[str, Any] = field(
        default_factory=lambda: {
            "n_dim": 128,
            "e_dim": 64,
            "a_dim": 32,
            "nlayers": 16,
            "e_rcut": 6.0,
            "e_rcut_smth": 5.3,
            "e_sel": 1200,
            "a_rcut": 4.0,
            "a_rcut_smth": 3.5,
            "a_sel": 300,
            "axis_neuron": 4,
            "fix_stat_std": 0.3,
            "a_compress_rate": 1,
            "a_compress_e_rate": 2,
            "a_compress_use_split": True,
            "update_angle": True,
            "smooth_edge_update": True,
            "use_dynamic_sel": True,
            "sel_reduce_factor": 10.0,
            "use_exp_switch": True,
            "update_style": "res_residual",
            "update_residual": 0.1,
            "update_residual_init": "const",
        }
    )


@dataclass
class FittingNetConfig(MSONable):
    """Fitting-network block of a DeePMD-kit configuration."""

    neuron: list[int] = field(default_factory=lambda: [240, 240, 240])
    dim_case_embd: int = 31
    resnet_dt: bool = True
    precision: str = "float32"
    activation_function: str = "silut:3.0"
    seed: int = 1


@dataclass
class LearningRateConfig(MSONable):
    """Learning-rate block of a DeePMD-kit configuration."""

    type: str = "exp"
    decay_steps: int = 5000
    start_lr: float = 1.0e-04
    stop_lr: float = 3e-8


@dataclass
class LossConfig(MSONable):
    """Loss block of a DeePMD-kit configuration, with the energy/force/virial prefactors."""

    type: str = "ener"
    start_pref_e: float = 0.2
    limit_pref_e: float = 20
    start_pref_f: float = 100
    limit_pref_f: float = 60
    start_pref_v: float = 0.02
    limit_pref_v: float = 1


@dataclass
class FinetuneResult(MSONable):
    """Outcome of a fine-tuning run.

    Attributes:
        success: Whether the training command exited with status 0.
        output_dir: Working directory the command ran in.
        config_file: Configuration file that was trained from.
        command: The command line that was run.
        return_code: Exit status of the command, ``-1`` when it could not be started.
        checkpoint: Newest checkpoint found in the working directory after the run.
        error: Error message when the command could not be started.
    """

    success: bool
    output_dir: str
    config_file: str
    command: str
    return_code: int
    checkpoint: str | None = None
    error: str | None = None


def build_dpa3_config(
    type_map: Sequence[str],
    train_systems: Sequence[str],
    val_systems: Sequence[str] | None = None,
    numb_steps: int = 100000,
    batch_size: int = 4,
    start_lr: float = 1e-4,
    stop_lr: float = 3e-8,
    decay_steps: int = 5000,
    warmup_steps: int = 1000,
    save_freq: int = 2000,
    disp_freq: int = 100,
    stat_file: str = "./dpa3.hdf5",
    seed: int = 10,
    start_pref_e: float = 0.2,
    limit_pref_e: float = 20,
    start_pref_f: float = 100,
    limit_pref_f: float = 60,
) -> dict[str, Any]:
    """Build a DeePMD-kit fine-tuning configuration for a DPA-3 model.

    Args:
        type_map: Elements the model is trained on, in the order the data uses.
        train_systems: Paths of the ``deepmd/npy`` training systems.
        val_systems: Paths of the validation systems; omitted from the configuration when empty.
        numb_steps: Number of training steps.
        batch_size: Training batch size, also used for validation.
        start_lr: Initial learning rate.
        stop_lr: Final learning rate.
        decay_steps: Steps between learning-rate decay events.
        warmup_steps: Number of warmup steps.
        save_freq: Checkpoint interval in steps.
        disp_freq: Interval in steps between learning-curve entries.
        stat_file: Path of the statistics file DeePMD-kit caches.
        seed: Training seed.
        start_pref_e: Initial energy prefactor in the loss.
        limit_pref_e: Final energy prefactor in the loss.
        start_pref_f: Initial force prefactor in the loss.
        limit_pref_f: Final force prefactor in the loss.

    Returns:
        The configuration dictionary, ready to be written as JSON.

    Raises:
        ValueError: If no elements or no training systems were given.
    """
    if not type_map:
        raise ValueError("type_map is empty; pass the elements the model should be trained on.")
    if not train_systems:
        raise ValueError("train_systems is empty; pass at least one training system directory.")

    model_config = {
        "type_map": list(type_map),
        "descriptor": asdict(DPA3DescriptorConfig()),
        "fitting_net": asdict(FittingNetConfig()),
    }
    learning_rate = asdict(LearningRateConfig(decay_steps=decay_steps, start_lr=start_lr, stop_lr=stop_lr))
    loss = asdict(
        LossConfig(
            start_pref_e=start_pref_e,
            limit_pref_e=limit_pref_e,
            start_pref_f=start_pref_f,
            limit_pref_f=limit_pref_f,
        )
    )

    training: dict[str, Any] = {
        "stat_file": stat_file,
        "training_data": {"systems": list(train_systems), "batch_size": batch_size},
        "numb_steps": numb_steps,
        "warmup_steps": warmup_steps,
        "gradient_max_norm": 5.0,
        "seed": seed,
        "disp_file": "lcurve.out",
        "disp_freq": disp_freq,
        "save_freq": save_freq,
        "max_ckpt_keep": 100,
    }
    if val_systems:
        training["validation_data"] = {"systems": list(val_systems), "batch_size": batch_size}

    logger.info(
        "Built DPA-3 configuration: %d elements, %d training systems, %d steps",
        len(type_map),
        len(train_systems),
        numb_steps,
    )
    return {"model": model_config, "learning_rate": learning_rate, "loss": loss, "training": training}


def write_dpa3_config(config: dict[str, Any], path: str | os.PathLike[str]) -> Path:
    """Write a fine-tuning configuration to a JSON file.

    Args:
        config: Configuration dictionary, e.g. from :func:`build_dpa3_config`.
        path: Destination file; the parent directory is created if missing.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    logger.info("Wrote fine-tuning configuration to %s", path)
    return path


def finetune(
    config_file: str | os.PathLike[str],
    pretrained_model: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    model_branch: str = DEFAULT_MODEL_BRANCH,
    log_file: str | None = None,
    use_gpu: bool = True,
    gpu_id: int | None = None,
    timeout: float | None = None,
) -> FinetuneResult:
    """Fine-tune a pretrained model by running the DeePMD-kit training command.

    The command runs to completion, so call this from a batch job for a real training run; use
    :func:`write_finetune_slurm_script` to produce that job script instead.

    Args:
        config_file: Training configuration written by :func:`write_dpa3_config`.
        pretrained_model: Checkpoint to fine-tune from. Defaults to ``$DPA3_MODEL_PATH``.
        output_dir: Working directory for the run. Defaults to the configuration file's directory.
        model_branch: Head of a multi-task checkpoint to start from.
        log_file: File under the working directory to send stdout and stderr to. ``None`` inherits
            the parent process's streams.
        use_gpu: When ``False``, ``CUDA_VISIBLE_DEVICES`` is cleared so training runs on CPU.
        gpu_id: Single GPU to expose through ``CUDA_VISIBLE_DEVICES``.
        timeout: Optional timeout in seconds.

    Returns:
        The result, including the newest checkpoint found after a successful run.

    Raises:
        RuntimeError: If the ``dp`` executable is not on ``PATH``.
        ValueError: If no pretrained checkpoint was given and the environment variable is not set.
    """
    if shutil.which(DEEPMD_EXECUTABLE) is None:
        raise RuntimeError(_DEEPMD_HINT)

    config_file = Path(config_file)
    checkpoint = resolve_pretrained_model(pretrained_model)
    work_dir = Path(output_dir) if output_dir is not None else (config_file.parent or Path("."))
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        DEEPMD_EXECUTABLE,
        "--pt",
        "train",
        str(config_file),
        "--finetune",
        checkpoint,
        "--model-branch",
        model_branch,
    ]
    cmd_str = " ".join(cmd)
    logger.info("Running: %s (cwd=%s)", cmd_str, work_dir)

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if not use_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""

    try:
        if log_file:
            log_path = work_dir / log_file
            with open(log_path, "w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    cmd, cwd=work_dir, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout, check=False
                )
            logger.info("Training log: %s", log_path)
        else:
            completed = subprocess.run(cmd, cwd=work_dir, env=env, timeout=timeout, check=False)
    except OSError as exc:
        logger.error("Could not start %s: %s", DEEPMD_EXECUTABLE, exc)
        return FinetuneResult(
            success=False,
            output_dir=str(work_dir),
            config_file=str(config_file),
            command=cmd_str,
            return_code=-1,
            error=str(exc),
        )

    success = completed.returncode == 0
    result = FinetuneResult(
        success=success,
        output_dir=str(work_dir),
        config_file=str(config_file),
        command=cmd_str,
        return_code=completed.returncode,
        checkpoint=find_latest_checkpoint(work_dir) if success else None,
    )
    if success:
        logger.info("Fine-tuning finished; newest checkpoint: %s", result.checkpoint)
    else:
        logger.error("Fine-tuning failed with exit code %d", completed.returncode)
    return result


def write_finetune_slurm_script(
    config_file: str | os.PathLike[str],
    pretrained_model: str | os.PathLike[str] | None = None,
    output_file: str | os.PathLike[str] = "run_finetune.sh",
    job_name: str = "finetune",
    n_nodes: int = 1,
    n_gpus: int = 1,
    conda_env: str = "deepmd-kit",
    model_branch: str = DEFAULT_MODEL_BRANCH,
    partition: str | None = None,
) -> Path:
    """Write a SLURM script that runs the fine-tuning command.

    The script activates a conda environment and runs ``dp --pt train``. Site-specific options
    (account, partition, module system) are deliberately not filled in: pass ``partition`` if your
    site needs one and add the rest to the generated script.

    Args:
        config_file: Training configuration to pass to the training command.
        pretrained_model: Checkpoint to fine-tune from. Defaults to ``$DPA3_MODEL_PATH``.
        output_file: Path of the script to write.
        job_name: SLURM job name.
        n_nodes: Number of nodes to request.
        n_gpus: Number of GPUs to request per node.
        conda_env: Name of the conda environment holding DeePMD-kit.
        model_branch: Head of a multi-task checkpoint to start from.
        partition: SLURM partition, omitted from the script when ``None``.

    Returns:
        The path written.

    Raises:
        ValueError: If no pretrained checkpoint was given and the environment variable is not set.
    """
    checkpoint = resolve_pretrained_model(pretrained_model)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -N {n_nodes}",
        f"#SBATCH --gres=gpu:{n_gpus}",
        "#SBATCH --output=%j.log",
        "#SBATCH --error=%j.err",
    ]
    if partition:
        lines.append(f"#SBATCH -p {partition}")
    lines += [
        "",
        "# Activate the environment that provides DeePMD-kit",
        "source $(conda info --base)/etc/profile.d/conda.sh",
        f"conda activate {conda_env}",
        "",
        "# Fine-tune the pretrained model",
        f"dp --pt train {config_file} --finetune {checkpoint} --model-branch {model_branch}",
        "",
    ]
    output_file.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote SLURM script to %s", output_file)
    return output_file


def check_training_progress(work_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Read the learning curve of a running or finished training job.

    Args:
        work_dir: Working directory of the training run, holding ``lcurve.out``.

    Returns:
        A dictionary with ``status`` (``not_started``, ``starting`` or ``running``), ``steps`` and,
        when they can be parsed, the energy and force RMSE of the last entry.
    """
    lcurve = Path(work_dir) / "lcurve.out"
    if not lcurve.is_file():
        return {"status": "not_started", "steps": 0}

    lines = lcurve.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return {"status": "starting", "steps": 0}

    parts = lines[-1].split()
    try:
        progress: dict[str, Any] = {
            "status": "running",
            "steps": int(parts[0]),
            "rmse_e": float(parts[4]) if len(parts) > 4 else None,
            "rmse_f": float(parts[5]) if len(parts) > 5 else None,
            "total_lines": len(lines) - 1,
        }
    except (ValueError, IndexError):
        return {"status": "running", "steps": 0}
    return progress


def find_latest_checkpoint(work_dir: str | os.PathLike[str]) -> str | None:
    """Find the checkpoint with the highest step count in a working directory.

    Args:
        work_dir: Working directory of the training run.

    Returns:
        Path of the newest ``model.ckpt-*.pt`` file, or ``None`` when there is none.
    """
    checkpoints = sorted(Path(work_dir).glob("model.ckpt-*.pt"), key=_checkpoint_step, reverse=True)
    return str(checkpoints[0]) if checkpoints else None


def resolve_pretrained_model(pretrained_model: str | os.PathLike[str] | None = None) -> str:
    """Resolve the pretrained checkpoint to fine-tune from.

    Args:
        pretrained_model: Explicit checkpoint path. When ``None``, ``$DPA3_MODEL_PATH`` is used.

    Returns:
        The checkpoint path as a string.

    Raises:
        ValueError: If no path was given and the environment variable is not set.
    """
    resolved = str(pretrained_model) if pretrained_model is not None else os.environ.get(MODEL_PATH_ENV_VAR, "")
    if not resolved:
        raise ValueError(
            "No pretrained checkpoint given. Pass pretrained_model=... or set the "
            f"{MODEL_PATH_ENV_VAR} environment variable to the checkpoint path."
        )
    return resolved


def _checkpoint_step(path: Path) -> int:
    """Extract the step count from a ``model.ckpt-<step>.pt`` filename, ``0`` when absent."""
    try:
        return int(path.stem.split("-")[1].split(".")[0])
    except (IndexError, ValueError):
        return 0
