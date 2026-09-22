"""Plain SLURM submission scripts for VASP runs.

The original workflow submitted VASP through an in-house job manager whose cluster template
carried the module stack, MPI tuning and login-node name of one particular machine. None of that
is portable, so this module writes a plain ``sbatch`` script instead: the resource request comes
from :class:`SlurmSettings`, the module loads and environment exports are whatever the caller
passes, and the VASP executable is taken from the ``VASP_CMD`` environment variable at run time::

    export VASP_CMD=/path/to/vasp_std

The resource presets in :data:`RESOURCE_PRESETS` are the node/core/walltime shapes the original
templates used; they are starting points, not settings for any particular cluster.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from monty.json import MSONable

from matdisc.common.logging import get_logger

__all__ = [
    "RESOURCE_PRESETS",
    "SlurmSettings",
    "cancel_jobs",
    "format_walltime",
    "get_preset",
    "query_jobs",
    "render_slurm_script",
    "settings_from_mapping",
    "submit_directory",
    "submit_job",
    "write_slurm_script",
]

logger = get_logger(__name__)

#: Resource shapes carried over from the original cluster templates. ``walltime_minutes`` is in
#: minutes, as in the original.
RESOURCE_PRESETS: dict[str, dict[str, int]] = {
    "small": {"nodes": 1, "ntasks": 48, "walltime_minutes": 120},
    "medium": {"nodes": 1, "ntasks": 96, "walltime_minutes": 600},
    "large": {"nodes": 2, "ntasks": 192, "walltime_minutes": 1440},
    "hse": {"nodes": 2, "ntasks": 192, "walltime_minutes": 2880},
}

_JOB_ID_PATTERN = re.compile(r"Submitted batch job (\d+)")


def format_walltime(minutes: int) -> str:
    """Format a walltime in minutes as a SLURM ``--time`` string.

    Args:
        minutes: Walltime in minutes. Must be positive.

    Returns:
        ``"HH:MM:SS"``, or ``"D-HH:MM:SS"`` for walltimes of a day or more.

    Raises:
        ValueError: If ``minutes`` is not positive.
    """
    if minutes <= 0:
        raise ValueError(f"walltime must be positive, got {minutes}")
    days, rest = divmod(int(minutes), 24 * 60)
    hours, mins = divmod(rest, 60)
    if days:
        return f"{days}-{hours:02d}:{mins:02d}:00"
    return f"{hours:02d}:{mins:02d}:00"


def get_preset(name: str) -> dict[str, int]:
    """Return a copy of one resource preset.

    Args:
        name: One of the keys of :data:`RESOURCE_PRESETS`.

    Returns:
        The preset's ``nodes``, ``ntasks`` and ``walltime_minutes``.

    Raises:
        KeyError: If ``name`` is not a known preset.
    """
    if name not in RESOURCE_PRESETS:
        raise KeyError(f"Unknown preset {name!r}; expected one of {', '.join(sorted(RESOURCE_PRESETS))}")
    return dict(RESOURCE_PRESETS[name])


@dataclass
class SlurmSettings(MSONable):
    """The resource request and environment of one SLURM job.

    Attributes:
        job_name: Value of ``--job-name``.
        partition: Value of ``--partition``. Omitted when ``None``.
        nodes: Number of nodes.
        ntasks: Total number of MPI ranks.
        cpus_per_task: Value of ``--cpus-per-task``.
        walltime_minutes: Walltime in minutes, formatted by :func:`format_walltime`.
        account: Value of ``--account``. Omitted when ``None``.
        qos: Value of ``--qos``. Omitted when ``None``.
        stdout: Value of ``--output``.
        stderr: Value of ``--error``.
        modules: Module names to ``module load``, in order. Site-specific; empty by default.
        exports: ``KEY=VALUE`` strings written as ``export`` lines.
        launcher: ``"mpirun"``, ``"srun"`` or ``""`` to run the executable directly.
        extra_directives: Further ``#SBATCH`` directives, without the ``#SBATCH`` prefix.
        extra_lines: Shell lines inserted just before the VASP command.
    """

    job_name: str = "vasp"
    partition: str | None = None
    nodes: int = 1
    ntasks: int = 96
    cpus_per_task: int = 1
    walltime_minutes: int = 600
    account: str | None = None
    qos: str | None = None
    stdout: str = "slurm-%j.out"
    stderr: str = "slurm-%j.err"
    modules: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=lambda: ["OMP_NUM_THREADS=1"])
    launcher: str = "mpirun"
    extra_directives: list[str] = field(default_factory=list)
    extra_lines: list[str] = field(default_factory=list)

    @classmethod
    def from_preset(cls, preset: str, **overrides: Any) -> SlurmSettings:
        """Build settings from a resource preset.

        Args:
            preset: One of the keys of :data:`RESOURCE_PRESETS`.
            **overrides: Fields to override on top of the preset.

        Returns:
            The settings object.
        """
        values = get_preset(preset)
        values.update(overrides)
        return cls(**values)


def _launcher_command(settings: SlurmSettings) -> str:
    """Return the shell command that starts VASP.

    Args:
        settings: The job settings.

    Returns:
        A shell command using the ``VASP_CMD`` environment variable.
    """
    launcher = (settings.launcher or "").strip()
    if launcher == "mpirun":
        return f'mpirun -np {settings.ntasks} "$VASP_CMD"'
    if launcher == "srun":
        return f'srun -n {settings.ntasks} "$VASP_CMD"'
    if not launcher:
        return '"$VASP_CMD"'
    return f'{launcher} "$VASP_CMD"'


def render_slurm_script(settings: SlurmSettings | None = None, **overrides: Any) -> str:
    """Render a SLURM submission script as text.

    Args:
        settings: The job settings. Defaults are used when ``None``.
        **overrides: Fields of :class:`SlurmSettings` to override.

    Returns:
        The script text, ending in a newline.
    """
    settings = settings or SlurmSettings()
    if overrides:
        settings = replace(settings, **overrides)

    lines: list[str] = ["#!/bin/bash"]
    lines.append(f"#SBATCH --job-name={settings.job_name}")
    if settings.partition:
        lines.append(f"#SBATCH --partition={settings.partition}")
    if settings.account:
        lines.append(f"#SBATCH --account={settings.account}")
    if settings.qos:
        lines.append(f"#SBATCH --qos={settings.qos}")
    lines.append(f"#SBATCH --nodes={settings.nodes}")
    lines.append(f"#SBATCH --ntasks={settings.ntasks}")
    lines.append(f"#SBATCH --cpus-per-task={settings.cpus_per_task}")
    lines.append(f"#SBATCH --time={format_walltime(settings.walltime_minutes)}")
    lines.append(f"#SBATCH --output={settings.stdout}")
    lines.append(f"#SBATCH --error={settings.stderr}")
    lines.extend(f"#SBATCH {directive}" for directive in settings.extra_directives)

    lines.append("")
    lines.append("set -euo pipefail")
    lines.append("")
    lines.append("# Site-specific setup: module loads and environment variables for your cluster.")
    lines.extend(f"module load {module}" for module in settings.modules)
    lines.extend(f"export {export}" for export in settings.exports)

    lines.append("")
    lines.append("# VASP executable, for example: export VASP_CMD=/path/to/vasp_std")
    lines.append('if [ -z "${VASP_CMD:-}" ]; then')
    lines.append('    echo "VASP_CMD is not set; point it at your vasp_std binary." >&2')
    lines.append("    exit 1")
    lines.append("fi")

    if settings.extra_lines:
        lines.append("")
        lines.extend(settings.extra_lines)

    lines.append("")
    lines.append(_launcher_command(settings))
    lines.append("")
    return "\n".join(lines)


def write_slurm_script(
    workdir: str | Path,
    settings: SlurmSettings | None = None,
    *,
    script_name: str = "submit.slurm",
    preset: str | None = None,
    **overrides: Any,
) -> Path:
    """Write a SLURM submission script into a calculation directory.

    Args:
        workdir: The calculation directory. It is created if it does not exist.
        settings: The job settings. Defaults are used when ``None``.
        script_name: File name of the script.
        preset: A key of :data:`RESOURCE_PRESETS` to start from, applied before ``overrides``
            and only when ``settings`` is ``None``.
        **overrides: Fields of :class:`SlurmSettings` to override.

    Returns:
        The path of the script that was written. The file is made executable.
    """
    if settings is None and preset is not None:
        settings = SlurmSettings.from_preset(preset)

    out_dir = Path(workdir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = out_dir / script_name
    script_path.write_text(render_slurm_script(settings, **overrides), encoding="utf-8")
    script_path.chmod(0o755)
    logger.info("Wrote SLURM script %s", script_path)
    return script_path


def _run(command: list[str], cwd: Path | None = None, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    """Run a scheduler command and return the completed process.

    Args:
        command: The command and its arguments.
        cwd: Working directory for the command.
        timeout: Timeout in seconds.

    Returns:
        The completed process.

    Raises:
        FileNotFoundError: If the executable is not on ``PATH``.
    """
    if shutil.which(command[0]) is None:
        raise FileNotFoundError(f"{command[0]} was not found on PATH; run this on a machine with SLURM.")
    return subprocess.run(  # noqa: S603 - fixed argument lists, no shell
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def submit_job(script: str | Path, workdir: str | Path | None = None, dry_run: bool = False) -> str | None:
    """Submit one SLURM script with ``sbatch``.

    Args:
        script: Path of the submission script.
        workdir: Directory to submit from. Defaults to the script's own directory.
        dry_run: When true, log what would be submitted and return ``None``.

    Returns:
        The SLURM job id, or ``None`` on a dry run or a failed submission.

    Raises:
        FileNotFoundError: If the script does not exist, or ``sbatch`` is not on ``PATH``.
    """
    script_path = Path(script).expanduser().resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"Submission script not found: {script_path}")
    cwd = Path(workdir).expanduser().resolve() if workdir else script_path.parent

    if dry_run:
        logger.info("Dry run: would submit %s from %s", script_path, cwd)
        return None

    result = _run(["sbatch", script_path.name], cwd=cwd)
    if result.returncode != 0:
        logger.error("sbatch failed for %s: %s", script_path, result.stderr.strip())
        return None

    match = _JOB_ID_PATTERN.search(result.stdout)
    if match is None:
        logger.warning("Could not read a job id from the sbatch output: %s", result.stdout.strip())
        return None

    job_id = match.group(1)
    logger.info("Submitted %s as job %s", script_path, job_id)
    return job_id


def submit_directory(
    root: str | Path,
    script_name: str = "submit.slurm",
    dry_run: bool = False,
) -> dict[str, str | None]:
    """Submit every calculation directory below a root that holds a submission script.

    Args:
        root: Directory whose immediate subdirectories are calculation directories.
        script_name: File name of the submission script in each subdirectory.
        dry_run: When true, nothing is submitted.

    Returns:
        A mapping from subdirectory name to job id, or to ``None`` when that submission was a
        dry run or failed.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {root_path}")

    jobs: dict[str, str | None] = {}
    for sub in sorted(p for p in root_path.iterdir() if p.is_dir()):
        script = sub / script_name
        if not script.is_file():
            logger.debug("No %s in %s, skipping", script_name, sub)
            continue
        jobs[sub.name] = submit_job(script, workdir=sub, dry_run=dry_run)

    logger.info("Handled %d submission director%s under %s", len(jobs), "y" if len(jobs) == 1 else "ies", root_path)
    return jobs


def query_jobs(job_ids: Iterable[str] | None = None, user: str | None = None) -> dict[str, str]:
    """Return the SLURM state of jobs, read from ``squeue``.

    Jobs that have already left the queue are absent from the result; check the calculation
    directories with :func:`matdisc.dft.outputs.check_calculation_status` for those.

    Args:
        job_ids: Job ids to query. Mutually exclusive with ``user``.
        user: Query all jobs of this user instead of specific ids.

    Returns:
        A mapping from job id to SLURM state, for example ``{"12345": "RUNNING"}``.

    Raises:
        ValueError: If both ``job_ids`` and ``user`` are given, or neither.
        FileNotFoundError: If ``squeue`` is not on ``PATH``.
    """
    ids = [str(job_id) for job_id in job_ids] if job_ids is not None else []
    if bool(ids) == bool(user):
        raise ValueError("Pass either job_ids or user, not both and not neither.")

    command = ["squeue", "-h", "-o", "%i %T"]
    command += ["-j", ",".join(ids)] if ids else ["-u", str(user)]

    result = _run(command)
    if result.returncode != 0:
        logger.error("squeue failed: %s", result.stderr.strip())
        return {}

    states: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            states[parts[0]] = parts[1]
    return states


def cancel_jobs(job_ids: Iterable[str]) -> bool:
    """Cancel SLURM jobs with ``scancel``.

    Args:
        job_ids: The job ids to cancel.

    Returns:
        True when ``scancel`` reported success, or when there was nothing to cancel.

    Raises:
        FileNotFoundError: If ``scancel`` is not on ``PATH``.
    """
    ids = [str(job_id) for job_id in job_ids]
    if not ids:
        logger.info("No job ids given, nothing to cancel.")
        return True

    result = _run(["scancel", *ids])
    if result.returncode != 0:
        logger.error("scancel failed: %s", result.stderr.strip())
        return False

    logger.info("Cancelled %d job(s)", len(ids))
    return True


def settings_from_mapping(mapping: Mapping[str, Any]) -> SlurmSettings:
    """Build :class:`SlurmSettings` from a plain mapping, ignoring unknown keys.

    Args:
        mapping: Field names and values, for example a block of a YAML configuration file.

    Returns:
        The settings object.
    """
    known = {f.name for f in SlurmSettings.__dataclass_fields__.values()}
    unknown = sorted(set(mapping) - known)
    if unknown:
        logger.warning("Ignoring unknown SLURM settings: %s", ", ".join(unknown))
    return SlurmSettings(**{key: value for key, value in mapping.items() if key in known})
