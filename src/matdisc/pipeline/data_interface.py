"""Structure handover between stages, and validation of what a stage produced.

Every stage of the pipeline hands the next one a directory of structure files. This module
holds the file conventions that make that work: how a structure file is recognised, how a
directory is read into :class:`~pymatgen.core.Structure` objects, how those objects are
written back out, and how a stage directory is checked for the files the next stage expects.

Structures are read and written through :mod:`matdisc.common.io`, so pymatgen decides the
format. The one case pymatgen does not cover is a multi-frame ``extxyz`` file -- what
MatterGen writes -- which is read through ASE and converted with the ASE bridge.

The names below are the contract between :mod:`matdisc.pipeline.stages` and
:func:`validate_stage_output`; a stage writes them and the validator looks for them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from monty.json import MSONable
from pymatgen.core import Structure

from matdisc.common.io import from_ase, read_structure, write_structure
from matdisc.common.logging import get_logger
from matdisc.pipeline.config import STAGE_ORDER

LOGGER = get_logger(__name__)

STRUCTURE_PATTERNS: tuple[str, ...] = ("*.vasp", "*.cif", "POSCAR*", "CONTCAR*", "*.xyz", "*.extxyz")
"""Glob patterns that match a structure file, in the order they are searched."""

MULTI_FRAME_SUFFIXES: tuple[str, ...] = (".extxyz",)
"""Suffixes of files that may hold more than one structure."""

STRUCTURES_SUBDIR = "structures"
"""Subdirectory a stage writes its structure output into."""

SELECTED_SUBDIR = "selected"
"""Subdirectory the clustering stage writes the representative structures into."""

STABLE_SUBDIR = "stable"
"""Subdirectory the screening stage writes the structures at or below the hull into."""

CANDIDATES_CSV = "candidates.csv"
"""Relaxed candidate energies written by the screening stage."""

COMPETING_CSV = "competing_phases.csv"
"""Competing-phase table written by the competing stage."""

COMPETING_RELAXED_CSV = "competing_relaxed.csv"
"""Competing-phase energies after relaxation with the screening calculator."""

HULL_CSV = "hull.csv"
"""Convex-hull result written by the screening stage."""

PHONON_CSV = "phonons.csv"
"""Dynamical-stability summary written by the phonon stage."""

RESULT_JSON = "pipeline_result.json"
"""Machine-readable summary of a run, written into the working directory."""

__all__ = [
    "CANDIDATES_CSV",
    "COMPETING_CSV",
    "COMPETING_RELAXED_CSV",
    "HULL_CSV",
    "MULTI_FRAME_SUFFIXES",
    "PHONON_CSV",
    "RESULT_JSON",
    "SELECTED_SUBDIR",
    "STABLE_SUBDIR",
    "STRUCTURES_SUBDIR",
    "STRUCTURE_PATTERNS",
    "StageCheck",
    "collect_structures",
    "detect_structure_format",
    "iter_structure_files",
    "load_structure_input",
    "load_structures",
    "read_frames",
    "validate_pipeline",
    "validate_stage_output",
    "write_structures",
]


def detect_structure_format(path: str | Path) -> str:
    """Name the format of a structure file from its name.

    Args:
        path: The file to classify.

    Returns:
        ``"vasp"``, ``"cif"``, ``"xyz"``, ``"extxyz"`` or ``"unknown"``.
    """
    file_path = Path(path)
    name = file_path.name.upper()
    if name.startswith(("POSCAR", "CONTCAR")):
        return "vasp"
    suffix = file_path.suffix.lower()
    if suffix in (".vasp", ".poscar"):
        return "vasp"
    if suffix == ".cif":
        return "cif"
    if suffix == ".extxyz":
        return "extxyz"
    if suffix == ".xyz":
        return "xyz"
    return "unknown"


def iter_structure_files(
    directory: str | Path,
    recursive: bool = True,
    patterns: Iterable[str] = STRUCTURE_PATTERNS,
) -> list[Path]:
    """List the structure files of a directory.

    Args:
        directory: Directory to search.
        recursive: Search subdirectories as well.
        patterns: Glob patterns to match, in order. A file matched twice is listed once.

    Returns:
        The matching files, sorted, without duplicates.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Structure directory not found: {root}")

    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = root.rglob(pattern) if recursive else root.glob(pattern)
        for path in sorted(matches):
            if path.is_file() and path not in seen:
                seen.add(path)
                found.append(path)
    return found


def read_frames(path: str | Path) -> list[Structure]:
    """Read every structure held by one file.

    Single-structure files go through pymatgen. A multi-frame ``extxyz`` file -- the format
    MatterGen writes -- is read through ASE and converted with the ASE bridge.

    Args:
        path: The file to read.

    Returns:
        One or more structures, in file order.

    Raises:
        ValueError: If the file holds no structure that could be read.
    """
    file_path = Path(path).expanduser()
    if file_path.suffix.lower() in MULTI_FRAME_SUFFIXES:
        from ase.io import read as ase_read

        frames = ase_read(str(file_path), index=":")
        if not isinstance(frames, list):
            frames = [frames]
        if not frames:
            raise ValueError(f"No structures in {file_path}")
        return [from_ase(atoms) for atoms in frames]
    return [read_structure(file_path)]


def load_structures(
    directory: str | Path,
    recursive: bool = True,
    patterns: Iterable[str] = STRUCTURE_PATTERNS,
) -> dict[str, Structure]:
    """Read a directory of structure files into named structures.

    Args:
        directory: Directory to read.
        recursive: Search subdirectories as well.
        patterns: Glob patterns to match.

    Returns:
        A mapping from name to structure. The name is the file stem, with an index appended
        for each frame of a multi-frame file, and a numeric suffix appended when two files
        would otherwise collide.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    structures: dict[str, Structure] = {}
    for path in iter_structure_files(directory, recursive=recursive, patterns=patterns):
        try:
            frames = read_frames(path)
        except Exception as error:  # noqa: BLE001 - one unreadable file must not stop the sweep
            LOGGER.warning("Skipping %s: %s", path, error)
            continue
        stem = path.stem or path.name
        for index, structure in enumerate(frames):
            name = stem if len(frames) == 1 else f"{stem}_{index}"
            structures[_unique_name(name, structures)] = structure
    LOGGER.info("Read %d structure(s) from %s", len(structures), directory)
    return structures


def _unique_name(name: str, taken: Mapping[str, Any]) -> str:
    """Return ``name``, or the first ``name_2``, ``name_3``, ... that is free.

    Args:
        name: The preferred name.
        taken: Names already in use.

    Returns:
        A name not present in ``taken``.
    """
    if name not in taken:
        return name
    index = 2
    while f"{name}_{index}" in taken:
        index += 1
    return f"{name}_{index}"


def load_structure_input(path: str | Path, recursive: bool = True) -> dict[str, Structure]:
    """Read a structure file or a directory of structure files.

    Every stage accepts either, so a single candidate can be run through a stage that
    normally consumes a directory.

    Args:
        path: A structure file or a directory holding some.
        recursive: Search subdirectories, when ``path`` is a directory.

    Returns:
        A mapping from name to structure, empty when nothing could be read.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    target = Path(path).expanduser()
    if target.is_dir():
        return load_structures(target, recursive=recursive)
    if not target.is_file():
        raise FileNotFoundError(f"No such structure file or directory: {target}")
    try:
        frames = read_frames(target)
    except Exception as error:  # noqa: BLE001 - reported as an empty result, like a directory sweep
        LOGGER.warning("Could not read %s: %s", target, error)
        return {}
    stem = target.stem or target.name
    return {(stem if len(frames) == 1 else f"{stem}_{index}"): frame for index, frame in enumerate(frames)}


def write_structures(
    structures: Mapping[str, Structure],
    output_dir: str | Path,
    fmt: str = "poscar",
    suffix: str = ".vasp",
) -> list[Path]:
    """Write named structures into a directory, one file each.

    Args:
        structures: Mapping from name to structure.
        output_dir: Directory to write into. It is created if missing.
        fmt: pymatgen format name.
        suffix: File suffix to give each file.

    Returns:
        The paths written, in the order of the mapping.
    """
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, structure in structures.items():
        written.append(write_structure(structure, out / f"{name}{suffix}", fmt=fmt))
    LOGGER.info("Wrote %d structure(s) to %s", len(written), out)
    return written


def collect_structures(
    input_dir: str | Path,
    output_dir: str | Path,
    fmt: str = "poscar",
    suffix: str = ".vasp",
    recursive: bool = True,
    prefix: str | None = None,
) -> list[Path]:
    """Read every structure below a directory and write them out in one format.

    This is the handover between a stage that writes CIF or ``extxyz`` -- MatterGen does
    both -- and a stage that reads POSCAR files.

    Args:
        input_dir: Directory to read.
        output_dir: Directory to write into.
        fmt: pymatgen format name for the output files.
        suffix: Suffix of the output files.
        recursive: Search subdirectories of ``input_dir``.
        prefix: Prepended to every output name, for example the chemical system. ``None``
            keeps the input names.

    Returns:
        The paths written.

    Raises:
        FileNotFoundError: If ``input_dir`` does not exist.
    """
    structures = load_structures(input_dir, recursive=recursive)
    if prefix:
        structures = {f"{prefix}_{name}": structure for name, structure in structures.items()}
    return write_structures(structures, output_dir, fmt=fmt, suffix=suffix)


@dataclass
class StageCheck(MSONable):
    """What a stage directory holds, and whether the next stage can use it.

    Attributes:
        stage: The stage name.
        output_dir: The directory that was checked.
        exists: Whether that directory exists.
        valid: Whether the files the next stage needs are present.
        stats: Counts found, for example ``n_structures``.
        messages: Why the check failed, or what is missing.
    """

    stage: str
    output_dir: str
    exists: bool = False
    valid: bool = False
    stats: dict[str, Any] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serialise the check.

        Returns:
            A JSON-serialisable dictionary.
        """
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "stage": self.stage,
            "output_dir": self.output_dir,
            "exists": self.exists,
            "valid": self.valid,
            "stats": dict(self.stats),
            "messages": list(self.messages),
        }


def _count(directory: Path, pattern: str, recursive: bool = False) -> int:
    """Count the files of a directory matching a pattern.

    Args:
        directory: Directory to look in.
        pattern: Glob pattern.
        recursive: Search subdirectories as well.

    Returns:
        The number of matching files, or 0 when the directory does not exist.
    """
    if not directory.is_dir():
        return 0
    matches = directory.rglob(pattern) if recursive else directory.glob(pattern)
    return sum(1 for path in matches if path.is_file())


def validate_stage_output(stage: str, output_dir: str | Path) -> StageCheck:
    """Check that a stage directory holds what the next stage reads.

    Args:
        stage: A stage name from :data:`matdisc.pipeline.config.STAGE_ORDER`.
        output_dir: The stage's output directory.

    Returns:
        A :class:`StageCheck`. Nothing is raised for a missing or empty directory; the
        result says so instead.

    Raises:
        ValueError: If the stage name is unknown.
    """
    if stage not in STAGE_ORDER:
        raise ValueError(f"Unknown stage {stage!r}. Known stages: {', '.join(STAGE_ORDER)}.")

    path = Path(output_dir).expanduser().resolve()
    check = StageCheck(stage=stage, output_dir=str(path), exists=path.is_dir())
    if not check.exists:
        check.messages.append("Output directory does not exist")
        return check

    if stage == "generation":
        count = _count(path / STRUCTURES_SUBDIR, "*", recursive=False)
        check.stats["n_structures"] = count
        check.valid = count > 0
        if not check.valid:
            check.messages.append(f"No structure files in {path / STRUCTURES_SUBDIR}")

    elif stage == "clustering":
        count = _count(path / SELECTED_SUBDIR, "*", recursive=False)
        check.stats["n_selected"] = count
        check.valid = count > 0
        if not check.valid:
            check.messages.append(f"No selected structures in {path / SELECTED_SUBDIR}")

    elif stage == "dft":
        prepared = sum(1 for incar in path.rglob("INCAR") if incar.is_file())
        finished = sum(1 for marker in path.rglob("vasprun.xml") if marker.is_file())
        check.stats["n_prepared"] = prepared
        check.stats["n_finished"] = finished
        check.valid = prepared > 0
        if not check.valid:
            check.messages.append("No calculation directories were prepared")
        elif finished == 0:
            check.messages.append("Inputs are prepared but no calculation has finished yet")

    elif stage == "finetune":
        checkpoints = _count(path, "model.ckpt*", recursive=True)
        config_files = _count(path, "*.json", recursive=False)
        check.stats["n_checkpoints"] = checkpoints
        check.stats["n_config_files"] = config_files
        check.valid = config_files > 0
        if not check.valid:
            check.messages.append("No training configuration was written")
        elif checkpoints == 0:
            check.messages.append("A configuration was written but training has not produced a checkpoint")

    elif stage == "competing":
        has_csv = (path / COMPETING_CSV).is_file()
        structures = _count(path / STRUCTURES_SUBDIR, "*", recursive=False)
        check.stats["n_structures"] = structures
        check.valid = has_csv
        if not has_csv:
            check.messages.append(f"{COMPETING_CSV} is missing")

    elif stage == "screening":
        has_hull = (path / HULL_CSV).is_file()
        stable = _count(path / STABLE_SUBDIR, "*", recursive=False)
        check.stats["n_stable"] = stable
        check.valid = has_hull
        if not has_hull:
            check.messages.append(f"{HULL_CSV} is missing")

    elif stage == "phonons":
        has_summary = (path / PHONON_CSV).is_file()
        spectra = _count(path, "band_structure.dat", recursive=True)
        check.stats["n_spectra"] = spectra
        check.valid = has_summary
        if not has_summary:
            check.messages.append(f"{PHONON_CSV} is missing")

    return check


def validate_pipeline(work_dir: str | Path, stages: Sequence[str] = STAGE_ORDER) -> dict[str, StageCheck]:
    """Check every stage directory of a run.

    Args:
        work_dir: The working directory of the run.
        stages: Stages to check.

    Returns:
        A mapping from stage name to its :class:`StageCheck`.

    Raises:
        ValueError: If a stage name is unknown.
    """
    root = Path(work_dir).expanduser().resolve()
    results: dict[str, StageCheck] = {}
    for stage in stages:
        if stage not in STAGE_ORDER:
            raise ValueError(f"Unknown stage {stage!r}. Known stages: {', '.join(STAGE_ORDER)}.")
        index = STAGE_ORDER.index(stage) + 1
        results[stage] = validate_stage_output(stage, root / f"{index:02d}_{stage}")
    return results
