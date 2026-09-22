"""Conversion of DFT results into a DeePMD-kit training dataset.

VASP relaxation trajectories are read with ``dpdata`` and written in the ``deepmd/npy`` layout, one
directory per system, then split into a training and a validation set.

The expected input layout is the one :func:`matdisc.dft.inputs.write_batch_inputs` writes and VASP
then runs in: one directory per calculation, holding the OUTCAR beside the inputs::

    <dft_base_dir>/               # e.g. <pipeline work dir>/03_dft/calculations
      <system>/
        INCAR POSCAR POTCAR OUTCAR ...
      <other system>/ ...

A single OUTCAR already carries every ionic step of the relaxation, so that is one system with as
many frames as the run took.

Output trees that nest the ionic steps in one directory each are also supported, but only on
request: ``output_subdir`` names an intermediate directory between ``dft_base_dir`` and the systems,
and ``relax_subdir`` names one between a system and its step directories::

    <dft_base_dir>/
      <output_subdir>/
        <system>/
          <relax_subdir>/
            S0/OUTCAR             # one directory per ionic step
            S1/OUTCAR

Both default to ``""``, which means "not nested". A trailing ``.poscar`` on a system directory
name is stripped, because some job preparers name the directory after the input file.

``dpdata`` is an optional extra and is imported inside the functions that need it.
:func:`find_system_outcars` does the directory walk and needs no optional backend, so a layout can
be checked before any conversion is attempted.
"""

from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from monty.json import MSONable
from pymatgen.core.periodic_table import Element

from matdisc.common.logging import get_logger

logger = get_logger(__name__)

_DPDATA_HINT = (
    "dpdata is required to convert DFT output into a training set but is not installed. Install the "
    "optional extra with: pip install 'materials-discovery-skills[mlip]'"
)


@dataclass
class ConversionResult(MSONable):
    """Outcome of converting one system's DFT output.

    Attributes:
        name: System name, taken from the directory name with a trailing ``.poscar`` removed.
        n_frames: Number of labelled frames written.
        output_path: Directory the ``deepmd/npy`` data was written to, empty on failure.
        error: Error message when the conversion failed, ``None`` when it succeeded.
    """

    name: str
    n_frames: int = 0
    output_path: str = ""
    error: str | None = None

    @property
    def success(self) -> bool:
        """``True`` when frames were written without error."""
        return self.error is None and self.n_frames > 0


def find_system_outcars(
    dft_base_dir: str | os.PathLike[str],
    *,
    output_subdir: str = "",
    relax_subdir: str = "",
    exclude_patterns: Sequence[str] | None = None,
) -> dict[str, list[Path]]:
    """Find the OUTCAR files of every system under a DFT output tree.

    Each immediate subdirectory of the (optionally nested) root is one system. A system
    directory that holds an OUTCAR itself is one calculation, which is the layout
    :func:`matdisc.dft.inputs.write_batch_inputs` produces and VASP runs in. Otherwise its
    subdirectories are taken as ionic-step directories and each of their OUTCARs is collected,
    in name order.

    No optional backend is needed, so a tree can be checked before a conversion is attempted.

    Args:
        dft_base_dir: Root of the DFT calculation tree.
        output_subdir: Directory between ``dft_base_dir`` and the systems, ``""`` for none.
        relax_subdir: Directory between a system and its ionic-step directories, ``""`` for none.
        exclude_patterns: Name fragments of the directory holding an OUTCAR that exclude it. In
            the nested layout that is the ionic-step directory, so ``["S0"]`` drops the unrelaxed
            starting geometry; in the flat layout it is the calculation directory itself, so a
            fragment there drops the whole system.

    Returns:
        A mapping from system name to its OUTCAR paths, in the order they should be appended. A
        system with no usable OUTCAR maps to an empty list rather than being left out, so the
        caller can report it.

    Raises:
        FileNotFoundError: If the root directory does not exist.
    """
    root = Path(dft_base_dir)
    if output_subdir:
        root = root / output_subdir
    if not root.is_dir():
        raise FileNotFoundError(f"No such directory: {root}")

    excludes = list(exclude_patterns or [])

    def excluded(name: str) -> bool:
        return any(pattern in name for pattern in excludes)

    systems: dict[str, list[Path]] = {}
    for system_folder in _subdirectories(root):
        name = system_folder.name
        if name.endswith(".poscar"):
            name = name[: -len(".poscar")]

        search_root = system_folder / relax_subdir if relax_subdir else system_folder
        if not search_root.is_dir():
            logger.warning("%s: no directory at %s", name, search_root)
            systems[name] = []
            continue

        own_outcar = search_root / "OUTCAR"
        if own_outcar.is_file():
            systems[name] = [] if excluded(search_root.name) else [own_outcar]
            continue

        outcars = [
            step_dir / "OUTCAR"
            for step_dir in _subdirectories(search_root)
            if not excluded(step_dir.name) and (step_dir / "OUTCAR").is_file()
        ]
        systems[name] = outcars

    return systems


def convert_dft_to_deepmd(
    dft_base_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    dft_type: str = "vasp",
    exclude_patterns: Sequence[str] | None = None,
    output_subdir: str = "",
    relax_subdir: str = "",
) -> tuple[list[ConversionResult], dict[str, Any]]:
    """Convert a tree of VASP relaxations into ``deepmd/npy`` training data.

    Every OUTCAR of a system is read and appended into one labelled system, which is then written
    as a single ``deepmd/npy`` directory. In the flat layout that is one OUTCAR carrying the whole
    ionic trajectory; in a nested layout it is one OUTCAR per ionic-step directory.

    Args:
        dft_base_dir: Root of the DFT calculation tree, for example the ``calculations``
            directory ``matdisc dft-inputs`` writes.
        output_dir: Directory the converted systems are written to; created if missing.
        dft_type: DFT code the output comes from. Only ``"vasp"`` is supported.
        exclude_patterns: See :func:`find_system_outcars`.
        output_subdir: See :func:`find_system_outcars`.
        relax_subdir: See :func:`find_system_outcars`.

    Returns:
        A tuple of the per-system results and a statistics dictionary with the keys
        ``total_systems``, ``successful_systems``, ``total_frames``, ``output_dir`` and ``errors``.

    Raises:
        ImportError: If ``dpdata`` is not installed.
        ValueError: If ``dft_type`` is not supported.
        FileNotFoundError: If the root directory does not exist.
    """
    dpdata = _require_dpdata()
    if dft_type != "vasp":
        raise ValueError(f"Unsupported DFT type {dft_type!r}; only 'vasp' is supported.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    found = find_system_outcars(
        dft_base_dir,
        output_subdir=output_subdir,
        relax_subdir=relax_subdir,
        exclude_patterns=exclude_patterns,
    )

    results: list[ConversionResult] = []
    errors: list[str] = []
    total_frames = 0

    for name, outcars in found.items():
        logger.info("Converting %s", name)

        labelled = None
        for outcar in outcars:
            try:
                step = dpdata.LabeledSystem(str(outcar), fmt="outcar")
            except Exception as exc:  # noqa: BLE001 - one unreadable step must not lose the system
                message = f"{name}: could not read {outcar}: {exc}"
                errors.append(message)
                logger.warning(message)
                continue
            if step is None or len(step) == 0:
                continue
            if labelled is None:
                labelled = step
            else:
                labelled.append(step)

        if labelled is None or len(labelled) == 0:
            message = f"{name}: no usable frames"
            errors.append(message)
            logger.warning(message)
            results.append(ConversionResult(name=name, error=message))
            continue

        n_frames = len(labelled["energies"])
        system_output = output_dir / name
        try:
            labelled.to_deepmd_npy(str(system_output))
        except Exception as exc:  # noqa: BLE001 - report and continue with the next system
            message = f"{name}: could not write {system_output}: {exc}"
            errors.append(message)
            logger.error(message)
            results.append(ConversionResult(name=name, error=message))
            continue

        total_frames += n_frames
        results.append(ConversionResult(name=name, n_frames=n_frames, output_path=str(system_output)))
        logger.info("%s: %d frames -> %s", name, n_frames, system_output)

    stats: dict[str, Any] = {
        "total_systems": len(results),
        "successful_systems": sum(1 for result in results if result.success),
        "total_frames": total_frames,
        "output_dir": str(output_dir),
        "errors": errors,
    }
    logger.info(
        "Converted %d of %d systems, %d frames total",
        stats["successful_systems"],
        stats["total_systems"],
        total_frames,
    )
    return results, stats


def convert_outcar(
    outcar_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    system_name: str | None = None,
) -> ConversionResult:
    """Convert a single VASP OUTCAR into ``deepmd/npy`` training data.

    Args:
        outcar_path: Path to the OUTCAR file.
        output_dir: Directory the converted system is written under; created if missing.
        system_name: Name of the written subdirectory. Defaults to the OUTCAR's parent directory name.

    Returns:
        The conversion result, carrying the error message when the conversion failed.

    Raises:
        ImportError: If ``dpdata`` is not installed.
    """
    dpdata = _require_dpdata()
    outcar_path = Path(outcar_path)
    name = system_name or outcar_path.parent.name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        labelled = dpdata.LabeledSystem(str(outcar_path), fmt="outcar")
    except Exception as exc:  # noqa: BLE001 - reported through the result object
        logger.warning("%s: could not read %s: %s", name, outcar_path, exc)
        return ConversionResult(name=name, error=str(exc))

    if labelled is None or len(labelled) == 0:
        logger.warning("%s: no usable frames in %s", name, outcar_path)
        return ConversionResult(name=name, error="no usable frames")

    system_output = output_dir / name
    labelled.to_deepmd_npy(str(system_output))
    n_frames = len(labelled["energies"])
    logger.info("%s: %d frames -> %s", name, n_frames, system_output)
    return ConversionResult(name=name, n_frames=n_frames, output_path=str(system_output))


def split_train_val(
    deepmd_dir: str | os.PathLike[str],
    train_dir: str | os.PathLike[str],
    val_dir: str | os.PathLike[str],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split converted systems into a training and a validation set.

    The split is by system, not by frame. With ``val_ratio`` above zero at least one system is
    held out, so a split needs at least two systems; ``val_ratio = 0`` means no validation set
    and every system trains.

    Args:
        deepmd_dir: Directory holding one ``deepmd/npy`` directory per system.
        train_dir: Destination directory for the training systems; created if missing.
        val_dir: Destination directory for the validation systems; created if missing.
        val_ratio: Fraction of systems held out for validation. ``0`` holds nothing out.
        seed: Seed for the shuffle, so the split is reproducible.

    Returns:
        A tuple of the training and validation system paths. The validation list is empty when
        ``val_ratio`` is zero.

    Raises:
        ValueError: If ``val_ratio`` is not in ``[0, 1)``, no systems were found, or a
            validation set was asked for and only one system exists -- which would leave
            nothing to train on.
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}.")

    deepmd_dir = Path(deepmd_dir)
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    systems = sorted(entry.name for entry in _subdirectories(deepmd_dir))
    if not systems:
        raise ValueError(f"No converted systems found in {deepmd_dir}.")
    if val_ratio > 0.0 and len(systems) < 2:
        raise ValueError(
            f"Only one converted system in {deepmd_dir}, and holding it out for validation would leave nothing "
            "to train on. Convert more calculations, or pass val_ratio=0 to train on the single system."
        )

    random.Random(seed).shuffle(systems)
    n_val = max(1, int(len(systems) * val_ratio)) if val_ratio > 0.0 else 0
    val_systems, train_systems = systems[:n_val], systems[n_val:]

    train_paths = [str(_copy_system(deepmd_dir / name, train_dir / name)) for name in train_systems]
    val_paths = [str(_copy_system(deepmd_dir / name, val_dir / name)) for name in val_systems]

    logger.info("Split %d systems: %d training, %d validation", len(systems), len(train_paths), len(val_paths))
    return train_paths, val_paths


def get_type_map(deepmd_dir: str | os.PathLike[str]) -> list[str]:
    """Collect the elements present in converted systems, ordered by atomic number.

    Args:
        deepmd_dir: Directory holding one ``deepmd/npy`` directory per system.

    Returns:
        The element symbols, sorted by atomic number.

    Raises:
        ValueError: If a ``type_map.raw`` file names something that is not an element.
    """
    elements: set[str] = set()
    for system_dir in _subdirectories(Path(deepmd_dir)):
        type_map_file = system_dir / "type_map.raw"
        if not type_map_file.is_file():
            continue
        with open(type_map_file, encoding="utf-8") as handle:
            for line in handle:
                symbol = line.strip()
                if symbol:
                    elements.add(symbol)

    for symbol in elements:
        if not Element.is_valid_symbol(symbol):
            raise ValueError(f"{symbol!r} in {deepmd_dir} is not a valid element symbol.")

    ordered = sorted(elements, key=lambda symbol: Element(symbol).Z)
    logger.info("Type map from %s: %s", deepmd_dir, ordered)
    return ordered


def _require_dpdata() -> Any:
    """Import ``dpdata`` on demand.

    Raises:
        ImportError: If ``dpdata`` is not installed.
    """
    try:
        import dpdata
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError(_DPDATA_HINT) from exc
    return dpdata


def _subdirectories(path: Path) -> Iterator[Path]:
    """Yield the immediate subdirectories of ``path`` in name order."""
    if not path.is_dir():
        return
    for entry in sorted(path.iterdir(), key=lambda item: item.name):
        if entry.is_dir():
            yield entry


def _copy_system(source: Path, destination: Path) -> Path:
    """Copy one converted system directory, replacing an existing copy."""
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination
