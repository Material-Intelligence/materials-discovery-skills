"""Structure generation with MatterGen.

MatterGen is driven through its ``mattergen-generate`` console script: this module builds the command
line, runs it, and reads the structures the run wrote back into :class:`pymatgen.core.Structure`
objects. MatterGen is an optional extra and is never imported as a Python package; the only
requirement is that ``mattergen-generate`` is on ``PATH`` and that a checkpoint directory is
available.

A MatterGen run writes ``generated_crystals_cif.zip`` (one CIF per structure),
``generated_crystals.extxyz`` (all frames) and ``generated_trajectories.zip`` into its output
directory. Structures are read from the CIF archive when it exists and from the extxyz file
otherwise; the ``*.cif`` files that a plain directory listing would look for are never written, so
counting them reports zero on a successful run.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

from monty.json import MSONable
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element

from matdisc.common.io import from_ase
from matdisc.common.logging import get_logger

logger = get_logger(__name__)

#: Console script installed by MatterGen.
MATTERGEN_EXECUTABLE = "mattergen-generate"
#: Environment variable consulted when no model path is passed explicitly.
MODEL_PATH_ENV_VAR = "MATTERGEN_MODEL_PATH"
#: Archive of CIF files a MatterGen run writes into its output directory.
CIF_ARCHIVE_NAME = "generated_crystals_cif.zip"
#: Multi-frame extxyz file a MatterGen run writes into its output directory.
EXTXYZ_NAME = "generated_crystals.extxyz"

DEFAULT_BATCH_SIZE = 24
DEFAULT_ENERGY_ABOVE_HULL = 0.1
"""Property-guidance target in eV/atom, the one value the library, the CLI and the pipeline use.

0.1 eV/atom is the metastability window a generated structure is asked to fall inside; it is a
sampling target handed to MatterGen, not a stability criterion (the hull decides that, with
``screening.tolerance``). Every entry point reads this constant, so two candidate sets produced
through different entry points come from the same distribution.
"""
DEFAULT_GUIDANCE_FACTOR = 2.0

_INSTALL_HINT = (
    f"'{MATTERGEN_EXECUTABLE}' was not found on PATH. MatterGen is an optional backend: install it "
    "from https://github.com/microsoft/mattergen (it is not part of this package's default "
    "dependencies) and make sure its console script is on PATH."
)


@dataclass
class GenerationResult(MSONable):
    """Outcome of generating structures for one chemical system.

    Attributes:
        chemical_system: Chemical system the run was conditioned on, e.g. ``"Ba-Cd-P"``.
        output_dir: Directory holding the raw MatterGen output for this system.
        structures: Structures read back from that directory.
        error: Error message when the run failed, ``None`` when it succeeded.
        reused: ``True`` when existing output was reused instead of running MatterGen again.
    """

    chemical_system: str
    output_dir: str
    structures: list[Structure] = field(default_factory=list)
    error: str | None = None
    reused: bool = False

    @property
    def n_structures(self) -> int:
        """Number of structures that were read back."""
        return len(self.structures)

    @property
    def success(self) -> bool:
        """``True`` when the run produced structures without error."""
        return self.error is None


def parse_chemical_system(chemsys: str) -> list[str]:
    """Split and validate a hyphen-separated chemical system.

    Args:
        chemsys: Chemical system such as ``"Ba-Cd-P"``. Surrounding whitespace is ignored.

    Returns:
        The element symbols in the order they were given.

    Raises:
        ValueError: If the string is empty or contains a token that is not an element symbol.
    """
    elements = [token.strip() for token in chemsys.split("-") if token.strip()]
    if not elements:
        raise ValueError(f"Empty chemical system: {chemsys!r}. Expected something like 'Ba-Cd-P'.")
    for symbol in elements:
        if not Element.is_valid_symbol(symbol):
            raise ValueError(f"{symbol!r} in chemical system {chemsys!r} is not a valid element symbol.")
    return elements


def resolve_model_path(model_path: str | os.PathLike[str] | None = None) -> str:
    """Resolve the MatterGen checkpoint directory.

    Args:
        model_path: Explicit checkpoint path. When ``None``, the ``MATTERGEN_MODEL_PATH``
            environment variable is used.

    Returns:
        The checkpoint path as a string.

    Raises:
        ValueError: If no path was given and the environment variable is not set.
    """
    resolved = str(model_path) if model_path is not None else os.environ.get(MODEL_PATH_ENV_VAR, "")
    if not resolved:
        raise ValueError(
            "No MatterGen checkpoint given. Pass model_path=... or set the "
            f"{MODEL_PATH_ENV_VAR} environment variable to the checkpoint directory."
        )
    return resolved


def generate(
    chemsys: str,
    n: int,
    output_dir: str | os.PathLike[str],
    model_path: str | os.PathLike[str] | None = None,
    energy_above_hull: float | None = DEFAULT_ENERGY_ABOVE_HULL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    guidance_factor: float = DEFAULT_GUIDANCE_FACTOR,
    skip_existing: bool = True,
    timeout: float | None = None,
) -> list[Structure]:
    """Generate candidate structures for one chemical system.

    MatterGen samples in batches, so ``ceil(n / batch_size)`` batches are requested and the first
    ``n`` structures of the run are returned. The complete raw output stays in ``output_dir``.

    Args:
        chemsys: Chemical system to condition on, e.g. ``"Ba-Cd-P"``.
        n: Number of structures to return.
        output_dir: Directory the MatterGen run writes into; created if missing.
        model_path: MatterGen checkpoint directory. Defaults to ``$MATTERGEN_MODEL_PATH``.
        energy_above_hull: Energy-above-hull value (eV/atom) to condition on. ``None`` conditions on
            the chemical system alone.
        batch_size: Number of structures sampled per batch.
        guidance_factor: Classifier-free diffusion guidance factor.
        skip_existing: When ``True`` and ``output_dir`` already holds MatterGen output, read that
            output instead of sampling again.
        timeout: Optional timeout in seconds for the MatterGen process.

    Returns:
        Up to ``n`` structures, in the order MatterGen wrote them.

    Raises:
        ValueError: If ``n`` or ``batch_size`` is not positive, or the chemical system is invalid.
        RuntimeError: If ``mattergen-generate`` is missing or exits with a non-zero status.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    parse_chemical_system(chemsys)

    output_dir = Path(output_dir)
    if skip_existing and _has_generated_output(output_dir):
        logger.info("Reusing existing MatterGen output in %s", output_dir)
        return read_generated_structures(output_dir)[:n]

    output_dir.mkdir(parents=True, exist_ok=True)
    num_batches = math.ceil(n / batch_size)
    _run_mattergen(
        chemsys=chemsys,
        output_dir=output_dir,
        model_path=resolve_model_path(model_path),
        energy_above_hull=energy_above_hull,
        batch_size=batch_size,
        num_batches=num_batches,
        guidance_factor=guidance_factor,
        timeout=timeout,
    )

    structures = read_generated_structures(output_dir)
    logger.info(
        "MatterGen produced %d structures for %s; returning %d", len(structures), chemsys, min(n, len(structures))
    )
    if len(structures) < n:
        logger.warning("Requested %d structures for %s but only %d were written", n, chemsys, len(structures))
    return structures[:n]


def generate_many(
    chemical_systems: Sequence[str],
    n: int,
    output_root: str | os.PathLike[str],
    model_path: str | os.PathLike[str] | None = None,
    energy_above_hull: float | None = DEFAULT_ENERGY_ABOVE_HULL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    guidance_factor: float = DEFAULT_GUIDANCE_FACTOR,
    skip_existing: bool = True,
    timeout: float | None = None,
) -> list[GenerationResult]:
    """Generate structures for several chemical systems, one subdirectory each.

    A failure for one system is recorded in its :class:`GenerationResult` and the remaining systems
    are still processed.

    Args:
        chemical_systems: Chemical systems to generate for, e.g. ``["Ba-Cd-P"]``.
        n: Number of structures to return per system.
        output_root: Directory under which one subdirectory per system is created.
        model_path: MatterGen checkpoint directory. Defaults to ``$MATTERGEN_MODEL_PATH``.
        energy_above_hull: Energy-above-hull value (eV/atom) to condition on.
        batch_size: Number of structures sampled per batch.
        guidance_factor: Classifier-free diffusion guidance factor.
        skip_existing: Reuse existing output directories instead of sampling again.
        timeout: Optional timeout in seconds for each MatterGen process.

    Returns:
        One :class:`GenerationResult` per chemical system, in input order.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[GenerationResult] = []
    for system in chemical_systems:
        system_dir = output_root / system.replace(" ", "")
        reused = skip_existing and _has_generated_output(system_dir)
        try:
            structures = generate(
                chemsys=system,
                n=n,
                output_dir=system_dir,
                model_path=model_path,
                energy_above_hull=energy_above_hull,
                batch_size=batch_size,
                guidance_factor=guidance_factor,
                skip_existing=skip_existing,
                timeout=timeout,
            )
            results.append(GenerationResult(system, str(system_dir), structures, reused=reused))
        except Exception as exc:  # noqa: BLE001 - one failing system must not stop the batch
            logger.error("Generation failed for %s: %s", system, exc)
            results.append(GenerationResult(system, str(system_dir), [], error=str(exc)))
    return results


def read_generated_structures(output_dir: str | os.PathLike[str]) -> list[Structure]:
    """Read the structures a MatterGen run wrote into ``output_dir``.

    The CIF archive is preferred; the multi-frame extxyz file is used when the archive is absent.

    Args:
        output_dir: Directory of a finished MatterGen run.

    Returns:
        The structures found, in file order. Empty when neither output file exists.
    """
    output_dir = Path(output_dir)
    archive = output_dir / CIF_ARCHIVE_NAME
    if archive.is_file():
        return _read_cif_archive(archive)

    extxyz = output_dir / EXTXYZ_NAME
    if extxyz.is_file():
        return _read_extxyz(extxyz)

    logger.warning("No %s or %s found in %s", CIF_ARCHIVE_NAME, EXTXYZ_NAME, output_dir)
    return []


def read_chemical_systems(path: str | os.PathLike[str]) -> list[str]:
    """Read a list of chemical systems from a text file, one per line.

    Blank lines and lines starting with ``#`` are ignored.

    Args:
        path: Path to the text file.

    Returns:
        The chemical systems in file order.
    """
    systems: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                systems.append(stripped)
    logger.info("Read %d chemical systems from %s", len(systems), path)
    return systems


def element_combinations(elements: Sequence[str], n_elements: int = 3) -> list[str]:
    """Enumerate chemical systems from a pool of elements.

    Args:
        elements: Element symbols to combine, e.g. ``["Ba", "Cd", "P"]``.
        n_elements: Number of elements per combination.

    Returns:
        Hyphen-separated chemical systems, one per combination.

    Raises:
        ValueError: If an element symbol is invalid or ``n_elements`` is not positive.
    """
    if n_elements <= 0:
        raise ValueError(f"n_elements must be positive, got {n_elements}.")
    for symbol in elements:
        if not Element.is_valid_symbol(symbol):
            raise ValueError(f"{symbol!r} is not a valid element symbol.")
    systems = ["-".join(combo) for combo in combinations(elements, n_elements)]
    logger.info("Enumerated %d %d-element systems", len(systems), n_elements)
    return systems


def _run_mattergen(
    chemsys: str,
    output_dir: Path,
    model_path: str,
    energy_above_hull: float | None,
    batch_size: int,
    num_batches: int,
    guidance_factor: float,
    timeout: float | None,
) -> None:
    """Run ``mattergen-generate`` for one chemical system.

    Raises:
        RuntimeError: If the executable is missing or the process exits with a non-zero status.
    """
    if shutil.which(MATTERGEN_EXECUTABLE) is None:
        raise RuntimeError(_INSTALL_HINT)

    properties: dict[str, Any] = {}
    if energy_above_hull is not None:
        properties["energy_above_hull"] = energy_above_hull
    properties["chemical_system"] = chemsys

    cmd = [
        MATTERGEN_EXECUTABLE,
        str(output_dir),
        f"--model_path={model_path}",
        f"--batch_size={batch_size}",
        f"--num_batches={num_batches}",
        f"--properties_to_condition_on={properties}",
        f"--diffusion_guidance_factor={guidance_factor}",
    ]
    logger.info("Running: %s", " ".join(cmd))

    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"{MATTERGEN_EXECUTABLE} exited with code {completed.returncode}: {detail[-2000:]}")


def _has_generated_output(output_dir: Path) -> bool:
    """Return whether ``output_dir`` already holds MatterGen output."""
    return (output_dir / CIF_ARCHIVE_NAME).is_file() or (output_dir / EXTXYZ_NAME).is_file()


def _read_cif_archive(archive: Path) -> list[Structure]:
    """Read every CIF member of a MatterGen CIF archive."""
    structures: list[Structure] = []
    with zipfile.ZipFile(archive) as zf:
        names = sorted((name for name in zf.namelist() if name.lower().endswith(".cif")), key=_natural_key)
        for name in names:
            text = zf.read(name).decode("utf-8")
            try:
                structures.append(Structure.from_str(text, fmt="cif"))
            except Exception as exc:  # noqa: BLE001 - one unreadable CIF must not lose the rest
                logger.warning("Could not parse %s in %s: %s", name, archive, exc)
    logger.info("Read %d structures from %s", len(structures), archive)
    return structures


def _read_extxyz(path: Path) -> list[Structure]:
    """Read every frame of a multi-frame extxyz file through the ASE bridge."""
    from ase.io import read as ase_read

    frames = ase_read(str(path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    structures = [from_ase(atoms) for atoms in frames]
    logger.info("Read %d structures from %s", len(structures), path)
    return structures


def _natural_key(name: str) -> tuple[Any, ...]:
    """Sort key that orders embedded integers numerically (``gen_2`` before ``gen_10``)."""
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))
