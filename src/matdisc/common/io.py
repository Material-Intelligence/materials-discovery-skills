"""Structure reading and writing, and the bridge to ASE.

All structure I/O in this package goes through :class:`pymatgen.core.Structure`, which infers
the format from the file name for POSCAR/CONTCAR, ``.vasp``, ``.cif``, ``.json``, ``.yaml``,
``.xsf`` and the other formats pymatgen supports. Stages that need ASE -- relaxation, phonons,
descriptors -- convert at the boundary with :func:`to_ase` and :func:`from_ase`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from pymatgen.core import Structure

from matdisc.common.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from ase import Atoms

__all__ = ["read_structure", "read_structures", "write_structure", "to_ase", "from_ase"]

logger = get_logger(__name__)


def read_structure(path: str | Path) -> Structure:
    """Read one structure file.

    Args:
        path: Path to a structure file. The format is taken from the file name.

    Returns:
        The structure.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Structure file not found: {file_path}")
    return Structure.from_file(str(file_path))


def read_structures(paths: Iterable[str | Path]) -> list[Structure]:
    """Read several structure files.

    Args:
        paths: Paths to structure files.

    Returns:
        The structures, in the order the paths were given.

    Raises:
        FileNotFoundError: If any path does not exist.
    """
    return [read_structure(path) for path in paths]


def write_structure(structure: Structure, path: str | Path, fmt: str | None = None) -> Path:
    """Write a structure to disk.

    Args:
        structure: The structure to write.
        path: Destination file. Parent directories are created if missing.
        fmt: An explicit pymatgen format name such as ``"poscar"``, ``"cif"`` or ``"json"``.
            When omitted the format is inferred from the file name.

    Returns:
        The path written.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt:
        structure.to(filename=str(file_path), fmt=fmt)
    else:
        structure.to(filename=str(file_path))
    logger.debug("Wrote %s (%d sites)", file_path, len(structure))
    return file_path


def to_ase(structure: Structure) -> Atoms:
    """Convert a pymatgen structure to an ASE ``Atoms`` object.

    Args:
        structure: The structure to convert.

    Returns:
        The equivalent :class:`ase.Atoms`.

    Raises:
        ImportError: If ASE is not installed.
    """
    return _adaptor().get_atoms(structure)


def from_ase(atoms: Atoms) -> Structure:
    """Convert an ASE ``Atoms`` object to a pymatgen structure.

    Args:
        atoms: The atoms to convert. It must carry a cell.

    Returns:
        The equivalent :class:`pymatgen.core.Structure`.

    Raises:
        ImportError: If ASE is not installed.
    """
    return _adaptor().get_structure(atoms)


def _adaptor():
    """Return pymatgen's ASE adaptor, with a clear error when ASE is missing.

    Returns:
        The :class:`pymatgen.io.ase.AseAtomsAdaptor` class.

    Raises:
        ImportError: If ASE is not installed.
    """
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("ASE is required for structure conversion. Install it with 'pip install ase'.") from exc
    return AseAtomsAdaptor
