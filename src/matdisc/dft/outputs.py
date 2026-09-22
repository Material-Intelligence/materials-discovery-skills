"""VASP output parsing via :mod:`pymatgen.io.vasp.outputs`.

A finished calculation directory is read with :func:`read_calculation`, which prefers
``vasprun.xml`` (it carries the forces, the stresses and the eigenvalues) and falls back to
``OUTCAR`` plus ``CONTCAR`` when the XML is missing or truncated.

:func:`collect_results` walks a tree of calculation directories and returns a table whose
``id``, ``composition`` and ``energy_per_atom`` columns are exactly the candidate columns that
:func:`matdisc.screening.hull.compute_e_above_hull` expects, so a DFT run can be fed straight
into the convex-hull stage. :func:`export_for_finetuning` writes the energies, forces and
stresses as JSON for the fine-tuning stage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from monty.json import MSONable
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Outcar, Vasprun

from matdisc.common.logging import get_logger

__all__ = [
    "CALCULATION_STATES",
    "VaspResult",
    "check_calculation_status",
    "collect_results",
    "export_for_finetuning",
    "parse_outcar",
    "parse_vasprun",
    "read_calculation",
]

logger = get_logger(__name__)

#: The states :func:`check_calculation_status` can report.
CALCULATION_STATES = ("pending", "running", "completed", "failed", "unknown")

#: Marker VASP writes at the end of a run that reached its normal end.
_COMPLETION_MARKER = "General timing and accounting"

#: Substrings that mark an aborted run.
_FAILURE_MARKERS = ("Error", "STOP")

#: Number of trailing bytes of OUTCAR inspected by :func:`check_calculation_status`. Reading the
#: tail keeps the scan cheap on the multi-gigabyte OUTCARs that long relaxations produce.
_OUTCAR_TAIL_BYTES = 64 * 1024


@dataclass
class VaspResult(MSONable):
    """The parsed outcome of one VASP calculation.

    Attributes:
        directory: The calculation directory this result came from.
        source: File the result was read from, ``"vasprun.xml"`` or ``"OUTCAR"``.
        converged: Whether the run converged electronically and ionically. ``None`` when the
            source does not say.
        energy: Final total energy in eV.
        energy_per_atom: Final total energy divided by the number of atoms, in eV/atom.
        natoms: Number of atoms.
        composition: Reduced formula of the final structure.
        band_gap: Band gap in eV, when eigenvalues were parsed.
        cbm: Conduction band minimum in eV.
        vbm: Valence band maximum in eV.
        is_gap_direct: Whether the gap is direct.
        efermi: Fermi level in eV.
        structure: The final structure.
        forces: Final forces in eV/A, as ``natoms`` rows of three.
        stress: Final stress tensor in kBar, as three rows of three.
    """

    directory: str
    source: str
    converged: bool | None = None
    energy: float | None = None
    energy_per_atom: float | None = None
    natoms: int | None = None
    composition: str | None = None
    band_gap: float | None = None
    cbm: float | None = None
    vbm: float | None = None
    is_gap_direct: bool | None = None
    efermi: float | None = None
    structure: Structure | None = None
    forces: list[list[float]] | None = None
    stress: list[list[float]] | None = None

    def summary(self) -> dict[str, Any]:
        """Return the scalar fields of this result, without the structure or the arrays.

        Returns:
            A flat dictionary suitable for a :class:`pandas.DataFrame` row.
        """
        return {
            "id": Path(self.directory).name,
            "directory": self.directory,
            "source": self.source,
            "composition": self.composition,
            "natoms": self.natoms,
            "energy": self.energy,
            "energy_per_atom": self.energy_per_atom,
            "converged": self.converged,
            "band_gap": self.band_gap,
            "efermi": self.efermi,
        }


def _band_properties(vasprun: Vasprun) -> dict[str, Any]:
    """Return the band-edge properties of a run, or empty values when they cannot be read.

    Args:
        vasprun: A parsed ``vasprun.xml``.

    Returns:
        A dictionary with ``band_gap``, ``cbm``, ``vbm`` and ``is_gap_direct``.
    """
    empty: dict[str, Any] = {"band_gap": None, "cbm": None, "vbm": None, "is_gap_direct": None}
    try:
        gap, cbm, vbm, is_direct = vasprun.eigenvalue_band_properties
    except Exception as exc:  # noqa: BLE001 - eigenvalues are optional output
        logger.debug("No band properties available: %s", exc)
        return empty
    return {
        "band_gap": float(gap),
        "cbm": float(cbm),
        "vbm": float(vbm),
        "is_gap_direct": bool(is_direct),
    }


def parse_vasprun(
    path: str | Path,
    *,
    parse_dos: bool = False,
    parse_eigen: bool = True,
    parse_potcar_file: bool = False,
) -> VaspResult:
    """Parse one ``vasprun.xml``.

    Args:
        path: Path of the ``vasprun.xml`` file.
        parse_dos: Whether to parse the density of states. Off by default; it is the most
            expensive part of the parse and nothing in this package uses it.
        parse_eigen: Whether to parse eigenvalues, which the band gap is taken from.
        parse_potcar_file: Whether to look for a POTCAR beside the XML to validate the
            pseudopotentials. Off by default so parsing works without VASP pseudopotentials.

    Returns:
        The parsed result.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    xml_path = Path(path).expanduser().resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(f"vasprun.xml not found: {xml_path}")

    vasprun = Vasprun(
        str(xml_path),
        parse_dos=parse_dos,
        parse_eigen=parse_eigen,
        parse_potcar_file=parse_potcar_file,
    )

    structure = getattr(vasprun, "final_structure", None)
    natoms = len(structure) if structure is not None else None
    energy = float(vasprun.final_energy) if vasprun.final_energy is not None else None

    last_step = vasprun.ionic_steps[-1] if getattr(vasprun, "ionic_steps", None) else {}
    forces = last_step.get("forces")
    stress = last_step.get("stress")
    band = _band_properties(vasprun) if parse_eigen else {}

    return VaspResult(
        directory=str(xml_path.parent),
        source="vasprun.xml",
        converged=bool(vasprun.converged),
        energy=energy,
        energy_per_atom=energy / natoms if energy is not None and natoms else None,
        natoms=natoms,
        composition=structure.composition.reduced_formula if structure is not None else None,
        efermi=float(vasprun.efermi) if getattr(vasprun, "efermi", None) is not None else None,
        structure=structure,
        forces=[[float(x) for x in row] for row in forces] if forces is not None else None,
        stress=[[float(x) for x in row] for row in stress] if stress is not None else None,
        **band,
    )


def parse_outcar(path: str | Path) -> dict[str, Any]:
    """Parse the scalar results of one ``OUTCAR``.

    Args:
        path: Path of the ``OUTCAR`` file.

    Returns:
        A dictionary with ``energy``, ``efermi`` and ``run_stats``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    outcar_path = Path(path).expanduser().resolve()
    if not outcar_path.is_file():
        raise FileNotFoundError(f"OUTCAR not found: {outcar_path}")

    outcar = Outcar(str(outcar_path))
    energy = outcar.final_energy
    return {
        "energy": float(energy) if energy is not None else None,
        "efermi": float(outcar.efermi) if outcar.efermi is not None else None,
        "run_stats": dict(outcar.run_stats) if outcar.run_stats else {},
    }


def read_calculation(directory: str | Path) -> VaspResult:
    """Read one calculation directory.

    ``vasprun.xml`` is tried first; if it is missing or unreadable, ``OUTCAR`` is parsed and the
    final structure is taken from ``CONTCAR`` (falling back to ``POSCAR``).

    Args:
        directory: The calculation directory.

    Returns:
        The parsed result. Fields the available files do not provide stay ``None``.

    Raises:
        FileNotFoundError: If the directory holds neither ``vasprun.xml`` nor ``OUTCAR``.
    """
    calc_dir = Path(directory).expanduser().resolve()
    xml_path = calc_dir / "vasprun.xml"
    outcar_path = calc_dir / "OUTCAR"

    if xml_path.is_file():
        try:
            return parse_vasprun(xml_path)
        except Exception as exc:  # noqa: BLE001 - a truncated XML is normal for a killed job
            logger.warning("Could not parse %s (%s); falling back to OUTCAR.", xml_path, exc)

    if not outcar_path.is_file():
        raise FileNotFoundError(f"Neither vasprun.xml nor OUTCAR found in {calc_dir}")

    parsed = parse_outcar(outcar_path)

    structure: Structure | None = None
    for name in ("CONTCAR", "POSCAR"):
        candidate = calc_dir / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            try:
                structure = Structure.from_file(str(candidate))
                break
            except Exception as exc:  # noqa: BLE001 - a half-written CONTCAR is normal
                logger.warning("Could not read %s (%s)", candidate, exc)

    natoms = len(structure) if structure is not None else None
    energy = parsed["energy"]
    return VaspResult(
        directory=str(calc_dir),
        source="OUTCAR",
        converged=None,
        energy=energy,
        energy_per_atom=energy / natoms if energy is not None and natoms else None,
        natoms=natoms,
        composition=structure.composition.reduced_formula if structure is not None else None,
        efermi=parsed["efermi"],
        structure=structure,
    )


def _calculation_directories(root: Path) -> list[Path]:
    """Return every directory below a root that looks like a VASP calculation directory.

    Args:
        root: Directory to search.

    Returns:
        Sorted directories holding an INCAR, a vasprun.xml or an OUTCAR.
    """
    found: set[Path] = set()
    for name in ("INCAR", "vasprun.xml", "OUTCAR"):
        found.update(path.parent for path in root.rglob(name) if path.is_file())
    return sorted(found)


def collect_results(root: str | Path, *, only_converged: bool = False) -> pd.DataFrame:
    """Collect the results of every calculation below a root directory.

    Args:
        root: Directory holding the calculation directories, at any depth.
        only_converged: Keep only rows whose run converged. Rows read from an ``OUTCAR``, which
            does not report convergence, are dropped by this filter as well.

    Returns:
        A table with the columns ``id``, ``directory``, ``source``, ``composition``, ``natoms``,
        ``energy``, ``energy_per_atom``, ``converged``, ``band_gap`` and ``efermi``. ``id``,
        ``composition`` and ``energy_per_atom`` are the columns the convex-hull stage reads. The
        table is empty when nothing could be parsed.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {root_path}")

    rows: list[dict[str, Any]] = []
    for calc_dir in _calculation_directories(root_path):
        try:
            result = read_calculation(calc_dir)
        except FileNotFoundError:
            logger.debug("No output in %s yet", calc_dir)
            continue
        except Exception as exc:  # noqa: BLE001 - one bad directory must not stop the sweep
            logger.error("Could not read %s: %s", calc_dir, exc)
            continue
        row = result.summary()
        row["id"] = str(calc_dir.relative_to(root_path)) if calc_dir != root_path else calc_dir.name
        rows.append(row)

    columns = [
        "id",
        "directory",
        "source",
        "composition",
        "natoms",
        "energy",
        "energy_per_atom",
        "converged",
        "band_gap",
        "efermi",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if only_converged and not frame.empty:
        frame = frame[frame["converged"].fillna(False).astype(bool)].reset_index(drop=True)

    logger.info("Collected %d calculation result(s) from %s", len(frame), root_path)
    return frame


def _read_tail(path: Path, nbytes: int = _OUTCAR_TAIL_BYTES) -> str:
    """Return the last bytes of a text file, decoded leniently.

    Args:
        path: The file to read.
        nbytes: How many trailing bytes to read.

    Returns:
        The decoded tail, or an empty string when the file cannot be read.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > nbytes:
                handle.seek(size - nbytes)
            return handle.read().decode("utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return ""


def check_calculation_status(root: str | Path) -> dict[str, str]:
    """Report the state of every calculation directory below a root.

    A directory counts as a calculation directory when it holds an INCAR, a ``vasprun.xml`` or
    an ``OUTCAR``. The state is read from the tail of the OUTCAR: ``completed`` when VASP
    reached its normal end, ``failed`` when the tail carries an error marker, ``running`` when
    the OUTCAR exists but says neither, and ``pending`` when there is no OUTCAR at all.

    Args:
        root: Directory holding the calculation directories, at any depth.

    Returns:
        A mapping from directory path (relative to ``root``) to one of
        :data:`CALCULATION_STATES`.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {root_path}")

    status: dict[str, str] = {}
    for calc_dir in _calculation_directories(root_path):
        key = str(calc_dir.relative_to(root_path)) if calc_dir != root_path else calc_dir.name
        outcar = calc_dir / "OUTCAR"
        if not outcar.is_file():
            status[key] = "pending"
            continue
        tail = _read_tail(outcar)
        if not tail:
            status[key] = "unknown"
        elif _COMPLETION_MARKER in tail:
            status[key] = "completed"
        elif any(marker in tail for marker in _FAILURE_MARKERS):
            status[key] = "failed"
        else:
            status[key] = "running"

    return status


def export_for_finetuning(
    root: str | Path,
    export_file: str | Path = "dft_results.json",
    *,
    only_converged: bool = True,
) -> Path:
    """Export energies, forces, stresses and structures as JSON for the fine-tuning stage.

    Args:
        root: Directory holding the calculation directories, at any depth.
        export_file: Path of the JSON file to write.
        only_converged: Skip runs that did not converge, and runs whose convergence is unknown
            because only an OUTCAR was available.

    Returns:
        The path of the file that was written.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {root_path}")

    payload: dict[str, Any] = {}
    for calc_dir in _calculation_directories(root_path):
        try:
            result = read_calculation(calc_dir)
        except FileNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001 - one bad directory must not stop the export
            logger.error("Could not read %s: %s", calc_dir, exc)
            continue

        if only_converged and not result.converged:
            logger.debug("Skipping %s: not a converged run", calc_dir)
            continue

        key = str(calc_dir.relative_to(root_path)) if calc_dir != root_path else calc_dir.name
        payload[key] = {
            "energy": result.energy,
            "energy_per_atom": result.energy_per_atom,
            "natoms": result.natoms,
            "composition": result.composition,
            "forces": result.forces,
            "stress": result.stress,
            "structure": result.structure.as_dict() if result.structure is not None else None,
        }

    out_path = Path(export_file).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Exported %d calculation(s) to %s", len(payload), out_path)
    return out_path
