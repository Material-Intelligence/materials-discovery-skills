"""Structure relaxation with a machine-learning interatomic potential.

The relaxation runs through ASE: the structure is converted to :class:`ase.Atoms`, the
calculator is attached with ``atoms.calc = calculator``, and cell relaxation is done with
:class:`ase.filters.FrechetCellFilter`. The calculator is supplied by the caller and reused
across structures, so a batch loads the model once
(:func:`matdisc.common.calculators.load_calculator` builds it).

Energies are reported both as a total and per atom. ``max_force`` is the largest force norm
on any atom, which is the quantity ASE compares against ``fmax``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pandas as pd
from monty.json import MSONable
from pymatgen.core import Structure

from matdisc.common.io import from_ase, to_ase, write_structure
from matdisc.common.logging import get_logger

if TYPE_CHECKING:  # imported for type checking only; ASE is imported where it is used
    from ase.calculators.calculator import Calculator
    from ase.optimize.optimize import Optimizer

LOGGER = get_logger(__name__)

OPTIMIZERS = ("BFGS", "FIRE")
"""Optimizer names accepted by :func:`relax`."""

DEFAULT_FMAX = 0.01
"""Default force convergence criterion, in eV/A."""

DEFAULT_MAX_STEPS = 500
"""Default cap on optimizer steps."""

SUMMARY_COLUMNS = [
    "id",
    "composition",
    "natoms",
    "energy",
    "energy_per_atom",
    "max_force",
    "n_steps",
    "converged",
    "error",
]
"""Columns of the table :func:`relax_many` returns."""

__all__ = [
    "DEFAULT_FMAX",
    "DEFAULT_MAX_STEPS",
    "OPTIMIZERS",
    "SUMMARY_COLUMNS",
    "RelaxationResult",
    "relax",
    "relax_many",
]


@dataclass
class RelaxationResult(MSONable):
    """Outcome of one relaxation.

    Attributes:
        converged: Whether the force criterion was met within ``max_steps``.
        natoms: Number of atoms in the relaxed cell.
        energy: Final potential energy, in eV.
        energy_per_atom: Final potential energy divided by ``natoms``, in eV/atom.
        max_force: Largest force norm on any atom after relaxation, in eV/A.
        n_steps: Number of optimizer steps taken.
        optimizer: Optimizer used, one of :data:`OPTIMIZERS`.
        fmax: Force criterion the run was asked to reach, in eV/A.
        relax_cell: Whether the cell was relaxed along with the positions.
        output_dir: Directory the run wrote to, or ``None`` when nothing was written.
    """

    converged: bool
    natoms: int
    energy: float
    energy_per_atom: float
    max_force: float
    n_steps: int
    optimizer: str
    fmax: float
    relax_cell: bool
    output_dir: str | None = field(default=None)


def _optimizer_class(name: str) -> type[Optimizer]:
    """Return the ASE optimizer class for a name.

    Args:
        name: Optimizer name, case-insensitive; see :data:`OPTIMIZERS`.

    Returns:
        The ASE optimizer class.

    Raises:
        ValueError: If the name is not a supported optimizer.
    """
    from ase.optimize import BFGS, FIRE

    classes = {"BFGS": BFGS, "FIRE": FIRE}
    try:
        return classes[name.upper()]
    except KeyError:
        raise ValueError(f"Unknown optimizer {name!r}; choose one of {', '.join(OPTIMIZERS)}") from None


def relax(
    structure: Structure,
    calculator: Calculator,
    optimizer: str = "BFGS",
    fmax: float = DEFAULT_FMAX,
    max_steps: int = DEFAULT_MAX_STEPS,
    relax_cell: bool = True,
    output_dir: str | Path | None = None,
) -> tuple[Structure, dict]:
    """Relax one structure with an ASE calculator.

    Args:
        structure: Structure to relax.
        calculator: An ASE calculator, for example the one returned by
            :func:`matdisc.common.calculators.load_calculator`.
        optimizer: ``"BFGS"`` or ``"FIRE"``.
        fmax: Force convergence criterion, in eV/A.
        max_steps: Maximum number of optimizer steps.
        relax_cell: Relax the cell as well as the positions, through
            :class:`ase.filters.FrechetCellFilter`.
        output_dir: Directory for ``relaxed.vasp`` and the optimizer log. Resolved to an
            absolute path and created if missing. Nothing is written when this is ``None``.

    Returns:
        A tuple of the relaxed structure and a dictionary describing the run, which is
        :meth:`RelaxationResult.as_dict` output and carries ``energy``,
        ``energy_per_atom``, ``max_force``, ``n_steps`` and ``converged``.

    Raises:
        ValueError: If ``optimizer`` is not a supported optimizer.
    """
    from ase.filters import FrechetCellFilter

    optimizer_class = _optimizer_class(optimizer)

    directory: Path | None = None
    logfile: str | None = None
    if output_dir is not None:
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        logfile = str(directory / "relaxation.log")

    atoms = to_ase(structure)
    atoms.calc = calculator
    target = FrechetCellFilter(atoms) if relax_cell else atoms

    run = optimizer_class(target, logfile=logfile)
    converged = bool(run.run(fmax=fmax, steps=max_steps))

    forces = np.asarray(atoms.get_forces())
    max_force = float(np.linalg.norm(forces, axis=1).max()) if forces.size else 0.0
    energy = float(atoms.get_potential_energy())
    natoms = len(atoms)
    relaxed = from_ase(atoms)

    if directory is not None:
        write_structure(relaxed, directory / "relaxed.vasp", fmt="poscar")

    result = RelaxationResult(
        converged=converged,
        natoms=natoms,
        energy=energy,
        energy_per_atom=energy / natoms,
        max_force=max_force,
        n_steps=int(run.nsteps),
        optimizer=optimizer.upper(),
        fmax=fmax,
        relax_cell=relax_cell,
        output_dir=str(directory) if directory is not None else None,
    )
    LOGGER.info(
        "Relaxed %s in %d step(s): E = %.4f eV/atom, max force %.4f eV/A, converged=%s",
        relaxed.composition.reduced_formula,
        result.n_steps,
        result.energy_per_atom,
        result.max_force,
        result.converged,
    )
    return relaxed, result.as_dict()


def relax_many(
    structures: Mapping[str, Structure],
    calculator: Calculator,
    output_dir: str | Path | None = None,
    **kwargs: Any,
) -> tuple[dict[str, Structure], pd.DataFrame]:
    """Relax a set of structures and collect their energies.

    A structure whose relaxation raises is recorded with an empty energy and the error
    message rather than aborting the batch.

    Args:
        structures: Mapping of identifier to structure. The identifier names the
            per-structure subdirectory of ``output_dir`` and the row in the returned table.
        calculator: An ASE calculator, reused for every structure.
        output_dir: Parent directory for per-structure output, resolved to an absolute path.
            Nothing is written when this is ``None``.
        **kwargs: Passed through to :func:`relax`.

    Returns:
        A tuple of the relaxed structures, keyed as the input was, and a table with the
        columns ``id``, ``composition``, ``natoms``, ``energy``, ``energy_per_atom``,
        ``max_force``, ``n_steps``, ``converged`` and ``error``. Those column names are the
        candidate schema :func:`matdisc.screening.hull.compute_e_above_hull` reads.
    """
    parent = Path(output_dir).expanduser().resolve() if output_dir is not None else None

    relaxed: dict[str, Structure] = {}
    rows: list[dict[str, Any]] = []
    for identifier, structure in structures.items():
        try:
            structure_out, info = relax(
                structure,
                calculator,
                output_dir=(parent / identifier) if parent is not None else None,
                **kwargs,
            )
        except Exception as error:  # one bad structure must not abort the batch
            LOGGER.warning("Relaxation of %s failed: %s", identifier, error)
            rows.append(
                {
                    "id": identifier,
                    "composition": structure.composition.reduced_formula,
                    "natoms": len(structure),
                    "energy": float("nan"),
                    "energy_per_atom": float("nan"),
                    "max_force": float("nan"),
                    "n_steps": 0,
                    "converged": False,
                    "error": str(error),
                }
            )
            continue

        relaxed[identifier] = structure_out
        rows.append(
            {
                "id": identifier,
                "composition": structure_out.composition.reduced_formula,
                "natoms": info["natoms"],
                "energy": info["energy"],
                "energy_per_atom": info["energy_per_atom"],
                "max_force": info["max_force"],
                "n_steps": info["n_steps"],
                "converged": info["converged"],
                "error": "",
            }
        )

    table = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    LOGGER.info("Relaxed %d of %d structure(s)", len(relaxed), len(structures))
    return relaxed, table
