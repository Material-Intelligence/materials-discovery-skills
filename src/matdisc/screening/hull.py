"""Convex-hull analysis via :mod:`pymatgen.analysis.phase_diagram`.

The hull is built once from a table of competing phases and every candidate is then placed
against it with :meth:`~pymatgen.analysis.phase_diagram.PhaseDiagram.get_decomp_and_e_above_hull`.

Energy convention
-----------------
Every energy read from either table is a **per-atom** energy and is multiplied by the atom
count of the reduced formula to obtain the total energy a
:class:`~pymatgen.analysis.phase_diagram.PDEntry` expects. Nothing is divided by an atom
count a second time.

Candidates and competing phases must be on **one** energy scale. Two combinations make
sense:

* ``energy_column="energy_per_atom"`` -- raw per-atom total energies, when candidates and
  competing phases come from the same calculator and settings.
* ``energy_column="formation_energy_per_atom"`` -- formation energies per atom, which is how
  candidate energies from a machine-learning potential are combined with Materials Project
  competing phases. :func:`formation_energy_per_atom` converts a per-atom energy into a
  formation energy given elemental reference energies.

Stability criterion
-------------------
A candidate counts as stable when ``e_above_hull <= tolerance``. The default tolerance is
1e-6 eV/atom, which admits a phase sitting exactly on the hull while absorbing the rounding
of the hull solve. A strict ``< 0`` test would report an on-hull phase as unstable.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import pandas as pd
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition

from matdisc.common.composition import parse
from matdisc.common.logging import get_logger

LOGGER = get_logger(__name__)

DEFAULT_TOLERANCE = 1e-6
"""Default stability tolerance in eV/atom; a candidate is stable when ``e_above_hull <= tolerance``."""

DEFAULT_ENERGY_COLUMN = "energy_per_atom"
"""Column read from both tables, in eV/atom."""

DEFAULT_COMPOSITION_COLUMN = "composition"
"""Column holding a formula string parsed by :class:`pymatgen.core.Composition`."""

DEFAULT_ID_COLUMN = "id"
"""Column identifying a candidate in the returned table."""

CONVERGENCE_COLUMN = "converged"
"""Column :func:`matdisc.screening.relax.relax_many` writes; absent from a harvested table."""

__all__ = [
    "CONVERGENCE_COLUMN",
    "DEFAULT_COMPOSITION_COLUMN",
    "DEFAULT_ENERGY_COLUMN",
    "DEFAULT_ID_COLUMN",
    "DEFAULT_TOLERANCE",
    "build_phase_diagram",
    "compute_e_above_hull",
    "formation_energy_per_atom",
    "load_element_references",
    "split_usable_rows",
    "toy_system",
]


def _require_columns(frame: pd.DataFrame, columns: list[str], what: str) -> None:
    """Raise if a table is missing a required column.

    Args:
        frame: Table to check.
        columns: Column names that must be present.
        what: Name of the table, used in the error message.

    Raises:
        KeyError: If any column is missing.
    """
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{what} table is missing column(s) {missing}; it has {list(frame.columns)}")


def split_usable_rows(
    table: pd.DataFrame,
    energy_column: str = DEFAULT_ENERGY_COLUMN,
    require_converged: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a table of phases into the rows a hull can be built from and the rest.

    A row is unusable when its energy is missing -- which is what
    :func:`matdisc.screening.relax.relax_many` records for a relaxation that raised -- or,
    with ``require_converged``, when it carries :data:`CONVERGENCE_COLUMN` set to false.

    Losing a competing phase is not neutral: a hull built from fewer phases can only sit
    lower or equal, so every candidate's energy above the hull moves toward zero and more
    candidates read as stable. The caller therefore has to see what was dropped rather than
    read a count of what survived.

    Args:
        table: A candidate or competing-phase table.
        energy_column: Per-atom energy column, in eV/atom.
        require_converged: Also drop rows whose relaxation did not reach its force criterion.

    Returns:
        ``(usable, dropped)``, two frames whose row counts sum to ``len(table)``.

    Raises:
        KeyError: If ``energy_column`` is missing.
    """
    _require_columns(table, [energy_column], "Phase")
    energies = pd.to_numeric(table[energy_column], errors="coerce")
    usable = energies.notna()
    if require_converged and CONVERGENCE_COLUMN in table.columns:
        usable &= table[CONVERGENCE_COLUMN].fillna(False).astype(bool)
    return table[usable], table[~usable]


def build_phase_diagram(
    competing: pd.DataFrame,
    energy_column: str = DEFAULT_ENERGY_COLUMN,
    composition_column: str = DEFAULT_COMPOSITION_COLUMN,
    id_column: str = "material_id",
) -> PhaseDiagram:
    """Build a phase diagram from a table of competing phases.

    One :class:`~pymatgen.analysis.phase_diagram.PDEntry` is created per row. Its total
    energy is ``row[energy_column] * n``, where ``n`` is the number of atoms in the reduced
    formula parsed from ``row[composition_column]``. Rows whose energy is missing are skipped
    with a warning; a caller that has to account for them -- every caller reporting a
    stability verdict -- filters the table with :func:`split_usable_rows` first and decides
    what to do with the rows that were dropped.

    Args:
        competing: Table of competing phases. Must carry ``composition_column`` and
            ``energy_column``; ``id_column`` is used only for entry names when present.
        energy_column: Per-atom energy column, in eV/atom.
        composition_column: Formula column, parsed by :class:`pymatgen.core.Composition`.
        id_column: Optional identifier column used to name entries.

    Returns:
        The phase diagram spanned by those entries.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If no usable entry remains, or if the entries do not span a phase
            diagram -- most often because an elemental reference is absent.
    """
    _require_columns(competing, [composition_column, energy_column], "Competing-phases")

    entries: list[PDEntry] = []
    skipped = 0
    for position, row in competing.iterrows():
        energy_per_atom = row[energy_column]
        if energy_per_atom is None or (isinstance(energy_per_atom, float) and math.isnan(energy_per_atom)):
            LOGGER.warning("Skipping competing phase at row %s: %s is empty", position, energy_column)
            skipped += 1
            continue
        composition = parse(str(row[composition_column]))
        name = str(row[id_column]) if id_column in competing.columns else composition.reduced_formula
        entries.append(PDEntry(composition, float(energy_per_atom) * composition.num_atoms, name=name))

    if skipped:
        LOGGER.warning(
            "%d of %d competing phases carried no %s and are not on this hull, which can only lower it",
            skipped,
            len(competing),
            energy_column,
        )

    if not entries:
        raise ValueError("No competing phase carried a usable energy; the phase diagram would be empty")

    try:
        diagram = PhaseDiagram(entries)
    except Exception as error:  # pymatgen raises PhaseDiagramError and ValueError here
        raise ValueError(
            f"Could not build a phase diagram from {len(entries)} competing phases: {error}. "
            "Every element appearing in the table needs an elemental reference phase of its own."
        ) from error

    LOGGER.info(
        "Phase diagram built from %d competing phases over %s (%d on the hull)",
        len(entries),
        "-".join(sorted(element.symbol for element in diagram.elements)),
        len(diagram.stable_entries),
    )
    return diagram


def _format_decomposition(decomposition: Mapping[object, float]) -> str:
    """Render a decomposition as ``amount*formula`` terms, largest amount first.

    Args:
        decomposition: Mapping of entry to fractional amount, as returned by pymatgen.

    Returns:
        A ``"; "``-joined string, empty when the decomposition is empty.
    """
    terms = sorted(decomposition.items(), key=lambda item: item[1], reverse=True)
    return "; ".join(f"{amount:.4f}*{entry.composition.reduced_formula}" for entry, amount in terms if amount > 1e-8)


def compute_e_above_hull(
    candidates: pd.DataFrame,
    competing: pd.DataFrame,
    tolerance: float = DEFAULT_TOLERANCE,
    energy_column: str = DEFAULT_ENERGY_COLUMN,
    composition_column: str = DEFAULT_COMPOSITION_COLUMN,
    id_column: str = DEFAULT_ID_COLUMN,
) -> pd.DataFrame:
    """Place candidates on the convex hull of a set of competing phases.

    Both tables are read on the same per-atom energy scale; see the module docstring. The
    hull is built once and reused for every candidate.

    Args:
        candidates: Table with at least ``id_column``, ``composition_column`` and
            ``energy_column``.
        competing: Table of competing phases, as returned by
            :func:`matdisc.competing.search.find_competing_phases`.
        tolerance: Stability tolerance in eV/atom. A candidate is stable when its energy
            above the hull is at or below this value.
        energy_column: Per-atom energy column read from both tables, in eV/atom.
        composition_column: Formula column read from both tables.
        id_column: Identifier column of the candidates table.

    Returns:
        A copy of ``candidates`` with four added columns: ``e_above_hull`` (eV/atom, NaN
        when the candidate could not be placed), ``hull_energy_per_atom`` (eV/atom on the
        same scale as the input energies), ``is_stable`` (bool) and ``decomposition`` (the
        hull decomposition as a string, or the reason the candidate was skipped).

    Raises:
        KeyError: If a required column is missing from either table.
        ValueError: If the phase diagram cannot be built.
    """
    _require_columns(candidates, [id_column, composition_column, energy_column], "Candidates")
    diagram = build_phase_diagram(competing, energy_column=energy_column, composition_column=composition_column)
    diagram_elements = set(diagram.elements)

    e_above_hull: list[float] = []
    hull_energy: list[float] = []
    is_stable: list[bool] = []
    decompositions: list[str] = []

    for _, row in candidates.iterrows():
        identifier = str(row[id_column])
        composition = parse(str(row[composition_column]))
        energy_per_atom = row[energy_column]

        if energy_per_atom is None or (isinstance(energy_per_atom, float) and math.isnan(energy_per_atom)):
            LOGGER.warning("Candidate %s has no %s and cannot be placed on the hull", identifier, energy_column)
            e_above_hull.append(float("nan"))
            hull_energy.append(float("nan"))
            is_stable.append(False)
            decompositions.append(f"no {energy_column}")
            continue

        if not set(composition.elements) <= diagram_elements:
            outside = sorted(element.symbol for element in set(composition.elements) - diagram_elements)
            LOGGER.warning(
                "Candidate %s (%s) contains %s, which the competing phases do not cover",
                identifier,
                composition.reduced_formula,
                ", ".join(outside),
            )
            e_above_hull.append(float("nan"))
            hull_energy.append(float("nan"))
            is_stable.append(False)
            decompositions.append(f"not covered by the competing phases: {', '.join(outside)}")
            continue

        entry = PDEntry(composition, float(energy_per_atom) * composition.num_atoms, name=identifier)
        try:
            decomposition, above_hull = diagram.get_decomp_and_e_above_hull(entry, allow_negative=True)
        except Exception as error:  # a failed hull solve must not abort the batch
            LOGGER.warning("Could not place candidate %s (%s) on the hull: %s", identifier, composition, error)
            e_above_hull.append(float("nan"))
            hull_energy.append(float("nan"))
            is_stable.append(False)
            decompositions.append(f"error: {error}")
            continue

        e_above_hull.append(float(above_hull))
        hull_energy.append(float(energy_per_atom) - float(above_hull))
        is_stable.append(bool(above_hull <= tolerance))
        decompositions.append(_format_decomposition(decomposition or {}))

    result = candidates.copy()
    result["e_above_hull"] = e_above_hull
    result["hull_energy_per_atom"] = hull_energy
    result["is_stable"] = is_stable
    result["decomposition"] = decompositions

    LOGGER.info(
        "%d of %d candidates are at or below the hull (tolerance %g eV/atom)",
        int(result["is_stable"].sum()),
        len(result),
        tolerance,
    )
    return result


def formation_energy_per_atom(
    energy_per_atom: float,
    composition: str | Composition,
    element_references: Mapping[str, float],
) -> float:
    """Convert a per-atom energy into a formation energy per atom.

    ``E_f = E - sum_i x_i * mu_i``, where ``x_i`` is the atomic fraction of element ``i``
    and ``mu_i`` its reference energy per atom. Reference energies must come from the same
    calculator and settings as ``energy_per_atom``.

    Args:
        energy_per_atom: Energy of the compound, in eV/atom.
        composition: Formula string or :class:`pymatgen.core.Composition`.
        element_references: Reference energy per atom of each element, in eV/atom.

    Returns:
        The formation energy in eV/atom.

    Raises:
        KeyError: If an element of the composition has no reference energy.
    """
    comp = composition if isinstance(composition, Composition) else parse(str(composition))
    reference = 0.0
    for element, amount in comp.get_el_amt_dict().items():
        if element not in element_references:
            raise KeyError(f"No reference energy for element {element} (composition {comp.reduced_formula})")
        reference += (amount / comp.num_atoms) * float(element_references[element])
    return float(energy_per_atom) - reference


def load_element_references(
    path: str | Path,
    composition_column: str = DEFAULT_COMPOSITION_COLUMN,
    energy_column: str = DEFAULT_ENERGY_COLUMN,
) -> dict[str, float]:
    """Read elemental reference energies from a CSV file.

    Rows whose composition is not a single element are ignored. When an element appears
    more than once the lowest energy wins, which is the usual choice of reference phase.

    Args:
        path: CSV file carrying ``composition_column`` and ``energy_column``.
        composition_column: Formula column.
        energy_column: Per-atom energy column, in eV/atom.

    Returns:
        Mapping of element symbol to reference energy per atom.

    Raises:
        KeyError: If a required column is missing.
    """
    frame = pd.read_csv(Path(path).expanduser().resolve())
    _require_columns(frame, [composition_column, energy_column], "Element-reference")

    references: dict[str, float] = {}
    for _, row in frame.iterrows():
        comp = parse(str(row[composition_column]))
        if len(comp.elements) != 1:
            continue
        symbol = comp.elements[0].symbol
        energy = float(row[energy_column])
        if symbol not in references or energy < references[symbol]:
            references[symbol] = energy
    LOGGER.info("Loaded reference energies for %d element(s) from %s", len(references), path)
    return references


def toy_system() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a small offline Ba-Cd-P system for examples and tests.

    The energies are round synthetic numbers chosen to exercise the three outcomes of the
    stability test, not measured or calculated values. The elemental references are set to
    zero, so ``energy_per_atom`` and ``formation_energy_per_atom`` coincide and either
    column can be used as ``energy_column``.

    The three candidates are:

    * ``toy-below-hull`` -- BaCdP, below the hull, stable;
    * ``toy-above-hull`` -- BaCdP at a higher energy, above the hull, unstable;
    * ``toy-on-hull`` -- Ba2Cd2P3, one formula unit of BaP plus one of BaCd2P2, so it sits
      exactly on the tie-line between two hull phases and its energy above the hull is zero
      to within the rounding of the hull solve (of order 1e-16 eV/atom). The
      ``<= tolerance`` criterion calls it stable whichever side of zero that rounding falls
      on; a strict ``< 0`` test would turn on it.

    No candidate composition appears in the competing-phase table, so no candidate is
    compared against itself.

    Returns:
        ``(candidates, competing)``, both ready for :func:`compute_e_above_hull`.
    """
    competing = pd.DataFrame(
        [
            ("toy-Ba", "Ba", 1, 0.00),
            ("toy-Cd", "Cd", 1, 0.00),
            ("toy-P", "P", 1, 0.00),
            ("toy-BaP", "BaP", 2, -0.60),
            ("toy-Cd3P2", "Cd3P2", 5, -0.30),
            ("toy-BaCd2P2", "BaCd2P2", 5, -0.50),
        ],
        columns=["material_id", "composition", "natoms", "energy_per_atom"],
    )
    competing["formula_pretty"] = competing["composition"]
    competing["formation_energy_per_atom"] = competing["energy_per_atom"]
    competing["is_stable_mp"] = True
    competing["icsd_ids"] = ""

    candidates = pd.DataFrame(
        [
            ("toy-below-hull", "BaCdP", 3, -0.50),
            ("toy-above-hull", "BaCdP", 3, -0.30),
            ("toy-on-hull", "Ba2Cd2P3", 7, (2 * -0.60 + 5 * -0.50) / 7),
        ],
        columns=["id", "composition", "natoms", "energy_per_atom"],
    )
    return candidates, competing
