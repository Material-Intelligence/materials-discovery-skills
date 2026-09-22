"""Competing-phase search against the Materials Project.

A convex hull for a chemical system needs the energies of every phase the candidate competes
with, including the phases of each subsystem: a Ba-Cd-P hull rests on Ba, Cd and P, on Ba-Cd,
Ba-P and Cd-P, and on the Ba-Cd-P phases themselves. Materials Project matches ``chemsys``
exactly, so :func:`find_competing_phases` enumerates the subsystems and queries each one.

Restricting the result to phases carrying an ICSD cross-reference (the default) keeps the
reference set to compounds that have been made and characterised experimentally.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Sequence

import pandas as pd
from pymatgen.core import Element

from matdisc.common import composition
from matdisc.common.logging import get_logger
from matdisc.common.mp import icsd_ids_from_doc, mprester, search_summary

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from mp_api.client import MPRester

__all__ = ["find_competing_phases", "COMPETING_PHASE_COLUMNS", "SUMMARY_FIELDS"]

logger = get_logger(__name__)

COMPETING_PHASE_COLUMNS = [
    "material_id",
    "formula_pretty",
    "composition",
    "natoms",
    "formation_energy_per_atom",
    "energy_per_atom",
    "is_stable_mp",
    "icsd_ids",
]
"""Columns of the table :func:`find_competing_phases` returns.

``composition`` is the canonical reduced formula and ``natoms`` the number of atoms in one
reduced formula unit, both from pymatgen. Both energies are per atom, as their names say and
as Materials Project reports them; nothing downstream needs to divide them again.
"""

SUMMARY_FIELDS = [
    "material_id",
    "formula_pretty",
    "formation_energy_per_atom",
    "energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "database_IDs",
]
"""Summary-document fields fetched per material -- only the ones used to build the table."""


def find_competing_phases(
    chemsys: str,
    include_unary: bool = True,
    include_binary: bool = True,
    include_higher_order: bool = True,
    only_icsd: bool = True,
    only_stable: bool = False,
    energy_above_hull_max: float | None = None,
    unique_formula: bool = True,
    allow_partial: bool = False,
    client: MPRester | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Collect the competing phases of a chemical system from the Materials Project.

    Args:
        chemsys: The chemical system, for example ``"Ba-Cd-P"``. A formula such as
            ``"BaCdP"`` is accepted too and is reduced to its elements.
        include_unary: Query the one-element subsystems (the elemental references a hull
            needs).
        include_binary: Query the two-element subsystems.
        include_higher_order: Query the subsystems of three or more elements, up to the
            full system.
        only_icsd: Keep only materials carrying an ICSD cross-reference.
        only_stable: Keep only materials Materials Project reports on its own hull.
        energy_above_hull_max: Drop materials further above the Materials Project hull than
            this, in eV/atom. ``None`` keeps them all.
        unique_formula: Keep one entry per reduced formula, the one with the lowest
            formation energy per atom. The polymorph is chosen on Materials Project energies,
            not on the calculator the candidates are relaxed with.
        allow_partial: Return the phases that were found even when a subsystem query failed.
            A subsystem that is missing takes its phases out of the reference set, and a hull
            built from fewer phases can only sit lower, so the default is to raise instead.
        client: An open :class:`mp_api.client.MPRester` to reuse. One is opened and closed
            here when omitted.
        api_key: Materials Project key. Read from ``MP_API_KEY`` when omitted.

    Returns:
        A table with the columns listed in :data:`COMPETING_PHASE_COLUMNS`, sorted by
        subsystem and then by formation energy. The table is empty, with those columns, when
        nothing matches. When ``allow_partial`` let a failed query through,
        ``frame.attrs["missing_subsystems"]`` names the subsystems that are not represented.

    Raises:
        ValueError: If ``chemsys`` names no element, or names something that is not one.
        RuntimeError: If no Materials Project API key is available, or a subsystem query
            failed and ``allow_partial`` is not set.
    """
    element_symbols = _elements_of(chemsys)
    subsystems = _subsystems(element_symbols, include_unary, include_binary, include_higher_order)
    if not subsystems:
        logger.warning("No subsystems selected for %s; nothing to query.", "-".join(element_symbols))
        return _empty_table()

    logger.info("Searching competing phases for %s (%d subsystems)", "-".join(element_symbols), len(subsystems))

    records: list[dict[str, object]] = []
    failures: dict[str, str] = {}
    with mprester(client, api_key) as mpr:
        for subsystem in subsystems:
            logger.info("Querying %s", subsystem)
            try:
                docs = search_summary(subsystem, fields=SUMMARY_FIELDS, client=mpr)
            except Exception as exc:  # noqa: BLE001 - every failure is reported together below
                logger.warning("Query for %s failed (%s).", subsystem, exc)
                failures[subsystem] = str(exc)
                continue

            kept = 0
            for doc in docs:
                record = _record_from_doc(
                    doc, only_icsd=only_icsd, only_stable=only_stable, energy_above_hull_max=energy_above_hull_max
                )
                if record is not None:
                    records.append(record)
                    kept += 1
            logger.info("  %s: %d of %d materials kept", subsystem, kept, len(docs))

    if failures and not allow_partial:
        detail = "; ".join(f"{subsystem}: {message}" for subsystem, message in sorted(failures.items()))
        raise RuntimeError(
            f"{len(failures)} of {len(subsystems)} subsystem queries failed, so the competing phases of "
            f"{'-'.join(element_symbols)} are incomplete and a hull built from them would sit too low. "
            f"Retry, or pass allow_partial=True to accept the gap knowingly. Failures: {detail}"
        )
    if failures:
        logger.warning(
            "Continuing without %d subsystem(s): %s. The hull is missing their phases.",
            len(failures),
            ", ".join(sorted(failures)),
        )

    if not records:
        logger.warning("No competing phases matched the filters.")
        empty = _empty_table()
        empty.attrs["missing_subsystems"] = sorted(failures)
        return empty

    frame = pd.DataFrame.from_records(records, columns=COMPETING_PHASE_COLUMNS + ["_chemsys"])
    frame = frame.sort_values(["_chemsys", "formation_energy_per_atom"], kind="stable")

    if unique_formula:
        before = len(frame)
        frame = _lowest_energy_per_formula(frame)
        frame = frame.sort_values(["_chemsys", "formation_energy_per_atom"], kind="stable")
        logger.info("One structure per reduced formula: %d -> %d", before, len(frame))

    frame = frame.drop(columns=["_chemsys"]).reset_index(drop=True)
    frame.attrs["missing_subsystems"] = sorted(failures)
    logger.info("Found %d competing phases", len(frame))
    return frame


def _elements_of(chemsys: str) -> list[str]:
    """Resolve a chemical system or formula to a sorted list of element symbols.

    Args:
        chemsys: ``"Ba-Cd-P"`` or ``"BaCdP"``.

    Returns:
        The element symbols in alphabetical order.

    Raises:
        ValueError: If the string names no element, or names something that is not one.
    """
    text = chemsys.strip()
    if not text:
        raise ValueError("A chemical system or formula is required, for example 'Ba-Cd-P'.")

    if "-" in text:
        symbols = [part.strip() for part in text.split("-") if part.strip()]
        for symbol in symbols:
            Element(symbol)  # raises ValueError on an unknown symbol
    else:
        symbols = composition.elements(text)

    if not symbols:
        raise ValueError(f"No elements found in {chemsys!r}.")
    return sorted(set(symbols))


def _subsystems(
    element_symbols: Sequence[str],
    include_unary: bool,
    include_binary: bool,
    include_higher_order: bool,
) -> list[str]:
    """Enumerate the subsystems to query.

    Args:
        element_symbols: Elements of the full system.
        include_unary: Include one-element subsystems.
        include_binary: Include two-element subsystems.
        include_higher_order: Include subsystems of three or more elements.

    Returns:
        Chemical-system strings such as ``["Ba", "Cd", "P", "Ba-Cd", ..., "Ba-Cd-P"]``,
        ordered by increasing number of elements.
    """
    n_elements = len(element_symbols)
    orders: set[int] = set()
    if include_unary:
        orders.add(1)
    if include_binary:
        orders.add(2)
    if include_higher_order:
        orders.update(range(3, n_elements + 1))

    systems: list[str] = []
    for order in sorted(order for order in orders if 1 <= order <= n_elements):
        for combo in combinations(sorted(element_symbols), order):
            systems.append("-".join(combo))
    return systems


def _record_from_doc(
    doc: object,
    only_icsd: bool,
    only_stable: bool,
    energy_above_hull_max: float | None,
) -> dict[str, object] | None:
    """Turn one summary document into a table row, or drop it.

    Args:
        doc: A Materials Project summary document.
        only_icsd: Drop documents with no ICSD cross-reference.
        only_stable: Drop documents above the Materials Project hull.
        energy_above_hull_max: Drop documents further above that hull than this, in eV/atom.

    Returns:
        The row, or ``None`` if the document was filtered out or carries no usable formula.
    """
    icsd_ids = icsd_ids_from_doc(doc)
    if only_icsd and not icsd_ids:
        return None

    energy_above_hull = getattr(doc, "energy_above_hull", None)
    if only_stable and (energy_above_hull is None or energy_above_hull > 0):
        return None
    if energy_above_hull_max is not None and energy_above_hull is not None:
        if energy_above_hull > energy_above_hull_max:
            return None

    formula_pretty = getattr(doc, "formula_pretty", None)
    if not formula_pretty:
        logger.warning("Skipping %s: the summary document carries no formula.", getattr(doc, "material_id", "?"))
        return None

    comp = composition.parse(formula_pretty)
    return {
        "material_id": str(getattr(doc, "material_id", "")),
        "formula_pretty": str(formula_pretty),
        "composition": composition.reduced_formula(comp),
        "natoms": composition.natoms(comp),
        "formation_energy_per_atom": getattr(doc, "formation_energy_per_atom", None),
        "energy_per_atom": getattr(doc, "energy_per_atom", None),
        "is_stable_mp": getattr(doc, "is_stable", None),
        "icsd_ids": icsd_ids,
        "_chemsys": composition.chemsys(comp),
    }


def _lowest_energy_per_formula(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per reduced formula: the one with the lowest formation energy per atom.

    Args:
        frame: Rows built by :func:`_record_from_doc`.

    Returns:
        The filtered rows. Formulas for which Materials Project reports no formation energy
        keep their first row rather than being dropped.
    """
    with_energy = frame[frame["formation_energy_per_atom"].notna()]
    without_energy = frame[frame["formation_energy_per_atom"].isna()]

    kept = with_energy.loc[with_energy.groupby("composition")["formation_energy_per_atom"].idxmin()]
    unmatched = without_energy[~without_energy["composition"].isin(kept["composition"])]
    if unmatched.empty:
        return kept
    return pd.concat([kept, unmatched.drop_duplicates("composition")])


def _empty_table() -> pd.DataFrame:
    """Return an empty table carrying the documented columns.

    Returns:
        An empty :class:`pandas.DataFrame` with the columns of
        :data:`COMPETING_PHASE_COLUMNS`, so that callers can index it without checking.
    """
    return pd.DataFrame(columns=COMPETING_PHASE_COLUMNS)
