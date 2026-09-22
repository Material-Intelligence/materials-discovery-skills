"""Shared fixtures and offline test doubles.

Nothing in the test suite touches the network, a GPU, a cluster or an optional backend
unless it is marked ``network``, ``gpu`` or ``extras``. The Materials Project client is
replaced by :class:`FakeMPRester`, which mimics the small part of
:class:`mp_api.client.MPRester` this package uses: ``materials.summary.search`` and
``get_structure_by_material_id``, plus the context-manager protocol.

Every public function that talks to the Materials Project takes a ``client`` argument, so a
fake can be injected without patching. The tests do both: injection where the API offers it,
and monkeypatching :func:`matdisc.common.mp.get_mprester` where a caller further up the stack
opens its own client.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import pytest
from pymatgen.core import Lattice, Structure

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
"""Directory holding the shipped example data."""


class FakeSummaryDoc:
    """A stand-in for a Materials Project summary document.

    Only the attributes this package reads are present, so a test that starts reading a new
    attribute fails loudly instead of silently picking up a Mock.
    """

    def __init__(
        self,
        material_id: str,
        formula_pretty: str,
        formation_energy_per_atom: float | None = -0.5,
        energy_per_atom: float | None = -5.0,
        energy_above_hull: float | None = 0.0,
        is_stable: bool = True,
        icsd_ids: Sequence[object] | None = (),
        database_ids: dict[str, list[object]] | None = None,
    ) -> None:
        """Build a fake summary document.

        Args:
            material_id: Materials Project id, for example ``"mp-8279"``.
            formula_pretty: The formula as Materials Project reports it.
            formation_energy_per_atom: Formation energy in eV/atom, or ``None``.
            energy_per_atom: Total energy in eV/atom, or ``None``.
            energy_above_hull: Energy above the Materials Project hull in eV/atom.
            is_stable: Whether Materials Project puts the material on its own hull.
            icsd_ids: ICSD collection codes to expose through ``database_IDs``.
            database_ids: A ``database_IDs`` mapping to use verbatim, overriding ``icsd_ids``.
        """
        self.material_id = material_id
        self.formula_pretty = formula_pretty
        self.formation_energy_per_atom = formation_energy_per_atom
        self.energy_per_atom = energy_per_atom
        self.energy_above_hull = energy_above_hull
        self.is_stable = is_stable
        if database_ids is not None:
            self.database_IDs = database_ids
        elif icsd_ids:
            self.database_IDs = {"icsd": list(icsd_ids)}
        else:
            self.database_IDs = {}


class FakeMPRester:
    """An offline stand-in for :class:`mp_api.client.MPRester`.

    Records every query so a test can assert which subsystems were asked for and which
    document fields were requested.
    """

    def __init__(
        self,
        docs_by_chemsys: dict[str, list[FakeSummaryDoc]] | None = None,
        structures: dict[str, Structure] | None = None,
        failing_ids: Iterable[str] = (),
    ) -> None:
        """Build a fake client.

        Args:
            docs_by_chemsys: Summary documents to return per chemical system. A system that
                is not a key returns no documents, as Materials Project does.
            structures: Structures to return per material id. Ids that are not keys fall
                back to a small cubic placeholder.
            failing_ids: Material ids whose structure download must raise.
        """
        self.docs_by_chemsys = docs_by_chemsys or {}
        self.structures = structures or {}
        self.failing_ids = set(failing_ids)
        self.searches: list[dict[str, Any]] = []
        self.requested_ids: list[str] = []
        self.closed = False
        self.materials = SimpleNamespace(summary=SimpleNamespace(search=self._search))

    def _search(self, chemsys: str | None = None, fields: Sequence[str] | None = None, **kwargs: Any) -> list[Any]:
        """Answer a summary search.

        Args:
            chemsys: The chemical system asked for.
            fields: The document fields asked for.
            **kwargs: Any further filters, recorded but ignored.

        Returns:
            The documents registered for that system.
        """
        self.searches.append({"chemsys": chemsys, "fields": list(fields) if fields else None, **kwargs})
        return list(self.docs_by_chemsys.get(str(chemsys), []))

    def get_structure_by_material_id(self, material_id: str, conventional_unit_cell: bool = False) -> Structure:
        """Answer a structure request.

        Args:
            material_id: The id asked for.
            conventional_unit_cell: Accepted and ignored; recorded only for signature parity.

        Returns:
            The registered structure, or a small cubic placeholder.

        Raises:
            RuntimeError: If the id was registered as failing.
        """
        self.requested_ids.append(material_id)
        if material_id in self.failing_ids:
            raise RuntimeError(f"simulated Materials Project failure for {material_id}")
        return self.structures.get(material_id, cubic_structure("Ba", "Cd"))

    def __enter__(self) -> FakeMPRester:
        """Enter the context manager.

        Returns:
            This client.
        """
        return self

    def __exit__(self, *exc_info: object) -> bool:
        """Leave the context manager and record that it closed.

        Args:
            *exc_info: Exception information, ignored.

        Returns:
            ``False``, so exceptions propagate.
        """
        self.closed = True
        return False


def cubic_structure(*species: str, a: float = 4.0) -> Structure:
    """Build a small cubic structure for tests that only need a valid one.

    Args:
        *species: Element symbols, one per site. Sites are spread along the cell diagonal.
        a: Cubic lattice parameter in Angstrom.

    Returns:
        The structure.
    """
    count = len(species)
    coords = [[i / count, i / count, i / count] for i in range(count)]
    return Structure(Lattice.cubic(a), list(species), coords)


def ba_cd_p_docs() -> dict[str, list[FakeSummaryDoc]]:
    """Return a small offline Ba-Cd-P set of summary documents.

    The values mirror the shape of a real harvest -- elemental references with zero
    formation energy, ICSD-backed binaries, and one ternary -- without claiming to be the
    Materials Project's current numbers.

    Returns:
        A mapping from chemical system to the documents that system returns.
    """
    return {
        "Ba": [FakeSummaryDoc("mp-122", "Ba", 0.0, -1.9, icsd_ids=[77367])],
        "Cd": [FakeSummaryDoc("mp-94", "Cd", 0.0, -0.9, icsd_ids=[53770])],
        "P": [FakeSummaryDoc("mp-568348", "P", 0.0, -5.4, icsd_ids=[29273])],
        "Ba-Cd": [
            FakeSummaryDoc("mp-527", "BaCd", -0.31, -1.6, icsd_ids=[58642, 615805]),
            FakeSummaryDoc("mp-9999", "BaCd3", -0.20, -1.3, icsd_ids=[]),
        ],
        "Ba-P": [FakeSummaryDoc("mp-1105095", "BaP2", -0.80, -4.6, icsd_ids=[429732])],
        "Cd-P": [FakeSummaryDoc("mp-2441", "Cd3P2", -0.12, -3.1, icsd_ids=[181134])],
        "Ba-Cd-P": [FakeSummaryDoc("mp-8279", "Ba(CdP)2", -0.565, -3.7, icsd_ids=[615814, 30915])],
    }


@pytest.fixture
def fake_mprester() -> FakeMPRester:
    """Return a fake Materials Project client loaded with the Ba-Cd-P documents.

    Returns:
        The client, ready to pass as ``client=`` or to return from a patched
        :func:`matdisc.common.mp.get_mprester`.
    """
    return FakeMPRester(docs_by_chemsys=ba_cd_p_docs())


@pytest.fixture
def example_competing_csv() -> Path:
    """Return the path of the shipped Ba-Cd-P competing-phase table.

    Returns:
        The CSV path.
    """
    path = DATA_DIR / "BaCdP_competing_phases.csv"
    if not path.is_file():  # pragma: no cover - only if the example data is removed
        pytest.skip(f"example data not present: {path}")
    return path


@pytest.fixture(autouse=True)
def _no_ambient_mp_key(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Hide any ambient ``MP_API_KEY`` from tests that are not marked ``network``.

    A developer with a key exported would otherwise have offline tests reach the real API
    the moment a mock was wired up wrongly, and the failure would only appear on a machine
    without a key.

    Args:
        monkeypatch: pytest's monkeypatch fixture.
        request: The test request, used to read the ``network`` marker.
    """
    if request.node.get_closest_marker("network") is None:
        monkeypatch.delenv("MP_API_KEY", raising=False)
