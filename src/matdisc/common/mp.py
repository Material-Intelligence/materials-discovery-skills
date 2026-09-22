"""Materials Project client access.

The API key is read from the ``MP_API_KEY`` environment variable at call time, which is also
where :class:`mp_api.client.MPRester` looks for it. No key is stored in this repository, and
:func:`get_api_key` raises with instructions when the variable is unset rather than falling
back to anything.

Get a key from https://materialsproject.org/api and export it::

    export MP_API_KEY=your_key_here
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator, Sequence

from matdisc.common.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from mp_api.client import MPRester
    from pymatgen.core import Structure

__all__ = [
    "MP_API_KEY_ENV",
    "get_api_key",
    "get_mprester",
    "mprester",
    "search_summary",
    "get_structure",
    "icsd_ids_from_doc",
]

logger = get_logger(__name__)

# The name of the variable, never a key. tools/release_check.py flags any credential-shaped
# assignment, and this one is deliberate, so the waiver names that rule and no other.
MP_API_KEY_ENV = "MP_API_KEY"  # release-check: allow credential-assignment
"""Environment variable holding the Materials Project API key."""

_MISSING_KEY_MESSAGE = (
    f"No Materials Project API key found. Set the {MP_API_KEY_ENV} environment variable "
    "or pass api_key=... explicitly. Keys are issued at https://materialsproject.org/api."
)

_MISSING_MP_API_MESSAGE = "mp-api is required for Materials Project access. Install it with 'pip install mp-api'."


def get_api_key(api_key: str | None = None) -> str:
    """Resolve the Materials Project API key.

    Args:
        api_key: An explicit key. When omitted, the ``MP_API_KEY`` environment variable is
            used.

    Returns:
        The resolved key.

    Raises:
        RuntimeError: If no key was passed and the environment variable is unset or empty.
    """
    key = api_key or os.environ.get(MP_API_KEY_ENV, "")
    key = key.strip()
    if not key:
        raise RuntimeError(_MISSING_KEY_MESSAGE)
    return key


def get_mprester(api_key: str | None = None, **kwargs: Any) -> MPRester:
    """Build a Materials Project client.

    Args:
        api_key: An explicit key. When omitted, the ``MP_API_KEY`` environment variable is
            used.
        **kwargs: Forwarded to :class:`mp_api.client.MPRester`.

    Returns:
        An :class:`mp_api.client.MPRester`. It holds an HTTP session, so close it or use it
        as a context manager.

    Raises:
        RuntimeError: If no API key is available.
        ImportError: If mp-api is not installed.
    """
    try:
        from mp_api.client import MPRester
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(_MISSING_MP_API_MESSAGE) from exc
    return MPRester(get_api_key(api_key), **kwargs)


@contextmanager
def mprester(client: MPRester | None = None, api_key: str | None = None) -> Iterator[MPRester]:
    """Yield a Materials Project client, reusing the caller's if one was given.

    A client passed in by the caller is yielded untouched and left open; a client created
    here is closed on exit. This lets a function take an optional ``mpr`` argument without
    each call paying for a new HTTP session.

    Args:
        client: An open client to reuse, or ``None`` to open one.
        api_key: Key used when a client has to be opened.

    Yields:
        The client to query.
    """
    if client is not None:
        yield client
        return
    with get_mprester(api_key) as new_client:
        yield new_client


def search_summary(
    chemsys: str | Sequence[str],
    fields: Sequence[str] | None = None,
    client: MPRester | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> list[Any]:
    """Run a Materials Project summary search for a chemical system.

    Args:
        chemsys: A chemical system such as ``"Ba-Cd-P"``, or several of them. Materials
            Project matches the system exactly, so subsystems have to be asked for
            separately.
        fields: Document fields to fetch. Passing the handful of fields actually used cuts
            the response by roughly an order of magnitude; ``None`` downloads the full
            summary document for every match.
        client: An open client to reuse. One is opened and closed here when omitted.
        api_key: Key used when a client has to be opened.
        **kwargs: Further filters forwarded to the summary endpoint.

    Returns:
        The matching summary documents.
    """
    with mprester(client, api_key) as mpr:
        return list(mpr.materials.summary.search(chemsys=chemsys, fields=list(fields) if fields else None, **kwargs))


def get_structure(
    material_id: str,
    client: MPRester | None = None,
    api_key: str | None = None,
    conventional_unit_cell: bool = False,
) -> Structure:
    """Fetch one structure by Materials Project id.

    Args:
        material_id: A Materials Project id such as ``"mp-8279"``.
        client: An open client to reuse. One is opened and closed here when omitted.
        api_key: Key used when a client has to be opened.
        conventional_unit_cell: Return the conventional cell instead of the primitive one.

    Returns:
        The relaxed structure Materials Project holds for that id.
    """
    with mprester(client, api_key) as mpr:
        return mpr.get_structure_by_material_id(material_id, conventional_unit_cell=conventional_unit_cell)


def icsd_ids_from_doc(doc: Any) -> str:
    """Extract the ICSD collection codes cross-referenced by a summary document.

    Args:
        doc: A Materials Project summary document. Documents fetched without the
            ``database_IDs`` field yield an empty string.

    Returns:
        The codes joined by ``"|"``, each prefixed with ``icsd-`` -- for example
        ``"icsd-260668|icsd-58643"``. Empty when the material has no ICSD cross-reference.
        These are Materials Project cross-references, not ICSD data.
    """
    database_ids = getattr(doc, "database_IDs", None)
    if not database_ids:
        return ""

    try:
        icsd_list = database_ids.get("icsd")
    except AttributeError:
        return ""
    if not icsd_list:
        return ""

    codes = []
    for entry in icsd_list:
        text = str(entry)
        codes.append(text if text.lower().startswith("icsd-") else f"icsd-{text}")
    return "|".join(codes)
