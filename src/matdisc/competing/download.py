"""Structure download for competing phases.

Takes the table :func:`matdisc.competing.search.find_competing_phases` returns and fetches
the relaxed Materials Project structure behind each ``material_id``. Structures downloaded
this way are Materials Project data, licensed CC BY 4.0; redistributing them requires the
attribution described in ``NOTICE.md``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import pandas as pd

from matdisc.common.io import write_structure
from matdisc.common.logging import get_logger
from matdisc.common.mp import get_structure, mprester

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from mp_api.client import MPRester

__all__ = ["download_structures", "download_structure", "download_by_ids", "SUPPORTED_FORMATS"]

logger = get_logger(__name__)

_FORMATS = {
    "vasp": (".vasp", "poscar"),
    "poscar": (".vasp", "poscar"),
    "cif": (".cif", "cif"),
    "json": (".json", "json"),
}

SUPPORTED_FORMATS = tuple(sorted(_FORMATS))
"""Output formats :func:`download_structures` writes. ``vasp`` and ``poscar`` both write POSCAR."""


def download_structures(
    df: pd.DataFrame,
    outdir: str | Path,
    fmt: str = "vasp",
    include_icsd_in_name: bool = True,
    skip_existing: bool = True,
    delay: float = 0.5,
    client: MPRester | None = None,
    api_key: str | None = None,
) -> dict[str, str | None]:
    """Download the structure of every material in a competing-phase table.

    Args:
        df: A table with a ``material_id`` column, such as the one
            :func:`matdisc.competing.search.find_competing_phases` returns. An ``icsd_ids``
            column, when present, is used in the file names.
        outdir: Directory to write into. Created if missing.
        fmt: Output format, one of :data:`SUPPORTED_FORMATS`.
        include_icsd_in_name: Append the first ICSD code to the file name, giving names such
            as ``mp-527_icsd-58642.vasp``.
        skip_existing: Leave files that already exist alone instead of downloading again.
        delay: Seconds to wait between downloads, to stay inside the Materials Project rate
            limit.
        client: An open :class:`mp_api.client.MPRester` to reuse. One is opened and closed
            here when omitted.
        api_key: Materials Project key. Read from ``MP_API_KEY`` when omitted.

    Returns:
        A mapping from material id to the file written, or to ``None`` for downloads that
        failed.

    Raises:
        ValueError: If ``fmt`` is unknown or the table has no ``material_id`` column.
        RuntimeError: If no Materials Project API key is available.
    """
    extension, pymatgen_fmt = _resolve_format(fmt)
    if "material_id" not in df.columns:
        raise ValueError("The table needs a 'material_id' column; got: " + ", ".join(map(str, df.columns)))

    directory = Path(outdir)
    directory.mkdir(parents=True, exist_ok=True)

    results: dict[str, str | None] = {}
    downloaded = skipped = failed = 0
    logger.info("Downloading %d structures to %s", len(df), directory)

    with mprester(client, api_key) as mpr:
        for _, row in df.iterrows():
            material_id = str(row["material_id"]).strip()
            stem = _file_stem(material_id, row.get("icsd_ids") if include_icsd_in_name else None)
            target = directory / f"{stem}{extension}"

            if skip_existing and target.exists():
                logger.debug("Already present, skipping: %s", target.name)
                results[material_id] = str(target)
                skipped += 1
                continue

            try:
                structure = get_structure(material_id, client=mpr)
                write_structure(structure, target, fmt=pymatgen_fmt)
            except Exception as exc:  # noqa: BLE001 - one failed material must not lose the rest
                logger.warning("Download of %s failed: %s", material_id, exc)
                results[material_id] = None
                failed += 1
            else:
                logger.info("Downloaded %s", target.name)
                results[material_id] = str(target)
                downloaded += 1

            if delay > 0:
                time.sleep(delay)

    logger.info("Download finished: %d written, %d already present, %d failed", downloaded, skipped, failed)
    return results


def download_structure(
    material_id: str,
    path: str | Path,
    fmt: str = "vasp",
    client: MPRester | None = None,
    api_key: str | None = None,
) -> Path:
    """Download one structure by Materials Project id.

    Args:
        material_id: A Materials Project id such as ``"mp-527"``.
        path: File to write. Parent directories are created if missing.
        fmt: Output format, one of :data:`SUPPORTED_FORMATS`.
        client: An open :class:`mp_api.client.MPRester` to reuse. One is opened and closed
            here when omitted.
        api_key: Materials Project key. Read from ``MP_API_KEY`` when omitted.

    Returns:
        The path written.

    Raises:
        ValueError: If ``fmt`` is unknown.
        RuntimeError: If no Materials Project API key is available.
    """
    _, pymatgen_fmt = _resolve_format(fmt)
    with mprester(client, api_key) as mpr:
        structure = get_structure(material_id, client=mpr)
    return write_structure(structure, path, fmt=pymatgen_fmt)


def download_by_ids(
    material_ids: Sequence[str],
    outdir: str | Path,
    fmt: str = "vasp",
    skip_existing: bool = True,
    delay: float = 0.5,
    client: MPRester | None = None,
    api_key: str | None = None,
) -> dict[str, str | None]:
    """Download structures for a list of Materials Project ids.

    Args:
        material_ids: The ids to fetch.
        outdir: Directory to write into. Created if missing.
        fmt: Output format, one of :data:`SUPPORTED_FORMATS`.
        skip_existing: Leave files that already exist alone instead of downloading again.
        delay: Seconds to wait between downloads.
        client: An open :class:`mp_api.client.MPRester` to reuse.
        api_key: Materials Project key. Read from ``MP_API_KEY`` when omitted.

    Returns:
        A mapping from material id to the file written, or to ``None`` for downloads that
        failed.

    Raises:
        ValueError: If ``fmt`` is unknown.
        RuntimeError: If no Materials Project API key is available.
    """
    return download_structures(
        pd.DataFrame({"material_id": list(material_ids)}),
        outdir,
        fmt=fmt,
        include_icsd_in_name=False,
        skip_existing=skip_existing,
        delay=delay,
        client=client,
        api_key=api_key,
    )


def _resolve_format(fmt: str) -> tuple[str, str]:
    """Map a format name to a file extension and a pymatgen format.

    Args:
        fmt: Format name, one of :data:`SUPPORTED_FORMATS`.

    Returns:
        The file extension and the format name to hand to pymatgen.

    Raises:
        ValueError: If the format is unknown.
    """
    try:
        return _FORMATS[fmt.strip().lower()]
    except KeyError:
        raise ValueError(f"Unknown format {fmt!r}. Supported: {', '.join(SUPPORTED_FORMATS)}.") from None


def _file_stem(material_id: str, icsd_ids: object) -> str:
    """Build the file name stem for a downloaded structure.

    Args:
        material_id: The Materials Project id.
        icsd_ids: The ``icsd_ids`` cell for that material, or ``None`` to use the id alone.

    Returns:
        ``"mp-527_icsd-58642"`` when an ICSD code is available, otherwise ``"mp-527"``.
    """
    if icsd_ids is None:
        return material_id
    text = str(icsd_ids).strip()
    if not text or text.lower() == "nan":
        return material_id
    first = text.split("|")[0].strip()
    return f"{material_id}_{first}" if first else material_id
