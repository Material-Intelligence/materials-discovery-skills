"""VASP input generation built on :mod:`pymatgen.io.vasp.sets`.

The original screening runs prepared VASP inputs through an in-house job-preparation tool. That
tool is not available outside the group that wrote it, so the input generation is reimplemented
here on top of pymatgen's Materials Project input sets. The convergence and accuracy settings of
the original ``relax``/``scf``/``band``/``hse`` templates are carried over and are collected in
:data:`PORTED_INCAR_SETTINGS`:

===========================  ==================  ====================================================
Original template setting    Value               INCAR equivalent written here
===========================  ==================  ====================================================
energy convergence           ``1e-5`` eV         ``EDIFF = 1e-05``
force convergence            ``0.01`` eV/A       ``EDIFFG = -0.01`` (relaxations only)
k-point spacing              ``0.189``           ``KSPACING = 0.189`` (no KPOINTS file is written)
plane-wave cutoff            ``1.3`` x ENMAX     ``ENCUT = ceil(1.3 * max(ENMAX))`` from the POTCARs
band-structure k-path        ``20``              line-mode k-point density for ``kind="band"``
spin polarisation            off                 ``ISPIN = 1`` and no ``MAGMOM``
exchange-correlation         ``pbe ldau``        PBE with **no** Hubbard U; see below
===========================  ==================  ====================================================

**No Hubbard U is applied to a phosphide.** The original template asked for ``pbe ldau``, but the
Materials Project GGA+U scheme that pymatgen implements applies a U only to oxides and fluorides.
For the chemistry this package is about there is therefore no INCAR equivalent of that request:
the INCAR written for a metal phosphide carries no ``LDAU``, ``LDAUU`` or ``LDAUJ`` tag at all, and
the run is plain PBE. If your own workflow does apply a U to these systems, pass it explicitly
through ``user_incar_settings`` (``LDAU``, ``LDAUTYPE``, ``LDAUL``, ``LDAUU``, ``LDAUJ``, with one
entry per element in POTCAR order) and say which elements carry which U. A hull built from PBE
energies and one built from PBE+U energies are not comparable, so both sides of a hull have to be
made the same way.

POTCAR files are never shipped with this repository. pymatgen looks them up in the directory
named by ``PMG_VASP_PSP_DIR`` (set it in ``~/.pmgrc.yaml`` or in the environment). When that
directory is not configured, :func:`write_vasp_inputs` still writes INCAR, KPOINTS and POSCAR and
replaces POTCAR with a ``POTCAR.spec`` file listing the pseudopotential symbols the run needs, so
inputs can be prepared and inspected on a machine without a VASP licence.

Three of the kinds are continuation runs and cannot be started from the directory as written:
``band`` (``ICHARG = 11``), ``hse-relax`` and ``hse-static`` (both ``ICHARG = 1``) all read a
charge density, so a converged CHGCAR from a preceding ``static`` run has to be copied in first.
That is how the Materials Project workflow runs a hybrid, and it is pymatgen's default for these
sets rather than a choice made here. :func:`write_vasp_inputs` logs a warning saying so, and
reports it in the returned summary under ``requires``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from pymatgen.core import Structure
from pymatgen.io.vasp.sets import (
    MPHSEBSSet,
    MPHSERelaxSet,
    MPNonSCFSet,
    MPRelaxSet,
    MPStaticSet,
    VaspInputSet,
)

from matdisc.common.io import read_structure
from matdisc.common.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from pymatgen.io.vasp.inputs import Potcar

__all__ = [
    "DEFAULT_EDIFF",
    "DEFAULT_EDIFFG",
    "DEFAULT_ENCUT_SCALE",
    "DEFAULT_KSPACING",
    "DEFAULT_LINE_DENSITY",
    "INPUT_SET_KINDS",
    "KINDS_NEEDING_CHGCAR",
    "PORTED_INCAR_SETTINGS",
    "RELAXATION_KINDS",
    "build_input_set",
    "potcar_symbols",
    "write_batch_inputs",
    "write_vasp_inputs",
]

logger = get_logger(__name__)

#: Electronic convergence criterion, in eV (original template: ``energy = 1e-5``).
DEFAULT_EDIFF = 1e-5

#: Ionic convergence criterion, in eV/A (original template: ``force = 0.01``). VASP reads a
#: negative EDIFFG as a force threshold.
DEFAULT_EDIFFG = -0.01

#: k-point spacing in A^-1 (original template: ``kpoints = 0.189``), written as ``KSPACING``.
DEFAULT_KSPACING = 0.189

#: ENCUT is set to this multiple of the largest ENMAX found in the POTCARs (original
#: template: ``cutoff = 1.3``).
DEFAULT_ENCUT_SCALE = 1.3

#: Line-mode k-point density for band-structure runs (original template: ``kpath.band = 20``).
DEFAULT_LINE_DENSITY = 20

#: The input-set kinds this module can write.
#:
#: ``hse-relax`` is a hybrid **ionic relaxation** (``NSW > 0``); ``hse-static`` is the
#: single-point hybrid run that a band gap is read from. They are not interchangeable: a hybrid
#: relaxation on the same k-mesh costs orders of magnitude more than the single point.
INPUT_SET_KINDS = ("relax", "static", "band", "hse-relax", "hse-static")

#: The kinds that move the ions, and are therefore the only ones ``EDIFFG`` applies to.
RELAXATION_KINDS = ("relax", "hse-relax")

#: The kinds whose INCAR reads a charge density (``ICHARG`` 1 or 11), so the directory as written
#: cannot be run until a CHGCAR from a converged ``static`` run is copied into it. Both hybrid
#: kinds are here because pymatgen's Materials Project hybrid sets set ``ICHARG = 1``, which is
#: how the MP workflow runs them: a PBE static run first, then the hybrid on its charge density.
KINDS_NEEDING_CHGCAR = ("band", "hse-relax", "hse-static")

#: The kinds that carry their own explicit KPOINTS file, so ``KSPACING`` is not written for them.
_EXPLICIT_KPOINTS_KINDS = ("band", "hse-static")

#: The settings carried over from the original templates, for reference and for tests.
PORTED_INCAR_SETTINGS: dict[str, float | int] = {
    "EDIFF": DEFAULT_EDIFF,
    "EDIFFG": DEFAULT_EDIFFG,
    "KSPACING": DEFAULT_KSPACING,
    "ENCUT_SCALE": DEFAULT_ENCUT_SCALE,
    "LINE_DENSITY": DEFAULT_LINE_DENSITY,
}

_SET_CLASSES: dict[str, type[VaspInputSet]] = {
    "relax": MPRelaxSet,
    "static": MPStaticSet,
    "band": MPNonSCFSet,
    "hse-relax": MPHSERelaxSet,
    "hse-static": MPHSEBSSet,
}


def _base_incar_settings(kind: str, kspacing: float | None, magnetic: bool) -> dict[str, Any]:
    """Return the INCAR overrides that reproduce the original templates for one kind.

    Args:
        kind: One of :data:`INPUT_SET_KINDS`.
        kspacing: k-point spacing in A^-1, or ``None`` to keep the input set's automatic
            k-point grid (a KPOINTS file is then written instead of a ``KSPACING`` tag).
        magnetic: Whether to run spin-polarised. The original templates ran with spin
            polarisation switched off.

    Returns:
        A mapping suitable for ``user_incar_settings``. A value of ``None`` tells pymatgen to
        drop that tag from the INCAR.
    """
    settings: dict[str, Any] = {"EDIFF": DEFAULT_EDIFF}

    # EDIFFG is an ionic convergence criterion, so it means nothing to a single-point run.
    if kind in RELAXATION_KINDS:
        settings["EDIFFG"] = DEFAULT_EDIFFG

    if not magnetic:
        settings["ISPIN"] = 1
        settings["MAGMOM"] = None

    # A band-structure run follows a k-path and the hybrid single point carries an explicit
    # k-point list, so KSPACING does not apply to either.
    if kspacing is not None and kind not in _EXPLICIT_KPOINTS_KINDS:
        settings["KSPACING"] = float(kspacing)

    return settings


def potcar_symbols(input_set: VaspInputSet) -> list[str]:
    """Return the POTCAR symbols an input set needs, without reading any POTCAR file.

    Args:
        input_set: A pymatgen input set.

    Returns:
        The pseudopotential symbols, for example ``["Ba_sv", "Cd", "P"]``.
    """
    return list(input_set.potcar_symbols)


def _load_potcar(input_set: VaspInputSet) -> Potcar | None:
    """Return the POTCAR object for an input set, or ``None`` when it cannot be built.

    Args:
        input_set: A pymatgen input set.

    Returns:
        The :class:`pymatgen.io.vasp.inputs.Potcar`, or ``None`` when ``PMG_VASP_PSP_DIR`` is
        unset or a required pseudopotential file is missing.
    """
    try:
        return input_set.potcar
    except (OSError, ValueError, KeyError) as exc:
        logger.debug("POTCAR unavailable: %s", exc)
        return None


def _encut_from_potcar(input_set: VaspInputSet, scale: float) -> float | None:
    """Return ``scale * max(ENMAX)`` over the POTCARs of an input set, rounded up.

    Args:
        input_set: A pymatgen input set.
        scale: The multiple of the largest ENMAX to use, as in the original templates.

    Returns:
        The cutoff in eV, or ``None`` when the POTCARs cannot be read.
    """
    potcar = _load_potcar(input_set)
    if potcar is None:
        return None
    enmax_values = [single.enmax for single in potcar]
    if not enmax_values:
        return None
    return float(math.ceil(max(enmax_values) * scale))


def build_input_set(
    structure: Structure,
    kind: str = "relax",
    *,
    kspacing: float | None = DEFAULT_KSPACING,
    encut_scale: float | None = DEFAULT_ENCUT_SCALE,
    magnetic: bool = False,
    line_density: int = DEFAULT_LINE_DENSITY,
    user_incar_settings: dict[str, Any] | None = None,
) -> VaspInputSet:
    """Build the pymatgen input set for one calculation kind.

    Args:
        structure: The structure to calculate.
        kind: ``"relax"`` (PBE ionic relaxation), ``"static"`` (PBE self-consistent field run),
            ``"band"`` (non-self-consistent PBE run along a k-path), ``"hse-relax"`` (HSE06
            ionic relaxation) or ``"hse-static"`` (single-point HSE06 run on a fixed geometry,
            which is what a hybrid band gap is read from).
        kspacing: k-point spacing in A^-1 written as ``KSPACING``. Pass ``None`` to use the
            input set's automatic k-point grid and write a KPOINTS file instead. Ignored for
            ``kind="band"`` and ``kind="hse-static"``, which carry their own k-point list.
        encut_scale: Multiple of the largest POTCAR ENMAX to use for ``ENCUT``. Pass ``None``
            to keep the input set's own ``ENCUT``. When the POTCARs cannot be read, the input
            set's ``ENCUT`` is kept and the substitution is logged.
        magnetic: Whether to run spin-polarised. The original templates did not.
        line_density: Line-mode k-point density, used only for ``kind="band"``.
        user_incar_settings: Extra INCAR tags, applied last so they win over everything above.
            A value of ``None`` removes that tag.

    Returns:
        The configured input set.

    Raises:
        ValueError: If ``kind`` is not one of :data:`INPUT_SET_KINDS`.
    """
    if kind not in _SET_CLASSES:
        raise ValueError(f"Unknown input set kind {kind!r}; expected one of {', '.join(INPUT_SET_KINDS)}")

    settings = _base_incar_settings(kind, kspacing, magnetic)
    if user_incar_settings:
        settings.update(user_incar_settings)

    kwargs: dict[str, Any] = {}
    if kind == "band":
        kwargs["mode"] = "line"
        kwargs["kpoints_line_density"] = int(line_density)
    elif kind == "hse-static":
        # A uniform hybrid single point. pymatgen's "gap" mode needs the VBM and CBM k-points of
        # a previous run, which a standalone directory does not have.
        kwargs["mode"] = "uniform"

    set_class = _SET_CLASSES[kind]
    input_set = set_class(structure, user_incar_settings=settings, **kwargs)

    if encut_scale is not None and "ENCUT" not in settings:
        encut = _encut_from_potcar(input_set, encut_scale)
        if encut is None:
            logger.info(
                "ENCUT left at the %s input set default; computing %.2f x max(ENMAX) needs the POTCARs.",
                kind,
                encut_scale,
            )
        else:
            settings = dict(settings)
            settings["ENCUT"] = encut
            input_set = set_class(structure, user_incar_settings=settings, **kwargs)

    return input_set


def write_vasp_inputs(
    structure: Structure,
    outdir: str | Path,
    kind: str = "relax",
    user_incar_settings: dict[str, Any] | None = None,
    *,
    kspacing: float | None = DEFAULT_KSPACING,
    encut_scale: float | None = DEFAULT_ENCUT_SCALE,
    magnetic: bool = False,
    line_density: int = DEFAULT_LINE_DENSITY,
) -> dict[str, Any]:
    """Write a complete set of VASP inputs for one structure.

    INCAR, POSCAR and, unless ``KSPACING`` is used, KPOINTS are always written. POTCAR is
    written when pymatgen can find the pseudopotentials; otherwise a ``POTCAR.spec`` file
    listing the required symbols is written in its place and a warning is logged.

    The kinds in :data:`KINDS_NEEDING_CHGCAR` are continuation runs: their INCAR sets ``ICHARG``
    to read a charge density, so the directory as written is not yet runnable and a CHGCAR from a
    converged ``static`` run has to be copied into it first. That is logged as a warning and
    named in the returned summary under ``requires``.

    Args:
        structure: The structure to calculate.
        outdir: Directory to write into. It is created if it does not exist.
        kind: One of :data:`INPUT_SET_KINDS`.
        user_incar_settings: Extra INCAR tags, applied after the ported settings.
        kspacing: See :func:`build_input_set`.
        encut_scale: See :func:`build_input_set`.
        magnetic: See :func:`build_input_set`.
        line_density: See :func:`build_input_set`.

    Returns:
        A summary dictionary with the keys ``outdir``, ``kind``, ``files`` (the file names
        written), ``potcar`` (``"POTCAR"`` or ``"POTCAR.spec"``), ``potcar_symbols``,
        ``incar`` (the INCAR tags as written) and ``requires`` (files that must be supplied
        before the directory can run, empty for a self-contained one).
    """
    out_path = Path(outdir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    input_set = build_input_set(
        structure,
        kind,
        kspacing=kspacing,
        encut_scale=encut_scale,
        magnetic=magnetic,
        line_density=line_density,
        user_incar_settings=user_incar_settings,
    )

    symbols = potcar_symbols(input_set)
    use_spec = _load_potcar(input_set) is None
    if use_spec:
        logger.warning(
            "POTCAR files are unavailable, so %s/POTCAR.spec lists the symbols (%s) instead. Set "
            "PMG_VASP_PSP_DIR to a VASP pseudopotential directory to write real POTCARs.",
            out_path,
            ", ".join(symbols),
        )

    input_set.write_input(str(out_path), potcar_spec=use_spec)

    files = sorted(p.name for p in out_path.iterdir() if p.is_file())
    logger.info("Wrote %s inputs for %s to %s", kind, structure.composition.reduced_formula, out_path)

    requires: list[str] = []
    if kind in KINDS_NEEDING_CHGCAR:
        requires.append("CHGCAR")
        logger.warning(
            "The %s input set sets ICHARG = %s, so %s cannot run until a CHGCAR from a converged "
            "'static' run on the same structure is copied into it.",
            kind,
            input_set.incar.get("ICHARG"),
            out_path,
        )

    return {
        "outdir": str(out_path),
        "kind": kind,
        "files": files,
        "potcar": "POTCAR.spec" if use_spec else "POTCAR",
        "potcar_symbols": symbols,
        "incar": dict(input_set.incar),
        "requires": requires,
    }


def _batch_sources(src: Path, patterns: Iterable[str]) -> list[Path]:
    """List the structure files a batch run covers.

    Args:
        src: A single structure file, or a directory of them.
        patterns: Glob patterns matched against ``src`` when it is a directory, in order. A file
            matched by more than one pattern appears once.

    Returns:
        The structure files to prepare, in glob order. A single file yields itself, whether or
        not its name matches any of the patterns: naming a file is an explicit choice.

    Raises:
        FileNotFoundError: If ``src`` is neither a file nor a directory.
    """
    if src.is_file():
        return [src]
    if not src.is_dir():
        raise FileNotFoundError(f"Structure file or directory not found: {src}")

    seen: set[Path] = set()
    found: list[Path] = []
    for pattern in patterns:
        for path in sorted(src.glob(pattern)):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            found.append(path)
    return found


def write_batch_inputs(
    structures: str | Path | Mapping[str, Structure],
    outdir: str | Path,
    kind: str = "relax",
    *,
    patterns: Iterable[str] = ("*.cif", "*.vasp", "POSCAR*", "CONTCAR*"),
    user_incar_settings: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Write VASP inputs for structures already in hand, one file, or a directory of them.

    Each structure gets its own subdirectory of ``outdir``, named after the structure file
    stem or the mapping key. This replaces the batch preparation step of the original
    workflow, which read a directory of relaxed structures and wrote one calculation
    directory per structure.

    Args:
        structures: Named structures to write, or a single structure file, or a directory
            holding structure files. A caller that has already read its structures -- the
            pipeline's DFT stage does -- passes the mapping, so one file selection governs
            both the validation and the writing and no file is parsed twice. A file is taken
            as given; a directory is globbed with ``patterns``.
        outdir: Directory to write the calculation directories into.
        kind: One of :data:`INPUT_SET_KINDS`.
        patterns: Glob patterns matched against ``structures`` when it is a directory, in order.
            A file matched by more than one pattern is only written once. Ignored when
            ``structures`` names a single file or is a mapping.
        user_incar_settings: Extra INCAR tags, applied after the ported settings.
        **kwargs: Forwarded to :func:`write_vasp_inputs`.

    Returns:
        A mapping from structure name to the summary returned by :func:`write_vasp_inputs`.

    Raises:
        FileNotFoundError: If ``structures`` is a path that is neither a file nor a directory.
    """
    out_root = Path(outdir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    named = dict(structures) if isinstance(structures, Mapping) else _read_batch_sources(structures, patterns)

    results: dict[str, dict[str, Any]] = {}
    for name, structure in named.items():
        results[name] = write_vasp_inputs(
            structure,
            out_root / name,
            kind=kind,
            user_incar_settings=user_incar_settings,
            **kwargs,
        )

    logger.info("Prepared %d %s calculation(s) under %s", len(results), kind, out_root)
    return results


def _read_batch_sources(structures: str | Path, patterns: Iterable[str]) -> dict[str, Structure]:
    """Read the structure files a batch was pointed at.

    Args:
        structures: A structure file or a directory of them.
        patterns: Glob patterns matched against a directory.

    Returns:
        A mapping from file stem to structure, in glob order. An unreadable file is logged
        and left out so that one bad file does not stop the batch.

    Raises:
        FileNotFoundError: If the path is neither a file nor a directory.
    """
    named: dict[str, Structure] = {}
    for path in _batch_sources(Path(structures).expanduser().resolve(), patterns):
        try:
            named[path.stem or path.name] = read_structure(path)
        except Exception as exc:  # noqa: BLE001 - one unreadable file must not stop the batch
            logger.error("Skipping %s: could not read the structure (%s)", path, exc)
    return named
