"""Dynamical-stability check from a phonon spectrum computed with ASE.

:func:`check_dynamical_stability` displaces atoms in a supercell with
:class:`ase.phonons.Phonons`, evaluates the forces with the calculator it is given, and
reports the lowest phonon frequency along a high-symmetry band path. A structure counts as
dynamically stable when that frequency sits above ``threshold`` (-0.1 THz by default), which
tolerates the small imaginary frequencies that finite displacements leave at the zone
centre.

All output paths are resolved to absolute paths when the function is entered and nothing
changes the working directory, so a relative ``output_dir`` lands where the caller expects
it. The ASE force cache lives in ``<output_dir>/phonon`` and is removed once the force
constants have been read.

Cost warning: the force evaluations scale with the number of atoms in the supercell. A
supercell chosen by ``min_cell_length`` can easily hold a few hundred atoms.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from monty.json import MSONable
from pymatgen.core import Structure

from matdisc.common.io import from_ase, to_ase, write_structure
from matdisc.common.logging import get_logger

if TYPE_CHECKING:  # imported for type checking only; ASE is imported where it is used
    from ase import Atoms
    from ase.calculators.calculator import Calculator

LOGGER = get_logger(__name__)

EV_TO_THZ = 241.79893
"""Conversion from eV to THz, used for every frequency this module reports."""

DEFAULT_THRESHOLD = -0.1
"""Default imaginary-frequency tolerance, in THz."""

DEFAULT_MIN_CELL_LENGTH = 10.0
"""Default minimum supercell edge length, in A, used when no supercell is given."""

DEFAULT_DELTA = 0.01
"""Default finite displacement, in A."""

DEFAULT_RELAX_FMAX = 0.001
"""Default force criterion of the pre-relaxation, in eV/A.

It is an order of magnitude tighter than the screening relaxation because the displacements
are read as derivatives at a stationary point.
"""

DEFAULT_RELAX_MAX_STEPS = 1000
"""Default step cap of the pre-relaxation."""

__all__ = [
    "DEFAULT_DELTA",
    "DEFAULT_MIN_CELL_LENGTH",
    "DEFAULT_RELAX_FMAX",
    "DEFAULT_RELAX_MAX_STEPS",
    "DEFAULT_THRESHOLD",
    "EV_TO_THZ",
    "PhononResult",
    "check_dynamical_stability",
    "min_frequency_from_band_file",
    "supercell_for",
]


@dataclass
class PhononResult(MSONable):
    """Outcome of one phonon calculation.

    Attributes:
        success: Whether the spectrum was computed.
        output_dir: Absolute path the run wrote to.
        supercell: Supercell repetitions used for the force constants.
        min_frequency: Lowest frequency along the band path, in THz. Negative values are
            imaginary frequencies. ``None`` when the run failed.
        threshold: Frequency threshold the verdict used, in THz.
        is_stable: Whether ``min_frequency`` is above ``threshold``.
        band_path: The high-symmetry path the spectrum was sampled on.
        relaxation_converged: Whether the pre-relaxation reached its force criterion.
            ``None`` when no pre-relaxation was asked for. A spectrum computed on a geometry
            that is not stationary carries imaginary frequencies of its own, so this field
            says whether the verdict can be read as a property of the material.
        error_message: Why the run failed, or ``None``.
    """

    success: bool
    output_dir: str
    supercell: tuple[int, int, int] | None = field(default=None)
    min_frequency: float | None = field(default=None)
    threshold: float = field(default=DEFAULT_THRESHOLD)
    is_stable: bool = field(default=False)
    band_path: str | None = field(default=None)
    relaxation_converged: bool | None = field(default=None)
    error_message: str | None = field(default=None)


def supercell_for(atoms: Atoms, min_length: float = DEFAULT_MIN_CELL_LENGTH) -> tuple[int, int, int]:
    """Choose supercell repetitions that bring every cell edge up to a minimum length.

    Args:
        atoms: An :class:`ase.Atoms` object carrying a cell.
        min_length: Minimum edge length of the supercell, in A.

    Returns:
        Repetitions along the three lattice vectors, each at least 1.
    """
    lengths = atoms.cell.lengths()
    return tuple(max(1, int(np.ceil(min_length / length))) for length in lengths)


def check_dynamical_stability(
    structure: Structure,
    calculator: Calculator,
    output_dir: str | Path = "phonon_output",
    supercell: tuple[int, int, int] | None = None,
    min_cell_length: float = DEFAULT_MIN_CELL_LENGTH,
    delta: float = DEFAULT_DELTA,
    relax_first: bool = True,
    fmax: float = DEFAULT_RELAX_FMAX,
    max_steps: int = DEFAULT_RELAX_MAX_STEPS,
    require_relaxation_converged: bool = True,
    threshold: float = DEFAULT_THRESHOLD,
    band_npoints: int = 100,
    dos_kpts: tuple[int, int, int] = (20, 20, 12),
    plot: bool = True,
) -> dict:
    """Compute a phonon spectrum and report whether the structure is dynamically stable.

    Args:
        structure: Structure to check.
        calculator: An ASE calculator, for example the one returned by
            :func:`matdisc.common.calculators.load_calculator`.
        output_dir: Directory for the spectrum, the density of states and the plot. It is
            resolved to an absolute path and created if missing.
        supercell: Explicit supercell repetitions. When ``None`` they are chosen by
            :func:`supercell_for` from ``min_cell_length``.
        min_cell_length: Minimum supercell edge length, in A, used only when ``supercell``
            is ``None``.
        delta: Finite displacement, in A.
        relax_first: Relax the atomic positions with BFGS before displacing them. The cell
            is left untouched.
        fmax: Force criterion of that relaxation, in eV/A.
        max_steps: Step cap of that relaxation.
        require_relaxation_converged: Stop when that relaxation hits the step cap without
            reaching ``fmax``, instead of displacing a geometry that is not stationary. The
            spectrum of a non-stationary geometry is not an approximation of the phonon
            spectrum, it is a different quantity, and its imaginary modes say nothing about
            the material.
        threshold: A structure is stable when its lowest frequency is above this value, in
            THz.
        band_npoints: Number of k-points along the band path.
        dos_kpts: Monkhorst-Pack grid for the phonon density of states.
        plot: Write ``phonon_spectrum.png`` alongside the data files.

    Returns:
        :meth:`PhononResult.as_dict` output, carrying ``success``, ``min_frequency`` (THz),
        ``is_stable``, ``supercell``, ``output_dir``, ``relaxation_converged`` and, on
        failure, ``error_message``. A failure is reported in the dictionary rather than
        raised, so that a batch of structures runs to the end.

    Files written into ``output_dir``:
        ``band_structure.dat`` (distance and frequencies in THz), ``dos.dat``,
        ``band_path_info.txt``, ``phonon_spectrum.png`` when ``plot`` is set, and
        ``relaxed.vasp`` plus ``relaxation.log`` when ``relax_first`` is set.
    """
    from ase.optimize import BFGS
    from ase.phonons import Phonons

    directory = Path(output_dir).expanduser().resolve()
    result = PhononResult(success=False, output_dir=str(directory), threshold=threshold)

    try:
        directory.mkdir(parents=True, exist_ok=True)

        cache_dir = directory / "phonon"
        if cache_dir.exists():
            LOGGER.info("Removing stale force cache %s", cache_dir)
            shutil.rmtree(cache_dir)

        atoms = to_ase(structure)
        atoms.calc = calculator

        if relax_first:
            LOGGER.info("Relaxing atomic positions before displacing them")
            optimizer = BFGS(atoms, logfile=str(directory / "relaxation.log"))
            converged = bool(optimizer.run(fmax=fmax, steps=max_steps))
            result.relaxation_converged = converged
            write_structure(from_ase(atoms), directory / "relaxed.vasp", fmt="poscar")
            if not converged and require_relaxation_converged:
                result.error_message = (
                    f"pre-relaxation did not converge: {max_steps} step(s) did not reach fmax = {fmax:g} eV/A. "
                    "The spectrum of a geometry that is not stationary is a different quantity, so it was not "
                    "computed; relax further, raise max_steps, or pass require_relaxation_converged=False."
                )
                LOGGER.warning("%s in %s", result.error_message, directory)
                return result.as_dict()

        repetitions = tuple(supercell) if supercell is not None else supercell_for(atoms, min_cell_length)
        result.supercell = repetitions
        LOGGER.info("Supercell %s, displacement %.3f A", repetitions, delta)

        phonons = Phonons(atoms.copy(), calculator, supercell=repetitions, delta=delta, name=str(cache_dir))
        LOGGER.info("Evaluating forces for the displaced configurations")
        phonons.run()
        phonons.read(acoustic=True)
        phonons.clean()

        try:
            path = atoms.cell.bandpath(npoints=band_npoints)
        except Exception as error:
            LOGGER.warning("Could not build an automatic band path (%s); falling back to G-X", error)
            path = atoms.cell.bandpath("GX", npoints=band_npoints)
        result.band_path = str(path.path)

        band_structure = phonons.get_band_structure(path, verbose=False)
        density_of_states = phonons.get_dos(kpts=dos_kpts, verbose=False).sample_grid(npts=200, width=1e-3)

        _write_band_path(path, directory)
        _write_spectrum(band_structure, density_of_states, path, directory)
        min_frequency = float(np.min(band_structure.energies[0]) * EV_TO_THZ)
        if plot:
            _plot_spectrum(band_structure, density_of_states, path, directory, min_frequency)

        result.success = True
        result.min_frequency = min_frequency
        result.is_stable = bool(min_frequency > threshold)
        LOGGER.info(
            "Lowest frequency %.3f THz (threshold %.3f THz): %s",
            min_frequency,
            threshold,
            "dynamically stable" if result.is_stable else "dynamically unstable",
        )
    except Exception as error:  # reported through the result so that a batch continues
        LOGGER.warning("Phonon calculation in %s failed: %s", directory, error, exc_info=True)
        result.success = False
        result.error_message = str(error)

    return result.as_dict()


def min_frequency_from_band_file(output_dir: str | Path) -> float:
    """Read the lowest frequency from a ``band_structure.dat`` written by an earlier run.

    Args:
        output_dir: Directory holding ``band_structure.dat``.

    Returns:
        The lowest frequency in THz. Compare it with a threshold to get a verdict.

    Raises:
        FileNotFoundError: If the file is not there.
    """
    path = Path(output_dir).expanduser().resolve() / "band_structure.dat"
    if not path.exists():
        raise FileNotFoundError(f"No phonon band data at {path}; run check_dynamical_stability first")
    data = np.loadtxt(path)
    return float(np.min(data[:, 1:]))


def _write_band_path(path: Any, directory: Path) -> None:
    """Write the high-symmetry path and its special points to ``band_path_info.txt``.

    Args:
        path: The ASE band path.
        directory: Absolute output directory.
    """
    lines = [
        f"# High-symmetry path: {path.path}",
        "",
        "# Special points (fractional coordinates):",
        "# Label k_x k_y k_z",
    ]
    for label, coords in path.special_points.items():
        lines.append(f"{label:<5} {coords[0]:8.5f} {coords[1]:8.5f} {coords[2]:8.5f}")
    (directory / "band_path_info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_spectrum(band_structure: Any, density_of_states: Any, path: Any, directory: Path) -> None:
    """Write the band structure and the density of states as text, in THz.

    Args:
        band_structure: The ASE band structure returned by ``Phonons.get_band_structure``.
        density_of_states: The sampled density of states.
        path: The ASE band path.
        directory: Absolute output directory.
    """
    distances, _, _ = path.get_linear_kpoint_axis()
    energies = band_structure.energies[0] * EV_TO_THZ
    np.savetxt(
        directory / "band_structure.dat",
        np.hstack((distances[:, np.newaxis], energies)),
        header=f"Distance(1/A) Frequencies(THz) - {energies.shape[1]} bands",
        fmt="%.8f",
    )

    dos_energies = density_of_states.get_energies() * EV_TO_THZ
    dos_weights = density_of_states.get_weights()
    np.savetxt(
        directory / "dos.dat",
        np.hstack((dos_energies[:, np.newaxis], dos_weights[:, np.newaxis])),
        header="Frequency(THz) DOS",
        fmt="%.8f",
    )


def _plot_spectrum(
    band_structure: Any,
    density_of_states: Any,
    path: Any,
    directory: Path,
    min_frequency: float,
) -> None:
    """Draw the phonon spectrum next to its density of states.

    The figure is built through matplotlib's object-oriented interface with the Agg canvas,
    so no interactive backend is selected and no global state is touched.

    Args:
        band_structure: The ASE band structure.
        density_of_states: The sampled density of states.
        path: The ASE band path.
        directory: Absolute output directory.
        min_frequency: Lowest frequency in THz, shown in the title.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    distances, tick_positions, tick_labels = path.get_linear_kpoint_axis()
    frequencies = band_structure.energies[0] * EV_TO_THZ

    ticks: list[float] = []
    labels: list[str] = []
    for index, position in enumerate(tick_positions):
        if index > 0 and abs(position - ticks[-1]) < 1e-4:
            labels[-1] = f"{labels[-1]}|{tick_labels[index]}"
        else:
            ticks.append(position)
            labels.append(tick_labels[index])

    figure = Figure(figsize=(10, 6))
    FigureCanvasAgg(figure)
    bands_axes, dos_axes = figure.subplots(1, 2, width_ratios=[3, 1])

    for band in range(frequencies.shape[1]):
        bands_axes.plot(distances, frequencies[:, band], color="tab:blue", lw=1.5)
    bands_axes.axhline(0, linestyle=":", color="black", lw=1)
    for position in ticks:
        bands_axes.axvline(x=position, color="gray", linestyle="--", lw=1.0)
    bands_axes.set_xticks(ticks)
    bands_axes.set_xticklabels(labels)
    bands_axes.set_xlim(0, distances[-1])
    bands_axes.set_ylabel("Frequency [THz]")
    bands_axes.set_title(f"Phonon spectrum (lowest frequency {min_frequency:.2f} THz)")

    dos_axes.fill_betweenx(
        density_of_states.get_energies() * EV_TO_THZ,
        density_of_states.get_weights(),
        x2=0,
        color="lightgray",
        edgecolor="black",
        lw=1,
    )
    dos_axes.set_xlabel("DOS")
    dos_axes.set_yticks([])
    dos_axes.set_ylim(bands_axes.get_ylim())
    dos_axes.set_xlim(left=0)

    figure.tight_layout()
    figure.savefig(directory / "phonon_spectrum.png", dpi=300)
