"""Representative-subset selection from a descriptor matrix.

Selection follows DIRECT (BIRCH clustering in a PCA-reduced descriptor space, then a fixed number of
samples per cluster) as implemented in ``maml.sampling.direct``. The rows of the descriptor matrix
are atoms, so the selected rows are mapped back onto the structures they came from before those
structures go to DFT.

``maml`` (and the scikit-learn it builds on) is an optional extra and is imported inside the
functions that need it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from pymatgen.core import Structure

from matdisc.clustering.descriptors import DescriptorSet
from matdisc.common.io import write_structure
from matdisc.common.logging import get_logger

logger = get_logger(__name__)

DEFAULT_THRESHOLD = 0.16
DEFAULT_K_PER_CLUSTER = 1

_MAML_HINT = (
    "maml is required for DIRECT sampling but is not installed. Install the optional extra with: "
    "pip install 'materials-discovery-skills[clustering]'"
)


@dataclass
class SamplingResult:
    """Outcome of one DIRECT sampling run.

    Attributes:
        selected_indices: Row indices of the descriptor matrix that were selected.
        threshold: BIRCH clustering threshold that produced this selection.
        n_total: Number of rows in the descriptor matrix.
        k_per_cluster: Number of rows taken from each cluster.
        pca_features: PCA-reduced descriptors, when the sampler returned them.
        explained_variance: Explained-variance ratio of the PCA components, when available.
    """

    selected_indices: list[int]
    threshold: float
    n_total: int
    k_per_cluster: int = DEFAULT_K_PER_CLUSTER
    pca_features: np.ndarray | None = None
    explained_variance: np.ndarray | None = None

    @property
    def n_selected(self) -> int:
        """Number of selected rows."""
        return len(self.selected_indices)

    @property
    def n_clusters(self) -> int:
        """Number of clusters implied by the selection size."""
        return self.n_selected // self.k_per_cluster if self.k_per_cluster else self.n_selected

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (the PCA features are left out)."""
        return {
            "selected_indices": [int(i) for i in self.selected_indices],
            "threshold": float(self.threshold),
            "n_total": int(self.n_total),
            "k_per_cluster": int(self.k_per_cluster),
            "n_selected": int(self.n_selected),
            "n_clusters": int(self.n_clusters),
            "explained_variance": (
                [float(v) for v in np.asarray(self.explained_variance).ravel()]
                if self.explained_variance is not None
                else None
            ),
        }


@dataclass
class StructureSelection:
    """One structure picked by the sampler, with how many of its atoms were selected.

    Attributes:
        index: Index of the structure in the descriptor set.
        name: Label of the structure in the descriptor set.
        n_atoms: Number of atoms in the structure.
        n_selected_atoms: Number of its atoms that the sampler selected.
        atom_indices: The selected descriptor rows belonging to this structure.
    """

    index: int
    name: str
    n_atoms: int
    n_selected_atoms: int
    atom_indices: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "index": int(self.index),
            "name": self.name,
            "n_atoms": int(self.n_atoms),
            "n_selected_atoms": int(self.n_selected_atoms),
            "atom_indices": [int(i) for i in self.atom_indices],
        }


def direct_sample(
    features: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    k_per_cluster: int = DEFAULT_K_PER_CLUSTER,
) -> SamplingResult:
    """Run DIRECT sampling on a descriptor matrix at a fixed clustering threshold.

    Args:
        features: Array of shape ``(n_rows, descriptor_dim)``.
        threshold: BIRCH clustering threshold; a smaller threshold yields more clusters and
            therefore a larger selection.
        k_per_cluster: Number of rows taken from each cluster.

    Returns:
        The selected row indices together with the PCA features the sampler produced.

    Raises:
        ImportError: If ``maml`` is not installed.
        ValueError: If ``features`` is not a 2-D array.
    """
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {features.shape}.")

    sampler = _build_sampler(threshold=threshold, k_per_cluster=k_per_cluster)
    logger.info("DIRECT sampling %d rows (threshold=%.4f, k=%d)", features.shape[0], threshold, k_per_cluster)
    selection = sampler.fit_transform(features)

    selected = [int(i) for i in selection["selected_indexes"]]
    explained_variance = getattr(getattr(getattr(sampler, "pca", None), "pca", None), "explained_variance_ratio_", None)

    result = SamplingResult(
        selected_indices=selected,
        threshold=threshold,
        n_total=int(features.shape[0]),
        k_per_cluster=k_per_cluster,
        pca_features=selection.get("PCAfeatures"),
        explained_variance=explained_variance,
    )
    logger.info("Selected %d of %d rows in about %d clusters", result.n_selected, result.n_total, result.n_clusters)
    return result


def tune_threshold(
    features: np.ndarray,
    target: int,
    k_per_cluster: int = DEFAULT_K_PER_CLUSTER,
    min_threshold: float = 0.05,
    max_threshold: float = 0.5,
    n_iterations: int = 10,
) -> float:
    """Bisect the BIRCH threshold so that DIRECT selects about ``target`` rows.

    The selection size is monotonic in the threshold only on average, so the result is the largest
    threshold seen during the bisection that still selected at least ``target`` rows.

    Args:
        features: Array of shape ``(n_rows, descriptor_dim)``.
        target: Desired number of selected rows.
        k_per_cluster: Number of rows taken from each cluster.
        min_threshold: Lower bound of the search interval.
        max_threshold: Upper bound of the search interval.
        n_iterations: Number of bisection steps.

    Returns:
        The tuned threshold.

    Raises:
        ImportError: If ``maml`` is not installed.
        ValueError: If ``target`` is not positive or the interval is empty.
    """
    if target <= 0:
        raise ValueError(f"target must be positive, got {target}.")
    if not min_threshold < max_threshold:
        raise ValueError(f"Empty threshold interval [{min_threshold}, {max_threshold}].")

    features = np.asarray(features)
    low, high = min_threshold, max_threshold
    best = (low + high) / 2

    for _ in range(n_iterations):
        mid = (low + high) / 2
        n_selected = len(direct_sample(features, threshold=mid, k_per_cluster=k_per_cluster).selected_indices)
        logger.info("threshold=%.4f selected %d rows (target %d)", mid, n_selected, target)
        if n_selected < target:
            high = mid
        else:
            low = mid
            best = mid

    logger.info("Tuned threshold: %.4f", best)
    return best


def select_representatives(
    X: np.ndarray,
    n: int,
    k_per_cluster: int = DEFAULT_K_PER_CLUSTER,
    threshold: float | None = None,
    min_threshold: float = 0.05,
    max_threshold: float = 0.5,
    n_iterations: int = 10,
) -> np.ndarray:
    """Select about ``n`` representative rows of a descriptor matrix.

    With an explicit ``threshold`` this is a single DIRECT run. Otherwise the threshold is tuned by
    bisection (:func:`tune_threshold`) so the selection lands at or just above ``n``; DIRECT takes
    ``k_per_cluster`` rows per cluster and the cluster count is only indirectly controlled by the
    threshold, so the returned count is close to ``n`` rather than exactly ``n``.

    Args:
        X: Descriptor matrix of shape ``(n_rows, descriptor_dim)``.
        n: Target number of representatives.
        k_per_cluster: Number of rows taken from each cluster.
        threshold: Fixed BIRCH threshold; when given, no tuning is done.
        min_threshold: Lower bound for threshold tuning.
        max_threshold: Upper bound for threshold tuning.
        n_iterations: Number of bisection steps during tuning.

    Returns:
        Sorted array of selected row indices.

    Raises:
        ImportError: If ``maml`` is not installed.
        ValueError: If ``n`` is not positive.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")

    if threshold is None:
        threshold = tune_threshold(
            X,
            target=n,
            k_per_cluster=k_per_cluster,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            n_iterations=n_iterations,
        )
    result = direct_sample(X, threshold=threshold, k_per_cluster=k_per_cluster)
    if result.n_selected != n:
        logger.info("Requested about %d representatives, selected %d", n, result.n_selected)
    return np.sort(np.asarray(result.selected_indices, dtype=int))


def map_atoms_to_structures(
    descriptors: DescriptorSet,
    atom_indices: Sequence[int],
) -> list[StructureSelection]:
    """Map selected descriptor rows back onto the structures they came from.

    Args:
        descriptors: The descriptor set the rows were selected from.
        atom_indices: Selected row indices.

    Returns:
        One :class:`StructureSelection` per distinct structure, ordered by structure index.

    Raises:
        IndexError: If a row index lies outside the descriptor set.
    """
    boundaries = np.cumsum(descriptors.atom_counts)
    per_structure: dict[int, list[int]] = {}
    for atom_index in atom_indices:
        if atom_index < 0 or atom_index >= descriptors.n_atoms:
            raise IndexError(
                f"Atom index {atom_index} is outside the {descriptors.n_atoms} rows of the descriptor set."
            )
        structure_index = int(np.searchsorted(boundaries, atom_index, side="right"))
        per_structure.setdefault(structure_index, []).append(int(atom_index))

    selections = [
        StructureSelection(
            index=structure_index,
            name=descriptors.names[structure_index],
            n_atoms=descriptors.atom_counts[structure_index],
            n_selected_atoms=len(rows),
            atom_indices=sorted(rows),
        )
        for structure_index, rows in sorted(per_structure.items())
    ]
    logger.info("%d selected atoms map onto %d structures", len(list(atom_indices)), len(selections))
    return selections


def select_representative_structures(
    descriptors: DescriptorSet,
    n: int,
    k_per_cluster: int = DEFAULT_K_PER_CLUSTER,
    threshold: float | None = None,
    min_threshold: float = 0.05,
    max_threshold: float = 0.5,
    n_iterations: int = 10,
) -> list[StructureSelection]:
    """Select representative structures by sampling their atoms.

    Sampling happens on atoms; a structure is selected as soon as at least one of its atoms is.
    The number of structures is therefore at most ``n`` and usually somewhat below it.

    Args:
        descriptors: Per-atom descriptors with the atom-to-structure mapping.
        n: Target number of selected atoms.
        k_per_cluster: Number of atoms taken from each cluster.
        threshold: Fixed BIRCH threshold; when given, no tuning is done.
        min_threshold: Lower bound for threshold tuning.
        max_threshold: Upper bound for threshold tuning.
        n_iterations: Number of bisection steps during tuning.

    Returns:
        The selected structures, ordered by their index in the descriptor set.

    Raises:
        ImportError: If ``maml`` is not installed.
    """
    indices = select_representatives(
        descriptors.features,
        n=n,
        k_per_cluster=k_per_cluster,
        threshold=threshold,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        n_iterations=n_iterations,
    )
    return map_atoms_to_structures(descriptors, indices.tolist())


def write_selected_structures(
    structures: Sequence[Structure],
    selections: Sequence[StructureSelection],
    output_dir: str | os.PathLike[str],
    fmt: str = "poscar",
    suffix: str = ".vasp",
) -> list[Path]:
    """Write the selected structures to disk, one file each.

    Args:
        structures: The structures the descriptor set was built from, in the same order.
        selections: Selections returned by :func:`select_representative_structures`.
        output_dir: Directory to write into; created if missing.
        fmt: Structure format passed to :func:`matdisc.common.io.write_structure`.
        suffix: File suffix for the written files.

    Returns:
        The paths written, in selection order.

    Raises:
        IndexError: If a selection refers to a structure outside ``structures``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for selection in selections:
        if selection.index >= len(structures):
            raise IndexError(f"Selection {selection.name} refers to structure {selection.index} of {len(structures)}.")
        path = output_dir / f"{Path(selection.name).stem}{suffix}"
        write_structure(structures[selection.index], path, fmt)
        written.append(path)
    logger.info("Wrote %d selected structures to %s", len(written), output_dir)
    return written


def plot_pca_coverage(
    result: SamplingResult,
    output_file: str | os.PathLike[str],
    title: str = "DIRECT sampling coverage",
) -> Path:
    """Plot the selected rows against all rows in the first two PCA components.

    Axis convention: each component is divided by its explained-variance ratio when the sampler
    reported one, which is how the DIRECT paper displays coverage. maml's own PCA output is
    already weighted by explained variance, so this division undoes that weighting and the axes
    read as unweighted components -- the low-variance direction is stretched, not shrunk. The
    picture is a diagnostic of coverage, not a metric; nothing downstream reads these numbers.

    The figure is built through the object-oriented interface with the Agg canvas, so importing
    or calling this function never selects a global matplotlib backend.

    Args:
        result: A sampling result carrying PCA features.
        output_file: Image file to write; the parent directory is created if missing.
        title: Plot title.

    Returns:
        The path written.

    Raises:
        ValueError: If the result carries no PCA features or fewer than two components.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if result.pca_features is None:
        raise ValueError("This sampling result carries no PCA features, so coverage cannot be plotted.")
    features = np.asarray(result.pca_features)
    if features.ndim != 2 or features.shape[1] < 2:
        raise ValueError(f"Need at least two PCA components to plot, got shape {features.shape}.")

    plotted = features[:, :2]
    if result.explained_variance is not None and len(result.explained_variance) >= 2:
        plotted = plotted / np.asarray(result.explained_variance)[:2]
    selected = plotted[result.selected_indices]

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(6, 6))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.scatter(plotted[:, 0], plotted[:, 1], alpha=0.3, s=10, label=f"All ({len(plotted):,} atoms)")
    axes.scatter(selected[:, 0], selected[:, 1], alpha=0.7, s=20, c="red", label=f"Selected ({len(selected):,} atoms)")
    axes.set_xlabel("PC 1", fontsize=14)
    axes.set_ylabel("PC 2", fontsize=14)
    axes.set_title(title, fontsize=16)
    axes.legend(fontsize=12)
    figure.tight_layout()
    figure.savefig(output_file, dpi=150)

    logger.info("Wrote PCA coverage plot to %s", output_file)
    return output_file


def _build_sampler(threshold: float, k_per_cluster: int) -> Any:
    """Build a DIRECT sampler.

    Raises:
        ImportError: If ``maml`` is not installed.
    """
    try:
        from maml.sampling.direct import BirchClustering, DIRECTSampler, SelectKFromClusters
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError(_MAML_HINT) from exc

    return DIRECTSampler(
        structure_encoder=None,
        clustering=BirchClustering(threshold_init=threshold),
        select_k_from_clusters=SelectKFromClusters(k=k_per_cluster),
    )
