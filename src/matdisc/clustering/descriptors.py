"""Per-atom structure descriptors from a machine-learning interatomic potential.

The descriptors are the per-atom feature vectors a DeePMD-kit model produces for a structure
(``DeepPot.eval_descriptor``). They are what the DIRECT clustering in :mod:`matdisc.clustering.sampling`
operates on: every atom of every candidate structure becomes one row of the feature matrix.

Coordinates, cells and atom types are taken straight from :class:`pymatgen.core.Structure`, and the
element order is read from the model itself, so any element the loaded model knows about is
supported and an unknown element raises a message naming it.

DeePMD-kit is an optional extra and is imported inside the functions that need it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from pymatgen.core import Structure

from matdisc.common.logging import get_logger

logger = get_logger(__name__)

#: Environment variable consulted when no model path is passed explicitly.
MODEL_PATH_ENV_VAR = "DPA3_MODEL_PATH"

_DEEPMD_HINT = (
    "DeePMD-kit is required for descriptor extraction but is not installed. Install the optional "
    "extra with: pip install 'materials-discovery-skills[mlip]'"
)


@dataclass
class DescriptorSet:
    """Per-atom descriptors for a set of structures, with the atom-to-structure mapping.

    Attributes:
        features: Array of shape ``(total_atoms, descriptor_dim)``; rows are ordered by structure and
            then by atom within the structure.
        names: One label per structure, in the same order.
        atom_counts: Number of atoms per structure, in the same order.
    """

    features: np.ndarray
    names: list[str]
    atom_counts: list[int]

    def __post_init__(self) -> None:
        """Validate that the mapping matches the feature matrix."""
        if len(self.names) != len(self.atom_counts):
            raise ValueError(f"{len(self.names)} names but {len(self.atom_counts)} atom counts.")
        total = int(sum(self.atom_counts))
        if self.features.shape[0] != total:
            raise ValueError(f"features has {self.features.shape[0]} rows but the atom counts sum to {total}.")

    @property
    def n_structures(self) -> int:
        """Number of structures described."""
        return len(self.names)

    @property
    def n_atoms(self) -> int:
        """Total number of atoms, i.e. the number of rows in :attr:`features`."""
        return int(self.features.shape[0])

    def structure_index_of_atom(self, atom_index: int) -> int:
        """Return the index of the structure a feature row belongs to.

        Args:
            atom_index: Row index into :attr:`features`.

        Returns:
            Index into :attr:`names` / :attr:`atom_counts`.

        Raises:
            IndexError: If ``atom_index`` is outside the feature matrix.
        """
        if atom_index < 0 or atom_index >= self.n_atoms:
            raise IndexError(f"Atom index {atom_index} is outside the {self.n_atoms} rows of this descriptor set.")
        return int(np.searchsorted(np.cumsum(self.atom_counts), atom_index, side="right"))

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the descriptor set to a compressed ``.npz`` file.

        Args:
            path: Destination file; the parent directory is created if missing.

        Returns:
            The path written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features,
            names=np.array(self.names, dtype=object),
            atom_counts=np.array(self.atom_counts, dtype=int),
        )
        logger.info("Wrote %d x %d descriptors to %s", *self.features.shape, path)
        return path

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "DescriptorSet":
        """Read a descriptor set written by :meth:`save`.

        Args:
            path: The ``.npz`` file to read.

        Returns:
            The descriptor set.
        """
        with np.load(path, allow_pickle=True) as data:
            return cls(
                features=data["features"],
                names=[str(name) for name in data["names"].tolist()],
                atom_counts=[int(count) for count in data["atom_counts"].tolist()],
            )


def load_descriptor_model(model_path: str | os.PathLike[str] | None = None, head: str | None = None) -> Any:
    """Load a DeePMD-kit model for descriptor extraction.

    Descriptor extraction needs the model object itself rather than an ASE calculator, and for a
    multi-task checkpoint it needs the head (model branch) to read descriptors from. Use
    :func:`matdisc.common.calculators.load_calculator` instead when you want a calculator for
    relaxation or phonons.

    Args:
        model_path: Path to the checkpoint. Defaults to ``$DPA3_MODEL_PATH``.
        head: Model branch of a multi-task checkpoint, e.g. the pretraining head the model was
            trained with. ``None`` uses the model's default.

    Returns:
        A ``deepmd.infer.DeepPot`` instance.

    Raises:
        ValueError: If no model path was given and the environment variable is not set.
        ImportError: If DeePMD-kit is not installed.
    """
    resolved = str(model_path) if model_path is not None else os.environ.get(MODEL_PATH_ENV_VAR, "")
    if not resolved:
        raise ValueError(
            "No machine-learning potential given. Pass model_path=... or set the "
            f"{MODEL_PATH_ENV_VAR} environment variable to the checkpoint path."
        )
    try:
        from deepmd.infer import DeepPot
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError(_DEEPMD_HINT) from exc

    logger.info("Loading descriptor model %s (head=%s)", resolved, head)
    return DeepPot(resolved, head=head) if head is not None else DeepPot(resolved)


def compute_descriptors(structures: Sequence[Structure], calculator: Any) -> np.ndarray:
    """Compute per-atom descriptors for a list of structures.

    Args:
        structures: Structures to describe.
        calculator: A ``deepmd.infer.DeepPot`` instance, or an ASE calculator wrapping one (the
            DeePMD-kit ASE calculator exposes it as ``calculator.dp``).

    Returns:
        Array of shape ``(total_atoms, descriptor_dim)``, stacked in structure order.

    Raises:
        TypeError: If ``calculator`` cannot provide descriptors.
        ValueError: If a structure contains an element the model does not know, or the list is empty.
    """
    return compute_descriptor_set(structures, calculator).features


def compute_descriptor_set(
    structures: Sequence[Structure],
    calculator: Any,
    names: Sequence[str] | None = None,
) -> DescriptorSet:
    """Compute per-atom descriptors and keep the atom-to-structure mapping.

    Args:
        structures: Structures to describe.
        calculator: A ``deepmd.infer.DeepPot`` instance, or an ASE calculator wrapping one.
        names: One label per structure for the mapping. Defaults to ``structure_0``, ``structure_1``
            and so on.

    Returns:
        The descriptors together with the per-structure labels and atom counts.

    Raises:
        TypeError: If ``calculator`` cannot provide descriptors.
        ValueError: If ``structures`` is empty, ``names`` has the wrong length, or a structure
            contains an element the model does not know.
    """
    if not structures:
        raise ValueError("No structures given.")
    labels = list(names) if names is not None else [f"structure_{i}" for i in range(len(structures))]
    if len(labels) != len(structures):
        raise ValueError(f"{len(labels)} names for {len(structures)} structures.")

    model = _resolve_descriptor_model(calculator)
    type_map = _model_type_map(model)

    blocks: list[np.ndarray] = []
    for label, structure in zip(labels, structures):
        descriptor = _descriptor_for_structure(structure, model, type_map, label)
        blocks.append(descriptor)
        logger.debug("Descriptors for %s: %s", label, descriptor.shape)

    features = np.vstack(blocks)
    logger.info("Computed descriptors for %d structures: %d x %d", len(structures), *features.shape)
    return DescriptorSet(features=features, names=labels, atom_counts=[len(s) for s in structures])


def _descriptor_for_structure(
    structure: Structure,
    model: Any,
    type_map: list[str],
    label: str,
) -> np.ndarray:
    """Evaluate the model descriptors for one structure.

    Returns:
        Array of shape ``(n_atoms, descriptor_dim)``.
    """
    coords = np.asarray(structure.cart_coords, dtype=float).reshape(1, -1)
    cells = np.asarray(structure.lattice.matrix, dtype=float).reshape(1, 9)
    atom_types = [_type_index(site.specie.symbol, type_map, label) for site in structure]

    descriptor = np.asarray(model.eval_descriptor(coords, cells, atom_types))
    if descriptor.ndim == 3:  # (n_frames, n_atoms, dim) with a single frame
        descriptor = descriptor[0]
    return descriptor


def _type_index(symbol: str, type_map: Sequence[str], label: str) -> int:
    """Map an element symbol onto the model's type index.

    Raises:
        ValueError: If the element is not in the model's type map.
    """
    try:
        return list(type_map).index(symbol)
    except ValueError as exc:
        raise ValueError(
            f"Element {symbol!r} in {label} is not in the type map of the loaded model "
            f"({len(type_map)} elements). Use a model whose type map covers it."
        ) from exc


def _resolve_descriptor_model(calculator: Any) -> Any:
    """Return the object that can evaluate descriptors.

    Raises:
        TypeError: If neither the calculator nor a wrapped model exposes ``eval_descriptor``.
    """
    if hasattr(calculator, "eval_descriptor"):
        return calculator
    inner = getattr(calculator, "dp", None)
    if inner is not None and hasattr(inner, "eval_descriptor"):
        return inner
    raise TypeError(
        f"{type(calculator).__name__} cannot provide descriptors. Pass a deepmd.infer.DeepPot "
        "instance (see load_descriptor_model) or an ASE calculator that wraps one as '.dp'."
    )


def _model_type_map(model: Any) -> list[str]:
    """Read the element order from the loaded model.

    Raises:
        TypeError: If the model does not expose a type map.
    """
    getter = getattr(model, "get_type_map", None)
    if getter is None:
        raise TypeError(
            f"{type(model).__name__} does not expose get_type_map(); the element order cannot be "
            "read from the model."
        )
    return [str(symbol) for symbol in getter()]
