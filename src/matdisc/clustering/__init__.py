"""Descriptor extraction and representative-subset selection stage.

Per-atom descriptors from a machine-learning interatomic potential feed DIRECT sampling, which picks
the structures that are worth a DFT calculation.
"""

from matdisc.clustering.descriptors import (
    DescriptorSet,
    compute_descriptor_set,
    compute_descriptors,
    load_descriptor_model,
)
from matdisc.clustering.sampling import (
    SamplingResult,
    StructureSelection,
    direct_sample,
    map_atoms_to_structures,
    plot_pca_coverage,
    select_representative_structures,
    select_representatives,
    tune_threshold,
    write_selected_structures,
)

__all__ = [
    "DescriptorSet",
    "SamplingResult",
    "StructureSelection",
    "compute_descriptor_set",
    "compute_descriptors",
    "direct_sample",
    "load_descriptor_model",
    "map_atoms_to_structures",
    "plot_pca_coverage",
    "select_representative_structures",
    "select_representatives",
    "tune_threshold",
    "write_selected_structures",
]
