"""Generative structure design stage.

Wraps MatterGen so that a chemical system and a structure count come in and
:class:`pymatgen.core.Structure` objects come out.
"""

from matdisc.generation.mattergen import (
    GenerationResult,
    element_combinations,
    generate,
    generate_many,
    parse_chemical_system,
    read_chemical_systems,
    read_generated_structures,
)

__all__ = [
    "GenerationResult",
    "element_combinations",
    "generate",
    "generate_many",
    "parse_chemical_system",
    "read_chemical_systems",
    "read_generated_structures",
]
