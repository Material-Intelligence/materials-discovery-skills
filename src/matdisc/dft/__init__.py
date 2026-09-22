"""VASP input generation, job submission and output parsing.

The three modules mirror the three steps of a DFT stage:

* :mod:`matdisc.dft.inputs` writes INCAR, KPOINTS, POSCAR and POTCAR through pymatgen's
  Materials Project input sets.
* :mod:`matdisc.dft.slurm` writes a plain ``sbatch`` script and submits it.
* :mod:`matdisc.dft.outputs` reads the finished runs back into a table the convex-hull and
  fine-tuning stages can consume.

Nothing here requires VASP itself to be installed. Input generation degrades to a
``POTCAR.spec`` listing when no pseudopotentials are configured, and submission is only
attempted when ``sbatch`` is on ``PATH``.
"""

from matdisc.dft.inputs import build_input_set, write_batch_inputs, write_vasp_inputs
from matdisc.dft.outputs import (
    VaspResult,
    check_calculation_status,
    collect_results,
    export_for_finetuning,
    parse_outcar,
    parse_vasprun,
    read_calculation,
)
from matdisc.dft.slurm import SlurmSettings, submit_directory, submit_job, write_slurm_script

__all__ = [
    "SlurmSettings",
    "VaspResult",
    "build_input_set",
    "check_calculation_status",
    "collect_results",
    "export_for_finetuning",
    "parse_outcar",
    "parse_vasprun",
    "read_calculation",
    "submit_directory",
    "submit_job",
    "write_batch_inputs",
    "write_slurm_script",
    "write_vasp_inputs",
]
