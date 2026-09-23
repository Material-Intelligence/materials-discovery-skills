# Third-party software and data

This project is released under the MIT License (see `LICENSE`). It **drives** third-party
software through public APIs and command-line interfaces; it does not vendor or redistribute
any of it. The licences below therefore apply to those projects when you install them, not to
the code in this repository.

## Required dependencies

| Component | Licence | Project |
|---|---|---|
| pymatgen | MIT | <https://github.com/materialsproject/pymatgen> |
| mp-api | BSD-3-Clause, Copyright (c) 2017 The Regents of the University of California, through Lawrence Berkeley National Laboratory | <https://github.com/materialsproject/api> |
| ASE (Atomic Simulation Environment) | LGPL-2.1-or-later | <https://gitlab.com/ase/ase> |
| monty | MIT | <https://github.com/materialsvirtuallab/monty> |
| NumPy | BSD-3-Clause | <https://github.com/numpy/numpy> |
| pandas | BSD-3-Clause | <https://github.com/pandas-dev/pandas> |
| Matplotlib | Matplotlib licence (BSD-compatible, PSF-derived) | <https://github.com/matplotlib/matplotlib> |
| PyYAML | MIT | <https://github.com/yaml/pyyaml> |

These are the packages `pyproject.toml` declares and `src/matdisc` imports directly. Installing
them pulls in further packages of their own — SciPy, spglib and the rest of the pymatgen stack —
each under its own licence, which their own distributions carry.

## Optional dependencies (extras)

These are **not** installed by default. Every module that uses one imports it lazily, inside
the function that needs it, so `import matdisc` and each submodule import succeed without them.

| Component | Licence | Extra | Project |
|---|---|---|---|
| DeePMD-kit | LGPL-3.0-or-later | `mlip` | <https://github.com/deepmodeling/deepmd-kit> |
| dpdata | LGPL-3.0-or-later | `mlip` | <https://github.com/deepmodeling/dpdata> |
| maml | BSD-3-Clause | `clustering` | <https://github.com/materialsvirtuallab/maml> |
| scikit-learn | BSD-3-Clause | `clustering` | <https://github.com/scikit-learn/scikit-learn> |
| pytest | MIT | `dev` | <https://github.com/pytest-dev/pytest> |
| ruff | MIT | `dev` | <https://github.com/astral-sh/ruff> |
| black | MIT | `dev` | <https://github.com/psf/black> |

MatterGen (MIT, Copyright (c) Microsoft Corporation,
<https://github.com/microsoft/mattergen>) is driven as the `mattergen-generate` console script
rather than imported, and is installed from its own repository. It is not declared as an extra
here, so `pip install` of this package can neither pull it in nor resolve its pinned CUDA build.

Method citations for the optional components:

- MatterGen: Zeni et al., *Nature* (2025), [doi:10.1038/s41586-025-08628-5](https://doi.org/10.1038/s41586-025-08628-5)
- DIRECT sampling (maml): *npj Computational Materials* (2024), [doi:10.1038/s41524-024-01227-4](https://doi.org/10.1038/s41524-024-01227-4)

Machine-learning interatomic potential checkpoints are referenced, never redistributed. Point
the code at your own checkpoint through `$DPA3_MODEL_PATH` or the corresponding argument.

## VASP

VASP is commercial software and requires a licence you obtain yourself. No VASP binaries and no
POTCAR pseudopotential files are included here, and none may be redistributed. The code invokes
your own VASP through `$VASP_CMD` and builds POTCAR files from your own pseudopotential
directory, which pymatgen locates through `$PMG_VASP_PSP_DIR`.

## Materials Project data

Data retrieved from the Materials Project, including the example structures and the
competing-phase table under `examples/data/`, is licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and is redistributed here under that
licence with attribution.

> A. Jain, S. P. Ong, G. Hautier, W. Chen, W. D. Richards, S. Dacek, S. Cholia, D. Gunter,
> D. Skinner, G. Ceder and K. A. Persson, "The Materials Project: A materials genome approach
> to accelerating materials innovation", *APL Materials* **1**(1), 011002 (2013),
> [doi:10.1063/1.4812323](https://doi.org/10.1063/1.4812323)

The bundled files are a frozen snapshot from an earlier harvest: neither the retrieval date nor
the database version current at that time was recorded, and `docs/data/README.md` says so. Treat
the values as demonstration data, not as the Materials Project's current numbers. The changes made
to the redistributed table — the `composition` and `natoms` columns were re-derived with
`pymatgen.core.Composition` — are listed there as CC BY 4.0 requires. ICSD collection codes that
appear in file names and in the `icsd_ids` column are Materials Project cross-references, not ICSD
data.

Access to the Materials Project API requires your own key, read from the `MP_API_KEY`
environment variable.
