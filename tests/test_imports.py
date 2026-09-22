"""Every module imports without the optional backends.

MatterGen, DeePMD-kit, dpdata and maml are heavy, and some of them need a GPU. They are
optional extras, and the code that uses them imports them inside the functions that need
them. This file is the check that keeps that true: it imports every module of the package in
an environment where none of those extras is installed, which is also what the ``--help`` of
every subcommand depends on.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

import pytest

import matdisc

OPTIONAL_BACKENDS = ("deepmd", "dpdata", "mattergen", "maml", "phonopy", "torch")
"""Packages that must not be needed to import ``matdisc``."""


def _module_names() -> list[str]:
    """List every importable module of the package.

    Returns:
        Fully qualified module names, sorted, including the package itself.
    """
    names = ["matdisc"]
    for info in pkgutil.walk_packages(matdisc.__path__, prefix="matdisc."):
        names.append(info.name)
    return sorted(names)


MODULE_NAMES = _module_names()


def test_the_package_has_a_version() -> None:
    assert isinstance(matdisc.__version__, str)
    assert matdisc.__version__


def test_every_module_was_found() -> None:
    """A rename that loses a module should fail here rather than silently shrink the sweep."""
    expected = {
        "matdisc",
        "matdisc.cli",
        "matdisc.clustering",
        "matdisc.clustering.descriptors",
        "matdisc.clustering.sampling",
        "matdisc.common",
        "matdisc.common.calculators",
        "matdisc.common.composition",
        "matdisc.common.io",
        "matdisc.common.logging",
        "matdisc.common.mp",
        "matdisc.competing",
        "matdisc.competing.download",
        "matdisc.competing.search",
        "matdisc.dft",
        "matdisc.dft.inputs",
        "matdisc.dft.outputs",
        "matdisc.dft.slurm",
        "matdisc.finetune",
        "matdisc.finetune.dataset",
        "matdisc.finetune.train",
        "matdisc.generation",
        "matdisc.generation.mattergen",
        "matdisc.pipeline",
        "matdisc.pipeline.config",
        "matdisc.pipeline.data_interface",
        "matdisc.pipeline.run",
        "matdisc.pipeline.stages",
        "matdisc.screening",
        "matdisc.screening.hull",
        "matdisc.screening.phonons",
        "matdisc.screening.relax",
    }
    assert expected <= set(MODULE_NAMES)


@pytest.mark.parametrize("name", MODULE_NAMES)
def test_module_imports(name: str) -> None:
    assert importlib.import_module(name) is not None


@pytest.mark.parametrize("backend", OPTIONAL_BACKENDS)
def test_importing_the_package_does_not_pull_in_an_optional_backend(backend: str) -> None:
    """Importing every module must not have imported an extra as a side effect."""
    for name in MODULE_NAMES:
        importlib.import_module(name)
    assert backend not in sys.modules, f"{backend} was imported at module level somewhere in matdisc"


def test_the_console_script_entry_point_is_importable() -> None:
    from matdisc.cli import main

    assert callable(main)


class TestOptionalBackends:
    """The extras, exercised only where they are installed.

    These are the tests ``pytest -m "not network and not gpu and not extras"`` leaves out.
    Each one also skips when its backend is absent, so a developer who installed an extra but
    ran the default selection does not get a surprise failure.
    """

    @pytest.mark.extras
    @pytest.mark.gpu
    def test_the_deepmd_calculator_loads(self) -> None:
        import os

        pytest.importorskip("deepmd", reason="install the 'mlip' extra to run this")
        model_path = os.environ.get("DPA3_MODEL_PATH")
        if not model_path:
            pytest.skip("set DPA3_MODEL_PATH to a DeePMD-kit checkpoint to run this")

        from matdisc.common.calculators import load_calculator

        assert load_calculator("dp", model_path=model_path) is not None

    @pytest.mark.extras
    def test_dpdata_is_reachable_from_the_dataset_module(self) -> None:
        pytest.importorskip("dpdata", reason="install the 'mlip' extra to run this")

        from matdisc.finetune.dataset import _require_dpdata

        assert _require_dpdata() is not None
