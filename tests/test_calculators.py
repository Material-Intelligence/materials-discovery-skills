"""Loading a calculator, and which model branch the energies then belong to.

The DeePMD backend is an optional extra and is not installed in the offline test environment,
so these tests put a stand-in for :mod:`deepmd.calculator` in ``sys.modules``. That is enough
to pin the policy that matters scientifically: a head that the checkpoint rejects is an error,
not a quiet reload of a different fitting network, and whichever head was loaded is recorded
on the calculator.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Callable

import pytest

from matdisc.common.calculators import DEFAULT_DP_HEAD, calculator_head, load_calculator


class FakeDP:
    """A stand-in for :class:`deepmd.calculator.DP` that records how it was built."""

    def __init__(self, model_path: str, head: str | None = None, **kwargs: Any) -> None:
        """Record the arguments the loader passed.

        Args:
            model_path: The checkpoint path.
            head: The head that was asked for, if any.
            **kwargs: Anything else the loader forwarded.
        """
        self.model_path = model_path
        self.head = head
        self.kwargs = kwargs


def install_fake_deepmd(monkeypatch: pytest.MonkeyPatch, factory: Callable[..., Any]) -> None:
    """Put a fake ``deepmd.calculator`` module in ``sys.modules``.

    Args:
        monkeypatch: pytest's monkeypatch fixture.
        factory: What ``DP`` should be.
    """
    package = types.ModuleType("deepmd")
    module = types.ModuleType("deepmd.calculator")
    module.DP = factory  # type: ignore[attr-defined]
    package.calculator = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepmd", package)
    monkeypatch.setitem(sys.modules, "deepmd.calculator", module)


class TestHeadSelection:
    """Which branch of a multi-task checkpoint the energies come from."""

    def test_the_requested_head_is_loaded_and_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_deepmd(monkeypatch, FakeDP)

        calculator = load_calculator("dp", model_path="/checkpoint.pt", head="Omat24")

        assert calculator.head == "Omat24"
        assert calculator_head(calculator) == "Omat24"

    def test_a_rejected_head_is_not_retried_without_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Another head is another fitting network, so its energies are not a fallback."""
        attempts: list[str | None] = []

        def explode(model_path: str, head: str | None = None, **kwargs: Any) -> FakeDP:
            attempts.append(head)
            if head is not None:
                raise KeyError(head)
            return FakeDP(model_path, head=head, **kwargs)

        install_fake_deepmd(monkeypatch, explode)

        with pytest.raises(KeyError):
            load_calculator("dp", model_path="/checkpoint.pt", head="Omat25")
        assert attempts == ["Omat25"]

    def test_a_single_task_checkpoint_is_loaded_without_a_head(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def single_task(model_path: str, head: str | None = None, **kwargs: Any) -> FakeDP:
            if head is not None:
                raise AssertionError("This is a single-task model and does not take a head.")
            return FakeDP(model_path, head=None, **kwargs)

        install_fake_deepmd(monkeypatch, single_task)

        calculator = load_calculator("dp", model_path="/checkpoint.pt", head="Omat24")

        assert calculator.head is None
        assert calculator_head(calculator) is None

    def test_a_multi_task_checkpoint_without_a_head_takes_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def multi_task(model_path: str, head: str | None = None, **kwargs: Any) -> FakeDP:
            if head is None:
                raise AssertionError("Head must be set for a multi-task model!")
            return FakeDP(model_path, head=head, **kwargs)

        install_fake_deepmd(monkeypatch, multi_task)

        calculator = load_calculator("dp", model_path="/checkpoint.pt")

        assert calculator.head == DEFAULT_DP_HEAD
        assert calculator_head(calculator) == DEFAULT_DP_HEAD

    def test_a_checkpoint_path_is_required(self) -> None:
        with pytest.raises(ValueError, match="checkpoint"):
            load_calculator("dp")


class TestEmt:
    """The offline calculator."""

    def test_emt_has_no_head(self) -> None:
        assert calculator_head(load_calculator("emt")) is None

    def test_an_unknown_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown calculator kind"):
            load_calculator("dpa4")
