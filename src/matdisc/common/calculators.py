"""Lazy loading of ASE calculators for machine-learning interatomic potentials.

The backends are optional extras: DeePMD-kit is imported inside :func:`load_calculator`, so
importing this module -- or any module that uses it -- works in an environment without a
machine-learning potential installed. The import cost is paid only when a calculator is
actually requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from matdisc.common.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from ase.calculators.calculator import Calculator

__all__ = ["load_calculator", "calculator_head", "DEFAULT_DP_HEAD", "HEAD_ATTRIBUTE", "SUPPORTED_KINDS"]

logger = get_logger(__name__)

DEFAULT_DP_HEAD = "Omat24"
"""Head used when a multi-task DeePMD model is loaded without one being named."""

SUPPORTED_KINDS = ("dp", "emt")
"""Calculator kinds :func:`load_calculator` understands. ``dpa3`` and ``deepmd`` alias ``dp``."""

HEAD_ATTRIBUTE = "matdisc_head"
"""Attribute :func:`load_calculator` sets on the calculator it returns; read it with :func:`calculator_head`.

The head of a multi-task checkpoint selects a fitting network, so two heads of one checkpoint
report energies on two different scales. The head that was actually loaded therefore belongs
with every number the calculator produces, and is recorded on the calculator object so that a
stage can carry it into its result table.
"""

_DP_ALIASES = {"dp", "deepmd", "dpa3", "dpa-3"}

_MISSING_DEEPMD_MESSAGE = (
    "deepmd-kit is required for the 'dp' calculator. Install the mlip extra "
    "('pip install materials-discovery-skills[mlip]') or deepmd-kit directly."
)


def load_calculator(
    kind: str, model_path: str | None = None, head: str | None = None, **backend_kwargs: Any
) -> Calculator:
    """Load an ASE calculator.

    Args:
        kind: Which backend to load. ``"dp"`` (also ``"deepmd"``, ``"dpa3"``) loads a
            DeePMD-kit model such as DPA-3 and needs ``model_path``. ``"emt"`` loads ASE's
            built-in effective-medium potential, which needs no model file; EMT is
            parameterised for a handful of metals and is meant for smoke tests, not for
            screening.
        model_path: Path to the model checkpoint, required by ``"dp"``.
        head: Head to select in a multi-task DeePMD model. When a multi-task model is loaded
            without one, :data:`DEFAULT_DP_HEAD` is used and a warning is logged.
        **backend_kwargs: Forwarded to the backend calculator's constructor.

    Returns:
        An ASE calculator ready to attach to an ``Atoms`` object.

    Raises:
        ValueError: If ``kind`` is unknown, or ``model_path`` is missing for a backend that
            needs one.
        ImportError: If the backend for ``kind`` is not installed.
    """
    key = kind.strip().lower()
    if key in _DP_ALIASES:
        if not model_path:
            raise ValueError("A model checkpoint path is required for the 'dp' calculator.")
        return _load_deepmd(model_path, head=head, **backend_kwargs)
    if key == "emt":
        return _tag_head(_load_emt(**backend_kwargs), None)
    raise ValueError(f"Unknown calculator kind {kind!r}. Supported kinds: {', '.join(SUPPORTED_KINDS)}.")


def calculator_head(calculator: Any) -> str | None:
    """Return the head a calculator was loaded with.

    Args:
        calculator: A calculator returned by :func:`load_calculator`.

    Returns:
        The head name, or ``None`` for a single-task model, a calculator that has no heads at
        all (EMT), or one that was not built here.
    """
    return getattr(calculator, HEAD_ATTRIBUTE, None)


def _tag_head(calculator: Any, head: str | None) -> Any:
    """Record on a calculator which head it was loaded with.

    Args:
        calculator: The calculator to tag.
        head: The head that was loaded, or ``None`` when there is none.

    Returns:
        The same calculator.
    """
    try:
        setattr(calculator, HEAD_ATTRIBUTE, head)
    except AttributeError:  # pragma: no cover - a backend with __slots__ and no such slot
        logger.warning("Could not record the head %r on %r; it will not appear in the results.", head, calculator)
    return calculator


_SINGLE_TASK_MARKERS = ("single-task", "single task", "not a multi-task", "not multi-task")
"""Phrases that identify the one failure meaning "this checkpoint has no heads to choose from"."""


def _is_single_task_failure(exc: BaseException) -> bool:
    """Say whether an exception means "this checkpoint has no heads".

    A head that is absent from the checkpoint, a mistyped head, a corrupt file or a device
    error must not be mistaken for this: loading a different branch of a multi-task model
    silently changes the energy scale, so everything else is re-raised.

    Args:
        exc: The exception :class:`deepmd.calculator.DP` raised.

    Returns:
        ``True`` only for the assertion deepmd-kit raises when a single-task model is handed
        a head.
    """
    if not isinstance(exc, AssertionError):
        return False
    message = str(exc).lower()
    return any(marker in message for marker in _SINGLE_TASK_MARKERS)


def _load_deepmd(model_path: str, head: str | None = None, **backend_kwargs: Any) -> Calculator:
    """Load a DeePMD-kit model as an ASE calculator.

    The head is not negotiable: when one is asked for and the checkpoint rejects it, the
    failure is raised rather than retried without a head, because the heads of a multi-task
    checkpoint are different fitting networks and their energies are not comparable. The one
    exception is the assertion deepmd-kit raises for a single-task model, which has no heads
    to choose between.

    Args:
        model_path: Path to the model checkpoint.
        head: Head to select in a multi-task model.
        **backend_kwargs: Forwarded to :class:`deepmd.calculator.DP`.

    Returns:
        The DeePMD calculator, carrying the loaded head under :data:`HEAD_ATTRIBUTE`.

    Raises:
        ImportError: If deepmd-kit is not installed.
        Exception: Whatever the backend raised, unless it means the model is single-task.
    """
    try:
        from deepmd.calculator import DP
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(_MISSING_DEEPMD_MESSAGE) from exc

    if head:
        try:
            return _tag_head(DP(model_path, head=head, **backend_kwargs), head)
        except Exception as exc:
            if not _is_single_task_failure(exc):
                logger.error(
                    "Loading %s with head=%r failed (%s). It is not retried without a head: another head of a "
                    "multi-task checkpoint is a different fitting network and reports energies on a different "
                    "scale. Name the head the checkpoint carries, or omit it for a single-task checkpoint.",
                    model_path,
                    head,
                    exc,
                )
                raise
            logger.warning("%s is a single-task model; the requested head %r does not apply.", model_path, head)
            return _tag_head(DP(model_path, **backend_kwargs), None)

    try:
        return _tag_head(DP(model_path, **backend_kwargs), None)
    except AssertionError as exc:
        if "Head must be set" not in str(exc):
            raise
        logger.warning("%s is a multi-task model; loading it with head=%r.", model_path, DEFAULT_DP_HEAD)
        return _tag_head(DP(model_path, head=DEFAULT_DP_HEAD, **backend_kwargs), DEFAULT_DP_HEAD)


def _load_emt(**backend_kwargs: Any) -> Calculator:
    """Load ASE's effective-medium-theory calculator.

    Args:
        **backend_kwargs: Forwarded to :class:`ase.calculators.emt.EMT`.

    Returns:
        The EMT calculator.

    Raises:
        ImportError: If ASE is not installed.
    """
    try:
        from ase.calculators.emt import EMT
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("ASE is required for the 'emt' calculator. Install it with 'pip install ase'.") from exc
    return EMT(**backend_kwargs)
