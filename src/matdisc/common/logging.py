"""Logging setup for the package.

Library modules take their logger from :func:`get_logger` and never print. Applications --
the ``matdisc`` command line, a script, a notebook -- call :func:`configure_logging` once to
attach a handler to the package logger; until they do, the package emits nothing.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

PACKAGE_LOGGER_NAME = "matdisc"
"""Name of the logger every module in this package logs through."""

DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
_package_logger.addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the logger a module should log through.

    Args:
        name: Usually ``__name__``. A name that does not already sit inside the package is
            attached below the package logger, so that one :func:`configure_logging` call
            controls every message this package emits. ``None`` returns the package logger.

    Returns:
        A :class:`logging.Logger` below ``matdisc``.
    """
    if not name or name == PACKAGE_LOGGER_NAME:
        return _package_logger
    if name.startswith(f"{PACKAGE_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name}")


def configure_logging(
    level: int | str = logging.INFO,
    stream: TextIO | None = None,
    fmt: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATE_FORMAT,
) -> logging.Logger:
    """Attach a stream handler to the package logger.

    Calling this more than once replaces the handler installed by the previous call rather
    than adding a second one, so repeated calls do not duplicate messages.

    Args:
        level: Logging level, either a :mod:`logging` constant or a level name such as
            ``"DEBUG"``.
        stream: Destination stream. Defaults to :data:`sys.stderr`.
        fmt: Format string for the handler.
        datefmt: Date format string for the handler.

    Returns:
        The package logger, now carrying the handler.
    """
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    handler.set_name(f"{PACKAGE_LOGGER_NAME}-stream")

    for existing in list(_package_logger.handlers):
        if existing.get_name() == handler.get_name():
            _package_logger.removeHandler(existing)

    _package_logger.addHandler(handler)
    _package_logger.setLevel(level)
    return _package_logger
