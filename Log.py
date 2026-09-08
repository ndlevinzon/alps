"""Fan companion logs into one ALPS stdout without stealing their loggers.

Each package keeps its own logger name (``alps``, ``ligandparam``,
``scission``, ``ffpopt``). ALPS attaches a shared stdout handler so a
``lig-getparam`` / ``lig-dihed-correct`` / ``lig-scission`` job has one
process stream. Standalone CLIs that already configured handlers are left
unchanged except for the extra ALPS tee.

ligandparam file logs stay as-is (timestamped). The tee is extra stdout.
ffpopt still prints ``[twist]`` / ``[frag-twist]`` lines itself; those
already land on this process stdout when ALPS calls the API in-process.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TextIO

_ATTACHED = False
_HANDLER: logging.Handler | None = None
_LOGGER_NAMES = ("alps", "ligandparam", "scission", "ffpopt")
_TEE_INSTALLED = False


class _CompanionFormatter(logging.Formatter):
    """Keep ligandparam message-only; tag the others so streams stay greppable."""

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if record.name == "ligandparam" or record.name.startswith("ligandparam."):
            return msg
        return f"[{record.name}] {msg}"


def setup_alps_stdout_logging(*, stream: TextIO | None = None) -> None:
    """Attach one stdout handler to companion loggers (idempotent)."""
    global _ATTACHED, _HANDLER
    if _ATTACHED:
        return
    out = stream if stream is not None else sys.stdout
    handler = logging.StreamHandler(out)
    handler.setLevel(logging.INFO)
    handler.setFormatter(_CompanionFormatter())
    handler._alps_stdout = True  # type: ignore[attr-defined]
    _HANDLER = handler
    _ATTACHED = True
    os.environ["ALPS_STDOUT_LOGGING"] = "1"
    for name in _LOGGER_NAMES:
        attach_logger(logging.getLogger(name), handler=handler)


def attach_logger(
    logger: logging.Logger,
    *,
    handler: logging.Handler | None = None,
) -> logging.Logger:
    """Tee ``logger`` onto the shared ALPS stdout handler."""
    if not _ATTACHED:
        setup_alps_stdout_logging()
    h = handler if handler is not None else _HANDLER
    if h is None:
        return logger
    if not any(getattr(existing, "_alps_stdout", False) for existing in logger.handlers):
        logger.addHandler(h)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def get_logger(name: str = "alps") -> logging.Logger:
    """Return an ALPS (or companion) logger after stdout logging is on."""
    setup_alps_stdout_logging()
    return logging.getLogger(name)


def install_ligandparam_stdout_tee() -> None:
    """After ligandparam opens a file logger, also tee that logger to stdout.

    ``lig-getparam`` logs under the residue name, not ``ligandparam``, so
    attaching only the package logger would miss the recipe stream.
    """
    global _TEE_INSTALLED
    setup_alps_stdout_logging()
    if _TEE_INSTALLED:
        return
    import ligandparam.Log as lp_log

    orig = lp_log.set_file_logger

    def _set_file_logger(*args, **kwargs):
        logger = orig(*args, **kwargs)
        attach_logger(logger)
        return logger

    lp_log.set_file_logger = _set_file_logger  # type: ignore[method-assign]
    _TEE_INSTALLED = True


def reset_for_tests() -> None:
    """Drop ALPS stdout handlers (unit tests only)."""
    global _ATTACHED, _HANDLER, _TEE_INSTALLED
    for name in _LOGGER_NAMES:
        log = logging.getLogger(name)
        log.handlers = [
            h for h in log.handlers if not getattr(h, "_alps_stdout", False)
        ]
    _ATTACHED = False
    _HANDLER = None
    _TEE_INSTALLED = False
    os.environ.pop("ALPS_STDOUT_LOGGING", None)
