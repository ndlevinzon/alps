"""Fan companion logs into one ALPS stdout without stealing their loggers.

Each package keeps its own logger name (``alps``, ``ligandparam``,
``scission``, ``ffpopt``). ALPS attaches a shared stdout handler so a
``lig-getparam`` / ``lig-dihed-correct`` / ``lig-scission`` job has one
process stream. Standalone CLIs that already configured handlers are left
unchanged except for the extra ALPS tee.

ligandparam file logs stay as-is (timestamped). The tee is extra stdout
(message-only, no ``[ligandparam]`` tag). ffpopt still prints ``[twist]`` /
``[wavefront]`` itself; those land on this process stdout when ALPS calls
the API in-process. ``lig-dihed-correct`` also reprints a live ASCII board
(``FRAG_STATUS.txt`` / ``WHOLE_STATUS.txt``), matching the ligandparam
recipe board.
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
_STDIO_CONFIGURED = False


class _CompanionFormatter(logging.Formatter):
    """Keep ligandparam message-only; tag the others so streams stay greppable.

    Callers may still write ``log.info("[alps] ...")``; peel one leading
    ``[name]`` so stdout never shows ``[alps] [alps] ...``.
    """

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if record.name == "ligandparam" or record.name.startswith("ligandparam."):
            return msg
        tag = record.name.split(".", 1)[0]
        prefix = f"[{tag}]"
        rest = msg
        if rest.startswith(prefix + " "):
            rest = rest[len(prefix) + 1 :]
        elif rest.startswith(prefix):
            rest = rest[len(prefix) :].lstrip()
        return f"{prefix} {rest}" if rest else prefix


class _FlushingStreamHandler(logging.StreamHandler):
    """Flush after every record so Slurm ``.out`` files update immediately."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        try:
            self.flush()
        except Exception:
            pass


def configure_line_buffered_stdio() -> None:
    """Force line-buffered stdout/stderr (Slurm redirects are fully buffered)."""
    global _STDIO_CONFIGURED
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if _STDIO_CONFIGURED:
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(line_buffering=True)
        except Exception:
            continue
    _STDIO_CONFIGURED = True


def setup_alps_stdout_logging(*, stream: TextIO | None = None) -> None:
    """Attach one stdout handler to companion loggers (idempotent)."""
    global _ATTACHED, _HANDLER
    configure_line_buffered_stdio()
    if _ATTACHED:
        return
    out = stream if stream is not None else sys.stdout
    handler = _FlushingStreamHandler(out)
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


def silence_wavefront_origin_filters(*, load_wavefront: bool = False) -> None:
    """Drop ffpopt ``ShowOriginFilter`` (``[LOG-ORIGIN]`` on stderr).

    WaveFront attaches that filter at import time. Under ALPS it floods
    Slurm ``.err`` on every log record; ligandparam-style stdout should
    stay status + board only.
    """
    if load_wavefront:
        for mod_name in ("ffpopt.WaveFront", "ffpopt.WaveFrontND"):
            try:
                __import__(mod_name)
            except Exception:
                continue
    for mod_name in ("ffpopt.WaveFront", "ffpopt.WaveFrontND"):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        cls = getattr(mod, "ShowOriginFilter", None)
        if cls is None:
            continue
        current = getattr(cls, "filter", None)
        if current is not None and getattr(current, "_alps_silenced", False):
            continue

        def _quiet(self, record):  # noqa: ARG001
            return True

        _quiet._alps_silenced = True  # type: ignore[attr-defined]
        cls.filter = _quiet  # type: ignore[method-assign]
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.filters = [
            f for f in handler.filters if type(f).__name__ != "ShowOriginFilter"
        ]


def install_ffpopt_stdio() -> None:
    """Match ligandparam's ALPS stdout contract for in-process ffpopt.

    Line-buffered ASCII stdio, shared logger tee, and no WaveFront
    ``[LOG-ORIGIN]`` spam. ffpopt ``print`` lines already hit this process
    stdout; the twist workflows add the live status board.
    """
    setup_alps_stdout_logging()
    try:
        from ligandparam.runtime.Console import ensure_ascii_stdio

        ensure_ascii_stdio()
    except Exception:
        pass
    silence_wavefront_origin_filters()


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
    global _ATTACHED, _HANDLER, _TEE_INSTALLED, _STDIO_CONFIGURED
    for name in _LOGGER_NAMES:
        log = logging.getLogger(name)
        log.handlers = [
            h for h in log.handlers if not getattr(h, "_alps_stdout", False)
        ]
    _ATTACHED = False
    _HANDLER = None
    _TEE_INSTALLED = False
    _STDIO_CONFIGURED = False
    os.environ.pop("ALPS_STDOUT_LOGGING", None)
