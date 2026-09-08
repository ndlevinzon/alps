"""One-job ASCII banner for ALPS ``lig-*`` CLIs.

Companion packages print their own banners when run standalone. ALPS prints
this once and sets the companion banner env flags so spawned workers do not
reprint ffpopt / ligandparam logos.
"""

from __future__ import annotations

import os
import sys
from typing import TextIO

_LOGO = r"""
    _    _     ____  ____
   / \  | |   |  _ \/ ___|
  / _ \ | |   | |_) \___ \
 / ___ \| |___|  __/ ___) |
/_/   \_\_____|_|   |____/
""".strip(
    "\n"
)

_BANNER_PRINTED = False


def package_version() -> str:
    try:
        from alps import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def format_startup_banner(*, version: str | None = None) -> str:
    ver = version if version is not None else package_version()
    return (
        f"{_LOGO}\n"
        f"\n"
        f"  ALPS  v{ver}\n"
        f"  Orchestrates ligandparam, scission, and ffpopt\n"
        f"\n"
    )


def print_startup_banner(
    *,
    stream: TextIO | None = None,
    force: bool = False,
) -> bool:
    """Print the ALPS banner once for this job.

    Also sets ``FFPOPT_BANNER_PRINTED`` and ``LIGANDPARAM_BANNER_PRINTED``
    so companion workers inherit the one-banner-per-job rule.
    """
    from alps.Log import configure_line_buffered_stdio, setup_alps_stdout_logging

    configure_line_buffered_stdio()
    global _BANNER_PRINTED
    if not force:
        if _BANNER_PRINTED or os.environ.get("ALPS_BANNER_PRINTED"):
            setup_alps_stdout_logging()
            return False
    out = stream if stream is not None else sys.__stdout__
    text = format_startup_banner()
    try:
        out.write(text if text.endswith("\n") else text + "\n")
        out.flush()
    except OSError:
        return False
    _BANNER_PRINTED = True
    os.environ["ALPS_BANNER_PRINTED"] = "1"
    os.environ["FFPOPT_BANNER_PRINTED"] = "1"
    os.environ["LIGANDPARAM_BANNER_PRINTED"] = "1"
    setup_alps_stdout_logging()
    return True


def print_progress_hint(*, stream: TextIO | None = None) -> None:
    """Remind operators that long external tools are silent without the board."""
    from alps.Log import configure_line_buffered_stdio

    configure_line_buffered_stdio()
    out = stream if stream is not None else sys.__stdout__
    try:
        out.write(
            "Progress: a live ASCII board is reprinted on this stdout while\n"
            "antechamber / sqm / g16 / xtb / geomeTRIC run. Those tools print\n"
            "little until they finish; a quiet Slurm .out is normal between\n"
            "board refreshes.\n"
            "\n"
        )
        out.flush()
    except OSError:
        return
