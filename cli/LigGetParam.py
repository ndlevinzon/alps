"""ALPS wrapper around ligandparam ``lig-getparam``.

Prints the ALPS banner, tees ligandparam file logs onto this process
stdout (ligandparam still writes ``{resname}.log``), then calls the
ligandparam CLI. Same flags as standalone ``lig-getparam``.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ALPS ``lig-getparam``."""
    from alps.cli.Banner import print_startup_banner
    from alps.companions import print_status_line
    from alps.Log import install_ligandparam_stdout_tee
    from ligandparam.cli.LigGetParam import main as ligandparam_main

    print_startup_banner()
    print_status_line()
    install_ligandparam_stdout_tee()
    raw = list(sys.argv[1:] if argv is None else argv)
    saved = sys.argv
    sys.argv = [saved[0], *raw]
    try:
        ligandparam_main()
    finally:
        sys.argv = saved
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
