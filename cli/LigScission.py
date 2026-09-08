"""Standalone CLI for scission ligand fragmentation.

Supports the full upstream ``scission`` subcommands (``fragment``, ``merge``,
``pick-bond``), plus a ligandparam-friendly shortcut that resolves mol2/lib/frcmod
from a ``lig-getparam`` output directory::

    lig-getparam -i chaps.mol2 -r CHA -d CHA3 -rn freeligand ...
    lig-scission fragment -d CHA3 -r CHA --label chaps

``lig-scission`` expands ``-d`` / ``-r`` / ``--label`` then delegates to
``scission.Cli.main``. The ``scission`` console script uses the same
sibling/installed tree without the ALPS banner.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ligandparam.io.AmberBundle import resolve_getparam_bundle
from scission.Cli import main as scission_main


def _expand_fragment_shortcuts(argv: list[str]) -> list[str]:
    """Rewrite ``fragment -d/-r/--label`` into explicit ``--mol2/--lib/--frcmod``."""
    if not argv or argv[0] != "fragment":
        return argv

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-d", "--data_cwd", type=Path, default=None)
    parser.add_argument("-r", "--resname", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--mol2", type=Path, default=None)
    parser.add_argument("--lib", type=Path, default=None)
    parser.add_argument("--frcmod", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    known, remaining = parser.parse_known_args(argv[1:])

    if known.data_cwd is None or known.resname is None:
        return argv
    if known.mol2 is not None and known.lib is not None and known.frcmod is not None:
        return argv

    bundle = resolve_getparam_bundle(
        cwd=Path.cwd(),
        data_cwd=known.data_cwd,
        resname=known.resname,
        label=known.label,
    )
    outdir = known.outdir or (bundle.work_dir / f"{bundle.stem}.scission_fragments")

    return [
        "fragment",
        "--mol2",
        str(bundle.mol2),
        "--lib",
        str(bundle.lib),
        "--frcmod",
        str(bundle.frcmod),
        "--outdir",
        str(outdir),
        *remaining,
    ]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``lig-scission`` / convenience wrapper."""
    from alps.cli.Banner import print_startup_banner
    from alps.companions import print_status_line

    raw = list(sys.argv[1:] if argv is None else argv)
    if not any(a in ("-h", "--help") for a in raw):
        print_startup_banner()
        print_status_line()
    try:
        expanded = _expand_fragment_shortcuts(raw)
    except FileNotFoundError as exc:
        print(f"lig-scission: {exc}", file=sys.stderr)
        return 2
    return scission_main(expanded)


def scission_console(argv: list[str] | None = None) -> int:
    """``scission`` console script: honor companion env, no ligandparam banner."""
    from scission.Cli import main as raw_scission_main

    raw = list(sys.argv[1:] if argv is None else argv)
    return raw_scission_main(raw)


if __name__ == "__main__":
    raise SystemExit(main())
