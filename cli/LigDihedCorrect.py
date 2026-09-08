"""Standalone CLI for ffpopt dihedral corrections after ligandparam.

Typical same-session workflow::

    lig-getparam -i chaps.mol2 -r CHA -d CHA3 -rn freeligand ...
    lig-dihed-correct -d CHA3 -r CHA --label chaps --model xtb -n 44
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ligandparam.io.AmberBundle import AmberLigandBundle, resolve_getparam_bundle
from ligandparam.Log import get_logger, set_file_logger
from alps.stages.FfpoptDihed import StageDihedTwistCorrection


def _build_fit_cli_args(args) -> list[str]:
    out: list[str] = []
    if getattr(args, "fit_full", False):
        out.append("--fit-full")
    if getattr(args, "barrier_only", False):
        out.append("--barrier-only")
    mode = getattr(args, "fit_mode", None)
    if mode:
        out.extend(["--fit-mode", str(mode)])
    backend = getattr(args, "fit_backend", None)
    if backend:
        out.extend(["--fit-backend", str(backend)])
    if getattr(args, "fit_phases", False):
        out.append("--fit-phases")
    if getattr(args, "fit_periods", False):
        out.append("--fit-periods")
    if getattr(args, "fit_scee_scnb", False):
        out.append("--fit-scee-scnb")
    return out


def run_dihed_correct(
    *,
    bundle: AmberLigandBundle | None = None,
    mol2: Path | None = None,
    lib: Path | None = None,
    frcmod: Path | None = None,
    work_dir: Path | None = None,
    out_frcmod: Path | None = None,
    out_dir: Path | None = None,
    model: str = "qdpi2",
    maxiter: int = 2,
    nprim: int = 3,
    delta: int = 10,
    nproc: int = 1,
    geometric_opt: bool = True,
    skip_existing: bool = True,
    dry_run: bool = False,
    fast_wavefront: bool | None = None,
    whole_ligand: bool = False,
    multi_centroid: int = 0,
    boltzmann_charges: bool = False,
    soft_dihed_restraint: bool = False,
    soft_dihed_k: float | None = None,
    soft_dihed_kmax: float | None = None,
    soft_dihed_tol: float | None = None,
    fit_cli_args: list[str] | None = None,
    logger=None,
):
    """Execute :class:`StageDihedTwistCorrection` on an Amber ligand bundle."""
    if logger is None:
        logger = get_logger()
    if bundle is None:
        if mol2 is None or lib is None or frcmod is None or work_dir is None:
            raise ValueError(
                "Provide bundle= or all of mol2, lib, frcmod, and work_dir"
            )
        bundle = AmberLigandBundle(
            mol2=Path(mol2),
            lib=Path(lib),
            frcmod=Path(frcmod),
            work_dir=Path(work_dir),
        )
    out_frcmod = out_frcmod or bundle.work_dir / f"{bundle.stem}.dihed.frcmod"
    default_out = (
        f"{bundle.stem}.dihed_whole"
        if whole_ligand
        else f"{bundle.stem}.dihed_fragments"
    )
    out_dir = out_dir or bundle.work_dir / default_out

    stage = StageDihedTwistCorrection(
        "DihedTwist",
        main_input=bundle.mol2,
        cwd=bundle.work_dir,
        in_lib=bundle.lib,
        in_frcmod=bundle.frcmod,
        out_frcmod=out_frcmod,
        out_dir=out_dir,
        model=model,
        maxiter=maxiter,
        nprim=nprim,
        delta=delta,
        nproc=nproc,
        geometric_opt=geometric_opt,
        skip_existing=skip_existing,
        fast_wavefront=fast_wavefront,
        whole_ligand=whole_ligand,
        multi_centroid=multi_centroid,
        boltzmann_charges=boltzmann_charges,
        soft_dihed_restraint=soft_dihed_restraint,
        soft_dihed_k=soft_dihed_k,
        soft_dihed_kmax=soft_dihed_kmax,
        soft_dihed_tol=soft_dihed_tol,
        fit_cli_args=fit_cli_args or [],
        logger=logger,
    )
    return stage.execute(dry_run=dry_run, nproc=nproc)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``lig-dihed-correct``."""
    from alps.cli.Banner import print_startup_banner
    from alps.companions import print_status_line

    parser = argparse.ArgumentParser(
        description=(
            "Fit dihedral corrections with ffpopt after ligandparam "
            "(default: fragmented twist -> merged frcmod; lib unchanged). "
            "Optional --whole-ligand skips scission."
        )
    )
    parser.add_argument(
        "-d",
        "--data_cwd",
        type=Path,
        default=None,
        help="Same --data_cwd used with lig-getparam (directory under CWD)",
    )
    parser.add_argument(
        "-r",
        "--resname",
        type=str,
        default=None,
        help="Same --resname used with lig-getparam (subdir under data_cwd)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Recipe file stem (default: resname, or auto-detect unique *.mol2)",
    )
    parser.add_argument("--mol2", type=Path, default=None, help="Parent ligand mol2")
    parser.add_argument("--lib", type=Path, default=None, help="Parent Amber lib")
    parser.add_argument("--frcmod", type=Path, default=None, help="Parent frcmod")
    parser.add_argument(
        "-o",
        "--out-frcmod",
        type=Path,
        default=None,
        help="Merged output frcmod (default: {stem}.dihed.frcmod)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Fragment / scan working directory (default: {stem}.dihed_fragments)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qdpi2",
        help="High-level ffpopt model (default: qdpi2). Light options: xtb, aimnet2.",
    )
    parser.add_argument("--maxiter", type=int, default=2, help="Fit iterations (default: 2)")
    parser.add_argument("--nprim", type=int, default=3, help="Cosine primitives (default: 3)")
    parser.add_argument(
        "--delta",
        type=int,
        default=10,
        help="Wavefront dihedral step in degrees (default: 10; try 5 if geomeTRIC is unstable)",
    )
    parser.add_argument("-n", "--nproc", type=int, default=1, help="Wavefront parallelism")
    parser.add_argument(
        "--no-geometric-opt",
        action="store_true",
        help=(
            "Use ASE BFGS instead of geomeTRIC for constrained scans "
            "(only if you intentionally want to skip geomeTRIC)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Do not reuse existing fragment/scan artifacts",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Accepted for compatibility with the old monorepo CLI. "
            "This ffpopt checkout has no GAU_LOOSE / optimizer-ladder presets; ignored."
        ),
    )
    parser.add_argument(
        "--whole-ligand",
        action="store_true",
        help="Skip scission fragmentation; twist the full parent ligand",
    )
    parser.add_argument(
        "--multi-centroid",
        type=int,
        default=0,
        help=(
            "ConfSearch centroids for HL scans; pick smoothest profile per "
            "torsion (Fourier+roughness). Default 0 (off)."
        ),
    )
    parser.add_argument(
        "--boltzmann-charges",
        action="store_true",
        help="Boltzmann-average charges over centroid mol2s (whole-ligand)",
    )
    parser.add_argument(
        "--soft-dihed-restraint",
        action="store_true",
        help=(
            "Soft harmonic dihedral restraint (AFFDO-style 500 kcal/mol/rad^2, "
            "+/-0.5 deg) with geomeTRIC instead of hard IC constraints"
        ),
    )
    parser.add_argument(
        "--soft-dihed-k",
        type=float,
        default=None,
        help="Soft dihedral k in kcal/mol/rad^2 (default 500)",
    )
    parser.add_argument(
        "--soft-dihed-kmax",
        type=float,
        default=None,
        help=(
            "Cap for k-doubling when the soft dihedral is out of band "
            "(kcal/mol/rad^2, default 8000). Then one hard-IC opt from last coords."
        ),
    )
    parser.add_argument(
        "--soft-dihed-tol",
        type=float,
        default=None,
        help="Soft dihedral tolerance in degrees (default 0.5)",
    )
    parser.add_argument(
        "--fit-mode",
        choices=("barrier", "torsion", "full"),
        default=None,
        help="GenDihedFit mode (default barrier / FC-only)",
    )
    parser.add_argument(
        "--fit-backend",
        choices=("lsq", "lbfgsb", "jax"),
        default=None,
        help="GenDihedFit solver (jax: pip install -e '.[jax]' from the clone, not PyPI)",
    )
    parser.add_argument("--fit-full", action="store_true", help="Fit FC+phase+period+scee/scnb")
    parser.add_argument("--barrier-only", action="store_true", help="Force FC-only fit")
    parser.add_argument("--fit-phases", action="store_true")
    parser.add_argument("--fit-periods", action="store_true")
    parser.add_argument("--fit-scee-scnb", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Log planned work only")
    parser.add_argument(
        "--logger",
        choices=("stream", "file"),
        default="stream",
        help="Logging destination (default: stream)",
    )

    args = parser.parse_args(argv)

    print_startup_banner()
    print_status_line()
    from alps.Log import attach_logger, get_logger as alps_get_logger

    try:
        bundle = resolve_getparam_bundle(
            cwd=Path.cwd(),
            data_cwd=args.data_cwd,
            resname=args.resname,
            label=args.label,
            mol2=args.mol2,
            lib=args.lib,
            frcmod=args.frcmod,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    if args.logger == "file":
        logger = attach_logger(
            set_file_logger(bundle.work_dir / f"{bundle.stem}.dihed.log")
        )
    else:
        logger = alps_get_logger("ligandparam")

    logger.info(
        "lig-dihed-correct: mol2=%s lib=%s frcmod=%s",
        bundle.mol2,
        bundle.lib,
        bundle.frcmod,
    )
    if args.fast:
        logger.warning(
            "[alps] --fast is ignored: this ffpopt checkout has no GAU_LOOSE presets"
        )
    if args.whole_ligand and (
        args.multi_centroid
        or args.boltzmann_charges
        or args.soft_dihed_restraint
        or _build_fit_cli_args(args)
    ):
        logger.warning(
            "[alps] AFFDO extras are not in this ffpopt checkout; running a plain parent twist"
        )
    result = run_dihed_correct(
        bundle=bundle,
        out_frcmod=args.out_frcmod,
        out_dir=args.out_dir,
        model=args.model,
        maxiter=args.maxiter,
        nprim=args.nprim,
        delta=args.delta,
        nproc=args.nproc,
        geometric_opt=not args.no_geometric_opt,
        skip_existing=not args.force,
        dry_run=args.dry_run,
        fast_wavefront=True if args.fast else None,
        whole_ligand=args.whole_ligand,
        multi_centroid=args.multi_centroid,
        boltzmann_charges=args.boltzmann_charges,
        soft_dihed_restraint=args.soft_dihed_restraint,
        soft_dihed_k=args.soft_dihed_k,
        soft_dihed_kmax=args.soft_dihed_kmax,
        soft_dihed_tol=args.soft_dihed_tol,
        fit_cli_args=_build_fit_cli_args(args),
        logger=logger,
    )
    if result is not None:
        key = "out_frcmod" if args.whole_ligand else "merged_frcmod"
        logger.info(
            "Done. %s=%s",
            key,
            result.get(key) or result.get("merged_frcmod") or result.get("out_frcmod"),
        )
        from ligandparam.Log import dihed_correct_ok, log_success_quote

        if dihed_correct_ok(result, dry_run=args.dry_run):
            log_success_quote(logger)
    return 0


if __name__ == "__main__":
    # Required for ffpopt wavefront spawn-mode multiprocessing.
    raise SystemExit(main())
