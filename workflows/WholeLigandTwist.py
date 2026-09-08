"""Whole-ligand dihedral twist: scission bonds + ffpopt API.

No AFFDO extras (those lived in the old split ``src/ffpopt`` tree).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Union

from alps.Log import get_logger
from alps.ffpopt_bridge import prepare_start_json, pushd, york_standard_kwargs

PathLike = Union[str, Path]


def _tleap_parent_parm_rst(
    mol2: Path, frcmod: Path, out_dir: Path, log: logging.Logger
) -> tuple[Path, Path]:
    """Build parent ``parm7`` / ``rst7`` with tleap (same idea as scission fragments)."""
    tleap = shutil.which("tleap")
    if tleap is None:
        raise RuntimeError("tleap not on PATH; install AmberTools for whole-ligand twist")
    from scission.Writers import _discover_leaprc_path

    mol2_copy = out_dir / "parent.mol2"
    frcmod_copy = out_dir / "parent.frcmod"
    shutil.copyfile(mol2, mol2_copy)
    shutil.copyfile(frcmod, frcmod_copy)
    leaprc = _discover_leaprc_path(tleap)
    lines = [
        f"source {leaprc}" if leaprc else "source leaprc.gaff2",
        "loadamberparams parent.frcmod",
        "lig = loadmol2 parent.mol2",
        "check lig",
        "saveamberparm lig parent.parm7 parent.rst7",
        "quit",
    ]
    leap_in = out_dir / "tleap.in"
    leap_in.write_text("\n".join(lines) + "\n")
    log.info("[alps] tleap parent parm7/rst7 in %s", out_dir)
    proc = subprocess.run(
        [tleap, "-f", leap_in.name], cwd=out_dir, text=True, capture_output=True
    )
    (out_dir / "tleap.stdout.log").write_text(proc.stdout)
    (out_dir / "tleap.stderr.log").write_text(proc.stderr)
    parm7 = out_dir / "parent.parm7"
    rst7 = out_dir / "parent.rst7"
    if proc.returncode != 0 or not parm7.is_file() or not rst7.is_file():
        raise RuntimeError(
            f"tleap failed for whole-ligand parent; see {out_dir / 'tleap.stdout.log'}"
        )
    return parm7, rst7


def run_whole_ligand_dihed_twist_workflow(
    *,
    mol2: PathLike,
    lib: PathLike,
    frcmod: PathLike,
    out_dir: PathLike = "whole_ligand_twist",
    out_frcmod: PathLike | None = None,
    rotatable_bond_smarts=None,
    delta: int = 10,
    nprim: int = 3,
    maxiter: int = 2,
    nlmaxiter: int = 300,
    nproc: int = 1,
    multi_centroid: int = 0,
    boltzmann_charges: bool = False,
    fit_cli_args: list | None = None,
    skip_existing: bool = True,
    logger: logging.Logger | None = None,
    fast_wavefront: bool | None = None,
    plot_comparisons: bool = True,
    **standard_kwargs,
) -> dict:
    """Discover rotatable bonds with scission, twist the parent with ffpopt."""
    from ffpopt.CpuThreads import pin_math_threads
    from ffpopt.Workflows import run_dihed_twist_workflow
    from scission.LigandIo import load_ligand_from_mol2
    from scission.Torsions import find_rotatable_bonds

    log = logger if logger is not None else get_logger("alps")
    if fast_wavefront:
        log.warning(
            "[alps] --fast is ignored: this ffpopt checkout has no GAU_LOOSE presets"
        )
    if multi_centroid or boltzmann_charges or fit_cli_args:
        log.warning(
            "[alps] AFFDO extras are not in this ffpopt checkout; running a plain parent twist"
        )
    for key in (
        "soft_dihed_restraint",
        "soft_dihed_k",
        "soft_dihed_kmax",
        "soft_dihed_tol",
    ):
        if standard_kwargs.pop(key, None):
            log.warning(
                "[alps] soft-dihed restraint flags are not in this ffpopt checkout; ignoring"
            )

    pin_math_threads(1)
    out_dir_path = Path(out_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)
    mol2_p = Path(mol2).resolve()
    Path(lib).resolve()
    frcmod_p = Path(frcmod).resolve()
    out_frcmod_path = (
        Path(out_frcmod).resolve()
        if out_frcmod is not None
        else out_dir_path / f"{mol2_p.stem}.dihed.frcmod"
    )

    extra_smarts = ()
    if rotatable_bond_smarts is not None:
        extra_smarts = (
            (rotatable_bond_smarts,)
            if isinstance(rotatable_bond_smarts, str)
            else tuple(rotatable_bond_smarts)
        )
    ligand = load_ligand_from_mol2(mol2_p)
    bonds1 = find_rotatable_bonds(ligand, rotatable_bond_smarts=extra_smarts)
    bonds0 = [(a - 1, b - 1) for a, b in bonds1]
    bond_args = [f"{a},{b}" for a, b in bonds0]
    if not bond_args:
        raise RuntimeError(f"no rotatable bonds found in {mol2_p}")

    parm7 = out_dir_path / "parent.parm7"
    rst7 = out_dir_path / "parent.rst7"
    start_json = out_dir_path / "start.json"
    log.info("[alps] whole-ligand: %s rotatable bond(s)", len(bond_args))
    from alps.Log import install_ffpopt_stdio, silence_wavefront_origin_filters
    from alps.Progress import (
        make_whole_board,
        print_whole_run_card,
        tee_job_stdio,
    )

    install_ffpopt_stdio()
    print_whole_run_card(
        ligand=mol2_p.stem,
        model=str(standard_kwargs.get("model") or "qdpi2"),
        nproc=int(nproc),
        delta=int(delta),
        n_bonds=len(bond_args),
        work_dir=out_dir_path,
    )
    store, watcher = make_whole_board(out_dir_path, logger=log)
    if store is not None:
        store.register(
            "parent",
            bonds=len(bond_args),
            log_path=str(out_dir_path / "whole-twist.log"),
        )
        if watcher is not None:
            watcher.start()
    try:
        if store is not None:
            store.update("parent", status="running", stage="prepare")
        if not (skip_existing and parm7.is_file() and rst7.is_file()):
            parm7, rst7 = _tleap_parent_parm_rst(mol2_p, frcmod_p, out_dir_path, log)
        if not (skip_existing and start_json.is_file()):
            prepare_start_json(parm7, rst7, start_json)

        log.info("[alps] twisting parent nproc=%s -> %s", nproc, out_dir_path)
        if store is not None:
            store.update("parent", status="running", stage="twist")
        york_kwargs = york_standard_kwargs(**standard_kwargs)
        silence_wavefront_origin_filters(load_wavefront=True)
        with tee_job_stdio(out_dir_path / "whole-twist.log"), pushd(out_dir_path):
            result = run_dihed_twist_workflow(
                inp=str(start_json),
                bond=bond_args,
                delta=delta,
                nprim=nprim,
                maxiter=maxiter,
                bytype=True,
                nlmaxiter=nlmaxiter,
                nproc=int(nproc),
                skip_existing=skip_existing,
                plot_comparisons=plot_comparisons,
                **york_kwargs,
            )
        if store is not None:
            store.update("parent", status="done", stage="finished")
    except Exception as exc:
        if store is not None:
            store.update(
                "parent",
                status="failed",
                stage="failed",
                error=str(exc)[:200],
            )
        raise
    finally:
        if watcher is not None:
            watcher.stop()
    latest = None
    for it in sorted(out_dir_path.glob("it*.frcmod")):
        latest = it
    if latest is not None:
        shutil.copyfile(latest, out_frcmod_path)
        log.info("[alps] copied %s -> %s", latest, out_frcmod_path)
    return {
        "out_frcmod": str(out_frcmod_path),
        "bonds": bonds0,
        "twist_result": result,
        "out_dir": str(out_dir_path),
    }
