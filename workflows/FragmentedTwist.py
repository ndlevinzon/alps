"""Fragmented dihedral-twist: scission API + ffpopt API, one ALPS process.

ALPS owns this loop. scission fragments and merges; ffpopt twists one
molecule given ``start.json`` and 0-based bonds. Each fragment runs in the
parent process so wavefront prints stay on ALPS stdout.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace as _dc_replace
from pathlib import Path
from types import SimpleNamespace
from typing import Union

from alps.Log import get_logger
from alps.ffpopt_bridge import prepare_start_json, pushd, york_standard_kwargs

PathLike = Union[str, Path]

_FRAG_TWIST_DONE = "frag-twist.done"


def bonds0_from_scission_fit_torsions(fit_torsions) -> list[tuple[int, int]]:
    """Map scission ``fit_torsions`` (1-based) to ffpopt central bonds (0-based)."""
    bonds: list[tuple[int, int]] = []
    for record in fit_torsions:
        pair = record["fragment_rotatable_bond"]
        if len(pair) != 2:
            raise ValueError(
                "fragment_rotatable_bond must have two 1-based indices; "
                f"got {pair!r}"
            )
        bonds.append((int(pair[0]) - 1, int(pair[1]) - 1))
    return bonds


def fragment_twist_done_path(frag_dir: PathLike) -> Path:
    return Path(frag_dir) / _FRAG_TWIST_DONE


def is_fragment_twist_done(frag_dir: PathLike) -> bool:
    return fragment_twist_done_path(frag_dir).is_file()


def mark_fragment_twist_done(frag_dir: PathLike) -> Path:
    path = fragment_twist_done_path(frag_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok\n", encoding="utf-8")
    return path


def clear_fragment_twist_done(frag_dir: PathLike) -> None:
    try:
        fragment_twist_done_path(frag_dir).unlink()
    except FileNotFoundError:
        pass


def partition_fragment_twist_jobs(
    jobs: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Split jobs into 1–2 bond fragments vs the rest (API compat)."""
    cheap: list[dict] = []
    correlated: list[dict] = []
    for job in jobs:
        n_bonds = len(job.get("bonds") or [])
        if n_bonds >= 3:
            correlated.append(job)
        else:
            cheap.append(job)
    return cheap, correlated


def _as_path(value: PathLike) -> Path:
    return Path(value)


def _parent_paths_from_args(*, mol2, lib, frcmod, bundle) -> tuple[Path, Path, Path]:
    if bundle is not None:
        if all(hasattr(bundle, attr) for attr in ("mol2", "lib", "frcmod")):
            return (
                _as_path(bundle.mol2).resolve(),
                _as_path(bundle.lib).resolve(),
                _as_path(bundle.frcmod).resolve(),
            )
        if all(
            hasattr(bundle, attr)
            for attr in ("mol2_path", "lib_path", "frcmod_path")
        ):
            return (
                _as_path(bundle.mol2_path).resolve(),
                _as_path(bundle.lib_path).resolve(),
                _as_path(bundle.frcmod_path).resolve(),
            )
        raise TypeError(
            "bundle must provide mol2/lib/frcmod or mol2_path/lib_path/frcmod_path"
        )
    if mol2 is None or lib is None or frcmod is None:
        raise TypeError(
            "run_fragmented_dihed_twist_workflow requires mol2, lib, and frcmod "
            "(or a bundle= AmberLigandBundle / InputBundle)"
        )
    return (
        _as_path(mol2).resolve(),
        _as_path(lib).resolve(),
        _as_path(frcmod).resolve(),
    )


def _load_existing_fragments(out_dir: Path):
    index_path = out_dir / "fragment_index.json"
    if not index_path.exists():
        return None
    index = json.loads(index_path.read_text())
    fragments = []
    for entry in index.get("fragments", []):
        frag_dir = Path(entry.get("dir") or entry.get("directory") or "")
        if not frag_dir.is_absolute():
            frag_dir = out_dir / frag_dir
        manifest = Path(entry.get("manifest_path") or (frag_dir / "manifest.json"))
        fit_path = frag_dir / "fit_torsions.json"
        fit_torsions = []
        if fit_path.is_file():
            payload = json.loads(fit_path.read_text())
            fit_torsions = payload if isinstance(payload, list) else payload.get(
                "torsions", payload.get("fit_torsions", [])
            )
        fragments.append(
            SimpleNamespace(
                fragment_id=entry.get("fragment_id") or frag_dir.name,
                manifest_path=manifest,
                parm7_path=frag_dir / "fragment.parm7",
                rst7_path=frag_dir / "fragment.rst7",
                fit_torsions=fit_torsions,
            )
        )
    return fragments or None


def _twist_one_fragment(
    fragment,
    *,
    skip_existing: bool,
    log: logging.Logger,
    **twist_kwargs,
) -> dict:
    from ffpopt.Workflows import run_dihed_twist_workflow

    frag_dir = Path(fragment.manifest_path).resolve().parent
    bonds = bonds0_from_scission_fit_torsions(fragment.fit_torsions)
    bond_args = [f"{a},{b}" for a, b in bonds]
    if skip_existing and is_fragment_twist_done(frag_dir):
        log.info("[alps] %s already complete - skipping twist", fragment.fragment_id)
        return {
            "fragment_id": fragment.fragment_id,
            "dir": str(frag_dir),
            "bonds": bonds,
            "twist_result": None,
            "skipped_complete": True,
        }
    if not skip_existing:
        clear_fragment_twist_done(frag_dir)
    if fragment.parm7_path is None or fragment.rst7_path is None:
        raise RuntimeError(
            f"fragment {fragment.fragment_id} has no parm7/rst7 - "
            "scission likely failed tleap (AmberTools on PATH?)"
        )
    start_json = frag_dir / "start.json"
    if not (skip_existing and start_json.exists()):
        log.info("[alps] PrepareInput %s -> %s", fragment.fragment_id, start_json)
        prepare_start_json(fragment.parm7_path, fragment.rst7_path, start_json)
    log.info(
        "[alps] twisting %s (%s bond(s)) nproc=%s",
        fragment.fragment_id,
        len(bonds),
        twist_kwargs.get("nproc"),
    )
    york_kwargs = york_standard_kwargs(**twist_kwargs)
    with pushd(frag_dir):
        result = run_dihed_twist_workflow(
            inp=str(start_json),
            bond=bond_args,
            **york_kwargs,
        )
    mark_fragment_twist_done(frag_dir)
    return {
        "fragment_id": fragment.fragment_id,
        "dir": str(frag_dir),
        "bonds": bonds,
        "twist_result": result,
    }


def run_fragmented_dihed_twist_workflow(
    *,
    mol2: PathLike | None = None,
    lib: PathLike | None = None,
    frcmod: PathLike | None = None,
    bundle=None,
    out_dir: PathLike = "fragments",
    merged_frcmod: PathLike = "merged.frcmod",
    fragment_config=None,
    rotatable_bond_smarts=None,
    delta: int = 10,
    nprim: int = 3,
    maxiter: int = 2,
    nlmaxiter: int = 300,
    nproc: int = 1,
    wf_starting_nodes: int = 4,
    wf_num_conformers: int = 0,
    wf_max_levels: int = -1,
    wf_convergence_threshold: float = 0.01,
    skip_existing: bool = True,
    compare_config=None,
    skip_converged_initial: bool = True,
    convergence_mode: str = "drop",
    plot_comparisons: bool = True,
    logger: logging.Logger | None = None,
    fast_wavefront: bool | None = None,
    multi_centroid: int = 0,
    centroid_mol2: PathLike | None = None,
    fit_cli_args: list | None = None,
    **standard_kwargs,
) -> dict:
    """Fragment with scission, twist each piece with ffpopt, merge DIHE terms."""
    from ffpopt.CpuThreads import pin_math_threads
    from scission import FragmentConfig, InputBundle, fragment_ligand
    from scission.Merge import merge_fragment_frcmods

    log = logger if logger is not None else get_logger("alps")
    if fast_wavefront:
        log.warning(
            "[alps] --fast is ignored: this ffpopt checkout has no GAU_LOOSE / optimizer-ladder presets"
        )
    if multi_centroid or fit_cli_args:
        log.warning(
            "[alps] AFFDO extras (multi-centroid / fit-cli) are not in this ffpopt checkout; ignoring"
        )
    unused = {k: standard_kwargs.pop(k) for k in ("soft_dihed_restraint", "soft_dihed_k",
              "soft_dihed_kmax", "soft_dihed_tol") if k in standard_kwargs}
    if unused:
        log.warning("[alps] soft-dihed restraint flags are not in this ffpopt checkout; ignoring")

    pin_math_threads(1)
    mol2_path, lib_path, parent_frcmod = _parent_paths_from_args(
        mol2=mol2, lib=lib, frcmod=frcmod, bundle=bundle
    )
    config = fragment_config if fragment_config is not None else FragmentConfig()
    if rotatable_bond_smarts is not None:
        extra = (
            (rotatable_bond_smarts,)
            if isinstance(rotatable_bond_smarts, str)
            else tuple(rotatable_bond_smarts)
        )
        if extra:
            config = _dc_replace(
                config,
                rotatable_bond_smarts=config.rotatable_bond_smarts + extra,
            )
    config = _dc_replace(config, nproc=max(1, int(nproc)))
    out_dir_path = _as_path(out_dir).resolve()
    merged_frcmod_path = _as_path(merged_frcmod).resolve()
    input_bundle = InputBundle(
        mol2_path=mol2_path,
        lib_path=lib_path,
        frcmod_path=parent_frcmod,
    )

    existing = _load_existing_fragments(out_dir_path) if skip_existing else None
    if existing is not None:
        log.info(
            "[alps] reusing %s fragment(s) under %s",
            len(existing),
            out_dir_path,
        )
        fragmentation_dump = None
        fragments_iter = existing
    else:
        log.info("[alps] scission fragment -> %s", out_dir_path)
        frag_result = fragment_ligand(input_bundle, out_dir_path, config)
        log.info("[alps] selected %s fragment(s)", len(frag_result.selected_fragments))
        fragmentation_dump = frag_result.to_dict()
        fragments_iter = frag_result.selected_fragments

    twist_kwargs = dict(
        delta=delta,
        nprim=nprim,
        maxiter=maxiter,
        bytype=True,
        nlmaxiter=nlmaxiter,
        nproc=int(nproc),
        wf_starting_nodes=wf_starting_nodes,
        wf_num_conformers=wf_num_conformers,
        wf_max_levels=wf_max_levels,
        wf_convergence_threshold=wf_convergence_threshold,
        skip_existing=skip_existing,
        compare_config=compare_config,
        skip_converged_initial=skip_converged_initial,
        convergence_mode=convergence_mode,
        plot_comparisons=plot_comparisons,
        **standard_kwargs,
    )

    per_fragment = []
    fragment_dirs = []
    for fragment in fragments_iter:
        if not getattr(fragment, "fit_torsions", None):
            log.info("[alps] %s: no fit_torsions - skip", fragment.fragment_id)
            continue
        rec = _twist_one_fragment(
            fragment, skip_existing=skip_existing, log=log, **twist_kwargs
        )
        per_fragment.append(rec)
        fragment_dirs.append(Path(rec["dir"]))

    if not fragment_dirs:
        raise RuntimeError("no fragments had fittable torsions - nothing to merge")

    report_path = merged_frcmod_path.with_name(
        merged_frcmod_path.name + ".merge_report.json"
    )
    log.info("[alps] scission merge -> %s", merged_frcmod_path)
    merge_report = merge_fragment_frcmods(
        parent_frcmod_path=parent_frcmod,
        output_frcmod_path=merged_frcmod_path,
        fragment_dirs=fragment_dirs,
        report_path=report_path,
    )
    return {
        "fragmentation": fragmentation_dump,
        "fragments": per_fragment,
        "merge_report": merge_report,
        "merged_frcmod": str(merged_frcmod_path),
    }
