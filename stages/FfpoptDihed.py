"""Optional ffpopt dihedral-twist correction stage.

Wraps ``alps.workflows.run_fragmented_dihed_twist_workflow`` so recipes and
the ``lig-dihed-correct`` CLI can apply torsion corrections to a parent
``mol2`` / ``lib`` / ``frcmod`` triplet. ALPS calls scission and ffpopt;
those packages do not import each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from ligandparam.stages.AbstractStage import AbstractStage


def coerce_fragment_config(value: Any):
    """Normalize ``fragment_config`` to a FragmentConfig or ``None``.

    Accepts ``None``, a :class:`scission.Models.FragmentConfig`, or a dict
    for ``FragmentConfig.from_dict``.
    """
    if value is None:
        return None
    try:
        from scission.Models import FragmentConfig
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fragment_config requires the integrated scission package"
        ) from exc
    if isinstance(value, FragmentConfig):
        return value
    if isinstance(value, dict):
        return FragmentConfig.from_dict(value)
    raise TypeError(
        "fragment_config must be None, a FragmentConfig, or a dict; "
        f"got {type(value).__name__}"
    )


class StageDihedTwistCorrection(AbstractStage):
    """Fit and merge dihedral corrections into a parent Amber frcmod.

Requires an installed ``ffpopt`` and ``scission`` (ALPS companions). The
parent ``lib`` is left unchanged; use the merged ``out_frcmod`` together
with the original library in LEaP.

    Parameters
    ----------
    stage_name : str
        Stage name.
    main_input : path-like
        Parent ligand ``mol2``.
    cwd : path-like
        Working directory (fragment outputs go under ``out_dir``).
    in_lib : path-like
        Parent Amber ``lib``.
    in_frcmod : path-like
        Parent ``frcmod`` produced by ``parmchk2`` / the recipe.
    out_frcmod : path-like
        Destination for the merged, torsion-corrected frcmod.
    out_dir : path-like, optional
        Directory for scission fragments and per-fragment twist outputs.
        Default ``{cwd}/dihed_fragments``.
    model : str, optional
        High-level ffpopt model chemistry. Default ``"qdpi2"``.
    maxiter : int, optional
        Fit-then-rescan iterations per fragment. Default ``2``.
    nprim : int, optional
        Cosine primitives per torsion family. Default ``3``.
    geometric_opt : bool, optional
        Use geomeTRIC for constrained optimizations. Default ``True``.
    skip_existing : bool, optional
        Restart-friendly reuse of on-disk artifacts. Default ``True``.
    delta : int, optional
        Wavefront dihedral step in degrees (CLI ``--delta``). Default ``10``.
    fragment_config : FragmentConfig or dict, optional
        Scission fragmentation settings forwarded to
        ``run_fragmented_dihed_twist_workflow``. Default ``None`` (scission
        defaults).
    """

    def __init__(
        self,
        stage_name: str,
        main_input: Union[Path, str],
        cwd: Union[Path, str],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(stage_name, main_input, cwd, *args, **kwargs)
        self.in_mol2 = Path(main_input)
        self.in_lib = Path(kwargs["in_lib"])
        self.in_frcmod = Path(kwargs["in_frcmod"])
        self.out_frcmod = Path(kwargs["out_frcmod"])
        self.out_dir = Path(kwargs.get("out_dir", self.cwd / "dihed_fragments"))
        self.model = kwargs.get("model", "qdpi2")
        self.maxiter = int(kwargs.get("maxiter", 2))
        self.nprim = int(kwargs.get("nprim", 3))
        self.delta = int(kwargs.get("delta", 10))
        self.geometric_opt = bool(kwargs.get("geometric_opt", True))
        self.skip_existing = bool(kwargs.get("skip_existing", True))
        self.rotatable_bond_smarts = kwargs.get("rotatable_bond_smarts")
        self.fragment_config = coerce_fragment_config(kwargs.get("fragment_config"))
        self.fast_wavefront = kwargs.get("fast_wavefront")
        self.geometric_maxiter = kwargs.get("geometric_maxiter")
        self.geometric_converge = kwargs.get("geometric_converge")
        self.ase_opt_tol = kwargs.get("ase_opt_tol")
        self.whole_ligand = bool(kwargs.get("whole_ligand", False))
        self.multi_centroid = int(kwargs.get("multi_centroid", 0) or 0)
        self.boltzmann_charges = bool(kwargs.get("boltzmann_charges", False))
        self.soft_dihed_restraint = bool(kwargs.get("soft_dihed_restraint", False))
        self.soft_dihed_k = kwargs.get("soft_dihed_k")
        self.soft_dihed_kmax = kwargs.get("soft_dihed_kmax")
        self.soft_dihed_tol = kwargs.get("soft_dihed_tol")
        self.fit_cli_args = list(kwargs.get("fit_cli_args") or [])
        self.add_required(self.in_mol2)
        self.add_required(self.in_lib)
        self.add_required(self.in_frcmod)

    def _run(
        self,
        dry_run: bool = False,
        nproc: Optional[int] = None,
        mem: Optional[int] = None,
    ) -> Any:
        """Run the fragmented or whole-ligand dihed-twist workflow."""
        nproc_eff = self.nproc if nproc is None else nproc

        self.logger.info(
            "Dihed twist correction: mol2=%s lib=%s frcmod=%s -> %s (model=%s)",
            self.in_mol2,
            self.in_lib,
            self.in_frcmod,
            self.out_frcmod,
            self.model,
        )
        if self.whole_ligand:
            self.logger.info("[alps] whole-ligand twist (no scission fragments)")
        if dry_run:
            which = (
                "run_whole_ligand_dihed_twist_workflow"
                if self.whole_ligand
                else "run_fragmented_dihed_twist_workflow"
            )
            self.logger.info(
                "Dry run: would call %s (out_dir=%s, maxiter=%s, nproc=%s)",
                which,
                self.out_dir,
                self.maxiter,
                nproc_eff,
            )
            return None

        try:
            from alps.workflows import (
                run_fragmented_dihed_twist_workflow,
                run_whole_ligand_dihed_twist_workflow,
            )
        except ImportError as exc:
            raise ImportError(
                "StageDihedTwistCorrection requires independent 'ffpopt' and "
                "'scission' packages (sibling checkouts ffpopt-main / "
                "scission-main, or pip install -e). AmberTools must be on PATH."
            ) from exc

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_frcmod.parent.mkdir(parents=True, exist_ok=True)

        extra = {}
        if self.geometric_maxiter is not None:
            extra["geometric_maxiter"] = int(self.geometric_maxiter)
        if self.geometric_converge is not None:
            extra["geometric_converge"] = self.geometric_converge
        if self.ase_opt_tol is not None:
            extra["ase_opt_tol"] = float(self.ase_opt_tol)
        if self.soft_dihed_restraint:
            extra["soft_dihed_restraint"] = True
            if self.soft_dihed_k is not None:
                extra["soft_dihed_k"] = float(self.soft_dihed_k)
            if self.soft_dihed_kmax is not None:
                extra["soft_dihed_kmax"] = float(self.soft_dihed_kmax)
            if self.soft_dihed_tol is not None:
                extra["soft_dihed_tol"] = float(self.soft_dihed_tol)

        if self.whole_ligand:
            result = run_whole_ligand_dihed_twist_workflow(
                mol2=self.in_mol2.resolve(),
                lib=self.in_lib.resolve(),
                frcmod=self.in_frcmod.resolve(),
                out_dir=self.out_dir.resolve(),
                out_frcmod=self.out_frcmod.resolve(),
                model=self.model,
                maxiter=self.maxiter,
                nprim=self.nprim,
                delta=self.delta,
                nproc=int(nproc_eff),
                geometric_opt=self.geometric_opt,
                skip_existing=self.skip_existing,
                rotatable_bond_smarts=self.rotatable_bond_smarts,
                fast_wavefront=self.fast_wavefront,
                multi_centroid=self.multi_centroid,
                boltzmann_charges=self.boltzmann_charges,
                fit_cli_args=self.fit_cli_args,
                logger=self.logger,
                **extra,
            )
            self.logger.info(
                "Whole-ligand dihed twist complete: out_frcmod=%s bonds=%s",
                result.get("out_frcmod"),
                result.get("bonds"),
            )
            if result.get("boltzmann_charges"):
                self.logger.info(
                    "Boltzmann charge rewrite: mol2=%s lib=%s",
                    (result["boltzmann_charges"] or {}).get("out_mol2"),
                    (result["boltzmann_charges"] or {}).get("out_lib"),
                )
            return result

        result = run_fragmented_dihed_twist_workflow(
            mol2=self.in_mol2.resolve(),
            lib=self.in_lib.resolve(),
            frcmod=self.in_frcmod.resolve(),
            out_dir=self.out_dir.resolve(),
            merged_frcmod=self.out_frcmod.resolve(),
            model=self.model,
            maxiter=self.maxiter,
            nprim=self.nprim,
            delta=self.delta,
            nproc=int(nproc_eff),
            geometric_opt=self.geometric_opt,
            skip_existing=self.skip_existing,
            rotatable_bond_smarts=self.rotatable_bond_smarts,
            fragment_config=self.fragment_config,
            fast_wavefront=self.fast_wavefront,
            multi_centroid=self.multi_centroid,
            centroid_mol2=self.in_mol2.resolve(),
            fit_cli_args=self.fit_cli_args,
            logger=self.logger,
            **extra,
        )
        self.logger.info(
            "Dihed twist complete: merged_frcmod=%s fragments=%s",
            result.get("merged_frcmod"),
            [f.get("fragment_id") for f in result.get("fragments", [])],
        )
        return result


def dihed_twist_stage_kwargs(
    *,
    mol2: Path,
    lib: Path,
    frcmod: Path,
    out_frcmod: Path,
    out_dir: Path,
    model: str,
    maxiter: int,
    nproc: int,
    logger,
    geometric_opt: bool = True,
    nprim: int = 3,
    delta: int = 10,
    skip_existing: bool = True,
    rotatable_bond_smarts=None,
    fragment_config=None,
    fast_wavefront=None,
) -> dict:
    """Keyword arguments for constructing :class:`StageDihedTwistCorrection`."""
    return {
        "main_input": mol2,
        "in_lib": lib,
        "in_frcmod": frcmod,
        "out_frcmod": out_frcmod,
        "out_dir": out_dir,
        "model": model,
        "maxiter": maxiter,
        "nproc": nproc,
        "nprim": nprim,
        "delta": delta,
        "geometric_opt": geometric_opt,
        "skip_existing": skip_existing,
        "rotatable_bond_smarts": rotatable_bond_smarts,
        "fragment_config": fragment_config,
        "fast_wavefront": fast_wavefront,
        "logger": logger,
    }
