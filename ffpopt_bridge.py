"""In-process ffpopt helpers used by ALPS workflows.

PrepareInput is still a bin script in ffpopt; run it in this interpreter
so its prints land on the ALPS stdout instead of a subprocess pipe.
"""

from __future__ import annotations

import os
import runpy
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def pushd(path: Path) -> Iterator[Path]:
    """Run York twist with cwd = fragment / parent out dir (outputs are relative)."""
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(prev)


def york_standard_kwargs(*, geometric_opt: bool | None = None, **kwargs) -> dict:
    """Map ALPS names onto independent ffpopt ``AddStandardOptions``.

    York ``--geometric-opt`` means ASE BFGS, which is the opposite of ALPS
    ``geometric_opt=True`` (use geomeTRIC). Default York is geomeTRIC.
    """
    out = dict(kwargs)
    alps_geo = out.pop("geometric_opt", geometric_opt)
    if alps_geo is False:
        out["geometric_opt"] = True
    else:
        out.pop("geometric_opt", None)
    for key in (
        "logger",
        "fast_wavefront",
        "multi_centroid",
        "boltzmann_charges",
        "fit_cli_args",
        "centroid_mol2",
        "soft_dihed_restraint",
        "soft_dihed_k",
        "soft_dihed_kmax",
        "soft_dihed_tol",
    ):
        out.pop(key, None)
    return out


def ffpopt_bin_script(name: str) -> Path:
    """Locate ``ffpopt-PrepareInput.py`` (installed ``ffpopt.bin`` or checkout)."""
    import ffpopt

    pkg = Path(ffpopt.__file__).resolve().parent
    candidates = [
        pkg / "bin" / name,
        pkg.parent.parent / "bin" / name,
    ]
    try:
        from importlib import resources

        import ffpopt.bin as bin_pkg

        candidates.insert(0, Path(str(resources.files(bin_pkg).joinpath(name))))
    except (ImportError, AttributeError, TypeError):
        pass
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"{name} not found next to ffpopt at {pkg}. "
        "Install ffpopt (pip install -e ffpopt) or keep ffpopt "
        "beside this ALPS checkout."
    )


def prepare_start_json(parm7: Path, rst7: Path, out_json: Path) -> Path:
    """Write ``start.json`` via ffpopt PrepareInput in this process."""
    script = ffpopt_bin_script("ffpopt-PrepareInput.py")
    out_json = Path(out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    argv = sys.argv
    sys.argv = [
        str(script),
        f"--parm={Path(parm7).resolve()}",
        f"--crd={Path(rst7).resolve()}",
        f"--out={out_json}",
    ]
    try:
        runpy.run_path(str(script), run_name="__ffpopt_prepare_input__")
    finally:
        sys.argv = argv
    if not out_json.is_file():
        raise RuntimeError(f"PrepareInput did not write {out_json}")
    return out_json
