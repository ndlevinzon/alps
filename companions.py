"""Bind sibling (or installed) ligandparam / ffpopt / scission trees.

ALPS is only the orchestrator. The three tools live as independent packages
at the repo root (or on ``sys.path`` after ``pip install -e``)::

    ligandparam/          # import ligandparam
    scission/             # import scission
    ffpopt/src/python/lib # import ffpopt

Older sibling names (``ligandparam-main``, ``scission-main``, ``ffpopt-main``)
are still discovered.

Environment (optional; auto-discovery is the default)::

    ALPS_LIGANDPARAM_PATH / LIGANDPARAM_PATH
    ALPS_FFPOPT_PATH      / LIGANDPARAM_FFPOPT_PATH
    ALPS_SCISSION_PATH    / LIGANDPARAM_SCISSION_PATH

A PATH value may be the directory that contains the package
(``.../lib`` with ``lib/ffpopt/``), or the package directory itself
(``scission/`` with ``__init__.py``).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

_COMPANION_NAMES = ("ligandparam", "scission", "ffpopt")
_HOOK_NAMES = frozenset(_COMPANION_NAMES)

_BOOTSTRAPPED = False
_HOOK_INSTALLED = False
_IN_BOOTSTRAP = False
_STATE: dict[str, "CompanionInfo"] | None = None


@dataclass(frozen=True)
class CompanionInfo:
    """Where one companion package was loaded from."""

    name: str
    mode: str
    origin: Path
    path_entry: Path
    layout: str

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "mode": self.mode,
            "origin": str(self.origin),
            "path_entry": str(self.path_entry),
            "layout": self.layout,
        }


def package_dir() -> Path:
    """This ALPS checkout (``alps/`` or a standalone clone)."""
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    """Directory that contains sibling companion checkouts.

    ``alps/companions.py`` -> parent of ``alps``. When ALPS sits beside
    ``ligandparam`` / ``scission`` / ``ffpopt``, that parent is the workspace.
    """
    return Path(__file__).resolve().parent.parent


def sibling_checkout(name: str, *, root: Path | None = None) -> Path | None:
    """Return ``<root>/<name>`` or the legacy ``<root>/<name>-main`` folder."""
    base = repo_root() if root is None else root
    for folder in (base / name, base / f"{name}-main"):
        if folder.is_dir():
            return folder
    return None


def bundled_src_root() -> Path:
    """Deprecated alias for :func:`repo_root`."""
    return repo_root()


def companion_status() -> dict[str, CompanionInfo]:
    """Return the resolved companion map (bootstraps if needed)."""
    return dict(bootstrap())


def format_status_line() -> str:
    """Companion versions from each checkout's ``pyproject.toml``, one per line."""
    return "\n".join(format_companion_lines())


def format_companion_lines() -> list[str]:
    """``ligandparam = v1.6.1 (C:/.../ligandparam-main)`` for each companion."""
    lines: list[str] = []
    for name in _COMPANION_NAMES:
        try:
            info = companion_status()[name]
        except Exception as exc:
            lines.append(f"  {name} = vunknown ({exc})")
            continue
        toml = _find_pyproject(info.origin)
        ver = _project_version(toml) if toml is not None else None
        if not ver:
            ver = _installed_version(name)
        path = toml.parent if toml is not None else info.origin
        ver_s = f"v{ver}" if ver else "vunknown"
        lines.append(f"  {name} = {ver_s} ({path})")
    return lines


def print_status_line(*, file=None) -> None:
    """Write companion versions unless the startup banner already did."""
    if os.environ.get("ALPS_BANNER_PRINTED"):
        return
    text = format_status_line()
    print(text, file=file if file is not None else sys.stdout, flush=True)


def _find_pyproject(origin: Path) -> Path | None:
    """Walk from the import origin up to the checkout ``pyproject.toml``."""
    here = Path(origin).resolve()
    for _ in range(10):
        cand = here / "pyproject.toml"
        if cand.is_file():
            return cand
        if here.parent == here:
            break
        here = here.parent
    return None


def _project_version(toml_path: Path) -> str | None:
    """Read ``[project].version`` from a PEP 621 ``pyproject.toml``."""
    try:
        text = toml_path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        import tomllib
    except ImportError:
        tomllib = None
    if tomllib is not None:
        try:
            data = tomllib.loads(text)
        except Exception:
            data = None
        if isinstance(data, dict):
            ver = (data.get("project") or {}).get("version")
            if ver:
                return str(ver)
    in_project = False
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            continue
        if not in_project:
            continue
        match = re.match(r'version\s*=\s*["\']([^"\']+)["\']', stripped)
        if match:
            return match.group(1)
    return None


def _installed_version(name: str) -> str | None:
    try:
        import importlib.metadata

        return importlib.metadata.version(name)
    except Exception:
        return None


def reset_for_tests() -> None:
    """Allow a later :func:`bootstrap` to run again (unit tests only)."""
    global _BOOTSTRAPPED, _STATE
    _BOOTSTRAPPED = False
    _STATE = None
    for name in _COMPANION_NAMES:
        _purge(name)


def install_import_hook() -> None:
    """Intercept companion imports until :func:`bootstrap` binds the trees."""
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    sys.meta_path.insert(0, _CompanionFinder())
    _HOOK_INSTALLED = True


class _CompanionFinder:
    """sys.meta_path entry that bootstraps, then defers to the normal finder."""

    def find_spec(self, fullname, path, target=None):  # noqa: ARG002
        top = fullname.split(".", 1)[0]
        if top not in _HOOK_NAMES:
            return None
        if _IN_BOOTSTRAP or _BOOTSTRAPPED:
            return None
        bootstrap()
        return None


def bootstrap() -> Mapping[str, CompanionInfo]:
    """Bind ligandparam, scission, and ffpopt. Idempotent."""
    global _BOOTSTRAPPED, _STATE, _IN_BOOTSTRAP
    if _BOOTSTRAPPED and _STATE is not None:
        return _STATE

    _IN_BOOTSTRAP = True
    try:
        infos: dict[str, CompanionInfo] = {}
        for name in _COMPANION_NAMES:
            infos[name] = _load(name)
        _apply_pythonpath(infos)
        _apply_ffpopt_bin_path(infos.get("ffpopt"))
        _STATE = infos
        _BOOTSTRAPPED = True
        return _STATE
    finally:
        _IN_BOOTSTRAP = False


_MODE_WORDS = frozenset({"internal", "external", "bundled", "auto", "sibling", "path"})


def _env_path(name: str) -> str | None:
    keys = [
        f"ALPS_{name.upper()}_PATH",
        f"LIGANDPARAM_{name.upper()}_PATH",
    ]
    if name == "ligandparam":
        keys.append("LIGANDPARAM_PATH")
    else:
        keys.append(f"LIGANDPARAM_{name.upper()}")
    for key in keys:
        raw = os.environ.get(key)
        if raw is None or not raw.strip():
            continue
        val = raw.strip()
        if val.lower() in _MODE_WORDS:
            continue
        return val
    return None


def _default_tree(name: str) -> Path | None:
    root = repo_root()
    folders = (root / name, root / f"{name}-main")
    if name == "ffpopt":
        for folder in folders:
            lib = folder / "src" / "python" / "lib"
            if (lib / "ffpopt" / "__init__.py").is_file():
                return lib
        return None
    for folder in folders:
        if (folder / "__init__.py").is_file():
            return folder
        nested = folder / "src" / name
        if (nested / "__init__.py").is_file():
            return nested.parent
    return None


def _load(name: str) -> CompanionInfo:
    hint = _env_path(name)
    if hint:
        entry, layout = _resolve_path_entry(name, Path(hint))
        _bind(name, entry, layout)
        origin = Path(sys.modules[name].__file__).resolve().parent
        return CompanionInfo(
            name=name,
            mode="path",
            origin=origin,
            path_entry=entry.resolve(),
            layout=layout,
        )

    sibling = _default_tree(name)
    if sibling is not None:
        entry, layout = _resolve_path_entry(name, sibling)
        _bind(name, entry, layout)
        origin = Path(sys.modules[name].__file__).resolve().parent
        return CompanionInfo(
            name=name,
            mode="sibling",
            origin=origin,
            path_entry=entry.resolve(),
            layout=layout,
        )

    importlib.invalidate_caches()
    mod = importlib.import_module(name)
    origin = Path(mod.__file__).resolve().parent
    return CompanionInfo(
        name=name,
        mode="installed",
        origin=origin,
        path_entry=origin.parent,
        layout="site",
    )


def _resolve_path_entry(name: str, hint: Path) -> tuple[Path, str]:
    hint = hint.expanduser().resolve()
    if (hint / name / "__init__.py").is_file():
        return hint, "parent"
    if (hint / "__init__.py").is_file():
        return hint, "flat"
    raise RuntimeError(
        f"{name} path {hint} is not a package tree. Pass the parent of "
        f"{name}/ or the directory that contains {name}/__init__.py."
    )


def _already_bound(name: str, origin: Path) -> bool:
    mod = sys.modules.get(name)
    file = getattr(mod, "__file__", None) if mod is not None else None
    if not file:
        return False
    return Path(file).resolve().parent == origin.resolve()


def _purge(name: str) -> None:
    prefix = name + "."
    for key in list(sys.modules):
        if key == name or key.startswith(prefix):
            del sys.modules[key]


def _prepend_sys_path(entry: str) -> None:
    while entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)


def _bind(name: str, path_entry: Path, layout: str) -> None:
    path_entry = path_entry.resolve()
    if layout == "flat":
        origin = path_entry
        if _already_bound(name, origin):
            return
        _purge(name)
        spec = importlib.util.spec_from_file_location(
            name,
            origin / "__init__.py",
            submodule_search_locations=[str(origin)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {name} from {origin}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return

    expected = path_entry / name
    if _already_bound(name, expected):
        _prepend_sys_path(str(path_entry))
        return
    _purge(name)
    _prepend_sys_path(str(path_entry))
    importlib.invalidate_caches()
    importlib.import_module(name)


def _apply_pythonpath(infos: Mapping[str, CompanionInfo]) -> None:
    """So spawn workers re-import the same ffpopt tree.

    Flat packages (this ALPS checkout, ligandparam, scission)
    are bound via ``spec_from_file_location`` / editable install, not
    ``PYTHONPATH`` folder names.
    """
    entries: list[str] = []
    for name in _COMPANION_NAMES:
        info = infos[name]
        if info.layout == "parent":
            item = str(info.path_entry)
        elif info.layout == "flat":
            continue
        else:
            item = str(info.path_entry)
        if item not in entries:
            entries.append(item)
    old = os.environ.get("PYTHONPATH", "")
    rest = [part for part in old.split(os.pathsep) if part and part not in entries]
    os.environ["PYTHONPATH"] = os.pathsep.join(entries + rest) if entries else old


def _apply_ffpopt_bin_path(info: CompanionInfo | None) -> None:
    """Put ``ffpopt-PrepareInput.py`` on PATH for York workflow subprocesses."""
    if info is None:
        return
    origin = info.origin
    candidates = (
        origin / "bin",
        origin.parent.parent / "bin",
    )
    bin_dir = next(
        (path for path in candidates if (path / "ffpopt-PrepareInput.py").is_file()),
        None,
    )
    if bin_dir is None:
        return
    item = str(bin_dir)
    parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    if item not in parts:
        os.environ["PATH"] = os.pathsep.join([item, *parts])
