"""Locate this ALPS checkout and the sibling ``*-main`` workspace."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def alps_checkout() -> Path:
    """``alps-main/`` (or a standalone ALPS clone)."""
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    """Parent of this checkout: ``ligandparam-main``, ``scission-main``, ``ffpopt-main``."""
    return alps_checkout().parent


def ensure_alps() -> None:
    """Make ``import alps`` work before ``pip install -e .`` (flat package dir)."""
    if "alps" in sys.modules:
        return
    pkg = alps_checkout()
    init = pkg / "__init__.py"
    if not init.is_file():
        raise RuntimeError(f"ALPS package init not found at {init}")
    spec = importlib.util.spec_from_file_location(
        "alps",
        init,
        submodule_search_locations=[str(pkg)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load alps from {pkg}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["alps"] = mod
    spec.loader.exec_module(mod)
