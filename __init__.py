"""ALPS: orchestrator for independent ligandparam, ffpopt, and scission installs.

``pip install -e .`` (from this directory) installs the orchestrator only.
Companion trees sit beside it (``ligandparam``, ``scission``,
``ffpopt``) or on ``sys.path``. Importing ``alps`` binds those trees,
then ``lig-getparam`` / ``lig-dihed-correct`` / ``lig-scission`` drive the
pipeline via APIs.
"""

from __future__ import annotations

__version__ = "1.6.1"

from .companions import bootstrap as _bootstrap
from .companions import install_import_hook as _install_companion_hook

_install_companion_hook()
_bootstrap()
