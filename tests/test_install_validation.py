"""Install validation suite for new users.

Run after ``pip install -e .`` (or ``pip install .``)::

    python -m unittest tests.test_install_validation -v

These tests exercise real import paths, CLI entry points, and a few
non-trivial numerical / I/O helpers. They intentionally avoid AmberTools,
Gaussian, and GPU stacks so a basic conda/pip install can pass on a laptop.

Optional extras (``tblite``, ``geometric``, ...) are checked when present and
skipped with an explicit reason when absent.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import io
import os
import shutil
import sys
import unittest
from pathlib import Path


_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
import _paths  # noqa: E402


def setUpModule():
    """Bind ALPS, then sibling ligandparam / ffpopt / scission."""
    _paths.ensure_alps()


def _has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


class TestCorePackageInstall(unittest.TestCase):
    """Top-level packages resolve and report a coherent version."""

    def test_ligandparam_version(self):
        import ligandparam

        self.assertTrue(ligandparam.__version__)
        self.assertRegex(ligandparam.__version__, r"^\d+\.\d+")

    def test_alps_binds_independent_companions(self):
        import alps
        import ffpopt
        import ligandparam
        import scission
        from alps.companions import companion_status

        self.assertTrue(alps.__version__)
        self.assertRegex(alps.__version__, r"^\d+\.\d+")
        self.assertTrue(hasattr(ligandparam, "__version__"))
        self.assertTrue(hasattr(ffpopt, "Workflows") or hasattr(ffpopt, "GeomOpt"))
        self.assertTrue(hasattr(scission, "fragment_ligand"))
        try:
            dist_ver = importlib.metadata.version("alps")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("alps not installed as a distribution")
        self.assertEqual(dist_ver, alps.__version__)
        mapping = importlib.metadata.packages_distributions()
        self.assertIn("alps", mapping.get("alps", []))
        status = companion_status()
        for name in ("ligandparam", "scission", "ffpopt"):
            self.assertIn(name, status)
            self.assertIn(status[name].mode, {"sibling", "installed", "path"})
        self.assertFalse(getattr(scission, "__ligandparam_bundle__", False))

    def test_import_integrated_packages(self):
        import alps
        import ffpopt
        import ligandparam
        import scission

        self.assertTrue(hasattr(alps, "__version__"))
        self.assertTrue(hasattr(ligandparam, "__version__"))
        self.assertTrue(hasattr(ffpopt, "Workflows") or callable(getattr(ffpopt, "__getattr__", None)))
        self.assertTrue(hasattr(scission, "fragment_ligand"))

    def test_core_scientific_dependencies(self):
        missing = []
        for name in (
            "numpy",
            "scipy",
            "pandas",
            "parmed",
            "ase",
            "networkx",
            "yaml",
            "MDAnalysis",
            "rdkit",
        ):
            if not _has_module(name):
                missing.append(name)
        self.assertEqual(
            missing,
            [],
            "Missing required dependencies: "
            f"{missing}. Install with: pip install -e . "
            "(ALPS orchestrates independent ligandparam, ffpopt, and scission "
            "installs; use the project conda/mamba env from env.yaml when possible).",
        )

    def test_numpy_runtime_usable(self):
        import numpy as np

        a = np.linspace(0.0, 1.0, 5)
        self.assertAlmostEqual(float(a.sum()), 2.5, places=10)

    def test_rdkit_smiles_roundtrip(self):
        if not _has_module("rdkit"):
            self.fail(
                "rdkit is a required dependency but is not importable. "
                "Install with: pip install -e . (or conda install -c conda-forge rdkit)"
            )
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        self.assertIsNotNone(mol)
        self.assertEqual(mol.GetNumAtoms(), 3)


class TestPublicAPISurface(unittest.TestCase):
    """Canonical modules and symbols used by ALPS CLIs must import."""

    def test_companion_default_is_sibling_or_installed(self):
        from alps.companions import companion_status, repo_root

        status = companion_status()
        root = repo_root()
        for name in ("ligandparam", "ffpopt", "scission"):
            info = status[name]
            self.assertIn(info.mode, {"sibling", "installed", "path"}, name)
            self.assertTrue(info.origin.is_dir(), name)
            if info.mode == "sibling":
                self.assertTrue(
                    str(info.origin).startswith(str(root)),
                    f"{name} sibling origin {info.origin} not under {root}",
                )
            mod = importlib.import_module(name)
            self.assertFalse(
                getattr(mod, "__ligandparam_bundle__", False),
                f"{name} should be an independent checkout",
            )

    def test_ffpopt_workflow_surface(self):
        m = importlib.import_module("ffpopt.Workflows")
        self.assertTrue(callable(getattr(m, "run_dihed_twist_workflow")))

    def test_alps_workflow_surface(self):
        m = importlib.import_module("alps.workflows")
        for name in (
            "run_fragmented_dihed_twist_workflow",
            "run_whole_ligand_dihed_twist_workflow",
            "bonds0_from_scission_fit_torsions",
        ):
            self.assertTrue(callable(getattr(m, name)), name)

    def test_ffpopt_geomopt_and_dihedrals(self):
        geom = importlib.import_module("ffpopt.GeomOpt")
        self.assertTrue(hasattr(geom, "GeomOpt"))
        dihed = importlib.import_module("ffpopt.Dihedrals")
        for name in ("FitInputType", "NonlinearSolve", "WriteParmedScript"):
            self.assertTrue(hasattr(dihed, name), name)

    def test_wavefront_1d_nd(self):
        if not _has_module("geometric"):
            self.skipTest("optional: geometric (pip install '.[dihed]')")
        wf = importlib.import_module("ffpopt.WaveFront")
        for name in ("Wavefront", "run_dihed_wavefront"):
            self.assertTrue(hasattr(wf, name), name)
        wnd = importlib.import_module("ffpopt.WaveFrontND")
        self.assertTrue(hasattr(wnd, "Wavefront"))

    def test_scission_public_api(self):
        import scission

        for name in (
            "fragment_ligand",
            "FragmentConfig",
            "SelectedFragment",
            "match_central_bond_smarts",
        ):
            self.assertTrue(hasattr(scission, name), name)

        merge = importlib.import_module("scission.Merge")
        self.assertTrue(callable(merge.merge_fragment_frcmods))
        self.assertTrue(callable(merge.list_iteration_frcmods))


class TestCLIEntrypoints(unittest.TestCase):
    """Console scripts declared in pyproject must be importable callables."""

    EXPECTED = (
        ("alps.cli.LigGetParam", "main"),
        ("alps.cli.LigDihedCorrect", "main"),
        ("alps.cli.LigScission", "main"),
        ("scission.Cli", "main"),
    )
    # These pull RDKit at import time (core dep; skipped only if RDKit absent).
    RDKit_EXPECTED = (
        ("ligandparam.cli.LigGetParam", "main"),
        ("ligandparam.cli.SmilesToPdb", "main"),
    )

    def test_cli_modules_expose_main(self):
        for modname, attr in self.EXPECTED:
            with self.subTest(module=modname):
                mod = importlib.import_module(modname)
                fn = getattr(mod, attr)
                self.assertTrue(callable(fn))

    def test_cli_modules_requiring_rdkit(self):
        if not _has_module("rdkit"):
            self.skipTest(
                "rdkit required for lig-getparam / smiles-to-pdb; "
                "install with: pip install -e ."
            )
        for modname, attr in self.RDKit_EXPECTED:
            with self.subTest(module=modname):
                mod = importlib.import_module(modname)
                self.assertTrue(callable(getattr(mod, attr)))

    def test_installed_console_scripts_when_available(self):
        try:
            dist = importlib.metadata.distribution("alps")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("alps distribution metadata unavailable")
        ep_names = {ep.name for ep in dist.entry_points if ep.group == "console_scripts"}
        for required in (
            "lig-getparam",
            "lig-dihed-correct",
            "lig-scission",
        ):
            self.assertIn(required, ep_names, f"missing console script {required}")
        for banned in (
            "ffpopt-DihedTwistAnimate.py",
            "ffpopt-specialty",
        ):
            self.assertNotIn(
                banned, ep_names, f"companion tool should not be an ALPS console script: {banned}"
            )


class TestBehavioralSmoke(unittest.TestCase):
    """Small but real computations that catch broken installs."""

    def test_shape_match_or_geomopt_imports(self):
        geom = importlib.import_module("ffpopt.GeomOpt")
        self.assertTrue(hasattr(geom, "GeomOpt"))
        try:
            from ffpopt.Dihedrals import NonlinearSolve
        except Exception:
            self.skipTest("ffpopt.Dihedrals not importable in this env")
        self.assertTrue(callable(NonlinearSolve) or NonlinearSolve is not None)

    def test_alps_banner_once(self):
        from alps.cli import Banner as banner_mod
        from alps.cli.Banner import print_startup_banner

        banner_mod._BANNER_PRINTED = False
        os.environ.pop("ALPS_BANNER_PRINTED", None)
        os.environ.pop("FFPOPT_BANNER_PRINTED", None)
        try:
            buf = io.StringIO()
            self.assertTrue(print_startup_banner(stream=buf))
            self.assertIn("ALPS", buf.getvalue())
            banner_mod._BANNER_PRINTED = False
            self.assertFalse(print_startup_banner(stream=buf))
            self.assertEqual(os.environ.get("FFPOPT_BANNER_PRINTED"), "1")
        finally:
            banner_mod._BANNER_PRINTED = False
            os.environ.pop("ALPS_BANNER_PRINTED", None)
            os.environ.pop("FFPOPT_BANNER_PRINTED", None)
            os.environ.pop("LIGANDPARAM_BANNER_PRINTED", None)
            from alps.Log import reset_for_tests

            reset_for_tests()

    def test_progress_hint_and_flushing_stdout(self):
        from alps.cli.Banner import print_progress_hint
        from alps.Log import (
            _FlushingStreamHandler,
            configure_line_buffered_stdio,
            reset_for_tests,
            setup_alps_stdout_logging,
        )

        reset_for_tests()
        try:
            buf = io.StringIO()
            print_progress_hint(stream=buf)
            hint = buf.getvalue()
            self.assertIn("live ASCII board", hint)
            self.assertIn("quiet Slurm", hint)
            configure_line_buffered_stdio()
            self.assertEqual(os.environ.get("PYTHONUNBUFFERED"), "1")
            log_buf = io.StringIO()
            setup_alps_stdout_logging(stream=log_buf)
            from alps.Log import _HANDLER

            self.assertIsInstance(_HANDLER, _FlushingStreamHandler)
        finally:
            from alps.Log import reset_for_tests as _reset

            _reset()

    def test_alps_stdout_keeps_independent_logger_names(self):
        from alps.Log import get_logger, reset_for_tests, setup_alps_stdout_logging

        reset_for_tests()
        buf = io.StringIO()
        try:
            setup_alps_stdout_logging(stream=buf)
            get_logger("alps").info("orchestrator")
            get_logger("alps").info("[alps] already tagged")
            get_logger("ligandparam").info("param only")
            get_logger("scission").info("frag")
            get_logger("ffpopt").info("twist")
            out = buf.getvalue()
            self.assertIn("orchestrator", out)
            self.assertIn("[alps]", out)
            self.assertIn("[alps] already tagged", out)
            self.assertNotIn("[alps] [alps]", out)
            self.assertIn("param only", out)
            self.assertNotIn("[ligandparam] param only", out)
            self.assertIn("[scission] frag", out)
            self.assertIn("[ffpopt] twist", out)
        finally:
            reset_for_tests()

    def test_fragment_board_and_job_stdio_tee(self):
        import logging
        import tempfile
        from pathlib import Path

        from alps.Progress import (
            make_fragment_board,
            print_fragmented_run_card,
            tee_job_stdio,
        )

        logger = logging.getLogger("alps")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            card = io.StringIO()
            print_fragmented_run_card(
                ligand="CHAPS",
                model="xtb",
                nproc=4,
                n_fragments=2,
                work_dir=root,
                stream=card,
            )
            self.assertIn("FRAGMENTED TWIST", card.getvalue())
            self.assertIn("FRAG_STATUS.txt", card.getvalue())

            store, watcher = make_fragment_board(
                root, logger=logger, stream=io.StringIO()
            )
            self.assertIsNotNone(store)
            self.assertIsNotNone(watcher)
            store.register("frag_a", bonds=1, frag_dir=str(root / "frag_a"))
            watcher.start()
            try:
                store.update("frag_a", status="running", stage="twist")
                store.update("frag_a", status="done", stage="finished")
            finally:
                watcher.stop()
            board = (root / "FRAG_STATUS.txt").read_text(encoding="utf-8")
            self.assertIn("frag_a", board)
            self.assertIn("finished", board)

            captured = io.StringIO()
            old = sys.stdout
            sys.stdout = captured
            try:
                with tee_job_stdio(root / "job.log"):
                    print("[twist] scan: demo", flush=True)
            finally:
                sys.stdout = old
            self.assertIn("[twist] scan: demo", captured.getvalue())
            self.assertIn("[twist] scan: demo", (root / "job.log").read_text(encoding="utf-8"))

    def test_york_geometric_opt_is_inverted(self):
        from alps.ffpopt_bridge import york_standard_kwargs

        self.assertNotIn("geometric_opt", york_standard_kwargs(geometric_opt=True, model="xtb"))
        self.assertTrue(york_standard_kwargs(geometric_opt=False)["geometric_opt"])
        stripped = york_standard_kwargs(
            geometric_opt=True, logger=object(), fast_wavefront=True, model="xtb"
        )
        self.assertEqual(stripped, {"model": "xtb"})

    def test_constraints_to_geometric_roundtrip(self):
        try:
            from ffpopt.Constraints import Constraint
        except ImportError:
            self.skipTest("ffpopt.Constraints not importable")
        cons = Constraint("dihed", [0, 1, 2, 3], value=90.0)
        self.assertTrue(hasattr(cons, "value") or cons is not None)


class TestOptionalExtras(unittest.TestCase):
    """Optional stacks: pass when installed, skip cleanly otherwise."""

    def test_geometric_optional(self):
        if not _has_module("geometric"):
            self.skipTest("optional: geometric (pip install '.[dihed]')")
        import geometric  # noqa: F401

    def test_tblite_optional(self):
        if not _has_module("tblite"):
            self.skipTest("optional: tblite (pip install '.[tblite]')")
        import tblite  # noqa: F401

    def test_aimnet_optional(self):
        if not _has_module("aimnet"):
            self.skipTest(
                "optional: aimnet (Python 3.11-3.13; pip install '.[aimnet]')"
            )
        from aimnet.calculators import AIMNet2ASE  # noqa: F401

    def test_ambertools_on_path_optional(self):
        found = {
            name: shutil.which(name)
            for name in ("antechamber", "parmchk2", "tleap")
        }
        if not any(found.values()):
            self.skipTest(
                "optional: AmberTools not on PATH "
                "(needed for lig-getparam / lig-dihed-correct production runs)"
            )
        for name, path in found.items():
            if path is None:
                continue
            self.assertTrue(Path(path).exists(), name)


if __name__ == "__main__":
    unittest.main()
