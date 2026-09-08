"""Developer regression suite for the ALPS orchestrator.

Companion unit tests live in ligandparam and scission. This file covers
ALPS glue plus leftover ffpopt helpers that still live only here
(no AmberTools / Gaussian)::

    python -m unittest tests.test_developer_regression -v
"""

from __future__ import annotations

import importlib
import importlib.util
import io
import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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


def _have_split_ffpopt() -> bool:
    """True only for the old monorepo ``src/ffpopt`` layout (runtime/geom/scan)."""
    try:
        return importlib.util.find_spec("ffpopt.runtime") is not None
    except ModuleNotFoundError:
        return False


@unittest.skipUnless(_have_split_ffpopt(), "split src/ffpopt not present (using independent ffpopt)")
class TestWavefrontMixinHelpers(unittest.TestCase):
    def test_precheck_geometry_clash_reports_error(self):
        from ffpopt.scan.WavefrontMixins import precheck_geometry_clash

        def boom():
            raise RuntimeError("bad geom")

        err = precheck_geometry_clash(get_atoms=boom, bonds=[], min_dist=0.8)
        self.assertIsNotNone(err)
        self.assertIn("precheck_error", err)

    def test_covalent_geometry_error_flags_flying_hydrogen(self):
        import numpy as np
        from ffpopt.geom.Constraints import covalent_geometry_error

        pos = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        err = covalent_geometry_error(pos, [(0, 1)], numbers=[6, 1])
        self.assertIsNotNone(err)
        self.assertIn("flew off", err)

        ok = covalent_geometry_error(
            np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]]),
            [(0, 1)],
            numbers=[6, 1],
        )
        self.assertIsNone(ok)

        nan = covalent_geometry_error(
            np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]),
            [(0, 1)],
            numbers=[6, 1],
        )
        self.assertIn("non-finite", nan)

    def test_precheck_skips_broken_covalent_bond(self):
        import numpy as np
        from ffpopt.scan.WavefrontMixins import precheck_geometry_clash

        class _Atoms:
            def get_positions(self):
                return np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])

            def get_atomic_numbers(self):
                return np.array([6, 1])

        err = precheck_geometry_clash(get_atoms=_Atoms, bonds=[(0, 1)])
        self.assertEqual(err, "broken_geometry")

    def test_replace_node_with_pickle_noop_when_missing(self):
        from ffpopt.scan.WavefrontMixins import replace_node_with_pickle

        node = SimpleNamespace(node_pkl=Path("definitely-missing-node.pkl"), node_id=1, los="keep")
        replace_node_with_pickle(node)
        self.assertEqual(node.los, "keep")

    def _butane_like_struct(self):
        from ffpopt.Struct import Struct

        # Planar cis-like C-C-C-C (dihedral ~0); RotateMask moves atom 3.
        pos = [
            [0.0, 1.54, 0.0],
            [0.0, 0.0, 0.0],
            [1.54, 0.0, 0.0],
            [1.54, 1.54, 0.0],
        ]
        return Struct(
            {
                "positions": pos,
                "elements": ["C", "C", "C", "C"],
                "charges": [0.0, 0.0, 0.0, 0.0],
                "names": ["C1", "C2", "C3", "C4"],
                "types": ["c", "c", "c", "c"],
                "bonds": [[0, 1], [1, 2], [2, 3]],
                "spin": 1,
                "energy": None,
                "forces": None,
            }
        )

    def test_rotate_coords_about_bond_hits_wrapped_target(self):
        import numpy as np
        from ffpopt.AmberParm import RotateMask, bonds2graph
        from ffpopt.geom.Geometry import CptDihed, rotate_coords_about_bond
        from ffpopt.scan.WavefrontMixins import (
            _wrapped_dihed_delta_deg,
            signed_wrapped_dihed_delta_deg,
        )

        crd = np.array(
            [
                [0.0, 1.54, 0.0],
                [0.0, 0.0, 0.0],
                [1.54, 0.0, 0.0],
                [1.54, 1.54, 0.0],
            ]
        )
        idxs = [0, 1, 2, 3]
        obs = float(CptDihed(crd[0], crd[1], crd[2], crd[3]))
        target = 250.0
        dphi = signed_wrapped_dihed_delta_deg(obs, target)
        self.assertLess(abs(dphi), 180.0 + 1e-9)
        mask = RotateMask(bonds2graph([[0, 1], [1, 2], [2, 3]]), idxs)
        out = rotate_coords_about_bond(crd, 1, 2, dphi, mask)
        new = float(CptDihed(out[0], out[1], out[2], out[3]))
        self.assertLess(_wrapped_dihed_delta_deg(new, target), 1e-6)
        np.testing.assert_allclose(out[0], crd[0], atol=1e-12)
        np.testing.assert_allclose(out[1], crd[1], atol=1e-12)

    def test_cpt_dihed_collinear_does_not_warn(self):
        import warnings
        import numpy as np
        from ffpopt.geom.Geometry import CptAngle, CptDihed

        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([2.0, 0.0, 0.0])
        d = np.array([3.0, 0.0, 0.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dih = CptDihed(a, b, c, d)
            ang = CptAngle(a, b, c)
        self.assertTrue(np.isfinite(dih))
        self.assertTrue(np.isfinite(ang))
        self.assertFalse(any("arccos" in str(w.message).lower() for w in caught))

    def test_seed_struct_rigid_dihed_rotates_and_clash_fallback(self):
        import numpy as np
        from unittest.mock import patch
        from ffpopt.geom.Geometry import CptDihed
        from ffpopt.scan.WavefrontMixins import (
            _wrapped_dihed_delta_deg,
            dihed_seed_targets,
            seed_struct_rigid_dihed_rotates,
        )

        struct = self._butane_like_struct()
        idxs = [0, 1, 2, 3]
        crd = np.asarray(struct.data["positions"], dtype=float)
        obs = float(CptDihed(crd[0], crd[1], crd[2], crd[3]))
        target = (obs + 90.0 + 360.0) % 360.0
        node = SimpleNamespace(
            is_nd=False,
            constraints=[SimpleNamespace(idxs=idxs, value=target)],
            angle=target,
        )
        self.assertEqual(dihed_seed_targets(node), [(idxs, float(target))])

        rotated = seed_struct_rigid_dihed_rotates(
            struct, [(idxs, target)], node_id=1
        )
        self.assertIsNot(rotated, struct)
        new = np.asarray(rotated.data["positions"], dtype=float)
        self.assertLess(
            _wrapped_dihed_delta_deg(
                float(CptDihed(new[0], new[1], new[2], new[3])), target
            ),
            1e-6,
        )

        with patch(
            "ffpopt.geom.Constraints.has_nonbonded_clash",
            return_value=(True, 0, 3, 0.1),
        ):
            kept = seed_struct_rigid_dihed_rotates(
                struct, [(idxs, target)], node_id=2
            )
        self.assertIs(kept, struct)
        np.testing.assert_allclose(
            np.asarray(kept.data["positions"], dtype=float), crd
        )

    def test_skip_hard_ic_after_inband_mm_only(self):
        from ffpopt.scan.WavefrontMixins import skip_hard_ic_after_inband

        self.assertTrue(
            skip_hard_ic_after_inband(two_stage=False, dphi_deg=0.2)
        )
        self.assertTrue(
            skip_hard_ic_after_inband(two_stage=True, dphi_deg=0.2)
        )
        self.assertTrue(
            skip_hard_ic_after_inband(two_stage=True, dphi_deg=0.01)
        )

    def test_node_wall_timeout_kills_hung_callable(self):
        import time
        from ffpopt.runtime.WallTimeout import (
            fork_wall_available,
            run_with_node_wall,
            timed_out_node_result,
        )

        payload = timed_out_node_result(
            {"angle": 280.0, "coords": None}, message="wall_timeout: 300s"
        )
        self.assertFalse(payload["active"])
        self.assertTrue(payload["complete"])
        self.assertEqual(payload["error"], "wall_timeout: 300s")
        self.assertEqual(run_with_node_wall(lambda: 7, wall_sec=0), 7)

        if not fork_wall_available():
            self.skipTest("fork wall-clock timeout is Linux-only")

        def _hang():
            time.sleep(5)
            return {"ok": True}

        out = run_with_node_wall(
            _hang, wall_sec=0.3, job={"angle": 280.0}
        )
        self.assertIn("wall_timeout", str(out.get("error")))
        self.assertEqual(out.get("complete"), True)

        def _fast():
            return {"energy": 1.0, "complete": True}

        ok = run_with_node_wall(_fast, wall_sec=2.0, job={"angle": 10.0})
        self.assertEqual(ok["energy"], 1.0)

    def test_format_drain_wait_line(self):
        from types import SimpleNamespace
        from ffpopt.scan.WavefrontMixins import format_drain_wait_line

        line = format_drain_wait_line(
            pending=2,
            in_flight=7,
            waited_sec=3600,
            nodes=[SimpleNamespace(angle=80), SimpleNamespace(angle=90)],
        )
        self.assertIn("waiting: pending=2 in-flight=7 for 3600s", line)
        self.assertIn("angles=80,90", line)


# ---------------------------------------------------------------------------
# Packaged FFPOPT_* defaults JSON
# ---------------------------------------------------------------------------


@unittest.skipUnless(_have_split_ffpopt(), "split src/ffpopt not present (using independent ffpopt)")
class TestEnvDefaults(unittest.TestCase):
    def tearDown(self):
        from ffpopt.runtime.EnvDefaults import clear_defaults_cache

        clear_defaults_cache()

    def test_packaged_json_loads_and_is_the_store(self):
        from ffpopt.runtime.EnvDefaults import (
            defaults_path,
            env_bool,
            env_int,
            env_str,
            env_value,
            packaged_defaults,
        )

        path = defaults_path()
        self.assertTrue(path.is_file(), path)
        data = packaged_defaults()
        self.assertFalse(data["FFPOPT_FAST_WAVEFRONT"])
        self.assertEqual(data["FFPOPT_MIN_WF_NPROC"], 2)
        self.assertIsNone(data["FFPOPT_ASE_FIRST"])
        self.assertEqual(data["FFPOPT_XTB_GUESS"], "eeq")
        self.assertTrue(data["FFPOPT_BOND_BATCH"])
        self.assertEqual(data["FFPOPT_FIT_MODE"], "barrier")
        self.assertEqual(data["FFPOPT_DIHED_FC_MAX"], 25.0)
        with patch.dict(os.environ, {}, clear=False):
            for key in (
                "FFPOPT_FAST_WAVEFRONT",
                "FFPOPT_DEFAULTS",
                "FFPOPT_MIN_WF_NPROC",
                "FFPOPT_XTB_GUESS",
            ):
                os.environ.pop(key, None)
            self.assertFalse(env_bool("FFPOPT_FAST_WAVEFRONT"))
            self.assertEqual(env_int("FFPOPT_MIN_WF_NPROC"), 2)
            self.assertIsNone(env_value("FFPOPT_ASE_FIRST"))
            self.assertEqual(env_str("FFPOPT_XTB_GUESS"), "eeq")

    def test_export_overrides_json(self):
        from ffpopt.runtime.EnvDefaults import env_bool, env_int, env_value

        with patch.dict(
            os.environ,
            {
                "FFPOPT_FAST_WAVEFRONT": "1",
                "FFPOPT_MIN_WF_NPROC": "8",
                "FFPOPT_ASE_FIRST": "0",
            },
            clear=False,
        ):
            os.environ.pop("FFPOPT_DEFAULTS", None)
            self.assertTrue(env_bool("FFPOPT_FAST_WAVEFRONT"))
            self.assertEqual(env_int("FFPOPT_MIN_WF_NPROC"), 8)
            self.assertFalse(env_value("FFPOPT_ASE_FIRST"))

    def test_overlay_file_then_export_wins(self):
        from ffpopt.runtime.EnvDefaults import clear_defaults_cache, env_bool, env_int

        with tempfile.TemporaryDirectory() as td:
            overlay = Path(td) / "mine.json"
            overlay.write_text(
                '// overlay\n{"FFPOPT_MIN_WF_NPROC": 4, "FFPOPT_FAST_WAVEFRONT": true}\n',
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"FFPOPT_DEFAULTS": str(overlay), "FFPOPT_FAST_WAVEFRONT": "0"},
                clear=False,
            ):
                os.environ.pop("FFPOPT_MIN_WF_NPROC", None)
                clear_defaults_cache()
                self.assertEqual(env_int("FFPOPT_MIN_WF_NPROC"), 4)
                self.assertFalse(env_bool("FFPOPT_FAST_WAVEFRONT"))

    def test_json_keys_cover_user_ffpopt_env(self):
        import ast
        import re
        from ffpopt.runtime.EnvDefaults import packaged_defaults

        allow = {
            "FFPOPT_IN_SPAWN_WORKER",
            "FFPOPT_DEFAULTS",
            "FFPOPT_IN_WALL_CHILD",
            "FFPOPT_BANNER_PRINTED",
        }
        keys = set(packaged_defaults())
        name_re = re.compile(r"^FFPOPT_[A-Z0-9_]+$")
        found: set[str] = set()
        root = Path(__file__).resolve().parents[1] / "src"
        for path in root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                if name_re.match(node.value):
                    found.add(node.value)
        missing = found - keys - allow
        self.assertEqual(
            missing,
            set(),
            f"FFPOPT_* used in code but missing from env_defaults.json: {missing}",
        )


# ---------------------------------------------------------------------------
# Logging / console contracts
# ---------------------------------------------------------------------------


@unittest.skipUnless(_have_split_ffpopt(), "split src/ffpopt not present (using independent ffpopt)")
class TestLoggingContracts(unittest.TestCase):
    def setUp(self):
        from ffpopt.runtime import Console as console_mod

        console_mod._BANNER_PRINTED = False
        os.environ.pop("FFPOPT_BANNER_PRINTED", None)

    def tearDown(self):
        from ffpopt.runtime import Console as console_mod

        console_mod._BANNER_PRINTED = False
        os.environ.pop("FFPOPT_BANNER_PRINTED", None)

    def test_ase_ignore_bad_restart_file_warning_filtered(self):
        import warnings
        from ffpopt.runtime.Console import install_ase_futurewarning_filter

        install_ase_futurewarning_filter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.warn(
                'The keyword "ignore_bad_restart_file" is deprecated and '
                "will be removed in a future version of ASE.  Passing more "
                "than one positional argument to Calculator is also deprecated.",
                FutureWarning,
                stacklevel=1,
            )
        self.assertFalse(
            any("ignore_bad_restart_file" in str(w.message) for w in caught)
        )
        self.assertIn(
            "ignore_bad_restart_file", os.environ.get("PYTHONWARNINGS", "")
        )

    def test_stale_file_handle_logging_guard(self):
        import io
        import logging
        from ffpopt.runtime.Console import (
            install_stale_handle_logging_guard,
            is_stale_file_handle,
        )

        self.assertTrue(is_stale_file_handle(OSError(116, "Stale file handle")))
        self.assertFalse(is_stale_file_handle(OSError(2, "No such file")))
        install_stale_handle_logging_guard()
        handler = logging.StreamHandler(io.StringIO())
        record = logging.LogRecord(
            "geometric", logging.INFO, __file__, 1, "step", (), None
        )
        try:
            raise OSError(116, "Stale file handle")
        except OSError:
            handler.handleError(record)

    def test_format_console_line_peels_scopes(self):
        from ffpopt.runtime.Console import format_console_line

        line = format_console_line("[frag-twist] hello", tag="ffpopt:fragment_1")
        self.assertRegex(line, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ")
        self.assertIn("[ffpopt:fragment_1]", line)
        self.assertIn("[frag-twist]", line)
        self.assertIn("hello", line)
        self.assertTrue(line.endswith("\n"))

    def test_ascii_for_stdio_maps_common_symbols(self):
        from ffpopt.runtime.Console import ascii_for_stdio, format_console_line

        self.assertEqual(ascii_for_stdio("k=500 +/-0.5 deg"), "k=500 +/-0.5 deg")
        self.assertEqual(ascii_for_stdio("k=500 \u00b10.5\u00b0"), "k=500 +/-0.5 deg")
        self.assertEqual(ascii_for_stdio("shape-match \u03c7\u00b2"), "shape-match chi^2")
        self.assertEqual(ascii_for_stdio("HL \u2192 LL"), "HL -> LL")
        mapped = ascii_for_stdio("ok")
        self.assertTrue(mapped.isascii())

        line = format_console_line("[affdo] band \u00b10.5\u00b0", tag="ffpopt")
        self.assertTrue(line.isascii())
        self.assertIn("+/-0.5 deg", line)

    def test_print_affdo_strips_non_ascii(self):
        from ffpopt.affdo.AffdoLog import print_affdo

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_affdo("soft band \u00b10.5\u00b0")
        out = buf.getvalue()
        self.assertTrue(out.isascii())
        self.assertIn("[affdo] soft band +/-0.5 deg", out)

    def test_affdo_scope_peels_on_console(self):
        from ffpopt.runtime.Console import format_console_line

        line = format_console_line("[affdo] extras: whole_ligand=True", tag="ffpopt")
        self.assertIn("[ffpopt]", line)
        self.assertIn("[affdo]", line)
        self.assertIn("extras: whole_ligand=True", line)

    def test_wavefront_scope_peels_on_console(self):
        from ffpopt.runtime.Console import format_console_line
        from ffpopt.scan.WavefrontMixins import print_wavefront

        line = format_console_line(
            "[wavefront] starting wavefront calculation", tag="ffpopt"
        )
        self.assertIn("[ffpopt]", line)
        self.assertIn("[wavefront]", line)
        self.assertIn("starting wavefront calculation", line)

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_wavefront("no checkpoint file found at /tmp/ckpt.pkl")
        out = buf.getvalue()
        self.assertTrue(out.isascii())
        self.assertIn("[wavefront] no checkpoint file found at /tmp/ckpt.pkl", out)

    def test_wavefront_node_lifecycle_logs_are_scoped(self):
        from ffpopt.scan.WavefrontMixins import mark_node_failed, print_wavefront

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_wavefront("New angle detected: 0, Energy: 5.711224")
            print_wavefront("Coalesce pending loc=350.0: better seed E=5.711224 < inf")
            print_wavefront(
                "Node 0 optimization error: BrokenGeometryError: "
                "ASE pre-opt fmax=94.5 eV/Ang exceeds 50"
            )
        out = buf.getvalue()
        self.assertTrue(out.isascii())
        for line in out.splitlines():
            self.assertTrue(line.startswith("[wavefront] "), line)

        node = SimpleNamespace(
            node_id=0,
            node_pkl="node.pckl",
            struct=SimpleNamespace(data={"elements": ["C"]}),
            angle=0,
            los=None,
        )
        buf = io.StringIO()
        with patch("ffpopt.scan.WavefrontMixins.write_node_pickle"):
            with patch("sys.stdout", buf):
                mark_node_failed(
                    node,
                    "optimization_error",
                    RuntimeError("ASE pre-opt fmax=94.5 eV/Ang exceeds 50"),
                )
        fail_out = buf.getvalue()
        self.assertIn("[wavefront] Node 0 failed at 0: optimization_error:", fail_out)

    def test_mark_node_failed_survives_pickle_io_error(self):
        import numpy as np
        from ffpopt.scan.WavefrontMixins import mark_node_failed

        node = SimpleNamespace(
            node_id=1,
            node_pkl="node.pckl",
            struct=SimpleNamespace(data={"elements": ["C"]}),
            angle=270,
            los=None,
        )
        with patch(
            "ffpopt.scan.WavefrontMixins.atomic_pickle_dump",
            side_effect=FileNotFoundError("node.pckl.tmp"),
        ):
            mark_node_failed(
                node,
                "optimization_error",
                RuntimeError("ASE pre-opt fmax=68.1 eV/Ang exceeds 50"),
            )
        self.assertTrue(node.complete)
        self.assertFalse(node.active)
        self.assertTrue(np.isinf(node.energy))
        self.assertIn("optimization_error", node.error)

    def test_atomic_pickle_dump_roundtrip(self):
        import tempfile
        from pathlib import Path
        from ffpopt.scan.WavefrontMixins import atomic_pickle_dump, pickle_load_compat

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sub" / "node.pckl"
            atomic_pickle_dump({"n": 3}, path)
            self.assertTrue(path.is_file())
            self.assertEqual(pickle_load_compat(path)["n"], 3)

    def test_wavefront_node_cleanup_missing_pickle_is_noop(self):
        import tempfile
        from ffpopt.scan.WavefrontEngine import WavefrontNode

        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "level_1_angle_110_id_2_node.pckl"
            node = SimpleNamespace(node_pkl=str(missing), node_id=2)
            WavefrontNode.cleanup(node)
            self.assertFalse(missing.exists())

    def test_attach_console_handlers_idempotent(self):
        from ffpopt.runtime.Console import attach_console_handlers

        logger = logging.getLogger("test.dev.console")
        logger.handlers.clear()
        attach_console_handlers(logger, tag="ligandparam")
        n = len(logger.handlers)
        attach_console_handlers(logger, tag="ligandparam")
        self.assertEqual(len(logger.handlers), n)
        logger.handlers.clear()

    def test_set_stream_logger_tags_ligandparam(self):
        from ligandparam.Log import set_stream_logger

        logger = set_stream_logger()
        markers = [
            getattr(h, "_lp_console_marker", None) for h in logger.handlers
        ]
        self.assertTrue(any(m and "ligandparam" in m for m in markers if m))

    def test_banner_not_from_attach(self):
        from ffpopt.runtime import Console as console_mod

        logger = logging.getLogger("test.dev.banner")
        logger.handlers.clear()
        console_mod.attach_console_handlers(logger, tag="x")
        self.assertFalse(console_mod._BANNER_PRINTED)
        logger.handlers.clear()


# ---------------------------------------------------------------------------
# AFFDO-style extras - logging helpers + pure scoring
# ---------------------------------------------------------------------------


@unittest.skipUnless(_have_split_ffpopt(), "split src/ffpopt not present (using independent ffpopt)")
class TestAffdoLogging(unittest.TestCase):
    def test_describe_affdo_extras_default_and_full(self):
        from ffpopt.affdo.AffdoLog import describe_affdo_extras

        default = describe_affdo_extras()
        self.assertIn("whole_ligand=False", default)
        self.assertIn("multi_centroid=0", default)
        self.assertIn("fit_flags=(barrier / default)", default)
        self.assertNotIn("k=", default)

        full = describe_affdo_extras(
            whole_ligand=True,
            multi_centroid=5,
            boltzmann_charges=True,
            soft_dihed_restraint=True,
            soft_dihed_k=400.0,
            soft_dihed_tol=0.25,
            fit_cli_args=["--fit-full", "--fit-backend", "jax"],
        )
        self.assertIn("whole_ligand=True", full)
        self.assertIn("multi_centroid=5", full)
        self.assertIn("boltzmann_charges=True", full)
        self.assertIn("k=400", full)
        self.assertIn("kmax=8000", full)
        self.assertIn("tol=0.25 deg", full)
        self.assertIn("--fit-full", full)
        self.assertIn("jax", full)

    def test_run_banner_is_rectangular_ascii(self):
        from ffpopt.runtime.Console import (
            format_fragmented_run_banner,
            format_whole_ligand_run_banner,
        )

        whole = format_whole_ligand_run_banner(
            ligand="CHAPS",
            model="xtb",
            nproc=44,
            delta=10,
            n_bonds=12,
            n_batches=6,
            extras="soft-dihed kmax=8000; fit-full/jax",
            work_dir="/scratch/chaps",
        )
        lines = [ln for ln in whole.splitlines() if ln]
        width = len(lines[0])
        self.assertGreaterEqual(width, 52)
        for ln in lines:
            self.assertEqual(len(ln), width, ln)
            self.assertTrue(ln.isascii())
        self.assertIn("WHOLE-LIGAND TWIST", whole)
        self.assertIn("CHAPS", whole)
        self.assertIn("WHOLE_STATUS.txt", whole)
        self.assertTrue(lines[0].startswith("+") and lines[0].endswith("+"))

        frag = format_fragmented_run_banner(
            ligand="DDM",
            model="xtb",
            nproc=56,
            n_fragments=8,
            work_dir="/tmp/ddm",
        )
        flines = [ln for ln in frag.splitlines() if ln]
        fw = len(flines[0])
        for ln in flines:
            self.assertEqual(len(ln), fw, ln)
        self.assertIn("FRAGMENTED TWIST", frag)
        self.assertIn("FRAG_STATUS.txt", frag)

    def test_whole_file_logger_identity_tag(self):
        import tempfile
        from pathlib import Path
        from ffpopt.runtime.ProgressBoard import make_whole_file_logger

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "whole-twist.log"
            logger = make_whole_file_logger("torsion_batch_00", log_path)
            logger.info("[whole-twist] hello from batch")
            for h in list(logger.handlers):
                h.flush()
                h.close()
            logger.handlers.clear()
            text = log_path.read_text(encoding="utf-8")
        self.assertIn("[ffpopt:torsion_batch_00]", text)
        self.assertIn("[whole-twist]", text)
        self.assertIn("hello from batch", text)

    def test_fit_backend_jax_falls_back_without_jax(self):
        from argparse import Namespace
        from unittest.mock import patch
        from ffpopt.dihed.ExtendedFit import apply_fit_flags_to_args, resolve_fit_backend

        with patch("ffpopt.dihed.ExtendedFit.jax_is_available", return_value=False):
            self.assertEqual(resolve_fit_backend("jax"), "lbfgsb")
            args = Namespace(
                fit_mode="full",
                fit_backend="jax",
                fit_full=True,
                barrier_only=False,
                fit_phases=False,
                fit_periods=False,
                fit_scee_scnb=False,
                opt_phase=False,
                opt_periods=False,
                opt_scee_scnb=False,
                scee=None,
                scnb=None,
            )
            apply_fit_flags_to_args(args)
            self.assertEqual(args.fit_backend, "lbfgsb")
            self.assertEqual(args.fit_mode, "full")

    def test_existing_scan_grid_mismatch(self):
        import json
        import tempfile
        from pathlib import Path
        from ffpopt.workflows.TwistHelpers import existing_scan_grid_mismatch
        from ffpopt.dihed.ExtendedFit import _lbfgsb_options

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "orig.json"
            p.write_text(json.dumps([{}] * 36))
            self.assertTrue(existing_scan_grid_mismatch(p, 15))
            self.assertFalse(existing_scan_grid_mismatch(p, 10))
            p.write_text(json.dumps({"structs": [{}] * 24}))
            self.assertFalse(existing_scan_grid_mismatch(p, 15))
            from ffpopt.workflows.TwistHelpers import (
                scan_outputs_complete,
                _promote_centroid_pick,
            )

            p.write_text(json.dumps([{}] * 36))
            self.assertFalse(scan_outputs_complete(p, 10))
            p.with_suffix(".dat").write_text("0 0 0\n", encoding="utf-8")
            self.assertTrue(scan_outputs_complete(p, 10))
            self.assertFalse(scan_outputs_complete(p, 15))
            log = logging.getLogger("test.centroid_promote")
            ok = _promote_centroid_pick(
                log,
                idx="0-1-2-3",
                hl_prefix="xtb",
                workdir=Path(td),
                candidates=[Path(td) / "xtb.c0_0-1-2-3.dat"],
            )
            self.assertFalse(ok)
        opts = _lbfgsb_options(type("A", (), {"nltol": 0.02, "nlmaxiter": 10})())
        self.assertNotIn("disp", opts)
        self.assertEqual(opts["ftol"], 0.02)

    def test_soft_dihed_skips_hard_twist_and_empty_scan_message(self):
        from types import SimpleNamespace
        from ffpopt.scan.WavefrontMixins import (
            uses_soft_dihed_restraint,
            empty_scan_error_message,
        )

        self.assertFalse(uses_soft_dihed_restraint(SimpleNamespace(args=None)))
        self.assertFalse(
            uses_soft_dihed_restraint(
                SimpleNamespace(args=SimpleNamespace(soft_dihed_restraint=False))
            )
        )
        self.assertTrue(
            uses_soft_dihed_restraint(
                SimpleNamespace(args=SimpleNamespace(soft_dihed_restraint=True))
            )
        )
        wf = SimpleNamespace(levels=[SimpleNamespace(nodes=[])])
        msg = empty_scan_error_message(wf, "xtb_0-1-2-3.json")
        self.assertIn("0 accepted scan angles", msg)
        self.assertIn("No seed nodes", msg)

    def test_soft_dihed_k_schedule_and_warm_start_hard_ic(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        from ffpopt.scan.WavefrontMixins import (
            soft_dihed_k_schedule,
            run_soft_dihed_opt,
        )

        self.assertEqual(
            soft_dihed_k_schedule(500, 8000),
            [500.0, 1000.0, 2000.0, 4000.0, 8000.0],
        )
        self.assertEqual(soft_dihed_k_schedule(500, 500), [500.0])
        self.assertEqual(soft_dihed_k_schedule(500, 600), [500.0, 600.0])

        seed_pos = np.zeros((4, 3))
        last_soft_pos = np.ones((4, 3))

        class Seed:
            def __init__(self, pos):
                self.data = {
                    "positions": np.asarray(pos, float),
                    "elements": ["C"] * 4,
                }

            def clone_geometry(self, coords=None, ene=None, frcs=None):
                return Seed(seed_pos if coords is None else coords)

            def Update(self, ene, coords, frcs):
                self.data["positions"] = np.asarray(coords, float)

        class Geom:
            def __init__(self, pos):
                self.data = {"positions": np.asarray(pos, float), "energy": 0.0}

        calls = []

        def fake_opt(los, struct, constraints=None, restraints=None, geom_prefix=None):
            pos = np.array(struct.data["positions"], dtype=float, copy=True)
            if restraints:
                k = float(restraints[0].k_kcal)
                calls.append(("soft", k, pos.copy()))
                return Geom(last_soft_pos)
            calls.append(("hard", None, pos.copy()))
            return Geom(pos)

        los = SimpleNamespace(
            args=SimpleNamespace(
                soft_dihed_k=500.0,
                soft_dihed_kmax=2000.0,
                soft_dihed_tol=0.5,
            )
        )

        with patch(
            "ffpopt.geom.Restraints.HarmonicDihedRestraint.within_tolerance",
            lambda self, crds: False,
        ), patch(
            "ffpopt.geom.Restraints.HarmonicDihedRestraint.GetCrdValue",
            lambda self, crds: 10.0,
        ):
            out = run_soft_dihed_opt(
                los,
                Seed(seed_pos),
                ["hard"],
                [0, 1, 2, 3],
                0.0,
                "x",
                node_id=1,
                opt_fn=fake_opt,
            )

        kinds = [c[0] for c in calls]
        ks = [c[1] for c in calls if c[0] == "soft"]
        self.assertEqual(ks, [500.0, 1000.0, 2000.0])
        self.assertEqual(kinds[-1], "hard")
        np.testing.assert_allclose(calls[-1][2], last_soft_pos)
        np.testing.assert_allclose(out.data["positions"], last_soft_pos)

        calls.clear()

        def fake_opt_hold(los, struct, constraints=None, restraints=None, geom_prefix=None):
            calls.append(float(restraints[0].k_kcal) if restraints else None)
            return Geom(last_soft_pos)

        with patch(
            "ffpopt.geom.Restraints.HarmonicDihedRestraint.within_tolerance",
            lambda self, crds: self.k_kcal >= 1000.0,
        ), patch(
            "ffpopt.geom.Restraints.HarmonicDihedRestraint.GetCrdValue",
            lambda self, crds: 0.1,
        ):
            run_soft_dihed_opt(
                los,
                Seed(seed_pos),
                ["hard"],
                [0, 1, 2, 3],
                0.0,
                "x",
                opt_fn=fake_opt_hold,
            )
        self.assertEqual(calls, [500.0, 1000.0])

        calls.clear()
        with patch(
            "ffpopt.geom.Restraints.HarmonicDihedRestraint.within_tolerance",
            lambda self, crds: self.k_kcal >= 1000.0,
        ), patch(
            "ffpopt.geom.Restraints.HarmonicDihedRestraint.GetCrdValue",
            lambda self, crds: 0.01,
        ):
            run_soft_dihed_opt(
                los,
                Seed(seed_pos),
                ["hard"],
                [0, 1, 2, 3],
                0.0,
                "x",
                opt_fn=fake_opt_hold,
            )
        self.assertEqual(calls, [500.0, 1000.0])

    def test_mm_then_hl_policy_and_two_stage_opts(self):
        import numpy as np
        from types import SimpleNamespace
        from unittest.mock import patch
        from ffpopt.runtime.EnvDefaults import clear_defaults_cache
        from ffpopt.runtime.FastWavefront import (
            cheap_preopt_model_name,
            mm_then_hl_enabled,
        )
        from ffpopt.scan.WavefrontMixins import (
            geomopt_mm_then_hl,
            run_soft_dihed_opt,
        )

        self.assertFalse(mm_then_hl_enabled("sander", fast=True))
        self.assertFalse(mm_then_hl_enabled("gfnff", fast=True))
        self.assertFalse(mm_then_hl_enabled("xtb", fast=False))
        self.assertTrue(mm_then_hl_enabled("xtb", fast=True))
        self.assertTrue(mm_then_hl_enabled("qdpi2", fast=True))
        with patch.dict(os.environ, {"FFPOPT_MM_THEN_HL": "0"}):
            clear_defaults_cache()
            self.assertFalse(mm_then_hl_enabled("xtb", fast=True))
        with patch.dict(os.environ, {"FFPOPT_MM_THEN_HL": "1"}):
            clear_defaults_cache()
            self.assertTrue(mm_then_hl_enabled("xtb", fast=False))
        clear_defaults_cache()

        with tempfile.NamedTemporaryFile(suffix=".parm7") as fh:
            struct = SimpleNamespace(data={"parm": fh.name})
            self.assertEqual(cheap_preopt_model_name(struct), "sander")
        import importlib.util

        noparm = cheap_preopt_model_name(
            SimpleNamespace(data={"parm": "/no/such/parm7"})
        )
        if importlib.util.find_spec("tblite") is not None:
            self.assertEqual(noparm, "gfnff")
        else:
            self.assertIsNone(noparm)

        class Seed:
            def __init__(self, pos):
                self.data = {
                    "positions": np.asarray(pos, float),
                    "elements": ["C"] * 4,
                }

            def clone_geometry(self, coords=None, ene=None, frcs=None):
                return Seed(self.data["positions"] if coords is None else coords)

            def Update(self, ene, coords, frcs):
                self.data["positions"] = np.asarray(coords, float)

        class Geom:
            def __init__(self, pos):
                self.data = {"positions": np.asarray(pos, float), "energy": 0.0}

        calls = []

        def fake_opt(los, struct, constraints=None, restraints=None, geom_prefix=None):
            model = getattr(los.args, "model", "?")
            pos = np.array(struct.data["positions"], dtype=float, copy=True)
            kind = "soft" if restraints else ("hard" if constraints else "free")
            calls.append((model, kind, float(pos.reshape(-1)[0])))
            shift = 1.0 if model == "sander" else 10.0
            return Geom(pos + shift)

        hl = SimpleNamespace(args=SimpleNamespace(model="xtb"))
        cheap = SimpleNamespace(args=SimpleNamespace(model="sander"))
        out = geomopt_mm_then_hl(
            hl,
            Seed(np.zeros((4, 3))),
            constraints=["c"],
            opt_fn=fake_opt,
            cheap_los=cheap,
            node_id=1,
        )
        self.assertEqual([c[0] for c in calls], ["sander", "xtb"])
        self.assertEqual([c[1] for c in calls], ["hard", "hard"])
        self.assertAlmostEqual(calls[1][2], 1.0)
        np.testing.assert_allclose(out.data["positions"], np.full((4, 3), 11.0))

        calls.clear()
        los = SimpleNamespace(
            args=SimpleNamespace(
                model="xtb",
                soft_dihed_k=500.0,
                soft_dihed_kmax=500.0,
                soft_dihed_tol=0.5,
            )
        )
        with patch(
            "ffpopt.geom.Restraints.HarmonicDihedRestraint.within_tolerance",
            lambda self, crds: True,
        ), patch(
            "ffpopt.geom.Restraints.HarmonicDihedRestraint.GetCrdValue",
            lambda self, crds: 0.01,
        ):
            run_soft_dihed_opt(
                los,
                Seed(np.zeros((4, 3))),
                ["hard"],
                [0, 1, 2, 3],
                0.0,
                "x",
                opt_fn=fake_opt,
                cheap_los=cheap,
            )
        self.assertEqual(calls, [("sander", "soft", 0.0), ("xtb", "soft", 1.0)])

    def test_format_boltzmann_summary(self):
        from ffpopt.affdo.AffdoLog import format_boltzmann_summary

        lines = format_boltzmann_summary(
            {
                "out_mol2": "lig.boltz.mol2",
                "out_lib": "lig.boltz.lib",
                "weights": [0.7, 0.3],
                "T": 298.15,
                "n_atom": 12,
                "equal_weights": False,
                "rms_vs_first": 0.01234,
                "max_abs_dq": 0.05,
            }
        )
        text = "\n".join(lines)
        self.assertIn("12 atom charges", text)
        self.assertIn("0.7000 0.3000", text)
        self.assertIn("lig.boltz.mol2", text)
        self.assertIn("lig.boltz.lib", text)
        self.assertNotIn("equal weights", text)

        eq = "\n".join(format_boltzmann_summary({"weights": [0.5, 0.5], "equal_weights": True}))
        self.assertIn("equal weights", eq)

    def test_print_affdo_stdout(self):
        from ffpopt.affdo.AffdoLog import print_affdo

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_affdo("soft restraint on")
        self.assertEqual(buf.getvalue(), "[affdo] soft restraint on\n")

    def test_struct_update_rejects_atom_count_mismatch(self):
        from ffpopt.Struct import Struct

        s = Struct(
            {
                "name": "frag",
                "positions": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
                "elements": ["C", "C"],
                "charges": [0.0, 0.0],
                "names": ["C1", "C2"],
                "types": ["c", "c"],
                "bonds": [[0, 1]],
                "spin": 1,
                "energy": None,
                "forces": None,
            }
        )
        with self.assertRaises(ValueError) as ctx:
            s.Update(None, [[0.0, 0.0, 0.0]] * 3, None)
        self.assertIn("3 coordinates onto 2 atoms", str(ctx.exception))

    def test_centroid_parent_mol2_does_not_overwrite_fragment(self):
        from ffpopt.Struct import ListOfStruct, Struct
        from ffpopt.affdo.CentroidProfiles import generate_centroid_start_jsons

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            data = {
                "name": "frag",
                "positions": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
                "elements": ["C", "C"],
                "charges": [0.0, 0.0],
                "names": ["C1", "C2"],
                "types": ["c", "c"],
                "bonds": [[0, 1]],
                "spin": 1,
                "energy": None,
                "forces": None,
                "constraints": [],
                "restraints": [],
            }
            start = td / "start.json"
            ListOfStruct([Struct(data)]).save(str(start))
            (td / "start.cent0.json").write_text(
                json.dumps([{**data, "positions": [[0.0, 0.0, 0.0]] * 3}]),
                encoding="utf-8",
            )
            mol2 = td / "parent.mol2"
            mol2.write_text(
                "@<TRIPOS>MOLECULE\nparent\n3 0 0 0 0\nSMALL\nUSER_CHARGES\n\n"
                "@<TRIPOS>ATOM\n"
                "1 C1 0.0 0.0 0.0 C 1 MOL 0.0\n"
                "2 C2 1.5 0.0 0.0 C 1 MOL 0.0\n"
                "3 C3 3.0 0.0 0.0 C 1 MOL 0.0\n"
                "@<TRIPOS>BOND\n",
                encoding="utf-8",
            )
            paths = generate_centroid_start_jsons(
                start, mol2_path=mol2, nkeep=2, workdir=td
            )
            self.assertEqual(len(paths), 1)
            los = ListOfStruct.from_file(str(paths[0]))
            self.assertEqual(len(los.structs[0].data["elements"]), 2)
            self.assertEqual(len(los.structs[0].data["positions"]), 2)

    def test_pick_smoothest_profile_logs_details(self):
        from ffpopt.affdo.CentroidProfiles import pick_smoothest_profile, score_profile_details

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            smooth = root / "c0.dat"
            jagged = root / "c1.dat"
            missing = root / "c2.dat"
            # Smooth cosine-like scan.
            smooth.write_text(
                "\n".join(f"{a} {0.5 * (1.0 - __import__('math').cos(__import__('math').radians(a)))} 0"
                          for a in range(-180, 181, 30)),
                encoding="utf-8",
            )
            jagged.write_text(
                "\n".join(
                    f"{a} {(0.0 if i % 2 == 0 else 8.0)} 0"
                    for i, a in enumerate(range(-180, 181, 30))
                ),
                encoding="utf-8",
            )
            best, score, rows = pick_smoothest_profile([smooth, jagged, missing])
            self.assertEqual(Path(best), smooth)
            self.assertTrue(score < rows[1]["score"] or Path(rows[0]["path"]) == smooth)
            names = {Path(r["path"]).name: r for r in rows}
            self.assertEqual(names["c2.dat"]["error"], "missing")
            self.assertIsNone(names["c0.dat"]["error"])
            d0 = score_profile_details(smooth)
            self.assertGreater(d0["npts"], 3)
            self.assertTrue(__import__("math").isfinite(d0["fourier"]))

    def test_profile_is_smooth_enough_uses_fourier(self):
        import math
        from ffpopt.affdo.CentroidProfiles import (
            profile_is_smooth_enough,
            score_profile_details,
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            smooth = root / "smooth.dat"
            jagged = root / "jagged.dat"
            barrier = root / "barrier.dat"
            angs = list(range(-180, 181, 30))
            smooth.write_text(
                "\n".join(
                    f"{a} {0.5 * (1.0 - math.cos(math.radians(a)))} 0" for a in angs
                ),
                encoding="utf-8",
            )
            jagged.write_text(
                "\n".join(
                    f"{a} {(0.0 if i % 2 == 0 else 8.0)} 0"
                    for i, a in enumerate(angs)
                ),
                encoding="utf-8",
            )
            barrier.write_text(
                "\n".join(
                    f"{a} {3.0 * (1.0 - math.cos(math.radians(2 * a)))} 0" for a in angs
                ),
                encoding="utf-8",
            )
            self.assertTrue(profile_is_smooth_enough(score_profile_details(smooth), fourier_max=0.5))
            self.assertTrue(profile_is_smooth_enough(score_profile_details(barrier), fourier_max=0.5))
            self.assertFalse(profile_is_smooth_enough(score_profile_details(jagged), fourier_max=0.5))
            self.assertFalse(profile_is_smooth_enough(score_profile_details(smooth), fourier_max=None))

    def test_hl_orig_and_centroid_jobs_share_pools(self):
        from ffpopt.workflows import TwistHelpers as th

        class _Scan:
            def __init__(self, idxs):
                self.idxs = list(idxs)

            def GetIdxStr(self):
                return "-".join(str(i) for i in self.idxs)

        scans = [_Scan([0, 1, 2, 3]), _Scan([4, 5, 6, 7])]
        captured = []

        def _fake_execute(jobs, **kwargs):
            captured.append({"label": kwargs.get("label"), "jobs": list(jobs)})
            return [
                (j["prefix"], tuple(j["dihed_idxs"]), None) for j in jobs
            ]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            starts = [root / "c0.json", root / "c1.json", root / "c2.json"]
            for p in starts:
                p.write_text("{}", encoding="utf-8")
            angs = list(range(-180, 181, 30))
            import math

            def _write_dat(name, jagged=False):
                path = root / name
                if jagged:
                    text = "\n".join(
                        f"{a} {(0.0 if i % 2 == 0 else 8.0)} 0"
                        for i, a in enumerate(angs)
                    )
                else:
                    text = "\n".join(
                        f"{a} {0.5 * (1.0 - math.cos(math.radians(a)))} 0" for a in angs
                    )
                path.write_text(text, encoding="utf-8")

            _write_dat("xtb.c0_0-1-2-3.dat", jagged=False)
            _write_dat("xtb.c0_4-5-6-7.dat", jagged=True)

            with patch.object(th, "_execute_bond_scan_jobs", side_effect=_fake_execute), patch(
                "ffpopt.affdo.CentroidProfiles.generate_centroid_start_jsons",
                return_value=starts,
            ):
                th._run_hl_and_orig_scans(
                    scans,
                    hl_prefix="xtb",
                    hl_model="xtb",
                    inp=str(root / "start.json"),
                    nproc=8,
                    skip_existing=True,
                    workdir=root,
                    logger=None,
                    wf_kwargs={},
                    multi_centroid=3,
                )
            self.assertTrue((root / "xtb_0-1-2-3.dat").is_file())
            self.assertTrue((root / "xtb_4-5-6-7.dat").is_file())

        self.assertEqual(len(captured), 2)
        first_prefixes = {j["prefix"] for j in captured[0]["jobs"]}
        self.assertEqual(first_prefixes, {"xtb.c0", "orig"})
        self.assertEqual(len(captured[0]["jobs"]), 4)
        extra_prefixes = {j["prefix"] for j in captured[1]["jobs"]}
        self.assertEqual(extra_prefixes, {"xtb.c1", "xtb.c2"})
        extra_idxs = {tuple(j["dihed_idxs"]) for j in captured[1]["jobs"]}
        self.assertEqual(extra_idxs, {(4, 5, 6, 7)})

    def test_interleave_job_groups(self):
        from ffpopt.workflows.TwistHelpers import _interleave_job_groups

        a = [{"id": "a0"}, {"id": "a1"}]
        b = [{"id": "b0"}, {"id": "b1"}, {"id": "b2"}]
        ids = [j["id"] for j in _interleave_job_groups(a, b)]
        self.assertEqual(ids, ["a0", "b0", "a1", "b1", "b2"])
        self.assertEqual(_interleave_job_groups([], a), a)

    def test_write_prefix_comparison_plot_waits_for_both_profiles(self):
        from unittest.mock import patch

        from ffpopt.workflows import TwistHelpers as th

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "orig_0-1-2-3.dat").write_text("0 0 0\n10 1 1\n", encoding="utf-8")
            self.assertIsNone(
                th._write_prefix_comparison_plot(
                    hl_prefix="xtb",
                    ll_prefix="orig",
                    idx="0-1-2-3",
                    workdir=root,
                    plot_dir=root,
                )
            )
            (root / "xtb_0-1-2-3.dat").write_text("0 0 0\n10 1 1\n", encoding="utf-8")
            fake = type(
                "R",
                (),
                {
                    "is_flat": False,
                    "is_close": False,
                    "barrier_hl": 2.0,
                    "barrier_ll": 1.0,
                },
            )()
            with patch(
                "ffpopt.scan.ScanAnalysis.compare_scan_files", return_value=fake
            ) as mocked:
                out = th._write_prefix_comparison_plot(
                    hl_prefix="xtb",
                    ll_prefix="orig",
                    idx="0-1-2-3",
                    workdir=root,
                    plot_dir=root,
                )
            self.assertIs(out, fake)
            self.assertEqual(
                Path(mocked.call_args.kwargs["plot_path"]).name,
                "compare_xtb_vs_orig_0-1-2-3.png",
            )

    def test_whole_ligand_defaults_plot_comparisons(self):
        import inspect

        from alps.workflows.WholeLigandTwist import (
            run_whole_ligand_dihed_twist_workflow,
        )

        param = inspect.signature(
            run_whole_ligand_dihed_twist_workflow
        ).parameters["plot_comparisons"]
        self.assertIs(param.default, True)

    def test_bond_scan_job_isolates_empty_wavefront(self):
        from unittest.mock import patch

        from ffpopt.workflows import TwistHelpers as th

        job = {
            "prefix": "xtb.c0",
            "inp": "start.json",
            "model": "xtb",
            "dihed_idxs": [2, 3, 4, 5],
            "out": "xtb.c0_2-3-4-5.json",
            "skip_existing": False,
            "workdir": None,
            "wf_kwargs": {},
        }
        with patch.object(
            th,
            "_run_one_scan",
            side_effect=RuntimeError("wavefront produced 0 accepted scan angles"),
        ):
            out = th._run_bond_scan_job(job)
        self.assertIsNone(out["result"])
        self.assertIn("0 accepted", out["error"])
        self.assertEqual(out["prefix"], "xtb.c0")

    def test_mm_preopt_los_uses_geometric(self):
        from types import SimpleNamespace

        from ffpopt.scan.WavefrontMixins import _copy_los_args_with_model

        args = _copy_los_args_with_model(
            SimpleNamespace(model="xtb", geometric_opt=False, no_opt=False),
            "sander",
        )
        self.assertEqual(args.model, "sander")
        self.assertTrue(args.geometric_opt)

    def test_boltzmann_average_summary_fields(self):
        from ffpopt.affdo.BoltzmannCharges import boltzmann_average_mol2_charges

        def _mol2(charges):
            lines = [
                "@<TRIPOS>MOLECULE\n",
                "LIG\n",
                "@<TRIPOS>ATOM\n",
            ]
            for i, q in enumerate(charges, start=1):
                name = f"C{i}"
                lines.append(
                    f"{i:7d} {name:<8s} {0.0:10.4f} {0.0:10.4f} {0.0:10.4f} "
                    f"{'C.3':<6s} {'1':>4s} {'LIG':<6s} {q:10.6f}\n"
                )
            lines.append("@<TRIPOS>BOND\n")
            return "".join(lines)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = root / "c0.mol2"
            b = root / "c1.mol2"
            a.write_text(_mol2([0.1, -0.1]), encoding="utf-8")
            b.write_text(_mol2([0.3, -0.3]), encoding="utf-8")
            out = root / "avg.mol2"
            info = boltzmann_average_mol2_charges([a, b], [0.0, 0.0], out)
            self.assertTrue(info["equal_weights"])
            self.assertEqual(info["n_atom"], 2)
            self.assertEqual(info["n_conf"], 2)
            self.assertAlmostEqual(info["charges"][0], 0.2, places=6)
            self.assertGreater(info["rms_vs_first"], 0.0)

    def test_format_extended_params(self):
        from ffpopt.dihed.ExtendedFit import format_extended_params

        prim = SimpleNamespace(fc=1.25, phase=180.0, per=2)
        finp = SimpleNamespace(
            ptypedict={"ca-c3-c-o": SimpleNamespace(dfcns=SimpleNamespace(prims=[prim]))},
            opt_phase=True,
            opt_periods=True,
            opt_scee_scnb=True,
            scee=1.2,
            scnb=2.0,
        )
        text = "\n".join(format_extended_params(finp))
        self.assertIn("ca-c3-c-o", text)
        self.assertIn("FC=[1.2500]", text)
        self.assertIn("phase_deg=[180.00]", text)
        self.assertIn("period=[2]", text)
        self.assertIn("scee=1.2000", text)


# ---------------------------------------------------------------------------
# Dihed option helpers + workflow bond contracts
# ---------------------------------------------------------------------------


class TestDihedOptionsAndBonds(unittest.TestCase):

    def test_normalize_bond_pairs0(self):
        if not _have_split_ffpopt():
            self.skipTest("normalize_bond_pairs0 lived in split ffpopt.workflows")
        from ffpopt.workflows import normalize_bond_pairs0

        self.assertEqual(normalize_bond_pairs0([(1, 2), [3, 4]]), [(1, 2), (3, 4)])
        self.assertEqual(normalize_bond_pairs0(["0,1", "10,11"]), [(0, 1), (10, 11)])
        with self.assertRaises(ValueError):
            normalize_bond_pairs0(["0-1"])
        with self.assertRaises(TypeError):
            normalize_bond_pairs0([42])

    def test_bonds0_from_scission_fit_torsions(self):
        from alps.workflows import bonds0_from_scission_fit_torsions

        pairs = bonds0_from_scission_fit_torsions(
            [
                {"fragment_rotatable_bond": [1, 2]},
                {"fragment_rotatable_bond": [5, 8]},
            ]
        )
        self.assertEqual(pairs, [(0, 1), (4, 7)])


# ---------------------------------------------------------------------------
# ffpopt - constraints, dihedrals, wavefront policy, runtime helpers
# ---------------------------------------------------------------------------


# ---
# Dihedral Fourier / ParmEd / scan alignment
# ---

@unittest.skipUnless(_have_split_ffpopt(), "split src/ffpopt not present (using independent ffpopt)")
class TestDihedFitHelpers(unittest.TestCase):
    def test_merge_duplicate_period_prims(self):
        from ffpopt.dihed.Dihedrals import (
            PrimDihedFcn,
            merge_duplicate_period_prims,
            parmed_dihedral_type_list_from_prims,
        )

        same_phase = merge_duplicate_period_prims(
            [PrimDihedFcn(1.5, 0.0, 2), PrimDihedFcn(0.5, 0.0, 2.4)],
            warn=False,
        )
        self.assertEqual(len(same_phase), 1)
        self.assertEqual(same_phase[0].per, 2)
        self.assertEqual(same_phase[0].phase, 0.0)
        self.assertAlmostEqual(same_phase[0].fc, 2.0)

        opposite = merge_duplicate_period_prims(
            [PrimDihedFcn(2.0, 0.0, 3), PrimDihedFcn(0.5, 180.0, 3)],
            warn=False,
        )
        self.assertEqual(len(opposite), 1)
        self.assertEqual(opposite[0].per, 3)
        self.assertEqual(opposite[0].phase, 0.0)
        self.assertAlmostEqual(opposite[0].fc, 1.5)

        distinct = merge_duplicate_period_prims(
            [PrimDihedFcn(1.0, 0.0, 1), PrimDihedFcn(2.0, 180.0, 2)],
            warn=False,
        )
        self.assertEqual([p.per for p in distinct], [1, 2])

        try:
            import parmed  # noqa: F401
        except ImportError:
            return
        typs = parmed_dihedral_type_list_from_prims(
            [PrimDihedFcn(1.0, 0.0, 2), PrimDihedFcn(0.25, 180.0, 2)],
            warn=False,
        )
        self.assertEqual(len(typs), 1)
        self.assertEqual(int(typs[0].per), 2)

    def test_write_parmed_script_merges_duplicate_periods(self):
        from types import SimpleNamespace
        from ffpopt.dihed.DihedFourier import PrimDihedFcn
        from ffpopt.dihed.DihedParmEd import WriteParmedScript

        atoms = [
            SimpleNamespace(name=n, residue=SimpleNamespace(name="LIG"))
            for n in ("C1", "C2", "C3", "C4")
        ]
        p = SimpleNamespace(atoms=atoms)
        dfcn = SimpleNamespace(
            idxs=[0, 1, 2, 3],
            prims=[PrimDihedFcn(1.0, 0.0, 2), PrimDihedFcn(0.5, 180.0, 2)],
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "it01.py"
            WriteParmedScript(str(path), p, [dfcn], scee=1.2, scnb=2.0)
            text = path.read_text(encoding="utf-8")
            self.assertFalse((path.parent / "it01.py.tmp").exists())
        self.assertIn("addDihedral", text)
        self.assertEqual(text.count("addDihedral("), 1)
        self.assertIn("0.5", text)
        self.assertIn("p.save(", text)

    def test_gendihedfit_outputs_complete_rejects_truncated_py(self):
        from ffpopt.workflows.TwistHelpers import gendihedfit_outputs_complete

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            py_path = td / "it01.py"
            frcmod = td / "it01.frcmod"
            py_path.write_text("#!/usr/bin/env python3\nprint('truncated')\n", encoding="utf-8")
            frcmod.write_text("FORCE_FIELD_TYPE PARM99\n", encoding="utf-8")
            self.assertFalse(gendihedfit_outputs_complete(py_path, frcmod))
            py_path.write_text(
                "p = load_file(args.iparm)\np.save(args.oparm,overwrite=True)\n",
                encoding="utf-8",
            )
            self.assertTrue(gendihedfit_outputs_complete(py_path, frcmod))
            frcmod.unlink()
            self.assertFalse(gendihedfit_outputs_complete(py_path, frcmod))

    def test_run_fit_script_inprocess_raises_if_parm7_missing(self):
        from ffpopt.workflows.TwistHelpers import _run_fit_script_inprocess

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            script = td / "it01.py"
            iparm = td / "orig.parm7"
            oparm = td / "it01.parm7"
            script.write_text("print('no p.save')\n", encoding="utf-8")
            iparm.write_text("placeholder\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError) as ctx:
                _run_fit_script_inprocess(script, iparm, oparm)
            self.assertIn("without writing", str(ctx.exception))
            self.assertFalse(oparm.exists())

    def test_dihed_facade_exports(self):
        from ffpopt.dihed import DihedFitSolve, DihedFourier, DihedParmEd
        from ffpopt.dihed.Dihedrals import (
            ChangeDihedrals,
            DeleteDihedrals,
            FindPuckers,
            FitInputType,
            IsolatedLinearSolve,
            MultiDihedFcn,
            NonlinearSolve,
            PrimDihedFcn,
            WriteParmedScript,
        )

        self.assertIs(PrimDihedFcn, DihedFourier.PrimDihedFcn)
        self.assertIs(MultiDihedFcn, DihedFourier.MultiDihedFcn)
        self.assertIs(DeleteDihedrals, DihedParmEd.DeleteDihedrals)
        self.assertIs(ChangeDihedrals, DihedParmEd.ChangeDihedrals)
        self.assertIs(WriteParmedScript, DihedParmEd.WriteParmedScript)
        self.assertIs(IsolatedLinearSolve, DihedFitSolve.IsolatedLinearSolve)
        self.assertIs(NonlinearSolve, DihedFitSolve.NonlinearSolve)
        self.assertTrue(isinstance(FitInputType, type))
        self.assertTrue(callable(FindPuckers))

        from ffpopt.dihed import DihedFitTypes, DihedMath
        self.assertIs(DihedFitTypes.WriteParmedScript, DihedParmEd.WriteParmedScript)
        self.assertIs(DihedFitTypes.ChangeDihedrals, DihedParmEd.ChangeDihedrals)
        self.assertIs(DihedFitSolve.shape_match_delta, DihedMath.shape_match_delta)

    def test_system_write_output_calls_parmed_script(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from ffpopt.dihed.DihedFitTypes import SystemType

        dfcn = SimpleNamespace(idxs=None)
        pinst = SimpleNamespace(
            ptype=SimpleNamespace(dfcns=dfcn),
            dihedidxs=[[0, 1, 2, 3]],
        )
        system = SystemType.__new__(SystemType)
        system.output = "it01.py"
        system.mol = object()
        system._fit_owner = SimpleNamespace(scee=1.2, scnb=2.0)
        system.pinstances = [pinst]
        with patch("ffpopt.dihed.DihedFitTypes.WriteParmedScript") as writer:
            system.write_output()
        writer.assert_called_once()
        self.assertEqual(writer.call_args.args[0], "it01.py")

    def test_clip_dihed_fcs_caps_unbounded_lstsq(self):
        import numpy as np
        from ffpopt.dihed.DihedFitSolve import clip_dihed_fcs, dihed_fc_abs_max

        cap = dihed_fc_abs_max()
        self.assertEqual(cap, 25.0)
        out = clip_dihed_fcs([6439.0, -1278.0, 1.4], where="test")
        np.testing.assert_allclose(out, [cap, -cap, 1.4])

    def test_fourier_ridge_beats_cancelling_harmonics(self):
        import numpy as np
        from ffpopt.dihed.DihedFitRegularize import tikhonov_svd_solve

        # Clustered scan + tiny singular modes: naive pinv (rcond=0) explodes.
        phi = np.deg2rad(np.array([0.0, 2.0, 4.0, 6.0, 8.0]))
        A = np.column_stack([np.cos(n * phi) for n in (1, 2, 3)])
        A = A - np.mean(A, axis=0, keepdims=True)
        y = 0.3 * A[:, 0] + 0.02 * np.array([0.0, 1.0, -1.0, 0.4, -0.3])
        y = y - np.mean(y)
        x_raw = np.linalg.pinv(A, rcond=0.0) @ y
        x_r, info = tikhonov_svd_solve(A, y, lam=0.0, rel_cutoff=1.0e-4)
        self.assertGreater(
            float(np.linalg.norm(x_raw)),
            10.0 * float(np.linalg.norm(x_r)) + 1.0,
        )
        self.assertGreaterEqual(info["n_kept"], 1)

    def test_fourier_nprim_aic_picks_single_harmonic(self):
        import numpy as np
        from ffpopt.dihed.DihedFitRegularize import fit_fourier_nprim

        angs = np.linspace(0.0, 350.0, 36)
        y = 1.5 * np.cos(np.deg2rad(angs))
        y = y - np.mean(y)
        with patch.dict(os.environ, {"FFPOPT_DIHED_NPRIM_SELECT": "1"}, clear=False):
            dfcn, x, info = fit_fourier_nprim(
                angs, y, 3, [0, 1, 2, 3], pname="test-n1"
            )
        self.assertEqual(info["nprim"], 1)
        self.assertEqual(len(dfcn.prims), 1)
        self.assertGreater(float(x[0]), 0.5)

    def test_fourier_energy_domain_barrier_scales_vphi(self):
        import numpy as np
        from ffpopt.dihed.DihedFitRegularize import (
            dense_torsion_ptp,
            solve_regularized_fcs,
        )
        from ffpopt.dihed.DihedFourier import GetDihedClasses

        dfcn = GetDihedClasses(idxs=[0, 1, 2, 3])[1][0]
        A = np.array([[1.0], [-1.0], [0.5], [-0.5]])
        y = np.array([80.0, -80.0, 40.0, -40.0])
        with patch.dict(
            os.environ,
            {
                "FFPOPT_DIHED_BARRIER_ABS": "30",
                "FFPOPT_DIHED_BARRIER_ALPHA": "1",
                "FFPOPT_DIHED_RIDGE_LAMBDA": "0",
            },
            clear=False,
        ):
            x, info = solve_regularized_fcs(A, y, dfcn=dfcn, where="test-barrier")
        self.assertLessEqual(dense_torsion_ptp(dfcn), 30.0 * 1.05)
        self.assertLessEqual(float(np.max(np.abs(x))), 25.0)
        self.assertIsNotNone(info.get("dense_ptp"))

    def test_sp3_rotor_policy_zero_and_alkane_cap(self):
        from ffpopt.dihed.DihedFitRegularize import (
            apply_chemical_rotor_policy,
            apply_sp3_rotor_policy,
            classify_dihed_rotor,
            dense_torsion_ptp,
            parse_dihed_type_key,
        )
        from ffpopt.dihed.DihedFourier import GetDihedClasses

        self.assertIs(apply_chemical_rotor_policy, apply_sp3_rotor_policy)
        self.assertEqual(
            parse_dihed_type_key("c -ns-c3-c3"), ("c", "ns", "c3", "c3")
        )
        self.assertEqual(classify_dihed_rotor("c3-c3-c3-c3"), "alkane")
        self.assertEqual(classify_dihed_rotor("hc-c3-c3-hc"), "alkane")
        self.assertEqual(classify_dihed_rotor("c3-c3-s6-o"), "sulfate_phosphate")
        self.assertEqual(classify_dihed_rotor("h1-c3-s6-o"), "sulfate_phosphate")
        self.assertEqual(classify_dihed_rotor("c3-c3-n4-c3"), "amine_ammonium")
        self.assertEqual(classify_dihed_rotor("h1-c3-oh-ho"), "alcohol_ether")
        self.assertEqual(classify_dihed_rotor("c3-c3-ss-c3"), "polar_sp3")
        self.assertEqual(classify_dihed_rotor("c3-c3-c3-oh"), "sp3_sp3")
        self.assertEqual(classify_dihed_rotor("o -c -ns-c3"), "unsaturated")
        # parmchk2 sugar/detergent carbon (DDM): analog of c3, not unsaturated.
        self.assertEqual(classify_dihed_rotor("h1-c3-c6-c6"), "alkane")
        self.assertEqual(classify_dihed_rotor("h1-c6-c6-h1"), "alkane")
        self.assertEqual(classify_dihed_rotor("oh-c3-c6-os"), "sp3_sp3")
        self.assertEqual(classify_dihed_rotor("oh-c3-c6-h1"), "sp3_sp3")
        self.assertEqual(classify_dihed_rotor("c6-c6-os-c3"), "alcohol_ether")
        self.assertEqual(classify_dihed_rotor("os-c6-os-c3"), "alcohol_ether")
        self.assertEqual(classify_dihed_rotor("h2-c6-os-c3"), "alcohol_ether")
        self.assertEqual(classify_dihed_rotor("h1-c6-os-c6"), "alcohol_ether")
        # Fit-input keys are "{res}_{types}"; the residue must not be a 5th type.
        self.assertEqual(
            parse_dihed_type_key("CHA_c3-c3-c3-h1"), ("c3", "c3", "c3", "h1")
        )
        self.assertEqual(classify_dihed_rotor("CHA_c3-c3-c3-c3"), "alkane")
        self.assertEqual(classify_dihed_rotor("CHA_c3-c3-s6-o"), "sulfate_phosphate")
        self.assertEqual(classify_dihed_rotor("CHA_c3-c3-n4-c3"), "amine_ammonium")
        self.assertEqual(classify_dihed_rotor("CHA_o-c-ns-c3"), "unsaturated")

        policy_env = {
            "FFPOPT_DIHED_SP3_BARRIER_MAX": "20",
            "FFPOPT_DIHED_ALKANE_BARRIER_MAX": "5",
            "FFPOPT_DIHED_POLAR_SP3_BARRIER_MAX": "8",
            "FFPOPT_DIHED_SULFATE_BARRIER_MAX": "10",
            "FFPOPT_DIHED_SULFATE_BARRIER_CAP": "4",
        }

        stiff = GetDihedClasses(idxs=[0, 1, 2, 3])[1][0]
        stiff.SetFCs([15.4])
        with patch.dict(os.environ, policy_env, clear=False):
            out, action, ptp = apply_sp3_rotor_policy(
                stiff, "CHA_c3-c3-s6-o", where="test"
            )
        self.assertEqual(action, "zero_sulfate_phosphate")
        self.assertAlmostEqual(float(out.prims[0].fc), 0.0)
        self.assertGreater(ptp, 10.0)

        alk = GetDihedClasses(idxs=[0, 1, 2, 3])[1][0]
        alk.SetFCs([6.0])
        with patch.dict(os.environ, policy_env, clear=False):
            out, action, _ptp = apply_sp3_rotor_policy(
                alk, "c3-c3-c3-c3", where="test"
            )
        self.assertEqual(action, "cap_alkane")
        self.assertLessEqual(dense_torsion_ptp(out), 5.0 * 1.05)

        amine = GetDihedClasses(idxs=[0, 1, 2, 3])[1][0]
        amine.SetFCs([6.0])
        with patch.dict(os.environ, policy_env, clear=False):
            out, action, _ptp = apply_sp3_rotor_policy(
                amine, "c3-c3-n4-c3", where="test"
            )
        self.assertEqual(action, "cap_amine_ammonium")
        self.assertLessEqual(dense_torsion_ptp(out), 8.0 * 1.05)

        sul_cap = GetDihedClasses(idxs=[0, 1, 2, 3])[1][0]
        sul_cap.SetFCs([3.5])
        with patch.dict(os.environ, policy_env, clear=False):
            out, action, _ptp = apply_sp3_rotor_policy(
                sul_cap, "c3-c3-s6-o", where="test"
            )
        self.assertEqual(action, "cap_sulfate_phosphate")
        self.assertLessEqual(dense_torsion_ptp(out), 4.0 * 1.05)

        generic = GetDihedClasses(idxs=[0, 1, 2, 3])[1][0]
        generic.SetFCs([6.0])
        with patch.dict(os.environ, policy_env, clear=False):
            out, action, _ptp = apply_sp3_rotor_policy(
                generic, "c3-c3-c3-oh", where="test"
            )
        self.assertEqual(action, "keep")
        self.assertAlmostEqual(float(out.prims[0].fc), 6.0)

        sugar_h = GetDihedClasses(idxs=[0, 1, 2, 3])[1][0]
        sugar_h.SetFCs([16.3])
        with patch.dict(os.environ, policy_env, clear=False):
            out, action, ptp = apply_sp3_rotor_policy(
                sugar_h, "h1-c3-c6-c6", where="test"
            )
        self.assertEqual(action, "zero_alkane")
        self.assertAlmostEqual(float(out.prims[0].fc), 0.0)
        self.assertGreater(ptp, 20.0)

        sugar_alk = GetDihedClasses(idxs=[0, 1, 2, 3])[1][0]
        sugar_alk.SetFCs([6.0])
        with patch.dict(os.environ, policy_env, clear=False):
            out, action, _ptp = apply_sp3_rotor_policy(
                sugar_alk, "h1-c3-c6-c6", where="test"
            )
        self.assertEqual(action, "cap_alkane")
        self.assertLessEqual(dense_torsion_ptp(out), 5.0 * 1.05)

        sugar_cc = GetDihedClasses(idxs=[0, 1, 2, 3])[1][0]
        sugar_cc.SetFCs([15.3])
        with patch.dict(os.environ, policy_env, clear=False):
            out, action, ptp = apply_sp3_rotor_policy(
                sugar_cc, "oh-c3-c6-os", where="test"
            )
        self.assertEqual(action, "zero_sp3_sp3")
        self.assertAlmostEqual(float(out.prims[0].fc), 0.0)
        self.assertGreater(ptp, 20.0)

        sugar_os = GetDihedClasses(idxs=[0, 1, 2, 3])[1][0]
        sugar_os.SetFCs([6.0])
        with patch.dict(os.environ, policy_env, clear=False):
            out, action, _ptp = apply_sp3_rotor_policy(
                sugar_os, "c6-c6-os-c3", where="test"
            )
        self.assertEqual(action, "cap_alcohol_ether")
        self.assertLessEqual(dense_torsion_ptp(out), 8.0 * 1.05)

        amide = GetDihedClasses(idxs=[0, 1, 2, 3])[1][0]
        amide.SetFCs([15.0])
        with patch.dict(os.environ, policy_env, clear=False):
            out, action, _ptp = apply_sp3_rotor_policy(
                amide, "o-c-ns-c3", where="test"
            )
        self.assertEqual(action, "keep")
        self.assertAlmostEqual(float(out.prims[0].fc), 15.0)

    def test_dihed_math_reexported_and_ipc_slim(self):
        from ffpopt.dihed import DihedMath as dihed_math
        from ffpopt.dihed.Dihedrals import shape_match_delta
        from ffpopt.runtime.SlimIpc import slim_scan_result, slim_twist_result

        self.assertIs(shape_match_delta, dihed_math.shape_match_delta)
        self.assertIsNone(slim_scan_result(None))
        self.assertEqual(slim_scan_result({"a": 1, "wf_run": object()})["a"], 1)
        slim_one = slim_scan_result(
            {"a": 1, "wf_run": object(), "structures": object()}
        )
        self.assertEqual(slim_one["a"], 1)
        self.assertNotIn("wf_run", slim_one)
        self.assertNotIn("structures", slim_one)
        slim = slim_twist_result(
            {
                "ok": True,
                "scans": [
                    (
                        "p",
                        (0, 1, 2, 3),
                        {"e": 1.0, "wf_run": object(), "structures": object()},
                    )
                ],
            }
        )
        self.assertNotIn("wf_run", slim["scans"][0][2])
        self.assertNotIn("structures", slim["scans"][0][2])

    def test_shape_match_and_joint_ls_symbols(self):
        import numpy as np
        from ffpopt.dihed.Dihedrals import shape_match_delta

        hl = np.array([1.0, 2.0, 3.0])
        ll = np.array([0.0, 1.0, 2.0])
        d = shape_match_delta(hl, ll)
        np.testing.assert_allclose(d, shape_match_delta(hl, ll + 9.0))

    def test_align_scan_profiles(self):
        from ffpopt.dihed.Dihedrals import align_scan_profiles
        from ffpopt.Struct import ListOfStruct

        def _frame(name, e=0.0):
            return SimpleNamespace(
                data={
                    "name": name,
                    "energy": e,
                    "positions": [[0.0, 0.0, 0.0]],
                    "constraints": [],
                },
                constraints=None,
            )

        hl = ListOfStruct.from_structs_shared(
            [_frame("d000"), _frame("d010"), _frame("d020"), _frame("d030")]
        )
        ll = ListOfStruct.from_structs_shared(
            [_frame("d010"), _frame("d020"), _frame("d030"), _frame("d040")]
        )
        ahl, all_, info = align_scan_profiles(hl, ll)
        self.assertEqual(info["n_common"], 3)
        self.assertEqual(len(ahl.structs), len(all_.structs))
        self.assertFalse(info.get("interpolated"))

    def test_align_scan_profiles_interpolates_mismatched_full_grids(self):
        from ffpopt.dihed.Dihedrals import align_scan_profiles
        from ffpopt.Struct import ListOfStruct

        def _frame(ang, e):
            return SimpleNamespace(
                data={
                    "name": f"d{int(ang):03d}",
                    "energy": e,
                    "positions": [[0.0, 0.0, 0.0]],
                    "constraints": [],
                },
                constraints=None,
            )

        hl_angs = list(range(0, 360, 15))
        ll_angs = list(range(0, 360, 10))
        hl = ListOfStruct.from_structs_shared(
            [_frame(a, float(a)) for a in hl_angs]
        )
        ll = ListOfStruct.from_structs_shared(
            [_frame(a, 0.0) for a in ll_angs]
        )
        ahl, all_, info = align_scan_profiles(hl, ll)
        self.assertTrue(info.get("interpolated"))
        self.assertEqual(info["n_common"], 36)
        self.assertEqual(len(ahl.structs), 36)
        self.assertEqual(len(all_.structs), 36)
        e10 = next(
            float(s.data["energy"])
            for s in ahl.structs
            if s.data["name"] == "d010"
        )
        self.assertAlmostEqual(e10, 10.0, places=5)

    def test_prim_dihed_energy_term(self):
        from ffpopt.dihed.Dihedrals import PrimDihedFcn

        prim = PrimDihedFcn(2.0, 0.0, 1)
        # CptEne(0 deg) = 2*(1+cos0) = 4
        self.assertAlmostEqual(float(prim.CptEne(0.0)), 4.0)


# ---
# AIMNet aliases, GPU worker cap, and fast model policy
# ---

@unittest.skipUnless(_have_split_ffpopt(), "split src/ffpopt not present (using independent ffpopt)")
class TestAimnetPolicy(unittest.TestCase):
    def test_aimnet_model_aliases_and_fast_policy(self):
        from ffpopt.ase.Aimnet import (
            is_aimnet_model,
            resolve_aimnet_model_name,
        )
        from ffpopt.runtime.FastWavefront import (
            is_aimnet_like_model,
            mm_then_hl_enabled,
            prefer_ase_first_model,
            prefer_wavefront_depth,
        )

        self.assertEqual(resolve_aimnet_model_name("aimnet2"), ("aimnet2", 0))
        self.assertEqual(resolve_aimnet_model_name("aimnet2_wb97m"), ("aimnet2", 0))
        self.assertEqual(resolve_aimnet_model_name("aimnet2-2025"), ("aimnet2-2025", 0))
        self.assertEqual(resolve_aimnet_model_name("aimnet2_b973c"), ("aimnet2-b973c", 0))
        self.assertEqual(resolve_aimnet_model_name("aimnet2_qr"), ("aimnet2-rxn", 0))
        self.assertEqual(resolve_aimnet_model_name("aimnet2-nse_2"), ("aimnet2-nse", 2))
        self.assertEqual(
            resolve_aimnet_model_name("isayevlab/aimnet2-wb97m-d3")[0],
            "isayevlab/aimnet2-wb97m-d3",
        )
        self.assertEqual(
            resolve_aimnet_model_name("ISAYEVLAB/AIMNET2-WB97M-D3")[0],
            "isayevlab/aimnet2-wb97m-d3",
        )
        self.assertTrue(is_aimnet_model("aimnet2"))
        self.assertTrue(is_aimnet_like_model("AIMNet2-2025"))
        self.assertFalse(is_aimnet_model("xtb"))
        self.assertTrue(prefer_ase_first_model("aimnet2", fast=True))
        self.assertFalse(prefer_ase_first_model("aimnet2", fast=False))
        self.assertTrue(prefer_wavefront_depth(model="aimnet2", fast=True))
        self.assertTrue(mm_then_hl_enabled("aimnet2", fast=True))
        from ffpopt.ase.Aimnet import _INSTALL_HINT, _import_aimnet

        self.assertIn(".[aimnet]", _INSTALL_HINT)
        import builtins

        real_import = builtins.__import__

        def _no_aimnet(name, *args, **kwargs):
            if str(name).startswith("aimnet"):
                raise ImportError("No module named 'aimnet'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", _no_aimnet):
            with self.assertRaises(ImportError) as ctx:
                _import_aimnet()
        self.assertIn(".[aimnet]", str(ctx.exception))

    def test_aimnet_gpu_worker_cap_and_pin(self):
        from ffpopt.ase.Aimnet import (
            cap_aimnet_nproc,
            pin_aimnet_worker_cuda,
            visible_cuda_ids,
        )
        from ffpopt.runtime.EnvDefaults import clear_defaults_cache

        env = {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "FFPOPT_AIMNET_PER_GPU": "4",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("FFPOPT_AIMNET_DEVICE", None)
            os.environ.pop("FFPOPT_AIMNET_WORKERS", None)
            clear_defaults_cache()
            self.assertEqual(visible_cuda_ids(), ["0", "1"])
            self.assertEqual(cap_aimnet_nproc(44, "aimnet2"), 8)
            self.assertEqual(cap_aimnet_nproc(44, "xtb"), 44)
            os.environ["FFPOPT_AIMNET_WORKERS"] = "2"
            clear_defaults_cache()
            self.assertEqual(cap_aimnet_nproc(44, "aimnet2"), 2)
            os.environ.pop("FFPOPT_AIMNET_WORKERS", None)
            os.environ["FFPOPT_AIMNET_DEVICE"] = "cpu"
            clear_defaults_cache()
            self.assertEqual(cap_aimnet_nproc(44, "aimnet2"), 44)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}, clear=False):
            self.assertEqual(pin_aimnet_worker_cuda(0), "2")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(os.environ.get("FFPOPT_AIMNET_DEVICE"), "cuda")
        os.environ.pop("FFPOPT_AIMNET_DEVICE", None)
        clear_defaults_cache()


# ---
# GeomOpt facades, geomeTRIC scratch, linear-torsion
# ---

@unittest.skipUnless(_have_split_ffpopt(), "split src/ffpopt not present (using independent ffpopt)")
class TestGeomOptHelpers(unittest.TestCase):
    def test_constraints_to_geometric(self):
        from ffpopt.geom.Constraints import Constraint, to_geometric

        lines = to_geometric([Constraint("dihed", [0, 1, 2, 3], value=45.0)])
        self.assertTrue(any("45.0" in ln and ln.startswith("dihedral") for ln in lines))

    def test_cleanup_geometric_scratch_nsf_and_tmp(self):
        import tempfile
        from pathlib import Path
        from ffpopt.geom.Geometric import (
            cleanup_geometric_scratch,
            geometric_prefix_from_node_pkl,
            sweep_geometric_scratch_dir,
        )
        from ffpopt.scan.WavefrontMixins import cleanup_wavefront_geometric_scratch

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            pkl = td / "level_1_angle_0.0_id_0_node.pckl"
            pkl.write_text("node")
            prefix = geometric_prefix_from_node_pkl(pkl)
            self.assertTrue(prefix.endswith("_geom"))
            tmp = Path(prefix + ".tmp")
            tmp.mkdir()
            (tmp / "junk").write_text("x")
            Path(prefix + ".nsf").write_text("geom log")
            Path(prefix + ".log").write_text("log")
            Path(prefix + "_optim.xyz").write_text("xyz")
            Path(prefix + ".r1.tmp").mkdir()
            (td / "log.nsf").write_text("cwd log")
            (td / "orphan_geom.log").write_text("old")

            n = cleanup_geometric_scratch(prefix, keep_optim=True)
            self.assertGreater(n, 0)
            self.assertFalse(tmp.exists())
            self.assertFalse(Path(prefix + ".nsf").exists())
            self.assertTrue(Path(prefix + "_optim.xyz").exists())

            class _Node:
                node_pkl = str(pkl)
                complete = False

            class _Level:
                nodes = [_Node()]

            class _WF:
                levels = [_Level()]
                workdir = str(td)
                checkpoint = str(td / "checkpoint.pkl")

            cleanup_wavefront_geometric_scratch(_WF(), keep_incomplete_optim=True)
            self.assertTrue(Path(prefix + "_optim.xyz").exists())
            self.assertFalse((td / "log.nsf").exists())

            _Node.complete = True
            cleanup_wavefront_geometric_scratch(_WF(), keep_incomplete_optim=False)
            self.assertFalse(Path(prefix + "_optim.xyz").exists())
            self.assertFalse((td / "orphan_geom.log").exists())
            self.assertEqual(sweep_geometric_scratch_dir(td), 0)

    def test_prepare_geometric_tmpdir_removes_leftover_r2(self):
        from ffpopt.geom.Geometric import prepare_geometric_tmpdir

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            prefix = str(td / "level_1_angle_270_id_1_node_geom")
            leftover = Path(prefix + ".r2.tmp")
            leftover.mkdir()
            (leftover / "x").write_text("old")
            Path(prefix + ".tmp").mkdir()
            prepare_geometric_tmpdir(prefix)
            self.assertFalse(Path(prefix + ".tmp").exists())
            self.assertFalse(leftover.exists())

    def test_linear_torsion_bend_detection_and_unkink(self):
        """Near-180 deg bend in a constrained dihedral is detected and unkinked."""
        try:
            from ase import Atoms
        except ImportError:
            self.skipTest("ase required")
        import numpy as np
        from ffpopt.geom.Constraints import Constraint
        from ffpopt.geom.LinearTorsion import (
            find_near_linear_bends,
            has_near_linear_dihedral_bend,
            is_linear_torsion_error,
            log_linear_torsion,
            unkink_near_linear_bends,
        )

        # A-B-C nearly linear, C-D off axis -> dihedral A-B-C-D ill-defined.
        atoms = Atoms(
            "CCCC",
            positions=[
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [3.0, 0.05, 0.0],
                [3.5, 1.0, 0.0],
            ],
        )
        # Make A-B-C almost linear explicitly.
        atoms.set_angle(0, 1, 2, 178.8)
        cons = [Constraint("dihed", [0, 1, 2, 3], value=60.0)]
        self.assertTrue(has_near_linear_dihedral_bend(atoms, cons))
        hits = find_near_linear_bends(atoms, cons)
        self.assertTrue(hits)
        self.assertGreaterEqual(hits[0]["angle_deg"], 175.0)
        unkink_near_linear_bends(atoms, hits, target_deg=170.0)
        self.assertAlmostEqual(atoms.get_angle(0, 1, 2), 170.0, places=1)
        self.assertFalse(has_near_linear_dihedral_bend(atoms, cons))

        # Log helper emits strict UTF-8 bytes (ASCII message body).
        msg = "[ffpopt] linear-torsion unkink 1-2-3: 178.80 deg -> 170.00 deg"
        data = (msg + "\n").encode("utf-8")
        self.assertEqual(data.decode("utf-8"), msg + "\n")
        log_linear_torsion(msg)

        class _E(Exception):
            pass

        class LinearTorsionError(_E):
            pass

        self.assertTrue(
            is_linear_torsion_error(
                LinearTorsionError(
                    "A constrained torsion has three consecutive atoms "
                    "forming a nearly linear angle"
                )
            )
        )
        self.assertFalse(is_linear_torsion_error(ValueError("other")))

    def test_geomopt_facade_exports(self):
        from ffpopt.geom import GeomOptAse, GeomOptParallel
        from ffpopt.geom.GeomOpt import CalcNode, GeomOpt, GeomOpt_ASE

        self.assertIs(GeomOpt_ASE, GeomOptAse.GeomOpt_ASE)
        self.assertIs(CalcNode, GeomOptParallel.CalcNode)
        self.assertTrue(callable(GeomOpt))

    def test_read_last_optim_xyz_warm_start_helper(self):
        """Interrupted geomopt leaves ``_optim.xyz``; helper reads last frame."""
        import numpy as np
        from ffpopt.geom.Geometric import read_last_optim_xyz, write_plain_xyz

        try:
            import ase
            from ase import Atoms
        except ImportError:
            self.skipTest("ase required")

        with tempfile.TemporaryDirectory() as td:
            prefix = Path(td) / "node_geom"
            atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
            write_plain_xyz(str(prefix) + ".xyz", atoms)
            atoms2 = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.90]])
            # geomeTRIC-style trajectory name
            write_plain_xyz(str(prefix) + "_optim.xyz", atoms2)
            last = read_last_optim_xyz(prefix)
            self.assertIsNotNone(last)
            np.testing.assert_allclose(last[1, 2], 0.90)


# ---
# Wavefront spawn, rescue, checkpoint, and pickle facades
# ---

@unittest.skipUnless(_have_split_ffpopt(), "split src/ffpopt not present (using independent ffpopt)")
class TestWavefrontPolicy(unittest.TestCase):
    def test_is_soft_opt_and_evaluate_policy(self):
        from ffpopt.geom.GeomOpt import is_soft_opt_recovery
        from ffpopt.scan.WavefrontMixins import evaluate_wavefront_minimum

        self.assertTrue(is_soft_opt_recovery("loose"))
        self.assertFalse(is_soft_opt_recovery("primary"))
        self.assertTrue(is_soft_opt_recovery("linear-torsion"))
        self.assertTrue(is_soft_opt_recovery("linear-torsion-soft"))
        d = evaluate_wavefront_minimum(
            energy=1.0,
            soft=True,
            has_incumbent=False,
            incumbent_energy=None,
            incumbent_soft=False,
            threshold_ev=0.1,
        )
        self.assertEqual(d["reason"], "soft_first_seed")

    def test_wavefront_policy_matrix(self):
        """Lock spawn / update decisions for soft and hard incumbents."""
        from ffpopt.scan.WavefrontMixins import evaluate_wavefront_minimum

        cases = [
            # energy, soft, has, inc_e, inc_soft, thr, reason, update, active
            (1.0, True, False, None, False, 0.1, "soft_first_seed", True, True),
            (0.9, True, True, 1.0, True, 0.1, "soft_improve", True, False),
            (1.1, True, True, 1.0, True, 0.1, "soft_demoted", False, False),
            (1.0, False, False, None, False, 0.1, "hard_first", True, True),
            (0.9, False, True, 1.0, True, 0.1, "hard_replace_soft", True, True),
            (1.1, False, True, 1.0, True, 0.1, "hard_worse_than_soft", False, False),
            (0.8, False, True, 1.0, False, 0.1, "hard_significant_improve", True, True),
            (0.95, False, True, 1.0, False, 0.1, "hard_quiet_improve", True, False),
            (1.0, False, True, 1.0, False, 0.1, "hard_not_lower", False, False),
            (float("nan"), False, False, None, False, 0.1, "nonfinite", False, False),
        ]
        for energy, soft, has, inc_e, inc_soft, thr, reason, upd, act in cases:
            with self.subTest(reason=reason):
                d = evaluate_wavefront_minimum(
                    energy=energy,
                    soft=soft,
                    has_incumbent=has,
                    incumbent_energy=inc_e,
                    incumbent_soft=inc_soft,
                    threshold_ev=thr,
                )
                self.assertEqual(d["reason"], reason)
                self.assertEqual(d["update_min"], upd)
                self.assertEqual(d["active"], act)

    def test_demote_redundant_spawn_guards(self):
        from ffpopt.scan.WavefrontMixins import demote_redundant_spawn

        kw = dict(
            loc=80.0,
            prior_expands=0,
            max_expand_per_loc=3,
            n_hard_bins=36,
            n_grid=36,
            improve_ev=0.2,
            threshold_ev=0.1,
            coverage_spawn_factor=4.0,
            recent_spawn_locs=[],
            pingpong_window=8,
        )
        self.assertIsNone(
            demote_redundant_spawn(reason="hard_first", **kw)
        )
        self.assertEqual(
            demote_redundant_spawn(reason="hard_significant_improve", **kw),
            "coverage_cauchy",
        )
        self.assertIsNone(
            demote_redundant_spawn(
                reason="hard_significant_improve",
                **{**kw, "improve_ev": 0.5, "n_hard_bins": 10},
            )
        )
        self.assertEqual(
            demote_redundant_spawn(
                reason="hard_significant_improve",
                **{**kw, "prior_expands": 3, "n_hard_bins": 10, "improve_ev": 0.5},
            ),
            "expand_cap",
        )
        ping = [80.0, 90.0] * 4
        self.assertEqual(
            demote_redundant_spawn(
                reason="hard_significant_improve",
                **{
                    **kw,
                    "n_hard_bins": 10,
                    "improve_ev": 0.5,
                    "recent_spawn_locs": ping,
                },
            ),
            "pingpong",
        )

    def test_discrete_energy_spike_and_rescue_seed(self):
        import numpy as np
        from ffpopt.scan.WavefrontMixins import (
            bin_needs_rescue,
            discrete_energy_spike,
            kabsch_lerp_coords,
            lerp_weight_on_cycle,
            lost_dihedral_well_delta,
            select_rescue_seed,
        )
        from ffpopt.scan.WavefrontMixins import kcal_threshold_to_ev

        e230 = -2016.354702
        e240 = -2016.099153
        e250 = -2016.370228
        spike = discrete_energy_spike(e240, e230, e250)
        self.assertGreater(spike, 0.25)
        thr = kcal_threshold_to_ev(2.0)
        self.assertTrue(
            bin_needs_rescue(e240, e230, e250, threshold_ev=thr, completed=True)
        )
        self.assertFalse(
            bin_needs_rescue(e250, e240, e230, threshold_ev=thr, completed=True)
        )
        self.assertTrue(
            bin_needs_rescue(
                None, e250, None, threshold_ev=thr, completed=True
            )
        )
        self.assertFalse(
            bin_needs_rescue(
                None, e250, None, threshold_ev=thr, completed=False
            )
        )
        self.assertIsNone(
            lost_dihedral_well_delta(229.8, 230.0, limit_deg=30.0)
        )
        lost = lost_dihedral_well_delta(50.62, 230.0, limit_deg=30.0)
        self.assertIsNotNone(lost)
        self.assertGreater(lost, 170.0)
        self.assertIsNone(lost_dihedral_well_delta(50.62, 230.0, limit_deg=0))

        crd_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        crd_b = crd_a + np.array([0.5, 0.0, 0.0])
        mixed = kabsch_lerp_coords(crd_a, crd_a, 0.5)
        np.testing.assert_allclose(mixed, crd_a, atol=1e-9)
        ident = kabsch_lerp_coords(crd_a, crd_b, 0.0)
        np.testing.assert_allclose(ident, crd_a, atol=1e-9)
        self.assertAlmostEqual(lerp_weight_on_cycle(230.0, 250.0, 240.0), 0.5)

        class _S:
            def __init__(self, coords, tag):
                self.data = {"positions": np.asarray(coords, dtype=float)}
                self.tag = tag

            def clone_geometry(self, coords, ene=0.0, frcs=None):
                out = _S(coords, self.tag + "+lerp")
                return out

        energies = {230.0: e230, 240.0: e240, 250.0: e250}
        structs = {
            230.0: _S(crd_a, "230"),
            250.0: _S(crd_b, "250"),
        }
        struct, seed_e, src = select_rescue_seed(
            240.0,
            230.0,
            250.0,
            energies=energies,
            structures=structs,
            lerp_ev=thr,
        )
        self.assertIn("lerp", src)
        self.assertEqual(seed_e, e250)
        self.assertIn("lerp", struct.tag)

        only = {250.0: e250}
        struct2, seed2, src2 = select_rescue_seed(
            240.0,
            230.0,
            250.0,
            energies=only,
            structures={250.0: structs[250.0]},
            lerp_ev=thr,
        )
        self.assertEqual(src2, "250")
        self.assertIs(struct2, structs[250.0])
        self.assertEqual(seed2, e250)

    def test_rescue_outlier_bins_enqueues_spike(self):
        import numpy as np
        from types import SimpleNamespace
        from ffpopt.scan.WavefrontEngine import Wavefront
        from ffpopt.scan.WavefrontMixins import kcal_threshold_to_ev

        class _S:
            def __init__(self, tag):
                self.data = {
                    "positions": np.array(
                        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
                    )
                }
                self.tag = tag

            def clone_geometry(self, coords, ene=0.0, frcs=None):
                out = _S(self.tag + "+lerp")
                out.data = {"positions": np.asarray(coords, dtype=float)}
                return out

        wf = Wavefront.__new__(Wavefront)
        wf.grid = None
        wf.delta = 10
        wf.min_energies = {
            230.0: -2016.354702,
            240.0: -2016.099153,
            250.0: -2016.370228,
        }
        wf.min_structures = {230.0: _S("230"), 250.0: _S("250")}
        wf._rescue_count = {}
        wf._completed_locs = {230.0, 240.0, 250.0}
        wf._pending_by_loc = {}
        wf._inflight_locs = set()
        wf._deferred_seeds = {}
        wf._expand_count = {}
        wf._recent_spawns = []
        enqueued = []

        def _enqueue(loc, **kwargs):
            enqueued.append((loc, kwargs.get("seed_energy"), kwargs.get("angle")))
            return SimpleNamespace(angle=kwargs.get("angle"), loc=loc)

        wf._enqueue_visit = _enqueue
        node = SimpleNamespace(angle=250.0, level=3)
        spawned = wf._rescue_outlier_bins(node)
        locs = [item[0] for item in enqueued]
        self.assertIn(240.0, locs)
        self.assertTrue(any(getattr(n, "angle", None) == 240.0 for n in spawned))
        self.assertGreater(wf._rescue_count[240.0], 0)
        self.assertGreater(kcal_threshold_to_ev(2.0), 0.0)

    def test_wavefront_apply_minimum_helper(self):
        """Shared evaluate-node tail updates the min store and node.active."""
        from types import SimpleNamespace

        from ffpopt.scan.WavefrontMixins import apply_wavefront_minimum_to_node

        store = {}

        def on_update(n, reason, _old):
            store["energy"] = n.energy
            store["reason"] = reason

        node = SimpleNamespace(
            energy=1.0,
            active=True,
            soft_opt=False,
            opt_geom=None,
            opt_recovery=None,
        )
        d = apply_wavefront_minimum_to_node(
            node,
            loc=90,
            threshold_kcal=2.306,
            has_incumbent=False,
            incumbent_energy=None,
            incumbent_soft=False,
            on_update=on_update,
            noun="angle",
        )
        self.assertEqual(d["reason"], "hard_first")
        self.assertTrue(node.active)
        self.assertEqual(store["energy"], 1.0)

    def test_wavefront_facade_exports(self):
        from ffpopt.scan.WaveFront import Wavefront, WavefrontNode, run_dihed_wavefront
        from ffpopt.scan.WaveFrontND import (
            GetGridNeighbors,
            Wavefront as NDWavefront,
            run_dihed_wavefront as run_ndim_wavefront,
        )
        from ffpopt.scan.WavefrontEngine import Wavefront as EngineWavefront
        from ffpopt.scan.WavefrontMixins import (
            apply_wavefront_minimum_to_node,
            finalize_successful_node_opt,
            slim_completed_nodes_for_checkpoint,
        )

        self.assertTrue(callable(run_dihed_wavefront))
        self.assertTrue(callable(run_ndim_wavefront))
        self.assertIsNot(run_dihed_wavefront, run_ndim_wavefront)
        self.assertTrue(callable(GetGridNeighbors))
        self.assertIs(Wavefront, NDWavefront)
        self.assertIs(Wavefront, EngineWavefront)
        self.assertTrue(callable(apply_wavefront_minimum_to_node))
        self.assertTrue(callable(finalize_successful_node_opt))
        self.assertTrue(callable(slim_completed_nodes_for_checkpoint))
        self.assertTrue(hasattr(Wavefront, "_evaluate_node"))
        self.assertTrue(hasattr(Wavefront, "calculate_mpi"))
        self.assertTrue(hasattr(WavefrontNode, "calculate"))

    def test_grid_neighbors_von_neumann(self):
        from ffpopt.scan.WavefrontEngine import GetGridNeighbors

        class _Dim:
            def __init__(self, size, isper):
                self.size = size
                self.isper = isper

        class _Grid:
            def __init__(self):
                self.dims = [_Dim(10, True), _Dim(8, False)]

            def CptGlbIdxFromBinIdx(self, b):
                return int(b[0]) * 8 + int(b[1])

        grid = _Grid()
        nbs = GetGridNeighbors([5, 4], grid)
        as_set = {tuple(p) for p in nbs}
        self.assertEqual(as_set, {(4, 4), (6, 4), (5, 3), (5, 5)})
        self.assertEqual(len(nbs), 4)

        edge = GetGridNeighbors([0, 0], grid)
        self.assertEqual({tuple(p) for p in edge}, {(9, 0), (1, 0), (0, 1)})

        valid = {grid.CptGlbIdxFromBinIdx([4, 4]), grid.CptGlbIdxFromBinIdx([6, 4])}
        clipped = GetGridNeighbors([5, 4], grid, validbins=valid)
        self.assertEqual({tuple(p) for p in clipped}, {(4, 4), (6, 4)})

    def test_wavefront_seed_coalesce(self):
        from types import SimpleNamespace
        from collections import deque
        from ffpopt.scan.WavefrontEngine import Wavefront

        wf = Wavefront.__new__(Wavefront)
        wf.grid = None
        wf.delta = 10
        wf.levels = []
        wf._pending_by_loc = {}
        wf._inflight_locs = set()
        wf._deferred_seeds = {}

        pending_node = SimpleNamespace(struct="old", seed_energy=2.0, angle=170.0)
        wf._pending_by_loc[170.0] = pending_node
        self.assertIsNone(
            wf._enqueue_visit(
                170.0,
                struct="better",
                seed_energy=1.0,
                level_id=2,
                angle=170.0,
            )
        )
        self.assertEqual(pending_node.struct, "better")
        self.assertEqual(pending_node.seed_energy, 1.0)
        self.assertIsNone(
            wf._enqueue_visit(
                170.0,
                struct="worse",
                seed_energy=3.0,
                level_id=2,
                angle=170.0,
            )
        )
        self.assertEqual(pending_node.struct, "better")

        wf._pending_by_loc.clear()
        wf._inflight_locs.add(180.0)
        self.assertIsNone(
            wf._enqueue_visit(
                180.0,
                struct="seed-a",
                seed_energy=4.0,
                level_id=3,
                angle=180.0,
            )
        )
        self.assertIsNone(
            wf._enqueue_visit(
                180.0,
                struct="seed-b",
                seed_energy=1.5,
                level_id=4,
                angle=180.0,
            )
        )
        self.assertEqual(wf._deferred_seeds[180.0]["struct"], "seed-b")
        self.assertEqual(wf._deferred_seeds[180.0]["energy"], 1.5)

        a = SimpleNamespace(angle=10.0, seed_energy=None)
        b = SimpleNamespace(angle=10.0, seed_energy=None)
        c = SimpleNamespace(angle=20.0, seed_energy=None)
        q = deque([a, b, c])
        wf._rebuild_occupancy(q)
        self.assertEqual(list(q), [a, c])
        self.assertIs(wf._pending_by_loc[10.0], a)

    def test_checkpoint_keeps_calc_cache(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from ffpopt.scan.WavefrontMixins import pickle_checkpoint_keep_calc_cache

        class _Los:
            def __init__(self):
                self.calc = object()
                self._ffpopt_calc_cache = ("key", object())
                self.cleared = 0

            def clear_runtime_caches(self):
                self.cleared += 1
                self.calc = None
                self._ffpopt_calc_cache = None

        los = _Los()
        calc = los.calc
        cache = los._ffpopt_calc_cache
        payload = SimpleNamespace(x=1)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ckpt.pkl"
            pickle_checkpoint_keep_calc_cache(payload, path, los)
            self.assertTrue(path.is_file())
        self.assertEqual(los.cleared, 1)
        self.assertIs(los.calc, calc)
        self.assertIs(los._ffpopt_calc_cache, cache)

    def test_list_of_struct_pickle_drops_calc_cache(self):
        import pickle
        from ffpopt.Struct import ListOfStruct

        class Boom:
            def __getstate__(self):
                raise TypeError("cannot pickle boom")

        los = ListOfStruct.__new__(ListOfStruct)
        los.structs = []
        los.args = None
        los.calc = Boom()
        los._ffpopt_calc_cache = ("k", Boom())
        los._ffpopt_qdpi2_full_cache = ("k2", Boom())
        los._ffpopt_mm_preopt_los = ("sander", Boom())

        restored = pickle.loads(pickle.dumps(los))
        self.assertIsNone(restored.calc)
        self.assertIsNone(restored._ffpopt_calc_cache)
        self.assertIsNone(restored._ffpopt_qdpi2_full_cache)
        self.assertIsNone(restored._ffpopt_mm_preopt_los)
        self.assertIsInstance(los.calc, Boom)
        self.assertEqual(los._ffpopt_calc_cache[0], "k")

    def test_pickle_compat_alias(self):
        from ffpopt.scan.WavefrontMixins import register_wavefront_pickle_aliases
        from ffpopt.scan.WaveFront import Wavefront
        from ffpopt.scan.WaveFrontND import Wavefront as NDWavefront

        register_wavefront_pickle_aliases()
        import ffpopt.WaveFront as legacy
        import ffpopt.WaveFrontND as legacy_nd

        self.assertIs(legacy.Wavefront, Wavefront)
        self.assertIs(legacy_nd.Wavefront, NDWavefront)
        self.assertIs(legacy.Wavefront, legacy_nd.Wavefront)


# ---
# Fast presets and sander-like low-level scans
# ---

@unittest.skipUnless(_have_split_ffpopt(), "split src/ffpopt not present (using independent ffpopt)")
class TestFastWavefront(unittest.TestCase):
    def test_fast_presets_keep_delta(self):
        from ffpopt.runtime.FastWavefront import (
            LIBRARY_DEFAULTS,
            apply_fast_wavefront_presets,
        )

        knobs = dict(LIBRARY_DEFAULTS)
        applied = apply_fast_wavefront_presets(knobs, enabled=True)
        self.assertNotIn("delta", applied)
        self.assertEqual(knobs["delta"], 10)

    def test_fast_wavefront_enabled(self):
        from ffpopt.runtime.FastWavefront import fast_wavefront_enabled

        self.assertTrue(fast_wavefront_enabled(explicit=True))
        self.assertFalse(fast_wavefront_enabled(explicit=False))
        with patch.dict(os.environ, {"FFPOPT_FAST_WAVEFRONT": "1"}):
            self.assertTrue(fast_wavefront_enabled())
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FFPOPT_FAST_WAVEFRONT", None)
            self.assertFalse(fast_wavefront_enabled())

    def test_sander_ll_scan_uses_ase_first_and_prefers_depth(self):
        from ffpopt.workflows.TwistHelpers import _is_sander_ll_model, _wf_kwargs_for_scan_model
        from ffpopt.runtime.FastWavefront import (
            prefer_ase_first_model,
            prefer_bond_pool_depth,
            prefer_fragment_pool_depth,
            prefer_wavefront_depth,
            qdpi2_opt_components,
            split_nproc_for_items,
        )

        self.assertTrue(_is_sander_ll_model("sander"))
        self.assertFalse(_is_sander_ll_model("xtb"))
        kw = _wf_kwargs_for_scan_model("sander", {"nproc": 4, "delta": 10})
        self.assertIs(kw["geometric_opt"], False)
        # Explicit override wins.
        kw2 = _wf_kwargs_for_scan_model("sander", {"geometric_opt": True})
        self.assertIs(kw2["geometric_opt"], True)
        # Sander no longer always prefers depth at the model level.
        self.assertFalse(prefer_wavefront_depth(model="sander", fast=False))
        self.assertFalse(prefer_wavefront_depth(model="qdpi2", fast=False))
        # Tiny lease + multi-bond -> bond breadth (concurrent bonds).
        self.assertFalse(
            prefer_bond_pool_depth(model="sander", nproc=3, n_bonds=3)
        )
        # Large lease can keep depth when the caller asks for it.
        self.assertTrue(
            prefer_bond_pool_depth(
                model="sander", nproc=12, n_bonds=3, prefer=True
            )
        )
        # Many fragments -> always breadth (xtb --fast used to serialize).
        self.assertFalse(
            prefer_fragment_pool_depth(
                model="xtb", nproc=8, n_fragments=6, fast=True
            )
        )
        self.assertFalse(
            prefer_fragment_pool_depth(
                model="xtb", nproc=60, n_fragments=11, fast=True
            )
        )
        self.assertEqual(
            split_nproc_for_items(
                60, 11, prefer_depth=False, flatten_nested=False
            ),
            (11, 5),
        )
        # Depth + flatten is 1 x nproc (wrong for the parent fragment pool).
        self.assertEqual(
            split_nproc_for_items(60, 11, prefer_depth=True, flatten_nested=True),
            (1, 60),
        )
        self.assertGreater(
            split_nproc_for_items(
                60, 11, prefer_depth=True, flatten_nested=False
            )[0],
            1,
        )
        # Flatten nested spawn: never both outer and inner > 1.
        n_outer, n_inner = split_nproc_for_items(8, 4, prefer_depth=False)
        self.assertTrue(n_outer == 1 or n_inner == 1)
        n_outer_d, n_inner_d = split_nproc_for_items(8, 4, prefer_depth=True)
        self.assertTrue(n_outer_d == 1 or n_inner_d == 1)
        # Top-level twist (not already in a spawn worker) keeps a 2-D split.
        n_o, n_i = split_nproc_for_items(
            44, 2, prefer_depth=True, flatten_nested=False
        )
        self.assertEqual((n_o, n_i), (2, 22))
        # Absorb leftover cores (old 8x5 left 4 idle; 16 HL+orig jobs were 16x2).
        self.assertEqual(
            split_nproc_for_items(44, 8, prefer_depth=True, flatten_nested=False),
            (4, 11),
        )
        self.assertEqual(
            split_nproc_for_items(44, 16, prefer_depth=True, flatten_nested=False),
            (11, 4),
        )
        # Nested correlated scans raise min_inner so 60/8 is 4x15, not 6x10.
        self.assertEqual(
            split_nproc_for_items(
                60, 8, prefer_depth=True, flatten_nested=False, min_inner=15
            ),
            (4, 15),
        )
        self.assertTrue(prefer_ase_first_model("xtb", fast=True))
        self.assertFalse(prefer_ase_first_model("xtb", fast=False))
        self.assertTrue(prefer_ase_first_model("aimnet2", fast=True))
        with patch.dict(os.environ, {"FFPOPT_FAST_WAVEFRONT": "1"}):
            self.assertEqual(qdpi2_opt_components(), "xtb")
        with patch.dict(os.environ, {"FFPOPT_QDPI2_OPT": "both"}):
            self.assertEqual(qdpi2_opt_components(), "both")


# ---
# CPU lease, resplit, and never-starve
# ---

@unittest.skipUnless(_have_split_ffpopt(), "split src/ffpopt not present (using independent ffpopt)")
class TestCpuBudget(unittest.TestCase):
    def test_cpu_budget_fair_share(self):
        from ffpopt.runtime.CpuBudget import cpu_lease_weight, fair_share_leases

        leases = fair_share_leases(8, ["a", "b", "c"])
        self.assertEqual(sum(leases.values()), 8)
        self.assertEqual(len(leases), 3)

        self.assertEqual(cpu_lease_weight(2, correlated=False), 1)
        self.assertEqual(cpu_lease_weight(8, correlated=True), 8)
        with patch.dict(os.environ, {"FFPOPT_WHOLE_MAX_BONDS_PER_TWIST": "8"}):
            self.assertEqual(cpu_lease_weight(12, correlated=True), 8)

        owners = ["tail", "a", "b", "c", "d", "e"]
        weights = {oid: (8 if oid == "tail" else 1) for oid in owners}
        weighted = fair_share_leases(44, owners, weights=weights)
        self.assertEqual(sum(weighted.values()), 44)
        self.assertEqual(weighted["tail"], 27)
        for oid in "abcde":
            self.assertGreaterEqual(weighted[oid], 3)
            self.assertLessEqual(weighted[oid], 4)

        packed = fair_share_leases(
            11, [f"f{i}" for i in range(11)], min_each=2
        )
        self.assertLessEqual(len(packed), 5)
        self.assertTrue(all(v >= 2 for v in packed.values()))
        self.assertEqual(sum(packed.values()), 11)

        reserved = fair_share_leases(
            44, ["first"], min_each=2, virtual_units=10
        )
        self.assertEqual(reserved["first"], 4)

    def test_cpu_budget_clear_leases_on_init(self):
        from ffpopt.runtime.CpuBudget import CpuBudget

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / ".cpu_budget.json"
            b = CpuBudget(path, 8)
            b.lease("fragment_1")
            b.lease("fragment_2")
            self.assertGreater(len(b.snapshot()["leases"]), 0)
            CpuBudget(path, 8, clear_leases=True)
            self.assertEqual(CpuBudget(path, 8).snapshot()["leases"], {})

    def test_cpu_budget_lease_persists_weights(self):
        from ffpopt.runtime.CpuBudget import CpuBudget

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / ".cpu_budget.json"
            b = CpuBudget(path, 44, n_alive=2)
            b.lease("tail", weight=8)
            b.lease("a", weight=1)
            snap = b.snapshot()
            self.assertEqual(snap["weights"]["tail"], 8)
            self.assertGreater(snap["leases"]["tail"], snap["leases"]["a"])
            self.assertGreaterEqual(min(snap["leases"].values()), 2)
            b.release("a")
            b.dec_alive()
            grown = CpuBudget(path, 44).lease("tail", weight=8)
            self.assertEqual(grown, 44)
            self.assertNotIn("a", CpuBudget(path, 44).snapshot()["leases"])

    def test_cpu_budget_never_starves_on_one_core(self):
        from ffpopt.runtime.CpuBudget import CpuBudget, cpu_min_lease

        self.assertGreaterEqual(cpu_min_lease(44), 2)
        self.assertEqual(cpu_min_lease(1), 1)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / ".cpu_budget.json"
            b = CpuBudget(path, 44, n_alive=11)
            first = b.lease("fragment_10", weight=1)
            self.assertEqual(first, 4)
            self.assertNotEqual(first, 1)
            CpuBudget(path, 1)
            self.assertEqual(CpuBudget(path, 1).snapshot()["total"], 44)
            for i in range(10):
                b.dec_alive()
            b.release("fragment_10")
            last = CpuBudget(path, 44).lease("fragment_10", weight=1)
            self.assertEqual(last, 44)
            fit_sib = CpuBudget(path, 44, clear_leases=True, n_alive=2)
            fit_sib.lease("fragment_11", weight=1)
            fit_sib.release("fragment_11")
            leftover = CpuBudget(path, 44).lease("fragment_10", weight=1)
            self.assertEqual(leftover, 44)
            tight = CpuBudget(path, 11, clear_leases=True, n_alive=11)
            held = []
            for i in range(11):
                got = tight.lease(f"f{i}", weight=1)
                if got:
                    held.append(got)
            self.assertTrue(held)
            self.assertTrue(all(v >= 2 for v in held))
            self.assertLessEqual(len(held), 5)

    def test_execute_bond_scan_jobs_resplit_remaining(self):
        from ffpopt.workflows.TwistHelpers import _execute_bond_scan_jobs

        seen_nproc = []
        pool_sizes = []

        def fake_run(job):
            seen_nproc.append(int(job["wf_kwargs"]["nproc"]))
            return {
                "prefix": job["prefix"],
                "dihed_idxs": job["dihed_idxs"],
                "result": None,
            }

        class FakePool:
            def __init__(self, n):
                pool_sizes.append(n)

            def imap_unordered(self, fn, jobs):
                for job in jobs:
                    yield fn(job)

            def close(self):
                return None

            def join(self):
                return None

        nproc_seq = [1, 12]

        def refresh(_phase):
            return nproc_seq.pop(0) if nproc_seq else 12

        jobs = [
            {
                "prefix": "hl",
                "dihed_idxs": (i, i + 1, i + 2, i + 3),
                "model": "xtb",
                "wf_kwargs": {"nproc": 1},
            }
            for i in range(4)
        ]
        with patch(
            "ffpopt.workflows.TwistHelpers._run_bond_scan_job",
            side_effect=fake_run,
        ), patch(
            "ffpopt.workflows.TwistHelpers.make_nondaemon_spawn_pool",
            side_effect=FakePool,
        ), patch(
            "ffpopt.scan.WaveFront.close_reused_wavefront_pool",
        ), patch.dict(
            os.environ,
            {
                "FFPOPT_MIN_WF_NPROC": "2",
                "FFPOPT_PREF_WF_DEPTH": "0",
                "FFPOPT_PREF_WF_BREADTH": "0",
            },
            clear=False,
        ):
            out = _execute_bond_scan_jobs(
                jobs,
                nproc=1,
                prefer_wf_depth=True,
                logger=None,
                nest_bond_pool=True,
                refresh_nproc=refresh,
            )
        self.assertEqual(len(out), 4)
        self.assertEqual(seen_nproc[0], 1)
        self.assertEqual(pool_sizes, [3])
        self.assertEqual(seen_nproc[1:], [4, 4, 4])

    def test_correlated_small_lease_stays_sequential_then_grows(self):
        from ffpopt.workflows.TwistHelpers import _execute_bond_scan_jobs

        seen_nproc = []
        pool_sizes = []

        def fake_run(job):
            seen_nproc.append(int(job["wf_kwargs"]["nproc"]))
            return {
                "prefix": job["prefix"],
                "dihed_idxs": job["dihed_idxs"],
                "result": None,
            }

        class FakePool:
            def __init__(self, n):
                pool_sizes.append(n)

            def imap_unordered(self, fn, jobs):
                for job in jobs:
                    yield fn(job)

            def close(self):
                return None

            def join(self):
                return None

        jobs = [
            {
                "prefix": "hl",
                "dihed_idxs": (i, i + 1, i + 2, i + 3),
                "model": "xtb",
                "wf_kwargs": {"nproc": 1},
            }
            for i in range(8)
        ]
        with patch(
            "ffpopt.workflows.TwistHelpers._run_bond_scan_job",
            side_effect=fake_run,
        ), patch(
            "ffpopt.workflows.TwistHelpers.make_nondaemon_spawn_pool",
            side_effect=FakePool,
        ), patch(
            "ffpopt.scan.WaveFront.close_reused_wavefront_pool",
        ), patch.dict(
            os.environ,
            {
                "FFPOPT_MIN_WF_NPROC": "2",
                "FFPOPT_PREF_WF_DEPTH": "0",
                "FFPOPT_PREF_WF_BREADTH": "0",
            },
            clear=False,
        ):
            stuck = _execute_bond_scan_jobs(
                jobs,
                nproc=4,
                prefer_wf_depth=True,
                logger=None,
                nest_bond_pool=True,
                refresh_nproc=lambda _p: 4,
            )
        self.assertEqual(len(stuck), 8)
        self.assertEqual(seen_nproc, [4] * 8)
        self.assertEqual(pool_sizes, [])

        seen_nproc.clear()
        pool_sizes.clear()
        nproc_seq = [4, 44]

        def refresh(_phase):
            return nproc_seq.pop(0) if nproc_seq else 44

        jobs2 = [
            {
                "prefix": "hl",
                "dihed_idxs": (i, i + 1, i + 2, i + 3),
                "model": "xtb",
                "wf_kwargs": {"nproc": 1},
            }
            for i in range(8)
        ]
        with patch(
            "ffpopt.workflows.TwistHelpers._run_bond_scan_job",
            side_effect=fake_run,
        ), patch(
            "ffpopt.workflows.TwistHelpers.make_nondaemon_spawn_pool",
            side_effect=FakePool,
        ), patch(
            "ffpopt.scan.WaveFront.close_reused_wavefront_pool",
        ), patch.dict(
            os.environ,
            {
                "FFPOPT_MIN_WF_NPROC": "2",
                "FFPOPT_PREF_WF_DEPTH": "0",
                "FFPOPT_PREF_WF_BREADTH": "0",
            },
            clear=False,
        ):
            grew = _execute_bond_scan_jobs(
                jobs2,
                nproc=4,
                prefer_wf_depth=True,
                logger=None,
                nest_bond_pool=True,
                refresh_nproc=refresh,
            )
        self.assertEqual(len(grew), 8)
        self.assertEqual(seen_nproc[0], 4)
        self.assertEqual(pool_sizes, [4, 2])
        self.assertEqual(seen_nproc[1:5], [11, 11, 11, 11])
        self.assertEqual(seen_nproc[5:], [22, 22, 44])

    def test_execute_bond_scan_jobs_wave_resplit_without_refresh(self):
        from ffpopt.workflows.TwistHelpers import _execute_bond_scan_jobs

        seen_nproc = []
        pool_sizes = []

        def fake_run(job):
            seen_nproc.append(int(job["wf_kwargs"]["nproc"]))
            return {
                "prefix": job["prefix"],
                "dihed_idxs": job["dihed_idxs"],
                "result": None,
            }

        class FakePool:
            def __init__(self, n):
                pool_sizes.append(n)

            def imap_unordered(self, fn, jobs):
                for job in jobs:
                    yield fn(job)

            def close(self):
                return None

            def join(self):
                return None

        jobs = [
            {
                "prefix": "it01",
                "dihed_idxs": (i, i + 1, i + 2, i + 3),
                "model": "sander",
                "wf_kwargs": {"nproc": 1},
            }
            for i in range(8)
        ]
        with patch(
            "ffpopt.workflows.TwistHelpers._run_bond_scan_job",
            side_effect=fake_run,
        ), patch(
            "ffpopt.workflows.TwistHelpers.make_nondaemon_spawn_pool",
            side_effect=FakePool,
        ), patch(
            "ffpopt.scan.WaveFront.close_reused_wavefront_pool",
        ), patch.dict(
            os.environ,
            {
                "FFPOPT_MIN_WF_NPROC": "2",
                "FFPOPT_PREF_WF_DEPTH": "0",
                "FFPOPT_PREF_WF_BREADTH": "0",
            },
            clear=False,
        ):
            out = _execute_bond_scan_jobs(
                jobs,
                nproc=60,
                prefer_wf_depth=True,
                logger=None,
                nest_bond_pool=True,
                refresh_nproc=None,
            )
        self.assertEqual(len(out), 8)
        # Not 6x10 with two leftover jobs stuck at wf=10.
        self.assertEqual(pool_sizes, [4, 4])
        self.assertEqual(seen_nproc, [15] * 8)

    def test_split_nproc_for_items(self):
        from ffpopt.runtime.FastWavefront import split_nproc_for_items
        from ffpopt.workflows.TwistHelpers import _split_fragment_nproc

        self.assertEqual(_split_fragment_nproc(8, 4), split_nproc_for_items(8, 4))
        n_outer, n_inner = split_nproc_for_items(8, 4)
        # Flattened: never nest both axes; product may be < nproc.
        self.assertTrue(n_outer == 1 or n_inner == 1)
        self.assertLessEqual(n_outer * n_inner, 8)
        n_outer2, n_inner2 = split_nproc_for_items(
            8, 4, flatten_nested=False
        )
        self.assertEqual(n_outer2 * n_inner2, 8)
        n_o, n_i = split_nproc_for_items(
            8, 4, prefer_depth=True, flatten_nested=False
        )
        self.assertGreater(n_o, 1)
        self.assertGreater(n_i, 1)
        self.assertEqual(n_o * n_i, 8)
        self.assertEqual(
            _split_fragment_nproc(
                44, 2, prefer_depth=True, flatten_nested=False
            ),
            (2, 22),
        )


# ---
# Fragment / whole-ligand twist batches
# ---

@unittest.skipUnless(_have_split_ffpopt(), "split src/ffpopt not present (using independent ffpopt)")
class TestTwistBatching(unittest.TestCase):
    def test_fragment_twist_done_sentinel(self):
        from alps.workflows import (
            clear_fragment_twist_done,
            is_fragment_twist_done,
            mark_fragment_twist_done,
        )

        with tempfile.TemporaryDirectory() as td:
            frag = Path(td) / "fragment_1"
            frag.mkdir()
            self.assertFalse(is_fragment_twist_done(frag))
            mark_fragment_twist_done(frag)
            self.assertTrue(is_fragment_twist_done(frag))
            clear_fragment_twist_done(frag)
            self.assertFalse(is_fragment_twist_done(frag))

    def test_pack_rotatable_bond_batches_conservative(self):
        from ffpopt.workflows.BondBatches import (
            adjacency_from_topology_bonds,
            pack_rotatable_bond_batches,
            should_batch_bonds,
        )

        # Linear chain 0-1-2-3-4-5-6-7: rotatable centrals (0,1),(2,3),(4,5),(6,7)
        topo = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
        adj = adjacency_from_topology_bonds(topo)
        bonds = [(0, 1), (2, 3), (4, 5), (6, 7)]
        self.assertTrue(should_batch_bonds(len(bonds), max_batch=2))
        batches = pack_rotatable_bond_batches(
            bonds, adj, max_batch=2, couple_radius=2
        )
        # All bonds land in some batch; no empty batches; size capped at 2.
        flat = [b for batch in batches for b in batch]
        self.assertEqual(len(flat), 4)
        self.assertTrue(all(len(batch) <= 2 for batch in batches))
        # Nearby (0,1) and (2,3) should prefer the same or adjacent batches.
        self.assertGreaterEqual(len(batches), 2)

        # Two distant bonds -> can be separate components but still <= max_batch each.
        far = pack_rotatable_bond_batches(
            [(0, 1), (6, 7)], adj, max_batch=2, couple_radius=1
        )
        self.assertEqual(sum(len(b) for b in far), 2)

    def test_fragment_uses_correlated_twist(self):
        from ffpopt.runtime.EnvDefaults import clear_defaults_cache
        from ffpopt.workflows.BondBatches import (
            adjacency_from_topology_bonds,
            fragment_uses_correlated_twist,
            max_bonds_per_twist_batch,
            pack_rotatable_bond_batches,
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FFPOPT_MAX_BONDS_PER_TWIST", None)
            os.environ.pop("FFPOPT_WHOLE_MAX_BONDS_PER_TWIST", None)
            os.environ.pop("FFPOPT_DEFAULTS", None)
            clear_defaults_cache()
            try:
                self.assertFalse(fragment_uses_correlated_twist(1))
                self.assertFalse(fragment_uses_correlated_twist(2))
                self.assertTrue(fragment_uses_correlated_twist(3))
                self.assertTrue(fragment_uses_correlated_twist(5))
                self.assertEqual(max_bonds_per_twist_batch(whole=False), 2)
                self.assertEqual(max_bonds_per_twist_batch(whole=True), 8)
            finally:
                clear_defaults_cache()

        # A 5-bond fragment packs as one joint batch under the whole-ligand cap.
        topo = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]
        adj = adjacency_from_topology_bonds(topo)
        bonds = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
        batches = pack_rotatable_bond_batches(
            bonds, adj, max_batch=8, couple_radius=2
        )
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 5)

    def test_partition_fragment_twist_jobs(self):
        from alps.workflows.FragmentedTwist import (
            partition_fragment_twist_jobs,
        )

        cheap, correlated = partition_fragment_twist_jobs(
            [
                {"fragment_id": "f1", "bonds": [(0, 1)]},
                {"fragment_id": "f2", "bonds": [(0, 1), (1, 2), (2, 3)]},
                {"fragment_id": "f3", "bonds": [(0, 1), (2, 3)]},
                {"fragment_id": "f4", "bonds": []},
            ]
        )
        self.assertEqual([j["fragment_id"] for j in cheap], ["f1", "f3", "f4"])
        self.assertEqual([j["fragment_id"] for j in correlated], ["f2"])
        empty_cheap, empty_corr = partition_fragment_twist_jobs([])
        self.assertEqual(empty_cheap, [])
        self.assertEqual(empty_corr, [])

    def test_whole_ligand_max_bonds_per_twist_batch(self):
        from ffpopt.runtime.EnvDefaults import clear_defaults_cache
        from ffpopt.workflows.BondBatches import max_bonds_per_twist_batch

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FFPOPT_MAX_BONDS_PER_TWIST", None)
            os.environ.pop("FFPOPT_WHOLE_MAX_BONDS_PER_TWIST", None)
            os.environ.pop("FFPOPT_DEFAULTS", None)
            clear_defaults_cache()
            try:
                self.assertEqual(max_bonds_per_twist_batch(whole=False), 2)
                self.assertEqual(max_bonds_per_twist_batch(whole=True), 8)
            finally:
                clear_defaults_cache()


# ---------------------------------------------------------------------------
# Parametrization defaults isolation
# ---------------------------------------------------------------------------


class TestGitTrackedModuleCase(unittest.TestCase):
    """Catch Windows case-only renames that Linux checkouts still see as snake_case."""

    _REQUIRED = (
        "__init__.py",
        "Log.py",
        "companions.py",
        "ffpopt_bridge.py",
        "cli/Banner.py",
        "cli/LigGetParam.py",
        "cli/LigDihedCorrect.py",
        "cli/LigScission.py",
        "workflows/FragmentedTwist.py",
        "workflows/WholeLigandTwist.py",
        "stages/FfpoptDihed.py",
        "pyproject.toml",
    )

    def test_pascalcase_modules_are_tracked_exactly(self):
        import shutil
        import subprocess

        from _paths import alps_checkout

        root = alps_checkout()
        missing_disk = [path for path in self._REQUIRED if not (root / path).is_file()]
        self.assertEqual(missing_disk, [], "ALPS modules missing on disk")
        if shutil.which("git") is None:
            self.skipTest("git checkout required")
        raw = subprocess.check_output(
            ["git", "ls-files", "."], cwd=root, text=True
        ).splitlines()
        tracked = {p.replace("\\", "/") for p in raw}
        for prefix in ("alps-main/", "alps/"):
            tracked = {
                p[len(prefix) :] if p.startswith(prefix) else p for p in tracked
            }
        if not tracked:
            self.skipTest("ALPS is not git-tracked yet")
        required = {p.replace("\\", "/") for p in self._REQUIRED}
        tracked_wrong = [
            path
            for path in tracked
            if path in required and path not in self._REQUIRED
        ]
        self.assertEqual(
            tracked_wrong,
            [],
            "git index still has lowercase names; Linux imports will fail. "
            "Use two-step git mv (file.py -> tmp -> File.py).",
        )


class TestSrcImportGraph(unittest.TestCase):
    """Every ALPS import of ffpopt / ligandparam / scission must resolve."""

    _TOP = frozenset({"ffpopt", "ligandparam", "scission", "alps"})
    _SKIP_DIRS = frozenset({"tests", "docs", "__pycache__"})

    def test_in_tree_imports_resolve(self):
        import ast
        import os

        from _paths import alps_checkout

        root = alps_checkout()
        modules: set[str] = {"alps"}
        files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in self._SKIP_DIRS]
            rel = Path(dirpath).relative_to(root)
            py_files = [f for f in filenames if f.endswith(".py")]
            prefix = ["alps", *rel.parts] if rel.parts else ["alps"]
            if "__init__.py" in filenames:
                modules.add(".".join(prefix))
            for name in py_files:
                fp = Path(dirpath) / name
                files.append(fp)
                if name == "__init__.py":
                    modules.add(".".join(prefix))
                else:
                    modules.add(".".join(prefix + [name[:-3]]))

        broken: list[str] = []
        for fp in files:
            text = fp.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(text, filename=str(fp))
            rel = fp.relative_to(root)
            parts = list(rel.parts)
            if parts[-1] == "__init__.py":
                file_pkg_parts = ["alps", *parts[:-1]]
            else:
                file_pkg_parts = ["alps", *parts[:-1]]
            pkg_parts = [p for p in file_pkg_parts if p]
            for node in ast.walk(tree):
                tgt = None
                if isinstance(node, ast.ImportFrom):
                    if node.level:
                        if node.level == 1:
                            base = pkg_parts
                        else:
                            trim = node.level - 1
                            if trim > len(pkg_parts):
                                broken.append(
                                    f"{rel}:{node.lineno} relative past package root"
                                )
                                continue
                            base = pkg_parts[: len(pkg_parts) - trim]
                        if node.module:
                            tgt = (
                                ".".join(base + node.module.split("."))
                                if base
                                else node.module
                            )
                        else:
                            tgt = ".".join(base)
                    elif node.module and node.module.split(".")[0] in self._TOP:
                        tgt = node.module
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if (
                            alias.name.split(".")[0] == "alps"
                            and alias.name not in modules
                        ):
                            broken.append(f"{rel}:{node.lineno} import {alias.name}")
                    continue
                if (
                    tgt
                    and tgt.split(".")[0] == "alps"
                    and tgt not in modules
                ):
                    broken.append(f"{rel}:{node.lineno} {tgt}")
        self.assertEqual(broken, [], "unresolved in-tree alps imports:\n" + "\n".join(broken))

    def _banned_top_level_imports(self, package_root: Path, banned: frozenset[str]) -> list[str]:
        import ast

        hits: list[str] = []
        if not package_root.is_dir():
            return hits
        for fp in package_root.rglob("*.py"):
            if "__pycache__" in fp.parts or "tests" in fp.parts:
                continue
            text = fp.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(text, filename=str(fp))
            except SyntaxError:
                continue
            rel = fp.relative_to(package_root)
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    top = name.split(".")[0]
                    if top in banned:
                        hits.append(f"{rel}:{node.lineno} {name}")
        return hits

    def test_ligandparam_does_not_import_ffpopt_scission_or_alps(self):
        from _paths import sibling_checkout

        root = sibling_checkout("ligandparam")
        if root is None:
            self.skipTest("ligandparam not checked out")
        hits = self._banned_top_level_imports(
            root, frozenset({"ffpopt", "scission", "alps"})
        )
        self.assertEqual(hits, [], "ligandparam imports companions:\n" + "\n".join(hits))

    def test_scission_does_not_import_ffpopt_ligandparam_or_alps(self):
        from _paths import sibling_checkout

        root = sibling_checkout("scission")
        if root is None:
            self.skipTest("scission not checked out")
        hits = self._banned_top_level_imports(
            root, frozenset({"ffpopt", "ligandparam", "alps"})
        )
        self.assertEqual(hits, [], "scission imports companions:\n" + "\n".join(hits))

    def test_ffpopt_does_not_import_scission_ligandparam_or_alps(self):
        from _paths import sibling_checkout

        checkout = sibling_checkout("ffpopt")
        if checkout is None:
            self.skipTest("ffpopt not checked out")
        root = checkout / "src" / "python" / "lib" / "ffpopt"
        if not root.is_dir():
            self.skipTest("ffpopt python package not found")
        hits = self._banned_top_level_imports(
            root, frozenset({"ligandparam", "alps"})
        )
        self.assertEqual(hits, [], "ffpopt imports ALPS/ligandparam:\n" + "\n".join(hits))


@unittest.skipUnless(_have_split_ffpopt(), "split src/ffpopt not present (using independent ffpopt)")
class TestGaussianOrientationBudget(unittest.TestCase):
    def test_mem_is_split_not_replicated(self):
        from ffpopt.runtime.FastWavefront import (
            split_core_budget,
            split_gaussian_orientation_budget,
        )

        # Old breadth+flatten plan: 28 workers x 1 core, each with the
        # full --mem (the Triton OOM).
        self.assertEqual(split_core_budget(60, 28), (28, 1))

        n_workers, job_nproc, job_mem = split_gaussian_orientation_budget(
            60, 28, 32
        )
        self.assertGreaterEqual(job_mem, 4)
        self.assertLessEqual(n_workers * job_nproc, 60)
        self.assertLessEqual(n_workers * job_mem, 32)
        self.assertLess(n_workers, 28)
        self.assertGreater(job_nproc, 1)

    def test_tiny_mem_stays_serial(self):
        from ffpopt.runtime.FastWavefront import split_gaussian_orientation_budget

        n_workers, job_nproc, job_mem = split_gaussian_orientation_budget(
            60, 28, 1
        )
        self.assertEqual((n_workers, job_nproc, job_mem), (1, 60, 1))


_COMPANION_ENV = (
    "LIGANDPARAM_FFPOPT",
    "LIGANDPARAM_SCISSION",
    "LIGANDPARAM_FFPOPT_PATH",
    "LIGANDPARAM_SCISSION_PATH",
    "ALPS_FFPOPT_PATH",
    "ALPS_SCISSION_PATH",
    "ALPS_LIGANDPARAM_PATH",
    "PYTHONPATH",
)


class TestCompanionResolver(unittest.TestCase):
    """Sibling auto-bind vs explicit ALPS_*_PATH."""

    def setUp(self):
        self._env = {key: os.environ.get(key) for key in _COMPANION_ENV}
        self._sys_path = list(sys.path)
        self.addCleanup(self._restore)

    def _restore(self):
        from alps.companions import bootstrap, reset_for_tests

        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        sys.path[:] = self._sys_path
        reset_for_tests()
        bootstrap()
        importlib.invalidate_caches()
        ffpopt = importlib.import_module("ffpopt")
        scission = importlib.import_module("scission")
        self.assertFalse(getattr(ffpopt, "__ligandparam_bundle__", False))
        self.assertFalse(getattr(scission, "__ligandparam_bundle__", False))

    def _stub_package(self, parent: Path, name: str, token: str) -> None:
        pkg = parent / name
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text(f"TOKEN = {token!r}\n", encoding="utf-8")

    def test_default_is_sibling_or_installed(self):
        from alps.companions import companion_status, repo_root

        status = companion_status()
        root = repo_root()
        for name in ("ffpopt", "scission", "ligandparam"):
            info = status[name]
            self.assertIn(info.mode, {"sibling", "installed", "path"}, name)
            if info.mode == "sibling":
                self.assertTrue(str(info.origin).startswith(str(root)), name)

    def test_mode_words_are_ignored(self):
        from alps.companions import bootstrap, reset_for_tests

        os.environ["LIGANDPARAM_FFPOPT"] = "external"
        os.environ.pop("LIGANDPARAM_FFPOPT_PATH", None)
        os.environ.pop("ALPS_FFPOPT_PATH", None)
        reset_for_tests()
        info = bootstrap()
        self.assertIn(info["ffpopt"].mode, {"sibling", "installed"})

    def test_path_override_loads_stub(self):
        from alps.companions import bootstrap, reset_for_tests

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src"
            self._stub_package(src, "ffpopt", "ext-ffpopt")
            os.environ["ALPS_FFPOPT_PATH"] = str(src)
            reset_for_tests()
            info = bootstrap()
            self.assertEqual(info["ffpopt"].mode, "path")
            ffpopt = importlib.import_module("ffpopt")
            self.assertEqual(ffpopt.TOKEN, "ext-ffpopt")

    def test_path_may_be_package_dir(self):
        from alps.companions import bootstrap, reset_for_tests

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src"
            self._stub_package(src, "scission", "ext-scission")
            os.environ["ALPS_SCISSION_PATH"] = str(src / "scission")
            reset_for_tests()
            info = bootstrap()
            self.assertEqual(info["scission"].mode, "path")
            scission = importlib.import_module("scission")
            self.assertEqual(scission.TOKEN, "ext-scission")


if __name__ == "__main__":
    unittest.main()
