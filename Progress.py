"""Live ASCII twist boards, matching ligandparam's recipe stdout board.

``lig-getparam`` reprints ``RECIPE_STATUS.txt`` on stdout while antechamber
and Gaussian run silently. ``lig-dihed-correct`` does the same for fragment
/ whole-ligand twist: ``FRAG_STATUS.txt`` or ``WHOLE_STATUS.txt``, plus a
run card and a raw copy of ffpopt ``print`` lines into per-job logs.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

PathLike = str | Path


def progress_stream() -> TextIO:
    """Real process stdout (not a fragment tee), same as the recipe board."""
    return sys.__stdout__


def print_fragmented_run_card(
    *,
    ligand: str,
    model: str,
    nproc: int,
    n_fragments: int,
    work_dir: PathLike,
    stream: TextIO | None = None,
) -> None:
    """ASCII FRAGMENTED TWIST card on stdout (before wavefront spam)."""
    from ligandparam.runtime.Console import (
        format_fragmented_run_banner,
        print_run_banner,
    )

    print_run_banner(
        format_fragmented_run_banner(
            ligand=ligand,
            model=model,
            nproc=int(nproc),
            n_fragments=int(n_fragments),
            work_dir=str(work_dir),
        ),
        stream=stream if stream is not None else progress_stream(),
    )


def print_whole_run_card(
    *,
    ligand: str,
    model: str,
    nproc: int,
    delta: int,
    n_bonds: int,
    work_dir: PathLike,
    stream: TextIO | None = None,
) -> None:
    """ASCII WHOLE-LIGAND TWIST card on stdout."""
    from ligandparam.runtime.Console import (
        format_whole_ligand_run_banner,
        print_run_banner,
    )

    print_run_banner(
        format_whole_ligand_run_banner(
            ligand=ligand,
            model=model,
            nproc=int(nproc),
            delta=int(delta),
            n_bonds=int(n_bonds),
            work_dir=str(work_dir),
        ),
        stream=stream if stream is not None else progress_stream(),
    )


def make_fragment_board(out_dir: PathLike, *, logger, stream: TextIO | None = None):
    """Return ``(store, watcher)`` or ``(None, None)`` if the board cannot start."""
    from ligandparam.runtime.ProgressBoard import (
        FragmentProgressStore,
        make_board_watcher,
    )

    out_dir = Path(out_dir)
    try:
        (out_dir / ".frag_progress.json").unlink()
    except FileNotFoundError:
        pass
    try:
        store = FragmentProgressStore(out_dir / ".frag_progress.json")
        watcher = make_board_watcher(
            "fragment",
            store,
            board_path=out_dir / "FRAG_STATUS.txt",
            logger=logger,
            interval_sec=5.0,
            heartbeat_sec=20.0,
            stream=stream if stream is not None else progress_stream(),
            log_root_hint=str(out_dir),
        )
    except Exception as exc:
        if logger is not None:
            try:
                logger.warning("Could not start fragment progress board: %s", exc)
            except Exception:
                pass
        return None, None
    return store, watcher


def make_whole_board(out_dir: PathLike, *, logger, stream: TextIO | None = None):
    """Return ``(store, watcher)`` or ``(None, None)`` if the board cannot start."""
    from ligandparam.runtime.ProgressBoard import (
        WholeProgressStore,
        make_board_watcher,
    )

    out_dir = Path(out_dir)
    try:
        (out_dir / ".whole_progress.json").unlink()
    except FileNotFoundError:
        pass
    try:
        store = WholeProgressStore(out_dir / ".whole_progress.json")
        watcher = make_board_watcher(
            "whole",
            store,
            board_path=out_dir / "WHOLE_STATUS.txt",
            logger=logger,
            interval_sec=5.0,
            heartbeat_sec=20.0,
            stream=stream if stream is not None else progress_stream(),
            log_root_hint=str(out_dir),
        )
    except Exception as exc:
        if logger is not None:
            try:
                logger.warning("Could not start whole-ligand progress board: %s", exc)
            except Exception:
                pass
        return None, None
    return store, watcher


class _MirrorTextIO:
    """Write the same bytes to a job log and the parent stdout/stderr."""

    def __init__(self, file_stream: TextIO, console_stream: TextIO) -> None:
        self.file_stream = file_stream
        self.console_stream = console_stream
        self.encoding = getattr(console_stream, "encoding", None) or "utf-8"
        self._console = console_stream

    def write(self, data: str) -> int:
        if not data:
            return 0
        try:
            self.file_stream.write(data)
        except OSError:
            pass
        return self._console.write(data)

    def flush(self) -> None:
        try:
            self.file_stream.flush()
        except OSError:
            pass
        try:
            self._console.flush()
        except OSError:
            pass

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def fileno(self) -> int:
        return self.file_stream.fileno()

    def __getattr__(self, name: str):
        return getattr(self._console, name)


@contextmanager
def tee_job_stdio(log_path: PathLike) -> Iterator[None]:
    """Copy stdout/stderr to ``log_path`` without extra prefixes.

    ffpopt ``[twist]`` / ``[wavefront]`` lines stay on ALPS stdout as-is
    (ligandparam's stream contract is also un-timestamped) and also land in
    the per-fragment / parent job log.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _MirrorTextIO(fh, old_out)  # type: ignore[assignment]
    sys.stderr = _MirrorTextIO(fh, old_err)  # type: ignore[assignment]
    try:
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except OSError:
            pass
        sys.stdout = old_out
        sys.stderr = old_err
        try:
            fh.close()
        except OSError:
            pass
