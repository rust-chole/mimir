"""Tiny console logger (stdout, UTF-8 safe)."""
from __future__ import annotations

import sys
import time

_START = time.perf_counter()
_QUIET = False


def set_quiet(quiet: bool) -> None:
    global _QUIET
    _QUIET = bool(quiet)


def info(message: str) -> None:
    if _QUIET:
        return
    elapsed = time.perf_counter() - _START
    line = f"[{elapsed:7.1f}s] {message}"
    try:
        sys.stdout.write(line + "\n")
    except UnicodeEncodeError:
        sys.stdout.write(line.encode("ascii", "replace").decode("ascii") + "\n")
    sys.stdout.flush()


def warn(message: str) -> None:
    info(f"WARNING: {message}")
