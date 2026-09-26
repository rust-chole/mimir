"""Safe FFmpeg/ffprobe execution and filter-argument escaping.

Commands are always argument lists (never a shell). Paths embedded inside a
filter graph (``ass=``/``subtitles=``) go through :func:`escape_filter_path`,
which handles drive colons, backslashes, quotes and graph separators.
"""
from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from mimir.errors import MediaError


def ffmpeg_bin() -> str:
    path = os.getenv("MIMIR_FFMPEG") or shutil.which("ffmpeg")
    if not path:
        raise MediaError("ffmpeg not found on PATH (set MIMIR_FFMPEG)")
    return path


def ffprobe_bin() -> str:
    path = os.getenv("MIMIR_FFPROBE") or shutil.which("ffprobe")
    if not path:
        raise MediaError("ffprobe not found on PATH (set MIMIR_FFPROBE)")
    return path


def run(args: Sequence[str], *, timeout: float | None = None, input_bytes: bytes | None = None,
        capture_stdout: bool = False, what: str = "ffmpeg") -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            list(args),
            input=input_bytes,
            stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as error:
        raise MediaError(f"{what} executable not found: {args[0]}") from error
    except subprocess.TimeoutExpired as error:
        raise MediaError(f"{what} timed out after {timeout}s") from error
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", "replace").strip().splitlines()
        tail = "\n".join(stderr[-25:])
        raise MediaError(f"{what} failed (exit {result.returncode}):\n{tail}")
    return result


def ffmpeg(args: Sequence[str], **kwargs) -> subprocess.CompletedProcess:
    return run([ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", *args], **kwargs)


def escape_filter_path(path: str | Path) -> str:
    """Escape a path for use as a filter option value inside single quotes.

    ``C:\\Users\\a'b\\x.ass`` -> ``C\\:/Users/a'\\''b/x.ass`` (the quote closes,
    is escaped, and reopens). Forward slashes avoid backslash escape ambiguity.
    """
    text = str(Path(path).resolve()).replace("\\", "/")
    text = text.replace(":", "\\:")
    text = text.replace("'", "'\\''")
    return text


@functools.lru_cache(maxsize=1)
def ffmpeg_major() -> int:
    result = run([ffmpeg_bin(), "-hide_banner", "-version"], capture_stdout=True, what="ffmpeg -version")
    match = re.search(r"ffmpeg version n?(\d+)", result.stdout.decode("utf-8", "replace"))
    return int(match.group(1)) if match else 0


def filter_complex_script(path: str | Path) -> list[str]:
    """Filter graph from a file (no command-line length limits on Windows)."""
    if ffmpeg_major() >= 7:
        return ["-/filter_complex", str(path)]
    return ["-filter_complex_script", str(path)]


def filter_has(name: str) -> bool:
    result = run([ffmpeg_bin(), "-hide_banner", "-filters"], capture_stdout=True, what="ffmpeg -filters")
    for line in result.stdout.decode("utf-8", "replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == name:
            return True
    return False
