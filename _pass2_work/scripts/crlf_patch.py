"""Exact-anchor patcher for CRLF files (edits are written in LF, file keeps CRLF).

Usage from other scripts:  apply(path, [(old_lf, new_lf), ...])
Each ``old`` must occur exactly once in the LF-normalized text; the result is
compiled (Python files) before it is written back with CRLF line endings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence


def apply(path: str | Path, replacements: Sequence[tuple[str, str]]) -> None:
    target = Path(path)
    raw = target.read_bytes()
    if raw.count(b"\n") != raw.count(b"\r\n"):
        raise SystemExit(f"{target}: expected pure CRLF line endings")
    text = raw.decode("utf-8").replace("\r\n", "\n")
    for old, new in replacements:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"{target}: anchor found {count} times:\n{old[:300]}")
        text = text.replace(old, new)
    if target.suffix == ".py":
        compile(text, str(target), "exec", dont_inherit=True)
    target.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
