"""Deterministic JSON I/O and content hashing."""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses / enums / paths / tuples into plain JSON values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("NaN/Inf is not valid artifact data")
        return value
    if hasattr(value, "item") and callable(value.item):  # numpy scalars
        return to_jsonable(value.item())
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(to_jsonable(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sampled_file_hash(path: str | Path, sample: int = 4 << 20) -> str:
    """Fast identity for large source media: size + first/middle/last blocks."""
    path = Path(path)
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with path.open("rb") as handle:
        for offset in sorted({0, max(0, size // 2 - sample // 2), max(0, size - sample)}):
            handle.seek(offset)
            digest.update(handle.read(sample))
    return digest.hexdigest()


def write_json(path: str | Path, data: Any) -> Path:
    """Atomic, pretty, deterministic JSON write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(to_jsonable(data), ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
