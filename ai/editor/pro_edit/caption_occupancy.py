"""Visual ACTIVITY occupancy for caption placement (local, deterministic).

This is motion / activity evidence, NOT object detection: it knows where
pixels change, not what they are. Per sample: blurred grayscale frame
difference, pooled to a coarse grid, minus the frame's median cell (removes
global camera motion and noise floor); scene cuts are skipped. Values are
normalized per clip (95th percentile) to [0, 1] and quantized to bytes.

Sampling: ~6 samples/s of the selected paced clip only (never the VOD), at
<= 320 px width, through the existing FFmpeg frame reader. The result is cached
in a versioned sidecar keyed by source content identity, clip timing identity,
analysis version and configuration.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

OCCUPANCY_VERSION = 1
DEFAULT_SAMPLE_FPS = 6.0
DEFAULT_COLS = 16
ANALYSIS_MAX_WIDTH = 320
CUT_MEAN_DIFF = 38.0          # mean abs difference (0-255) above which a sample is a scene cut
_IDENTITY_CHUNK = 1 << 20


@dataclass(frozen=True)
class VisualOccupancyMap:
    cols: int
    rows: int
    sample_fps: float
    times: tuple[float, ...]
    cells: tuple[bytes, ...]          # row-major uint8 activity per sample
    cut_samples: int
    fingerprint: str
    version: int = OCCUPANCY_VERSION

    @property
    def confidence(self) -> float:
        total = len(self.times) + self.cut_samples
        return 0.0 if total == 0 else len(self.times) / total

    def window(self, start: float, end: float) -> list[float]:
        """Mean activity per cell over samples in [start, end] (0..1)."""
        picked = [cells for t, cells in zip(self.times, self.cells) if start - 1e-6 <= t <= end + 1e-6]
        if not picked:
            # nearest sample (short windows between samples)
            if not self.times:
                return [0.0] * (self.cols * self.rows)
            mid = (start + end) / 2.0
            index = min(range(len(self.times)), key=lambda i: abs(self.times[i] - mid))
            if abs(self.times[index] - mid) > 1.0 / self.sample_fps:
                return [0.0] * (self.cols * self.rows)
            picked = [self.cells[index]]
        size = self.cols * self.rows
        return [sum(cells[i] for cells in picked) / (255.0 * len(picked)) for i in range(size)]

    def region_score(self, box: Sequence[float], values: Sequence[float]) -> float:
        """Area-weighted mean activity of normalized box (x0, y0, x1, y1)."""
        x0, y0, x1, y1 = (min(1.0, max(0.0, v)) for v in box)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        total = weight = 0.0
        for row in range(self.rows):
            cy0, cy1 = row / self.rows, (row + 1) / self.rows
            h = min(y1, cy1) - max(y0, cy0)
            if h <= 0:
                continue
            for col in range(self.cols):
                cx0, cx1 = col / self.cols, (col + 1) / self.cols
                w = min(x1, cx1) - max(x0, cx0)
                if w <= 0:
                    continue
                total += values[row * self.cols + col] * w * h
                weight += w * h
        return total / weight if weight else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "fingerprint": self.fingerprint, "cols": self.cols, "rows": self.rows,
                "sample_fps": self.sample_fps, "times": [round(t, 3) for t in self.times],
                "cells": [c.hex() for c in self.cells], "cut_samples": self.cut_samples,
                "kind": "motion activity (frame difference), not object detection"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisualOccupancyMap":
        return cls(cols=int(data["cols"]), rows=int(data["rows"]), sample_fps=float(data["sample_fps"]),
                   times=tuple(float(t) for t in data["times"]),
                   cells=tuple(bytes.fromhex(c) for c in data["cells"]), cut_samples=int(data["cut_samples"]),
                   fingerprint=str(data["fingerprint"]), version=int(data["version"]))


def source_identity(path: str | Path) -> dict[str, Any]:
    """Content identity (size + head/tail hash): stable across copies/mtime."""
    file = Path(path)
    size = file.stat().st_size
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        digest.update(handle.read(_IDENTITY_CHUNK))
        if size > 2 * _IDENTITY_CHUNK:
            handle.seek(size - _IDENTITY_CHUNK)
            digest.update(handle.read(_IDENTITY_CHUNK))
    return {"size": size, "sha256_head_tail": digest.hexdigest()}


def occupancy_fingerprint(*, source: dict[str, Any], clip_identity: str, width: int, height: int,
                          frame_count: int, fps: str, sample_fps: float, cols: int) -> str:
    payload = {"version": OCCUPANCY_VERSION, "source": source, "clip_identity": clip_identity, "size": [width, height],
               "frames": frame_count, "fps": fps, "sample_fps": sample_fps, "cols": cols,
               "max_width": ANALYSIS_MAX_WIDTH, "cut": CUT_MEAN_DIFF}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def grid_rows(width: int, height: int, cols: int) -> int:
    return max(6, min(32, int(round(cols * height / max(1, width)))))


def analyze_frames(frames: Iterable[tuple[Any, float]], *, cols: int, rows: int, sample_fps: float,
                   fingerprint: str) -> VisualOccupancyMap:
    """Activity map from streamed (BGR image, t) pairs; keeps only pooled grids in memory."""
    import numpy as np

    from ai.editor.pro_edit.vision.cv_runtime import load_opencv

    cv2 = load_opencv()

    raw: list[Any] = []
    kept_times: list[float] = []
    cuts = 0
    previous = None
    for image, t in frames:
        gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
        if previous is None:
            previous = gray
            continue
        diff = np.abs(gray - previous)
        previous = gray
        if float(diff.mean()) > CUT_MEAN_DIFF:
            cuts += 1
            continue
        pooled = cv2.resize(diff, (cols, rows), interpolation=cv2.INTER_AREA)
        pooled = np.maximum(pooled - float(np.median(pooled)), 0.0)
        raw.append(pooled)
        kept_times.append(float(t))
    if raw:
        stack = np.stack(raw)
        positive = stack[stack > 0]
        scale = float(np.percentile(positive, 95)) if positive.size else 0.0
        scale = max(scale, 4.0)                       # below ~4 grey levels is sensor noise
        quantized = [bytes(np.clip(frame / scale * 255.0, 0, 255).astype(np.uint8).ravel()) for frame in raw]
    else:
        quantized = []
    return VisualOccupancyMap(cols, rows, sample_fps, tuple(kept_times), tuple(quantized), cuts, fingerprint)


def analyze_clip(path: str | Path, width: int, height: int, *, fingerprint: str,
                 sample_fps: float = DEFAULT_SAMPLE_FPS, cols: int = DEFAULT_COLS) -> VisualOccupancyMap:
    from ai.editor.pro_edit.vision.frames import iterate_frames

    rows = grid_rows(width, height, cols)
    stream = ((sample.image, sample.t) for sample in
              iterate_frames(path, width, height, sample_fps=sample_fps, max_width=ANALYSIS_MAX_WIDTH))
    return analyze_frames(stream, cols=cols, rows=rows, sample_fps=sample_fps, fingerprint=fingerprint)


def load_or_analyze(cache_path: Path, media: Any, clip_identity: str, *, force: bool = False,
                    sample_fps: float = DEFAULT_SAMPLE_FPS, cols: int = DEFAULT_COLS
                    ) -> tuple[VisualOccupancyMap, str]:
    """(map, status) with status "cached" | "analyzed". Raises on analysis failure."""
    fingerprint = occupancy_fingerprint(
        source=source_identity(media.path), clip_identity=clip_identity, width=media.width, height=media.height,
        frame_count=media.frame_count, fps=str(media.fps), sample_fps=sample_fps, cols=cols)
    if not force and cache_path.is_file():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if data.get("fingerprint") == fingerprint and int(data.get("version", -1)) == OCCUPANCY_VERSION:
                return VisualOccupancyMap.from_dict(data), "cached"
        except (OSError, ValueError, KeyError, TypeError):
            pass
    occupancy = analyze_clip(media.path, media.width, media.height, fingerprint=fingerprint,
                             sample_fps=sample_fps, cols=cols)
    if not occupancy.times:
        raise ValueError("no usable activity samples")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp = cache_path.with_name(cache_path.name + ".tmp")
    temp.write_text(json.dumps(occupancy.to_dict(), separators=(",", ":")), encoding="utf-8")
    os.replace(temp, cache_path)
    return occupancy, "analyzed"


@dataclass(frozen=True)
class OutputMappedOccupancy:
    """Occupancy (source frame) queried with OUTPUT-frame boxes (base-window crop)."""

    inner: VisualOccupancyMap
    base: tuple[int, int, int, int]
    source_size: tuple[int, int]

    def window(self, start: float, end: float) -> list[float]:
        return self.inner.window(start, end)

    def region_score(self, box: Sequence[float], values: Sequence[float]) -> float:
        bx, by, bw, bh = self.base
        width, height = self.source_size
        x0, y0, x1, y1 = box
        mapped = ((bx + x0 * bw) / width, (by + y0 * bh) / height, (bx + x1 * bw) / width, (by + y1 * bh) / height)
        return self.inner.region_score(mapped, values)


def activity_is_finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(v) for v in values)
