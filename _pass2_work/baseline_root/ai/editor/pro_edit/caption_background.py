"""Caption BACKGROUND analysis: legibility statistics + TEXT_LIKE occupancy.

Local and deterministic, one sparse decode of the selected paced clip (never
the VOD): ~2 samples/s at <= 640 px through the existing FFmpeg frame reader.
Per sample and per grid cell it keeps

* relative luminance (linear sRGB, WCAG) mean / 10th / 90th percentile,
* edge density (fraction of strong-gradient pixels = background complexity),
* TEXT_LIKE coverage: fraction of the cell inside components that look like
  rows of glyph strokes (gradient -> threshold -> horizontal closing ->
  connected components filtered by height, elongation, fill and stroke
  transitions). This is text-LIKE geometry, NOT OCR: nothing is recognised.

and per sample the positions of full-span straight lines (panel boundaries).
Temporal persistence (per scene segment) turns per-sample text-like cells into
occupancy: a cell must be text-like in at least half of a segment's samples,
so transient texture never becomes permanent UI.

Cached in a versioned sidecar keyed by source content identity, clip
identity, analysis version and configuration (same scheme as the activity map).
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ai.editor.pro_edit.caption_occupancy import source_identity

BACKGROUND_VERSION = 1
DEFAULT_SAMPLE_FPS = 2.0
ANALYSIS_MAX_WIDTH = 640
DEFAULT_COLS = 24
CUT_MEAN_DIFF = 45.0            # mean abs grey difference between samples (0.5 s apart) = new segment
EDGE_THRESHOLD = 48.0           # gradient magnitude (0-255 grey) counted as an edge
TEXT_CELL_COVERAGE = 0.12       # cell counts as text-like in a sample above this coverage
TEXT_PERSISTENCE = 0.5          # ... and as occupancy when text-like in >= half of its segment's samples
MIN_SEGMENT_SAMPLES = 3
STATIC_CELL_DIFF = 6.0          # mean abs grey difference below which a cell is static between samples
TEXT_BIMODALITY = 0.8          # Otsu separability of a component's pixels
PANEL_LINE_SPAN = 0.7           # fraction of a column/row that must be edge for a panel line
PANEL_PERSISTENCE = 0.6
_CHANNELS = ("lum_mean", "lum_p10", "lum_p90", "edges", "text", "diff")


def srgb_to_linear(value: float) -> float:
    c = value / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(r: float, g: float, b: float) -> float:
    """WCAG relative luminance of an sRGB colour (0-255 channels)."""
    return 0.2126 * srgb_to_linear(r) + 0.7152 * srgb_to_linear(g) + 0.0722 * srgb_to_linear(b)


def _pack(values: bytes) -> str:
    return base64.b64encode(zlib.compress(values, 6)).decode("ascii")


def _unpack(text: str) -> bytes:
    return zlib.decompress(base64.b64decode(text.encode("ascii")))


@dataclass(frozen=True)
class BackgroundSample:
    t: float
    lum_mean: bytes      # per cell, linear relative luminance x 255
    lum_p10: bytes
    lum_p90: bytes
    edges: bytes         # edge density x 255
    text: bytes          # text-like coverage x 255
    diff: bytes          # mean abs grey difference to the previous sample (0 for a segment's first)
    segment: int
    panel_x: tuple[float, ...] = ()     # normalized x of vertical full-span lines
    panel_y: tuple[float, ...] = ()


@dataclass(frozen=True)
class RegionStats:
    """Background under a caption region over a time window (output frame)."""

    samples: int
    lum_p10: float
    lum_p50: float
    lum_p90: float
    edge_density: float
    text_coverage: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {"samples": self.samples, "lum_p10": round(self.lum_p10, 4), "lum_p50": round(self.lum_p50, 4),
                "lum_p90": round(self.lum_p90, 4), "edge_density": round(self.edge_density, 4),
                "text_coverage": round(self.text_coverage, 4), "confidence": round(self.confidence, 3)}


def _box_weights(box: Sequence[float], cols: int, rows: int) -> list[tuple[int, float]]:
    x0, y0, x1, y1 = (min(1.0, max(0.0, v)) for v in box)
    if x1 <= x0 or y1 <= y0:
        return []
    found = []
    for row in range(rows):
        h = min(y1, (row + 1) / rows) - max(y0, row / rows)
        if h <= 0:
            continue
        for col in range(cols):
            w = min(x1, (col + 1) / cols) - max(x0, col / cols)
            if w > 0:
                found.append((row * cols + col, w * h))
    return found


@dataclass(frozen=True)
class CaptionBackgroundMap:
    cols: int
    rows: int
    sample_fps: float
    samples: tuple[BackgroundSample, ...]
    cut_samples: int
    fingerprint: str
    version: int = BACKGROUND_VERSION

    # ---------------------------------------------------------------- queries
    def _in_window(self, start: float, end: float) -> list[BackgroundSample]:
        picked = [s for s in self.samples if start - 1e-6 <= s.t <= end + 1e-6]
        if picked or not self.samples:
            return picked
        mid = (start + end) / 2.0
        nearest = min(self.samples, key=lambda s: (abs(s.t - mid), s.t))
        return [nearest] if abs(nearest.t - mid) <= 1.0 / self.sample_fps else []

    def region_stats(self, start: float, end: float, box: Sequence[float]) -> RegionStats | None:
        """Luminance distribution / complexity under ``box`` (normalized, this map's frame)."""
        picked = self._in_window(start, end)
        weights = _box_weights(box, self.cols, self.rows)
        if not picked or not weights:
            return None
        p10: list[tuple[float, float]] = []
        mean: list[tuple[float, float]] = []
        p90: list[tuple[float, float]] = []
        edges = text = total = 0.0
        for sample in picked:
            for index, weight in weights:
                p10.append((sample.lum_p10[index] / 255.0, weight))
                mean.append((sample.lum_mean[index] / 255.0, weight))
                p90.append((sample.lum_p90[index] / 255.0, weight))
                edges += sample.edges[index] / 255.0 * weight
                text += sample.text[index] / 255.0 * weight
                total += weight
        span = max(end - start, 1e-3)
        expected = max(1.0, span * self.sample_fps)
        confidence = min(1.0, len(picked) / expected) if span > 1.0 / self.sample_fps else 1.0
        return RegionStats(len(picked), _weighted_quantile(p10, 0.10), _weighted_quantile(mean, 0.5),
                           _weighted_quantile(p90, 0.90), edges / total, text / total, confidence)

    def text_occupancy(self, start: float, end: float) -> list[float]:
        """Per cell persistence (0..1) of TEXT_LIKE content in the segments overlapping [start, end];
        cells below TEXT_PERSISTENCE (or segments with too few samples) are 0."""
        size = self.cols * self.rows
        best = [0.0] * size
        for segment in sorted({s.segment for s in self._in_window(start, end)}):
            for index, persistence in enumerate(self._persistence(segment)):
                if persistence > best[index]:
                    best[index] = persistence
        return best

    def _persistence(self, segment: int) -> tuple[float, ...]:
        cache: dict[int, tuple[float, ...]] = self.__dict__.setdefault("_persistence_cache", {})
        found = cache.get(segment)
        if found is None:
            size = self.cols * self.rows
            members = [s for s in self.samples if s.segment == segment]
            if len(members) < MIN_SEGMENT_SAMPLES:
                found = (0.0,) * size
            else:
                threshold = int(math.ceil(TEXT_CELL_COVERAGE * 255.0))
                values = []
                for index in range(size):
                    persistence = sum(1 for s in members if s.text[index] >= threshold) / len(members)
                    values.append(persistence if persistence >= TEXT_PERSISTENCE else 0.0)
                found = tuple(values)
            cache[segment] = found
        return found

    def static_fraction(self, start: float, end: float) -> list[float]:
        """Per cell fraction of sample pairs in the window whose pixels did not change."""
        picked = [s for s in self._in_window(start, end) if s.diff and any(s.diff)]
        size = self.cols * self.rows
        if not picked:
            return [0.0] * size
        return [sum(1 for s in picked if s.diff[i] < STATIC_CELL_DIFF) / len(picked) for i in range(size)]

    def window(self, start: float, end: float) -> list[float]:
        """Occupancy-map protocol (placement): text-like persistence per cell."""
        return self.text_occupancy(start, end)

    def region_score(self, box: Sequence[float], values: Sequence[float]) -> float:
        weights = _box_weights(box, self.cols, self.rows)
        total = sum(w for _i, w in weights)
        return sum(values[i] * w for i, w in weights) / total if total else 0.0

    def panel_lines(self, start: float, end: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Persistent full-span vertical / horizontal lines (normalized positions)."""
        picked = self._in_window(start, end)
        if len(picked) < MIN_SEGMENT_SAMPLES:
            return (), ()

        def persistent(values: list[tuple[float, ...]]) -> tuple[float, ...]:
            buckets: dict[int, int] = {}
            for positions in values:
                for key in {int(round(p * 100)) for p in positions}:
                    buckets[key] = buckets.get(key, 0) + 1
            keep = sorted(k for k, n in buckets.items() if n / len(values) >= PANEL_PERSISTENCE)
            merged: list[int] = []
            for key in keep:
                if merged and key - merged[-1] <= 1:
                    continue
                merged.append(key)
            return tuple(k / 100.0 for k in merged)

        return persistent([s.panel_x for s in picked]), persistent([s.panel_y for s in picked])

    @property
    def confidence(self) -> float:
        total = len(self.samples) + self.cut_samples
        return 0.0 if total == 0 else len(self.samples) / total

    # ---------------------------------------------------------------- io
    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "fingerprint": self.fingerprint, "cols": self.cols, "rows": self.rows,
                "sample_fps": self.sample_fps, "cut_samples": self.cut_samples,
                "kind": "luminance/edge statistics + TEXT_LIKE occupancy (no OCR, no object detection)",
                "samples": [{"t": round(s.t, 3), "segment": s.segment,
                             **{name: _pack(getattr(s, name)) for name in _CHANNELS},
                             "panel_x": list(s.panel_x), "panel_y": list(s.panel_y)} for s in self.samples]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaptionBackgroundMap":
        cols, rows = int(data["cols"]), int(data["rows"])
        samples = []
        for raw in data["samples"]:
            channels = {name: _unpack(raw[name]) for name in _CHANNELS}
            if any(len(v) != cols * rows for v in channels.values()):
                raise ValueError("background sidecar grid size mismatch")
            samples.append(BackgroundSample(float(raw["t"]), segment=int(raw["segment"]),
                                            panel_x=tuple(float(v) for v in raw.get("panel_x", [])),
                                            panel_y=tuple(float(v) for v in raw.get("panel_y", [])), **channels))
        return cls(cols, rows, float(data["sample_fps"]), tuple(samples), int(data["cut_samples"]),
                   str(data["fingerprint"]), int(data["version"]))


def _weighted_quantile(pairs: list[tuple[float, float]], q: float) -> float:
    ordered = sorted(pairs)
    total = sum(w for _v, w in ordered)
    if total <= 0:
        return 0.0
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= q * total - 1e-12:
            return value
    return ordered[-1][0]


# ============================================================
# ANALYSIS
# ============================================================

def grid_rows(width: int, height: int, cols: int) -> int:
    return max(8, min(48, int(round(cols * height / max(1, width)))))


def text_like_mask(gray: Any) -> Any:
    """uint8 mask (1 = inside a text-like component) for one grey frame."""
    import cv2
    import numpy as np

    h, w = gray.shape[:2]
    unit = max(1.0, w / 640.0)
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    otsu, _ = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary = (gradient >= max(float(otsu), EDGE_THRESHOLD)).astype(np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(9 * unit)), 1)))
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    mask = np.zeros((h, w), dtype=np.uint8)
    min_h, max_h = max(5, int(5 * unit)), max(8, int(0.09 * h))
    for index in range(1, count):
        x, y, bw, bh, area = (int(v) for v in stats[index])
        if not min_h <= bh <= max_h or bw < 1.5 * bh or bw > 0.95 * w:
            continue
        if not 0.25 <= area / float(bw * bh) <= 0.95:
            continue
        strokes = binary[y:y + bh, x:x + bw]
        density = float(strokes.mean())
        if not 0.12 <= density <= 0.80:
            continue
        # Glyph rows alternate stroke / gap many times per character height.
        transitions = float(np.count_nonzero(np.diff(strokes, axis=1))) / bh
        if transitions / max(1.0, bw / bh) < 1.2:
            continue
        if _bimodality(gray[y:y + bh, x:x + bw]) < TEXT_BIMODALITY:
            continue
        mask[y:y + bh, x:x + bw] = 1
    return mask


def _bimodality(patch: Any) -> float:
    """Otsu between-class / total variance: glyphs are two-tone against their local background."""
    import numpy as np

    values = patch.astype(np.float64).ravel()
    total = float(values.var())
    if total < 25.0:                   # < 5 grey levels of spread: no visible strokes
        return 0.0
    hist = np.bincount(patch.ravel(), minlength=256).astype(np.float64)
    prob = hist / hist.sum()
    levels = np.arange(256, dtype=np.float64)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * levels)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega))
    between = np.nan_to_num(between, nan=0.0, posinf=0.0, neginf=0.0)
    return float(between.max() / total)


def _panel_positions(gray: Any, axis: int) -> tuple[float, ...]:
    """Normalized positions of straight lines spanning most of the frame (axis 0: vertical lines)."""
    import cv2
    import numpy as np

    if axis == 0:
        gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        profile = (gradient > 4 * EDGE_THRESHOLD).mean(axis=0)
    else:
        gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        profile = (gradient > 4 * EDGE_THRESHOLD).mean(axis=1)
    size = profile.shape[0]
    margin = max(2, int(0.03 * size))
    found = []
    for index in range(margin, size - margin):
        if profile[index] >= PANEL_LINE_SPAN and profile[index] >= profile[index - 1] \
                and profile[index] >= profile[index + 1]:
            found.append(round(index / size, 3))
    return tuple(found)


def _block_stats(values: Any, cols: int, rows: int) -> tuple[Any, Any, Any]:
    """Per cell mean / 10th / 90th percentile of a float image."""
    import cv2
    import numpy as np

    k = 8
    resized = cv2.resize(values, (cols * k, rows * k), interpolation=cv2.INTER_AREA)
    blocks = resized.reshape(rows, k, cols, k).transpose(0, 2, 1, 3).reshape(rows, cols, k * k)
    return blocks.mean(axis=2), np.percentile(blocks, 10, axis=2), np.percentile(blocks, 90, axis=2)


def _cell_mean(values: Any, cols: int, rows: int) -> Any:
    import cv2

    return cv2.resize(values, (cols, rows), interpolation=cv2.INTER_AREA)


def analyze_frames(frames: Iterable[tuple[Any, float]], *, cols: int, rows: int, sample_fps: float,
                   fingerprint: str) -> CaptionBackgroundMap:
    """Background map from streamed (BGR image, t) pairs; keeps only per-cell grids."""
    import cv2
    import numpy as np

    lut = np.array([srgb_to_linear(v) for v in range(256)], dtype=np.float32)

    def u8(values: Any) -> bytes:
        return bytes(np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8).ravel())

    samples: list[BackgroundSample] = []
    previous = None
    segment = 0
    cuts = 0
    for image, t in frames:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (cols * 4, rows * 4), interpolation=cv2.INTER_AREA).astype(np.float32)
        diff_cells = bytes(cols * rows)
        if previous is not None:
            delta = np.abs(small - previous)
            if float(delta.mean()) > CUT_MEAN_DIFF:
                segment += 1
                cuts += 1
            else:
                diff_cells = bytes(np.clip(np.rint(_cell_mean(delta, cols, rows)), 0, 255).astype(np.uint8).ravel())
        previous = small
        b, g, r = (lut[image[:, :, i]] for i in range(3))
        luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)
        mean, p10, p90 = _block_stats(luminance, cols, rows)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = ((np.abs(gx) + np.abs(gy)) > 4 * EDGE_THRESHOLD).astype(np.float32)
        text = text_like_mask(gray).astype(np.float32)
        samples.append(BackgroundSample(
            float(t), u8(mean), u8(p10), u8(p90), u8(_cell_mean(edges, cols, rows)),
            u8(_cell_mean(text, cols, rows)), diff_cells, segment,
            _panel_positions(gray, 0), _panel_positions(gray, 1)))
    return CaptionBackgroundMap(cols, rows, sample_fps, tuple(samples), cuts, fingerprint)


def background_fingerprint(*, source: dict[str, Any], clip_identity: str, width: int, height: int,
                           frame_count: int, fps: str, sample_fps: float, cols: int) -> str:
    payload = {"version": BACKGROUND_VERSION, "source": source, "clip_identity": clip_identity,
               "size": [width, height], "frames": frame_count, "fps": fps, "sample_fps": sample_fps, "cols": cols,
               "max_width": ANALYSIS_MAX_WIDTH, "cut": CUT_MEAN_DIFF, "edge": EDGE_THRESHOLD}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def analyze_clip(path: str | Path, width: int, height: int, *, fingerprint: str,
                 sample_fps: float = DEFAULT_SAMPLE_FPS, cols: int = DEFAULT_COLS) -> CaptionBackgroundMap:
    from ai.editor.pro_edit.vision.frames import iterate_frames

    rows = grid_rows(width, height, cols)
    stream = ((sample.image, sample.t) for sample in
              iterate_frames(path, width, height, sample_fps=sample_fps, max_width=ANALYSIS_MAX_WIDTH))
    return analyze_frames(stream, cols=cols, rows=rows, sample_fps=sample_fps, fingerprint=fingerprint)


def load_or_analyze(cache_path: Path, media: Any, clip_identity: str, *, force: bool = False,
                    sample_fps: float = DEFAULT_SAMPLE_FPS, cols: int = DEFAULT_COLS
                    ) -> tuple[CaptionBackgroundMap, str]:
    """(map, "cached" | "analyzed"). Raises on analysis failure (caller falls back)."""
    fingerprint = background_fingerprint(
        source=source_identity(media.path), clip_identity=clip_identity, width=media.width, height=media.height,
        frame_count=media.frame_count, fps=str(media.fps), sample_fps=sample_fps, cols=cols)
    if not force and cache_path.is_file():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if data.get("fingerprint") == fingerprint and int(data.get("version", -1)) == BACKGROUND_VERSION:
                return CaptionBackgroundMap.from_dict(data), "cached"
        except (OSError, ValueError, KeyError, TypeError, zlib.error):
            pass
    found = analyze_clip(media.path, media.width, media.height, fingerprint=fingerprint, sample_fps=sample_fps,
                         cols=cols)
    if not found.samples:
        raise ValueError("no usable background samples")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp = cache_path.with_name(cache_path.name + ".tmp")
    temp.write_text(json.dumps(found.to_dict(), separators=(",", ":")), encoding="utf-8")
    os.replace(temp, cache_path)
    return found, "analyzed"


@dataclass(frozen=True)
class OutputMappedBackground:
    """Background map (source frame) queried with OUTPUT-frame boxes (base-window crop)."""

    inner: CaptionBackgroundMap
    base: tuple[int, int, int, int]
    source_size: tuple[int, int]

    def _map(self, box: Sequence[float]) -> tuple[float, float, float, float]:
        bx, by, bw, bh = self.base
        width, height = self.source_size
        x0, y0, x1, y1 = box
        return ((bx + x0 * bw) / width, (by + y0 * bh) / height, (bx + x1 * bw) / width, (by + y1 * bh) / height)

    def region_stats(self, start: float, end: float, box: Sequence[float]) -> RegionStats | None:
        return self.inner.region_stats(start, end, self._map(box))

    def window(self, start: float, end: float) -> list[float]:
        return self.inner.window(start, end)

    def region_score(self, box: Sequence[float], values: Sequence[float]) -> float:
        return self.inner.region_score(self._map(box), values)

    @property
    def cols(self) -> int:
        return self.inner.cols

    @property
    def rows(self) -> int:
        return self.inner.rows


def ui_boxes(background: CaptionBackgroundMap, start: float, end: float, *, min_cells: int = 2
             ) -> list[tuple[tuple[float, float, float, float], float]]:
    """Connected groups of persistent text-like cells -> (normalized box, mean persistence)."""
    values = background.text_occupancy(start, end)
    cols, rows = background.cols, background.rows
    seen: set[int] = set()
    boxes = []
    for start_index, value in enumerate(values):
        if value <= 0 or start_index in seen:
            continue
        stack, members = [start_index], []
        seen.add(start_index)
        while stack:
            index = stack.pop()
            members.append(index)
            row, col = divmod(index, cols)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r, c = row + dr, col + dc
                other = r * cols + c
                if 0 <= r < rows and 0 <= c < cols and other not in seen and values[other] > 0:
                    seen.add(other)
                    stack.append(other)
        if len(members) < min_cells:
            continue
        rs = [i // cols for i in members]
        cs = [i % cols for i in members]
        boxes.append(((min(cs) / cols, min(rs) / rows, (max(cs) + 1) / cols, (max(rs) + 1) / rows),
                      sum(values[i] for i in members) / len(members)))
    return boxes


def finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(v) for v in values)
