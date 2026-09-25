"""Camera mathematics: easing, crop windows, frame-domain camera paths.

Pure, deterministic, dependency-free. The FFmpeg expression builder encodes
exactly these formulas, and tests compare both.

Camera state per frame: zoom z >= 1 and a normalized desired crop center
(cx, cy). For a (base) frame W x H:

    crop_w = W / z,  crop_h = H / z
    x = clamp(cx * W - crop_w / 2, 0, W - crop_w)
    y = clamp(cy * H - crop_h / 2, 0, H - crop_h)

The crop is scaled back to the full frame in one resampling step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

from ai.editor.pro_edit.errors import PresetResolutionError

HARD_MIN_ZOOM = 1.0
HARD_MAX_ZOOM = 1.35
_EPS = 1e-9


def _smoothstep(u: float) -> float:
    return u * u * (3.0 - 2.0 * u)


def _ease_out_cubic(u: float) -> float:
    return 1.0 - (1.0 - u) ** 3


EASINGS: dict[str, Callable[[float], float]] = {
    "linear": lambda u: u,
    "smoothstep": _smoothstep,
    "ease_out_cubic": _ease_out_cubic,
}


def ease(name: str, u: float) -> float:
    try:
        function = EASINGS[name]
    except KeyError as error:
        raise PresetResolutionError(f"unknown easing {name!r}") from error
    return function(max(0.0, min(1.0, float(u))))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


@dataclass(frozen=True)
class CameraState:
    zoom: float
    cx: float
    cy: float

    def close_to(self, other: "CameraState", tol: float = 1e-6) -> bool:
        return (abs(self.zoom - other.zoom) <= tol and abs(self.cx - other.cx) <= tol
                and abs(self.cy - other.cy) <= tol)


IDENTITY = CameraState(1.0, 0.5, 0.5)


def crop_window(state: CameraState, width: float, height: float) -> tuple[float, float, float, float]:
    """(x, y, crop_w, crop_h) in pixels; always inside the frame."""
    z = clamp(float(state.zoom), HARD_MIN_ZOOM, HARD_MAX_ZOOM)
    crop_w = width / z
    crop_h = height / z
    x = clamp(state.cx * width - crop_w / 2.0, 0.0, width - crop_w)
    y = clamp(state.cy * height - crop_h / 2.0, 0.0, height - crop_h)
    return x, y, crop_w, crop_h


def effective_center(state: CameraState) -> tuple[float, float]:
    """Normalized center actually shown after clamping (resolution-free)."""
    x, y, w, h = crop_window(state, 1.0, 1.0)
    return x + w / 2.0, y + h / 2.0


class OutputProfile(str, Enum):
    PRESERVE = "preserve"
    LANDSCAPE_16_9 = "16:9"
    PORTRAIT_9_16 = "9:16"
    SQUARE_1_1 = "1:1"


_ASPECTS = {
    OutputProfile.LANDSCAPE_16_9: 16 / 9,
    OutputProfile.PORTRAIT_9_16: 9 / 16,
    OutputProfile.SQUARE_1_1: 1.0,
}


def base_window(width: int, height: int, profile: OutputProfile,
                anchor: tuple[float, float] = (0.5, 0.5)) -> tuple[int, int, int, int]:
    """Largest even-sized window of the profile aspect inside the source.

    PRESERVE returns the full frame (no geometry change). Other profiles are
    explicit configuration only; the planner can never select them.
    """
    if profile is OutputProfile.PRESERVE:
        return 0, 0, int(width), int(height)
    aspect = _ASPECTS[profile]
    if width / height > aspect:
        bh = int(height) - int(height) % 2
        bw = int(bh * aspect)
    else:
        bw = int(width) - int(width) % 2
        bh = int(bw / aspect)
    bw -= bw % 2
    bh -= bh % 2
    bx = int(round(clamp(anchor[0] * width - bw / 2.0, 0, width - bw)))
    by = int(round(clamp(anchor[1] * height - bh / 2.0, 0, height - bh)))
    bx -= bx % 2
    by -= by % 2
    return bx, by, bw, bh


# ============================================================
# FRAME-DOMAIN PATH
# ============================================================

@dataclass(frozen=True)
class CameraSegment:
    """Interpolates start->end over frames [start_frame, end_frame).

    ``u = (f - start_frame) / (end_frame - 1 - start_frame)`` so the LAST frame
    of the segment shows exactly ``end``; the next segment starts from that
    same state -> no discontinuity at any segment boundary.
    """

    start_frame: int
    end_frame: int
    start: CameraState
    end: CameraState
    easing: str = "linear"
    source_event_id: str = ""
    phase: str = ""

    def __post_init__(self) -> None:
        if self.end_frame <= self.start_frame:
            raise PresetResolutionError(f"empty camera segment {self.start_frame}-{self.end_frame}")
        if self.easing not in EASINGS:
            raise PresetResolutionError(f"unknown easing {self.easing!r}")

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def is_constant(self) -> bool:
        return self.start.close_to(self.end, 1e-12)

    def progress(self, frame: int) -> float:
        span = max(1, self.end_frame - 1 - self.start_frame)
        return ease(self.easing, (frame - self.start_frame) / span)

    def state_at(self, frame: int) -> CameraState:
        s = self.progress(frame)
        return CameraState(
            lerp(self.start.zoom, self.end.zoom, s),
            lerp(self.start.cx, self.end.cx, s),
            lerp(self.start.cy, self.end.cy, s),
        )


@dataclass(frozen=True)
class CameraPath:
    frame_count: int
    segments: tuple[CameraSegment, ...]

    def __post_init__(self) -> None:
        previous_end = 0
        for segment in self.segments:
            if segment.start_frame < previous_end:
                raise PresetResolutionError("camera segments overlap")
            if segment.end_frame > self.frame_count:
                raise PresetResolutionError("camera segment beyond clip frame count")
            previous_end = segment.end_frame

    def state_at(self, frame: int) -> CameraState:
        lo, hi = 0, len(self.segments) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            segment = self.segments[mid]
            if frame < segment.start_frame:
                hi = mid - 1
            elif frame >= segment.end_frame:
                lo = mid + 1
            else:
                return segment.state_at(frame)
        return IDENTITY

    def active_ranges(self) -> list[tuple[int, int]]:
        """Merged [start, end) frame ranges where the camera is not identity."""
        ranges: list[tuple[int, int]] = []
        for segment in self.segments:
            if segment.is_constant and segment.start.close_to(IDENTITY, 1e-12):
                continue
            if ranges and ranges[-1][1] == segment.start_frame:
                ranges[-1] = (ranges[-1][0], segment.end_frame)
            else:
                ranges.append((segment.start_frame, segment.end_frame))
        return ranges

    @property
    def is_identity(self) -> bool:
        return not self.active_ranges()

    def sample(self, frames: Sequence[int] | None = None) -> list[CameraState]:
        indices = range(self.frame_count) if frames is None else frames
        return [self.state_at(f) for f in indices]


def verify_path(
    path: CameraPath,
    *,
    max_zoom: float,
    fps: float,
    max_follow_speed: float | None = None,
    follow_phases: frozenset[str] = frozenset({"follow"}),
) -> dict[str, float]:
    """Hard invariants checked on EVERY frame; raises on any violation."""
    if max_zoom > HARD_MAX_ZOOM + _EPS:
        raise PresetResolutionError(f"style max zoom {max_zoom} exceeds hard limit {HARD_MAX_ZOOM}")
    peak_zoom = 1.0
    max_step_zoom = 0.0
    previous: CameraState | None = None
    for frame in range(path.frame_count):
        state = path.state_at(frame)
        if not (HARD_MIN_ZOOM - _EPS <= state.zoom <= max_zoom + _EPS):
            raise PresetResolutionError(f"frame {frame}: zoom {state.zoom:.5f} outside [1, {max_zoom}]")
        if not (0.0 <= state.cx <= 1.0 and 0.0 <= state.cy <= 1.0):
            raise PresetResolutionError(f"frame {frame}: anchor outside normalized frame")
        x, y, w, h = crop_window(state, 1.0, 1.0)
        if x < -_EPS or y < -_EPS or x + w > 1.0 + _EPS or y + h > 1.0 + _EPS:
            raise PresetResolutionError(f"frame {frame}: crop window leaves the source")
        if previous is not None:
            max_step_zoom = max(max_step_zoom, abs(state.zoom - previous.zoom))
        previous = state
        peak_zoom = max(peak_zoom, state.zoom)
    # Boundary continuity: the step INTO a segment may not exceed the largest
    # per-frame step inside the two segments it joins (no hidden jump cuts).
    def step(a: CameraState, b: CameraState) -> float:
        return max(abs(b.zoom - a.zoom), abs(b.cx - a.cx), abs(b.cy - a.cy))

    def inner_step(segment: CameraSegment | None) -> float:
        if segment is None or segment.length < 2:
            return 0.0
        states = [segment.state_at(f) for f in range(segment.start_frame, segment.end_frame)]
        return max(step(x, y) for x, y in zip(states, states[1:]))

    by_end = {segment.end_frame: segment for segment in path.segments}
    for segment in path.segments:
        if segment.start_frame == 0:
            continue
        jump = step(path.state_at(segment.start_frame - 1), path.state_at(segment.start_frame))
        allowed = max(inner_step(segment), inner_step(by_end.get(segment.start_frame))) + 1e-6
        if jump > allowed:
            raise PresetResolutionError(
                f"discontinuity at frame {segment.start_frame} ({segment.source_event_id}/{segment.phase}): "
                f"step {jump:.5f} > {allowed:.5f}"
            )
    max_speed = 0.0
    if max_follow_speed is not None:
        for segment in path.segments:
            if segment.phase not in follow_phases or segment.length < 2:
                continue
            for frame in range(segment.start_frame + 1, segment.end_frame):
                a = effective_center(path.state_at(frame - 1))
                b = effective_center(path.state_at(frame))
                speed = math.hypot(b[0] - a[0], b[1] - a[1]) * fps
                max_speed = max(max_speed, speed)
        if max_speed > max_follow_speed * 1.05 + 1e-6:
            raise PresetResolutionError(f"follow speed {max_speed:.4f} > limit {max_follow_speed}")
    return {"peak_zoom": round(peak_zoom, 6), "max_zoom_step": round(max_step_zoom, 6),
            "max_follow_speed": round(max_speed, 6)}
