"""Cheap, deterministic CONTENT LAYOUT classification for caption placement.

Classes: TALKING_HEAD, DUAL_TALKING_HEAD, GAMEPLAY_FACE_CAM, FULLSCREEN_GAMEPLAY,
SCREEN_SHARE, MULTI_PANEL, UNKNOWN.

Evidence (all already computed locally; no LLM, no new model):
    face tracks (count, persistence, size, position), motion activity,
    TEXT_LIKE persistence (caption_background), persistent full-span lines
    (panel boundaries).

Conservative by design: every rule needs its evidence to clear explicit
thresholds and the winning rule's confidence must reach MIN_CONFIDENCE;
otherwise the result is UNKNOWN and placement behaves exactly as without a
layout. The layout only re-weights existing placement evidence (and adds the
face-cam box as an avoid region); it never adds zones or moves captions by
itself. The weights are calibrations (MANUAL_VISUAL_CALIBRATION).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

LAYOUT_VERSION = 1
MIN_CONFIDENCE = 0.6
FACE_PERSISTENCE = 0.5          # fraction of 0.5 s bins with a reliable face sample
TALKING_FACE_HEIGHT = 0.15      # median face height / frame height
DUAL_FACE_HEIGHT = 0.10
CAM_FACE_HEIGHT = 0.12          # a face-cam face is small ...
CAM_CORNER = 0.32               # ... and inside a corner band
ACTIVE_SCENE = 0.10             # mean activity (0..1) of a gameplay-like scene
CALM_SCENE = 0.06
UI_AREA_GAMEPLAY = 0.015        # persistent text-like area fraction (HUD present)
UI_AREA_SCREEN = 0.10           # ... dense text (screen share)
FACECAM_EXPAND = 2.2
BIN_S = 0.5


class LayoutClass(str, Enum):
    TALKING_HEAD = "TALKING_HEAD"
    DUAL_TALKING_HEAD = "DUAL_TALKING_HEAD"
    GAMEPLAY_FACE_CAM = "GAMEPLAY_FACE_CAM"
    FULLSCREEN_GAMEPLAY = "FULLSCREEN_GAMEPLAY"
    SCREEN_SHARE = "SCREEN_SHARE"
    MULTI_PANEL = "MULTI_PANEL"
    UNKNOWN = "UNKNOWN"


# Placement weight multipliers per class (1.0 = V4 weights).
LAYOUT_WEIGHTS: Mapping[LayoutClass, Mapping[str, float]] = {
    LayoutClass.GAMEPLAY_FACE_CAM: {"ui": 1.25, "activity": 1.0},
    LayoutClass.FULLSCREEN_GAMEPLAY: {"ui": 1.25, "activity": 1.5},
    LayoutClass.SCREEN_SHARE: {"ui": 1.5, "activity": 1.0},
}


@dataclass(frozen=True)
class FaceSummary:
    subject_id: str
    persistence: float
    height: float
    cx: float
    cy: float
    box: tuple[float, float, float, float]     # union-ish (median-centred) source-normalized box


@dataclass(frozen=True)
class LayoutResult:
    layout: LayoutClass
    confidence: float
    evidence: Mapping[str, Any] = field(default_factory=dict)
    facecam_box: tuple[float, float, float, float] | None = None    # source-normalized
    weights: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"version": LAYOUT_VERSION, "layout": self.layout.value, "confidence": round(self.confidence, 3),
                "evidence": dict(self.evidence),
                "facecam_box": [round(v, 4) for v in self.facecam_box] if self.facecam_box else None,
                "weights": dict(self.weights)}


UNKNOWN_LAYOUT = LayoutResult(LayoutClass.UNKNOWN, 0.0, {"reason": "no evidence"})


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def summarize_faces(tracks: Sequence[Any], duration: float) -> list[FaceSummary]:
    bins_total = max(1, int(duration / BIN_S + 0.999))
    found = []
    for track in tracks:
        if getattr(getattr(track, "kind", None), "value", getattr(track, "kind", "")) != "face":
            continue
        reliable = [s for s in track.samples if s.confidence >= 0.5]
        if not reliable:
            continue
        bins = {int(s.t // BIN_S) for s in reliable}
        cx, cy = _median([s.cx for s in reliable]), _median([s.cy for s in reliable])
        w, h = _median([s.width for s in reliable]), _median([s.height for s in reliable])
        found.append(FaceSummary(str(track.subject_id), min(1.0, len(bins) / bins_total), h, cx, cy,
                                 (max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2),
                                  min(1.0, cy + h / 2))))
    return sorted(found, key=lambda f: (-f.persistence, f.subject_id))


def _mean_activity(occupancy: Any, duration: float, exclude: tuple[float, float, float, float] | None) -> float:
    if occupancy is None or not getattr(occupancy, "times", ()):
        return 0.0
    values = occupancy.window(0.0, duration)
    cols, rows = occupancy.cols, occupancy.rows
    kept = []
    for index, value in enumerate(values):
        row, col = divmod(index, cols)
        cx, cy = (col + 0.5) / cols, (row + 0.5) / rows
        if exclude and exclude[0] <= cx <= exclude[2] and exclude[1] <= cy <= exclude[3]:
            continue
        kept.append(value)
    return sum(kept) / len(kept) if kept else 0.0


def _ui_area(background: Any, duration: float) -> float:
    if background is None or not getattr(background, "samples", ()):
        return 0.0
    values = background.text_occupancy(0.0, duration)
    return sum(1 for v in values if v > 0) / max(1, len(values))


def _expand(box: tuple[float, float, float, float], factor: float) -> tuple[float, float, float, float]:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = (box[2] - box[0]) * factor, (box[3] - box[1]) * factor
    return (max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2))


def classify_layout(tracks: Sequence[Any], duration: float, *, occupancy: Any = None,
                    background: Any = None) -> LayoutResult:
    """Deterministic rules; UNKNOWN unless one class is clearly supported."""
    if duration <= 0:
        return UNKNOWN_LAYOUT
    faces = [f for f in summarize_faces(tracks, duration) if f.persistence >= FACE_PERSISTENCE]
    ui_area = _ui_area(background, duration)
    panels_x, panels_y = background.panel_lines(0.0, duration) if background is not None else ((), ())
    dividers = [p for p in (*panels_x, *panels_y) if 0.2 <= p <= 0.8]
    cam = next((f for f in faces if f.height < CAM_FACE_HEIGHT and (f.cx < CAM_CORNER or f.cx > 1 - CAM_CORNER)
                and (f.cy < CAM_CORNER or f.cy > 1 - CAM_CORNER)), None)
    cam_box = _expand(cam.box, FACECAM_EXPAND) if cam else None
    activity = _mean_activity(occupancy, duration, cam_box)
    evidence = {"faces": [{"id": f.subject_id, "persistence": round(f.persistence, 3), "height": round(f.height, 3),
                           "center": [round(f.cx, 3), round(f.cy, 3)]} for f in faces],
                "activity_mean": round(activity, 4), "ui_area": round(ui_area, 4),
                "panel_lines": {"x": list(panels_x), "y": list(panels_y)},
                "activity_available": occupancy is not None, "background_available": background is not None}
    candidates: list[tuple[float, LayoutClass, tuple[float, float, float, float] | None]] = []
    large = [f for f in faces if f.height >= TALKING_FACE_HEIGHT]
    medium = [f for f in faces if f.height >= DUAL_FACE_HEIGHT]
    if len(medium) >= 2 and len(faces) == 2:
        candidates.append((min(f.persistence for f in medium[:2]), LayoutClass.DUAL_TALKING_HEAD, None))
    elif len(large) == 1 and len(faces) == 1:
        candidates.append((large[0].persistence, LayoutClass.TALKING_HEAD, None))
    if cam is not None and occupancy is not None and (activity >= ACTIVE_SCENE or ui_area >= UI_AREA_GAMEPLAY):
        candidates.append((cam.persistence, LayoutClass.GAMEPLAY_FACE_CAM, cam_box))
    if not faces and occupancy is not None and background is not None:
        if activity >= ACTIVE_SCENE and ui_area >= UI_AREA_GAMEPLAY:
            candidates.append((min(1.0, 0.5 + activity + ui_area * 5), LayoutClass.FULLSCREEN_GAMEPLAY, None))
        elif activity < CALM_SCENE and ui_area >= UI_AREA_SCREEN:
            candidates.append((min(1.0, 0.4 + ui_area * 3), LayoutClass.SCREEN_SHARE, None))
    if dividers and not large and not candidates:
        candidates.append((0.65 if len(dividers) >= 2 else 0.6, LayoutClass.MULTI_PANEL, None))
    if not candidates:
        return LayoutResult(LayoutClass.UNKNOWN, 0.0, {**evidence, "reason": "no rule matched"})
    candidates.sort(key=lambda c: (-c[0], c[1].value))
    confidence, layout, box = candidates[0]
    if confidence < MIN_CONFIDENCE:
        return LayoutResult(LayoutClass.UNKNOWN, confidence,
                            {**evidence, "reason": f"{layout.value} below confidence {MIN_CONFIDENCE}"})
    if len(candidates) > 1 and candidates[1][0] >= confidence - 0.05 and candidates[1][1] is not layout:
        return LayoutResult(LayoutClass.UNKNOWN, confidence,
                            {**evidence, "reason": f"ambiguous: {layout.value} vs {candidates[1][1].value}"})
    return LayoutResult(layout, confidence, evidence, box, dict(LAYOUT_WEIGHTS.get(layout, {})))
