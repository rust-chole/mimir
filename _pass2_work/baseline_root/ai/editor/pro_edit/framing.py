"""Framing solver: semantic camera intent + evidence -> one concrete crop.

The planner chooses intent (camera mode, motion, intensity). This solver
chooses the actual crop from subject boxes, story geometry and the caption
safe region. All coordinates are normalized to the base window (the frame
the camera crops from, which is the output frame after scaling).

Constraint priority (highest first):
1. frame bounds (crop always inside the source)
2. story geometry: required regions / required subjects stay inside the
   crop, and ``min_visible_fraction`` of each dimension stays on screen
3. subject safety box: forehead/chin/hands margins for the camera mode
4. caption avoidance: the primary face stays above the burned caption band
   (soft: dropped, with a note, only when it cannot coexist with 1-3);
   burned text bands (intro hook) : the primary face sits fully above or
   fully below each band (soft, same rule)
5. composition preference: face slightly above center (headroom), subject
   centered horizontally (dual subject: pair centered)
Zoom is reduced step by step until 1-3 are satisfiable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ai.editor.pro_edit.camera import HARD_MAX_ZOOM, clamp
from ai.editor.pro_edit.schema import CameraMode

ZOOM_STEP = 0.005

# Margins around a FACE box per camera mode, as multiples of the face box size:
# (each side, above, below). Medium keeps shoulders and gesturing hands.
FACE_MARGINS: dict[CameraMode, tuple[float, float, float]] = {
    CameraMode.PRESERVE: (0.60, 0.60, 1.00),
    CameraMode.SPEAKER_MEDIUM: (1.00, 0.70, 1.80),
    CameraMode.SPEAKER_CLOSE: (0.45, 0.50, 0.60),
    CameraMode.REACTION_CLOSE: (0.35, 0.45, 0.45),
    CameraMode.DUAL_SUBJECT: (0.40, 0.50, 0.70),
    CameraMode.OBJECT_FOCUS: (0.60, 0.60, 1.00),
}
KIND_MARGINS: dict[str, tuple[float, float, float]] = {
    "person": (0.08, 0.06, 0.04),
    "object": (0.20, 0.20, 0.20),
    "unknown": (0.15, 0.15, 0.15),
}
# Where the face center should sit inside the crop (fraction of crop height).
HEADROOM_TARGET: dict[CameraMode, float] = {
    CameraMode.SPEAKER_MEDIUM: 0.38,
    CameraMode.SPEAKER_CLOSE: 0.42,
    CameraMode.REACTION_CLOSE: 0.44,
}
DEFAULT_HEADROOM_TARGET = 0.42
CAPTION_CLEARANCE = 0.01


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_center(cls, cx: float, cy: float, w: float, h: float) -> "Box":
        return cls(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def clipped(self) -> "Box":
        return Box(clamp(self.x0, 0.0, 1.0), clamp(self.y0, 0.0, 1.0), clamp(self.x1, 0.0, 1.0),
                   clamp(self.y1, 0.0, 1.0))

    def union(self, other: "Box") -> "Box":
        return Box(min(self.x0, other.x0), min(self.y0, other.y0), max(self.x1, other.x1), max(self.y1, other.y1))

    def expanded(self, side: float, top: float, bottom: float) -> "Box":
        return Box(self.x0 - side * self.w, self.y0 - top * self.h, self.x1 + side * self.w,
                   self.y1 + bottom * self.h).clipped()

    def to_list(self) -> list[float]:
        return [round(self.x0, 4), round(self.y0, 4), round(self.x1, 4), round(self.y1, 4)]


@dataclass(frozen=True)
class SubjectBox:
    box: Box
    kind: str
    subject_id: str = ""


@dataclass(frozen=True)
class FramingRequest:
    desired_zoom: float
    camera: CameraMode
    max_zoom: float
    subjects: tuple[SubjectBox, ...] = ()
    required: tuple[Box, ...] = ()
    min_visible_fraction: float = 0.0
    caption_top: float | None = None
    avoid_bands: tuple[tuple[float, float], ...] = ()   # output-normalized (y0, y1) text bands


@dataclass(frozen=True)
class Framing:
    zoom: float
    anchor: tuple[float, float]
    origin: tuple[float, float]
    notes: tuple[str, ...] = field(default_factory=tuple)
    caption_clear: bool | None = None
    text_clear: bool | None = None

    def to_dict(self) -> dict[str, object]:
        return {"zoom": round(self.zoom, 4), "anchor": [round(self.anchor[0], 4), round(self.anchor[1], 4)],
                "origin": [round(self.origin[0], 4), round(self.origin[1], 4)], "notes": list(self.notes),
                "caption_clear": self.caption_clear, "text_clear": self.text_clear}


def safety_box(subject: SubjectBox, camera: CameraMode) -> Box:
    if subject.kind == "face":
        side, top, bottom = FACE_MARGINS.get(camera, FACE_MARGINS[CameraMode.PRESERVE])
    else:
        side, top, bottom = KIND_MARGINS.get(subject.kind, KIND_MARGINS["unknown"])
    return subject.box.expanded(side, top, bottom)


def _interval(lo: float, hi: float, constraints: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    for a, b in constraints:
        lo, hi = max(lo, a), min(hi, b)
    return (lo, hi) if lo <= hi + 1e-9 else None


def _contain(box: Box, size: float, axis: str) -> tuple[float, float]:
    """Origin range so that [origin, origin+size] contains the box on ``axis``."""
    if axis == "x":
        return box.x1 - size, box.x0
    return box.y1 - size, box.y0


def solve_framing(request: FramingRequest) -> Framing:
    notes: list[str] = []
    safety = [safety_box(s, request.camera) for s in request.subjects]
    subject_union = None
    for box in safety:
        subject_union = box if subject_union is None else subject_union.union(box)
    required_union = None
    for box in request.required:
        clipped = box.clipped()
        required_union = clipped if required_union is None else required_union.union(clipped)

    cap = min(request.max_zoom, HARD_MAX_ZOOM)
    if request.min_visible_fraction > 0:
        cap = min(cap, 1.0 / request.min_visible_fraction)
    for label, union in (("subject_safety_box", subject_union), ("story_required_region", required_union)):
        if union is not None and union.w > 0 and union.h > 0:
            fit = max(1.0, min(1.0 / union.w, 1.0 / union.h))
            if fit < cap:
                cap = fit
                notes.append(f"zoom_limited_by_{label}")
    zoom = clamp(min(request.desired_zoom, cap), 1.0, HARD_MAX_ZOOM)
    if zoom < request.desired_zoom - 1e-9:
        notes.append(f"zoom_capped:{request.desired_zoom:.3f}->{zoom:.3f}")

    primary = request.subjects[0] if request.subjects else None
    while True:
        size = 1.0 / zoom
        hard_x = [(0.0, 1.0 - size)]
        hard_y = [(0.0, 1.0 - size)]
        for union in (required_union, subject_union):
            if union is not None:
                hard_x.append(_contain(union, size, "x"))
                hard_y.append(_contain(union, size, "y"))
        range_x = _interval(-1e9, 1e9, hard_x)
        range_y = _interval(-1e9, 1e9, hard_y)
        if range_x is not None and range_y is not None:
            break
        if zoom <= 1.0 + 1e-9:
            # Identity always satisfies containment of in-frame boxes.
            range_x, range_y = (0.0, 0.0), (0.0, 0.0)
            break
        zoom = max(1.0, zoom - ZOOM_STEP)
        if "zoom_reduced_for_containment" not in notes:
            notes.append("zoom_reduced_for_containment")

    caption_clear: bool | None = None
    if request.caption_top is not None and primary is not None and primary.kind == "face" and zoom > 1.0 + 1e-9:
        # Face bottom must map above the caption band: (fb - y0) / size <= top.
        needed = primary.box.y1 - (request.caption_top - CAPTION_CLEARANCE) * size
        with_caption = _interval(range_y[0], range_y[1], [(needed, 1e9)])
        if with_caption is not None:
            range_y = with_caption
            caption_clear = True
        else:
            caption_clear = False
            notes.append("caption_overlap_unavoidable")

    size = 1.0 / zoom
    pref_x, pref_y = _preferred_origin(request, primary, subject_union, required_union, size)

    text_clear: bool | None = None
    if request.avoid_bands and primary is not None and primary.kind == "face" and zoom > 1.0 + 1e-9:
        text_clear = True
        for band_y0, band_y1 in request.avoid_bands:
            # Face fully below the band, or fully above it (whichever keeps the
            # composition closest to the preference).
            below = _interval(range_y[0], range_y[1], [(-1e9, primary.box.y0 - (band_y1 + CAPTION_CLEARANCE) * size)])
            above = _interval(range_y[0], range_y[1], [(primary.box.y1 - (band_y0 - CAPTION_CLEARANCE) * size, 1e9)])
            options = [o for o in (below, above) if o is not None]
            if not options:
                text_clear = False
                if "text_overlap_unavoidable" not in notes:
                    notes.append("text_overlap_unavoidable")
                continue
            range_y = min(options, key=lambda o: abs(clamp(pref_y, o[0], o[1]) - pref_y))
    x0 = clamp(pref_x, range_x[0], range_x[1])
    y0 = clamp(pref_y, range_y[0], range_y[1])
    anchor = (clamp(x0 + size / 2.0, 0.0, 1.0), clamp(y0 + size / 2.0, 0.0, 1.0))
    return Framing(zoom, anchor, (x0, y0), tuple(notes), caption_clear, text_clear)


def _preferred_origin(request: FramingRequest, primary: SubjectBox | None, subject_union: Box | None,
                      required_union: Box | None, size: float) -> tuple[float, float]:
    """Composition preference (lowest priority): headroom / centering."""
    if primary is not None:
        if request.camera is CameraMode.DUAL_SUBJECT and subject_union is not None:
            return subject_union.cx - size / 2.0, subject_union.cy - size / 2.0
        pref_x = primary.box.cx - size / 2.0
        if primary.kind == "face":
            return pref_x, primary.box.cy - HEADROOM_TARGET.get(request.camera, DEFAULT_HEADROOM_TARGET) * size
        return pref_x, primary.box.cy - size / 2.0
    if required_union is not None:
        return required_union.cx - size / 2.0, required_union.cy - size / 2.0
    return (1.0 - size) / 2.0, (1.0 - size) / 2.0


def crop_contains(zoom: float, anchor: tuple[float, float], box: Box, tol: float = 1e-6) -> bool:
    size = 1.0 / max(1.0, zoom)
    x0 = clamp(anchor[0] - size / 2.0, 0.0, 1.0 - size)
    y0 = clamp(anchor[1] - size / 2.0, 0.0, 1.0 - size)
    b = box.clipped()
    return (x0 <= b.x0 + tol and y0 <= b.y0 + tol and x0 + size >= b.x1 - tol and y0 + size >= b.y1 - tol)
