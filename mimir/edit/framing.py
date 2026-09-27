"""Framing geometry: one virtual camera window per frame (source-normalized coordinates).

A window is (cx, cy, h): its center in source-normalized coordinates and its
height in source-height units; its width follows from the output aspect:

    w = h * a / A        (a = output aspect W/H, A = source aspect)

``h <= h_inside`` keeps the window inside the source (a classic crop);
larger windows show more than the source in one dimension and the renderer
fills the rest with a blurred copy of the frame (the "fit" look). Crop and
fit are therefore one continuum, so moving between them is a smooth zoom.

Constraint priority (highest first): frame validity -> REQUIRED story content
inside the window -> subject safety margins -> caption clearance -> headroom
and centering. When requirements conflict the window only ever gets WIDER.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

CONTENT_Y = 0.42          # where fitted content is centered in the output (captions live below)
HEADROOM = 0.40           # face center at this fraction of the window height
CAPTION_TOP = 0.60        # output-y above which a framed face should end (caption band below)
REQUIRED_MARGIN = 0.03

# face-box multiples: (each side, above, below)
MARGINS = {  # sized for a 9:16 window (~3 face widths wide at full crop height)
    "medium": (0.55, 0.60, 1.60),
    "punch": (0.45, 0.50, 0.60),
    "reaction": (0.35, 0.40, 0.45),
    "pair": (0.35, 0.45, 0.60),
    "context": (0.50, 0.50, 0.90),
}


@dataclass(frozen=True)
class Geometry:
    src_w: int
    src_h: int
    out_w: int
    out_h: int
    max_zoom: float
    max_upscale: float

    @property
    def src_aspect(self) -> float:
        return self.src_w / self.src_h

    @property
    def out_aspect(self) -> float:
        return self.out_w / self.out_h

    @property
    def h_inside(self) -> float:
        """Tallest window that stays inside the source."""
        return min(1.0, self.src_aspect / self.out_aspect)

    @property
    def h_full(self) -> float:
        """Smallest window that shows the entire source."""
        return max(1.0, self.src_aspect / self.out_aspect)

    @property
    def h_min(self) -> float:
        """Tightest allowed window (zoom and upscale limits)."""
        zoom_cap = max(1.0, min(self.max_zoom, self.max_upscale * self.h_inside * self.src_h / self.out_h))
        return self.h_inside / zoom_cap

    def width(self, h: float) -> float:
        return h * self.out_aspect / self.src_aspect

    def zoom(self, h: float) -> float:
        return self.h_inside / h


@dataclass(frozen=True)
class Window:
    cx: float
    cy: float
    h: float

    def box(self, geo: Geometry) -> tuple[float, float, float, float]:
        w = geo.width(self.h)
        return self.cx - w / 2, self.cy - self.h / 2, self.cx + w / 2, self.cy + self.h / 2

    def contains(self, geo: Geometry, box: Sequence[float], tol: float = 1e-4) -> bool:
        x0, y0, x1, y1 = self.box(geo)
        bx0, by0, bx1, by1 = clip_box(box)
        return x0 <= bx0 + tol and y0 <= by0 + tol and x1 >= bx1 - tol and y1 >= by1 - tol

    def to_output(self, geo: Geometry, box: Sequence[float]) -> tuple[float, float, float, float]:
        """Map a source-normalized box into output-normalized coordinates."""
        x0, y0, _, _ = self.box(geo)
        w = geo.width(self.h)
        return ((box[0] - x0) / w, (box[1] - y0) / self.h, (box[2] - x0) / w, (box[3] - y0) / self.h)

    def inside(self, geo: Geometry, tol: float = 1e-6) -> bool:
        x0, y0, x1, y1 = self.box(geo)
        return x0 >= -tol and y0 >= -tol and x1 <= 1 + tol and y1 <= 1 + tol


def clip_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    return (max(0.0, min(1.0, box[0])), max(0.0, min(1.0, box[1])), max(0.0, min(1.0, box[2])),
            max(0.0, min(1.0, box[3])))


def expand(box: Sequence[float], margins: tuple[float, float, float]) -> tuple[float, float, float, float]:
    side, top, bottom = margins
    w, h = box[2] - box[0], box[3] - box[1]
    return clip_box((box[0] - side * w, box[1] - top * h, box[2] + side * w, box[3] + bottom * h))


def union(boxes: Sequence[Sequence[float]]) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def pad(box: Sequence[float], amount: float) -> tuple[float, float, float, float]:
    return clip_box((box[0] - amount, box[1] - amount, box[2] + amount, box[3] + amount))


def _axis_range(center_lo: float, center_hi: float, size: float, lo: float | None, hi: float | None
                ) -> tuple[float, float]:
    """Center range keeping [c - size/2, c + size/2] inside the source (if it fits) and around [lo, hi]."""
    if size <= 1.0:
        a, b = size / 2, 1 - size / 2
    else:
        a, b = 1 - size / 2, size / 2          # shows the whole source on this axis
    if lo is not None and hi is not None:
        a, b = max(a, hi - size / 2), min(b, lo + size / 2)
    return (a, b) if a <= b + 1e-9 else ((a + b) / 2, (a + b) / 2)


AVOID_MARGIN = 0.012


def _clean_edges(cx: float, w: float, avoid: Sequence[Sequence[float]]) -> bool:
    left, right = cx - w / 2, cx + w / 2
    for box in avoid:
        if box[0] + 1e-4 < left < box[2] - 1e-4 or box[0] + 1e-4 < right < box[2] - 1e-4:
            return False
    return True


def _avoid_partial(pref: float, lo: float, hi: float, w: float, avoid: Sequence[Sequence[float]]) -> float:
    """Horizontal position closest to ``pref`` that leaves every other face fully in or fully out."""
    start = min(max(pref, lo), hi)
    if not avoid or _clean_edges(start, w, avoid):
        return start
    candidates = []
    for box in avoid:
        m = AVOID_MARGIN
        candidates += [box[2] + m + w / 2, box[0] - m - w / 2, box[0] - m + w / 2, box[2] + m - w / 2]
    valid = [c for c in candidates if lo - 1e-9 <= c <= hi + 1e-9 and _clean_edges(c, w, avoid)]
    return min(valid, key=lambda c: abs(c - pref)) if valid else start


def solve(geo: Geometry, *, desired_h: float, required: Sequence[Sequence[float]] = (),
          subject: Sequence[float] | None = None, subject_margins: tuple[float, float, float] | None = None,
          pair: Sequence[Sequence[float]] = (), caption_clear: bool = True,
          avoid: Sequence[Sequence[float]] = ()) -> tuple[Window, list[str]]:
    """Window for one framing decision. ``required`` boxes always end up inside (window widens)."""
    notes: list[str] = []
    must = [pad(b, REQUIRED_MARGIN) for b in required]
    focus: tuple[float, float, float, float] | None = None
    if subject is not None:
        focus = expand(subject, subject_margins or MARGINS["context"])
        must.append(focus)
    for face in pair:
        must.append(expand(face, MARGINS["pair"]))
    need = union(must)
    h_need = 0.0
    if need is not None:
        h_need = max(need[3] - need[1], (need[2] - need[0]) * geo.src_aspect / geo.out_aspect)
    h = max(desired_h, geo.h_min)
    if h_need > h:
        notes.append(f"widened_for_required:{h:.3f}->{h_need:.3f}")
        h = h_need
    h = min(h, geo.h_full)
    w = geo.width(h)
    x_lo = x_hi = y_lo = y_hi = None
    if need is not None:
        x_lo, x_hi, y_lo, y_hi = need[0], need[2], need[1], need[3]
    cx_range = _axis_range(0, 1, w, x_lo, x_hi)
    cy_range = _axis_range(0, 1, h, y_lo, y_hi)
    # composition preference (lowest priority)
    if subject is not None and not pair:
        pref_cx = (subject[0] + subject[2]) / 2
        face_cy = (subject[1] + subject[3]) / 2
        pref_cy = face_cy - HEADROOM * h + h / 2
        if caption_clear:  # face bottom above the caption band when the constraints allow it
            cy_limit = subject[3] - CAPTION_TOP * h + h / 2
            if cy_limit >= cy_range[0]:
                cy_range = (cy_range[0], min(cy_range[1], max(cy_range[0], cy_limit)))
    elif need is not None:
        pref_cx = (need[0] + need[2]) / 2
        content_mid = (need[1] + need[3]) / 2
        pref_cy = content_mid - CONTENT_Y * h + h / 2 if h > 1.0 else content_mid
    else:
        pref_cx, pref_cy = 0.5, 0.5 - CONTENT_Y * h + h / 2 if h > 1.0 else 0.5
    if h > 1.0 and need is None:
        pref_cy = 0.5 - CONTENT_Y * h + h / 2
    cx = _avoid_partial(pref_cx, cx_range[0], cx_range[1], w, avoid) if w < 1.0 else \
        min(max(pref_cx, cx_range[0]), cx_range[1])
    cy = min(max(pref_cy, cy_range[0]), cy_range[1])
    window = Window(round(cx, 6), round(cy, 6), round(h, 6))
    for box in required:
        if not window.contains(geo, clip_box(box), tol=REQUIRED_MARGIN + 1e-3):
            notes.append("required_not_contained")
    return window, notes


def clamp_window(geo: Geometry, window: Window) -> Window:
    """Project a (possibly interpolated) window into its valid position range for its size."""
    h = min(max(window.h, geo.h_min), geo.h_full)
    cx_range = _axis_range(0, 1, geo.width(h), None, None)
    cy_range = _axis_range(0, 1, h, None, None)
    return Window(min(max(window.cx, cx_range[0]), cx_range[1]), min(max(window.cy, cy_range[0]), cy_range[1]), h)


def full_frame(geo: Geometry) -> Window:
    h = geo.h_full
    return Window(0.5, 0.5 - CONTENT_Y * h + h / 2 if h > 1.0 else 0.5, h)


# ----------------------------------------------------------------- stack layout

STACK_SPLIT = 0.40


def panel_box(geo: Geometry, box: Sequence[float], panel_aspect: float) -> tuple[float, float, float, float]:
    """Expand a source box to the panel's pixel aspect (inside the frame when possible)."""
    x0, y0, x1, y1 = box
    w_px, h_px = (x1 - x0) * geo.src_w, (y1 - y0) * geo.src_h
    if w_px / max(h_px, 1e-6) < panel_aspect:
        w_px = h_px * panel_aspect
    else:
        h_px = w_px / panel_aspect
    w, h = min(1.0, w_px / geo.src_w), min(1.0, h_px / geo.src_h)
    cx = min(max((x0 + x1) / 2, w / 2), 1 - w / 2)
    cy = min(max((y0 + y1) / 2, h / 2), 1 - h / 2)
    return (round(cx - w / 2, 6), round(cy - h / 2, 6), round(cx + w / 2, 6), round(cy + h / 2, 6))


def stack_bottom_geometry(geo: Geometry) -> Geometry:
    height = int(round(geo.out_h * (1 - STACK_SPLIT)))
    return Geometry(geo.src_w, geo.src_h, geo.out_w, height, geo.max_zoom, geo.max_upscale)
