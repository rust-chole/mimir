"""ACTION REGION evidence for caption placement (geometry only, never story semantics).

During PAYOFF / REACTION story spans the place where the action happens should
not be covered by captions. Without a semantic object model MIMIR combines
evidence it already has:

    story span (role, time)  x  motion activity hotspot  [x tracked subject]
        -> ActionRegionEvidence(story_span_id, start, end, bbox, confidence, source)

Optional object evidence: ``MIMIR_CAPTION_OBJECTS=hog_person`` runs OpenCV's
built-in HOG pedestrian detector (coefficients ship inside OpenCV, Apache-2.0;
no download, no extra dependency) on sparse frames inside those spans. It
detects upright full-body people only; faces stay the tracker's job; hands and
generic objects are NOT supported (no license-verified local model is bundled)
and are never claimed. Detector failure never fails the short: the activity
path remains.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

ACTION_VERSION = 2
HOTSPOT_MIN_ACTIVITY = 0.30      # peak cell activity (0..1) needed to call something an action
HOTSPOT_REL = 0.5                # cells >= this fraction of the peak belong to the hotspot
HOTSPOT_MAX_AREA = 0.40          # a hotspot larger than this is global motion, not an action
# Restraint (V6 lesson, generic): the highest-motion cell group is not automatically "the action".
UI_EXPLAINED_SHARE = 0.5         # a hotspot this much inside persistent UI/HUD/chat text is UI motion
PERSON_EXPLAINED_SHARE = 0.6     # ... this much inside a tracked person's body region is that person moving
MIN_PERSISTENCE = 0.35           # share of the window's samples in which the hotspot actually moves
SMALL_AREA = 0.03                # a hotspot this small ...
CORNER = 0.25                    # ... centred in a corner band is ambiguous without other support
ACTION_ROLES = ("payoff", "reaction")
OBJECT_MODES = ("off", "hog_person")
HOG_SAMPLE_FPS = 1.0
HOG_MIN_SCORE = 0.5


@dataclass(frozen=True)
class ActionRegionEvidence:
    story_span_id: str
    start: float
    end: float
    bbox: tuple[float, float, float, float]      # source-normalized
    confidence: float
    evidence_source: str                          # story_span+activity | story_span+hog_person
    status: str = "action"                        # action | ambiguous | ui_motion | person_motion
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"story_span_id": self.story_span_id, "window": [round(self.start, 3), round(self.end, 3)],
                "bbox": [round(v, 4) for v in self.bbox], "confidence": round(self.confidence, 3),
                "evidence_source": self.evidence_source, "status": self.status, "note": self.note}


def _role(span: Any) -> str:
    role = getattr(span, "role", "")
    return str(getattr(role, "value", role))


def activity_hotspot(occupancy: Any, start: float, end: float
                     ) -> tuple[tuple[float, float, float, float], float] | None:
    """Largest connected high-activity group in [start, end] -> (normalized box, mean activity)."""
    values = occupancy.window(start, end)
    if not values:
        return None
    peak = max(values)
    if peak < HOTSPOT_MIN_ACTIVITY:
        return None
    cols, rows = occupancy.cols, occupancy.rows
    hot = {i for i, v in enumerate(values) if v >= HOTSPOT_REL * peak}
    best: list[int] = []
    seen: set[int] = set()
    for first in sorted(hot):
        if first in seen:
            continue
        group, stack = [], [first]
        seen.add(first)
        while stack:
            index = stack.pop()
            group.append(index)
            row, col = divmod(index, cols)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r, c = row + dr, col + dc
                other = r * cols + c
                if 0 <= r < rows and 0 <= c < cols and other in hot and other not in seen:
                    seen.add(other)
                    stack.append(other)
        if sum(values[i] for i in group) > sum(values[i] for i in best):
            best = group
    rs = [i // cols for i in best]
    cs = [i % cols for i in best]
    box = (min(cs) / cols, min(rs) / rows, (max(cs) + 1) / cols, (max(rs) + 1) / rows)
    if (box[2] - box[0]) * (box[3] - box[1]) > HOTSPOT_MAX_AREA:
        return None
    return box, sum(values[i] for i in best) / len(best)


def _overlap_share(box: Sequence[float], other: Sequence[float]) -> float:
    area = max(1e-9, (box[2] - box[0]) * (box[3] - box[1]))
    ix = max(0.0, min(box[2], other[2]) - max(box[0], other[0]))
    iy = max(0.0, min(box[3], other[3]) - max(box[1], other[1]))
    return ix * iy / area


def _person_region(face: Sequence[float]) -> tuple[float, float, float, float]:
    """Head + shoulders + torso implied by a face box: motion there is the person, not an action."""
    x0, y0, x1, y1 = face
    w, h = x1 - x0, y1 - y0
    return max(0.0, x0 - 1.3 * w), max(0.0, y0 - 0.7 * h), min(1.0, x1 + 1.3 * w), min(1.0, y1 + 4.5 * h)


def _face_boxes(tracks: Sequence[Any], start: float, end: float) -> list[tuple[float, float, float, float]]:
    boxes = []
    for track in tracks:
        if getattr(track, "kind", "") != "face":
            continue
        samples = [x for x in getattr(track, "samples", ()) if start <= x.t <= end and x.confidence >= 0.5]
        if samples:
            mid = samples[len(samples) // 2]
            boxes.append((mid.cx - mid.width / 2, mid.cy - mid.height / 2, mid.cx + mid.width / 2,
                          mid.cy + mid.height / 2))
    return boxes


def _persistence(occupancy: Any, box: Sequence[float], start: float, end: float) -> float | None:
    """Share of the window's samples in which the hotspot itself moves (None: no per-sample data)."""
    times = getattr(occupancy, "times", None)
    cells = getattr(occupancy, "cells", None)
    if not times or cells is None or not hasattr(occupancy, "region_score"):
        return None
    picked = [c for t, c in zip(times, cells) if start - 1e-6 <= t <= end + 1e-6]
    if not picked:
        return None
    moving = sum(1 for c in picked if occupancy.region_score(box, [v / 255.0 for v in c]) >= HOTSPOT_MIN_ACTIVITY * 0.5)
    return moving / len(picked)


def classify_hotspot(box: Sequence[float], *, occupancy: Any, start: float, end: float,
                     ui_boxes: Sequence[Sequence[float]] = (), faces: Sequence[Sequence[float]] = ()
                     ) -> tuple[str, str]:
    """(status, note): is this motion hotspot a story ACTION, or better explained by something else?"""
    ui_share = max((_overlap_share(box, ui) for ui in ui_boxes), default=0.0)
    if ui_share >= UI_EXPLAINED_SHARE:
        return "ui_motion", f"{ui_share:.0%} inside persistent UI/HUD/chat text"
    person_share = max((_overlap_share(box, _person_region(face)) for face in faces), default=0.0)
    if person_share >= PERSON_EXPLAINED_SHARE:
        return "person_motion", f"{person_share:.0%} inside a tracked person's body region"
    persistence = _persistence(occupancy, box, start, end)
    if persistence is not None and persistence < MIN_PERSISTENCE:
        return "ambiguous", f"moves in only {persistence:.0%} of the window (transient)"
    area = (box[2] - box[0]) * (box[3] - box[1])
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    if area < SMALL_AREA and (cx < CORNER or cx > 1 - CORNER) and (cy < CORNER or cy > 1 - CORNER):
        return "ambiguous", "small corner motion with no other support (overlay-like)"
    return "action", ""


def derive_action_regions(spans: Iterable[Any], occupancy: Any, *, objects: Sequence[ActionRegionEvidence] = (),
                          ui: Any = None, faces: Sequence[Any] = ()) -> list[ActionRegionEvidence]:
    """Activity hotspots inside PAYOFF / REACTION spans (+ optional detector evidence), each
    classified: only a corroborated ``action`` is an action region; UI/chat motion, a person's own
    movement, transient or overlay-like corner motion are kept as evidence with their status."""
    from ai.editor.pro_edit.caption_background import ui_boxes as find_ui_boxes

    found: list[ActionRegionEvidence] = []
    for span in spans:
        if _role(span) not in ACTION_ROLES or occupancy is None:
            continue
        hotspot = activity_hotspot(occupancy, float(span.start), float(span.end))
        if hotspot is None:
            continue
        box, mean = hotspot
        uis = [b for b, _v in find_ui_boxes(ui, float(span.start), float(span.end))] if ui is not None else []
        status, note = classify_hotspot(box, occupancy=occupancy, start=float(span.start), end=float(span.end),
                                        ui_boxes=uis, faces=_face_boxes(faces, float(span.start), float(span.end)))
        confidence = min(1.0, mean) * (1.0 if status in ("action", "person_motion") else 0.5)
        found.append(ActionRegionEvidence(str(span.span_id), float(span.start), float(span.end), box,
                                          confidence, "story_span+activity", status, note))
    found.extend(objects)
    return found


class HogPersonDetector:
    """OpenCV built-in HOG + linear SVM pedestrian detector (upright full bodies only)."""

    name = "hog_person"

    def __init__(self) -> None:
        from ai.editor.pro_edit.vision.cv_runtime import load_opencv

        cv2 = load_opencv()
        self._cv2 = cv2
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, image: Any) -> list[tuple[tuple[float, float, float, float], float]]:
        h, w = image.shape[:2]
        rects, weights = self._hog.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.05)
        found = []
        for (x, y, bw, bh), score in zip(rects if len(rects) else [], weights if len(weights) else []):
            if float(score) >= HOG_MIN_SCORE:
                found.append(((x / w, y / h, (x + bw) / w, (y + bh) / h), min(1.0, float(score))))
        return found


def detect_people_in_spans(path: str | Path, width: int, height: int, spans: Iterable[Any],
                           detector: Any | None = None) -> tuple[list[ActionRegionEvidence], str]:
    """Optional detector pass over PAYOFF / REACTION spans. Never raises: (evidence, status)."""
    wanted = [s for s in spans if _role(s) in ACTION_ROLES]
    if not wanted:
        return [], "no action spans"
    try:
        detector = detector or HogPersonDetector()
        from ai.editor.pro_edit.vision.frames import iterate_frames

        last_end = max(float(s.end) for s in wanted)
        boxes: dict[str, list[tuple[tuple[float, float, float, float], float]]] = {}
        for sample in iterate_frames(path, width, height, sample_fps=HOG_SAMPLE_FPS, max_width=640):
            if sample.t > last_end:
                break
            inside = [s for s in wanted if float(s.start) <= sample.t <= float(s.end)]
            if inside:
                detections = detector.detect(sample.image)
                for span in inside:
                    boxes.setdefault(str(span.span_id), []).extend(detections)
        found: list[ActionRegionEvidence] = []
        for span in wanted:
            rows = boxes.get(str(span.span_id), [])
            if rows:
                found.append(ActionRegionEvidence(
                    str(span.span_id), float(span.start), float(span.end),
                    (min(b[0][0] for b in rows), min(b[0][1] for b in rows), max(b[0][2] for b in rows),
                     max(b[0][3] for b in rows)), max(b[1] for b in rows), "story_span+hog_person"))
        return found, f"{getattr(detector, 'name', 'detector')}: {len(found)} span(s) with people"
    except Exception as error:  # optional evidence: the activity path remains
        return [], f"object detector unavailable ({type(error).__name__}: {str(error)[:160]})"
