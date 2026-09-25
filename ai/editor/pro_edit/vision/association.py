"""Detection <-> track association (IoU, center distance, cheap appearance)."""
from __future__ import annotations

import math
from typing import Any, Sequence

from ai.editor.pro_edit.vision.cv_runtime import OpenCVUnavailable, load_opencv
from ai.editor.pro_edit.vision.detectors import Detection

IOU_WEIGHT = 0.6
DISTANCE_WEIGHT = 0.3
APPEARANCE_WEIGHT = 0.1
DISTANCE_GATE = 1.2      # in units of the track's larger box side
MIN_MATCH_SCORE = 0.25


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Center distance normalized by a's larger side."""
    scale = max(a[2] - a[0], a[3] - a[1], 1e-6)
    return math.hypot((a[0] + a[2] - b[0] - b[2]) / 2.0, (a[1] + a[3] - b[1] - b[3]) / 2.0) / scale


def appearance(image: Any, box: Sequence[float]) -> Any | None:
    """Normalized hue/saturation histogram of the box (None if unavailable)."""
    try:
        cv2 = load_opencv()
    except OpenCVUnavailable:
        return None
    h, w = image.shape[:2]
    x0, y0 = max(0, int(box[0] * w)), max(0, int(box[1] * h))
    x1, y1 = min(w, int(box[2] * w)), min(h, int(box[3] * h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    hsv = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
    return hist


def appearance_similarity(a: Any | None, b: Any | None) -> float:
    if a is None or b is None:
        return 0.5  # unknown: neutral
    cv2 = load_opencv()

    return max(0.0, float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL)))


def match_score(track_box: Sequence[float], det: Detection, similarity: float) -> float | None:
    det_box = (det.x0, det.y0, det.x1, det.y1)
    iou = box_iou(track_box, det_box)
    dist = center_distance(track_box, det_box)
    if iou < 0.05 and dist > DISTANCE_GATE:
        return None
    score = IOU_WEIGHT * iou + DISTANCE_WEIGHT * max(0.0, 1.0 - dist / DISTANCE_GATE) + APPEARANCE_WEIGHT * similarity
    return score if score >= MIN_MATCH_SCORE else None


def greedy_assign(scores: dict[tuple[int, int], float]) -> list[tuple[int, int]]:
    """Highest score first; each track and detection used at most once."""
    used_t: set[int] = set()
    used_d: set[int] = set()
    pairs = []
    for (t, d), _score in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
        if t in used_t or d in used_d:
            continue
        used_t.add(t)
        used_d.add(d)
        pairs.append((t, d))
    return pairs
