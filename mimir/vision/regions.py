"""Motion/action regions, static HUD/UI text regions and content-layout classification."""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

MOTION_COLS, MOTION_ROWS = 16, 9
EDGE_COLS, EDGE_ROWS = 32, 18
ACTION_WINDOW_S = 0.5
MIN_ACTION_AREA = 0.01
STATIC_EDGE_DENSITY = 0.07
STATIC_MAX_MOTION = 2.5
SCREEN_COVERAGE = 0.22
FACECAM_MOTION = 3.0          # full-frame motion that alone marks gameplay under a corner face
OVERLAY_MOTION = 1.5          # enough motion when the corner face is a pixel-stable overlay
OVERLAY_POSITION_STD = 0.004  # composited facecam: face box position barely varies


@dataclass
class RegionAccumulator:
    times: list[float] = field(default_factory=list)
    motion: list[np.ndarray] = field(default_factory=list)
    edges: list[np.ndarray] = field(default_factory=list)
    edge_motion: list[np.ndarray] = field(default_factory=list)
    _prev: np.ndarray | None = None

    def update(self, image: np.ndarray, t: float, cut: bool) -> None:
        gray = cv2.cvtColor(cv2.resize(image, (320, int(round(320 * image.shape[0] / image.shape[1]))),
                                       interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180).astype(np.float32) / 255.0
        if self._prev is None or cut:
            diff = np.zeros_like(gray, dtype=np.float32)
        else:
            diff = cv2.absdiff(gray, self._prev).astype(np.float32)
        self.times.append(t)
        self.motion.append(cv2.resize(diff, (MOTION_COLS, MOTION_ROWS), interpolation=cv2.INTER_AREA))
        self.edges.append(cv2.resize(edges, (EDGE_COLS, EDGE_ROWS), interpolation=cv2.INTER_AREA))
        self.edge_motion.append(cv2.resize(diff, (EDGE_COLS, EDGE_ROWS), interpolation=cv2.INTER_AREA))
        self._prev = gray


def _components(mask: np.ndarray) -> list[tuple[float, float, float, float, float]]:
    """Connected components of a cell mask -> normalized boxes (x0, y0, x1, y1, area)."""
    rows, cols = mask.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        boxes.append((x / cols, y / rows, (x + w) / cols, (y + h) / rows, area / float(rows * cols)))
    return boxes


def _person_region(face: Sequence[float]) -> tuple[float, float, float, float]:
    """Head + shoulders + torso implied by a face box (motion here is the person, not an action)."""
    x0, y0, x1, y1 = face
    w, h = x1 - x0, y1 - y0
    return x0 - 1.3 * w, y0 - 0.7 * h, x1 + 1.3 * w, y1 + 4.5 * h


def _explained_by_people(box: Sequence[float], faces: Sequence[Sequence[float]]) -> bool:
    area = max(1e-9, (box[2] - box[0]) * (box[3] - box[1]))
    covered = 0.0
    for face in faces:
        px0, py0, px1, py1 = _person_region(face)
        ix = max(0.0, min(box[2], px1) - max(box[0], px0))
        iy = max(0.0, min(box[3], py1) - max(box[1], py0))
        covered = max(covered, ix * iy / area)
    return covered > 0.6


def action_regions(acc: RegionAccumulator, face_boxes_at: Any, cuts: Sequence[float]) -> list[dict[str, Any]]:
    """Motion hotspots grouped into regions over time (not explained by talking faces or edit cuts)."""
    if len(acc.times) < 4:
        return []
    motion = np.stack(acc.motion)
    baseline = float(np.median(motion))
    mad = float(np.median(np.abs(motion - baseline))) or 0.5
    threshold = max(8.0, baseline + 6.0 * 1.4826 * mad)
    step = max(1, int(round(ACTION_WINDOW_S / max(1e-3, (acc.times[-1] - acc.times[0]) / max(1, len(acc.times) - 1)))))
    windows = []
    for start in range(0, len(acc.times), step):
        stop = min(len(acc.times), start + step)
        t0, t1 = acc.times[start], acc.times[stop - 1]
        if any(t0 - 0.2 <= c <= t1 + 0.2 for c in cuts):
            continue
        mean = motion[start:stop].mean(axis=0)
        for box in _components(mean >= threshold):
            if box[4] < MIN_ACTION_AREA:
                continue
            cells = mean[int(box[1] * MOTION_ROWS):int(np.ceil(box[3] * MOTION_ROWS)),
                         int(box[0] * MOTION_COLS):int(np.ceil(box[2] * MOTION_COLS))]
            if _explained_by_people(box[:4], face_boxes_at((t0 + t1) / 2)):
                continue
            windows.append({"t0": t0, "t1": t1, "box": list(box[:4]), "intensity": float(cells.max())})
    regions: list[dict[str, Any]] = []
    for window in windows:
        match = None
        for region in regions:
            b = region["box"]
            overlap = max(0.0, min(b[2], window["box"][2]) - max(b[0], window["box"][0])) * \
                max(0.0, min(b[3], window["box"][3]) - max(b[1], window["box"][1]))
            if window["t0"] - region["t1"] <= ACTION_WINDOW_S * 1.5 and overlap > 0:
                match = region
                break
        if match is None:
            regions.append({"t0": window["t0"], "t1": window["t1"], "box": list(window["box"]),
                            "peak": window["intensity"], "windows": 1})
        else:
            match["t1"] = window["t1"]
            match["box"] = [min(match["box"][0], window["box"][0]), min(match["box"][1], window["box"][1]),
                            max(match["box"][2], window["box"][2]), max(match["box"][3], window["box"][3])]
            match["peak"] = max(match["peak"], window["intensity"])
            match["windows"] += 1
    result = []
    for index, region in enumerate(sorted(regions, key=lambda r: r["t0"])):
        result.append({"id": f"act_{index:02d}", "t0": round(region["t0"], 3),
                       "t1": round(region["t1"] + ACTION_WINDOW_S * 0.5, 3),
                       "box": [round(v, 4) for v in region["box"]], "intensity": round(region["peak"], 2),
                       "relative_intensity": round(region["peak"] / max(1.0, threshold), 3)})
    return result


def static_text_regions(acc: RegionAccumulator) -> tuple[list[dict[str, Any]], float, np.ndarray]:
    """Persistently textured, motionless cells: HUD, overlays, UI panels, on-screen text."""
    if not acc.edges:
        return [], 0.0, np.zeros((EDGE_ROWS, EDGE_COLS), dtype=bool)
    edges = np.stack(acc.edges)
    motion = np.stack(acc.edge_motion)
    density = edges.mean(axis=0)
    stability = edges.std(axis=0)
    still = motion.mean(axis=0)
    mask = (density >= STATIC_EDGE_DENSITY) & (stability <= density * 0.6 + 0.02) & (still <= STATIC_MAX_MOTION)
    regions = []
    for index, box in enumerate(b for b in _components(mask) if b[4] >= 0.004):
        regions.append({"id": f"ui_{index:02d}", "box": [round(v, 4) for v in box[:4]], "area": round(box[4], 4)})
    return regions, float(mask.mean()), mask


def classify_layout(faces: Sequence[dict[str, Any]], static_coverage: float, motion_level: float,
                    window: tuple[float, float]) -> dict[str, Any]:
    """faces: persistent (non-static) face subjects with coverage/median box. Rules are general, not per-clip."""
    span = max(1e-6, window[1] - window[0])
    persistent = [f for f in faces if f["coverage"] >= 0.25]
    big = [f for f in persistent if f["median_box"][3] - f["median_box"][1] >= 0.12]
    facecams = []
    for face in persistent:
        x0, y0, x1, y1 = face["median_box"]
        cx, cy, h = (x0 + x1) / 2, (y0 + y1) / 2, y1 - y0
        cornered = (cx < 0.33 or cx > 0.67) and (cy < 0.40 or cy > 0.60)
        if h < 0.24 and cornered and face["position_std"] < 0.02:
            facecams.append(face)
    # a composited facecam is pixel-stable; a real person in a corner of the scene still drifts
    overlay = [f for f in facecams if f["position_std"] < OVERLAY_POSITION_STD]
    if (facecams and motion_level >= FACECAM_MOTION) or (overlay and motion_level >= OVERLAY_MOTION):
        facecams = overlay or facecams
        face = max(facecams, key=lambda f: f["coverage"])
        x0, y0, x1, y1 = face["median_box"]
        w, h = x1 - x0, y1 - y0
        box = [max(0.0, x0 - 0.7 * w), max(0.0, y0 - 0.8 * h), min(1.0, x1 + 0.7 * w), min(1.0, y1 + 0.8 * h)]
        box = [0.0 if box[0] < 0.05 else box[0], 0.0 if box[1] < 0.05 else box[1],
               1.0 if box[2] > 0.95 else box[2], 1.0 if box[3] > 0.95 else box[3]]
        return {"class": "facecam_gameplay", "confidence": 0.8, "facecam_track": face["id"],
                "facecam_box": [round(v, 4) for v in box], "gameplay_box": [0.0, 0.0, 1.0, 1.0],
                "reason": "small fixed face in a corner over high-motion full-frame content"}
    if static_coverage >= SCREEN_COVERAGE and not big:
        return {"class": "screen_content", "confidence": min(0.95, 0.5 + static_coverage),
                "reason": f"static textured UI/text covers {static_coverage:.0%} of the frame without a large face"}
    together = 0.0
    if len(big) >= 2:
        a, b = sorted(big, key=lambda f: -f["coverage"])[:2]
        overlap = max(0.0, min(a["t1"], b["t1"]) - max(a["t0"], b["t0"]))
        together = overlap / span
    if len(big) >= 2 and together >= 0.4:
        return {"class": "multi_person", "confidence": 0.75, "reason": f"{len(big)} persistent faces on screen together"}
    if len(big) == 1:
        return {"class": "talking_head", "confidence": 0.75, "reason": "one persistent prominent face"}
    if not big and motion_level >= 3.0 and static_coverage < 0.08:
        return {"class": "gameplay", "confidence": 0.55, "reason": "high full-frame motion without faces"}
    return {"class": "scene", "confidence": 0.5, "reason": "general scene (IRL / mixed)"}
