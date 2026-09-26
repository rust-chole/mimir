"""Face tracking: detect sparsely, track densely (LK flow), reacquire, consolidate.

Ported from the proven CLEAN V3 tracker. Per sample:
* live tracks are propagated with pyramidal Lucas-Kanade flow inside the box;
* the detector runs every ``detect_every`` samples, on shot cuts, and when flow
  quality or confidence drops;
* detections associate by IoU + center distance + appearance; unmatched tracks
  are held, then marked lost but reacquirable (no identity swap on occlusion);
* mouth-region motion energy (``activity``) is measured for speaker linking.
Fragments are consolidated into persistent subjects, never across a shot cut.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

from mimir.vision.faces import Detection, FaceDetector

DETECT_EVERY = 3
CONFIRM_HITS = 2
MAX_MISSED_S = 1.0
MAX_FLOW_ONLY_S = 3.0
REACQUIRE_WINDOW_S = 3.0
REACQUIRE_MIN_SIMILARITY = 0.45
MIN_FLOW_QUALITY = 0.35
REDETECT_CONFIDENCE = 0.45
FLOW_DECAY = 0.96
MAX_TRACKS = 6
MIN_SAMPLES = 3
MIN_MEDIAN_CONFIDENCE = 0.35
MIN_DETECTIONS = 2
STITCH_MAX_GAP_S = 6.0
STITCH_MAX_OVERLAP_S = 0.3
STITCH_EDGE_S = 1.0
STITCH_MAX_DISTANCE = 0.9
STITCH_SIZE_RATIO = (0.6, 1.6)


@dataclass(frozen=True)
class Sample:
    t: float
    cx: float
    cy: float
    w: float
    h: float
    confidence: float
    activity: float | None
    source: str             # detected | tracked

    def to_list(self) -> list[Any]:
        return [round(self.t, 3), round(self.cx, 4), round(self.cy, 4), round(self.w, 4), round(self.h, 4),
                round(self.confidence, 3), None if self.activity is None else round(self.activity, 5), self.source]

    @classmethod
    def from_list(cls, row: Sequence[Any]) -> "Sample":
        return cls(float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]),
                   None if row[6] is None else float(row[6]), str(row[7]))


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _appearance(image: np.ndarray, box: Sequence[float]) -> np.ndarray | None:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = max(0, int(box[0] * w)), max(0, int(box[1] * h)), min(w, int(box[2] * w)), min(h, int(box[3] * h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    hsv = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
    return hist


def _similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.5
    return max(0.0, float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL)))


def _match_score(track_box: Sequence[float], det: Detection, similarity: float) -> float | None:
    det_box = (det.x0, det.y0, det.x1, det.y1)
    overlap = box_iou(track_box, det_box)
    scale = max(track_box[2] - track_box[0], track_box[3] - track_box[1], 1e-6)
    distance = math.hypot((track_box[0] + track_box[2] - det.x0 - det.x1) / 2,
                          (track_box[1] + track_box[3] - det.y0 - det.y1) / 2) / scale
    if overlap < 0.05 and distance > 1.2:
        return None
    score = 0.6 * overlap + 0.3 * max(0.0, 1.0 - distance / 1.2) + 0.1 * similarity
    return score if score >= 0.25 else None


def _greedy(scores: dict[tuple[int, int], float]) -> list[tuple[int, int]]:
    used_t: set[int] = set()
    used_d: set[int] = set()
    pairs = []
    for (t, d), _ in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
        if t not in used_t and d not in used_d:
            used_t.add(t)
            used_d.add(d)
            pairs.append((t, d))
    return pairs


def _clip(box: Sequence[float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (max(0.0, min(1.0, float(v))) for v in box)
    return x0, y0, max(x1, x0 + 1e-3), max(y1, y0 + 1e-3)


@dataclass
class _Track:
    track_id: int
    box: tuple[float, float, float, float]
    confidence: float
    hits: int = 1
    last_seen: float = 0.0
    status: str = "tentative"
    samples: list[Sample] = field(default_factory=list)
    appearance: Any = None
    mouth_prev: Any = None
    upper_prev: Any = None


class FaceTracker:
    def __init__(self, detector: FaceDetector, detect_every: int = DETECT_EVERY) -> None:
        self.detector = detector
        self.detect_every = max(1, int(detect_every))
        self.tracks: list[_Track] = []
        self.next_id = 0
        self.prev_gray: np.ndarray | None = None
        self.index = 0
        self.cuts: list[float] = []
        self.stats = {"frames": 0, "detector_runs": 0, "detections": 0, "reacquired": 0}

    def _flow(self, gray: np.ndarray, track: _Track) -> tuple[tuple[float, float, float, float], float]:
        assert self.prev_gray is not None
        h, w = gray.shape[:2]
        x0, y0, x1, y1 = track.box
        mask = np.zeros_like(self.prev_gray)
        mask[int(y0 * h):max(int(y0 * h) + 1, int(y1 * h)), int(x0 * w):max(int(x0 * w) + 1, int(x1 * w))] = 255
        points = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=40, qualityLevel=0.01, minDistance=3, mask=mask)
        if points is None or len(points) < 4:
            return track.box, 0.0
        moved, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, points, None, winSize=(15, 15), maxLevel=2)
        ok = status.flatten() == 1
        old, new = points[ok].reshape(-1, 2), moved[ok].reshape(-1, 2)
        quality = len(new) / max(1, len(points))
        if len(new) < 4:
            return track.box, 0.0
        dx, dy = np.median(new - old, axis=0)
        spread_old = np.median(np.linalg.norm(old - old.mean(axis=0), axis=1)) + 1e-6
        spread_new = np.median(np.linalg.norm(new - new.mean(axis=0), axis=1)) + 1e-6
        scale = float(max(0.85, min(1.18, spread_new / spread_old)))
        cx, cy = (x0 + x1) / 2 + float(dx) / w, (y0 + y1) / 2 + float(dy) / h
        bw, bh = (x1 - x0) * scale, (y1 - y0) * scale
        return _clip((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)), float(quality)

    def _activity(self, gray: np.ndarray, track: _Track) -> float | None:
        """Mouth-region temporal change minus half the upper-face change (head motion)."""
        h, w = gray.shape[:2]
        x0, y0, x1, y1 = track.box
        px0, px1 = int(x0 * w), int(x1 * w)
        top, mid, bottom = int(y0 * h), int((y0 + 0.62 * (y1 - y0)) * h), int(y1 * h)
        if px1 - px0 < 6 or bottom - mid < 3 or mid - top < 3:
            return None
        mouth = cv2.resize(gray[mid:bottom, px0:px1], (24, 10)).astype(np.float32)
        upper = cv2.resize(gray[top:mid, px0:px1], (24, 16)).astype(np.float32)
        activity = None
        if track.mouth_prev is not None and track.upper_prev is not None:
            activity = max(0.0, float(np.mean(np.abs(mouth - track.mouth_prev)))
                           - 0.5 * float(np.mean(np.abs(upper - track.upper_prev)))) / 255.0
        track.mouth_prev, track.upper_prev = mouth, upper
        return activity

    def _record(self, track: _Track, t: float, gray: np.ndarray, source: str) -> None:
        x0, y0, x1, y1 = track.box
        track.samples.append(Sample(t, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0,
                                    max(0.0, min(1.0, track.confidence)), self._activity(gray, track), source))

    def update(self, image: np.ndarray, t: float, cut: bool) -> None:
        self.stats["frames"] += 1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if cut:
            self.cuts.append(round(t, 3))
            for track in self.tracks:
                if track.status != "lost":
                    track.status = "lost"
                    track.mouth_prev = track.upper_prev = None
        live = [track for track in self.tracks if track.status != "lost"]
        need_detect = cut or self.index % self.detect_every == 0
        flow_ok: set[int] = set()
        if self.prev_gray is not None and not cut:
            for track in live:
                box, quality = self._flow(gray, track)
                if quality < MIN_FLOW_QUALITY:
                    need_detect = True
                    track.confidence *= 0.85
                else:
                    track.box = box
                    track.confidence *= FLOW_DECAY * (0.7 + 0.3 * quality)
                    flow_ok.add(track.track_id)
                if track.confidence < REDETECT_CONFIDENCE:
                    need_detect = True
        detected: set[int] = set()
        if need_detect:
            self.stats["detector_runs"] += 1
            detections = self.detector.detect(image)
            self.stats["detections"] += len(detections)
            looks = [_appearance(image, (d.x0, d.y0, d.x1, d.y1)) for d in detections]
            candidates = [tr for tr in self.tracks if tr.status != "lost" or t - tr.last_seen <= REACQUIRE_WINDOW_S]
            scores: dict[tuple[int, int], float] = {}
            for ti, track in enumerate(candidates):
                for di, det in enumerate(detections):
                    similarity = _similarity(track.appearance, looks[di])
                    if track.status == "lost":
                        ratio = det.h / max(1e-6, track.box[3] - track.box[1])
                        if similarity < REACQUIRE_MIN_SIMILARITY or not 0.6 <= ratio <= 1.6:
                            continue
                        if any(c > track.last_seen for c in self.cuts):
                            continue  # never reacquire across a shot cut
                        scores[(ti, di)] = 0.2 + 0.8 * similarity
                    else:
                        score = _match_score(track.box, det, similarity)
                        if score is not None:
                            scores[(ti, di)] = score
            used: set[int] = set()
            for ti, di in _greedy(scores):
                track, det = candidates[ti], detections[di]
                if track.status == "lost":
                    self.stats["reacquired"] += 1
                track.box = _clip((det.x0, det.y0, det.x1, det.y1))
                track.confidence = det.confidence
                track.hits += 1
                track.last_seen = t
                track.status = "active" if track.hits >= CONFIRM_HITS else "tentative"
                if looks[di] is not None:
                    track.appearance = looks[di]
                detected.add(track.track_id)
                used.add(di)
            for di, det in enumerate(detections):
                if di in used or len([tr for tr in self.tracks if tr.status != "lost"]) >= MAX_TRACKS:
                    continue
                if any(box_iou(tr.box, (det.x0, det.y0, det.x1, det.y1)) > 0.3 for tr in self.tracks
                       if tr.status != "lost"):
                    continue
                self.tracks.append(_Track(self.next_id, _clip((det.x0, det.y0, det.x1, det.y1)), det.confidence,
                                          last_seen=t, appearance=looks[di]))
                detected.add(self.next_id)
                self.next_id += 1
        for track in self.tracks:
            if track.status == "lost":
                continue
            if track.track_id in detected:
                self._record(track, t, gray, "detected")
            elif (t - track.last_seen > MAX_MISSED_S and track.track_id not in flow_ok) \
                    or t - track.last_seen > MAX_FLOW_ONLY_S:
                track.status = "lost"
                track.mouth_prev = track.upper_prev = None
            elif track.status == "active":
                self._record(track, t, gray, "tracked")
        self.prev_gray = gray
        self.index += 1

    def finish(self, cuts: Sequence[float] | None = None) -> list[list[Sample]]:
        """Consolidate tracks. ``cuts`` are the confirmed (persistent) shot cuts; online cut flags
        that turned out to be flashes still reset tracking but do not block stitching."""
        raw = [sorted(tr.samples, key=lambda s: s.t) for tr in sorted(self.tracks, key=lambda tr: tr.track_id)
               if tr.hits >= CONFIRM_HITS and len(tr.samples) >= MIN_SAMPLES]
        return consolidate(raw, self.cuts if cuts is None else cuts)


def _edge_box(samples: Sequence[Sample], first: bool) -> tuple[float, float, float, float]:
    if first:
        edge = [s for s in samples if s.t <= samples[0].t + STITCH_EDGE_S]
    else:
        edge = [s for s in samples if s.t >= samples[-1].t - STITCH_EDGE_S]
    return (statistics.median(s.cx for s in edge), statistics.median(s.cy for s in edge),
            statistics.median(s.w for s in edge), statistics.median(s.h for s in edge))


def _continues(earlier: Sequence[Sample], later: Sequence[Sample], cuts: Sequence[float]) -> float | None:
    gap = later[0].t - earlier[-1].t
    if gap < -STITCH_MAX_OVERLAP_S or gap > STITCH_MAX_GAP_S:
        return None
    if any(earlier[-1].t - 0.05 < cut <= later[0].t + 0.05 for cut in cuts):
        return None
    ax, ay, aw, ah = _edge_box(earlier, first=False)
    bx, by, bw, bh = _edge_box(later, first=True)
    if not STITCH_SIZE_RATIO[0] <= bh / max(ah, 1e-6) <= STITCH_SIZE_RATIO[1]:
        return None
    distance = math.hypot(ax - bx, ay - by) / max(aw, bw, 1e-6)
    return distance if distance <= STITCH_MAX_DISTANCE else None


def consolidate(tracks: Sequence[Sequence[Sample]], cuts: Sequence[float]) -> list[list[Sample]]:
    """Drop weak fragments; stitch continuations into persistent subjects (never across a cut)."""
    kept = [list(t) for t in tracks if t and sum(1 for s in t if s.source == "detected") >= MIN_DETECTIONS
            and statistics.median(s.confidence for s in t) >= MIN_MEDIAN_CONFIDENCE]
    kept.sort(key=lambda t: t[0].t)
    merged: list[list[Sample]] = []
    for samples in kept:
        best: tuple[float, int] | None = None
        for index, existing in enumerate(merged):
            distance = _continues(existing, samples, cuts)
            if distance is not None and (best is None or distance < best[0]):
                best = (distance, index)
        if best is None:
            merged.append(samples)
            continue
        by_time = {round(s.t, 4): s for s in merged[best[1]]}
        for sample in samples:
            key = round(sample.t, 4)
            if key not in by_time or sample.confidence > by_time[key].confidence:
                by_time[key] = sample
        merged[best[1]] = sorted(by_time.values(), key=lambda s: s.t)
    return sorted(merged, key=lambda t: t[0].t)
