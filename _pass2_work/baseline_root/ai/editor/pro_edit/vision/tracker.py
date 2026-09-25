"""Temporal multi-subject tracking: detect sparsely, track densely.

Per sampled frame (default 10 fps of the paced clip):

* active tracks are propagated with pyramidal Lucas-Kanade optical flow on
  features inside the box (translation + scale from point spread);
* the detector runs every ``detect_every`` samples AND immediately when a
  scene cut is seen, flow quality drops, a track's confidence decays, or no
  track is active;
* detections are associated to tracks (IoU + distance + appearance);
  unmatched tracks are held (not dropped) for ``max_missed_s``, then marked
  lost but kept reacquirable for ``reacquire_window_s`` so a face that
  reappears keeps its identity (no ID swap on brief occlusion);
* per face sample, mouth-region motion energy (``activity``) is measured for
  the speaker association layer.

Output: SubjectTrack objects in normalized PACED_CLIP coordinates. Raw boxes
are never used as camera centers directly (see subjects.smooth_center_path).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ai.editor.pro_edit.subjects import SubjectSample, SubjectTrack
from ai.editor.pro_edit.vision.association import (
    appearance,
    appearance_similarity,
    box_iou,
    greedy_assign,
    match_score,
)
from ai.editor.pro_edit.vision.detectors import Detection, FaceDetector
from ai.editor.pro_edit.vision.frames import SampledFrame

TRACKER_VERSION = 1


@dataclass(frozen=True)
class TrackerConfig:
    detect_every: int = 3
    confirm_hits: int = 2
    max_missed_s: float = 1.0
    max_flow_only_s: float = 3.0
    reacquire_window_s: float = 3.0
    reacquire_min_similarity: float = 0.45
    min_flow_quality: float = 0.35
    redetect_confidence: float = 0.45
    flow_confidence_decay: float = 0.96
    scene_cut_threshold: float = 0.5
    max_tracks: int = 6
    min_samples: int = 3


@dataclass
class _Track:
    track_id: int
    box: tuple[float, float, float, float]
    confidence: float
    kind: str
    hits: int = 1
    last_seen: float = 0.0
    status: str = "tentative"  # tentative | active | lost
    samples: list[SubjectSample] = field(default_factory=list)
    appearance: Any = None
    mouth_prev: Any = None
    upper_prev: Any = None


def _clip_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(max(0.0, min(1.0, float(v)))) for v in box)
    return (x0, y0, max(x1, x0 + 1e-3), max(y1, y0 + 1e-3))


class MultiSubjectTracker:
    def __init__(self, detector: FaceDetector, config: TrackerConfig | None = None) -> None:
        import cv2
        import numpy as np

        self._cv2 = cv2
        self._np = np
        self.detector = detector
        self.config = config or TrackerConfig()
        self.stats = {"frames": 0, "detector_runs": 0, "detections": 0, "scene_cuts": 0, "reacquired": 0,
                      "flow_updates": 0}

    # --- helpers ---------------------------------------------------------------

    def _scene_cut(self, prev_hist: Any, gray: Any) -> tuple[bool, Any]:
        cv2 = self._cv2
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
        if prev_hist is None:
            return False, hist
        distance = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        return distance > self.config.scene_cut_threshold, hist

    def _flow(self, prev_gray: Any, gray: Any, track: _Track) -> tuple[tuple[float, float, float, float], float]:
        cv2, np = self._cv2, self._np
        h, w = gray.shape[:2]
        x0, y0, x1, y1 = track.box
        mask = np.zeros_like(prev_gray)
        mask[int(y0 * h):max(int(y0 * h) + 1, int(y1 * h)), int(x0 * w):max(int(x0 * w) + 1, int(x1 * w))] = 255
        points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=40, qualityLevel=0.01, minDistance=3, mask=mask)
        if points is None or len(points) < 4:
            return track.box, 0.0
        moved, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None, winSize=(15, 15), maxLevel=2)
        good_old = points[status.flatten() == 1].reshape(-1, 2)
        good_new = moved[status.flatten() == 1].reshape(-1, 2)
        quality = len(good_new) / max(1, len(points))
        if len(good_new) < 4:
            return track.box, 0.0
        dx, dy = np.median(good_new - good_old, axis=0)
        spread_old = np.median(np.linalg.norm(good_old - good_old.mean(axis=0), axis=1)) + 1e-6
        spread_new = np.median(np.linalg.norm(good_new - good_new.mean(axis=0), axis=1)) + 1e-6
        scale = float(max(0.85, min(1.18, spread_new / spread_old)))
        cx = (x0 + x1) / 2.0 + float(dx) / w
        cy = (y0 + y1) / 2.0 + float(dy) / h
        bw, bh = (x1 - x0) * scale, (y1 - y0) * scale
        return _clip_box((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)), float(quality)

    def _activity(self, gray: Any, track: _Track) -> float | None:
        """Mouth-region temporal change minus half the upper-face change."""
        cv2, np = self._cv2, self._np
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
            mouth_change = float(np.mean(np.abs(mouth - track.mouth_prev)))
            upper_change = float(np.mean(np.abs(upper - track.upper_prev)))
            activity = max(0.0, mouth_change - 0.5 * upper_change) / 255.0
        track.mouth_prev, track.upper_prev = mouth, upper
        return activity

    def _record(self, track: _Track, t: float, gray: Any, source: str) -> None:
        x0, y0, x1, y1 = track.box
        activity = self._activity(gray, track) if track.kind == "face" else None
        track.samples.append(SubjectSample(t, (x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0,
                                           round(max(0.0, min(1.0, track.confidence)), 4), activity, source))

    # --- main loop -------------------------------------------------------------

    def run(self, frames: Iterable[SampledFrame], scene_changes: Sequence[float] = ()) -> tuple[SubjectTrack, ...]:
        cv2 = self._cv2
        cfg = self.config
        tracks: list[_Track] = []
        next_id = 0
        prev_gray = None
        prev_hist = None
        cut_times = sorted(scene_changes)
        cut_index = 0
        for frame in frames:
            self.stats["frames"] += 1
            gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
            cut, prev_hist = self._scene_cut(prev_hist, gray)
            while cut_index < len(cut_times) and cut_times[cut_index] <= frame.t:
                cut = cut or frame.t - cut_times[cut_index] < 0.2
                cut_index += 1
            if cut:
                self.stats["scene_cuts"] += 1
                for track in tracks:
                    if track.status != "lost":
                        track.status = "lost"
                        track.mouth_prev = track.upper_prev = None
            live = [t for t in tracks if t.status != "lost"]
            # Sparse cadence even when nothing is tracked: acquisition within one
            # detect interval, without running the detector on every sample.
            need_detect = cut or frame.index % cfg.detect_every == 0
            flow_ok: set[int] = set()
            if prev_gray is not None and not cut:
                for track in live:
                    box, quality = self._flow(prev_gray, gray, track)
                    self.stats["flow_updates"] += 1
                    if quality < cfg.min_flow_quality:
                        need_detect = True
                        track.confidence *= 0.85
                    else:
                        track.box = box
                        track.confidence *= cfg.flow_confidence_decay * (0.7 + 0.3 * quality)
                        flow_ok.add(id(track))
                    if track.confidence < cfg.redetect_confidence:
                        need_detect = True
            detected: set[int] = set()
            if need_detect:
                self.stats["detector_runs"] += 1
                detections = self.detector.detect(frame.image)
                self.stats["detections"] += len(detections)
                det_appearance = [appearance(frame.image, (d.x0, d.y0, d.x1, d.y1)) for d in detections]
                candidates = [t for t in tracks if t.status != "lost"
                              or frame.t - t.last_seen <= cfg.reacquire_window_s]
                scores: dict[tuple[int, int], float] = {}
                for ti, track in enumerate(candidates):
                    for di, det in enumerate(detections):
                        similarity = appearance_similarity(track.appearance, det_appearance[di])
                        if track.status == "lost":
                            size_ratio = (det.h / max(1e-6, track.box[3] - track.box[1]))
                            if similarity < cfg.reacquire_min_similarity or not 0.6 <= size_ratio <= 1.6:
                                continue
                            score = 0.2 + 0.8 * similarity
                        else:
                            score = match_score(track.box, det, similarity)
                            if score is None:
                                continue
                        scores[(ti, di)] = score
                used = set()
                for ti, di in greedy_assign(scores):
                    track, det = candidates[ti], detections[di]
                    if track.status == "lost":
                        self.stats["reacquired"] += 1
                    track.box = _clip_box((det.x0, det.y0, det.x1, det.y1))
                    track.confidence = det.confidence
                    track.hits += 1
                    track.last_seen = frame.t
                    track.status = "active" if track.hits >= cfg.confirm_hits else "tentative"
                    if det_appearance[di] is not None:
                        track.appearance = det_appearance[di]
                    detected.add(id(track))
                    used.add(di)
                for di, det in enumerate(detections):
                    if di in used or len([t for t in tracks if t.status != "lost"]) >= cfg.max_tracks:
                        continue
                    if any(box_iou(t.box, (det.x0, det.y0, det.x1, det.y1)) > 0.3 for t in tracks
                           if t.status != "lost"):
                        continue
                    tracks.append(_Track(next_id, _clip_box((det.x0, det.y0, det.x1, det.y1)), det.confidence,
                                         det.kind, last_seen=frame.t, appearance=det_appearance[di]))
                    detected.add(id(tracks[-1]))
                    next_id += 1
            for track in tracks:
                if track.status == "lost":
                    continue
                if id(track) in detected:
                    self._record(track, frame.t, gray, "detected")
                elif ((frame.t - track.last_seen > cfg.max_missed_s and id(track) not in flow_ok)
                      or frame.t - track.last_seen > cfg.max_flow_only_s):
                    track.status = "lost"  # held long enough; keep reacquirable
                    track.mouth_prev = track.upper_prev = None
                elif track.status == "active":
                    self._record(track, frame.t, gray, "tracked")
            prev_gray = gray
        return self._finalize(tracks)

    def _finalize(self, tracks: Sequence[_Track]) -> tuple[SubjectTrack, ...]:
        result = []
        ordinal = 0
        for track in sorted(tracks, key=lambda t: t.track_id):
            if track.hits < self.config.confirm_hits or len(track.samples) < self.config.min_samples:
                continue
            result.append(SubjectTrack(f"{track.kind}_{ordinal:02d}", track.kind,  # type: ignore[arg-type]
                                       tuple(sorted(track.samples, key=lambda s: s.t))))
            ordinal += 1
        return tuple(result)
