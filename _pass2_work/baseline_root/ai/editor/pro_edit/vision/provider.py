"""Cached local-vision subject provider (writes/reads the standard sidecar)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

from ai.editor.pro_edit.errors import SubjectResolutionError
from ai.editor.pro_edit.subjects import JsonSidecarSubjectProvider, SubjectTrack, write_sidecar
from ai.editor.pro_edit.vision.detectors import FaceDetector, load_detector
from ai.editor.pro_edit.vision.frames import DEFAULT_MAX_WIDTH, DEFAULT_SAMPLE_FPS, iterate_frames
from ai.editor.pro_edit.vision.tracker import TRACKER_VERSION, MultiSubjectTracker, TrackerConfig


def _fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class LocalVisionSubjectProvider:
    """Runs detection + tracking once per paced clip; cached by fingerprint."""

    name = "local_vision"

    def __init__(self, video_path: str | Path, *, width: int, height: int, cache_path: str | Path,
                 scene_changes: Sequence[float] = (), force: bool = False,
                 detector: FaceDetector | None = None, config: TrackerConfig | None = None,
                 sample_fps: float = DEFAULT_SAMPLE_FPS) -> None:
        self.video_path = Path(video_path)
        self.width, self.height = int(width), int(height)
        self.cache_path = Path(cache_path)
        self.scene_changes = tuple(scene_changes)
        self.force = force
        self._detector = detector
        self.config = config or TrackerConfig()
        self.sample_fps = sample_fps
        self.diagnostics: dict[str, Any] = {}

    def _meta(self, detector_name: str) -> dict[str, Any]:
        return {"tracker_version": TRACKER_VERSION, "detector": detector_name,
                "video": _fingerprint(self.video_path), "sample_fps": self.sample_fps,
                "config": self.config.__dict__}

    def _cached(self, detector_name: str) -> tuple[SubjectTrack, ...] | None:
        if self.force or not self.cache_path.is_file():
            return None
        try:
            meta = json.loads(self.cache_path.read_text(encoding="utf-8")).get("meta", {})
        except (OSError, ValueError):
            return None
        if meta != json.loads(json.dumps(self._meta(detector_name))):
            return None
        return JsonSidecarSubjectProvider(self.cache_path).load(float("inf"))

    def load(self, duration_s: float) -> tuple[SubjectTrack, ...]:
        detector = self._detector
        reason = "injected"
        if detector is None:
            detector, reason = load_detector()
        if detector is None:
            self.diagnostics = {"status": "unavailable", "reason": reason}
            return ()
        cached = self._cached(detector.name)
        if cached is not None:
            self.diagnostics = {"status": "cache", "detector": detector.name, "tracks": len(cached)}
            return tuple(t for t in cached if t.samples and t.samples[0].t <= duration_s + 0.5)
        started = time.perf_counter()
        tracker = MultiSubjectTracker(detector, self.config)
        try:
            tracks = tracker.run(iterate_frames(self.video_path, self.width, self.height,
                                                sample_fps=self.sample_fps, max_width=DEFAULT_MAX_WIDTH),
                                 self.scene_changes)
        except SubjectResolutionError as error:
            self.diagnostics = {"status": "failed", "detector": detector.name, "reason": str(error)}
            return ()
        write_sidecar(self.cache_path, tracks, meta=self._meta(detector.name))
        self.diagnostics = {"status": "tracked", "detector": detector.name, "detector_reason": reason,
                            "tracks": len(tracks), "seconds": round(time.perf_counter() - started, 2),
                            **tracker.stats}
        return tracks

    def load_regions(self, duration_s: float) -> tuple:
        return ()
