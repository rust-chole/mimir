"""Stage ``vision``: one streaming pass over the selected story window (source time)."""
from __future__ import annotations

import statistics
from typing import Any, Callable

import cv2
import numpy as np

from mimir.config import Settings, routes_for, section
from mimir.core.stage import StageContext, StageOutput
from mimir.media.frames import frame_at, iter_frames
from mimir.media.motion import (SETTLED_RATIO, MotionSample, color_hist, dedupe_cuts, hist_distance,
                                settled_distances, shot_boundaries, visual_peaks)
from mimir.media.probe import MediaInfo
from mimir.vision.faces import detector_fingerprint, load_detector
from mimir.vision.observer import observe
from mimir.vision.regions import RegionAccumulator, action_regions, classify_layout, static_text_regions
from mimir.vision.speaker_link import associate
from mimir.vision.tracking import FaceTracker, Sample

WINDOW_PAD = 0.5
ONLINE_CUT_DISTANCE = 0.5
STATIC_PATTERN_MAX_STD = 0.004
STATIC_PATTERN_MAX_ACTIVITY = 0.003


def _jpeg(image: np.ndarray, width: int = 640) -> bytes:
    h, w = image.shape[:2]
    if w > width:
        image = cv2.resize(image, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return buffer.tobytes()


def summarize_track(track_id: str, samples: list[Sample], span: float, fps: float) -> dict[str, Any]:
    t0, t1 = samples[0].t, samples[-1].t
    boxes = [(s.cx - s.w / 2, s.cy - s.h / 2, s.cx + s.w / 2, s.cy + s.h / 2) for s in samples]
    median_box = [round(statistics.median(b[i] for b in boxes), 4) for i in range(4)]
    position_std = float(np.hypot(np.std([s.cx for s in samples]), np.std([s.cy for s in samples])))
    activities = [s.activity for s in samples if s.activity is not None]
    activity = float(np.median(activities)) if activities else 0.0
    return {"id": track_id, "t0": round(t0, 3), "t1": round(t1, 3), "samples": [s.to_list() for s in samples],
            "coverage": round(min(1.0, len(samples) / max(1e-6, span * fps)), 3),
            "median_box": median_box, "position_std": round(position_std, 5), "median_activity": round(activity, 5),
            "static_pattern": position_std <= STATIC_PATTERN_MAX_STD and activity <= STATIC_PATTERN_MAX_ACTIVITY}


def face_boxes_at_factory(tracks: dict[str, list[Sample]]) -> Callable[[float], list[list[float]]]:
    def boxes(t: float) -> list[list[float]]:
        found = []
        for samples in tracks.values():
            nearest = min(samples, key=lambda s: abs(s.t - t))
            if abs(nearest.t - t) <= 0.3:
                found.append([nearest.cx - nearest.w / 2, nearest.cy - nearest.h / 2,
                              nearest.cx + nearest.w / 2, nearest.cy + nearest.h / 2])
        return found
    return boxes


class VisionStage:
    name = "vision"
    version = 1
    deps = ("source", "probe", "story", "speakers")

    def params(self, settings: Settings) -> Any:
        return {"vision": section(settings, "vision"), "routes": routes_for(settings, "visual_observer"),
                "detector": detector_fingerprint()}

    def run(self, ctx: StageContext) -> StageOutput:
        info = MediaInfo.from_dict(ctx.dep("probe").json("media"))
        story = ctx.dep("story").json("story")
        speakers = ctx.dep("speakers").json("speakers")
        cfg = ctx.settings.vision
        start = max(0.0, float(story["start"]) - WINDOW_PAD)
        end = min(info.duration, float(story["end"]) + WINDOW_PAD)
        detector, fallback = load_detector()
        if fallback:
            ctx.ledger.warning("face_detector_fallback", f"{fallback}; using the OpenCV Haar cascades "
                               "(lower recall on small, turned and partly covered faces)")
        tracker = FaceTracker(detector, cfg.detect_every)
        regions = RegionAccumulator()
        times: list[float] = []
        energies: list[float] = []
        jumps: list[float] = []
        hists: list[np.ndarray] = []
        previous_gray = None
        for frame in iter_frames(ctx.source.path, start=start, duration=end - start, fps=cfg.analysis_fps,
                                 width=cfg.analysis_width, src_width=info.width, src_height=info.height):
            hist = color_hist(frame.image)
            gray = cv2.cvtColor(cv2.resize(frame.image, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            distance = hist_distance(hists[-1], hist) if hists else 0.0
            energy = float(cv2.absdiff(gray, previous_gray).mean()) if previous_gray is not None else 0.0
            # online cut: tracking state resets immediately; only persistent cuts are published below
            cut = bool(hists) and distance > ONLINE_CUT_DISTANCE and energy >= 12.0
            tracker.update(frame.image, frame.t, cut)
            regions.update(frame.image, frame.t, cut)
            times.append(frame.t)
            energies.append(energy)
            jumps.append(distance)
            hists.append(hist)
            previous_gray = gray
        settled = settled_distances(times, hists, jumps)
        motion = [MotionSample(t, e, np.zeros((1, 1)), d, s) for t, e, d, s in zip(times, energies, jumps, settled)]
        index_of = {round(t, 3): i for i, t in enumerate(times)}
        confirmed = {c for c in tracker.cuts
                     if settled[index_of[c]] >= SETTLED_RATIO * ONLINE_CUT_DISTANCE}
        span = end - start
        cuts = dedupe_cuts(sorted(set(shot_boundaries(motion)) | confirmed))
        raw_tracks = tracker.finish(cuts)
        faces = [summarize_track(f"face_{index:02d}", samples, span, cfg.analysis_fps)
                 for index, samples in enumerate(raw_tracks)]
        live_tracks = {f["id"]: samples for f, samples in zip(faces, raw_tracks) if not f["static_pattern"]}
        links = associate(live_tracks, speakers.get("segments", []))
        for face in faces:
            link = links.get(face["id"])
            face["speaker"] = link["speaker"] if link else None
            face["speaker_confidence"] = link["confidence"] if link else 0.0
            face["speaker_evidence"] = link["evidence"] if link else ""
        actions = action_regions(regions, face_boxes_at_factory(live_tracks), cuts)
        ui_regions, static_coverage, _ = static_text_regions(regions)
        motion_level = float(np.median([m.energy for m in motion])) if motion else 0.0
        layout = classify_layout([f for f in faces if not f["static_pattern"]], static_coverage, motion_level,
                                 (start, end))
        peaks = visual_peaks(motion, cuts)
        observations: list[dict[str, Any]] = []
        if cfg.observer:
            times = self._observer_times(story, cuts, peaks, cfg.observer_max_frames, start, end)
            frames = [(t, _jpeg(frame_at(ctx.source.path, t, src_width=info.width, src_height=info.height,
                                         width=640))) for t in times]
            observations = observe(ctx.provider, ctx.settings.route("visual_observer"), frames,
                                   f"{story['title']}: {story['reason']}")
        else:
            ctx.ledger.info("observer_disabled", "multimodal observer disabled by configuration")
        return StageOutput(data={"vision": {
            "window": [round(start, 3), round(end, 3)], "analysis_fps": cfg.analysis_fps,
            "source_size": [info.width, info.height], "detector": detector.name,
            "shot_cuts": cuts, "faces": faces, "action_regions": actions, "ui_regions": ui_regions,
            "static_coverage": round(static_coverage, 4), "motion_level": round(motion_level, 3),
            "motion": [[round(m.t, 3), round(m.energy, 3)] for m in motion],
            "layout": layout, "visual_peaks": peaks, "observations": observations,
            "tracker_stats": tracker.stats,
        }})

    @staticmethod
    def _observer_times(story: dict[str, Any], cuts: list[float], peaks: list[dict[str, Any]], limit: int,
                        start: float, end: float) -> list[float]:
        times = [story["payoff"]["start"], (story["payoff"]["start"] + story["payoff"]["end"]) / 2,
                 story["payoff"]["end"]]
        times += [(b["start"] + b["end"]) / 2 for b in story["beats"] if b["duration"] > 0.2]
        times += [p["t"] for p in sorted(peaks, key=lambda p: -p["score"])[:3]]
        times += [c + 0.3 for c in cuts]
        chosen: list[float] = []
        for t in times:
            t = min(end - 0.1, max(start + 0.05, t))
            if all(abs(t - c) >= 0.6 for c in chosen):
                chosen.append(round(t, 3))
            if len(chosen) >= limit:
                break
        return sorted(chosen)
