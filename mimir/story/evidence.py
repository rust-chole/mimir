"""Stage ``vod_evidence``: cheap whole-VOD signals so discovery is not transcript-blind.

Local and deterministic over the whole source (streamed, low resolution):
audio loudness/suddenness transients and motion bursts. Only the strongest few
regions are then shown to a multimodal observer (before / at / after frames),
which is the ONLY model call here - bounded by ``peak_probe_max_regions``.
Visual-only payoffs (a door breaking, a fall) therefore reach story discovery
as REQUIRED-REVIEW events without a whole-VOD video-model pass.
"""
from __future__ import annotations

import math

import io
from typing import Any

import cv2
import numpy as np

from mimir.config import Settings, routes_for
from mimir.core.stage import StageContext, StageOutput
from mimir.media.audio import audio_peaks, decode_pcm, envelope
from mimir.media.frames import frame_at, iter_frames
from mimir.media.motion import SETTLE_WINDOW, MotionSample, motion_series, shot_boundaries, visual_peaks
from mimir.media.probe import MediaInfo
from mimir.models.provider import ImageInput
from mimir.story.prompts import PEAK_PROBE_INSTRUCTIONS, PEAK_PROBE_SCHEMA

AUDIO_CHUNK_SECONDS = 600.0
MOTION_FPS = 2.0
MOTION_WIDTH = 160
GUARD_MIN_AUDIO_SCORE = 0.82
GUARD_MIN_PEAK_DBFS = -13.0
DECISIVE_TYPES = {"object_break", "destruction", "impact", "crash", "fall", "explosion", "visual_payoff",
                  "physical_action", "reveal", "sudden_change", "gameplay_event"}
REQUIRED_REVIEW_MIN_CONFIDENCE = 0.55


def _jpeg(image: np.ndarray, width: int = 512) -> bytes:
    h, w = image.shape[:2]
    if w > width:
        image = cv2.resize(image, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buffer.tobytes()


def whole_audio_peaks(path: str, duration: float, max_peaks: int = 24) -> list[dict[str, Any]]:
    peaks: list[dict[str, Any]] = []
    offset = 0.0
    while offset < duration - 0.05:
        length = min(AUDIO_CHUNK_SECONDS, duration - offset)
        samples = decode_pcm(path, start=offset, duration=length, rate=8000)
        env = envelope(samples, 8000)
        for peak in audio_peaks(env, max_peaks=max_peaks):
            peaks.append({k: (round(v + offset, 3) if k in ("center", "start", "end") else v) for k, v in peak.items()})
        offset += length
    peaks.sort(key=lambda p: -p["score"])
    return sorted(peaks[:max_peaks], key=lambda p: p["center"])


def _merge_block(samples: list[MotionSample], rows: list[MotionSample]) -> None:
    """Append a block's rows. Consecutive blocks overlap by two settle windows: rows in the first
    half of the overlap keep the previous block's values (full look-ahead there), rows after it
    come from the new block (full look-back there)."""
    if not rows:
        return
    if samples:
        seam = rows[0].t + SETTLE_WINDOW[1] + 1e-6
        while samples and samples[-1].t > seam:
            samples.pop()
        rows = [row for row in rows if row.t > seam]
    samples.extend(rows)


def whole_motion(path: str, info: MediaInfo) -> tuple[list[dict[str, Any]], list[float]]:
    frames = iter_frames(path, fps=MOTION_FPS, width=MOTION_WIDTH, src_width=info.width, src_height=info.height)
    # stream in blocks so memory stays flat for multi-hour VODs; blocks overlap so every sample's
    # settled (cut-persistence) distance sees its full look-back and look-ahead
    overlap = 2 * int(math.ceil(SETTLE_WINDOW[1] * MOTION_FPS)) + 2
    samples: list[MotionSample] = []
    block: list = []
    for frame in frames:
        block.append(frame)
        if len(block) >= 1200:
            _merge_block(samples, motion_series(block))
            block = block[-overlap:]
    if len(block) > 1 or not samples:
        _merge_block(samples, motion_series(block))
    cuts = shot_boundaries(samples)
    return visual_peaks(samples, cuts, max_peaks=24), cuts


class VodEvidenceStage:
    name = "vod_evidence"
    version = 1
    deps = ("source", "probe")

    def params(self, settings: Settings) -> Any:
        return {"max_regions": settings.story.peak_probe_max_regions, "routes": routes_for(settings, "peak_probe")}

    def run(self, ctx: StageContext) -> StageOutput:
        info = MediaInfo.from_dict(ctx.dep("probe").json("media"))
        audio = whole_audio_peaks(str(ctx.source.path), info.duration) if info.has_audio else []
        motion, cuts = whole_motion(str(ctx.source.path), info)
        regions = self._regions(audio, motion, info.duration, ctx.settings.story.peak_probe_max_regions)
        events: list[dict[str, Any]] = []
        if regions:
            events = self._probe(ctx, info, regions)
        guards = self._guards(audio, events)
        return StageOutput(data={"vod_evidence": {
            "audio_peaks": audio, "motion_peaks": motion, "shot_cuts": cuts,
            "probe_regions": regions, "visual_events": events + guards,
        }})

    @staticmethod
    def _regions(audio: list[dict[str, Any]], motion: list[dict[str, Any]], duration: float, limit: int
                 ) -> list[dict[str, Any]]:
        candidates = [{"t": p["center"], "score": p["score"], "signals": ["audio_transient"]} for p in audio
                      if p["score"] >= 0.55]
        candidates += [{"t": p["t"], "score": p["score"], "signals": ["motion_burst"]} for p in motion]
        candidates.sort(key=lambda c: -c["score"])
        regions: list[dict[str, Any]] = []
        for candidate in candidates:
            near = next((r for r in regions if abs(r["t"] - candidate["t"]) < 2.0), None)
            if near is not None:
                near["signals"] = sorted(set(near["signals"]) | set(candidate["signals"]))
                near["score"] = max(near["score"], candidate["score"])
                continue
            if len(regions) < limit:
                regions.append({"t": candidate["t"], "score": candidate["score"], "signals": candidate["signals"]})
        for index, region in enumerate(sorted(regions, key=lambda r: r["t"])):
            region["region_id"] = f"r{index + 1:02d}"
            region["times"] = [round(max(0.0, region["t"] - 0.8), 3), round(region["t"], 3),
                               round(min(duration - 0.05, region["t"] + 0.9), 3)]
        return sorted(regions, key=lambda r: r["t"])

    @staticmethod
    def _probe(ctx: StageContext, info: MediaInfo, regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        images: list[ImageInput] = []
        lines = []
        for region in regions:
            lines.append(f"{region['region_id']}: measured {', '.join(region['signals'])} at {region['t']:.2f}s")
            for label, t in zip(("before", "at", "after"), region["times"]):
                image = frame_at(ctx.source.path, t, src_width=info.width, src_height=info.height, width=512)
                images.append(ImageInput(_jpeg(image), f"{region['region_id']} {label} ({t:.2f}s)"))
        result = ctx.provider.json_task(
            "peak_probe", ctx.settings.route("peak_probe"), instructions=PEAK_PROBE_INSTRUCTIONS,
            input_text="REGIONS:\n" + "\n".join(lines), schema=PEAK_PROBE_SCHEMA, schema_name="mimir_peak_probe_v1",
            images=images)
        by_id = {r["region_id"]: r for r in regions}
        events: list[dict[str, Any]] = []
        for row in result.get("regions", []):
            region = by_id.get(str(row.get("region_id", "")))
            if region is None:
                continue
            for event in row.get("events", []):
                confidence = max(0.0, min(1.0, float(event.get("confidence", 0.0))))
                kind = str(event.get("type", "other"))
                events.append({
                    "event_id": f"v{len(events) + 1:03d}", "start": region["times"][0], "end": region["times"][2],
                    "type": kind, "description": " ".join(str(event.get("description", "")).split())[:300],
                    "confidence": round(confidence, 3), "signals": region["signals"],
                    "required_review": kind in DECISIVE_TYPES and confidence >= REQUIRED_REVIEW_MIN_CONFIDENCE,
                })
        return events

    @staticmethod
    def _guards(audio: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Very strong audio transients no semantic event explains: review evidence, never automatic winners."""
        guards: list[dict[str, Any]] = []
        for peak in sorted(audio, key=lambda p: -p["score"]):
            if peak["score"] < GUARD_MIN_AUDIO_SCORE or peak["peak_dbfs"] < GUARD_MIN_PEAK_DBFS:
                continue
            if any(e["start"] - 0.6 <= peak["center"] <= e["end"] + 0.6 and e["required_review"] for e in events):
                continue
            if any(abs(g["center"] - peak["center"]) < 1.2 for g in guards):
                continue
            guards.append({"event_id": f"g{len(guards) + 1:02d}", "start": peak["start"], "end": peak["end"],
                           "center": peak["center"], "type": "source_peak_guard",
                           "description": (f"abrupt source transient at {peak['center']:.2f}s (audio_score="
                                           f"{peak['score']:.2f}, peak={peak['peak_dbfs']:.1f} dBFS); judge whether "
                                           "this is a real payoff, a reaction, or only loud speech"),
                           "confidence": round(min(0.93, 0.72 + (peak["score"] - GUARD_MIN_AUDIO_SCORE) * 0.9), 3),
                           "signals": ["audio_transient"], "required_review": True})
            if len(guards) >= 2:
                break
        return guards
