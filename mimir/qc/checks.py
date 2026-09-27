"""Deterministic checks over the rendered Short and the truths it must honor."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from mimir.captions.layout import MIN_VISIBLE_FRACTION
from mimir.config import Settings
from mimir.edit.context import box_at, samples_of
from mimir.edit.framing import Geometry, Window, stack_bottom_geometry
from mimir.media.audio import cross_correlation_lag, decode_pcm, envelope
from mimir.media.ffmpeg import ffmpeg
from mimir.media.frames import frame_at, frames_at_indices
from mimir.media.probe import MediaInfo, probe
from mimir.qc.pixels import band_ink, ncc, texture
from mimir.render.compositor import Compositor, apply_flash, flash_alpha, top_boxes
from mimir.story.package import MIN_BEAT
from mimir.timeline.schema import COLD_OPEN, STORY, Timeline
from mimir.transcript.truth import truth_signature

CAPTION_BAND_LOW = (0.60, 0.86)
CAPTION_BAND_SEAM = (0.30, 0.50)
HOOK_BAND = (0.10, 0.40)
MIN_PIXEL_MATCH = 0.60
MIN_BASE_MATCH = 0.55
CAPTION_INK_PER_CHAR = 0.0003   # burned text: strongly changed band pixels per visible character
NO_CAPTION_INK = 0.001


@dataclass
class Check:
    name: str
    passed: bool
    severity: str = "fail"                    # fail | warn
    details: dict[str, Any] = field(default_factory=dict)
    repair: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "severity": self.severity, "details": self.details,
                "repair": self.repair}


@dataclass
class QCInputs:
    settings: Settings
    source: Path
    media: MediaInfo
    story: dict[str, Any]
    cold_open: dict[str, Any]
    truth: dict[str, Any]
    captions: dict[str, Any]
    ass_path: Path
    timeline: Timeline
    pacing: dict[str, Any]
    vision: dict[str, Any]
    context: dict[str, Any]
    plan: dict[str, Any]
    effects: dict[str, Any]
    base_video: Path
    final_video: Path
    ledger: list[dict[str, Any]]


# ------------------------------------------------------------------ helpers

def geometry(inputs: QCInputs) -> Geometry:
    cam = inputs.settings.camera
    return Geometry(inputs.media.width, inputs.media.height, inputs.plan["output"][0], inputs.plan["output"][1],
                    cam.max_zoom, cam.max_upscale)


def predicted_frame(inputs: QCInputs, base_frame: np.ndarray, index: int, compositor: Compositor,
                    tops: dict[int, list[float]]) -> np.ndarray:
    image = compositor.frame(base_frame, inputs.plan["layout"][index], inputs.plan["windows"][index], tops.get(index))
    flash = inputs.effects.get("flash", {})
    if flash:
        image = apply_flash(image, flash_alpha(index, flash["frame"], flash["frames"]))
    return image


def active_bands(inputs: QCInputs, index: int) -> list[tuple[float, float]]:
    t = index / inputs.timeline.fps
    bands = [CAPTION_BAND_SEAM if inputs.plan["caption_band"][index] == "seam" else CAPTION_BAND_LOW]
    hook = inputs.captions.get("hook", {}).get("window")
    if hook and hook[0] <= t < hook[1] + 0.1:
        bands.append(HOOK_BAND)
    if inputs.plan["layout"][index] == 1:
        bands.append(CAPTION_BAND_SEAM)
    return bands


def event_at(inputs: QCInputs, t: float) -> dict[str, Any] | None:
    for event in inputs.captions["events"]:
        if event["start"] <= t < event["end"]:
            return event
    return None


# ------------------------------------------------------------------ checks

def check_file(inputs: QCInputs) -> Check:
    info = probe(inputs.final_video, count_frames=True)
    expected = inputs.plan["frame_count"]
    problems = []
    if (info.width, info.height) != tuple(inputs.plan["output"]):
        problems.append(f"size {info.width}x{info.height}")
    if info.video_codec != "h264" or info.audio_codec != "aac":
        problems.append(f"codecs {info.video_codec}/{info.audio_codec}")
    if info.frame_count != expected:
        problems.append(f"frames {info.frame_count} != {expected}")
    if abs(info.fps - inputs.timeline.fps) > 0.01:
        problems.append(f"fps {info.fps}")
    result = ffmpeg(["-v", "error", "-i", str(inputs.final_video), "-f", "null", "-"], timeout=1800)
    decode_errors = result.stderr.decode("utf-8", "replace").strip()
    if decode_errors:
        problems.append("decode errors: " + decode_errors[:200])
    return Check("file_validity", not problems, details={"problems": problems, "frames": info.frame_count,
                                                        "duration": info.duration}, repair={"rerender": True})


def check_story(inputs: QCInputs) -> Check:
    tl = inputs.timeline
    problems = []
    beats = {}
    for beat in inputs.story["beats"]:
        shown = sum(b - a for a, b, _ in tl.map_interval(beat["start"], beat["end"], kinds=(STORY,)))
        beats[beat["role"]] = round(shown, 3)
        minimum = MIN_BEAT.get(beat["role"])
        if minimum is not None and shown + 1.0 / tl.fps < minimum * 0.6:
            problems.append(f"{beat['role']} shows only {shown:.2f}s")
    for rng in inputs.story["protected_ranges"]:
        a, b = max(rng["start"], tl.story_start), min(rng["end"], tl.story_end)
        if b <= a:
            continue
        shown = sum(y - x for x, y, _ in tl.map_interval(a, b, kinds=(STORY,)))
        if shown + 1.5 / tl.fps < b - a:
            problems.append(f"protected range {a:.2f}-{b:.2f} ({rng.get('kind')}) lost {b - a - shown:.2f}s")
    order = [beats.get(r, 0.0) > 0 for r in ("setup", "escalation", "payoff", "reaction")]
    if not all(order):
        problems.append("causal beats missing from the main story")
    return Check("story_completeness", not problems, details={"beats_shown": beats, "problems": problems})


def check_cold_open(inputs: QCInputs) -> Check:
    tl = inputs.timeline
    cfg = inputs.settings.cold_open
    problems = []
    first = tl.segments[0]
    window = inputs.cold_open["window"]
    peak = inputs.cold_open["peak"]
    duration = first.frames / tl.fps
    if first.kind != COLD_OPEN:
        problems.append("first segment is not the cold open")
    if abs(first.source_start - window["start"]) > 1.0 / tl.fps + 1e-6:
        problems.append("cold open segment does not start at the planned window")
    if not (cfg.min_understandable - 0.05 <= duration <= cfg.max_duration + 1.0 / tl.fps):
        problems.append(f"cold open duration {duration:.2f}s outside policy")
    core_a, core_b = peak["start"], min(peak["end"], first.source_start + cfg.max_duration)
    if not (first.source_start <= peak["start"] + 0.05 and first.source_end >= core_b - 0.05):
        problems.append(f"peak {peak['start']:.2f}-{peak['end']:.2f} not inside cold open "
                        f"{first.source_start:.2f}-{first.source_end:.2f}")
    if not (tl.story_start - 0.01 <= peak["start"] and peak["end"] <= tl.story_end + 0.01):
        problems.append("peak is not part of the story (the main story must show how it happened)")
    shown_again = sum(b - a for a, b, _ in tl.map_interval(core_a, max(core_a + 0.01, core_b), kinds=(STORY,)))
    if shown_again <= 0:
        problems.append("the peak is never reached again in the main story")
    return Check("mandatory_cold_open", not problems, details={"duration": round(duration, 3), "problems": problems,
                                                              "peak": [peak["start"], peak["end"]]})


def check_main_restart(inputs: QCInputs) -> Check:
    tl = inputs.timeline
    story_segments = tl.kind_segments(STORY)
    first = story_segments[0]
    leading = [c for c in inputs.pacing.get("suggestions", []) if c["kind"] == "leading_dead_air"
               and c["action"] == "cut"]
    expected = leading[0]["end"] if leading else tl.story_start
    ok = abs(first.source_start - expected) <= 0.15 + 1.0 / tl.fps
    ok = ok and abs(tl.story_start - float(inputs.story["start"])) <= 0.5
    return Check("main_story_restart", ok, details={"main_start_source": first.source_start,
                                                     "expected": round(expected, 3), "story_start": tl.story_start,
                                                     "main_start_frame": first.start_frame})


def check_captions(inputs: QCInputs) -> Check:
    problems = []
    truth = inputs.truth
    if truth_signature(truth["words"]) != truth["signature"]:
        problems.append("caption truth signature mismatch")
    if inputs.captions["truth_signature"] != truth["signature"]:
        problems.append("captions were built from a different truth")
    by_id = {w["id"]: w for w in truth["words"]}
    tl = inputs.timeline
    frame = 1.0 / tl.fps
    tolerance = inputs.settings.qc.caption_timing_tolerance_frames * frame + 1e-3
    expected_keys = []
    for word in truth["words"]:
        for a, b, index in tl.map_interval(float(word["start"]), float(word["end"]), kinds=(COLD_OPEN, STORY)):
            if b - a >= MIN_VISIBLE_FRACTION * (float(word["end"]) - float(word["start"])) - 1e-6:
                expected_keys.append((f"{word['id']}@{tl.segments[index].kind}", a))
    shown = {w["key"]: w for w in inputs.captions["words"]}
    missing = [k for k, _ in expected_keys if k not in shown]
    if missing:
        problems.append(f"{len(missing)} truth words not captioned: {missing[:5]}")
    for key, out_start in expected_keys:
        row = shown.get(key)
        if row is None:
            continue
        truth_word = by_id[row["word_id"]]
        if row["text"] != truth_word["text"]:
            problems.append(f"{key}: caption text {row['text']!r} != truth {truth_word['text']!r}")
        if abs(row["start"] - out_start) > tolerance:
            problems.append(f"{key}: caption at {row['start']:.3f}s, truth maps to {out_start:.3f}s")
    events_by_word = {e["word_key"]: e for e in inputs.captions["events"]}
    for key, out_start in expected_keys:
        event = events_by_word.get(key)
        if event is None:
            continue
        if abs(event["start"] - out_start) > tolerance:
            problems.append(f"{key}: event starts {event['start']:.3f}s vs word {out_start:.3f}s")
    ass_text = inputs.ass_path.read_text(encoding="utf-8")
    dialogues = [line for line in ass_text.splitlines() if line.startswith("Dialogue:")]
    plain = [re.sub(r"\{[^}]*\}", "", line.split(",", 9)[9]) for line in dialogues]
    visible_words = set()
    for line in plain:
        visible_words |= set(line.replace("\\N", " ").split())
    for row in inputs.captions["words"]:
        text = row["text"].upper() if inputs.settings.captions.uppercase else row["text"]
        missing_tokens = [t for t in text.replace("{", "(").replace("}", ")").split() if t not in visible_words]
        if missing_tokens:
            problems.append(f"burned ASS lacks word {row['text']!r}")
            break
    lanes: dict[str, list[tuple[float, float]]] = {}
    for event in inputs.captions["events"]:
        lanes.setdefault(event["lane"], []).append((event["start"], event["end"]))
    for lane, spans in lanes.items():
        spans.sort()
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            if b0 < a1 - 1e-3:
                problems.append(f"overlapping caption events on lane {lane} at {b0:.2f}s")
                break
    return Check("caption_truth_and_timing", not problems, details={"problems": problems[:12],
                                                                   "words": len(expected_keys),
                                                                   "events": len(inputs.captions["events"])})


def check_speakers(inputs: QCInputs) -> Check:
    problems = []
    truth = {w["id"]: w for w in inputs.truth["words"]}
    names = inputs.truth.get("confirmed_names", {})
    for row in inputs.captions["words"]:
        word = truth[row["word_id"]]
        if (word.get("speaker") or "") != row["speaker"]:
            problems.append(f"{row['key']}: speaker {row['speaker']} != truth {word.get('speaker')}")
        if row["label"] and names.get(row["speaker"]) != row["label"]:
            problems.append(f"{row['key']}: label {row['label']!r} is not a confirmed name")
    windows = inputs.captions.get("overlap_windows", [])
    for row in inputs.captions["words"]:
        if row["lane"] == "secondary" and not any(a <= row["start"] <= b for a, b in windows):
            problems.append(f"{row['key']}: secondary lane outside measured overlap")
    by_key = {w["key"]: w for w in inputs.captions["words"]}
    for group in inputs.captions["groups"]:
        labels = {by_key[k]["label"] for k in group["words"]}
        if len(labels) > 1:
            problems.append(f"group {group['index']} mixes visible speaker labels {labels}")
    return Check("speaker_ownership", not problems, details={"problems": problems[:12]})


SPEAKER_EMPHASIS = {"SPEAKER_MEDIUM", "SPEAKER_PUNCH", "REACTION"}


def check_speaker_resolution(inputs: QCInputs) -> Check:
    """Confirmed speaker evidence vs an unresolved diarization, and no fabricated ownership."""
    resolution = inputs.truth.get("speaker_resolution") or {"status": "confirmed"}
    status = resolution.get("status", "confirmed")
    details = {**resolution, "mode": inputs.truth.get("speaker_mode", ""), "violations": []}
    if status == "confirmed":
        return Check("speaker_resolution", True, details=details)
    violations = details["violations"]
    owned = [w["id"] for w in inputs.truth["words"] if w.get("speaker")]
    if owned:
        violations.append(f"{len(owned)} words carry a speaker although diarization is unresolved")
    if inputs.truth.get("confirmed_names") or any(r.get("label") for r in inputs.captions["words"]):
        violations.append("a speaker name is shown although no speaker was resolved")
    emphasis = sorted({s["intent"] for s in inputs.plan["spans"]} & SPEAKER_EMPHASIS)
    if emphasis:
        violations.append(f"speaker-focused framing {emphasis} without resolved speakers")
    if any(r.get("lane") == "secondary" for r in inputs.captions["words"]):
        violations.append("an interrupter caption lane without resolved speakers")
    # unresolved but handled conservatively: visible in the report, not a failure
    return Check("speaker_resolution", False, "fail" if violations else "warn", details=details)


def _window_for(inputs: QCInputs, index: int, geo: Geometry) -> tuple[Window, Geometry]:
    cx, cy, h = inputs.plan["windows"][index]
    if inputs.plan["layout"][index] == 1:
        return Window(cx, cy, h), stack_bottom_geometry(geo)
    return Window(cx, cy, h), geo


def check_required_content(inputs: QCInputs) -> Check:
    geo = geometry(inputs)
    tops = top_boxes(inputs.plan)
    failures = []
    spans_to_widen = []
    context_spans = {s["id"]: s for s in inputs.context["spans"]}
    plan_spans = {s["id"]: s for s in inputs.plan["spans"]}
    faces = {f["id"]: samples_of(f) for f in inputs.vision["faces"] if not f["static_pattern"]}
    for span_id, span in context_spans.items():
        f0, f1 = span["frames"]
        target = plan_spans[span_id]["target_id"]
        boxes = [(r["id"], r["box"]) for r in span["required"]]
        settle = f0 + plan_spans[span_id]["transition_frames"] if not plan_spans[span_id]["hard_start"] else f0
        for index in sorted({f0, (f0 + f1) // 2, f1 - 1, min(f1 - 1, settle)}):
            window, g = _window_for(inputs, index, geo)
            top = tops.get(index)
            items = list(boxes)
            if target in faces and index >= settle:
                box = box_at(faces[target], inputs.timeline.source_time(index))
                if box is not None:
                    items.append((target, box))
            for item_id, box in items:
                inside_top = top is not None and top[0] - 0.02 <= box[0] and top[2] + 0.02 >= box[2] \
                    and top[1] - 0.02 <= box[1] and top[3] + 0.02 >= box[3]
                if not (window.contains(g, box, tol=0.02) or inside_top):
                    failures.append({"span": span_id, "frame": index, "item": item_id})
                    spans_to_widen.append(span_id)
    return Check("required_content_visible", not failures, details={"failures": failures[:12]},
                 repair={"widen_spans": sorted(set(spans_to_widen))})


def check_crops(inputs: QCInputs) -> Check:
    geo = geometry(inputs)
    problems = []
    for index, (layout, (cx, cy, h)) in enumerate(zip(inputs.plan["layout"], inputs.plan["windows"])):
        g = stack_bottom_geometry(geo) if layout == 1 else geo
        window = Window(cx, cy, h)
        if not (g.h_min - 1e-4 <= h <= g.h_full + 1e-4):
            problems.append(f"frame {index}: window height {h:.4f} outside [{g.h_min:.4f}, {g.h_full:.4f}]")
        elif h <= g.h_inside + 1e-6 and not window.inside(g, tol=2e-3):
            problems.append(f"frame {index}: crop leaves the source")
        if problems and len(problems) > 5:
            break
    for run in inputs.plan.get("stack_tops", []):
        x0, y0, x1, y1 = run["box"]
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            problems.append(f"stack top box invalid {run['box']}")
    return Check("crop_validity", not problems, details={"problems": problems[:8]}, repair={"rerender": True})


def check_stability(inputs: QCInputs) -> Check:
    phases = inputs.plan["phases"]
    windows = inputs.plan["windows"]
    fps = inputs.timeline.fps
    limit = inputs.settings.qc.max_center_step
    worst = 0.0
    reversals = 0
    follow_frames = 0
    previous_velocity = None
    for index in range(1, len(windows)):
        if phases[index] in ("cut", "transition"):
            previous_velocity = None
            continue
        dx = windows[index][0] - windows[index - 1][0]
        dy = windows[index][1] - windows[index - 1][1]
        worst = max(worst, math.hypot(dx, dy))
        if phases[index] == "follow":
            follow_frames += 1
            if abs(dx) > 1e-4:
                if previous_velocity is not None and np.sign(dx) != np.sign(previous_velocity):
                    reversals += 1
                previous_velocity = dx
    reversal_rate = reversals / max(1.0, follow_frames / fps)
    ok = worst <= limit + 1e-6 and reversal_rate <= 1.5
    return Check("camera_stability", ok, details={"max_step": round(worst, 5), "limit": limit,
                                                  "reversals_per_second": round(reversal_rate, 3)},
                 repair={"conservative_camera": True})


def check_broken_faces(inputs: QCInputs) -> Check:
    geo = geometry(inputs)
    faces = {f["id"]: samples_of(f) for f in inputs.vision["faces"] if not f["static_pattern"]}
    required_ids = {r["id"] for s in inputs.context["spans"] for r in s["required"]}
    targets = {s["target_id"] for s in inputs.plan["spans"] if s["target_id"]}
    step = max(1, inputs.timeline.fps // 2)
    broken, warnings = [], []
    for index in range(0, inputs.plan["frame_count"], step):
        if inputs.plan["layout"][index] == 1 or inputs.plan["phases"][index] == "transition":
            continue  # planned moves are brief; settled framing must never cut a face
        window, g = _window_for(inputs, index, geo)
        t = inputs.timeline.source_time(index)
        for face_id, samples in faces.items():
            box = box_at(samples, t)
            if box is None or box[3] - box[1] < 0.06:
                continue
            ox0, oy0, ox1, oy1 = window.to_output(g, box)
            area = max(1e-9, (ox1 - ox0) * (oy1 - oy0))
            vx = max(0.0, min(1.0, ox1) - max(0.0, ox0))
            vy = max(0.0, min(1.0, oy1) - max(0.0, oy0))
            visible = vx * vy / area
            if 0.15 < visible < 0.85:
                row = {"frame": index, "face": face_id, "visible": round(visible, 2)}
                (broken if face_id in required_ids or face_id in targets else warnings).append(row)
    span_ids = sorted({s["id"] for s in inputs.context["spans"] for row in broken
                       if s["frames"][0] <= row["frame"] < s["frames"][1]})
    return Check("no_broken_faces", not broken, details={"broken": broken[:10], "partial_non_required": warnings[:10]},
                 repair={"widen_spans": span_ids})


def check_pixels(inputs: QCInputs, samples: int) -> Check:
    """The render plan (and the cold open / main restart) reached the rendered pixels."""
    n = inputs.plan["frame_count"]
    tl = inputs.timeline
    main = tl.main_start_frame
    picks = {1, max(0, main - 3), min(n - 1, main + 4), n - 2}
    picks |= {int(k) for k in np.linspace(2, n - 3, max(2, samples))}
    picks = sorted(i for i in picks if 0 <= i < n)
    flash = inputs.effects.get("flash", {})
    picks = [i for i in picks if not flash or abs(i - flash["frame"]) > flash["frames"]]
    base = frames_at_indices(inputs.base_video, picks, fps=tl.fps, src_width=inputs.media.width,
                             src_height=inputs.media.height)
    width, height = inputs.plan["output"]
    rendered = frames_at_indices(inputs.final_video, picks, fps=tl.fps, src_width=width, src_height=height)
    compositor = Compositor(inputs.media.width, inputs.media.height, width, height,
                            inputs.plan["geometry"]["stack_split"])
    tops = top_boxes(inputs.plan)
    rows, failures = [], []
    for index in picks:
        if index not in base or index not in rendered:
            failures.append({"frame": index, "problem": "frame not decodable"})
            continue
        predicted = predicted_frame(inputs, base[index], index, compositor, tops)
        bands = active_bands(inputs, index)
        if texture(predicted) < 1.0:
            rows.append({"frame": index, "skipped": "flat frame"})
            continue
        score = ncc(predicted, rendered[index], bands)
        source_t = tl.source_time(index)
        source = frame_at(inputs.source, source_t + 0.5 / tl.fps, src_width=inputs.media.width,
                          src_height=inputs.media.height, width=640)
        base_small = frame_at(inputs.base_video, index / tl.fps + 0.5 / tl.fps, src_width=inputs.media.width,
                              src_height=inputs.media.height, width=640)
        base_score = ncc(source, base_small)
        rows.append({"frame": index, "plan_match": round(score, 3), "timeline_match": round(base_score, 3)})
        if score < MIN_PIXEL_MATCH:
            failures.append({"frame": index, "problem": f"rendered frame does not match the plan ({score:.2f})"})
        if base_score < MIN_BASE_MATCH and texture(source) > 1.0:
            failures.append({"frame": index, "problem": f"base frame does not show source t={source_t:.2f} "
                                                        f"({base_score:.2f})"})
    return Check("plan_reached_pixels", not failures, details={"samples": rows, "failures": failures[:8]},
                 repair={"rerender": True})


def check_caption_pixels(inputs: QCInputs) -> Check:
    tl = inputs.timeline
    n = inputs.plan["frame_count"]
    with_caption, without = [], []
    for event in inputs.captions["events"][::max(1, len(inputs.captions["events"]) // 6)]:
        index = int(math.floor((event["start"] + event["end"]) / 2 * tl.fps))
        if 0 <= index < n:
            with_caption.append(index)
    hook = inputs.captions.get("hook", {}).get("window")
    for index in range(0, n, max(1, tl.fps // 3)):
        t = index / tl.fps
        if event_at(inputs, t - 0.2) is None and event_at(inputs, t) is None and event_at(inputs, t + 0.2) is None \
                and not (hook and t < hook[1] + 0.2):
            without.append(index)
    without = without[:4]
    picks = sorted(set(with_caption[:6]) | set(without))
    flash = inputs.effects.get("flash", {})
    picks = [i for i in picks if not flash or abs(i - flash["frame"]) > flash["frames"]]
    width, height = inputs.plan["output"]
    base = frames_at_indices(inputs.base_video, picks, fps=tl.fps, src_width=inputs.media.width,
                             src_height=inputs.media.height)
    rendered = frames_at_indices(inputs.final_video, picks, fps=tl.fps, src_width=width, src_height=height)
    compositor = Compositor(inputs.media.width, inputs.media.height, width, height,
                            inputs.plan["geometry"]["stack_split"])
    tops = top_boxes(inputs.plan)
    words = {w["key"]: w["text"] for w in inputs.captions["words"]}
    groups = {g["index"]: g["words"] for g in inputs.captions["groups"]}
    rows, problems = [], []
    for index in picks:
        predicted = predicted_frame(inputs, base[index], index, compositor, tops)
        band = CAPTION_BAND_SEAM if inputs.plan["caption_band"][index] == "seam" else CAPTION_BAND_LOW
        ink = band_ink(rendered[index], predicted, band)
        expects = index in with_caption
        chars = 0
        event = event_at(inputs, index / tl.fps + 0.5 / tl.fps)
        if event is not None:
            keys = groups.get(event["group"], [])
            shown = keys[:keys.index(event["word_key"]) + 1] if event["word_key"] in keys else keys
            chars = sum(len(words.get(k, "")) for k in shown)
        rows.append({"frame": index, "caption_expected": expects, "ink": round(ink, 4), "visible_chars": chars})
        if expects and ink < CAPTION_INK_PER_CHAR * max(1, chars):
            problems.append(f"frame {index}: caption event not visible in pixels (ink {ink:.4f}, {chars} chars)")
        if not expects and ink > NO_CAPTION_INK:
            problems.append(f"frame {index}: unexpected ink in the caption band (ink {ink:.4f})")
    return Check("captions_reached_pixels", not problems, details={"samples": rows, "problems": problems},
                 repair={"rerender": True})


def check_av_sync(inputs: QCInputs) -> Check:
    tl = inputs.timeline
    tolerance = inputs.settings.qc.av_sync_tolerance
    rate = 8000
    rendered = decode_pcm(inputs.final_video, rate=rate)
    reference = decode_pcm(inputs.base_video, rate=rate)
    rows, problems = [], []
    hop = 0.01
    for segment in tl.segments:
        a, b = segment.start_frame / tl.fps, segment.end_frame / tl.fps
        if b - a < 1.0:
            continue
        ra = rendered[int(a * rate):int(b * rate)]
        rb = reference[int(a * rate):int(b * rate)]
        if len(ra) < rate // 2 or len(rb) < rate // 2:
            continue
        ea = envelope(ra, rate, window=0.02, hop=hop).dbfs
        eb = envelope(rb, rate, window=0.02, hop=hop).dbfs
        if eb.std() < 1.5:
            continue
        lag, corr = cross_correlation_lag(eb, ea, max_lag=12)
        rows.append({"segment": segment.index, "lag_ms": round(lag * hop * 1000, 1), "correlation": round(corr, 3)})
        if abs(lag * hop) > tolerance or corr < 0.5:
            problems.append(f"segment {segment.index}: lag {lag * hop * 1000:.0f} ms, corr {corr:.2f}")
    info = probe(inputs.final_video)
    duration_error = abs(info.duration - tl.duration)
    if duration_error > 0.05:
        problems.append(f"container duration off by {duration_error:.3f}s")
    return Check("av_sync", not problems, details={"segments": rows, "problems": problems},
                 repair={"rerender": True})


def check_cold_open_audio(inputs: QCInputs) -> Check:
    """The cold open's sound is the peak's sound (not any other part of the source)."""
    tl = inputs.timeline
    first = tl.segments[0]
    length = first.frames / tl.fps - 0.25
    if length < 0.6:
        return Check("cold_open_contains_peak_audio", True, "warn", details={"skipped": "too short"})
    rate = 8000
    rendered = decode_pcm(inputs.final_video, start=0.0, duration=length, rate=rate)
    source = decode_pcm(inputs.source, start=first.source_start, duration=length, rate=rate)
    ea = envelope(rendered, rate, window=0.03, hop=0.01).dbfs
    eb = envelope(source, rate, window=0.03, hop=0.01).dbfs
    if eb.std() < 1.5:
        return Check("cold_open_contains_peak_audio", True, "warn", details={"skipped": "flat source audio"})
    lag, corr = cross_correlation_lag(eb, ea, max_lag=5)
    return Check("cold_open_contains_peak_audio", corr >= 0.6 and abs(lag) <= 4,
                 details={"correlation": round(corr, 3), "lag_ms": lag * 10})


def check_ledger(inputs: QCInputs) -> Check:
    degraded = [n for n in inputs.ledger if n.get("level") == "degraded"]
    return Check("no_silent_fallback", not degraded, details={"degraded": degraded[:10],
                                                              "warnings": [n for n in inputs.ledger
                                                                           if n.get("level") == "warning"][:10]})


CHECKS: Sequence[Callable[[QCInputs], Check]] = (
    check_file, check_story, check_cold_open, check_main_restart, check_captions, check_speakers,
    check_speaker_resolution, check_required_content, check_crops, check_stability, check_broken_faces, check_caption_pixels,
    check_av_sync, check_cold_open_audio, check_ledger,
)


def run_checks(inputs: QCInputs) -> list[Check]:
    results = [check(inputs) for check in CHECKS]
    results.insert(11, check_pixels(inputs, inputs.settings.qc.pixel_samples))
    return results
