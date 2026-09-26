"""Stage ``edit_context``: EditContextBuilder.

Splits the output timeline into spans (at segment boundaries, shot cuts, story
beats and speaker turns) and attaches, per span, only evidence MIMIR already
has: story role, who is talking, which subjects are visible (with speaker-face
link confidence), action regions, important observed elements, the content
layout, what MUST stay visible, and which intents the evidence allows.
"""
from __future__ import annotations

import statistics
from typing import Any, Sequence

from mimir.config import Settings, section
from mimir.core.stage import StageContext, StageOutput
from mimir.edit.intents import Intent
from mimir.timeline.schema import COLD_OPEN, STORY, Timeline
from mimir.vision.tracking import Sample

TURN_MIN_SECONDS = 1.2
DOMINANT_SHARE = 0.6
VISIBLE_COVERAGE = 0.5
STRONG_LINK = 0.7
WEAK_LINK = 0.5
MULTI_SUBJECT_ROLES = {"payoff", "reaction", "cold_open"}


def samples_of(face: dict[str, Any]) -> list[Sample]:
    return [Sample.from_list(row) for row in face["samples"]]


def box_at(samples: Sequence[Sample], t: float, tolerance: float = 0.35) -> list[float] | None:
    """Interpolated face box at source time ``t`` (None when not tracked near ``t``)."""
    if not samples:
        return None
    before = [s for s in samples if s.t <= t]
    after = [s for s in samples if s.t >= t]
    a = before[-1] if before else None
    b = after[0] if after else None
    if a is None and b is None:
        return None
    if a is None or b is None or b.t - a.t > 2 * tolerance:
        nearest = min((s for s in (a, b) if s is not None), key=lambda s: abs(s.t - t))
        if abs(nearest.t - t) > tolerance:
            return None
        return [nearest.cx - nearest.w / 2, nearest.cy - nearest.h / 2, nearest.cx + nearest.w / 2,
                nearest.cy + nearest.h / 2]
    u = 0.0 if b.t == a.t else (t - a.t) / (b.t - a.t)
    cx, cy = a.cx + (b.cx - a.cx) * u, a.cy + (b.cy - a.cy) * u
    w, h = a.w + (b.w - a.w) * u, a.h + (b.h - a.h) * u
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def speaker_runs(words: Sequence[dict[str, Any]]) -> list[tuple[float, float, str]]:
    runs: list[list[Any]] = []
    for word in words:
        speaker = word.get("speaker") or ""
        if runs and runs[-1][2] == speaker and float(word["start"]) - runs[-1][1] < 0.8:
            runs[-1][1] = float(word["end"])
        else:
            runs.append([float(word["start"]), float(word["end"]), speaker])
    return [(a, b, s) for a, b, s in runs]


def build_spans(timeline: Timeline, story: dict[str, Any], vision: dict[str, Any], words: Sequence[dict[str, Any]],
                min_shot_seconds: float) -> list[dict[str, Any]]:
    fps = timeline.fps
    hard = {0, timeline.frame_count}
    hard |= {s.start_frame for s in timeline.segments}
    for cut in vision.get("shot_cuts", []):
        for out_t in timeline.map_time(cut, kinds=(STORY, COLD_OPEN)):
            hard.add(int(round(out_t * fps)))
    soft: set[int] = set()
    for beat in story["beats"]:
        for out_t in timeline.map_time(beat["start"], kinds=(STORY,)):
            soft.add(int(round(out_t * fps)))
    for a, b, speaker in speaker_runs(words):
        if speaker and b - a >= TURN_MIN_SECONDS:
            for out_t in timeline.map_time(a, kinds=(STORY,)):
                soft.add(int(round(out_t * fps)))
    minimum = int(round(min_shot_seconds * fps))
    boundaries = sorted(hard)
    for frame in sorted(soft):
        if all(abs(frame - b) >= minimum for b in boundaries):
            boundaries.append(frame)
            boundaries.sort()
    spans = []
    for f0, f1 in zip(boundaries, boundaries[1:]):
        if f1 > f0:
            spans.append((f0, f1))
    return [{"frames": [f0, f1]} for f0, f1 in spans]


class EditContextStage:
    name = "edit_context"
    version = 1
    deps = ("story", "caption_truth", "vision", "timeline")

    def params(self, settings: Settings) -> Any:
        return {"min_shot_seconds": settings.camera.min_shot_seconds}

    def run(self, ctx: StageContext) -> StageOutput:
        story = ctx.dep("story").json("story")
        truth = ctx.dep("caption_truth").json("caption_truth")
        vision = ctx.dep("vision").json("vision")
        timeline = Timeline.from_dict(ctx.dep("timeline").json("timeline"))
        words = truth["words"]
        spans = build_spans(timeline, story, vision, words, ctx.settings.camera.min_shot_seconds)
        faces = [f for f in vision["faces"] if not f["static_pattern"]]
        face_samples = {f["id"]: samples_of(f) for f in faces}
        cuts = vision.get("shot_cuts", [])
        layout = vision["layout"]
        mode = truth.get("speaker_mode", "single")
        observations = vision.get("observations", [])
        rows = []
        for index, span in enumerate(spans):
            f0, f1 = span["frames"]
            segment = timeline.segment_at(f0)
            src_a = timeline.source_time(f0)
            src_b = timeline.source_time(f1 - 1) + 1.0 / timeline.fps
            mid = (src_a + src_b) / 2
            role = "cold_open" if segment.kind == COLD_OPEN else next(
                (b["role"] for b in story["beats"] if b["start"] <= mid < b["end"]), "aftermath")
            talk: dict[str, float] = {}
            span_words = [w for w in words if float(w["end"]) > src_a and float(w["start"]) < src_b]
            for word in span_words:
                if word.get("speaker"):
                    overlap = min(src_b, float(word["end"])) - max(src_a, float(word["start"]))
                    talk[word["speaker"]] = talk.get(word["speaker"], 0.0) + max(0.0, overlap)
            total_talk = sum(talk.values())
            dominant = max(talk, key=talk.get) if talk and talk[max(talk, key=talk.get)] >= DOMINANT_SHARE * total_talk else ""
            visible = []
            for face in faces:
                samples = [s for s in face_samples[face["id"]] if src_a - 0.05 <= s.t <= src_b + 0.05]
                dt = 1.0 / vision["analysis_fps"]
                if len(samples) * dt < VISIBLE_COVERAGE * (src_b - src_a):
                    continue
                boxes = [(s.cx - s.w / 2, s.cy - s.h / 2, s.cx + s.w / 2, s.cy + s.h / 2) for s in samples]
                median_box = [round(statistics.median(b[i] for b in boxes), 4) for i in range(4)]
                visible.append({"id": face["id"], "speaker": face.get("speaker"),
                                "link": face.get("speaker_confidence", 0.0), "box": median_box,
                                "size": round(median_box[3] - median_box[1], 4),
                                "position": "left" if median_box[0] + median_box[2] < 0.8 else (
                                    "right" if median_box[0] + median_box[2] > 1.2 else "center")})
            actions = [{"id": a["id"], "box": a["box"], "intensity": a["relative_intensity"]}
                       for a in vision.get("action_regions", []) if a["t1"] > src_a and a["t0"] < src_b]
            elements = [{"kind": e["kind"], "description": e["description"], "box": e["box"],
                         "importance": e["importance"]}
                        for o in observations if src_a - 0.5 <= o["t"] <= src_b + 0.5
                        for e in o["elements"] if e["importance"] in ("high", "medium")]
            # a face is "the speaker" only when mouth activity linked it to the dominant voice;
            # without that evidence the camera stays conservative (never guesses a lone face)
            speaking_face = next((v for v in visible if v["speaker"] and v["speaker"] == dominant), None)
            overlay_face = layout["class"] == "facecam_gameplay"
            required = []
            if role in MULTI_SUBJECT_ROLES and len(visible) >= 2:
                required += [{"kind": "subject", "id": v["id"], "box": v["box"],
                              "reason": f"{role}: every visible participant carries the moment"} for v in visible]
            if role in ("escalation", "payoff", "reaction", "cold_open"):
                required += [{"kind": "action", "id": a["id"], "box": a["box"],
                              "reason": f"{role}: the action must stay visible"} for a in actions
                             if a["intensity"] >= 1.0]
            required += [{"kind": "element", "id": f"el_{index}_{k}", "box": e["box"],
                          "reason": f"observer: {e['description'][:80]}"}
                         for k, e in enumerate(el for el in elements if el["importance"] == "high"
                                               and el["kind"] in ("object", "action", "gameplay", "screen", "text"))]
            allowed = [Intent.HOLD.value, Intent.WIDE_CONTEXT.value]
            # facecam overlays stay visible in the stacked panel; face-only framings would drop the
            # gameplay and upscale a small overlay, so they are not offered there
            if len(visible) >= 2 and not overlay_face:
                allowed.append(Intent.TWO_SHOT.value)
            if speaking_face and speaking_face["link"] >= WEAK_LINK and not overlay_face:
                allowed.append(Intent.SPEAKER_MEDIUM.value)
            if speaking_face and speaking_face["link"] >= STRONG_LINK and not overlay_face:
                allowed.append(Intent.SPEAKER_PUNCH.value)
            if visible and role in ("payoff", "reaction", "cold_open", "escalation") and not overlay_face:
                allowed.append(Intent.REACTION.value)
            if actions or any(e["kind"] in ("object", "action") for e in elements):
                allowed.append(Intent.ACTION_REGION.value)
            if layout["class"] in ("facecam_gameplay", "gameplay"):
                allowed.append(Intent.GAMEPLAY_PRIORITY.value)
            if layout["class"] == "screen_content":
                allowed.append(Intent.SCREEN_PRIORITY.value)
            text = " ".join(w["text"] for w in span_words)
            rows.append({
                "id": f"s{index:02d}", "frames": [f0, f1], "seconds": [round(f0 / timeline.fps, 3),
                                                                       round(f1 / timeline.fps, 3)],
                "source": [round(src_a, 3), round(src_b, 3)], "segment": segment.kind, "role": role,
                "shot": sum(1 for c in cuts if c <= src_a), "talk": {k: round(v, 2) for k, v in talk.items()},
                "dominant_speaker": dominant, "visible": visible, "actions": actions, "elements": elements[:6],
                "required": required, "allowed": allowed, "text": text[:240],
            })
        return StageOutput(data={"edit_context": {
            "fps": timeline.fps, "frame_count": timeline.frame_count, "layout": layout, "speaker_mode": mode,
            "story": {"title": story["title"], "reason": story["reason"], "emotion": story["emotion"]},
            "spans": rows,
        }})
