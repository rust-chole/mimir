"""Optional bounded multimodal final reviewer: SOURCE story evidence vs the RENDERED Short.

One call, at most MAX_PAIRS source/render frame pairs chosen at the story's own moments
(cold-open peak, main restart, setup, escalation, payoff, reaction). Every pair carries
the evidence the render must honour at that moment: beat, planned intent, REQUIRED
content (drawn as numbered boxes on the source frame), who is speaking and which face
is linked to them, the burned caption text and the caption truth. The reviewer reports
only concrete failures in fixed categories; visual failures map to the deterministic
repair (widen that span); story and caption failures are not auto-repairable and stop
the gate. The pipeline applies at most one repair round.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from mimir.models.provider import ImageInput
from mimir.timeline.schema import COLD_OPEN, STORY, Timeline

MAX_PAIRS = 8
ISSUE_TYPES = ["missing_story_beat", "weak_or_wrong_cold_open", "causal_story_damage", "important_visual_cropped",
               "speaker_camera_mismatch", "caption_contradiction", "plan_not_reached_pixels"]
VISUAL_REPAIRABLE = {"important_visual_cropped", "speaker_camera_mismatch", "plan_not_reached_pixels"}

INSTRUCTIONS = """
You are the final reviewer of a vertical Short cut from a longer source video. Structure: COLD OPEN (the story's
peak) -> hard restart -> SETUP -> ESCALATION -> PAYOFF -> REACTION.

For each frame id you get the SOURCE frame (landscape, REQUIRED content drawn as numbered green boxes R1, R2, ...)
and the RENDERED frame of the Short at the same moment, plus the evidence the render must honour at that moment.
Compare them and report ONLY concrete, checkable failures:
- missing_story_beat: a beat's moment is absent or shows something unrelated to that beat.
- weak_or_wrong_cold_open: the cold-open frame does not show the peak/payoff/reaction named in the evidence.
- causal_story_damage: the order/content makes the cause of the payoff impossible to follow (e.g. the main story
  does not restart at the setup).
- important_visual_cropped: a REQUIRED box (person, action, object, gameplay, screen text) is cut off or missing
  in the rendered frame, or a face is cut in half.
- speaker_camera_mismatch: the rendered frame is framed on a person other than the named speaker while the plan
  says SPEAKER_*, or on nobody while the plan targets a person.
- caption_contradiction: the burned caption text differs from the caption truth, belongs to the wrong speaker,
  or covers REQUIRED content.
- plan_not_reached_pixels: the rendered framing clearly differs from the planned intent (e.g. plan says
  GAMEPLAY_PRIORITY but the gameplay is not visible).
Never judge taste, style or whether the video "looks good". Use only the supplied frame ids. severity=high only
when the failure would make a viewer miss or misread the story; verdict=fail if and only if any issue is high.
""".strip()


def schema_for(frame_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verdict": {"type": "string", "enum": ["pass", "fail"]},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "type": {"type": "string", "enum": ISSUE_TYPES},
                        "frame_id": {"type": "string", "enum": list(frame_ids)},
                        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                        "expected": {"type": "string"},
                        "observed": {"type": "string"},
                    },
                    "required": ["type", "frame_id", "severity", "expected", "observed"],
                },
            },
        },
        "required": ["verdict", "issues"],
    }


@dataclass
class Moment:
    frame: int
    label: str          # cold_open_peak | main_restart | setup | escalation | payoff | reaction


def pick_moments(timeline: Timeline, story: dict[str, Any], cold_open: dict[str, Any], frame_count: int
                 ) -> list[Moment]:
    """The story's own moments in output frames (bounded, unique, inside the render)."""
    fps = timeline.fps
    moments: list[Moment] = []

    def add(label: str, source_t: float, kinds: tuple[str, ...]) -> None:
        for out_t in timeline.map_time(source_t, kinds=kinds):
            frame = min(frame_count - 1, max(0, int(round(out_t * fps))))
            if all(abs(frame - m.frame) > fps // 4 for m in moments):
                moments.append(Moment(frame, label))
            return

    add("cold_open_peak", float(cold_open["peak"]["center"]), (COLD_OPEN,))
    moments.append(Moment(min(frame_count - 1, timeline.main_start_frame + 3), "main_restart"))
    for beat in story["beats"]:
        if beat["role"] == "payoff":
            center = float(cold_open["peak"]["center"])
            t = center if beat["start"] <= center <= beat["end"] else (beat["start"] + beat["end"]) / 2
        else:
            t = (float(beat["start"]) + float(beat["end"])) / 2
        add(beat["role"], t, (STORY,))
    return sorted(moments[:MAX_PAIRS], key=lambda m: m.frame)


def _jpeg(image: np.ndarray, width: int) -> bytes:
    h, w = image.shape[:2]
    image = cv2.resize(image, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()


def draw_required(source: np.ndarray, required: Sequence[dict[str, Any]]) -> np.ndarray:
    image = source.copy()
    h, w = image.shape[:2]
    for k, row in enumerate(required, 1):
        x0, y0, x1, y1 = row["box"]
        p0, p1 = (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h))
        cv2.rectangle(image, p0, p1, (40, 220, 40), 2)
        cv2.putText(image, f"R{k}", (p0[0] + 3, max(12, p0[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (40, 220, 40), 1, cv2.LINE_AA)
    return image


def moment_evidence(moment: Moment, timeline: Timeline, story: dict[str, Any], context: dict[str, Any],
                    plan: dict[str, Any], captions: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    fps = timeline.fps
    out_t = moment.frame / fps
    src_t = timeline.source_time(moment.frame)
    span = next((s for s in context["spans"] if s["frames"][0] <= moment.frame < s["frames"][1]), None)
    planned = next((s for s in plan["spans"] if s["frames"][0] <= moment.frame < s["frames"][1]), None)
    beat = next((b for b in story["beats"] if b["start"] <= src_t < b["end"]), None)
    words_by_key = {w["key"]: w for w in captions["words"]}
    shown = [e for e in captions["events"] if e["start"] - 0.35 <= out_t <= e["end"] + 0.35]
    burned = {}
    for event in shown:
        word = words_by_key.get(event["word_key"])
        if word:
            burned.setdefault(event["lane"], [])
            if word["text"] not in burned[event["lane"]]:
                burned[event["lane"]].append(f"{word['text']}({word['speaker']})")
    spoken = [f"{w['text']}({w.get('speaker') or '?'})" for w in truth["words"]
              if src_t - 1.0 <= float(w["start"]) <= src_t + 1.0]
    required = span["required"] if span else []
    return {
        "frame_id": f"f{moment.frame}", "moment": moment.label, "output_time": round(out_t, 2),
        "segment": timeline.segment_at(moment.frame).kind, "source_time": round(src_t, 2),
        "beat": f"{beat['role']}: {beat.get('note', '')}" if beat else "(outside the story beats)",
        "intent": planned["intent"] if planned else "", "target": planned["target_id"] if planned else "",
        "dominant_speaker": span["dominant_speaker"] if span else "",
        "linked_faces": [f"{v['id']}={v['speaker'] or 'unlinked'}" for v in (span["visible"] if span else [])],
        "required": [f"R{k} {r['kind']}: {r['reason']}" for k, r in enumerate(required, 1)],
        "burned_captions": {lane: " ".join(words) for lane, words in burned.items()},
        "caption_truth": " ".join(spoken),
        "_required_boxes": required,
    }


def review(provider, route, pairs: Sequence[tuple[dict[str, Any], np.ndarray, np.ndarray]], story: dict[str, Any],
           cold_open: dict[str, Any]) -> dict[str, Any]:
    """pairs: (evidence, source frame, rendered frame)."""
    images = []
    lines = [f"STORY: {story['title']} | {story.get('reason', '')}",
             "BEATS: " + "; ".join(f"{b['role']} [{b['start']:.2f}-{b['end']:.2f}] {b.get('note', '')}"
                                   for b in story["beats"]),
             f"COLD OPEN PEAK: source {cold_open['peak']['start']:.2f}-{cold_open['peak']['end']:.2f} "
             f"(window {cold_open['window']['start']:.2f}-{cold_open['window']['end']:.2f}) "
             f"{cold_open.get('reason', '')}", "", "FRAMES:"]
    for evidence, source, rendered in pairs:
        fid = evidence["frame_id"]
        images.append(ImageInput(_jpeg(draw_required(source, evidence["_required_boxes"]), 640), f"{fid} SOURCE"))
        images.append(ImageInput(_jpeg(rendered, 360), f"{fid} RENDERED"))
        lines.append(" | ".join(f"{k}={v}" for k, v in evidence.items() if not k.startswith("_")))
    ids = [e["frame_id"] for e, _, _ in pairs]
    return provider.json_task("final_reviewer", route, instructions=INSTRUCTIONS, input_text="\n".join(lines),
                              schema=schema_for(ids), schema_name="mimir_final_review_v2", images=images)


def verdict_to_check(verdict: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Map the reviewer's findings to a QC check with deterministic repair directives."""
    issues = verdict.get("issues", [])
    high = [i for i in issues if i["severity"] == "high"]
    spans: set[str] = set()
    for issue in high:
        if issue["type"] not in VISUAL_REPAIRABLE:
            continue
        frame = int(issue["frame_id"].lstrip("f"))
        spans |= {s["id"] for s in context["spans"] if s["frames"][0] <= frame < s["frames"][1]}
    repairable = bool(high) and all(i["type"] in VISUAL_REPAIRABLE for i in high)
    return {"name": "final_review", "passed": not high, "severity": "fail" if high else "warn",
            "details": {"verdict": verdict.get("verdict"), "issues": issues},
            "repairable": repairable,
            "repair": {"widen_spans": sorted(spans), "conservative_camera": bool(spans)}}
