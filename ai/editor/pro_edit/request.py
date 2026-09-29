"""Production planner request: compact, complete, secret-free.

EditContext -> PlannerRequest -> provider adapter -> structured output ->
parser -> validator -> resolver. The request carries semantics (story spans,
caption word references, speaker timeline, subject tracks, visual/audio
events, style, budget, constraints) - never raw pipeline dumps, paths or keys.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ai.editor.pro_edit.context import EditContext
from ai.editor.pro_edit.policy import effect_budget
from ai.editor.pro_edit.schema import (
    EDIT_PLAN_SCHEMA_VERSION,
    INTRO_CAMERAS,
    INTRO_MOTIONS,
    PLAN_TIMELINE_DOMAIN,
    planner_json_schema,
)
from ai.editor.pro_edit.style import StylePack

PLANNER_REQUEST_VERSION = 2
SCHEMA_NAME = "mimir_pro_edit_plan_v2"
MAX_WORDS = 360
MAX_VISUAL_EVENTS = 24
MAX_SPEAKER_SEGMENTS = 80
MAX_SHOT_CUTS = 64

SYSTEM_PROMPT = """
You are MIMIR's edit director for an ALREADY-SELECTED short.
MIMIR already decided WHAT story is shown and which intro (cold-open) is used.
You only decide HOW it is presented: the editorial INTENT of each moment.

Think like a professional short-form editor. Before choosing anything, reason over the evidence:
story beats (story_spans: hook/setup/escalation/payoff/reaction and their protection), who speaks when
(speaker_timeline; names only where verified), reactions and laughter, face tracks (faces: where each
subject sits and which speaker it belongs to), shot cuts (shot_cuts_s), motion and sound events, the
content layout (talking head, gameplay + facecam, screen share, panels), persistent game UI / HUD / chat
regions (ui_regions), action regions inside payoff/reaction spans, and required visual content that must
stay visible. Ask per moment: what must the viewer SEE right now to follow the story? Very often the
answer is the whole frame: a stable wide shot is a professional choice, not a failure.

You do not render. You do not write FFmpeg, code, or commands.
You do not modify story boundaries, clip boundaries, intro selection, or pacing.
You do not modify transcript text, caption text, word timestamps, or speaker labels.
You never emit replacement text or replacement timing of any kind.
Main-story events use paced_clip seconds inside [visible_start_s, duration_s] and must name
the story span they belong to (story_span_id from story_spans). The optional "intro" directive
presents the EXISTING cold-open as a whole: it has no times, and you cannot choose another intro.
You select bounded behaviors from the allowed enums only. You never choose zoom scales, crop
coordinates, milliseconds, output size, or aspect ratio: the engine resolves all physical
parameters from your enum + intensity (0..1) and the story/caption/subject constraints.

Rules:
- Strong effects (punch_in, punch_in_fast, snap_reframe) are scarce; respect effect_budget.
  Professional editing is selective, not constant motion.
- PAYOFF VISIBILITY beats all decoration. Spans marked SOURCE_REQUIRED never get meme/broll.
  Spans with visual_importance "action" keep the whole action visible; prefer wider framing there.
- Use the role of the story span; you cannot redefine roles.
- reaction_close only where reaction evidence exists; dual_subject only with two subject tracks.
- Reference subjects/speakers only by ids present in the input. A speaker id steers the camera only
  when a subject track is linked to it. With no subject tracks use target center_safe.
- emphasis_word_ids may only contain word ids from the input words inside the event window.
- caption_style (default|emphasis|impact) is caption INTENT only; the engine owns typography, layout and
  timing. impact belongs to payoff moments; emphasis is rare. Emphasize at most one or two words per sentence.
- emphasis_reasons optionally explains an emphasized id (name, number, payoff, reaction, contrast, surprise,
  generic). Use [] when unsure; reasons never change text, timing or speakers.
- Gameplay / screen content with UI or an action region: keep it readable and whole; never frame it away to
  chase a face. A facecam reaction can be featured only when the action is not what matters in that moment.
- Never cross a shot cut with a subject-locked move; a new shot restarts the framing decision.
- scene.action_regions are corroborated story action. scene.ambiguous_motion is motion better explained by
  UI/chat/HUD, a person's own movement, a transient burst or an overlay-like corner: it is NEVER a reason to
  move the camera there. When the action location is unclear, keep the context wide.
- When scene.visual_context says frames are attached, use them to understand what the structured evidence
  refers to and what must remain visible. They are coarse LOW-detail evidence, not a geometry authority.
  Never derive crop coordinates from the frames and never contradict protected/required visual evidence.
- Evidence boxes are given as region words (where/size). You never output positions, boxes or numbers
  for the frame; the engine computes every crop from your intent and the evidence.
- Prefer static_clean (or no event) when uncertain. UNCERTAIN => LESS EDITING, never invent an effect.
- confidence is your editorial confidence (0..1); low confidence is muted by the engine.
- Return only the JSON object required by the schema.
""".strip()

REPAIR_INSTRUCTIONS = """
Your previous edit plan was rejected by MIMIR's validator.
Fix ONLY the listed problems and return the corrected JSON object only.
Do not add new effects. When unsure, remove the offending event or set intro to null.
""".strip()


@dataclass(frozen=True)
class PlannerRequest:
    request_id: str
    kind: str  # "plan" | "repair"
    instructions: str
    payload: Mapping[str, Any]
    schema: Mapping[str, Any]
    schema_name: str = SCHEMA_NAME
    visual_inputs: tuple[Any, ...] = ()

    def input_text(self) -> str:
        return json.dumps(self.payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    def input_value(self) -> Any:
        """Responses API input. Base64 images are sent live and never written to artifacts."""
        if not self.visual_inputs:
            return self.input_text()
        content: list[dict[str, Any]] = [{"type": "input_text", "text": self.input_text()}]
        for index, frame in enumerate(self.visual_inputs, start=1):
            manifest = frame.manifest()
            content.append({
                "type": "input_text",
                "text": (
                    f"VISUAL_FRAME_{index}: sampled_t={manifest['sampled_t']}s; "
                    f"reason={manifest['reason']}. Read-only evidence; do not output crop coordinates."
                ),
            })
            content.append(frame.api_item())
        return [{"role": "user", "content": content}]

    def to_artifact(self) -> dict[str, Any]:
        return {"request_version": PLANNER_REQUEST_VERSION, "request_id": self.request_id, "kind": self.kind,
                "schema_name": self.schema_name, "instructions": self.instructions,
                "payload": dict(self.payload), "schema": dict(self.schema),
                "visual_inputs": [frame.manifest() for frame in self.visual_inputs]}


def _request_id(instructions: str, payload: Mapping[str, Any], schema: Mapping[str, Any]) -> str:
    blob = json.dumps([instructions, payload, schema], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _r(value: float) -> float:
    return round(float(value), 2)


def build_payload(context: EditContext, style: StylePack) -> dict[str, Any]:
    visible = context.clip.visible_start_s
    budget = effect_budget(context.clip.duration_s - visible, style)
    words = [w for w in context.words if w.end > visible][:MAX_WORDS]
    intro = None
    if context.intro is not None and context.intro_span is not None:
        span = context.intro_span
        intro = {
            "span_id": span.span_id,
            "duration_s": _r(context.intro.teaser_duration),
            "visual_importance": span.visual_importance,
            "reaction_evidence": any(r.start < span.end and span.start < r.end
                                     for r in context.story.reaction_ranges),
            "allowed_motion": sorted(m.value for m in INTRO_MOTIONS),
            "allowed_camera": sorted(c.value for c in INTRO_CAMERAS),
            "note": "presentation only; the intro selection and handoff are fixed by MIMIR",
        }
    return {
        "request_version": PLANNER_REQUEST_VERSION,
        "clip": {
            "duration_s": _r(context.clip.duration_s), "fps": round(context.clip.fps.fps, 3),
            "width": context.clip.width, "height": context.clip.height,
            "visible_start_s": _r(visible), "timeline_domain": PLAN_TIMELINE_DOMAIN.value,
            "layout": context.layout_content_type or "unknown",
        },
        "story_spans": [
            {"id": s.span_id, "role": s.role.value, "start": _r(s.start), "end": _r(s.end),
             "protection": s.to_dict()["protection"], "visual_importance": s.visual_importance,
             "required_subject_ids": list(s.required_subject_ids), "must_keep": s.must_keep,
             "has_required_regions": bool(s.required_regions), "evidence": list(s.evidence)[:6]}
            for s in context.spans if s.end > visible
        ],
        "intro": intro,
        "caption_words": [[w.id, w.text, _r(w.start), _r(w.end), w.speaker_id] for w in words],
        "caption_words_note": "read-only references; only ids may be used (emphasis_word_ids)",
        "speaker_timeline": [[_r(a), _r(b), spk] for a, b, spk in context.speaker_segments][:MAX_SPEAKER_SEGMENTS],
        "speaker_names": {spk: (context.speaker_identities.get(spk) or "unverified (anonymous)")
                          for spk in sorted(context.speaker_ids())},
        "subjects": [
            {"id": t.subject_id, "kind": t.kind, "speaker_id": t.speaker_id,
             "speaker_link_confidence": round(t.speaker_confidence, 2),
             "t_range": t.summary()["t_range"], "mean_confidence": t.summary()["mean_confidence"]}
            for t in context.subject_tracks
        ],
        "visual_events": [
            {"start": _r(v.start), "end": _r(v.end), "type": v.type, "desc": v.description}
            for v in context.visual_events[:MAX_VISUAL_EVENTS]
        ],
        "audio_events": {
            "speech_gaps": [[_r(r.start), _r(r.end)] for r in context.speech_gap_ranges],
            "laughter": [[_r(r.start), _r(r.end)] for r in context.laughter_events],
            "energy": [[_r(r.start), _r(r.end)] for r in context.energy_events],
            "transients": [[_r(r.start), _r(r.end)] for r in context.transient_events][:MAX_VISUAL_EVENTS],
        },
        "shot_cuts_s": [_r(t) for t in context.scene_changes if t >= visible][:MAX_SHOT_CUTS],
        "motion_events": [[_r(r.start), _r(r.end)] for r in context.motion_events][:MAX_VISUAL_EVENTS],
        "scene": dict(context.director_evidence),
        "caption_region": "burned captions occupy the band shown in constraints; faces are kept above it",
        "effect_budget": dict(budget),
        "style": style.describe_for_planner(),
        "constraints": {
            "schema_version": EDIT_PLAN_SCHEMA_VERSION,
            "style_pack": style.name,
            "timeline_domain": PLAN_TIMELINE_DOMAIN.value,
            "caption_band": [b.to_dict() if hasattr(b, "to_dict") else [b.x0, b.y0, b.x1, b.y1]
                             for b in context.caption_region.bands],
            "forbidden": ["caption/word text", "word timing", "speaker labels", "story/clip boundaries",
                          "intro selection or timing", "zoom/crop numbers", "output geometry", "commands"],
        },
    }


def build_planner_request(context: EditContext, style: StylePack,
                          visual_inputs: Sequence[Any] = ()) -> PlannerRequest:
    payload = build_payload(context, style)
    schema = planner_json_schema(style.name)
    return PlannerRequest(_request_id(SYSTEM_PROMPT, payload, schema), "plan", SYSTEM_PROMPT, payload, schema,
                          visual_inputs=tuple(visual_inputs))


def build_repair_request(original: PlannerRequest, invalid_text: str, errors: list[str],
                         context: EditContext) -> PlannerRequest:
    """Validator errors only (no re-sent analysis payload)."""
    payload = {
        "invalid": invalid_text[:12000],
        "errors": errors,
        "allowed": {
            "story_span_ids": [s.span_id for s in context.spans],
            "subject_ids": sorted(context.subject_ids()),
            "speaker_ids": sorted(context.speaker_ids()),
            "word_id_range": [min(w.id for w in context.words), max(w.id for w in context.words)]
            if context.words else [],
            "duration_s": round(context.clip.duration_s, 3),
            "visible_start_s": round(context.clip.visible_start_s, 3),
            "intro_available": context.intro_span is not None,
            **dict(original.payload["constraints"]),
        },
    }
    instructions = original.instructions + "\n\n" + REPAIR_INSTRUCTIONS
    return PlannerRequest(_request_id(instructions, payload, original.schema), "repair", instructions, payload,
                          original.schema)
