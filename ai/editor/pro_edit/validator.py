"""Strict EditPlan validator. No planner output is trusted before this runs.

Result classes (spec section 29):

* VALID       - plan accepted unchanged
* SANITIZED   - some fields clamped/defaulted or events rejected; plan usable
* FATAL       - whole plan rejected (wrong schema/domain/style, attempted
                truth mutation, command/geometry injection, broken structure)

Validation is idempotent: validating a validator-produced plan is VALID.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Mapping

from ai.editor.pro_edit.context import EditContext
from ai.editor.pro_edit.errors import EditPlanTimelineError, EditPlanValidationError
from ai.editor.pro_edit.policy import (
    PolicyNote,
    effect_budget,
    enforce_density,
    enforce_role_policy,
    gate_confidence,
    resolve_camera_overlaps,
)
from ai.editor.pro_edit.schema import (
    MAX_EMPHASIS_WORDS_PER_EVENT,
    MAX_EVENT_NOTE_CHARS,
    INTRO_CAMERAS,
    INTRO_MOTIONS,
    MAX_PLAN_EVENTS,
    PLAN_TIMELINE_DOMAIN,
    STRONG_MOTIONS,
    SUPPORTED_PLAN_SCHEMA_VERSIONS,
    CameraMode,
    CaptionStyle,
    EmphasisReason,
    EditEvent,
    EditPlan,
    EditTarget,
    IntroDirective,
    MotionPreset,
    ReasonCode,
    SfxCue,
    StoryRole,
    SupportVisual,
    TargetType,
)
from ai.editor.pro_edit.style import StylePack


class Severity(str, Enum):
    SANITIZED = "sanitized"
    REJECTED = "rejected"
    FATAL = "fatal"


class PlanStatus(str, Enum):
    VALID = "valid"
    SANITIZED = "sanitized"
    FATAL = "fatal"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: Severity
    message: str
    event_id: str | None = None
    category: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "severity": self.severity.value, "category": self.category,
                "event_id": self.event_id, "message": self.message}


@dataclass(frozen=True)
class ValidationReport:
    status: PlanStatus
    plan: EditPlan | None
    issues: tuple[ValidationIssue, ...] = ()
    events_received: int = 0
    events_valid: int = 0
    events_sanitized: int = 0
    events_rejected: int = 0
    budget: Mapping[str, int] = field(default_factory=dict)

    @property
    def fatal(self) -> bool:
        return self.status is PlanStatus.FATAL

    def raise_if_fatal(self) -> EditPlan:
        if self.fatal or self.plan is None:
            messages = "; ".join(i.message for i in self.issues if i.severity is Severity.FATAL)
            raise EditPlanValidationError(f"edit plan rejected: {messages or 'fatal'}", list(self.issues))
        return self.plan

    def error_summary(self, limit: int = 20) -> list[str]:
        rows = [i for i in self.issues if i.severity in (Severity.FATAL, Severity.REJECTED)]
        return [f"{i.event_id or 'plan'}: {i.code}: {i.message}" for i in rows[:limit]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "events_received": self.events_received,
            "events_valid": self.events_valid,
            "events_sanitized": self.events_sanitized,
            "events_rejected": self.events_rejected,
            "budget": dict(self.budget),
            "issues": [i.to_dict() for i in self.issues],
        }


# Keys that would let a planner rewrite MIMIR-owned truth -> FATAL.
TRUTH_MUTATION_KEYS = frozenset({
    "caption", "captions", "caption_text", "text", "word_text", "words", "transcript", "subtitle",
    "subtitles", "replacement_text", "word_start", "word_end", "word_timings", "timestamps",
    "speaker", "speaker_label", "speaker_labels", "speaker_name", "speaker_assignment",
    "clip_start", "clip_end", "clip_boundaries", "story", "story_boundaries", "payoff_start",
    "payoff_end", "payoff_range", "must_keep", "must_keep_ranges", "cut_ranges", "cuts", "trim",
    "intro_source", "clip_index", "selected_clip", "replacement_text", "replacement_start",
    "replacement_end", "replacement_speaker", "replacement_words", "word_text_override",
    "teaser", "teaser_start", "teaser_end", "intro_start", "intro_end", "intro_clip", "peak_id",
    "peak_time", "main_restart", "handoff", "story_spans", "span_start", "span_end",
})
# Executable or render-backend payloads -> FATAL (never executed anyway).
COMMAND_KEYS = frozenset({"ffmpeg", "filter", "filter_complex", "command", "cmd", "shell", "python",
                          "code", "script", "exec", "eval"})
# Output geometry is config-owned -> FATAL (no aspect/resolution mutation).
GEOMETRY_KEYS = frozenset({"aspect", "aspect_ratio", "resolution", "width", "height", "output_profile",
                           "output_width", "output_height"})
# Physical render parameters are engine-owned -> ignored (SANITIZED).
RENDER_PARAMETER_KEYS = frozenset({"scale", "zoom", "scale_peak", "scale_start", "crop", "x", "y",
                                   "anchor_x", "anchor_y", "attack_ms", "hold_ms", "release_ms",
                                   "attack_s", "hold_s", "release_s", "easing", "velocity"})
TOP_LEVEL_KEYS = frozenset({"schema_version", "style_pack", "timeline_domain", "events", "planner", "note",
                            "intro"})
EVENT_KEYS = frozenset({"event_id", "start", "end", "role", "camera", "motion", "target", "intensity",
                        "caption_style", "emphasis_word_ids", "emphasis_reasons", "sfx", "support_visual",
                        "confidence",
                        "reason_code", "note", "story_span_id"})
INTRO_KEYS = frozenset({"camera", "motion", "target", "intensity", "confidence", "reason_code", "note"})
EVENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
EMPHASIS_WINDOW_TOLERANCE_S = 0.25


class _Collector:
    def __init__(self) -> None:
        self.issues: list[ValidationIssue] = []
        self.touched: set[str] = set()
        self.rejected: set[str] = set()

    def add(self, code: str, severity: Severity, message: str, event_id: str | None = None,
            category: str = "") -> None:
        self.issues.append(ValidationIssue(code, severity, message, event_id, category))
        if event_id is not None:
            if severity is Severity.REJECTED:
                self.rejected.add(event_id)
            elif severity is Severity.SANITIZED:
                self.touched.add(event_id)

    def notes(self, notes: Iterable[PolicyNote], category: str) -> None:
        for note in notes:
            self.add(note.code, Severity.SANITIZED, note.message, note.event_id, category)

    @property
    def fatal(self) -> bool:
        return any(i.severity is Severity.FATAL for i in self.issues)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _contains_non_finite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_non_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_non_finite(v) for v in value)
    return False


def _forbidden_key_scan(data: Mapping[str, Any], where: str, out: _Collector, event_id: str | None) -> None:
    for key in data:
        name = str(key).strip().casefold()
        if name in TRUTH_MUTATION_KEYS:
            out.add("truth_mutation_attempt", Severity.FATAL,
                    f"{where} contains MIMIR-owned truth field {key!r}", event_id, "caption/story")
        elif name in COMMAND_KEYS:
            out.add("command_injection_attempt", Severity.FATAL,
                    f"{where} contains executable/backend field {key!r}", event_id, "schema")
        elif name in GEOMETRY_KEYS:
            out.add("output_geometry_mutation", Severity.FATAL,
                    f"{where} tries to set output geometry via {key!r}", event_id, "output")


def _enum(enum_type: type[Enum], value: Any) -> Any:
    if not isinstance(value, str):
        return None
    try:
        return enum_type(value.strip())
    except ValueError:
        return None


def _parse_event(raw: Any, index: int, context: EditContext, style: StylePack, seen: set[str],
                 out: _Collector) -> EditEvent | None:
    label = f"event[{index}]"
    if not isinstance(raw, dict):
        out.add("event_not_object", Severity.REJECTED, f"{label} is not an object", label, "schema")
        return None
    raw_id = raw.get("event_id")
    event_id = raw_id.strip() if isinstance(raw_id, str) else ""
    if not EVENT_ID_PATTERN.match(event_id):
        out.add("event_id_invalid", Severity.REJECTED, f"{label} has invalid event_id {raw_id!r}", label, "schema")
        return None
    _forbidden_key_scan(raw, f"event {event_id}", out, event_id)
    for key in raw:
        name = str(key).strip().casefold()
        if name in RENDER_PARAMETER_KEYS:
            out.add("render_parameter_ignored", Severity.SANITIZED,
                    f"planner may not choose physical parameter {key!r}; engine resolves it", event_id, "motion")
        elif name not in EVENT_KEYS and name not in TRUTH_MUTATION_KEYS | COMMAND_KEYS | GEOMETRY_KEYS:
            out.add("unknown_field_ignored", Severity.SANITIZED, f"unknown field {key!r}", event_id, "schema")
    if event_id in seen:
        out.add("duplicate_event_id", Severity.REJECTED, f"duplicate event_id {event_id}", event_id, "schema")
        return None
    seen.add(event_id)
    if _contains_non_finite(raw):
        out.add("non_finite_number", Severity.REJECTED, "NaN/Infinity is not allowed", event_id, "schema")
        return None

    missing = [k for k in ("start", "end", "role", "camera", "motion", "intensity", "confidence", "reason_code")
               if k not in raw]
    if missing:
        out.add("missing_required_field", Severity.REJECTED, f"missing {', '.join(missing)}", event_id, "schema")
        return None
    for key in ("start", "end", "intensity", "confidence"):
        if not _is_number(raw.get(key)):
            out.add("numeric_type_invalid", Severity.REJECTED, f"{key} must be a number", event_id, "schema")
            return None

    role = _enum(StoryRole, raw.get("role"))
    if role is None:
        out.add("unknown_role", Severity.REJECTED, f"unknown role {raw.get('role')!r}", event_id, "schema")
        return None
    motion = _enum(MotionPreset, raw.get("motion"))
    if motion is None:
        out.add("unknown_motion_static_clean", Severity.REJECTED,
                f"unknown motion {raw.get('motion')!r}; event rejected (static_clean)", event_id, "motion")
        return None
    camera = _enum(CameraMode, raw.get("camera"))
    if camera is None:
        out.add("unknown_camera_static_clean", Severity.REJECTED,
                f"unknown camera {raw.get('camera')!r}; event rejected (static_clean)", event_id, "camera")
        return None

    # --- TIME (PACED_CLIP) ---
    duration = context.clip.duration_s
    tol = context.clip.frame_tolerance_s
    start, end = float(raw["start"]), float(raw["end"])
    if start < -tol:
        out.add("time_negative_start", Severity.REJECTED, f"start {start:.3f} < 0", event_id, "time")
        return None
    if end > duration + tol:
        out.add("time_end_out_of_range", Severity.REJECTED, f"end {end:.3f} > duration {duration:.3f}",
                event_id, "time")
        return None
    if start < 0:
        out.add("time_start_clamped", Severity.SANITIZED, f"start {start:.4f} -> 0 (within frame tolerance)",
                event_id, "time")
        start = 0.0
    if end > duration:
        out.add("time_end_clamped", Severity.SANITIZED, f"end {end:.4f} -> {duration:.4f} (frame tolerance)",
                event_id, "time")
        end = duration
    if end <= start:
        out.add("time_end_not_after_start", Severity.REJECTED, f"end {end:.3f} <= start {start:.3f}",
                event_id, "time")
        return None
    visible = context.clip.visible_start_s
    if end <= visible + tol:
        out.add("time_not_visible", Severity.REJECTED,
                f"event ends before the visible main starts ({visible:.3f}s after intro handoff)", event_id, "time")
        return None
    if start < visible:
        out.add("time_start_visible_clamp", Severity.SANITIZED, f"start {start:.3f} -> visible {visible:.3f}",
                event_id, "time")
        start = visible
    # --- STORY SPAN reference (semantic link; engine owns the timestamps) ---
    span_id: str | None = None
    raw_span = raw.get("story_span_id")
    if raw_span is not None:
        span = context.span(raw_span.strip()) if isinstance(raw_span, str) else None
        if span is None or span is context.intro_span:
            out.add("story_span_unknown", Severity.SANITIZED,
                    f"story_span_id {raw_span!r} is not a main story span; derived from window", event_id, "story")
        elif not span.overlaps(start, end):
            out.add("event_outside_story_span", Severity.REJECTED,
                    f"window {start:.3f}-{end:.3f} does not touch span {span.span_id}", event_id, "story")
            return None
        else:
            span_id = span.span_id
            clamped = (max(start, span.start), min(end, span.end))
            if abs(clamped[0] - start) > tol or abs(clamped[1] - end) > tol:
                out.add("event_clamped_to_story_span", Severity.SANITIZED,
                        f"{start:.3f}-{end:.3f} -> {clamped[0]:.3f}-{clamped[1]:.3f} ({span.span_id})",
                        event_id, "story")
            start, end = clamped
    if span_id is None:
        # One presentation event belongs to exactly one story span: the dominant one.
        owner = context.dominant_span(start, end)
        if owner is not None:
            span_id = owner.span_id
            clamped = (max(start, owner.start), min(end, owner.end))
            if abs(clamped[0] - start) > tol or abs(clamped[1] - end) > tol:
                out.add("event_clamped_to_story_span", Severity.SANITIZED,
                        f"{start:.3f}-{end:.3f} -> {clamped[0]:.3f}-{clamped[1]:.3f} ({owner.span_id})",
                        event_id, "story")
            start, end = clamped
    limits = style.durations
    if end - start > limits.max_event_s:
        out.add("time_max_duration_clamp", Severity.SANITIZED, f"duration clamped to {limits.max_event_s}s",
                event_id, "time")
        end = start + limits.max_event_s
    if end - start < limits.min_event_s - 1e-9:
        out.add("time_below_min_duration", Severity.REJECTED,
                f"duration {end - start:.3f}s < {limits.min_event_s}s", event_id, "time")
        return None

    # --- MOTION numbers ---
    intensity, confidence = float(raw["intensity"]), float(raw["confidence"])
    if not 0.0 <= intensity <= 1.0:
        out.add("intensity_clamped", Severity.SANITIZED, f"intensity {intensity} -> [0,1]", event_id, "motion")
        intensity = max(0.0, min(1.0, intensity))
    if not 0.0 <= confidence <= 1.0:
        out.add("confidence_clamped", Severity.SANITIZED, f"confidence {confidence} -> [0,1]", event_id, "motion")
        confidence = max(0.0, min(1.0, confidence))

    # --- TARGET ---
    target = EditTarget(TargetType.CENTER_SAFE, None)
    raw_target = raw.get("target", {"type": "center_safe", "id": None})
    if isinstance(raw_target, dict):
        target_type = _enum(TargetType, raw_target.get("type"))
        target_id = raw_target.get("id")
        if target_type is None or (target_id is not None and not isinstance(target_id, str)):
            out.add("target_invalid_center_safe", Severity.SANITIZED, f"target {raw_target!r} -> center_safe",
                    event_id, "target")
        else:
            target = EditTarget(target_type, target_id.strip() if isinstance(target_id, str) and target_id.strip()
                                else None)
    else:
        out.add("target_invalid_center_safe", Severity.SANITIZED, "target must be an object", event_id, "target")

    # --- OPTIONAL CHANNELS ---
    def optional(key: str, enum_type: type[Enum], default: Enum, category: str) -> Any:
        if key not in raw:
            return default
        value = _enum(enum_type, raw.get(key))
        if value is None:
            out.add(f"{key}_invalid_default", Severity.SANITIZED, f"{key} {raw.get(key)!r} -> {default.value}",
                    event_id, category)
            return default
        return value

    caption_style = optional("caption_style", CaptionStyle, CaptionStyle.DEFAULT, "caption")
    sfx = optional("sfx", SfxCue, SfxCue.NONE, "sfx")
    support_visual = optional("support_visual", SupportVisual, SupportVisual.NONE, "support_visual")
    reason = _enum(ReasonCode, raw.get("reason_code"))
    if reason is None:
        out.add("reason_code_invalid", Severity.SANITIZED, f"reason_code {raw.get('reason_code')!r} not controlled",
                event_id, "schema")
        reason = ReasonCode.VALIDATOR_FALLBACK

    # --- CAPTION references (IDs only; text/timing are immutable) ---
    emphasis: list[int] = []
    raw_ids = raw.get("emphasis_word_ids", [])
    if not isinstance(raw_ids, list):
        out.add("emphasis_ids_invalid", Severity.SANITIZED, "emphasis_word_ids must be a list", event_id, "caption")
        raw_ids = []
    words = context.word_index()
    for value in raw_ids:
        if not isinstance(value, int) or isinstance(value, bool):
            out.add("emphasis_id_type", Severity.SANITIZED, f"word id {value!r} is not an integer", event_id,
                    "caption")
            continue
        word = words.get(value)
        if word is None:
            out.add("caption_unknown_word_id", Severity.SANITIZED, f"word id {value} does not exist; removed",
                    event_id, "caption")
            continue
        if word.end < start - EMPHASIS_WINDOW_TOLERANCE_S or word.start > end + EMPHASIS_WINDOW_TOLERANCE_S:
            out.add("caption_word_outside_event", Severity.SANITIZED, f"word id {value} outside event window",
                    event_id, "caption")
            continue
        if value not in emphasis:
            emphasis.append(value)
    if len(emphasis) > MAX_EMPHASIS_WORDS_PER_EVENT:
        out.add("emphasis_truncated", Severity.SANITIZED, "too many emphasis words", event_id, "caption")
        emphasis = emphasis[:MAX_EMPHASIS_WORDS_PER_EVENT]

    # Optional reasons (compatible extension): only for accepted emphasis ids.
    reasons: list[tuple[int, EmphasisReason]] = []
    raw_reasons = raw.get("emphasis_reasons", [])
    if not isinstance(raw_reasons, list):
        out.add("emphasis_reasons_invalid", Severity.SANITIZED, "emphasis_reasons must be a list", event_id,
                "caption")
        raw_reasons = []
    for item in raw_reasons:
        word_id = item.get("word_id") if isinstance(item, dict) else None
        # Own name: the event's ReasonCode (``reason``) must not be overwritten here.
        word_reason = _enum(EmphasisReason, item.get("reason")) if isinstance(item, dict) else None
        if not isinstance(word_id, int) or isinstance(word_id, bool) or word_id not in emphasis or word_reason is None:
            out.add("emphasis_reason_dropped", Severity.SANITIZED, f"emphasis reason {item!r} ignored", event_id,
                    "caption")
            continue
        if all(existing != word_id for existing, _ in reasons):
            reasons.append((word_id, word_reason))

    note = raw.get("note", "")
    note = " ".join(note.split())[:MAX_EVENT_NOTE_CHARS] if isinstance(note, str) else ""

    return EditEvent(
        event_id=event_id, start=round(start, 6), end=round(end, 6), role=role, camera=camera, motion=motion,
        target=target, intensity=round(intensity, 6), confidence=round(confidence, 6), reason_code=reason,
        caption_style=caption_style, emphasis_word_ids=tuple(emphasis), sfx=sfx, support_visual=support_visual,
        story_span_id=span_id, note=note, emphasis_reasons=tuple(reasons),
    )


def _parse_intro(raw: Any, context: EditContext, style: StylePack, out: _Collector) -> IntroDirective | None:
    """Presentation of the EXISTING mandatory intro. No times: intro selection,
    source range and handoff stay owned by MIMIR's intro system."""
    if raw is None:
        return None
    label = "intro"
    if not isinstance(raw, dict):
        out.add("intro_not_object", Severity.REJECTED, "intro directive must be an object or null", label, "intro")
        return None
    _forbidden_key_scan(raw, "intro", out, label)
    for key in raw:
        name = str(key).strip().casefold()
        if name in {"start", "end", "duration", "source_start", "source_end"}:
            out.add("intro_timing_owned_by_mimir", Severity.FATAL,
                    f"intro directive may not carry time field {key!r}", label, "intro")
        elif name in RENDER_PARAMETER_KEYS:
            out.add("render_parameter_ignored", Severity.SANITIZED, f"engine resolves {key!r}", label, "intro")
        elif name not in INTRO_KEYS and name not in TRUTH_MUTATION_KEYS | COMMAND_KEYS | GEOMETRY_KEYS:
            out.add("unknown_field_ignored", Severity.SANITIZED, f"unknown intro field {key!r}", label, "intro")
    if out.fatal:
        return None
    if context.intro_span is None:
        out.add("intro_directive_without_intro", Severity.REJECTED, "no selected intro in context", label, "intro")
        return None
    if _contains_non_finite(raw) or not _is_number(raw.get("intensity")) or not _is_number(raw.get("confidence")):
        out.add("intro_numeric_invalid", Severity.REJECTED, "intensity/confidence must be finite numbers",
                label, "intro")
        return None
    camera, motion = _enum(CameraMode, raw.get("camera")), _enum(MotionPreset, raw.get("motion"))
    reason = _enum(ReasonCode, raw.get("reason_code")) or ReasonCode.VALIDATOR_FALLBACK
    if camera is None or motion is None:
        out.add("intro_unknown_enum", Severity.REJECTED, "unknown intro camera/motion", label, "intro")
        return None
    if motion not in INTRO_MOTIONS:
        out.add("intro_motion_not_allowed", Severity.SANITIZED, f"{motion.value} -> punch_in", label, "intro")
        motion = MotionPreset.PUNCH_IN
    if camera not in INTRO_CAMERAS:
        out.add("intro_camera_not_allowed", Severity.SANITIZED, f"{camera.value} -> preserve", label, "intro")
        camera = CameraMode.PRESERVE
    intro = context.intro_span
    reaction_seen = any(r.start < intro.end and intro.start < r.end for r in context.story.reaction_ranges)
    if camera is CameraMode.REACTION_CLOSE and not reaction_seen:
        out.add("reaction_framing_without_evidence", Severity.SANITIZED, "intro has no reaction evidence",
                label, "intro")
        camera = CameraMode.PRESERVE
    intensity = max(0.0, min(1.0, float(raw["intensity"])))
    confidence = max(0.0, min(1.0, float(raw["confidence"])))
    target = EditTarget(TargetType.CENTER_SAFE, None)
    raw_target = raw.get("target")
    if isinstance(raw_target, dict):
        target_type = _enum(TargetType, raw_target.get("type"))
        target_id = raw_target.get("id") if isinstance(raw_target.get("id"), str) else None
        if target_type is TargetType.SUBJECT and target_id in context.subject_ids():
            target = EditTarget(TargetType.SUBJECT, target_id)
        elif target_type in (TargetType.ACTIVE_SPEAKER, TargetType.DOMINANT_SUBJECT) and context.subject_tracks:
            target = EditTarget(target_type, None)
        elif target_type not in (None, TargetType.CENTER_SAFE):
            out.add("intro_target_center_safe", Severity.SANITIZED, "no matching subject evidence", label, "intro")
    policy = style.confidence
    if confidence < policy.low:
        out.add("confidence_low_static", Severity.SANITIZED, "intro directive muted (low confidence)", label, "intro")
        return None
    if confidence < policy.medium:
        motion = MotionPreset.SLOW_PUSH if motion in STRONG_MOTIONS else motion
        intensity = min(intensity, policy.mild_intensity_cap)
    note = raw.get("note", "")
    return IntroDirective(camera, motion, target, round(intensity, 6), round(confidence, 6), reason,
                          " ".join(note.split())[:MAX_EVENT_NOTE_CHARS] if isinstance(note, str) else "")


def validate_plan_payload(
    raw: Any,
    context: EditContext,
    style: StylePack,
    *,
    planner: str = "unknown",
    executor_domain: str = PLAN_TIMELINE_DOMAIN.value,
) -> ValidationReport:
    out = _Collector()
    budget = effect_budget(context.clip.duration_s - context.clip.visible_start_s, style)

    def fatal_report(received: int = 0) -> ValidationReport:
        return ValidationReport(PlanStatus.FATAL, None, tuple(out.issues), received, 0, 0, received, budget)

    if not isinstance(raw, dict):
        out.add("plan_not_object", Severity.FATAL, "plan must be a JSON object", None, "schema")
        return fatal_report()
    _forbidden_key_scan(raw, "plan", out, None)
    for key in raw:
        if str(key) not in TOP_LEVEL_KEYS and str(key).casefold() not in TRUTH_MUTATION_KEYS | COMMAND_KEYS | GEOMETRY_KEYS:
            out.add("unknown_top_level_field_ignored", Severity.SANITIZED, f"unknown field {key!r}", None, "schema")
    version = raw.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version not in SUPPORTED_PLAN_SCHEMA_VERSIONS:
        out.add("schema_version_unsupported", Severity.FATAL, f"schema_version {version!r} unsupported", None,
                "schema")
    domain = raw.get("timeline_domain")
    if domain != executor_domain or domain != context.clip.timeline_domain.value:
        out.add("timeline_domain_mismatch", Severity.FATAL,
                f"plan domain {domain!r} != executor domain {executor_domain!r}; no silent conversion", None,
                "timeline_domain")
    if raw.get("style_pack") != style.name:
        out.add("style_pack_mismatch", Severity.FATAL, f"style_pack {raw.get('style_pack')!r} != {style.name!r}",
                None, "schema")
    events_raw = raw.get("events")
    if not isinstance(events_raw, list):
        out.add("events_not_list", Severity.FATAL, "events must be a list", None, "schema")
        return fatal_report()
    received = len(events_raw)
    if out.fatal:
        return fatal_report(received)
    if received > MAX_PLAN_EVENTS:
        out.add("events_truncated", Severity.SANITIZED, f"{received} events > {MAX_PLAN_EVENTS}", None, "schema")
        for extra in events_raw[MAX_PLAN_EVENTS:]:
            eid = extra.get("event_id") if isinstance(extra, dict) and isinstance(extra.get("event_id"), str) else None
            out.add("event_over_limit", Severity.REJECTED, "beyond max event count", eid or "overflow", "schema")
        events_raw = events_raw[:MAX_PLAN_EVENTS]

    seen: set[str] = set()
    parsed: list[EditEvent] = []
    original: dict[str, EditEvent] = {}
    for index, item in enumerate(events_raw):
        event = _parse_event(item, index, context, style, seen, out)
        if out.fatal:
            return fatal_report(received)
        if event is not None:
            parsed.append(event)
            original[event.event_id] = event

    # --- policy passes ---
    def apply_event_policy(event: EditEvent) -> EditEvent:
        gated, note = gate_confidence(event, style)
        if note:
            out.notes([note], "motion")
        gated, notes = enforce_role_policy(gated, context, style)
        out.notes(notes, "story")
        # Re-gate after the role change; a clamp may have moved intensity.
        gated, note = gate_confidence(gated, style)
        if note:
            out.notes([note], "motion")
        return gated

    policed = [apply_event_policy(event) for event in parsed]
    windows = {e.event_id: (e.start, e.end) for e in policed}
    policed, notes, rejected = resolve_camera_overlaps(policed, style)
    out.notes(notes, "camera")
    for event_id in rejected:
        out.add("camera_overlap_rejected", Severity.REJECTED, "rejected by camera overlap policy", event_id, "camera")
    policed = [e for e in policed if e.event_id not in rejected]
    # Trimmed windows may fall into another story role: re-apply role policy.
    policed = [apply_event_policy(e) if windows.get(e.event_id) != (e.start, e.end) else e for e in policed]
    policed, notes = enforce_density(policed, context.clip.duration_s - context.clip.visible_start_s, style, budget)
    out.notes(notes, "density")

    intro_directive = _parse_intro(raw.get("intro"), context, style, out)
    if out.fatal:
        return fatal_report(received)
    final = tuple(sorted(policed, key=lambda e: (e.start, e.end, e.event_id)))
    for event in final:
        if event != original.get(event.event_id):
            out.touched.add(event.event_id)
    rejected_count = len(out.rejected - {"intro"})
    sanitized_count = len({e.event_id for e in final} & out.touched)
    valid_count = len(final) - sanitized_count
    status = PlanStatus.VALID if not out.issues else PlanStatus.SANITIZED
    plan = EditPlan(
        schema_version=int(version),
        style_pack=style.name,
        timeline_domain=PLAN_TIMELINE_DOMAIN,
        events=final,
        planner=planner,
        intro=intro_directive,
    )
    return ValidationReport(status, plan, tuple(out.issues), received, valid_count, sanitized_count,
                            rejected_count, budget)


def validate_plan(plan: EditPlan, context: EditContext, style: StylePack) -> ValidationReport:
    """Re-validate an EditPlan object (e.g. a cached artifact)."""
    if plan.timeline_domain is not PLAN_TIMELINE_DOMAIN:
        raise EditPlanTimelineError(f"plan domain {plan.timeline_domain.value} is not executable")
    return validate_plan_payload(plan.to_dict(), context, style, planner=plan.planner)
