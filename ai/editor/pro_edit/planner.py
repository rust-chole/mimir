"""Provider-independent edit planners.

AI_DECIDES_EDITORIAL_INTENT / ENGINE_DECIDES_RENDERING_PARAMETERS:
planners only emit enum choices + normalized intensity/confidence + time
windows + existing word IDs. Every planner result is validated before use,
and model output is data only - it is never executed.

Budget: at most ONE planning call per selected short, plus at most ONE
repair call that carries only validator errors.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from ai.editor.pro_edit.context import EditContext
from ai.editor.pro_edit.errors import PlannerOutputError
from ai.editor.pro_edit.providers import PlannerProvider, ProviderResponse
from ai.editor.pro_edit.request import PlannerRequest, build_planner_request, build_repair_request
from ai.editor.pro_edit.schema import (
    EDIT_PLAN_SCHEMA_VERSION,
    PLAN_TIMELINE_DOMAIN,
    CameraMode,
    EditPlan,
    MotionPreset,
    StoryRole,
    empty_plan,
)
from ai.editor.pro_edit.style import StylePack
from ai.editor.pro_edit.validator import PlanStatus, ValidationReport, validate_plan_payload

MAX_REPAIR_ATTEMPTS = 1


@dataclass(frozen=True)
class PlannerOutcome:
    planner: str
    report: ValidationReport
    raw: dict[str, Any] | None = None
    model_calls: int = 0
    repair_attempted: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)
    exchanges: tuple[tuple[PlannerRequest, ProviderResponse], ...] = ()

    @property
    def plan(self) -> EditPlan:
        return self.report.raise_if_fatal()


class EditPlanner(Protocol):
    name: str

    def plan(self, context: EditContext, style: StylePack) -> PlannerOutcome:
        ...


def _validate(raw: Any, context: EditContext, style: StylePack, name: str) -> ValidationReport:
    return validate_plan_payload(raw, context, style, planner=name)


# ============================================================
# STATIC / RULE-BASED PLANNERS (no model call)
# ============================================================

class StaticEditPlanner:
    """Fallback: no presentation changes (identical to feature-off pixels)."""

    name = "static"

    def plan(self, context: EditContext, style: StylePack) -> PlannerOutcome:
        raw = empty_plan(style.name, self.name).to_dict()
        return PlannerOutcome(self.name, _validate(raw, context, style, self.name), raw)


class RuleBasedEditPlanner:
    """Deterministic, evidence-only planner (zero cost, fully reproducible)."""

    name = "rules_v1"

    def plan(self, context: EditContext, style: StylePack) -> PlannerOutcome:
        visible = context.clip.visible_start_s
        events: list[dict[str, Any]] = []
        has_subjects = bool(context.subject_tracks)

        def add(start: float, end: float, role: StoryRole, motion: MotionPreset, camera: CameraMode,
                intensity: float, reason: str, **extra: Any) -> None:
            start = max(start, visible)
            if end - start < style.durations.min_event_s:
                return
            events.append({
                "event_id": f"evt_{len(events) + 1:03d}", "start": round(start, 3), "end": round(end, 3),
                "role": role.value, "camera": camera.value, "motion": motion.value,
                "target": {"type": "dominant_subject" if has_subjects else "center_safe", "id": None},
                "intensity": intensity, "caption_style": extra.get("caption_style", "default"),
                "emphasis_word_ids": extra.get("emphasis_word_ids", []),
                "emphasis_reasons": extra.get("emphasis_reasons", []), "sfx": "none",
                "support_visual": "none", "confidence": 0.8, "reason_code": reason,
                "story_span_id": extra.get("span_id"),
            })

        for segment in context.spans:
            if segment.end <= visible or not segment.camera_allowed:
                continue
            length = segment.end - max(segment.start, visible)
            if segment.role is StoryRole.PAYOFF:
                peak = segment.visual_importance == "action"
                end = segment.start + min(length, 1.8)
                words = [w.id for w in context.words if segment.start <= w.start < min(end, segment.start + 1.0)
                         and len(w.text.strip(".,!?")) >= 3][:2]
                add(segment.start, end, StoryRole.PAYOFF,
                    MotionPreset.PUNCH_IN_FAST if peak else MotionPreset.PUNCH_IN,
                    CameraMode.SPEAKER_CLOSE if has_subjects else CameraMode.PRESERVE,
                    0.75, "payoff_hit", caption_style="impact", emphasis_word_ids=words,
                    emphasis_reasons=[{"word_id": w, "reason": "payoff"} for w in words], span_id=segment.span_id)
            elif segment.role is StoryRole.REACTION and length >= 0.5:
                add(segment.start, segment.start + min(length, 2.0), StoryRole.REACTION, MotionPreset.PUNCH_IN,
                    CameraMode.REACTION_CLOSE, 0.6, "reaction_hold", span_id=segment.span_id)
            elif segment.role is StoryRole.ESCALATION and length >= 1.0:
                add(segment.start, segment.start + min(length, 3.0), StoryRole.ESCALATION, MotionPreset.SLOW_PUSH,
                    CameraMode.SPEAKER_MEDIUM if has_subjects else CameraMode.PRESERVE, 0.5, "escalation_rise",
                    span_id=segment.span_id)
            elif segment.role is StoryRole.HOOK and length >= 0.8:
                add(segment.start, segment.start + min(length, 2.2), StoryRole.HOOK, MotionPreset.SLOW_PUSH,
                    CameraMode.PRESERVE, 0.45, "hook_emphasis", span_id=segment.span_id)
        intro = None
        if context.intro_span is not None:
            intro = {"camera": CameraMode.SPEAKER_CLOSE.value if has_subjects else CameraMode.PRESERVE.value,
                     "motion": MotionPreset.PUNCH_IN.value,
                     "target": {"type": "dominant_subject" if has_subjects else "center_safe", "id": None},
                     "intensity": 0.6, "confidence": 0.8, "reason_code": "hook_emphasis"}
        raw = {"schema_version": EDIT_PLAN_SCHEMA_VERSION, "style_pack": style.name,
               "timeline_domain": PLAN_TIMELINE_DOMAIN.value, "intro": intro, "events": events}
        return PlannerOutcome(self.name, _validate(raw, context, style, self.name), raw)


# ============================================================
# STRICT JSON PARSING
# ============================================================

def _reject_constant(value: str) -> Any:
    raise PlannerOutputError(f"non-standard JSON constant {value!r} rejected")


def parse_planner_json(text: str) -> dict[str, Any]:
    """Exactly one JSON object; NaN/Infinity rejected; ambiguity rejected."""
    body = (text or "").strip()
    if not body:
        raise PlannerOutputError("planner returned empty output")
    if body.startswith("```"):
        lines = body.splitlines()
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            raise PlannerOutputError("unterminated fenced JSON block")
        body = "\n".join(lines[1:-1]).strip()
    if not body.startswith("{"):
        raise PlannerOutputError("planner output is not a bare JSON object")
    decoder = json.JSONDecoder(parse_constant=_reject_constant)
    try:
        data, end = decoder.raw_decode(body)
    except PlannerOutputError:
        raise
    except json.JSONDecodeError as error:
        raise PlannerOutputError(f"invalid/partial JSON: {error.msg} at {error.pos}") from error
    if body[end:].strip():
        raise PlannerOutputError("trailing content after JSON object (ambiguous output)")
    if not isinstance(data, dict):
        raise PlannerOutputError("planner JSON root is not an object")
    return data


# ============================================================
# PROVIDER PLANNER (live model or replay: same boundary)
# ============================================================

class ProviderEditPlanner:
    """request -> provider -> parse -> validate (-> one repair) -> outcome."""

    def __init__(self, provider: PlannerProvider, *, max_repairs: int = MAX_REPAIR_ATTEMPTS,
                 visual_inputs: tuple[Any, ...] = ()) -> None:
        self.provider = provider
        self.max_repairs = max(0, min(MAX_REPAIR_ATTEMPTS, int(max_repairs)))
        self.visual_inputs = tuple(visual_inputs)
        self.name = f"{provider.name}:{provider.model}"

    def plan(self, context: EditContext, style: StylePack) -> PlannerOutcome:
        request = build_planner_request(context, style, self.visual_inputs)
        response = self.provider.complete(request)
        exchanges = [(request, response)]
        notes: list[str] = []
        try:
            raw = parse_planner_json(response.text)
            report = _validate(raw, context, style, self.name)
            if not report.fatal:
                return PlannerOutcome(self.name, report, raw, 1, False, tuple(notes), tuple(exchanges))
            errors = report.error_summary()
        except PlannerOutputError as error:
            errors = [str(error)]
        notes.append("first answer invalid: " + "; ".join(errors[:5]))
        if self.max_repairs < 1:
            raise PlannerOutputError("planner output invalid and repair disabled: " + "; ".join(errors[:5]))
        repair = build_repair_request(request, response.text, errors, context)
        repaired = self.provider.complete(repair)
        exchanges.append((repair, repaired))
        raw = parse_planner_json(repaired.text)
        report = _validate(raw, context, style, self.name)
        if report.fatal:
            raise PlannerOutputError("planner output still invalid after one repair: "
                                     + "; ".join(report.error_summary()[:5]))
        notes.append("repaired")
        return PlannerOutcome(self.name, report, raw, 2, True, tuple(notes), tuple(exchanges))


def outcome_from_cached(raw: dict[str, Any], context: EditContext, style: StylePack, planner: str) -> PlannerOutcome:
    """Re-validate a cached plan artifact against the fresh context (no model call)."""
    report = _validate(raw, context, style, planner)
    if report.status is PlanStatus.FATAL:
        raise PlannerOutputError("cached plan no longer valid: " + "; ".join(report.error_summary()[:5]))
    return PlannerOutcome(planner, report, raw, 0, False, ("cache",))
