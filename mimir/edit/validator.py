"""Stage ``edit_validation``: evidence and restraint policy applied to the director's plan.

Every correction is recorded with a reason code; nothing is silently replaced:
* intents the span's evidence does not allow walk down the downgrade ladder;
* the story outranks the voice: a single-subject intent is widened when the
  span REQUIRES several participants / actions to stay visible;
* targets must be visible subjects / actions of the span;
* restraint: limited punches per 10 s, no punch on very short spans;
* repair directives from a failed QC (at most one round) widen named spans.
"""
from __future__ import annotations

from typing import Any

from mimir.config import Settings, section
from mimir.core.stage import StageContext, StageOutput
from mimir.edit.intents import DOWNGRADE, SINGLE_SUBJECT, Intent

MIN_PUNCH_SECONDS = 0.6


def _default_target(span: dict[str, Any], intent: Intent) -> str:
    if intent in (Intent.SPEAKER_MEDIUM, Intent.SPEAKER_PUNCH):
        face = next((v for v in span["visible"] if v["speaker"] and v["speaker"] == span["dominant_speaker"]), None)
        if face is None and len(span["visible"]) == 1:
            face = span["visible"][0]
        return face["id"] if face else ""
    if intent is Intent.REACTION:
        others = [v for v in span["visible"] if v["speaker"] != span["dominant_speaker"]]
        pool = others or span["visible"]
        return max(pool, key=lambda v: v["size"])["id"] if pool else ""
    if intent is Intent.ACTION_REGION:
        return max(span["actions"], key=lambda a: a["intensity"])["id"] if span["actions"] else ""
    return ""


def validate_plan(context: dict[str, Any], plan: dict[str, Any], settings: Settings) -> dict[str, Any]:
    fps = context["fps"]
    camera = settings.camera
    repair = settings.repair
    decisions = plan["decisions"]
    rows: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []
    punches: list[float] = []

    def correct(span_id: str, code: str, before: str, after: str, detail: str = "") -> None:
        corrections.append({"span": span_id, "code": code, "from": before, "to": after, "detail": detail})

    for index, span in enumerate(context["spans"]):
        raw = decisions.get(span["id"])
        if raw is None:
            intent = Intent.WIDE_CONTEXT if index == 0 else Intent.HOLD
            correct(span["id"], "missing_decision", "-", intent.value)
            raw = {"intent": intent.value, "target_id": "", "intensity": "normal", "reason": "not decided"}
        intent = Intent(raw["intent"])
        original = intent
        if index == 0 and intent is Intent.HOLD:
            intent = Intent.WIDE_CONTEXT
            correct(span["id"], "hold_needs_a_previous_framing", original.value, intent.value)
        while intent.value not in span["allowed"] and intent is not Intent.WIDE_CONTEXT:
            lower = DOWNGRADE[intent]
            correct(span["id"], "evidence_does_not_allow", intent.value, lower.value)
            intent = lower
        required_subjects = [r for r in span["required"] if r["kind"] == "subject"]
        if intent in SINGLE_SUBJECT and len(required_subjects) >= 2:
            wider = Intent.TWO_SHOT if Intent.TWO_SHOT.value in span["allowed"] else Intent.WIDE_CONTEXT
            correct(span["id"], "story_outranks_voice", intent.value, wider.value,
                    "several participants carry this moment")
            intent = wider
        target = raw.get("target_id", "")
        valid_targets = {v["id"] for v in span["visible"]} | {a["id"] for a in span["actions"]}
        if intent in SINGLE_SUBJECT | {Intent.ACTION_REGION}:
            if target not in valid_targets:
                fixed = _default_target(span, intent)
                if fixed and not target:
                    # the director may leave the subject to the evidence: a resolution, not a disagreement
                    resolved.append({"span": span["id"], "intent": intent.value, "target": fixed})
                    target = fixed
                elif fixed:
                    correct(span["id"], "target_not_visible", target, fixed)
                    target = fixed
                else:
                    correct(span["id"], "no_target_evidence", intent.value, Intent.WIDE_CONTEXT.value)
                    intent, target = Intent.WIDE_CONTEXT, ""
        else:
            target = target if target in valid_targets else ""
        seconds = (span["frames"][1] - span["frames"][0]) / fps
        if intent is Intent.SPEAKER_PUNCH:
            start = span["frames"][0] / fps
            recent = [t for t in punches if start - t < 10.0]
            if seconds < MIN_PUNCH_SECONDS or len(recent) >= camera.max_punches_per_10s:
                correct(span["id"], "punch_restraint", intent.value, Intent.SPEAKER_MEDIUM.value,
                        f"{len(recent)} punches in the last 10 s / span {seconds:.2f}s")
                intent = Intent.SPEAKER_MEDIUM
            else:
                punches.append(start)
        if span["id"] in repair.widen_spans or (repair.conservative_camera and intent in SINGLE_SUBJECT):
            wider = Intent.TWO_SHOT if (intent in SINGLE_SUBJECT and Intent.TWO_SHOT.value in span["allowed"]) \
                else Intent.WIDE_CONTEXT
            if wider is not intent:
                correct(span["id"], "qc_repair_widen", intent.value, wider.value)
                intent, target = wider, ""
        rows.append({"id": span["id"], "intent": intent.value, "target_id": target,
                     "intensity": raw.get("intensity", "normal"), "reason": raw.get("reason", ""),
                     "director_intent": original.value})
    return {"spans": rows, "corrections": corrections, "resolved_targets": resolved,
            "correction_ratio": round(len({c["span"] for c in corrections}) / max(1, len(rows)), 3)}


class EditValidationStage:
    name = "edit_validation"
    version = 1
    deps = ("edit_context", "edit_direction")

    def params(self, settings: Settings) -> Any:
        camera = section(settings, "camera")
        return {"max_punches_per_10s": camera["max_punches_per_10s"], "repair": section(settings, "repair")}

    def run(self, ctx: StageContext) -> StageOutput:
        context = ctx.dep("edit_context").json("edit_context")
        plan = ctx.dep("edit_direction").json("edit_plan")
        validated = validate_plan(context, plan, ctx.settings)
        for row in validated["corrections"]:
            ctx.ledger.info("plan_correction", f"{row['span']}: {row['code']} {row['from']} -> {row['to']}")
        if validated["correction_ratio"] > 0.5:
            ctx.ledger.warning("director_disagrees_with_evidence",
                               f"{validated['correction_ratio']:.0%} of spans needed correction")
        return StageOutput(data={"validated_plan": validated})
