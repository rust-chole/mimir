"""Stage ``edit_direction``: the AI Edit Director chooses editorial intent per span.

It answers "what must the viewer see right now?" with one intent from a small
vocabulary, an optional target id from the span's evidence and an intensity.
It never returns coordinates, zoom values or commands.
"""
from __future__ import annotations

import json
from typing import Any

from mimir.config import Settings, routes_for
from mimir.core.stage import StageContext, StageOutput
from mimir.edit.intents import INTENSITIES, Intent
from mimir.errors import StageError

INSTRUCTIONS = """
You are MIMIR's edit director for a vertical 9:16 Short. For every span decide WHAT THE VIEWER MUST SEE RIGHT NOW.
You choose editorial intent only; deterministic code computes every crop, zoom and movement.

Intents:
- HOLD: keep the current framing (a deliberate non-move; restraint is professional).
- WIDE_CONTEXT: everything story-relevant visible (establishing, groups, unclear action).
- TWO_SHOT: two or more participants together (exchanges, a reaction to someone else's line).
- SPEAKER_MEDIUM: the active speaker, shoulders and hands.
- SPEAKER_PUNCH: a quick push onto the speaker for a peak line (rare, only on real peaks).
- REACTION: the person reacting (payoff / aftermath).
- ACTION_REGION: where the physical action or important object is.
- GAMEPLAY_PRIORITY: gameplay owns the frame (the facecam stays visible).
- SCREEN_PRIORITY: on-screen UI/text owns the frame and must stay readable.

Rules:
- Choose ONLY from the span's ALLOWED intents; they encode the available evidence.
- The story outranks the voice: never isolate one speaker when REQUIRED items would leave the frame.
- Setup establishes context; escalation may tighten; payoff shows the action/peak clearly; reaction shows who
  reacts. The cold open should be the strongest, clearest view of the peak.
- No rhythmic zooming: at most a couple of SPEAKER_PUNCH per 10 seconds, only on peaks.
- target_id must be one of the span's visible subject ids or action ids, or empty.
Return a decision for EVERY span.
""".strip()


def span_lines(context: dict[str, Any]) -> str:
    lines = []
    for span in context["spans"]:
        visible = ", ".join(f"{v['id']}(speaker={v['speaker'] or '?'} link={v['link']:.2f} size={v['size']:.2f} "
                            f"{v['position']})" for v in span["visible"]) or "none"
        actions = ", ".join(f"{a['id']}(intensity={a['intensity']:.1f})" for a in span["actions"]) or "none"
        required = "; ".join(f"{r['kind']}:{r['id']}" for r in span["required"]) or "none"
        elements = "; ".join(f"{e['kind']}:{e['description'][:50]}({e['importance']})" for e in span["elements"]) or "none"
        lines.append(
            f"{span['id']} [{span['seconds'][0]:.2f}-{span['seconds'][1]:.2f}s] {span['segment']}/{span['role']} "
            f"shot={span['shot']} talk={json.dumps(span['talk'])} dominant={span['dominant_speaker'] or '-'}\n"
            f"  visible: {visible}\n  actions: {actions}\n  elements: {elements}\n  REQUIRED: {required}\n"
            f"  ALLOWED: {', '.join(span['allowed'])}\n  words: {span['text'] or '(none)'}")
    return "\n".join(lines)


class EditDirectionStage:
    name = "edit_direction"
    version = 1
    deps = ("edit_context",)

    def params(self, settings: Settings) -> Any:
        return {"routes": routes_for(settings, "edit_director")}

    def run(self, ctx: StageContext) -> StageOutput:
        context = ctx.dep("edit_context").json("edit_context")
        span_ids = [s["id"] for s in context["spans"]]
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "decisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "span_id": {"type": "string", "enum": span_ids},
                            "intent": {"type": "string", "enum": [i.value for i in Intent]},
                            "target_id": {"type": "string"},
                            "intensity": {"type": "string", "enum": list(INTENSITIES)},
                            "reason": {"type": "string"},
                        },
                        "required": ["span_id", "intent", "target_id", "intensity", "reason"],
                    },
                },
                "notes": {"type": "string"},
            },
            "required": ["decisions", "notes"],
        }
        text = (f"STORY: {context['story']['title']} ({context['story']['emotion']}): {context['story']['reason']}\n"
                f"LAYOUT: {context['layout']['class']} - {context['layout']['reason']}\n"
                f"SPEAKER MODE: {context['speaker_mode']}\n\nSPANS:\n{span_lines(context)}")
        raw = ctx.provider.json_task("edit_director", ctx.settings.route("edit_director"), instructions=INSTRUCTIONS,
                                     input_text=text, schema=schema, schema_name="mimir_edit_director_v1")
        decisions = {}
        for row in raw.get("decisions", []):
            if row.get("span_id") in span_ids and row.get("intent") in {i.value for i in Intent}:
                decisions[row["span_id"]] = {"intent": row["intent"], "target_id": str(row.get("target_id", "")),
                                             "intensity": row.get("intensity", "normal"),
                                             "reason": " ".join(str(row.get("reason", "")).split())[:240]}
        coverage = len(decisions) / max(1, len(span_ids))
        if coverage < 0.5:
            raise StageError(self.name, f"edit director covered only {coverage:.0%} of spans")
        if coverage < 1.0:
            ctx.ledger.warning("director_missing_spans", "director skipped spans; validation will HOLD them",
                               missing=[s for s in span_ids if s not in decisions])
        return StageOutput(data={"edit_plan": {"decisions": decisions, "notes": raw.get("notes", ""),
                                               "model": ctx.settings.route("edit_director").model}})
