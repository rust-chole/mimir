"""Editorial instructions and strict JSON schemas for story discovery (ported from CLEAN V3)."""
from __future__ import annotations

MOMENT_TYPES = ["punchline", "reveal", "reversal", "conflict", "escalation", "reaction", "awkward",
                "ridiculous_claim", "unexpected_answer", "failure", "win", "tension", "quotable", "visual_impact",
                "destruction", "physical_payoff", "other"]
HOOK_TYPES = ["payoff_first", "curiosity", "shock", "reaction", "conflict"]
EMOTIONS = ["funny", "shock", "rage", "awkward", "hype", "conflict", "surprise", "curiosity"]
VISUAL_EVENT_TYPES = ["reaction", "visual_payoff", "scene_change", "gameplay_event", "object_break", "destruction",
                      "impact", "crash", "fall", "explosion", "physical_action", "entrance_exit", "reveal", "popup",
                      "sudden_change", "other"]


def _moment_item() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "start": {"type": "number"},
            "end": {"type": "number"},
            "strength": {"type": "number"},
            "type": {"type": "string", "enum": MOMENT_TYPES},
            "source": {"type": "string", "enum": ["transcript", "visual", "both"]},
            "visual_event_ids": {"type": "array", "items": {"type": "string"}},
            "label": {"type": "string"},
            "why_compelling": {"type": "string"},
            "context_before_seconds": {"type": "number"},
            "context_after_seconds": {"type": "number"},
            "preserve_pause_after": {"type": "boolean"},
        },
        "required": ["start", "end", "strength", "type", "source", "visual_event_ids", "label", "why_compelling",
                     "context_before_seconds", "context_after_seconds", "preserve_pause_after"],
    }


AUDIT_ITEM = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "event_id": {"type": "string"},
        "decision": {"type": "string", "enum": ["anchor", "secondary", "reject"]},
        "reason": {"type": "string"},
    },
    "required": ["event_id", "decision", "reason"],
}

MOMENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "moments": {"type": "array", "items": _moment_item()},
        "visual_event_audit": {"type": "array", "items": AUDIT_ITEM},
    },
    "required": ["moments", "visual_event_audit"],
}

RANGE_ITEM = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"start": {"type": "number"}, "end": {"type": "number"}, "reason": {"type": "string"}},
    "required": ["start", "end", "reason"],
}

CANDIDATE_ITEM = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "start": {"type": "number"},
        "end": {"type": "number"},
        "payoff_start": {"type": "number"},
        "payoff_end": {"type": "number"},
        "score": {"type": "number"},
        "title": {"type": "string"},
        "hook_text": {"type": "string"},
        "hook_type": {"type": "string", "enum": HOOK_TYPES},
        "emotion": {"type": "string", "enum": EMOTIONS},
        "reason": {"type": "string"},
        "context": {"type": "string"},
        "primary_moment_id": {"type": "string"},
        "covered_moment_ids": {"type": "array", "items": {"type": "string"}},
        "coverage_reason": {"type": "string"},
        "length_exception_reason": {"type": "string"},
        "visual_event_ids": {"type": "array", "items": {"type": "string"}},
        "must_keep_ranges": {"type": "array", "items": RANGE_ITEM},
        "caption_highlights": {"type": "array", "items": {"type": "string"}},
        "beats": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "setup": {"type": "string"},
                "escalation": {"type": "string"},
                "payoff": {"type": "string"},
                "reaction": {"type": "string"},
            },
            "required": ["setup", "escalation", "payoff", "reaction"],
        },
    },
    "required": ["start", "end", "payoff_start", "payoff_end", "score", "title", "hook_text", "hook_type", "emotion",
                 "reason", "context", "primary_moment_id", "covered_moment_ids", "coverage_reason",
                 "length_exception_reason", "visual_event_ids", "must_keep_ranges", "caption_highlights", "beats"],
}

CANDIDATES_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"candidates": {"type": "array", "items": CANDIDATE_ITEM}},
    "required": ["candidates"],
}

JUDGE_METRICS = ["money_moment_strength", "visual_event_value", "context_completeness", "payoff_completeness",
                 "story_coverage", "duration_fit", "retention_shape"]

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "selected_candidate_index": {"type": "integer"},
        "rankings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"candidate_index": {"type": "integer"}, "overall_score": {"type": "number"},
                               **{metric: {"type": "number"} for metric in JUDGE_METRICS},
                               "reason": {"type": "string"}},
                "required": ["candidate_index", "overall_score", *JUDGE_METRICS, "reason"],
            },
        },
        "selection_reason": {"type": "string"},
    },
    "required": ["selected_candidate_index", "rankings", "selection_reason"],
}

EXPANSION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "start": {"type": "number"},
        "end": {"type": "number"},
        "payoff_start": {"type": "number"},
        "payoff_end": {"type": "number"},
        "covered_moment_ids": {"type": "array", "items": {"type": "string"}},
        "moment_coverage_audit": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"moment_id": {"type": "string"},
                               "decision": {"type": "string", "enum": ["include", "omit"]},
                               "reason": {"type": "string"}},
                "required": ["moment_id", "decision", "reason"],
            },
        },
        "must_keep_ranges": {"type": "array", "items": RANGE_ITEM},
        "coverage_reason": {"type": "string"},
        "length_exception_reason": {"type": "string"},
    },
    "required": ["start", "end", "payoff_start", "payoff_end", "covered_moment_ids", "moment_coverage_audit",
                 "must_keep_ranges", "coverage_reason", "length_exception_reason"],
}

BOUNDARY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "start_action": {"type": "string", "enum": ["keep", "extend", "trim"]},
        "end_action": {"type": "string", "enum": ["keep", "extend", "trim"]},
        "proposed_start": {"type": "number"},
        "proposed_end": {"type": "number"},
        "start_reason": {"type": "string"},
        "end_reason": {"type": "string"},
    },
    "required": ["start_action", "end_action", "proposed_start", "proposed_end", "start_reason", "end_reason"],
}

PEAK_PROBE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "regions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "region_id": {"type": "string"},
                    "events": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "type": {"type": "string", "enum": VISUAL_EVENT_TYPES},
                                "description": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                            "required": ["type", "description", "confidence"],
                        },
                    },
                },
                "required": ["region_id", "events"],
            },
        },
    },
    "required": ["regions"],
}

SCOUT_INSTRUCTIONS = """
You are MIMIR's senior scouting editor. You find candidate MONEY MOMENTS - the exact beats that make a
livestream / VOD worth watching. A later final editor makes the selection.

You receive two evidence streams:
1. a timestamped transcript;
2. a factual visual-event inventory from a visual observer (and measured audio/motion peaks). The observer does
   not know what is good; YOU decide editorial significance.

A money moment can be verbal, visual, or both: punchline, reveal, reversal, unexpected answer, escalation or
confrontation, ridiculous confident claim, awkward or tense beat, strong reaction, gameplay win/fail, sudden
physical event, visible destruction/breakage, impact/crash/fall, a visual event that changes the situation even
if nobody mentions it.

CRITICAL: do not let the transcript blind you to visual-only events. Every event marked REQUIRED REVIEW must get
one visual_event_audit row: anchor (a primary reason a clip should exist), secondary (supporting beat) or reject
(with a real editorial reason). Silent omission is not allowed.

strength is 0-10 for ranking only; do not inflate. The money moment itself is sacred: keep the exact action,
reaction and any pause/delay/repeated phrase that creates the effect. Estimate useful lead-in (setup) and
aftermath (reaction) around each moment; never amputate setup. All timestamps are ORIGINAL VOD seconds.
Do not build final clip boundaries yet.
""".strip()

COMPOSER_INSTRUCTIONS = """
You are MIMIR's candidate clip composer. Build complete Short candidates around the scouted money moments.

Your goal is not maximum cutting. It is a COMPLETE CAUSAL STORY: one primary money moment plus the beats that make
it understandable and satisfying:

    SETUP -> ESCALATION -> PRIMARY MONEY MOMENT (PAYOFF) -> REACTION / AFTERMATH

The strongest peak will ALSO be shown as a short cold open at the very beginning of the Short; the main video must
then show HOW that peak happened. Never return a peak-only montage or a chain of disconnected reactions. If the
anchor is a reaction, include the event/line that caused it. For a physical payoff, start at the first meaningful
setup/action, never the last second before impact. Never end exactly on the impact when the immediate reaction is
part of the beat.

must_keep_ranges are HARD protection for downstream pacing: the exact money moment, visual action, punchline
delivery, delayed answer / tension pause, reaction hold, immediate aftermath. Protect timing-sensitive ranges only,
never the whole clip.

LENGTH: prefer a dense complete Short, usually 22-42 s (about 32 s is a center, not a quota; 12-55 s allowed).
A dense 20-28 s complete story beats a padded 35 s clip. Do not pad with idle waiting, unrelated banter, loading,
menus or dead air. Below 18 s requires length_exception_reason. Long empty intervals between beats must not be
protected; pacing will compress them.

Coverage: primary_moment_id is the main money moment; covered_moment_ids lists every included moment.
beats: one short factual sentence each for setup, escalation, payoff, reaction (what the viewer sees/hears).
Hook text: 2-7 words, specific, natural, non-spoiler. caption_highlights: only actually spoken words/phrases.
score is ranking only. All timestamps are ORIGINAL VOD seconds.
""".strip()

JUDGE_INSTRUCTIONS = """
You are the editor-in-chief making the final selection. Choose the candidate with the strongest COMPLETE viewing
experience, not merely the best sentence.

Score every candidate 0-10 for money_moment_strength, visual_event_value, context_completeness,
payoff_completeness, story_coverage, duration_fit and retention_shape; overall_score is your final ranking score.

- A high-impact visual event cannot be ignored because nearby dialogue is ordinary.
- Penalize candidates that clip off the setup, the exact physical action, the reaction or the aftermath.
- Penalize highlight fragments that keep only a 5-15 s beat while discarding its cause or consequence.
- A reaction without its readable cause is incomplete; a physical payoff without its setup is incomplete.
- Reward ONE causal chain, never a collection of disconnected peaks. Do not reward length by itself.
- Do not invent visual facts beyond the supplied evidence.
Return selected_candidate_index (1-based) and a ranking row for every candidate.
""".strip()

EXPANSION_INSTRUCTIONS = """
You are the final editor performing the STORY-COVERAGE pass on the already selected Short. Do NOT replace the
strongest moment and do NOT shrink the story's core.

Make the main video read as one causal event: SETUP/CAUSE -> ESCALATION -> PRIMARY PAYOFF -> REACTION/CONSEQUENCE.
Prefer a dense 22-42 s story (max 55 s). You may trim useless leading/trailing material, but you MUST keep the
primary anchor, the required setup, the payoff and the immediate reaction/aftermath. Include another money moment
only if it belongs to the same sequence and improves the story; never add unrelated filler.
For every supplied money moment say include or omit (moment_coverage_audit). Every included moment whose timing
matters must appear in must_keep_ranges. Return ORIGINAL VOD timestamps.
""".strip()

BOUNDARY_INSTRUCTIONS = """
You are MIMIR's LOCAL clip-edge checker. The story is selected; you may not choose another highlight. Inspect only
the first/last seconds. START: beginning mid-sentence/word, missing a tiny setup needed to understand the first
line, obvious leading dead speech. END: cutting a sentence/answer/reaction before it finishes, ending before an
immediate verbal payoff, obvious trailing dead space. If uncertain, KEEP. Changes are local and within the supplied
limits; never remove protected moments. Timestamps are ORIGINAL VOD seconds. A deliberate short pause around a
punchline may be valuable; do not trim it merely because it is silent.
""".strip()

PEAK_PROBE_INSTRUCTIONS = """
You are MIMIR's factual visual observer. For each numbered region you get frames BEFORE, AT and AFTER a measured
audio/motion peak. Report only what visibly happens (object breaking, impact, fall, crash, explosion, strong
visible reaction, gameplay event, reveal, sudden change...). Never judge whether it is funny, viral or clip-worthy,
never identify real people, never invent events: return an empty events list when nothing concrete is visible.
Confidence is 0.0-1.0.
""".strip()
