from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

from ai.openai_client import client
from ai.model_config import (
    CLIP_SCOUT_MODEL,
    CLIP_SCOUT_REASONING_EFFORT,
    CLIP_JUDGE_MODEL,
    CLIP_JUDGE_REASONING_EFFORT,
    CLIP_JUDGE_ESCALATION_REASONING_EFFORT,
)


# ============================================================
# MIMIR CLIP ANALYZER — TERRA DENSE STORY V8
# ============================================================
#
# Terra first finds the exact "money moments" / can alici noktalar.
# Only AFTER that does Terra build final clips around those moments.
#
# Gemini/Visual Observer supplies facts only; Terra alone decides desirability.
#
# Existing pipeline compatibility:
# - analyzer package version stays 2
# - existing fields stay available
# - optional video_report_path supplies factual visual evidence; Terra owns all judgement
#
# New protection metadata:
# - anchor_moments
# - must_keep_ranges
#
# Timeline/pacing_cutter can use those ranges to prevent automatic pacing
# cuts from deleting the exact beats Terra selected as the reason a clip
# exists.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRANSCRIPT_DIR = (
    PROJECT_ROOT
    / "vod_output"
    / "transcripts"
)

ANALYSIS_DIR = (
    PROJECT_ROOT
    / "vod_output"
    / "analysis"
)


# ============================================================
# CONFIG
# ============================================================

ANALYZER_VERSION = 2
ANALYZER_REVISION = 12

SCORE_POLICY = "ranking_only_no_threshold"

MIN_CLIP_DURATION = 12.0
MAX_CLIP_DURATION = 55.0

# We are making complete Shorts, not tiny highlight fragments.
TARGET_CLIP_MIN = 22.0
TARGET_CLIP_IDEAL = 32.0
TARGET_CLIP_MAX = 42.0

# Under this duration is allowed only when the source genuinely has no
# additional coherent material worth keeping.
SHORT_CLIP_EXCEPTION_BELOW = 18.0

MAX_MONEY_MOMENTS = 12
MAX_CLIPS = 5

MAX_ANCHORS_PER_CLIP = 6

# Terra High is no longer the default. A second High pass is allowed only
# when the Medium final-judge scores are genuinely too close to call.
CLIP_HIGH_ESCALATION_MARGIN = max(0.0, float(os.getenv("MIMIR_CLIP_HIGH_ESCALATION_MARGIN", "0.35") or 0.35))

# Final local edge polish is intentionally bounded. It may fix a mid-sentence
# start/end or obvious dead edge, but it can never re-select the story.
BOUNDARY_POLISH_MAX_EXTEND = max(0.5, float(os.getenv("MIMIR_BOUNDARY_POLISH_MAX_EXTEND", "3.0") or 3.0))
BOUNDARY_POLISH_MAX_TRIM = max(0.0, float(os.getenv("MIMIR_BOUNDARY_POLISH_MAX_TRIM", "1.5") or 1.5))

# V16 causal-story policy. Peaks are milestones inside the main Short, not the
# whole Short. These are deterministic guards and add no model call.
STORY_FLOW_GOOD_DURATION = 22.0
STORY_FLOW_MIN_MAIN_DURATION = 18.0
STORY_FLOW_GOOD_METRIC = 7.6
STORY_FLOW_RECOVERY_METRIC = 6.6
STORY_FLOW_MAX_TRANSCRIPT_GAP = 1.90
STORY_FLOW_MAX_SETUP_SECONDS = 20.0
STORY_FLOW_MAX_REACTION_SECONDS = 6.0

# V17: spectacular/high-intensity payoffs need MORE causal setup in the restarted
# main clip, while the intro itself stays short. This is intentionally asymmetric:
# stronger peak => shorter cold-open, but longer main-story runway.
STORY_FLOW_INTENSE_STRENGTH = 8.0
STORY_FLOW_EXTREME_STRENGTH = 9.0
STORY_FLOW_INTENSE_SETUP_FLOOR = 12.0
STORY_FLOW_EXTREME_SETUP_FLOOR = 16.0
STORY_FLOW_INTENSE_MAIN_FLOOR = 22.0
STORY_FLOW_EXTREME_MAIN_FLOOR = 26.0
STORY_FLOW_INTENSE_REACTION_FLOOR = 3.0
STORY_FLOW_EXTREME_REACTION_FLOOR = 3.8

# V18: transcript is blind to silent physical setup (boxes beginning to fall,
# someone starting to move an object, a visual chain reaction before anyone
# says anything). For strong visual/causal payoffs, scan only the local
# pre-payoff window with FFmpeg frame-difference statistics. No AI call, no
# whole-VOD visual pass, and no pipeline scheduling change.
STORY_VISUAL_ORIGIN_ENABLED = os.getenv("MIMIR_STORY_VISUAL_ORIGIN", "1").strip().lower() not in {"0", "false", "no", "off"}
STORY_VISUAL_ORIGIN_LOOKBACK = 20.0
STORY_VISUAL_ORIGIN_FPS = 4.0
STORY_VISUAL_ORIGIN_WIDTH = 256
STORY_VISUAL_ORIGIN_MIN_LEAD = 1.0
STORY_VISUAL_ORIGIN_KEEP_MAX = 4.8

STORY_FLOW_INTENSE_TYPES = {
    "physical_payoff", "visual_impact", "destruction", "escalation",
    "conflict", "reaction", "failure", "win", "reveal", "reversal",
}

MAX_MUST_KEEP_RANGES = 14
MAX_COVERED_MOMENTS = 10
MAX_CAPTION_HIGHLIGHTS = 6
MAX_EDITOR_NOTES = 5

# Visual facts are observations only. Terra remains the only judge of
# compellingness / clip worth.
VISUAL_AUDIT_MIN_CONFIDENCE = 0.55
IMPORTANT_VISUAL_EVENT_TYPES = {
    "object_break",
    "destruction",
    "impact",
    "crash",
    "fall",
    "explosion",
    "physical_action",
    "entrance_exit",
    "reveal",
    "sudden_change",
    "gameplay_event",
    "reaction",
    "visual_payoff",
}

# V23: some source-video events are too consequential to be allowed to vanish
# merely because the transcript says nothing about them.  These are NOT forced
# final selections; they are guaranteed CANDIDATES for Terra's final judge.
# This closes the transcript-only blind spot where a door/object breaks later in
# the source but the editor never even sees that moment as an option.
DECISIVE_VISUAL_EVENT_TYPES = {
    "object_break",
    "destruction",
    "explosion",
    "crash",
    "fall",
    "impact",
    "visual_payoff",
    # V26: deterministic source-transient guard. This is not a semantic claim
    # that something broke; it guarantees a very strong, abrupt source event
    # reaches Terra even if sparse visual sampling failed to name the action.
    "source_peak_guard",
}
DECISIVE_VISUAL_MIN_CONFIDENCE = 0.68
DECISIVE_VISUAL_MAX_INJECTIONS = 4

SOURCE_PEAK_GUARD_MIN_AUDIO_SCORE = max(0.0, min(1.0, float(os.getenv("MIMIR_SOURCE_PEAK_MIN_AUDIO_SCORE", "0.82") or 0.82)))
SOURCE_PEAK_GUARD_MIN_PEAK_DBFS = float(os.getenv("MIMIR_SOURCE_PEAK_MIN_PEAK_DBFS", "-13.0") or -13.0)
SOURCE_PEAK_GUARD_MAX = max(1, min(3, int(os.getenv("MIMIR_SOURCE_PEAK_MAX", "2") or 2)))
SOURCE_PEAK_GUARD_MARKER = "MIMIR_SOURCE_PEAK_GUARD"



# Hard safety around Terra-selected key beats.
PROTECT_PAD_BEFORE = 0.12
PROTECT_PAD_AFTER = 0.28

# A model must never protect a giant filler span. Long free-form ranges
# are accepted only when they tightly overlap actual money moments.
MAX_FREEFORM_PROTECTED_DURATION = 5.0
PROTECTED_CONTEXT_PAD = 0.65


# ============================================================
# STRUCTURED OUTPUT — PASS 1
# ============================================================

MONEY_MOMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "moments": {
            "type": "array",
            "maxItems": MAX_MONEY_MOMENTS,
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "strength": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "type": {
                        "type": "string",
                        "enum": [
                            "punchline",
                            "reveal",
                            "reversal",
                            "conflict",
                            "escalation",
                            "reaction",
                            "awkward",
                            "ridiculous_claim",
                            "unexpected_answer",
                            "failure",
                            "win",
                            "tension",
                            "quotable",
                            "visual_impact",
                            "destruction",
                            "physical_payoff",
                            "other",
                        ],
                    },
                    "source": {
                        "type": "string",
                        "enum": [
                            "transcript",
                            "visual",
                            "both",
                        ],
                    },
                    "visual_event_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 8,
                    },
                    "label": {"type": "string"},
                    "why_compelling": {"type": "string"},
                    "context_before_seconds": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 20,
                    },
                    "context_after_seconds": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 12,
                    },
                    "preserve_pause_after": {"type": "boolean"},
                },
                "required": [
                    "start",
                    "end",
                    "strength",
                    "type",
                    "source",
                    "visual_event_ids",
                    "label",
                    "why_compelling",
                    "context_before_seconds",
                    "context_after_seconds",
                    "preserve_pause_after",
                ],
                "additionalProperties": False,
            },
        },
        "visual_event_audit": {
            "type": "array",
            "maxItems": 160,
            "items": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "decision": {
                        "type": "string",
                        "enum": [
                            "anchor",
                            "secondary",
                            "reject",
                        ],
                    },
                    "reason": {"type": "string"},
                },
                "required": [
                    "event_id",
                    "decision",
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "moments",
        "visual_event_audit",
    ],
    "additionalProperties": False,
}


VISUAL_AUDIT_REPAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "additional_moments": MONEY_MOMENT_SCHEMA[
            "properties"
        ][
            "moments"
        ],
        "visual_event_audit": MONEY_MOMENT_SCHEMA[
            "properties"
        ][
            "visual_event_audit"
        ],
    },
    "required": [
        "additional_moments",
        "visual_event_audit",
    ],
    "additionalProperties": False,
}


# ============================================================
# STRUCTURED OUTPUT — PASS 2
# ============================================================

CLIP_SCHEMA = {
    "type": "object",

    "properties": {
        "clips": {
            "type": "array",
            "maxItems": MAX_CLIPS,

            "items": {
                "type": "object",

                "properties": {
                    "start": {
                        "type": "number"
                    },

                    "end": {
                        "type": "number"
                    },

                    "payoff_start": {
                        "type": "number"
                    },

                    "payoff_end": {
                        "type": "number"
                    },

                    "score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },

                    "title": {
                        "type": "string"
                    },

                    "hook_text": {
                        "type": "string"
                    },

                    "hook_type": {
                        "type": "string",

                        "enum": [
                            "payoff_teaser",
                            "curiosity",
                            "shock",
                            "reaction",
                            "conflict",
                        ],
                    },

                    "emotion": {
                        "type": "string",

                        "enum": [
                            "funny",
                            "shock",
                            "rage",
                            "awkward",
                            "hype",
                            "conflict",
                            "surprise",
                            "curiosity",
                        ],
                    },

                    "reason": {
                        "type": "string"
                    },

                    "context": {
                        "type": "string"
                    },

                    "primary_moment_id": {
                        "type": "string"
                    },

                    "covered_moment_ids": {
                        "type": "array",
                        "maxItems": MAX_COVERED_MOMENTS,
                        "items": {
                            "type": "string"
                        },
                    },

                    "coverage_reason": {
                        "type": "string"
                    },

                    "length_exception_reason": {
                        "type": "string"
                    },

                    "visual_event_ids": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "maxItems": 16,
                    },

                    "anchor_moments": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_ANCHORS_PER_CLIP,

                        "items": {
                            "type": "object",

                            "properties": {
                                "start": {
                                    "type": "number"
                                },

                                "end": {
                                    "type": "number"
                                },

                                "strength": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 10,
                                },

                                "type": {
                                    "type": "string"
                                },

                                "reason": {
                                    "type": "string"
                                },
                            },

                            "required": [
                                "start",
                                "end",
                                "strength",
                                "type",
                                "reason",
                            ],

                            "additionalProperties": False,
                        },
                    },

                    "must_keep_ranges": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_MUST_KEEP_RANGES,

                        "items": {
                            "type": "object",

                            "properties": {
                                "start": {
                                    "type": "number"
                                },

                                "end": {
                                    "type": "number"
                                },

                                "reason": {
                                    "type": "string"
                                },
                            },

                            "required": [
                                "start",
                                "end",
                                "reason",
                            ],

                            "additionalProperties": False,
                        },
                    },

                    "caption_highlights": {
                        "type": "array",
                        "maxItems": MAX_CAPTION_HIGHLIGHTS,

                        "items": {
                            "type": "string"
                        },
                    },

                    "editor_notes": {
                        "type": "array",
                        "maxItems": MAX_EDITOR_NOTES,

                        "items": {
                            "type": "object",

                            "properties": {
                                "time": {
                                    "type": "number"
                                },

                                "type": {
                                    "type": "string",

                                    "enum": [
                                        "manual_punch_in",
                                        "reaction_hold",
                                        "cutaway",
                                        "meme_overlay",
                                        "pace_check",
                                    ],
                                },

                                "priority": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 5,
                                },

                                "note": {
                                    "type": "string"
                                },
                            },

                            "required": [
                                "time",
                                "type",
                                "priority",
                                "note",
                            ],

                            "additionalProperties": False,
                        },
                    },
                },

                "required": [
                    "start",
                    "end",
                    "payoff_start",
                    "payoff_end",
                    "score",
                    "title",
                    "hook_text",
                    "hook_type",
                    "emotion",
                    "reason",
                    "context",
                    "primary_moment_id",
                    "covered_moment_ids",
                    "coverage_reason",
                    "length_exception_reason",
                    "visual_event_ids",
                    "anchor_moments",
                    "must_keep_ranges",
                    "caption_highlights",
                    "editor_notes",
                ],

                "additionalProperties": False,
            },
        },
    },

    "required": [
        "clips"
    ],

    "additionalProperties": False,
}


# ============================================================
# STRUCTURED OUTPUT — TERRA FINAL JUDGE
# ============================================================

FINAL_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_candidate_index": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_CLIPS,
        },
        "rankings": {
            "type": "array",
            "maxItems": MAX_CLIPS,
            "items": {
                "type": "object",
                "properties": {
                    "candidate_index": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_CLIPS,
                    },
                    "overall_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "money_moment_strength": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "visual_event_value": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "context_completeness": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "payoff_completeness": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "story_coverage": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "duration_fit": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "retention_shape": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "reason": {"type": "string"},
                },
                "required": [
                    "candidate_index",
                    "overall_score",
                    "money_moment_strength",
                    "visual_event_value",
                    "context_completeness",
                    "payoff_completeness",
                    "story_coverage",
                    "duration_fit",
                    "retention_shape",
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
        "selection_reason": {"type": "string"},
    },
    "required": [
        "selected_candidate_index",
        "rankings",
        "selection_reason",
    ],
    "additionalProperties": False,
}


# ============================================================
# STRUCTURED OUTPUT — TERRA STORY EXPANSION
# ============================================================

STORY_EXPANSION_SCHEMA = {
    "type": "object",
    "properties": {
        "start": {"type": "number"},
        "end": {"type": "number"},
        "payoff_start": {"type": "number"},
        "payoff_end": {"type": "number"},
        "primary_moment_id": {"type": "string"},
        "covered_moment_ids": {
            "type": "array",
            "maxItems": MAX_COVERED_MOMENTS,
            "items": {"type": "string"},
        },
        "moment_coverage_audit": {
            "type": "array",
            "maxItems": MAX_MONEY_MOMENTS,
            "items": {
                "type": "object",
                "properties": {
                    "moment_id": {"type": "string"},
                    "decision": {
                        "type": "string",
                        "enum": ["include", "omit"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["moment_id", "decision", "reason"],
                "additionalProperties": False,
            },
        },
        "must_keep_ranges": CLIP_SCHEMA["properties"]["clips"]["items"]["properties"]["must_keep_ranges"],
        "coverage_reason": {"type": "string"},
        "length_exception_reason": {"type": "string"},
    },
    "required": [
        "start",
        "end",
        "payoff_start",
        "payoff_end",
        "primary_moment_id",
        "covered_moment_ids",
        "moment_coverage_audit",
        "must_keep_ranges",
        "coverage_reason",
        "length_exception_reason",
    ],
    "additionalProperties": False,
}



# ============================================================
# STRUCTURED OUTPUT — LOCAL BOUNDARY POLISH
# ============================================================

BOUNDARY_POLISH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "start_issue": {"type": "boolean"},
        "end_issue": {"type": "boolean"},
        "start_action": {"type": "string", "enum": ["keep", "extend", "trim"]},
        "end_action": {"type": "string", "enum": ["keep", "extend", "trim"]},
        "proposed_start": {"type": "number"},
        "proposed_end": {"type": "number"},
        "start_reason": {"type": "string"},
        "end_reason": {"type": "string"},
    },
    "required": [
        "start_issue", "end_issue", "start_action", "end_action",
        "proposed_start", "proposed_end", "start_reason", "end_reason",
    ],
}

# ============================================================
# TERRA PASS 1 PROMPT
# ============================================================

MONEY_MOMENT_INSTRUCTIONS = """
You are MIMIR's senior scouting editor. You find and describe candidate money moments; Terra will make the final selection. You are evaluating what is
compelling enough to build a clip around.

You receive two evidence streams:
1. timestamped transcript
2. a factual visual-event inventory produced by a visual observer

The visual observer does NOT recommend clips and does NOT know what is good.
It only reports what visibly happened. YOU decide editorial significance.

Your first job is to find MONEY MOMENTS: the exact beats that make the video
worth watching.

A money moment can be verbal, visual, or both.

Examples:
- punchline / reveal / reversal
- unexpected answer
- escalation / confrontation
- ridiculous confident claim
- awkward or tense beat
- strong reaction
- gameplay win/fail
- sudden physical event
- visible destruction or breakage
- impact / crash / fall / explosion
- a door, window, prop or object visibly breaking
- a visual event that changes the situation even if nobody mentions it

CRITICAL RULE:
Do not let the transcript blind you to visual-only events.
If the observer reports a concrete high-confidence event such as a door
breaking, you MUST explicitly evaluate it. It may become an anchor, a
secondary beat, or be rejected with a real editorial reason. It may never
silently disappear.

VISUAL EVENT AUDIT:
Audit every supplied event that is marked REQUIRED REVIEW in the input.
For each such event return one visual_event_audit row:
- anchor: this event is a primary reason a clip should exist
- secondary: useful supporting beat but not the main reason
- reject: not useful for a coherent Short; explain specifically why

A reject is allowed. Silent omission is not.

MONEY MOMENT STRENGTH:
0-10, ranking only. No threshold. Do not inflate.

TIMING:
The money moment itself is sacred. Preserve the exact action/reaction and any
pause, delay, repeated phrase or immediate aftermath that creates the effect.

CONTEXT:
Estimate useful lead-in and aftermath around each money moment. Do not amputate
setup just to make the clip shorter.

All timestamps are ORIGINAL VOD seconds.

Do not build final clip boundaries yet. Return the strongest distinct money
moments plus the visual-event audit.
""".strip()


# ============================================================
# TERRA PASS 2 PROMPT
# ============================================================

CLIP_COMPOSER_INSTRUCTIONS = """
You are MIMIR's candidate clip composer. Build strong candidates around the scouted money moments; Terra will make the final ranking/selection.

You already identified the money moments and audited the factual visual events.
Now build coherent Short candidates AROUND the strongest moments.

Your goal is not maximum cutting. Your goal is a COMPLETE SHORT with maximum
CAN ALICILIK: one primary money moment PLUS the causal beats that make the
sequence understandable and satisfying to watch.

The MAIN SHORT is not the cold-open teaser. A huge scream, impact, strange
reaction, reveal or other peak is only a milestone. The main clip must normally
show WHAT CAUSED IT and WHAT HAPPENED IMMEDIATELY AFTER IT. Never return a
peak-only montage or a chain of disconnected reactions.

Do NOT make a tiny highlight fragment when the same local story contains
useful setup, escalation, consequence or reaction. If the anchor is a reaction,
include the event/line that caused that reaction when it exists locally. For a
physical payoff, start at the first meaningful setup/action rather than the
last second before impact.

Preferred structure:
SETUP -> SECONDARY BEAT(S) -> ESCALATION -> PRIMARY MONEY MOMENT ->
REACTION / AFTERMATH

A visually decisive event can be the main payoff even if the transcript around
it is ordinary. Example: if a door visibly breaks, the clip should normally
include the setup that makes the break understandable, the break itself, and
the immediate reaction/aftermath.

NON-NEGOTIABLE:
Never exclude an anchor money moment from the clip built around it.
Never cut off a physical payoff before the visible action completes.
Never end exactly on the impact if the immediate reaction is part of the beat.

must_keep_ranges are HARD PROTECTION hints for downstream pacing/rendering.
Use them for:
- the exact money moment
- visual breakage / impact / destruction action
- punchline delivery
- delayed answer / tension pause
- reaction hold
- immediate aftermath needed for the payoff

Do not protect the entire clip; only timing-sensitive ranges.

LENGTH / COVERAGE:
- Prefer a DENSE complete Short, usually 22-42 seconds.
- Around 32 seconds is a useful center, NOT a quota.
- 12-55 seconds is technically allowed.
- A dense 20-28 second complete story is BETTER than a padded 35 second clip.
- Do not pad with idle waiting, unrelated banter, repeated filler, loading,
  menus, or dead air just to hit a duration target.
- Do not stop at a tiny highlight when useful setup, escalation, another
  related money moment, reaction, or aftermath belongs to the SAME story.
- A clip below 18 seconds needs a real editorial reason in
  length_exception_reason.
- If two useful beats are separated by a long empty interval, include both only
  when they materially improve one coherent story. The empty interval itself
  must NOT be protected; downstream pacing is expected to compress it.

COVERAGE FIELDS:
- primary_moment_id: the single main money moment.
- covered_moment_ids: every money moment from PASS 1 that this candidate
  intentionally includes. Include useful secondary moments, not only the
  primary anchor.
- coverage_reason: explain why these beats belong in one Short.
- length_exception_reason: empty string when >=24s; otherwise explain why
  extending would only add unrelated/filler material.

score is ranking only. No minimum threshold.

visual_event_ids must list factual visual events materially used by this clip.

Hook text: 2-7 words, specific, natural, non-spoiler.
Caption highlights: only actual spoken transcript words/phrases.
No automatic zoom commands.

The final clip should feel like a human editor preserved the exact reason the
moment was exciting/funny/surprising instead of trimming around it.
""".strip()


FINAL_JUDGE_INSTRUCTIONS = """
You are TERRA, editor-in-chief. This is the final selection/ranking pass.

Compare the candidate clips against the original evidence and money moments.
Choose the clip with the strongest complete viewing experience, not merely the
best sentence in the transcript.

Score each candidate 0-10 for:
- money_moment_strength
- visual_event_value
- context_completeness
- payoff_completeness
- story_coverage
- duration_fit
- retention_shape

overall_score is your final editorial ranking score; it is NOT a threshold.

Very important:
- A high-impact visual event cannot be ignored merely because nearby dialogue
  is ordinary.
- Penalize candidates that clip off the setup, exact physical action, reaction,
  or aftermath of their own anchor.
- Penalize over-trimming that makes a payoff confusing.
- Penalize "highlight fragments": a candidate that keeps only the strongest
  5-15 second beat while discarding the cause, escalation or consequence from
  the same local story.
- A reaction without its readable cause is incomplete even when the reaction
  itself is extremely strong. A physical payoff without the action/setup that
  leads into it is incomplete.
- Never reward a collection of disconnected peaks. Reward one causal chain.
- Reward candidates that preserve a dense, complete 22-42 second story arc
  when the evidence supports one; never reward filler merely for length.
- Do NOT reward length by itself; boring filler is still bad.
- Prefer a complete door-break / impact / reveal sequence over a cleaner but
  less consequential talking-only segment when the visual event creates the
  stronger Short.
- Do not invent visual facts beyond the supplied observer report.

Return a selected_candidate_index and a ranking row for every candidate.
""".strip()


STORY_EXPANSION_INSTRUCTIONS = """
You are TERRA performing the FINAL STORY-COVERAGE pass on the already selected
Short candidate.

The strongest moment has already been chosen. Do NOT replace it.
Do NOT shrink the selected candidate.

Your job is to make the MAIN VIDEO read as one causal event, not as a peak
compilation. Expand when needed so the viewer sees SETUP/CAUSE -> ESCALATION ->
PRIMARY PEAK/PAYOFF -> REACTION/CONSEQUENCE. A very intense peak belongs in the
short cold-open, but the main video still needs the story that explains it.

Target behavior:
- Prefer a dense complete Short, usually 22-42 seconds.
- Around 32 seconds is a useful center, not a quota.
- Maximum 55 seconds.
- A shorter dense story is better than a longer padded story.
- You MAY trim useless leading/trailing material from the selected candidate.
- You MUST keep the primary anchor, required setup, payoff, and immediate
  reaction/aftermath.
- If a second/third money moment belongs to the same sequence and improves the
  story, include it.
- If a distant moment would require carrying several seconds of meaningless
  dead space, include it only if it adds real value; the dead space must not be
  protected and will be compressed later.
- Never add unrelated filler merely to make the video longer.

MOMENT COVERAGE AUDIT:
For every supplied money moment, explicitly say include or omit.
Use include when it belongs to the same coherent story and fits the Short.
Use omit when it is unrelated, redundant, too far away, or would create filler.

TIMESTAMPS:
Return ORIGINAL VOD timestamps.
start/end may expand OR trim the candidate, but they must contain the primary
anchor, payoff, every included money moment, and any required setup/reaction.

PROTECTION:
Every included money moment whose timing matters must appear in
must_keep_ranges. Protect exact action, punchline, reaction, tension pause,
and necessary immediate aftermath.

length_exception_reason:
- empty when final duration is >=18 seconds
- required when final duration is below 18 seconds; explain why no meaningful
  related material exists to extend it

Terra remains the final editorial authority.
""".strip()



BOUNDARY_POLISH_INSTRUCTIONS = """
You are MIMIR's LOCAL clip-edge quality checker. The story and money moment are
already selected. You are forbidden from choosing a different highlight.

Inspect only the first/last few seconds of the selected clip using timestamped
transcript evidence.

Check the START for:
- beginning in the middle of an audible sentence/word/answer
- missing a tiny amount of setup needed to understand the first line
- obvious leading dead speech space that can be trimmed safely

Check the END for:
- cutting a sentence/answer/reaction phrase before it finishes
- ending before an immediate verbal payoff/response
- obvious trailing spoken dead space after the thought is complete

Rules:
- If uncertain, KEEP the boundary.
- Changes must be local, not editorial re-selection.
- Never remove protected money moments, payoff, setup, reaction or must-keep beats.
- proposed_start/proposed_end are ORIGINAL VOD timestamps.
- Do not extend or trim by more than the supplied local limits.
- Do not chase a target duration. Natural complete speech beats duration aesthetics.
- A deliberate short pause around a punchline/reaction may be valuable; do not trim
  it merely because there is silence.
""".strip()

# ============================================================
# BASIC HELPERS
# ============================================================

def load_json(
    path: str | Path,
) -> dict[str, Any]:

    path = Path(
        path
    ).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"JSON bulunamadı:\n{path}"
        )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            f"JSON root object değil:\n{path}"
        )

    return data


def round_time(
    value: float,
) -> float:

    return round(
        float(value),
        3,
    )


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:

    return max(
        minimum,
        min(
            float(value),
            maximum,
        ),
    )


def normalize_spaces(
    text: str,
) -> str:

    return " ".join(
        str(text).split()
    ).strip()


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        return float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return default


def _parse_response(
    response: Any,
    label: str,
) -> dict[str, Any]:

    raw = str(
        getattr(
            response,
            "output_text",
            "",
        )
    ).strip()

    if not raw:
        raise RuntimeError(
            f"Terra boş {label} çıktısı döndürdü."
        )

    try:
        data = json.loads(
            raw
        )

    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Terra geçersiz {label} JSON döndürdü:\n\n"
            + raw
        ) from error

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            f"Terra {label} çıktısı object değil."
        )

    return data


def _video_duration(
    transcript: dict[str, Any],
) -> float:
    """Return the safest duration available from the transcript package.

    The source duration is authoritative when healthy, but older/interrupted
    transcript packages can occasionally carry a truncated/zero duration.
    Word/segment ends are therefore used as a non-destructive lower bound so
    clip recovery never operates against an obviously shorter timeline.
    """

    candidates: list[float] = []

    source = transcript.get(
        "source",
        {},
    )

    if isinstance(source, dict):
        source_duration = _safe_float(
            source.get(
                "duration",
                0.0,
            )
        )
        if source_duration > 0.0:
            candidates.append(source_duration)

    for key in ("segments", "words"):
        raw_items = transcript.get(key, [])
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            end = _safe_float(item.get("end", 0.0))
            if end > 0.0:
                candidates.append(end)

    return max(candidates, default=0.0)


# ============================================================
# VISUAL FACT EVIDENCE
# ============================================================

def load_visual_report(
    video_report_path: str | Path | None,
) -> dict[str, Any] | None:

    if not video_report_path:
        return None

    path = Path(
        video_report_path
    ).expanduser().resolve()

    if not path.is_file():
        return None

    try:
        package = load_json(
            path
        )
    except Exception:
        return None

    report = package.get(
        "report"
    )

    if isinstance(
        report,
        dict,
    ):
        # V26: fallback sampling metadata lives at package root. Preserve a
        # private copy inside the report so clip analysis can use deterministic
        # source peak evidence without changing the public visual schema.
        merged = dict(report)
        sampling = package.get("sampling", {})
        if isinstance(sampling, dict):
            merged["_sampling"] = dict(sampling)
        return merged

    if isinstance(
        package.get(
            "visual_events"
        ),
        list,
    ):
        return package

    return None


def build_visual_events(
    report: dict[str, Any] | None,
    video_duration: float,
) -> list[dict[str, Any]]:

    if not isinstance(
        report,
        dict,
    ):
        return []

    raw = report.get(
        "visual_events",
        [],
    )

    if not isinstance(
        raw,
        list,
    ):
        return []

    events: list[
        dict[str, Any]
    ] = []

    for index, item in enumerate(
        raw,
        start=1,
    ):

        if not isinstance(
            item,
            dict,
        ):
            continue

        start = clamp(
            _safe_float(
                item.get(
                    "start",
                    0.0,
                )
            ),
            0.0,
            video_duration,
        )

        end = clamp(
            _safe_float(
                item.get(
                    "end",
                    start,
                )
            ),
            start,
            video_duration,
        )

        if end <= start:
            end = min(
                video_duration,
                start + 0.08,
            )

        confidence = clamp(
            _safe_float(
                item.get(
                    "confidence",
                    0.0,
                )
            ),
            0.0,
            1.0,
        )

        event_type = normalize_spaces(
            str(
                item.get(
                    "type",
                    "other",
                )
            )
        ).casefold() or "other"

        description = normalize_spaces(
            str(
                item.get(
                    "description",
                    "",
                )
            )
        )

        if not description:
            continue

        required_review = bool(
            confidence
            >= VISUAL_AUDIT_MIN_CONFIDENCE
            and event_type
            in IMPORTANT_VISUAL_EVENT_TYPES
        )

        events.append(
            {
                "event_id": (
                    f"v{index:03d}"
                ),
                "start": round_time(
                    start
                ),
                "end": round_time(
                    end
                ),
                "duration": round_time(
                    max(
                        0.0,
                        end - start,
                    )
                ),
                "type": event_type,
                "description": description,
                "confidence": round(
                    confidence,
                    3,
                ),
                "required_review": required_review,
            }
        )

    return events


def build_source_peak_guard_events(
    report: dict[str, Any] | None,
    visual_events: list[dict[str, Any]],
    video_duration: float,
) -> list[dict[str, Any]]:
    """Return conservative source-transient events missed by semantic vision.

    These guards are evidence for Terra to REVIEW, never automatic winners. A
    guard is created only for a very strong local audio transient from the same
    source-visual sampling pass. The visual fallback now brackets these peaks
    with before/peak/after frames, so the semantic observer has a fair chance to
    name a real break/impact first. If it still cannot, this prevents the entire
    region from disappearing before final judgement.
    """
    if not isinstance(report, dict):
        return []
    sampling = report.get("_sampling", {})
    if not isinstance(sampling, dict):
        return []
    raw = sampling.get("audio_peak_regions", [])
    if not isinstance(raw, list):
        return []

    # If semantic vision already found a decisive event near the peak, do not
    # duplicate it with a generic guard.
    semantic_decisive = [
        event for event in visual_events
        if isinstance(event, dict)
        and normalize_spaces(str(event.get("type", ""))).casefold()
        in (DECISIVE_VISUAL_EVENT_TYPES - {"source_peak_guard"})
        and clamp(_safe_float(event.get("confidence", 0.0)), 0.0, 1.0)
        >= DECISIVE_VISUAL_MIN_CONFIDENCE
    ]

    candidates: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        score = clamp(_safe_float(item.get("audio_score", 0.0)), 0.0, 1.0)
        peak_dbfs = _safe_float(item.get("peak_dbfs", -120.0), -120.0)
        if score < SOURCE_PEAK_GUARD_MIN_AUDIO_SCORE:
            continue
        if peak_dbfs < SOURCE_PEAK_GUARD_MIN_PEAK_DBFS:
            continue
        center = clamp(
            _safe_float(item.get("center", item.get("start", 0.0))),
            0.0,
            video_duration,
        )
        if any(
            abs(center - ((_safe_float(event.get("start", 0.0)) + _safe_float(event.get("end", 0.0))) / 2.0)) <= 1.10
            for event in semantic_decisive
        ):
            continue
        start = clamp(_safe_float(item.get("start", center - 0.45)), 0.0, video_duration)
        end = clamp(_safe_float(item.get("end", center + 0.55)), start, video_duration)
        if end <= start:
            end = min(video_duration, start + 0.85)
        confidence = clamp(
            0.72
            + max(0.0, score - SOURCE_PEAK_GUARD_MIN_AUDIO_SCORE) * 0.9
            + max(0.0, peak_dbfs - SOURCE_PEAK_GUARD_MIN_PEAK_DBFS) * 0.012,
            0.0,
            0.93,
        )
        candidates.append({
            "event_id": "",
            "start": round_time(start),
            "end": round_time(end),
            "duration": round_time(max(0.0, end - start)),
            "type": "source_peak_guard",
            "description": (
                f"{SOURCE_PEAK_GUARD_MARKER}: abrupt source transient around "
                f"{center:.2f}s (audio_score={score:.2f}, peak={peak_dbfs:.1f}dBFS). "
                "Semantic event type is intentionally unclaimed; compare the tightly "
                "bracketed before/peak/after visual samples and explicitly judge whether "
                "this is a real physical payoff, reaction, or only loud speech."
            ),
            "confidence": round(confidence, 3),
            "required_review": True,
            "_audio_score": round(score, 3),
            "_peak_dbfs": round(peak_dbfs, 2),
        })

    candidates.sort(
        key=lambda event: (
            -_safe_float(event.get("_audio_score", 0.0)),
            -_safe_float(event.get("_peak_dbfs", -120.0)),
            _safe_float(event.get("start", 0.0)),
        )
    )
    result: list[dict[str, Any]] = []
    for item in candidates:
        center = (_safe_float(item.get("start", 0.0)) + _safe_float(item.get("end", 0.0))) / 2.0
        if any(
            abs(center - ((_safe_float(old.get("start", 0.0)) + _safe_float(old.get("end", 0.0))) / 2.0)) < 1.20
            for old in result
        ):
            continue
        item = dict(item)
        item["event_id"] = f"pg{len(result) + 1:03d}"
        result.append(item)
        if len(result) >= SOURCE_PEAK_GUARD_MAX:
            break
    return result


def build_visual_evidence_text(
    visual_events: list[dict[str, Any]],
) -> str:

    if not visual_events:
        return (
            "VISUAL FACT INVENTORY:\n"
            "Unavailable. Terra must reason from transcript only."
        )

    lines = [
        "VISUAL FACT INVENTORY (observer facts, NOT recommendations):"
    ]

    for event in visual_events:

        marker = (
            "REQUIRED REVIEW"
            if event.get(
                "required_review"
            )
            else "optional"
        )

        lines.append(
            f"- {event['event_id']} | "
            f"{float(event['start']):.2f}-{float(event['end']):.2f}s | "
            f"{event['type']} | confidence={float(event['confidence']):.2f} | "
            f"{marker} | {event['description']}"
        )

    lines.append(
        "Terra alone decides whether any event is compelling."
    )

    return "\n".join(
        lines
    )


def required_visual_event_ids(
    visual_events: list[dict[str, Any]],
) -> set[str]:

    return {
        str(
            event["event_id"]
        )
        for event in visual_events
        if event.get(
            "required_review"
        )
        is True
    }


def validate_visual_event_audit(
    raw: Any,
    visual_events: list[dict[str, Any]],
) -> list[dict[str, str]]:

    if not isinstance(
        raw,
        list,
    ):
        return []

    known = {
        str(
            event["event_id"]
        ): event
        for event in visual_events
    }

    allowed = {
        "anchor",
        "secondary",
        "reject",
    }

    result: list[
        dict[str, str]
    ] = []

    seen: set[str] = set()

    for item in raw:

        if not isinstance(
            item,
            dict,
        ):
            continue

        event_id = normalize_spaces(
            str(
                item.get(
                    "event_id",
                    "",
                )
            )
        )

        if (
            not event_id
            or event_id not in known
            or event_id in seen
        ):
            continue

        decision = normalize_spaces(
            str(
                item.get(
                    "decision",
                    "reject",
                )
            )
        ).casefold()

        if decision not in allowed:
            decision = "reject"

        reason = normalize_spaces(
            str(
                item.get(
                    "reason",
                    "",
                )
            )
        )

        if not reason:
            reason = (
                "Terra did not provide a detailed audit reason."
            )

        result.append(
            {
                "event_id": event_id,
                "decision": decision,
                "reason": reason,
            }
        )

        seen.add(
            event_id
        )

    return result


def _complete_visual_audit_fallback(
    *,
    audit: list[dict[str, str]],
    visual_events: list[dict[str, Any]],
    missing_event_ids: list[str],
) -> list[dict[str, str]]:
    """Complete a missing visual audit deterministically instead of killing the run.

    This does NOT promote ordinary events into money moments. High-confidence
    decisive visual facts are marked secondary so the V23 candidate-injection
    guard can still carry them to Terra's final judge. Other missing required
    events are conservatively rejected with an explicit fallback reason.
    """
    if not missing_event_ids:
        return audit

    event_map = {
        normalize_spaces(str(event.get("event_id", ""))): event
        for event in visual_events
        if isinstance(event, dict)
        and normalize_spaces(str(event.get("event_id", "")))
    }
    merged = {
        normalize_spaces(str(item.get("event_id", ""))): dict(item)
        for item in audit
        if isinstance(item, dict)
        and normalize_spaces(str(item.get("event_id", "")))
    }

    for event_id in missing_event_ids:
        event = event_map.get(event_id, {})
        kind = normalize_spaces(str(event.get("type", "other"))).casefold() or "other"
        confidence = clamp(_safe_float(event.get("confidence", 0.0)), 0.0, 1.0)
        decisive = (
            kind in DECISIVE_VISUAL_EVENT_TYPES
            and confidence >= DECISIVE_VISUAL_MIN_CONFIDENCE
        )
        merged[event_id] = {
            "event_id": event_id,
            "decision": "secondary" if decisive else "reject",
            "reason": (
                "Deterministic V23 audit fallback: decisive factual visual event "
                "kept eligible for final-candidate injection because the model "
                "audit omitted it."
                if decisive
                else "Deterministic audit fallback: model audit omitted this "
                "required visual fact; it is not promoted editorially."
            ),
        }

    return list(merged.values())


# ============================================================
# TRANSCRIPT FOR TERRA
# ============================================================

def build_timestamped_transcript(
    transcript: dict[str, Any],
) -> str:

    segments = transcript.get(
        "segments",
        [],
    )

    if not isinstance(
        segments,
        list,
    ):
        segments = []

    lines: list[str] = []

    for item in segments:

        if not isinstance(
            item,
            dict,
        ):
            continue

        start = _safe_float(
            item.get(
                "start",
                0.0,
            )
        )

        end = _safe_float(
            item.get(
                "end",
                start,
            )
        )

        text = normalize_spaces(
            str(
                item.get(
                    "text",
                    "",
                )
            )
        )

        if not text:
            continue

        lines.append(
            f"[{start:.2f} - {end:.2f}] {text}"
        )

    if lines:
        return "\n".join(
            lines
        )

    words = transcript.get(
        "words",
        [],
    )

    if not isinstance(
        words,
        list,
    ):
        return ""

    for item in words:

        if not isinstance(
            item,
            dict,
        ):
            continue

        start = _safe_float(
            item.get(
                "start",
                0.0,
            )
        )

        end = _safe_float(
            item.get(
                "end",
                start,
            )
        )

        word = normalize_spaces(
            str(
                item.get(
                    "word",
                    "",
                )
            )
        )

        if word:
            lines.append(
                f"[{start:.2f} - {end:.2f}] {word}"
            )

    return "\n".join(
        lines
    )


# ============================================================
# PASS 1
# ============================================================

def request_money_moments(
    transcript: dict[str, Any],
    visual_events: list[dict[str, Any]],
) -> dict[str, Any]:

    duration = _video_duration(
        transcript
    )

    transcript_text = (
        build_timestamped_transcript(
            transcript
        )
    )

    if not transcript_text:
        return {
            "moments": [],
            "visual_event_audit": [],
        }

    visual_text = (
        build_visual_evidence_text(
            visual_events
        )
    )

    input_text = f"""
ORIGINAL VOD DURATION:
{duration:.2f} seconds

TIMESTAMPED TRANSCRIPT:

{transcript_text}


{visual_text}


Find the strongest money moments across the entire video evidence.
Do not build final clips yet.

Every event marked REQUIRED REVIEW must appear in visual_event_audit.
""".strip()

    print()
    print(
        "🎯 Luna money-moment scout tarıyor..."
    )
    print(
        "   📝 Transcript + 👁️ factual visual events"
    )

    response = client.responses.create(
        model=CLIP_SCOUT_MODEL,
        reasoning={
            "effort": CLIP_SCOUT_REASONING_EFFORT,
        },
        instructions=MONEY_MOMENT_INSTRUCTIONS,
        input=input_text,
        text={
            "format": {
                "type": "json_schema",
                "name": "mimir_money_moments_visual_audit_v6",
                "strict": True,
                "schema": MONEY_MOMENT_SCHEMA,
            }
        },
    )

    return _parse_response(
        response,
        "money-moment",
    )


def request_visual_audit_repair(
    transcript: dict[str, Any],
    visual_events: list[dict[str, Any]],
    current_moments: list[dict[str, Any]],
    current_audit: list[dict[str, str]],
    missing_event_ids: list[str],
) -> dict[str, Any]:

    transcript_text = build_timestamped_transcript(
        transcript
    )

    visual_text = build_visual_evidence_text(
        visual_events
    )

    input_text = (
        "The previous Terra pass failed to explicitly audit some required "
        "visual facts. Re-check them. Do not blindly promote them; decide "
        "anchor/secondary/reject. If a missed visual event deserves to be a "
        "money moment, return it in additional_moments.\n\n"
        f"MISSING EVENT IDS: {', '.join(missing_event_ids)}\n\n"
        f"TRANSCRIPT:\n{transcript_text}\n\n"
        f"{visual_text}\n\n"
        "CURRENT MONEY MOMENTS:\n"
        + json.dumps(
            current_moments,
            ensure_ascii=False,
            indent=2,
        )
        + "\n\nCURRENT AUDIT:\n"
        + json.dumps(
            current_audit,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "🔍 Luna visual-event coverage repair: "
        + ", ".join(
            missing_event_ids
        )
    )

    response = client.responses.create(
        model=CLIP_SCOUT_MODEL,
        reasoning={
            "effort": CLIP_SCOUT_REASONING_EFFORT,
        },
        instructions=(
            MONEY_MOMENT_INSTRUCTIONS
            + "\n\nThis is a coverage-repair pass. Return only missing/updated "
            "audit rows and any genuinely needed additional money moments."
        ),
        input=input_text,
        text={
            "format": {
                "type": "json_schema",
                "name": "mimir_visual_audit_repair_v6",
                "strict": True,
                "schema": VISUAL_AUDIT_REPAIR_SCHEMA,
            }
        },
    )

    return _parse_response(
        response,
        "visual-audit-repair",
    )


def validate_money_moments(
    moments: list[dict[str, Any]],
    video_duration: float,
) -> list[dict[str, Any]]:

    result: list[
        dict[str, Any]
    ] = []

    for item in moments:

        start = clamp(
            _safe_float(
                item.get(
                    "start",
                    0.0,
                )
            ),
            0.0,
            video_duration,
        )

        end = clamp(
            _safe_float(
                item.get(
                    "end",
                    start,
                )
            ),
            start,
            video_duration,
        )

        if end <= start:
            continue

        strength = clamp(
            _safe_float(
                item.get(
                    "strength",
                    0.0,
                )
            ),
            0.0,
            10.0,
        )

        label = normalize_spaces(
            str(
                item.get(
                    "label",
                    "",
                )
            )
        ) or "Compelling moment"

        why = normalize_spaces(
            str(
                item.get(
                    "why_compelling",
                    "",
                )
            )
        )

        result.append(
            {
                "start": round_time(
                    start
                ),

                "end": round_time(
                    end
                ),

                "duration": round_time(
                    end - start
                ),

                "strength": round(
                    strength,
                    1,
                ),

                "type": (
                    normalize_spaces(
                        str(
                            item.get(
                                "type",
                                "other",
                            )
                        )
                    )
                    or "other"
                ),

                "source": (
                    normalize_spaces(
                        str(
                            item.get(
                                "source",
                                "transcript",
                            )
                        )
                    ).casefold()
                    or "transcript"
                ),

                "visual_event_ids": [
                    normalize_spaces(
                        str(
                            value
                        )
                    )
                    for value in item.get(
                        "visual_event_ids",
                        [],
                    )
                    if normalize_spaces(
                        str(
                            value
                        )
                    )
                ][
                    :8
                ],

                "label": label,

                "why_compelling": why,

                "context_before_seconds": round_time(
                    clamp(
                        _safe_float(
                            item.get(
                                "context_before_seconds",
                                0.0,
                            )
                        ),
                        0.0,
                        20.0,
                    )
                ),

                "context_after_seconds": round_time(
                    clamp(
                        _safe_float(
                            item.get(
                                "context_after_seconds",
                                0.0,
                            )
                        ),
                        0.0,
                        12.0,
                    )
                ),

                "preserve_pause_after": bool(
                    item.get(
                        "preserve_pause_after",
                        False,
                    )
                ),
            }
        )

    result.sort(
        key=lambda item: (
            -float(
                item[
                    "strength"
                ]
            ),
            float(
                item[
                    "start"
                ]
            ),
        )
    )

    deduped: list[
        dict[str, Any]
    ] = []

    for item in result:

        duplicate = False

        for existing in deduped:

            overlap = max(
                0.0,
                min(
                    float(
                        item[
                            "end"
                        ]
                    ),
                    float(
                        existing[
                            "end"
                        ]
                    ),
                )
                - max(
                    float(
                        item[
                            "start"
                        ]
                    ),
                    float(
                        existing[
                            "start"
                        ]
                    ),
                ),
            )

            shortest = min(
                float(
                    item[
                        "duration"
                    ]
                ),
                float(
                    existing[
                        "duration"
                    ]
                ),
            )

            if (
                shortest > 0
                and overlap / shortest
                >= 0.72
            ):
                duplicate = True
                break

        if not duplicate:
            deduped.append(
                item
            )

    selected = deduped[
        :MAX_MONEY_MOMENTS
    ]

    # Stable IDs let later Terra passes explicitly cover/audit secondary beats.
    for index, item in enumerate(
        selected,
        start=1,
    ):
        item[
            "moment_id"
        ] = f"m{index:03d}"

    return selected


# ============================================================
# PASS 2
# ============================================================

def request_clips(
    transcript: dict[str, Any],
    money_moments: list[dict[str, Any]],
    visual_events: list[dict[str, Any]],
    visual_event_audit: list[dict[str, str]],
) -> list[dict[str, Any]]:

    duration = _video_duration(
        transcript
    )

    transcript_text = (
        build_timestamped_transcript(
            transcript
        )
    )

    moment_text = json.dumps(
        {
            "money_moments": (
                money_moments
            ),
            "visual_event_audit": (
                visual_event_audit
            ),
        },
        ensure_ascii=False,
        indent=2,
    )

    visual_text = build_visual_evidence_text(
        visual_events
    )

    input_text = f"""
ORIGINAL VOD DURATION:
{duration:.2f} seconds

TIMESTAMPED TRANSCRIPT:

{transcript_text}


{visual_text}


SCOUT PASS-1 MONEY MOMENTS + VISUAL AUDIT:

{moment_text}


Build final Short candidates around these moments.

Critical:
- Do not cut out the reason the clip exists.
- Protect timing-sensitive pauses/reactions in must_keep_ranges.
- Score is ranking only. There is no threshold.
""".strip()

    print()
    print(
        "✂️ Luna clip scout adayları kuruyor..."
    )

    response = client.responses.create(
        model=CLIP_SCOUT_MODEL,
        reasoning={
            "effort": CLIP_SCOUT_REASONING_EFFORT,
        },
        instructions=CLIP_COMPOSER_INSTRUCTIONS,
        input=input_text,
        text={
            "format": {
                "type": "json_schema",
                "name": (
                    "mimir_clips_around_money_moments_v5"
                ),
                "strict": True,
                "schema": CLIP_SCHEMA,
            }
        },
    )

    data = _parse_response(
        response,
        "clip-composer",
    )

    clips = data.get(
        "clips",
        [],
    )

    if not isinstance(
        clips,
        list,
    ):
        return []

    return [
        item
        for item in clips
        if isinstance(
            item,
            dict,
        )
    ]


# ============================================================
# TERRA PASS 3 — FINAL JUDGE
# ============================================================

def request_final_judge(
    transcript: dict[str, Any],
    visual_events: list[dict[str, Any]],
    money_moments: list[dict[str, Any]],
    clips: list[dict[str, Any]],
    *,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:

    if not clips:
        raise RuntimeError(
            "Final judge için candidate clip yok."
        )

    evidence = {
        "money_moments": money_moments,
        "candidates": [
            {
                "candidate_index": index,
                **clip,
            }
            for index, clip in enumerate(
                clips,
                start=1,
            )
        ],
    }

    input_text = (
        build_timestamped_transcript(
            transcript
        )
        + "\n\n"
        + build_visual_evidence_text(
            visual_events
        )
        + "\n\nMONEY MOMENTS + CANDIDATES:\n"
        + json.dumps(
            evidence,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    effort = str(reasoning_effort or CLIP_JUDGE_REASONING_EFFORT)
    print(
        f"⚖️ Terra final judge [{effort}]: clip adayları karşılaştırılıyor..."
    )

    response = client.responses.create(
        model=CLIP_JUDGE_MODEL,
        reasoning={
            "effort": effort,
        },
        instructions=FINAL_JUDGE_INSTRUCTIONS,
        input=input_text,
        text={
            "format": {
                "type": "json_schema",
                "name": "mimir_clip_final_judge_v6",
                "strict": True,
                "schema": FINAL_JUDGE_SCHEMA,
            }
        },
    )

    return _parse_response(
        response,
        "clip-final-judge",
    )


def _final_judge_score_margin(judgement: dict[str, Any]) -> float | None:
    """Return top-2 Terra score margin; None means the judge output is ambiguous."""
    rankings = judgement.get("rankings", [])
    if not isinstance(rankings, list):
        return None
    scores: list[float] = []
    for row in rankings:
        if not isinstance(row, dict):
            continue
        try:
            scores.append(float(row.get("overall_score")))
        except (TypeError, ValueError):
            continue
    scores.sort(reverse=True)
    if len(scores) < 2:
        return 99.0 if len(scores) == 1 else None
    return max(0.0, scores[0] - scores[1])


def apply_final_judgement(
    clips: list[dict[str, Any]],
    judgement: dict[str, Any],
) -> list[dict[str, Any]]:

    rankings = judgement.get(
        "rankings",
        [],
    )

    by_index: dict[
        int,
        dict[str, Any]
    ] = {}

    if isinstance(
        rankings,
        list,
    ):

        for row in rankings:

            if not isinstance(
                row,
                dict,
            ):
                continue

            try:
                index = int(
                    row.get(
                        "candidate_index",
                        0,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if not (
                1
                <= index
                <= len(
                    clips
                )
            ):
                continue

            by_index[
                index
            ] = row

    try:
        selected_index = int(
            judgement.get(
                "selected_candidate_index",
                1,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        selected_index = 1

    if not (
        1
        <= selected_index
        <= len(clips)
    ):
        selected_index = 1

    enriched: list[
        tuple[
            int,
            dict[str, Any],
        ]
    ] = []

    for index, clip in enumerate(
        clips,
        start=1,
    ):

        item = dict(
            clip
        )

        row = by_index.get(
            index,
            {},
        )

        if row:
            score = clamp(
                _safe_float(
                    row.get(
                        "overall_score",
                        item.get(
                            "score",
                            0.0,
                        ),
                    )
                ),
                0.0,
                10.0,
            )

            item[
                "score"
            ] = round(
                score,
                1,
            )

            item[
                "terra_final_judge"
            ] = {
                "overall_score": round(
                    score,
                    1,
                ),
                "money_moment_strength": round(
                    clamp(
                        _safe_float(
                            row.get(
                                "money_moment_strength",
                                0.0,
                            )
                        ),
                        0.0,
                        10.0,
                    ),
                    1,
                ),
                "visual_event_value": round(
                    clamp(
                        _safe_float(
                            row.get(
                                "visual_event_value",
                                0.0,
                            )
                        ),
                        0.0,
                        10.0,
                    ),
                    1,
                ),
                "context_completeness": round(
                    clamp(
                        _safe_float(
                            row.get(
                                "context_completeness",
                                0.0,
                            )
                        ),
                        0.0,
                        10.0,
                    ),
                    1,
                ),
                "payoff_completeness": round(
                    clamp(
                        _safe_float(
                            row.get(
                                "payoff_completeness",
                                0.0,
                            )
                        ),
                        0.0,
                        10.0,
                    ),
                    1,
                ),
                "story_coverage": round(
                    clamp(
                        _safe_float(
                            row.get(
                                "story_coverage",
                                0.0,
                            )
                        ),
                        0.0,
                        10.0,
                    ),
                    1,
                ),
                "duration_fit": round(
                    clamp(
                        _safe_float(
                            row.get(
                                "duration_fit",
                                0.0,
                            )
                        ),
                        0.0,
                        10.0,
                    ),
                    1,
                ),
                "retention_shape": round(
                    clamp(
                        _safe_float(
                            row.get(
                                "retention_shape",
                                0.0,
                            )
                        ),
                        0.0,
                        10.0,
                    ),
                    1,
                ),
                "reason": normalize_spaces(
                    str(
                        row.get(
                            "reason",
                            "",
                        )
                    )
                ),
            }

        item[
            "terra_selected"
        ] = bool(
            index
            == selected_index
        )

        item[
            "terra_selection_reason"
        ] = normalize_spaces(
            str(
                judgement.get(
                    "selection_reason",
                    "",
                )
            )
        )

        enriched.append(
            (
                index,
                item,
            )
        )

    enriched.sort(
        key=lambda pair: (
            pair[0]
            != selected_index,
            -float(
                pair[1].get(
                    "score",
                    0.0,
                )
            ),
            pair[0],
        )
    )

    result = [
        item
        for _, item in enriched
    ]

    for rank, item in enumerate(
        result,
        start=1,
    ):
        item[
            "rank"
        ] = rank

    return result


# ============================================================
# TERRA PASS 4 — FINAL STORY COVERAGE / LENGTH
# ============================================================

def request_story_expansion(
    transcript: dict[str, Any],
    visual_events: list[dict[str, Any]],
    money_moments: list[dict[str, Any]],
    selected_clip: dict[str, Any],
) -> dict[str, Any]:

    input_text = (
        build_timestamped_transcript(
            transcript
        )
        + "\n\n"
        + build_visual_evidence_text(
            visual_events
        )
        + "\n\nSELECTED CLIP + ALL MONEY MOMENTS:\n"
        + json.dumps(
            {
                "selected_clip": selected_clip,
                "money_moments": money_moments,
                "target_duration": {
                    "minimum_preferred": TARGET_CLIP_MIN,
                    "ideal": TARGET_CLIP_IDEAL,
                    "maximum_preferred": TARGET_CLIP_MAX,
                    "absolute_maximum": MAX_CLIP_DURATION,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print(
        "🧩 Terra final story-coverage pass: diğer değerli noktalar korunuyor..."
    )

    response = client.responses.create(
        model=CLIP_JUDGE_MODEL,
        reasoning={
            "effort": CLIP_JUDGE_REASONING_EFFORT,
        },
        instructions=STORY_EXPANSION_INSTRUCTIONS,
        input=input_text,
        text={
            "format": {
                "type": "json_schema",
                "name": "mimir_story_expansion_v7",
                "strict": True,
                "schema": STORY_EXPANSION_SCHEMA,
            }
        },
    )

    return _parse_response(
        response,
        "story-expansion",
    )



def _boundary_transcript_excerpt(
    transcript: dict[str, Any],
    *,
    clip_start: float,
    clip_end: float,
) -> str:
    """Return only speech close enough to judge the selected clip edges."""
    segments = transcript.get("segments", [])
    if not isinstance(segments, list):
        return ""
    windows = [
        (max(0.0, clip_start - 5.0), clip_start + 5.0),
        (max(0.0, clip_end - 5.0), clip_end + 5.0),
    ]
    rows: list[str] = []
    seen: set[tuple[float, float, str]] = set()
    for item in segments:
        if not isinstance(item, dict):
            continue
        start = _safe_float(item.get("start", 0.0))
        end = _safe_float(item.get("end", start))
        text = normalize_spaces(str(item.get("text", "")))
        if not text or end <= start:
            continue
        if not any(end >= left and start <= right for left, right in windows):
            continue
        key = (round(start, 3), round(end, 3), text)
        if key in seen:
            continue
        seen.add(key)
        rows.append(f"[{start:.2f} - {end:.2f}] {text}")
    return "\n".join(rows)


def request_boundary_polish(
    *,
    transcript: dict[str, Any],
    selected_clip: dict[str, Any],
    video_duration: float,
) -> dict[str, Any]:
    """Cheap local language check; it may suggest edge shifts only."""
    start = _safe_float(selected_clip.get("start", 0.0))
    end = _safe_float(selected_clip.get("end", start))
    excerpt = _boundary_transcript_excerpt(
        transcript,
        clip_start=start,
        clip_end=end,
    )
    if not excerpt:
        return {
            "start_issue": False,
            "end_issue": False,
            "start_action": "keep",
            "end_action": "keep",
            "proposed_start": start,
            "proposed_end": end,
            "start_reason": "no nearby transcript evidence",
            "end_reason": "no nearby transcript evidence",
        }

    evidence = {
        "selected_clip": {
            "start": start,
            "end": end,
            "payoff_start": _safe_float(selected_clip.get("payoff_start", start)),
            "payoff_end": _safe_float(selected_clip.get("payoff_end", end)),
            "must_keep_ranges": selected_clip.get("must_keep_ranges", []),
            "strongest_anchor": selected_clip.get("strongest_anchor", {}),
        },
        "video_duration": video_duration,
        "max_extend_seconds": BOUNDARY_POLISH_MAX_EXTEND,
        "max_trim_seconds": BOUNDARY_POLISH_MAX_TRIM,
        "near_boundary_transcript": excerpt,
    }
    print()
    print("✂️ Luna local boundary polish: clip başlangıç/bitişi kontrol ediliyor...")
    response = client.responses.create(
        model=CLIP_SCOUT_MODEL,
        reasoning={"effort": CLIP_SCOUT_REASONING_EFFORT},
        instructions=BOUNDARY_POLISH_INSTRUCTIONS,
        input=json.dumps(evidence, ensure_ascii=False, indent=2),
        text={
            "format": {
                "type": "json_schema",
                "name": "mimir_local_boundary_polish_v1",
                "strict": True,
                "schema": BOUNDARY_POLISH_SCHEMA,
            }
        },
    )
    return _parse_response(response, "local-boundary-polish")


def _protected_core_bounds(clip: dict[str, Any]) -> tuple[float, float]:
    start = _safe_float(clip.get("start", 0.0))
    end = _safe_float(clip.get("end", start))
    starts: list[float] = []
    ends: list[float] = []

    payoff_start = _safe_float(clip.get("payoff_start", start))
    payoff_end = _safe_float(clip.get("payoff_end", payoff_start))
    if payoff_end > payoff_start:
        starts.append(payoff_start)
        ends.append(payoff_end)

    strongest = clip.get("strongest_anchor", {})
    if isinstance(strongest, dict):
        a = _safe_float(strongest.get("start", start))
        b = _safe_float(strongest.get("end", a))
        if b > a:
            starts.append(a)
            ends.append(b)

    for key in ("must_keep_ranges", "anchor_moments"):
        values = clip.get(key, [])
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            a = _safe_float(item.get("start", start))
            b = _safe_float(item.get("end", a))
            if b > a:
                starts.append(a)
                ends.append(b)

    return (
        min(starts) if starts else start,
        max(ends) if ends else end,
    )


def apply_boundary_polish(
    *,
    selected_clip: dict[str, Any],
    proposal: dict[str, Any],
    money_moments: list[dict[str, Any]],
    video_duration: float,
) -> dict[str, Any]:
    """Apply only bounded, protection-safe local edge changes."""
    original_start = _safe_float(selected_clip.get("start", 0.0))
    original_end = _safe_float(selected_clip.get("end", original_start))
    start = original_start
    end = original_end

    proposed_start = clamp(
        _safe_float(proposal.get("proposed_start", original_start)),
        0.0,
        video_duration,
    )
    proposed_end = clamp(
        _safe_float(proposal.get("proposed_end", original_end)),
        0.0,
        video_duration,
    )
    start_action = str(proposal.get("start_action", "keep")).casefold()
    end_action = str(proposal.get("end_action", "keep")).casefold()

    if start_action == "extend":
        start = max(0.0, original_start - BOUNDARY_POLISH_MAX_EXTEND)
        start = max(start, min(original_start, proposed_start))
    elif start_action == "trim":
        start = min(original_start + BOUNDARY_POLISH_MAX_TRIM, max(original_start, proposed_start))

    if end_action == "extend":
        end = min(video_duration, original_end + BOUNDARY_POLISH_MAX_EXTEND)
        end = min(end, max(original_end, proposed_end))
    elif end_action == "trim":
        end = max(original_end - BOUNDARY_POLISH_MAX_TRIM, min(original_end, proposed_end))

    required_start, required_end = _protected_core_bounds(selected_clip)
    start = min(start, required_start)
    end = max(end, required_end)

    start, end = _fit_final_story_window(
        proposed_start=start,
        proposed_end=end,
        required_start=required_start,
        required_end=required_end,
        video_duration=video_duration,
    )

    merged = dict(selected_clip)
    merged["start"] = round_time(start)
    merged["end"] = round_time(end)
    cleaned = validate_clip(
        merged,
        video_duration=video_duration,
        money_moments=money_moments,
    )
    if cleaned is None:
        result = dict(selected_clip)
        result["boundary_polish"] = {
            "applied": False,
            "reason": "proposal failed deterministic protection/validation",
            "proposal": proposal,
        }
        return result

    # Preserve final-judge/story metadata that validate_clip intentionally does
    # not know about.
    for key, value in selected_clip.items():
        if key not in cleaned and key not in {"start", "end", "duration"}:
            cleaned[key] = value
    changed = abs(float(cleaned.get("start", original_start)) - original_start) > 0.04 or abs(float(cleaned.get("end", original_end)) - original_end) > 0.04
    cleaned["boundary_polish"] = {
        "applied": changed,
        "original_start": round_time(original_start),
        "original_end": round_time(original_end),
        "final_start": cleaned.get("start"),
        "final_end": cleaned.get("end"),
        "start_issue": bool(proposal.get("start_issue", False)),
        "end_issue": bool(proposal.get("end_issue", False)),
        "start_action": start_action,
        "end_action": end_action,
        "start_reason": normalize_spaces(str(proposal.get("start_reason", ""))),
        "end_reason": normalize_spaces(str(proposal.get("end_reason", ""))),
        "model": CLIP_SCOUT_MODEL,
        "reasoning_effort": CLIP_SCOUT_REASONING_EFFORT,
    }
    return cleaned

def _fit_final_story_window(
    *,
    proposed_start: float,
    proposed_end: float,
    required_start: float,
    required_end: float,
    video_duration: float,
) -> tuple[float, float]:
    """
    Terra's final pass may trim useless edges or expand to related beats.

    Deterministic rule:
    - never exclude the required editorial core
    - never exceed MAX_CLIP_DURATION
    - never invent padding merely to chase a duration target
    """

    required_start = clamp(
        required_start,
        0.0,
        video_duration,
    )

    required_end = clamp(
        required_end,
        required_start,
        video_duration,
    )

    start = clamp(
        proposed_start,
        0.0,
        required_start,
    )

    end = clamp(
        proposed_end,
        required_end,
        video_duration,
    )

    start = min(
        start,
        max(
            0.0,
            required_start - 0.18,
        ),
    )

    end = max(
        end,
        min(
            video_duration,
            required_end + 0.28,
        ),
    )

    if (
        end - start
        <= MAX_CLIP_DURATION
    ):
        return (
            start,
            end,
        )

    excess = (
        end - start
        - MAX_CLIP_DURATION
    )

    removable_before = max(
        0.0,
        required_start - start,
    )

    removable_after = max(
        0.0,
        end - required_end,
    )

    optional_total = (
        removable_before
        + removable_after
    )

    if optional_total > 0:
        trim_before = min(
            removable_before,
            excess
            * removable_before
            / optional_total,
        )

        start += trim_before
        excess -= trim_before

        trim_after = min(
            removable_after,
            excess,
        )

        end -= trim_after
        excess -= trim_after

    if excess > 1e-6:
        # Required core itself should not be amputated.
        return (
            required_start,
            required_end,
        )

    return (
        start,
        end,
    )


def apply_story_expansion(
    selected_clip: dict[str, Any],
    proposal: dict[str, Any],
    money_moments: list[dict[str, Any]],
    video_duration: float,
) -> dict[str, Any]:

    known_moment_ids = {
        normalize_spaces(
            str(
                item.get(
                    "moment_id",
                    "",
                )
            )
        )
        for item in money_moments
        if isinstance(
            item,
            dict,
        )
        and normalize_spaces(
            str(
                item.get(
                    "moment_id",
                    "",
                )
            )
        )
    }

    coverage_audit = proposal.get(
        "moment_coverage_audit",
        [],
    )

    audited_ids: set[str] = set()
    included_by_audit: list[str] = []

    if isinstance(
        coverage_audit,
        list,
    ):
        for row in coverage_audit:
            if not isinstance(
                row,
                dict,
            ):
                continue

            moment_id = normalize_spaces(
                str(
                    row.get(
                        "moment_id",
                        "",
                    )
                )
            )

            if moment_id not in known_moment_ids:
                continue

            audited_ids.add(
                moment_id
            )

            if str(
                row.get(
                    "decision",
                    "",
                )
            ).casefold() == "include":
                included_by_audit.append(
                    moment_id
                )

    missing_audit = sorted(
        known_moment_ids
        - audited_ids
    )

    if missing_audit:
        raise RuntimeError(
            "Terra final story-coverage audit'i eksik bıraktı: "
            + ", ".join(
                missing_audit
            )
        )

    original_start = _safe_float(
        selected_clip.get(
            "start",
            0.0,
        )
    )

    original_end = _safe_float(
        selected_clip.get(
            "end",
            original_start,
        )
    )

    proposed_start = clamp(
        _safe_float(
            proposal.get(
                "start",
                original_start,
            )
        ),
        0.0,
        video_duration,
    )

    proposed_end = clamp(
        _safe_float(
            proposal.get(
                "end",
                original_end,
            )
        ),
        proposed_start,
        video_duration,
    )

    # Required editorial core: strongest anchor, payoff and every money moment
    # the final coverage audit keeps.
    required_starts: list[float] = []
    required_ends: list[float] = []

    strongest_anchor = selected_clip.get(
        "strongest_anchor",
        {},
    )

    if isinstance(
        strongest_anchor,
        dict,
    ):
        anchor_start = _safe_float(
            strongest_anchor.get(
                "start",
                original_start,
            )
        )

        anchor_end = _safe_float(
            strongest_anchor.get(
                "end",
                anchor_start,
            )
        )

        if anchor_end > anchor_start:
            required_starts.append(
                anchor_start
            )
            required_ends.append(
                anchor_end
            )

    original_payoff_start_for_core = _safe_float(
        selected_clip.get(
            "payoff_start",
            original_start,
        )
    )

    original_payoff_end_for_core = _safe_float(
        selected_clip.get(
            "payoff_end",
            original_payoff_start_for_core,
        )
    )

    if (
        original_payoff_end_for_core
        > original_payoff_start_for_core
    ):
        required_starts.append(
            original_payoff_start_for_core
        )
        required_ends.append(
            original_payoff_end_for_core
        )

    moment_by_id = {
        normalize_spaces(
            str(
                item.get(
                    "moment_id",
                    "",
                )
            )
        ): item
        for item in money_moments
        if isinstance(
            item,
            dict,
        )
    }

    for moment_id in included_by_audit:
        moment = moment_by_id.get(
            moment_id
        )

        if not isinstance(
            moment,
            dict,
        ):
            continue

        moment_start = _safe_float(
            moment.get(
                "start",
                original_start,
            )
        )

        moment_end = _safe_float(
            moment.get(
                "end",
                moment_start,
            )
        )

        if moment_end <= moment_start:
            continue

        required_starts.append(
            moment_start
        )
        required_ends.append(
            moment_end
        )

    original_ids_for_core = selected_clip.get(
        "covered_moment_ids",
        [],
    )

    if isinstance(
        original_ids_for_core,
        list,
    ):
        audit_decisions = {
            normalize_spaces(
                str(
                    row.get(
                        "moment_id",
                        "",
                    )
                )
            ): str(
                row.get(
                    "decision",
                    "",
                )
            ).casefold()
            for row in coverage_audit
            if isinstance(
                row,
                dict,
            )
        }

        for raw_id in original_ids_for_core:
            moment_id = normalize_spaces(
                str(
                    raw_id
                )
            )

            if (
                audit_decisions.get(
                    moment_id
                )
                == "omit"
            ):
                continue

            moment = moment_by_id.get(
                moment_id
            )

            if not isinstance(
                moment,
                dict,
            ):
                continue

            moment_start = _safe_float(
                moment.get(
                    "start",
                    original_start,
                )
            )

            moment_end = _safe_float(
                moment.get(
                    "end",
                    moment_start,
                )
            )

            if moment_end > moment_start:
                required_starts.append(
                    moment_start
                )
                required_ends.append(
                    moment_end
                )

    required_start = (
        min(
            required_starts
        )
        if required_starts
        else original_start
    )

    required_end = (
        max(
            required_ends
        )
        if required_ends
        else original_end
    )

    start, end = _fit_final_story_window(
        proposed_start=proposed_start,
        proposed_end=proposed_end,
        required_start=required_start,
        required_end=required_end,
        video_duration=video_duration,
    )

    merged = dict(
        selected_clip
    )

    merged[
        "start"
    ] = round_time(
        start
    )

    merged[
        "end"
    ] = round_time(
        end
    )

    # Payoff itself may expand but may never be amputated.
    original_payoff_start = _safe_float(
        selected_clip.get(
            "payoff_start",
            start,
        )
    )

    original_payoff_end = _safe_float(
        selected_clip.get(
            "payoff_end",
            original_payoff_start,
        )
    )

    merged[
        "payoff_start"
    ] = round_time(
        clamp(
            min(
                original_payoff_start,
                _safe_float(
                    proposal.get(
                        "payoff_start",
                        original_payoff_start,
                    )
                ),
            ),
            start,
            end,
        )
    )

    merged[
        "payoff_end"
    ] = round_time(
        clamp(
            max(
                original_payoff_end,
                _safe_float(
                    proposal.get(
                        "payoff_end",
                        original_payoff_end,
                    )
                ),
            ),
            merged[
                "payoff_start"
            ],
            end,
        )
    )

    proposal_ids = proposal.get(
        "covered_moment_ids",
        [],
    )

    if isinstance(
        proposal_ids,
        list,
    ):
        proposal_ids = (
            list(
                proposal_ids
            )
            + included_by_audit
        )
    else:
        proposal_ids = list(
            included_by_audit
        )

    original_ids = selected_clip.get(
        "covered_moment_ids",
        [],
    )

    combined_ids: list[str] = []
    seen: set[str] = set()

    for raw in (
        list(original_ids)
        if isinstance(
            original_ids,
            list,
        )
        else []
    ) + (
        list(proposal_ids)
        if isinstance(
            proposal_ids,
            list,
        )
        else []
    ):
        moment_id = normalize_spaces(
            str(
                raw
            )
        )

        if (
            moment_id
            and moment_id not in seen
        ):
            seen.add(
                moment_id
            )
            combined_ids.append(
                moment_id
            )

    merged[
        "covered_moment_ids"
    ] = combined_ids[
        :MAX_COVERED_MOMENTS
    ]

    primary = normalize_spaces(
        str(
            proposal.get(
                "primary_moment_id",
                selected_clip.get(
                    "primary_moment_id",
                    "",
                ),
            )
        )
    )

    if primary:
        merged[
            "primary_moment_id"
        ] = primary

    proposal_keep = proposal.get(
        "must_keep_ranges",
        [],
    )

    original_keep = selected_clip.get(
        "must_keep_ranges",
        [],
    )

    combined_keep: list[
        dict[str, Any]
    ] = []

    for item in (
        list(original_keep)
        if isinstance(
            original_keep,
            list,
        )
        else []
    ) + (
        list(proposal_keep)
        if isinstance(
            proposal_keep,
            list,
        )
        else []
    ):
        if isinstance(
            item,
            dict,
        ):
            combined_keep.append(
                dict(
                    item
                )
            )

    merged[
        "must_keep_ranges"
    ] = combined_keep[
        :MAX_MUST_KEEP_RANGES
    ]

    merged[
        "coverage_reason"
    ] = normalize_spaces(
        str(
            proposal.get(
                "coverage_reason",
                selected_clip.get(
                    "coverage_reason",
                    "",
                ),
            )
        )
    )

    merged[
        "length_exception_reason"
    ] = normalize_spaces(
        str(
            proposal.get(
                "length_exception_reason",
                selected_clip.get(
                    "length_exception_reason",
                    "",
                ),
            )
        )
    )

    cleaned = validate_clip(
        merged,
        video_duration=video_duration,
        money_moments=money_moments,
    )

    if cleaned is None:
        raise RuntimeError(
            "Terra story-expansion geçersiz final clip üretti."
        )

    if (
        float(
            cleaned.get(
                "duration",
                0.0,
            )
        )
        < SHORT_CLIP_EXCEPTION_BELOW
        and not normalize_spaces(
            str(
                cleaned.get(
                    "length_exception_reason",
                    "",
                )
            )
        )
    ):
        raise RuntimeError(
            "Terra final Short'u 18 saniyenin altında bıraktı fakat "
            "length_exception_reason vermedi. Aşırı kısa output engellendi."
        )

    # Preserve final-judge metadata and selection identity.
    for key in (
        "terra_final_judge",
        "terra_selected",
        "terra_selection_reason",
        "rank",
    ):
        if key in selected_clip:
            cleaned[
                key
            ] = selected_clip[
                key
            ]

    cleaned[
        "story_expansion"
    ] = {
        "applied": bool(
            cleaned[
                "duration"
            ]
            > _safe_float(
                selected_clip.get(
                    "duration",
                    original_end - original_start,
                )
            )
            + 0.05
        ),
        "original_duration": round_time(
            _safe_float(
                selected_clip.get(
                    "duration",
                    original_end - original_start,
                )
            )
        ),
        "final_duration": cleaned[
            "duration"
        ],
        "target_min": TARGET_CLIP_MIN,
        "target_ideal": TARGET_CLIP_IDEAL,
        "target_max": TARGET_CLIP_MAX,
        "moment_coverage_audit": (
            proposal.get(
                "moment_coverage_audit",
                [],
            )
            if isinstance(
                proposal.get(
                    "moment_coverage_audit",
                    [],
                ),
                list,
            )
            else []
        ),
    }

    return cleaned



# ============================================================
# V16 — CAUSAL STORY FLOW GUARD
# ============================================================

_FLOW_CONTEXT_DEFAULTS: dict[str, tuple[float, float]] = {
    "physical_payoff": (6.5, 3.0),
    "visual_impact": (5.5, 2.6),
    "destruction": (6.5, 3.2),
    "escalation": (6.0, 3.0),
    "conflict": (5.0, 3.0),
    "reaction": (4.2, 2.2),
    "failure": (5.2, 3.0),
    "win": (4.8, 2.6),
    "reveal": (4.8, 2.5),
    "reversal": (4.5, 2.5),
    "unexpected_answer": (3.8, 2.0),
    "awkward": (3.6, 2.2),
    "ridiculous_claim": (3.8, 2.0),
    "punchline": (3.4, 1.8),
    "tension": (4.5, 2.4),
    "quotable": (2.8, 1.5),
    "other": (3.8, 2.0),
}


def _judge_metric(clip: dict[str, Any], key: str, default: float = 10.0) -> float:
    row = clip.get("terra_final_judge", {})
    if not isinstance(row, dict):
        return default
    return clamp(_safe_float(row.get(key, default), default), 0.0, 10.0)


def _story_expansion_needed(clip: dict[str, Any], video_duration: float) -> bool:
    """Use the expensive Terra story pass only when it can improve the clip."""
    duration = _safe_float(clip.get("duration", 0.0))
    if video_duration <= STORY_FLOW_MIN_MAIN_DURATION + 0.05:
        return False
    if duration < STORY_FLOW_GOOD_DURATION:
        return True
    for key in ("context_completeness", "payoff_completeness", "story_coverage"):
        if _judge_metric(clip, key) < STORY_FLOW_GOOD_METRIC:
            return True
    return False


def _source_video_from_transcript(transcript: dict[str, Any]) -> Path | None:
    source = transcript.get("source", {})
    if not isinstance(source, dict):
        return None
    raw = normalize_spaces(str(source.get("video_path", "")))
    if not raw:
        return None
    try:
        path = Path(raw).expanduser()
    except Exception:
        return None
    return path if path.exists() and path.is_file() else None


def _scan_local_motion_series(
    *,
    video_path: Path,
    scan_start: float,
    scan_end: float,
) -> list[tuple[float, float]]:
    """Return low-cost local frame-difference luminance scores via FFmpeg.

    The seek happens before input decode, and only a <=20s local window is read.
    This intentionally avoids OpenCV/new Python dependencies and adds zero API
    calls. Failures are soft: story selection falls back to the existing V17
    transcript/money-moment logic.
    """
    if scan_end - scan_start < 1.0:
        return []

    duration = max(0.25, scan_end - scan_start)
    vf = (
        f"fps={STORY_VISUAL_ORIGIN_FPS:g},"
        f"scale={STORY_VISUAL_ORIGIN_WIDTH}:-2:flags=fast_bilinear,"
        "tblend=all_mode=difference,signalstats,metadata=print"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "info",
        "-ss", f"{scan_start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(video_path),
        "-vf", vf,
        "-an",
        "-f", "null",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception:
        return []

    series: list[tuple[float, float]] = []
    current_time: float | None = None
    for line in (result.stderr or "").splitlines():
        time_match = re.search(r"pts_time:([0-9]+(?:\.[0-9]+)?)", line)
        if time_match:
            try:
                current_time = scan_start + float(time_match.group(1))
            except (TypeError, ValueError):
                current_time = None
            continue
        score_match = re.search(r"lavfi\.signalstats\.YAVG=([0-9]+(?:\.[0-9]+)?)", line)
        if score_match and current_time is not None:
            try:
                series.append((current_time, float(score_match.group(1))))
            except (TypeError, ValueError):
                pass
            current_time = None
    return series


def _detect_visual_causal_origin(
    *,
    transcript: dict[str, Any],
    moment_start: float,
    video_duration: float,
) -> dict[str, float] | None:
    """Find the first sustained physical-motion burst leading into a payoff.

    This is deliberately conservative: one isolated gesture is not enough.
    We require a sustained threshold run, a stronger crest soon afterwards,
    and continuing elevated activity toward the primary moment. The result is
    an *earlier boundary hint*, never a new highlight selection.
    """
    if not STORY_VISUAL_ORIGIN_ENABLED or moment_start <= STORY_VISUAL_ORIGIN_MIN_LEAD:
        return None
    video_path = _source_video_from_transcript(transcript)
    if video_path is None:
        return None

    scan_start = max(0.0, moment_start - min(STORY_VISUAL_ORIGIN_LOOKBACK, STORY_FLOW_MAX_SETUP_SECONDS))
    scan_end = min(video_duration, moment_start + 0.35)
    series = [
        (t, score)
        for t, score in _scan_local_motion_series(
            video_path=video_path,
            scan_start=scan_start,
            scan_end=scan_end,
        )
        if t <= moment_start + 0.05
    ]
    if len(series) < 12:
        return None

    values = sorted(score for _, score in series)
    lower_count = max(4, int(len(values) * 0.55))
    baseline_values = values[:lower_count]
    baseline = statistics.median(baseline_values)
    mad = statistics.median(abs(value - baseline) for value in baseline_values)
    threshold = max(baseline * 1.50, baseline + 2.50 * max(1.0, mad))
    strong_threshold = max(baseline * 2.00, threshold * 1.25)

    onset_index: int | None = None
    for index in range(0, len(series) - 3):
        window = series[index:index + 4]
        if sum(score >= threshold for _, score in window) < 3:
            continue
        candidate_time = window[0][0]
        if moment_start - candidate_time < STORY_VISUAL_ORIGIN_MIN_LEAD:
            continue

        future_scores = [
            score for t, score in series
            if candidate_time <= t <= min(moment_start, candidate_time + 5.5)
        ]
        remaining_scores = [score for t, score in series if candidate_time <= t <= moment_start]
        if not future_scores or not remaining_scores:
            continue
        activity_ratio = sum(score >= threshold for score in remaining_scores) / len(remaining_scores)
        if max(future_scores) < strong_threshold or activity_ratio < 0.25:
            continue
        onset_index = index
        break

    if onset_index is None:
        # V26: IMPACT-STYLE ORIGIN.
        #
        # The loop above models a *sustained struggle* (e.g. a physical
        # scuffle): motion has to stay elevated for a while after the onset.
        # A single percussive event - a door being kicked/broken, a slam, a
        # crash - looks completely different on the frame-difference curve:
        # ONE huge isolated spike, then the room often goes nearly still
        # (people freeze/react verbally) instead of staying active. That
        # pattern fails the `activity_ratio < 0.25` and "sustained 3-of-4"
        # checks above and was silently dropped, so the exported Short never
        # included the setup - only the reaction to it.
        #
        # Detect that second shape separately: a sudden isolated jump to
        # strong_threshold that is immediately preceded by calm (near
        # baseline) frames, with no sustained-activity requirement afterward.
        for index in range(1, len(series)):

            candidate_time, candidate_score = series[index]

            if moment_start - candidate_time < STORY_VISUAL_ORIGIN_MIN_LEAD:
                continue

            if candidate_score < strong_threshold:
                continue

            lookback_window = [
                score for _, score in series[max(0, index - 3):index]
            ]

            if not lookback_window or max(lookback_window) >= threshold:
                # Not a rise from calm - either noisy/already-elevated, or
                # the sustained-motion branch above should already own it.
                continue

            onset_index = index
            break

    if onset_index is None:
        return None

    onset_time = max(scan_start, series[onset_index][0] - 0.35)
    early_limit = min(moment_start, onset_time + STORY_VISUAL_ORIGIN_KEEP_MAX)
    early_rows = [(t, score) for t, score in series if onset_time <= t <= early_limit]
    if early_rows:
        first_crest_time, first_crest_score = max(early_rows, key=lambda item: item[1])
    else:
        first_crest_time, first_crest_score = onset_time, threshold

    keep_end = min(
        max(onset_time + 2.2, first_crest_time + 0.65),
        min(moment_start - 0.35, onset_time + STORY_VISUAL_ORIGIN_KEEP_MAX),
    )
    keep_end = max(onset_time + 0.6, keep_end)
    return {
        "start": round_time(onset_time),
        "keep_end": round_time(keep_end),
        "baseline": round(float(baseline), 3),
        "threshold": round(float(threshold), 3),
        "crest_score": round(float(first_crest_score), 3),
        "crest_ratio": round(float(first_crest_score / max(1.0, baseline)), 3),
    }


def _story_segments(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw = transcript.get("segments", [])
    if not isinstance(raw, list):
        return rows
    for item in raw:
        if not isinstance(item, dict):
            continue
        start = _safe_float(item.get("start", 0.0))
        end = _safe_float(item.get("end", start))
        text = normalize_spaces(str(item.get("text", "")))
        if end <= start or not text:
            continue
        rows.append({"start": start, "end": end, "text": text})
    rows.sort(key=lambda item: (item["start"], item["end"]))
    return rows


def _primary_story_moment(
    clip: dict[str, Any],
    money_moments: list[dict[str, Any]],
) -> dict[str, Any] | None:
    moment_map = _money_moment_map(money_moments)
    primary_id = normalize_spaces(str(clip.get("primary_moment_id", "")))
    if primary_id in moment_map:
        return moment_map[primary_id]

    strongest = clip.get("strongest_anchor", {})
    if isinstance(strongest, dict):
        a = _safe_float(strongest.get("start", 0.0))
        b = _safe_float(strongest.get("end", a))
        candidates: list[tuple[float, float, dict[str, Any]]] = []
        for moment in money_moments:
            if not isinstance(moment, dict):
                continue
            m_start = _safe_float(moment.get("start", 0.0))
            m_end = _safe_float(moment.get("end", m_start))
            overlap = max(0.0, min(b, m_end) - max(a, m_start))
            distance = abs(((a + b) * 0.5) - ((m_start + m_end) * 0.5))
            candidates.append((overlap, -distance, moment))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1], _safe_float(item[2].get("strength", 0.0))), reverse=True)
            return candidates[0][2]

    usable = [item for item in money_moments if isinstance(item, dict)]
    if not usable:
        return None
    return max(usable, key=lambda item: _safe_float(item.get("strength", 0.0)))


def _align_causal_envelope_to_speech(
    *,
    segments: list[dict[str, Any]],
    core_start: float,
    core_end: float,
    desired_start: float,
    desired_end: float,
    video_duration: float,
) -> tuple[float, float]:
    """Expand to nearby contiguous speech without crossing large scene-like gaps."""
    if not segments:
        return clamp(desired_start, 0.0, video_duration), clamp(desired_end, 0.0, video_duration)

    center = (core_start + core_end) * 0.5
    nearest_index = min(
        range(len(segments)),
        key=lambda idx: 0.0 if segments[idx]["start"] <= center <= segments[idx]["end"] else min(abs(center - segments[idx]["start"]), abs(center - segments[idx]["end"])),
    )

    left = nearest_index
    right = nearest_index
    while left > 0:
        prev = segments[left - 1]
        current = segments[left]
        gap = max(0.0, current["start"] - prev["end"])
        if gap > STORY_FLOW_MAX_TRANSCRIPT_GAP:
            break
        if core_start - prev["start"] > STORY_FLOW_MAX_SETUP_SECONDS + 0.5:
            break
        left -= 1
        if segments[left]["start"] <= desired_start:
            break

    while right + 1 < len(segments):
        current = segments[right]
        nxt = segments[right + 1]
        gap = max(0.0, nxt["start"] - current["end"])
        if gap > STORY_FLOW_MAX_TRANSCRIPT_GAP:
            break
        if nxt["end"] - core_end > STORY_FLOW_MAX_REACTION_SECONDS + 0.5:
            break
        right += 1
        if segments[right]["end"] >= desired_end:
            break

    start = min(desired_start, segments[left]["start"])
    end = max(desired_end, segments[right]["end"])
    return clamp(start, 0.0, video_duration), clamp(end, 0.0, video_duration)



def _extend_contiguous_story_to_minimum(
    *,
    segments: list[dict[str, Any]],
    start: float,
    end: float,
    core_start: float,
    core_end: float,
    minimum_duration: float,
    video_duration: float,
    left_floor: float | None = None,
) -> tuple[float, float]:
    """Use real nearby speech to reach a readable main-story length when possible.

    This never pads blindly. It only consumes contiguous transcript segments and
    stops at large gaps or the causal setup/reaction safety caps.
    """
    start = clamp(start, 0.0, video_duration)
    end = clamp(end, start, video_duration)
    if end - start >= minimum_duration or not segments:
        return start, end

    # Find the transcript interval already covered by the current story window.
    covered = [
        idx for idx, seg in enumerate(segments)
        if seg["end"] >= start - 0.05 and seg["start"] <= end + 0.05
    ]
    if not covered:
        return start, end

    left = min(covered)
    right = max(covered)
    changed = True
    while end - start < minimum_duration and changed:
        changed = False

        # Favor setup first: curiosity peaks are explained in the main video.
        if left > 0:
            prev = segments[left - 1]
            current = segments[left]
            gap = max(0.0, current["start"] - prev["end"])
            if (
                gap <= STORY_FLOW_MAX_TRANSCRIPT_GAP
                and core_start - prev["start"] <= STORY_FLOW_MAX_SETUP_SECONDS + 0.5
                and (left_floor is None or prev["start"] >= left_floor - 0.05)
            ):
                left -= 1
                start = min(start, segments[left]["start"])
                changed = True
                if end - start >= minimum_duration:
                    break

        if right + 1 < len(segments):
            current = segments[right]
            nxt = segments[right + 1]
            gap = max(0.0, nxt["start"] - current["end"])
            if (
                gap <= STORY_FLOW_MAX_TRANSCRIPT_GAP
                and nxt["end"] - core_end <= STORY_FLOW_MAX_REACTION_SECONDS + 0.5
            ):
                right += 1
                end = max(end, segments[right]["end"])
                changed = True

    return clamp(start, 0.0, video_duration), clamp(end, start, video_duration)


def enforce_causal_story_flow(
    *,
    transcript: dict[str, Any],
    selected_clip: dict[str, Any],
    money_moments: list[dict[str, Any]],
    video_duration: float,
) -> dict[str, Any]:
    """Keep the main Short causal even when the selected peak is extremely strong.

    This does NOT select a new highlight and makes no AI call. It can only expand
    the already-selected story around its primary money moment and add narrow
    setup/reaction protection so pacing cannot turn it back into a peak montage.
    """
    primary = _primary_story_moment(selected_clip, money_moments)
    if not isinstance(primary, dict):
        return selected_clip

    original_start = _safe_float(selected_clip.get("start", 0.0))
    original_end = _safe_float(selected_clip.get("end", original_start))
    original_duration = max(0.0, original_end - original_start)

    moment_start = clamp(_safe_float(primary.get("start", original_start)), 0.0, video_duration)
    moment_end = clamp(_safe_float(primary.get("end", moment_start)), moment_start, video_duration)
    moment_type = normalize_spaces(str(primary.get("type", "other"))).casefold() or "other"
    moment_strength = clamp(_safe_float(primary.get("strength", 0.0)), 0.0, 10.0)
    moment_source = normalize_spaces(str(primary.get("source", "transcript"))).casefold() or "transcript"
    default_before, default_after = _FLOW_CONTEXT_DEFAULTS.get(moment_type, _FLOW_CONTEXT_DEFAULTS["other"])

    # V17 asymmetric story rule:
    # - the intro duration is handled by teaser_analyzer and gets SHORTER as peak intensity rises;
    # - the restarted main clip gets MORE setup as the payoff becomes more spectacular.
    # This prevents a loud/visual payoff from becoming an unexplained highlight montage.
    visual_or_causal = moment_type in STORY_FLOW_INTENSE_TYPES or moment_source in {"visual", "both"}
    intense_causal = bool(visual_or_causal and moment_strength >= STORY_FLOW_INTENSE_STRENGTH)
    extreme_causal = bool(visual_or_causal and moment_strength >= STORY_FLOW_EXTREME_STRENGTH)

    requested_before = clamp(_safe_float(primary.get("context_before_seconds", 0.0)), 0.0, STORY_FLOW_MAX_SETUP_SECONDS)
    requested_after = clamp(_safe_float(primary.get("context_after_seconds", 0.0)), 0.0, STORY_FLOW_MAX_REACTION_SECONDS)
    before = min(STORY_FLOW_MAX_SETUP_SECONDS, max(default_before, requested_before))
    after = min(STORY_FLOW_MAX_REACTION_SECONDS, max(default_after, requested_after))

    target_main_duration = STORY_FLOW_MIN_MAIN_DURATION
    if extreme_causal:
        before = min(STORY_FLOW_MAX_SETUP_SECONDS, max(before, STORY_FLOW_EXTREME_SETUP_FLOOR))
        after = min(STORY_FLOW_MAX_REACTION_SECONDS, max(after, STORY_FLOW_EXTREME_REACTION_FLOOR))
        target_main_duration = STORY_FLOW_EXTREME_MAIN_FLOOR
    elif intense_causal:
        before = min(STORY_FLOW_MAX_SETUP_SECONDS, max(before, STORY_FLOW_INTENSE_SETUP_FLOOR))
        after = min(STORY_FLOW_MAX_REACTION_SECONDS, max(after, STORY_FLOW_INTENSE_REACTION_FLOOR))
        target_main_duration = STORY_FLOW_INTENSE_MAIN_FLOOR

    # A weak context/story score means Terra itself is telling us the selected
    # bounds do not explain the event well enough. Spend zero extra AI calls:
    # deterministically look farther toward the cause and immediate consequence.
    weakest_flow_metric = min(
        _judge_metric(selected_clip, "context_completeness"),
        _judge_metric(selected_clip, "story_coverage"),
    )
    if weakest_flow_metric < STORY_FLOW_RECOVERY_METRIC:
        before = min(STORY_FLOW_MAX_SETUP_SECONDS, before + 2.4)
        after = min(STORY_FLOW_MAX_REACTION_SECONDS, after + 1.0)
    elif original_duration < STORY_FLOW_MIN_MAIN_DURATION and video_duration > STORY_FLOW_MIN_MAIN_DURATION + 0.05:
        before = min(STORY_FLOW_MAX_SETUP_SECONDS, before + 1.2)
        after = min(STORY_FLOW_MAX_REACTION_SECONDS, after + 0.45)

    if bool(primary.get("preserve_pause_after", False)):
        after = min(STORY_FLOW_MAX_REACTION_SECONDS, after + 0.55)

    desired_start = max(0.0, moment_start - before)
    desired_end = min(video_duration, moment_end + after)

    # V18: speech timestamps cannot see silent physical setup. For a strong
    # visual/causal payoff, recover the earliest sustained motion burst in the
    # same local pre-payoff window and treat it as part of the causal story.
    visual_origin: dict[str, float] | None = None
    visual_scan_candidate = bool(
        intense_causal
        or (visual_or_causal and moment_strength >= 7.0)
        or (original_duration < STORY_FLOW_MIN_MAIN_DURATION and weakest_flow_metric < STORY_FLOW_RECOVERY_METRIC)
    )
    if visual_scan_candidate:
        candidate_origin = _detect_visual_causal_origin(
            transcript=transcript,
            moment_start=moment_start,
            video_duration=video_duration,
        )
        if isinstance(candidate_origin, dict):
            crest_score = _safe_float(candidate_origin.get("crest_score", 0.0))
            crest_ratio = _safe_float(candidate_origin.get("crest_ratio", 0.0))
            # If Terra did not already classify this as an intense causal event,
            # require very strong local visual evidence before changing bounds.
            if intense_causal or (crest_score >= 42.0 and crest_ratio >= 2.70):
                visual_origin = candidate_origin
                # The measured physical onset is a better causal start than a
                # generic "N seconds before peak" budget. Related covered money
                # moments may still extend this earlier below.
                desired_start = _safe_float(visual_origin.get("start", desired_start))

    # Preserve every beat Terra already said belongs to this story.
    moment_map = _money_moment_map(money_moments)
    covered_ids = selected_clip.get("covered_moment_ids", [])
    if isinstance(covered_ids, list):
        for raw_id in covered_ids:
            moment = moment_map.get(normalize_spaces(str(raw_id)))
            if not isinstance(moment, dict):
                continue
            m_start = _safe_float(moment.get("start", desired_start))
            m_end = _safe_float(moment.get("end", desired_end))
            m_type = normalize_spaces(str(moment.get("type", "other"))).casefold() or "other"
            m_default_before, m_default_after = _FLOW_CONTEXT_DEFAULTS.get(m_type, _FLOW_CONTEXT_DEFAULTS["other"])
            m_before = min(
                STORY_FLOW_MAX_SETUP_SECONDS,
                max(
                    m_default_before,
                    clamp(_safe_float(moment.get("context_before_seconds", 0.0)), 0.0, STORY_FLOW_MAX_SETUP_SECONDS),
                ),
            )
            m_after = min(
                STORY_FLOW_MAX_REACTION_SECONDS,
                max(
                    m_default_after,
                    clamp(_safe_float(moment.get("context_after_seconds", 0.0)), 0.0, STORY_FLOW_MAX_REACTION_SECONDS),
                ),
            )
            desired_start = min(desired_start, max(0.0, m_start - m_before))
            desired_end = max(desired_end, min(video_duration, m_end + m_after))

    # Once a high-confidence visual origin exists, do not let generic speech
    # padding drift the start back into unrelated pre-event chatter. Earlier
    # Terra-covered moments are still honored because they already lowered
    # desired_start above.
    visual_story_floor = desired_start if isinstance(visual_origin, dict) else None

    story_segments = _story_segments(transcript)
    flow_start, flow_end = _align_causal_envelope_to_speech(
        segments=story_segments,
        core_start=moment_start,
        core_end=moment_end,
        desired_start=desired_start,
        desired_end=desired_end,
        video_duration=video_duration,
    )
    if visual_story_floor is not None:
        flow_start = max(flow_start, visual_story_floor)

    if (
        video_duration > target_main_duration + 0.05
        and (
            original_duration < target_main_duration
            or weakest_flow_metric < STORY_FLOW_RECOVERY_METRIC
            or intense_causal
        )
    ):
        flow_start, flow_end = _extend_contiguous_story_to_minimum(
            segments=story_segments,
            start=flow_start,
            end=flow_end,
            core_start=moment_start,
            core_end=moment_end,
            minimum_duration=target_main_duration,
            video_duration=video_duration,
            left_floor=visual_story_floor,
        )

    # Never shrink the main story here. Intro compactness is handled elsewhere.
    start = min(original_start, flow_start)
    end = max(original_end, flow_end)
    if end - start > MAX_CLIP_DURATION + 0.02:
        return selected_clip

    # If Terra already judged the clip complete and the envelope adds almost
    # nothing, leave bytes/metadata as stable as possible.
    if start >= original_start - 0.05 and end <= original_end + 0.05:
        return selected_clip

    merged = dict(selected_clip)
    merged["start"] = round_time(start)
    merged["end"] = round_time(end)
    merged["length_exception_reason"] = "" if end - start >= STORY_FLOW_MIN_MAIN_DURATION else normalize_spaces(str(selected_clip.get("length_exception_reason", "")))

    keep = selected_clip.get("must_keep_ranges", [])
    combined_keep = [dict(item) for item in keep if isinstance(item, dict)] if isinstance(keep, list) else []

    # Protect causal handles, not the entire expanded clip. For spectacular
    # events keep BOTH the visual origin/setup and the final escalation into the
    # peak. This lets pacing remove genuine filler without deleting how the chaos
    # began (e.g. boxes starting to fall before the later payoff).
    if isinstance(visual_origin, dict):
        visual_start = max(start, _safe_float(visual_origin.get("start", start)))
        visual_keep_end = min(end, _safe_float(visual_origin.get("keep_end", visual_start)))
        if visual_keep_end > visual_start + 0.35:
            combined_keep.append({
                "start": round_time(visual_start),
                "end": round_time(visual_keep_end),
                "reason": "V18 visual causal origin; preserve the silent physical setup that starts the chain reaction.",
            })
    elif intense_causal and moment_start - start > 4.0:
        origin_end = min(moment_start - 2.0, start + (3.2 if extreme_causal else 2.6))
        if origin_end > start + 0.35:
            combined_keep.append({
                "start": round_time(start),
                "end": round_time(origin_end),
                "reason": "V17 causal origin; preserve how the high-intensity event started.",
            })

    escalation_guard = 5.2 if extreme_causal else (4.5 if intense_causal else 3.8)
    setup_start = max(start, moment_start - min(before, escalation_guard))
    setup_end = min(end, moment_start + 0.10)
    if setup_end > setup_start:
        combined_keep.append({
            "start": round_time(setup_start),
            "end": round_time(setup_end),
            "reason": "V17 causal escalation bridge; keeps the lead-in readable before the primary peak.",
        })
    reaction_start = max(start, moment_end - 0.08)
    reaction_end = min(end, moment_end + min(after, 2.8))
    if reaction_end > reaction_start:
        combined_keep.append({
            "start": round_time(reaction_start),
            "end": round_time(reaction_end),
            "reason": "V16 causal reaction/consequence bridge after the primary peak.",
        })
    merged["must_keep_ranges"] = combined_keep[:MAX_MUST_KEEP_RANGES]

    cleaned = validate_clip(merged, video_duration=video_duration, money_moments=money_moments)
    if cleaned is None:
        return selected_clip

    for key in (
        "terra_final_judge",
        "terra_selected",
        "terra_selection_reason",
        "rank",
        "story_expansion",
        "boundary_polish",
    ):
        if key in selected_clip:
            cleaned[key] = selected_clip[key]

    cleaned["story_flow_recovery"] = {
        "applied": True,
        "policy": "setup/cause -> escalation -> primary peak -> reaction/consequence",
        "primary_moment_id": normalize_spaces(str(primary.get("moment_id", ""))),
        "primary_type": moment_type,
        "primary_strength": round(moment_strength, 2),
        "intense_causal": intense_causal,
        "extreme_causal": extreme_causal,
        "target_main_duration": round_time(target_main_duration),
        "original_duration": round_time(original_duration),
        "final_duration": round_time(_safe_float(cleaned.get("duration", end - start))),
        "context_before": round_time(before),
        "context_after": round_time(after),
        "visual_causal_origin": dict(visual_origin) if isinstance(visual_origin, dict) else None,
        "ai_calls_added": 0,
    }
    return cleaned


# ============================================================
# RANGE HELPERS
# ============================================================

def _merge_ranges(
    ranges: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    if not ranges:
        return []

    ranges = sorted(
        ranges,
        key=lambda item: (
            float(
                item[
                    "start"
                ]
            ),
            float(
                item[
                    "end"
                ]
            ),
        ),
    )

    merged: list[
        dict[str, Any]
    ] = []

    current = dict(
        ranges[0]
    )

    current_reasons = [
        normalize_spaces(
            str(
                current.get(
                    "reason",
                    "",
                )
            )
        )
    ]

    for item in ranges[
        1:
    ]:

        start = float(
            item[
                "start"
            ]
        )

        end = float(
            item[
                "end"
            ]
        )

        if (
            start
            <= float(
                current[
                    "end"
                ]
            )
            + 0.03
        ):

            current[
                "end"
            ] = max(
                float(
                    current[
                        "end"
                    ]
                ),
                end,
            )

            reason = normalize_spaces(
                str(
                    item.get(
                        "reason",
                        "",
                    )
                )
            )

            if (
                reason
                and reason not in current_reasons
            ):
                current_reasons.append(
                    reason
                )

            current[
                "reason"
            ] = " | ".join(
                value
                for value in current_reasons
                if value
            )

            continue

        current[
            "start"
        ] = round_time(
            current[
                "start"
            ]
        )

        current[
            "end"
        ] = round_time(
            current[
                "end"
            ]
        )

        current[
            "duration"
        ] = round_time(
            float(
                current[
                    "end"
                ]
            )
            - float(
                current[
                    "start"
                ]
            )
        )

        merged.append(
            current
        )

        current = dict(
            item
        )

        current_reasons = [
            normalize_spaces(
                str(
                    current.get(
                        "reason",
                        "",
                    )
                )
            )
        ]

    current[
        "start"
    ] = round_time(
        current[
            "start"
        ]
    )

    current[
        "end"
    ] = round_time(
        current[
            "end"
        ]
    )

    current[
        "duration"
    ] = round_time(
        float(
            current[
                "end"
            ]
        )
        - float(
            current[
                "start"
            ]
        )
    )

    merged.append(
        current
    )

    return merged


def _expand_short_clip(
    start: float,
    end: float,
    video_duration: float,
) -> tuple[
    float,
    float,
]:

    duration = end - start

    if duration >= MIN_CLIP_DURATION:
        return (
            start,
            end,
        )

    missing = (
        MIN_CLIP_DURATION
        - duration
    )

    start = max(
        0.0,
        start
        - missing * 0.55,
    )

    end = min(
        video_duration,
        end
        + missing * 0.45,
    )

    duration = end - start

    if duration < MIN_CLIP_DURATION:

        if start <= 0.001:

            end = min(
                video_duration,
                MIN_CLIP_DURATION,
            )

        elif (
            end
            >= video_duration
            - 0.001
        ):

            start = max(
                0.0,
                video_duration
                - MIN_CLIP_DURATION,
            )

    return (
        start,
        end,
    )


# ============================================================
# CLIP VALIDATION
# ============================================================

def _money_moment_map(
    money_moments: list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:

    if not isinstance(
        money_moments,
        list,
    ):
        return {}

    result: dict[
        str,
        dict[str, Any],
    ] = {}

    for item in money_moments:

        if not isinstance(
            item,
            dict,
        ):
            continue

        moment_id = normalize_spaces(
            str(
                item.get(
                    "moment_id",
                    "",
                )
            )
        )

        if moment_id:
            result[
                moment_id
            ] = item

    return result


def _clean_covered_moment_ids(
    clip: dict[str, Any],
    money_moments: list[dict[str, Any]] | None,
    clip_start: float,
    clip_end: float,
) -> tuple[str, list[str], list[dict[str, Any]]]:

    moment_map = _money_moment_map(
        money_moments
    )

    raw_ids = clip.get(
        "covered_moment_ids",
        [],
    )

    cleaned_ids: list[str] = []
    seen: set[str] = set()

    if isinstance(
        raw_ids,
        list,
    ):

        for raw in raw_ids:

            moment_id = normalize_spaces(
                str(
                    raw
                )
            )

            if (
                not moment_id
                or moment_id in seen
                or moment_id not in moment_map
            ):
                continue

            moment = moment_map[
                moment_id
            ]

            start = _safe_float(
                moment.get(
                    "start",
                    -1.0,
                ),
                -1.0,
            )

            end = _safe_float(
                moment.get(
                    "end",
                    -1.0,
                ),
                -1.0,
            )

            # A covered moment must actually be inside the proposed clip.
            if (
                start < clip_start - 0.05
                or end > clip_end + 0.05
            ):
                continue

            seen.add(
                moment_id
            )
            cleaned_ids.append(
                moment_id
            )

    primary = normalize_spaces(
        str(
            clip.get(
                "primary_moment_id",
                "",
            )
        )
    )

    if primary not in moment_map:
        primary = ""

    # Deterministic fallback: if Terra omitted explicit IDs, map every money
    # moment fully contained by the clip. This keeps secondary-beat protection
    # from depending on one fragile field.
    if not cleaned_ids:

        contained = [
            item
            for item in (
                money_moments
                if isinstance(
                    money_moments,
                    list,
                )
                else []
            )
            if isinstance(
                item,
                dict,
            )
            and _safe_float(
                item.get(
                    "start",
                    -1.0,
                ),
                -1.0,
            ) >= clip_start - 0.05
            and _safe_float(
                item.get(
                    "end",
                    -1.0,
                ),
                -1.0,
            ) <= clip_end + 0.05
        ]

        contained.sort(
            key=lambda item: (
                -_safe_float(
                    item.get(
                        "strength",
                        0.0,
                    )
                ),
                _safe_float(
                    item.get(
                        "start",
                        0.0,
                    )
                ),
            )
        )

        for item in contained[
            :MAX_COVERED_MOMENTS
        ]:

            moment_id = normalize_spaces(
                str(
                    item.get(
                        "moment_id",
                        "",
                    )
                )
            )

            if (
                moment_id
                and moment_id not in seen
            ):
                seen.add(
                    moment_id
                )
                cleaned_ids.append(
                    moment_id
                )

    if (
        primary
        and primary not in seen
    ):
        cleaned_ids.insert(
            0,
            primary,
        )
        seen.add(
            primary
        )

    if not primary and cleaned_ids:
        primary = max(
            cleaned_ids,
            key=lambda moment_id: _safe_float(
                moment_map[
                    moment_id
                ].get(
                    "strength",
                    0.0,
                )
            ),
        )

    cleaned_ids = cleaned_ids[
        :MAX_COVERED_MOMENTS
    ]

    covered = [
        moment_map[
            moment_id
        ]
        for moment_id in cleaned_ids
        if moment_id in moment_map
    ]

    return (
        primary,
        cleaned_ids,
        covered,
    )


def _clean_anchor_moments(
    clip: dict[str, Any],
    clip_start: float,
    clip_end: float,
    payoff_start: float,
    payoff_end: float,
) -> list[dict[str, Any]]:

    raw = clip.get(
        "anchor_moments",
        [],
    )

    result: list[
        dict[str, Any]
    ] = []

    if isinstance(
        raw,
        list,
    ):

        for item in raw[
            :MAX_ANCHORS_PER_CLIP
        ]:

            if not isinstance(
                item,
                dict,
            ):
                continue

            start = clamp(
                _safe_float(
                    item.get(
                        "start",
                        payoff_start,
                    )
                ),
                clip_start,
                clip_end,
            )

            end = clamp(
                _safe_float(
                    item.get(
                        "end",
                        start,
                    )
                ),
                start,
                clip_end,
            )

            if end <= start:
                continue

            result.append(
                {
                    "start": round_time(
                        start
                    ),

                    "end": round_time(
                        end
                    ),

                    "duration": round_time(
                        end - start
                    ),

                    "strength": round(
                        clamp(
                            _safe_float(
                                item.get(
                                    "strength",
                                    0.0,
                                )
                            ),
                            0.0,
                            10.0,
                        ),
                        1,
                    ),

                    "type": (
                        normalize_spaces(
                            str(
                                item.get(
                                    "type",
                                    "payoff",
                                )
                            )
                        )
                        or "payoff"
                    ),

                    "reason": normalize_spaces(
                        str(
                            item.get(
                                "reason",
                                "",
                            )
                        )
                    ),
                }
            )

    if not result:

        result.append(
            {
                "start": round_time(
                    payoff_start
                ),

                "end": round_time(
                    payoff_end
                ),

                "duration": round_time(
                    payoff_end
                    - payoff_start
                ),

                "strength": 0.0,

                "type": "payoff",

                "reason": (
                    "Fallback anchor derived from Terra payoff range."
                ),
            }
        )

    result.sort(
        key=lambda item: (
            -float(
                item[
                    "strength"
                ]
            ),
            float(
                item[
                    "start"
                ]
            ),
        )
    )

    return result[
        :MAX_ANCHORS_PER_CLIP
    ]


def _clean_must_keep_ranges(
    clip: dict[str, Any],
    anchors: list[dict[str, Any]],
    covered_moments: list[dict[str, Any]],
    clip_start: float,
    clip_end: float,
    payoff_start: float,
    payoff_end: float,
) -> list[dict[str, Any]]:

    ranges: list[
        dict[str, Any]
    ] = []

    raw = clip.get(
        "must_keep_ranges",
        [],
    )

    if isinstance(
        raw,
        list,
    ):

        for item in raw:

            if not isinstance(
                item,
                dict,
            ):
                continue

            start = clamp(
                _safe_float(
                    item.get(
                        "start",
                        clip_start,
                    )
                ),
                clip_start,
                clip_end,
            )

            end = clamp(
                _safe_float(
                    item.get(
                        "end",
                        start,
                    )
                ),
                start,
                clip_end,
            )

            if end <= start:
                continue

            raw_duration = (
                end - start
            )

            # Prevent a broad model-generated range from protecting filler.
            if (
                raw_duration
                > MAX_FREEFORM_PROTECTED_DURATION
            ):
                relevant_ranges: list[
                    tuple[
                        float,
                        float,
                    ]
                ] = []

                for candidate in (
                    list(anchors)
                    + list(covered_moments)
                ):
                    if not isinstance(
                        candidate,
                        dict,
                    ):
                        continue

                    candidate_start = _safe_float(
                        candidate.get(
                            "start",
                            start,
                        )
                    )

                    candidate_end = _safe_float(
                        candidate.get(
                            "end",
                            candidate_start,
                        )
                    )

                    if (
                        candidate_end <= start
                        or candidate_start >= end
                    ):
                        continue

                    relevant_ranges.append(
                        (
                            max(
                                start,
                                candidate_start
                                - PROTECTED_CONTEXT_PAD,
                            ),
                            min(
                                end,
                                candidate_end
                                + PROTECTED_CONTEXT_PAD,
                            ),
                        )
                    )

                if not relevant_ranges:
                    continue

                for (
                    protected_start,
                    protected_end,
                ) in relevant_ranges:
                    if (
                        protected_end
                        <= protected_start
                    ):
                        continue

                    ranges.append(
                        {
                            "start": protected_start,
                            "end": protected_end,
                            "reason": normalize_spaces(
                                str(
                                    item.get(
                                        "reason",
                                        "Terra narrowed must-keep range.",
                                    )
                                )
                            ),
                        }
                    )

                continue

            ranges.append(
                {
                    "start": start,
                    "end": end,
                    "reason": normalize_spaces(
                        str(
                            item.get(
                                "reason",
                                "Terra must-keep range.",
                            )
                        )
                    ),
                }
            )

    # Terra anchor is always protected even when the model forgot to duplicate
    # it into must_keep_ranges.
    for anchor in anchors:

        ranges.append(
            {
                "start": max(
                    clip_start,
                    float(
                        anchor[
                            "start"
                        ]
                    )
                    - PROTECT_PAD_BEFORE,
                ),

                "end": min(
                    clip_end,
                    float(
                        anchor[
                            "end"
                        ]
                    )
                    + PROTECT_PAD_AFTER,
                ),

                "reason": (
                    "Terra anchor protection: "
                    + str(
                        anchor.get(
                            "reason",
                            anchor.get(
                                "type",
                                "money moment",
                            ),
                        )
                    )
                ),
            }
        )

    # Every covered money moment is protected, not only the primary anchor.
    # This is the key difference between a tiny highlight and a complete Short:
    # secondary beats selected by Terra are not allowed to disappear in pacing.
    for moment in covered_moments:

        moment_start = clamp(
            _safe_float(
                moment.get(
                    "start",
                    clip_start,
                )
            ),
            clip_start,
            clip_end,
        )

        moment_end = clamp(
            _safe_float(
                moment.get(
                    "end",
                    moment_start,
                )
            ),
            moment_start,
            clip_end,
        )

        if moment_end <= moment_start:
            continue

        ranges.append(
            {
                "start": max(
                    clip_start,
                    moment_start
                    - PROTECT_PAD_BEFORE,
                ),
                "end": min(
                    clip_end,
                    moment_end
                    + PROTECT_PAD_AFTER,
                ),
                "reason": (
                    "Terra covered money moment "
                    + str(
                        moment.get(
                            "moment_id",
                            "",
                        )
                    )
                    + ": "
                    + str(
                        moment.get(
                            "label",
                            moment.get(
                                "why_compelling",
                                "secondary story beat",
                            ),
                        )
                    )
                ),
            }
        )

    # Payoff remains protected too.
    ranges.append(
        {
            "start": max(
                clip_start,
                payoff_start
                - PROTECT_PAD_BEFORE,
            ),

            "end": min(
                clip_end,
                payoff_end
                + PROTECT_PAD_AFTER,
            ),

            "reason": (
                "Terra payoff protection."
            ),
        }
    )

    cleaned: list[
        dict[str, Any]
    ] = []

    for item in ranges:

        start = clamp(
            _safe_float(
                item.get(
                    "start",
                    clip_start,
                )
            ),
            clip_start,
            clip_end,
        )

        end = clamp(
            _safe_float(
                item.get(
                    "end",
                    start,
                )
            ),
            start,
            clip_end,
        )

        if end <= start:
            continue

        cleaned.append(
            {
                "start": start,
                "end": end,
                "reason": normalize_spaces(
                    str(
                        item.get(
                            "reason",
                            "",
                        )
                    )
                ),
            }
        )

    return _merge_ranges(
        cleaned
    )


def _clean_caption_highlights(
    raw: Any,
) -> list[str]:

    if not isinstance(
        raw,
        list,
    ):
        return []

    result: list[
        str
    ] = []

    seen: set[
        str
    ] = set()

    for item in raw:

        text = normalize_spaces(
            str(
                item
            )
        )

        if not text:
            continue

        key = text.casefold()

        if key in seen:
            continue

        seen.add(
            key
        )

        result.append(
            text
        )

    return result[
        :MAX_CAPTION_HIGHLIGHTS
    ]


def _clean_editor_notes(
    raw: Any,
    clip_start: float,
    clip_end: float,
) -> list[dict[str, Any]]:

    if not isinstance(
        raw,
        list,
    ):
        return []

    result: list[
        dict[str, Any]
    ] = []

    allowed_types = {
        "manual_punch_in",
        "reaction_hold",
        "cutaway",
        "meme_overlay",
        "pace_check",
    }

    for item in raw[
        :MAX_EDITOR_NOTES
    ]:

        if not isinstance(
            item,
            dict,
        ):
            continue

        source_time = clamp(
            _safe_float(
                item.get(
                    "time",
                    clip_start,
                )
            ),
            clip_start,
            clip_end,
        )

        note_type = normalize_spaces(
            str(
                item.get(
                    "type",
                    "pace_check",
                )
            )
        )

        if note_type not in allowed_types:
            note_type = "pace_check"

        priority = int(
            clamp(
                _safe_float(
                    item.get(
                        "priority",
                        3,
                    )
                ),
                1,
                5,
            )
        )

        note = normalize_spaces(
            str(
                item.get(
                    "note",
                    "",
                )
            )
        )

        if not note:
            continue

        result.append(
            {
                "time": round_time(
                    source_time
                ),

                "clip_time": round_time(
                    source_time
                    - clip_start
                ),

                "type": note_type,

                "priority": priority,

                "note": note,
            }
        )

    result.sort(
        key=lambda item: (
            float(
                item[
                    "time"
                ]
            ),
            -int(
                item[
                    "priority"
                ]
            ),
        )
    )

    return result


def validate_clip(
    clip: dict[str, Any],
    video_duration: float,
    money_moments: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:

    start = clamp(
        _safe_float(
            clip.get(
                "start",
                0.0,
            )
        ),
        0.0,
        video_duration,
    )

    end = clamp(
        _safe_float(
            clip.get(
                "end",
                start,
            )
        ),
        start,
        video_duration,
    )

    if end <= start:
        return None

    start, end = _expand_short_clip(
        start,
        end,
        video_duration,
    )

    duration = end - start

    # If the entire source is shorter than the normal 12s floor, the source
    # itself is the only coherent clip available. Do not make short source
    # files mathematically impossible to process.
    effective_min_duration = min(
        MIN_CLIP_DURATION,
        max(0.0, video_duration),
    )

    if (
        duration
        < effective_min_duration
        - 0.02
    ):
        return None

    if duration > MAX_CLIP_DURATION:
        return None

    payoff_start = clamp(
        _safe_float(
            clip.get(
                "payoff_start",
                start,
            )
        ),
        start,
        end,
    )

    payoff_end = clamp(
        _safe_float(
            clip.get(
                "payoff_end",
                payoff_start,
            )
        ),
        payoff_start,
        end,
    )

    if payoff_end <= payoff_start:

        payoff_end = min(
            end,
            payoff_start + 0.25,
        )

    if payoff_end <= payoff_start:
        return None

    score = clamp(
        _safe_float(
            clip.get(
                "score",
                0.0,
            )
        ),
        0.0,
        10.0,
    )

    (
        primary_moment_id,
        covered_moment_ids,
        covered_moments,
    ) = _clean_covered_moment_ids(
        clip=clip,
        money_moments=money_moments,
        clip_start=start,
        clip_end=end,
    )

    anchors = _clean_anchor_moments(
        clip=clip,
        clip_start=start,
        clip_end=end,
        payoff_start=payoff_start,
        payoff_end=payoff_end,
    )

    must_keep = _clean_must_keep_ranges(
        clip=clip,
        anchors=anchors,
        covered_moments=covered_moments,
        clip_start=start,
        clip_end=end,
        payoff_start=payoff_start,
        payoff_end=payoff_end,
    )

    strongest_anchor = max(
        anchors,
        key=lambda item: float(
            item.get(
                "strength",
                0.0,
            )
        ),
    )

    return {
        "start": round_time(
            start
        ),

        "end": round_time(
            end
        ),

        "duration": round_time(
            duration
        ),

        "payoff_start": round_time(
            payoff_start
        ),

        "payoff_end": round_time(
            payoff_end
        ),

        "score": round(
            score,
            1,
        ),

        "score_policy": (
            SCORE_POLICY
        ),

        "title": (
            normalize_spaces(
                str(
                    clip.get(
                        "title",
                        "",
                    )
                )
            )
            or "Untitled Clip"
        ),

        "hook_text": normalize_spaces(
            str(
                clip.get(
                    "hook_text",
                    "",
                )
            )
        ).upper(),

        "hook_type": (
            normalize_spaces(
                str(
                    clip.get(
                        "hook_type",
                        "curiosity",
                    )
                )
            )
            or "curiosity"
        ),

        "emotion": (
            normalize_spaces(
                str(
                    clip.get(
                        "emotion",
                        "curiosity",
                    )
                )
            )
            or "curiosity"
        ),

        "reason": normalize_spaces(
            str(
                clip.get(
                    "reason",
                    "",
                )
            )
        ),

        "context": normalize_spaces(
            str(
                clip.get(
                    "context",
                    "",
                )
            )
        ),

        "primary_moment_id": (
            primary_moment_id
        ),

        "covered_moment_ids": (
            covered_moment_ids
        ),

        "coverage_reason": normalize_spaces(
            str(
                clip.get(
                    "coverage_reason",
                    "",
                )
            )
        ),

        "length_exception_reason": (
            normalize_spaces(
                str(
                    clip.get(
                        "length_exception_reason",
                        "",
                    )
                )
            )
            or (
                "Source video is shorter than the normal Short target; full "
                "available source was preserved."
                if video_duration < SHORT_CLIP_EXCEPTION_BELOW + 0.02
                and start <= 0.02
                and end >= video_duration - 0.02
                else ""
            )
        ),

        "story_coverage": {
            "covered_count": len(
                covered_moment_ids
            ),
            "target_min_seconds": TARGET_CLIP_MIN,
            "target_ideal_seconds": TARGET_CLIP_IDEAL,
            "target_max_seconds": TARGET_CLIP_MAX,
            "short_exception_required": bool(
                duration
                < SHORT_CLIP_EXCEPTION_BELOW
            ),
        },

        "visual_event_ids": [
            normalize_spaces(
                str(
                    value
                )
            )
            for value in clip.get(
                "visual_event_ids",
                [],
            )
            if normalize_spaces(
                str(
                    value
                )
            )
        ][
            :16
        ],

        "strongest_anchor": {
            "start": (
                strongest_anchor[
                    "start"
                ]
            ),

            "end": (
                strongest_anchor[
                    "end"
                ]
            ),

            "strength": (
                strongest_anchor[
                    "strength"
                ]
            ),

            "type": (
                strongest_anchor[
                    "type"
                ]
            ),

            "reason": (
                strongest_anchor[
                    "reason"
                ]
            ),
        },

        "anchor_moments": anchors,

        "must_keep_ranges": must_keep,

        "caption_highlights": (
            _clean_caption_highlights(
                clip.get(
                    "caption_highlights",
                    [],
                )
            )
        ),

        "editor_notes": (
            _clean_editor_notes(
                clip.get(
                    "editor_notes",
                    [],
                ),
                clip_start=start,
                clip_end=end,
            )
        ),
    }



def _build_money_moment_fallback_clips(
    money_moments: list[dict[str, Any]],
    video_duration: float,
) -> list[dict[str, Any]]:
    """Recover model-invalid clip bounds from already-valid money moments.

    This is boundary recovery only: the function never invents a new editorial
    highlight. It builds deterministic windows around the money moments that
    already passed MIMIR's money-moment validation, then Terra still performs
    the normal final ranking/judgement over the recovered candidates.
    """

    if video_duration <= 0.02:
        return []

    usable = [
        item
        for item in money_moments
        if isinstance(item, dict)
        and _safe_float(item.get("end", 0.0))
        > _safe_float(item.get("start", 0.0))
    ]

    usable.sort(
        key=lambda item: (
            -_safe_float(item.get("strength", 0.0)),
            _safe_float(item.get("start", 0.0)),
        )
    )

    raw_candidates: list[dict[str, Any]] = []

    for moment in usable[:MAX_CLIPS]:
        moment_start = clamp(
            _safe_float(moment.get("start", 0.0)),
            0.0,
            video_duration,
        )
        moment_end = clamp(
            _safe_float(moment.get("end", moment_start)),
            moment_start,
            video_duration,
        )
        moment_duration = moment_end - moment_start

        if moment_duration <= 0.0:
            continue

        preferred_before = clamp(
            _safe_float(moment.get("context_before_seconds", 0.0)),
            0.0,
            20.0,
        )
        preferred_after = clamp(
            _safe_float(moment.get("context_after_seconds", 0.0)),
            0.0,
            12.0,
        )

        # Prefer the normal ~32s dense-story target. If Terra returned an
        # over-broad money-moment span (>55s), do not discard the moment and
        # kill the run: recover a bounded window around its temporal center.
        desired_duration = min(
            max(video_duration, 0.0),
            MAX_CLIP_DURATION,
            max(
                min(MIN_CLIP_DURATION, max(video_duration, 0.0)),
                min(TARGET_CLIP_IDEAL, max(video_duration, 0.0)),
                min(
                    moment_duration + preferred_before + preferred_after,
                    MAX_CLIP_DURATION,
                ),
            ),
        )

        if moment_duration > desired_duration + 0.02:
            center = (moment_start + moment_end) / 2.0
            start = center - desired_duration * 0.5
            end = center + desired_duration * 0.5
        else:
            extra = max(0.0, desired_duration - moment_duration)
            preferred_total = preferred_before + preferred_after

            if preferred_total > extra and preferred_total > 0.0:
                scale = extra / preferred_total
                before = preferred_before * scale
                after = preferred_after * scale
            else:
                before = preferred_before
                after = preferred_after
                remaining = max(0.0, extra - preferred_total)
                before += remaining * 0.58
                after += remaining * 0.42

            start = moment_start - before
            end = moment_end + after

        # At the source edges shift the whole window instead of chopping the
        # money moment or needlessly shortening the recovered candidate.
        if start < 0.0:
            end = min(video_duration, end - start)
            start = 0.0

        if end > video_duration:
            start = max(0.0, start - (end - video_duration))
            end = video_duration

        start, end = _expand_short_clip(
            clamp(start, 0.0, video_duration),
            clamp(end, 0.0, video_duration),
            video_duration,
        )

        effective_min_duration = min(MIN_CLIP_DURATION, video_duration)
        if (
            end - start < effective_min_duration - 0.02
            or end - start > MAX_CLIP_DURATION + 0.02
            or moment_end <= start + 0.02
            or moment_start >= end - 0.02
        ):
            continue

        contained = [
            item
            for item in usable
            if _safe_float(item.get("start", -1.0), -1.0) >= start - 0.05
            and _safe_float(item.get("end", -1.0), -1.0) <= end + 0.05
        ]
        contained.sort(
            key=lambda item: (
                -_safe_float(item.get("strength", 0.0)),
                _safe_float(item.get("start", 0.0)),
            )
        )
        contained = contained[:MAX_COVERED_MOMENTS]

        primary_id = normalize_spaces(str(moment.get("moment_id", "")))
        covered_ids = [
            normalize_spaces(str(item.get("moment_id", "")))
            for item in contained
            if normalize_spaces(str(item.get("moment_id", "")))
        ]
        if primary_id and primary_id not in covered_ids:
            covered_ids.insert(0, primary_id)
            covered_ids = covered_ids[:MAX_COVERED_MOMENTS]

        anchors: list[dict[str, Any]] = []
        for item in contained[:MAX_ANCHORS_PER_CLIP]:
            anchors.append(
                {
                    "start": _safe_float(item.get("start", moment_start)),
                    "end": _safe_float(item.get("end", moment_end)),
                    "strength": _safe_float(item.get("strength", 0.0)),
                    "type": normalize_spaces(str(item.get("type", "other"))) or "other",
                    "reason": normalize_spaces(
                        str(item.get("why_compelling", item.get("label", "money moment")))
                    ),
                }
            )

        if not anchors:
            anchors = [
                {
                    "start": moment_start,
                    "end": moment_end,
                    "strength": _safe_float(moment.get("strength", 0.0)),
                    "type": normalize_spaces(str(moment.get("type", "other"))) or "other",
                    "reason": normalize_spaces(
                        str(moment.get("why_compelling", moment.get("label", "money moment")))
                    ),
                }
            ]

        must_keep = [
            {
                "start": _safe_float(item.get("start", moment_start)),
                "end": _safe_float(item.get("end", moment_end)),
                "reason": "Protected money moment fallback boundary.",
            }
            for item in contained[:MAX_MUST_KEEP_RANGES]
            if _safe_float(item.get("end", 0.0))
            > _safe_float(item.get("start", 0.0))
        ]
        if not must_keep:
            must_keep = [
                {
                    "start": moment_start,
                    "end": moment_end,
                    "reason": "Protected primary money moment fallback boundary.",
                }
            ]

        visual_event_ids: list[str] = []
        for item in contained:
            for raw_id in item.get("visual_event_ids", []):
                event_id = normalize_spaces(str(raw_id))
                if event_id and event_id not in visual_event_ids:
                    visual_event_ids.append(event_id)

        raw_candidates.append(
            {
                "start": round_time(start),
                "end": round_time(end),
                "payoff_start": round_time(max(start, moment_start)),
                "payoff_end": round_time(min(end, moment_end)),
                "score": round(
                    clamp(_safe_float(moment.get("strength", 0.0)), 0.0, 10.0),
                    1,
                ),
                "title": (
                    normalize_spaces(str(moment.get("label", "Money moment fallback")))
                    or "Money moment fallback"
                ),
                "hook_text": "",
                "hook_type": "curiosity",
                "emotion": "curiosity",
                "reason": (
                    "Deterministic boundary recovery around an already-approved "
                    "money moment; the model candidate bounds were unusable."
                ),
                "context": normalize_spaces(str(moment.get("why_compelling", ""))),
                "primary_moment_id": primary_id,
                "covered_moment_ids": covered_ids,
                "coverage_reason": (
                    "Recovered from validated money-moment timing and its requested context."
                ),
                "length_exception_reason": (
                    "Source video is shorter than the normal Short target; full "
                    "available source was preserved."
                    if video_duration < SHORT_CLIP_EXCEPTION_BELOW + 0.02
                    else ""
                ),
                "visual_event_ids": visual_event_ids[:16],
                "anchor_moments": anchors,
                "must_keep_ranges": must_keep,
                "caption_highlights": [],
                "editor_notes": [],
            }
        )

    return raw_candidates



def _build_emergency_money_moment_clip(
    money_moments: list[dict[str, Any]],
    video_duration: float,
) -> dict[str, Any] | None:
    """Canonical last-resort clip around the strongest valid money moment.

    This function exists only to prevent a valid editorial anchor from being
    lost because model-provided boundaries are malformed. It does not discover
    a new highlight and it never runs when normal/fallback candidates validate.
    """

    if video_duration <= 0.02:
        return None

    valid = [
        item
        for item in money_moments
        if isinstance(item, dict)
        and _safe_float(item.get("end", 0.0))
        > _safe_float(item.get("start", 0.0))
    ]
    if not valid:
        return None

    valid.sort(
        key=lambda item: (
            -_safe_float(item.get("strength", 0.0)),
            _safe_float(item.get("start", 0.0)),
        )
    )
    moment = valid[0]
    ms = clamp(_safe_float(moment.get("start", 0.0)), 0.0, video_duration)
    me = clamp(_safe_float(moment.get("end", ms)), ms, video_duration)
    if me <= ms:
        return None

    window = min(
        MAX_CLIP_DURATION,
        video_duration,
        max(
            min(TARGET_CLIP_IDEAL, video_duration),
            min(MIN_CLIP_DURATION, video_duration),
        ),
    )
    if window <= 0.02:
        return None

    center = (ms + me) / 2.0
    start = center - window * 0.5
    end = center + window * 0.5

    if start < 0.0:
        end -= start
        start = 0.0
    if end > video_duration:
        start = max(0.0, start - (end - video_duration))
        end = video_duration

    # For very short sources, preserve the entire available source.
    if video_duration <= MIN_CLIP_DURATION + 0.02:
        start = 0.0
        end = video_duration

    payoff_start = max(start, ms)
    payoff_end = min(end, me)
    if payoff_end <= payoff_start:
        # Should not happen for a centered window, but keep a tiny valid payoff
        # range inside the clip instead of aborting the entire pipeline.
        payoff_start = clamp(center - 0.125, start, end)
        payoff_end = clamp(center + 0.125, payoff_start, end)
        if payoff_end <= payoff_start:
            return None

    moment_id = normalize_spaces(str(moment.get("moment_id", "")))
    anchor = {
        "start": round_time(payoff_start),
        "end": round_time(payoff_end),
        "strength": round(clamp(_safe_float(moment.get("strength", 0.0)), 0.0, 10.0), 1),
        "type": normalize_spaces(str(moment.get("type", "other"))) or "other",
        "reason": normalize_spaces(
            str(moment.get("why_compelling", moment.get("label", "money moment")))
        ),
    }

    raw = {
        "start": round_time(start),
        "end": round_time(end),
        "payoff_start": round_time(payoff_start),
        "payoff_end": round_time(payoff_end),
        "score": round(clamp(_safe_float(moment.get("strength", 0.0)), 0.0, 10.0), 1),
        "title": normalize_spaces(str(moment.get("label", "Money moment recovery"))) or "Money moment recovery",
        "hook_text": "",
        "hook_type": "curiosity",
        "emotion": "curiosity",
        "reason": "Canonical boundary recovery around Terra's strongest validated money moment.",
        "context": normalize_spaces(str(moment.get("why_compelling", ""))),
        "primary_moment_id": moment_id,
        "covered_moment_ids": [moment_id] if moment_id else [],
        "coverage_reason": "Emergency boundary-only recovery; editorial anchor remains Terra's money moment.",
        "length_exception_reason": (
            "Source video is shorter than the normal Short target; full available source was preserved."
            if video_duration < SHORT_CLIP_EXCEPTION_BELOW + 0.02
            else ""
        ),
        "visual_event_ids": [
            normalize_spaces(str(value))
            for value in moment.get("visual_event_ids", [])
            if normalize_spaces(str(value))
        ][:16],
        "anchor_moments": [anchor],
        "must_keep_ranges": [
            {
                "start": round_time(payoff_start),
                "end": round_time(payoff_end),
                "reason": "Protected strongest money moment emergency boundary.",
            }
        ],
        "caption_highlights": [],
        "editor_notes": [],
    }

    return validate_clip(
        raw,
        video_duration=video_duration,
        money_moments=money_moments,
    )


def validate_analysis(
    clips: list[dict[str, Any]],
    video_duration: float,
    money_moments: list[dict[str, Any]] | None = None,
    *,
    limit: int | None = MAX_CLIPS,
) -> list[dict[str, Any]]:

    valid: list[
        dict[str, Any]
    ] = []

    for clip in clips:

        cleaned = validate_clip(
            clip,
            video_duration,
            money_moments=money_moments,
        )

        if cleaned is not None:

            valid.append(
                cleaned
            )

    valid.sort(
        key=lambda item: (
            -float(
                item[
                    "score"
                ]
            ),
            -float(
                item.get(
                    "strongest_anchor",
                    {},
                ).get(
                    "strength",
                    0.0,
                )
            ),
            -len(
                item.get(
                    "covered_moment_ids",
                    [],
                )
            ),
            abs(
                float(
                    item[
                        "duration"
                    ]
                )
                - TARGET_CLIP_IDEAL
            ),
        )
    )

    deduped: list[
        dict[str, Any]
    ] = []

    for item in valid:

        duplicate = False

        for existing in deduped:

            overlap = max(
                0.0,
                min(
                    float(
                        item[
                            "end"
                        ]
                    ),
                    float(
                        existing[
                            "end"
                        ]
                    ),
                )
                - max(
                    float(
                        item[
                            "start"
                        ]
                    ),
                    float(
                        existing[
                            "start"
                        ]
                    ),
                ),
            )

            shortest = min(
                float(
                    item[
                        "duration"
                    ]
                ),
                float(
                    existing[
                        "duration"
                    ]
                ),
            )

            if (
                shortest > 0
                and overlap / shortest
                >= 0.78
            ):

                duplicate = True
                break

        if not duplicate:

            deduped.append(
                item
            )

    if limit is None:
        return deduped
    return deduped[:max(0, int(limit))]




def _visual_event_overlap_ratio(
    event: dict[str, Any],
    moment: dict[str, Any],
) -> float:
    e_start = _safe_float(event.get("start", 0.0))
    e_end = _safe_float(event.get("end", e_start))
    m_start = _safe_float(moment.get("start", 0.0))
    m_end = _safe_float(moment.get("end", m_start))
    overlap = max(0.0, min(e_end, m_end) - max(e_start, m_start))
    e_len = max(0.08, e_end - e_start)
    return overlap / e_len


def _moment_covers_visual_event(
    moment: dict[str, Any],
    event: dict[str, Any],
) -> bool:
    event_id = normalize_spaces(str(event.get("event_id", "")))
    ids = {
        normalize_spaces(str(value))
        for value in moment.get("visual_event_ids", [])
        if normalize_spaces(str(value))
    }
    if event_id and event_id in ids:
        return True
    return _visual_event_overlap_ratio(event, moment) >= 0.55


def _decisive_visual_priority(event: dict[str, Any]) -> tuple[float, float]:
    kind = normalize_spaces(str(event.get("type", "other"))).casefold()
    base = {
        "object_break": 1.00,
        "destruction": 0.98,
        "explosion": 0.97,
        "crash": 0.94,
        "fall": 0.90,
        "impact": 0.88,
        "visual_payoff": 0.86,
        "source_peak_guard": 0.92,
    }.get(kind, 0.0)
    confidence = clamp(_safe_float(event.get("confidence", 0.0)), 0.0, 1.0)
    return (base, confidence)


def _inject_decisive_visual_money_moments(
    *,
    money_moments: list[dict[str, Any]],
    visual_events: list[dict[str, Any]],
    video_duration: float,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Guarantee decisive visual facts reach Terra as candidates, not winners."""
    additions: list[dict[str, Any]] = []
    injected_event_ids: set[str] = set()

    candidates = [
        event for event in visual_events
        if isinstance(event, dict)
        and normalize_spaces(str(event.get("type", ""))).casefold() in DECISIVE_VISUAL_EVENT_TYPES
        and clamp(_safe_float(event.get("confidence", 0.0)), 0.0, 1.0) >= DECISIVE_VISUAL_MIN_CONFIDENCE
    ]
    candidates.sort(key=lambda event: _decisive_visual_priority(event), reverse=True)

    for event in candidates[:DECISIVE_VISUAL_MAX_INJECTIONS]:
        if any(_moment_covers_visual_event(moment, event) for moment in money_moments):
            continue

        kind = normalize_spaces(str(event.get("type", "other"))).casefold()
        event_id = normalize_spaces(str(event.get("event_id", "")))
        confidence = clamp(_safe_float(event.get("confidence", 0.0)), 0.0, 1.0)
        start = clamp(_safe_float(event.get("start", 0.0)), 0.0, video_duration)
        end = clamp(_safe_float(event.get("end", start + 0.15)), start, video_duration)
        if end <= start:
            end = min(video_duration, start + 0.20)
        if end <= start:
            continue

        mapped_type = {
            "object_break": "destruction",
            "destruction": "destruction",
            "explosion": "physical_payoff",
            "crash": "physical_payoff",
            "fall": "physical_payoff",
            "impact": "visual_impact",
            "visual_payoff": "visual_impact",
            "source_peak_guard": "visual_impact",
        }.get(kind, "visual_impact")

        # A decisive factual event should compete strongly, but Terra still owns
        # the final ranking. Confidence influences strength without hard-wiring a
        # particular uploaded video or timestamp.
        strength_floor = {
            "object_break": 8.9,
            "destruction": 8.8,
            "explosion": 8.8,
            "crash": 8.5,
            "fall": 8.2,
            "impact": 8.0,
            "visual_payoff": 7.9,
            "source_peak_guard": 8.7,
        }.get(kind, 7.8)
        strength = min(9.7, strength_floor + max(0.0, confidence - 0.68) * 1.8)
        before = 12.0 if kind in {"object_break", "destruction", "crash", "explosion"} else 8.0
        after = 6.0 if kind == "source_peak_guard" else (4.5 if kind in {"object_break", "destruction", "crash", "explosion", "fall"} else 3.0)
        description = normalize_spaces(str(event.get("description", ""))) or f"{kind} visual event"
        if kind == "source_peak_guard":
            label = f"Abrupt source event around {((start + end) / 2.0):.2f}s"
            why_compelling = (
                "V26 source-peak guard: a very strong abrupt source transient was detected and "
                "the visual observer was given tightly bracketed before/peak/after frames. "
                "The semantic event type is intentionally unclaimed; Terra must compare it with "
                "the transcript and visual facts instead of letting the region disappear unseen."
            )
        else:
            label = description[:160]
            why_compelling = (
                "High-confidence decisive source-video state change. "
                "V23 guarantees it reaches Terra as a candidate so silent visual payoffs cannot vanish before final judgement."
            )

        additions.append({
            "start": round_time(start),
            "end": round_time(end),
            "strength": round(strength, 1),
            "type": mapped_type,
            "source": "visual",
            "visual_event_ids": [event_id] if event_id else [],
            "label": label,
            "why_compelling": why_compelling,
            "context_before_seconds": before,
            "context_after_seconds": after,
            "preserve_pause_after": True,
        })
        if event_id:
            injected_event_ids.add(event_id)

    if not additions:
        return money_moments, injected_event_ids

    merged = validate_money_moments(
        list(money_moments) + additions,
        video_duration=video_duration,
    )
    return merged, injected_event_ids


def _clip_covers_visual_event_id(
    clip: dict[str, Any],
    event_id: str,
    money_moments: list[dict[str, Any]],
) -> bool:
    direct = {
        normalize_spaces(str(value))
        for value in clip.get("visual_event_ids", [])
        if normalize_spaces(str(value))
    }
    if event_id in direct:
        return True

    covered = {
        normalize_spaces(str(value))
        for value in clip.get("covered_moment_ids", [])
        if normalize_spaces(str(value))
    }
    for moment in money_moments:
        if normalize_spaces(str(moment.get("moment_id", ""))) not in covered:
            continue
        mids = {
            normalize_spaces(str(value))
            for value in moment.get("visual_event_ids", [])
            if normalize_spaces(str(value))
        }
        if event_id in mids:
            return True
    return False


def _ensure_decisive_visual_clip_candidates(
    *,
    clips: list[dict[str, Any]],
    money_moments: list[dict[str, Any]],
    injected_event_ids: set[str],
    video_duration: float,
) -> list[dict[str, Any]]:
    """Guarantee decisive-event coverage survives MAX_CLIPS truncation.

    Every injected decisive event that has a valid candidate gets a reserved
    representation in the final <=MAX_CLIPS candidate set. Multiple decisive
    events may share one candidate, so this does not blindly consume one slot
    per event when a single coherent story covers several of them.
    """
    if not injected_event_ids:
        return validate_analysis(
            clips,
            video_duration=video_duration,
            money_moments=money_moments,
        )

    result = list(clips)

    # First create a valid fallback candidate for any decisive event the model
    # composer omitted entirely.
    for event_id in sorted(injected_event_ids):
        if any(_clip_covers_visual_event_id(clip, event_id, money_moments) for clip in result):
            continue

        moment = next(
            (
                item
                for item in money_moments
                if event_id in {
                    normalize_spaces(str(value))
                    for value in item.get("visual_event_ids", [])
                }
            ),
            None,
        )
        if not isinstance(moment, dict):
            continue

        fallback_raw = _build_money_moment_fallback_clips(
            money_moments=[moment],
            video_duration=video_duration,
        )
        fallback_valid = validate_analysis(
            fallback_raw,
            video_duration=video_duration,
            money_moments=money_moments,
            limit=None,
        )
        if fallback_valid:
            result.append(fallback_valid[0])

    # Validate/dedupe WITHOUT truncating first. Truncating here would recreate
    # the exact bug this guard is meant to prevent.
    all_valid = validate_analysis(
        result,
        video_duration=video_duration,
        money_moments=money_moments,
        limit=None,
    )
    if not all_valid:
        return []

    reserved: list[dict[str, Any]] = []

    def _candidate_rank(clip: dict[str, Any]) -> tuple[float, float, float]:
        anchor = clip.get("strongest_anchor", {})
        if not isinstance(anchor, dict):
            anchor = {}
        return (
            _safe_float(clip.get("score", 0.0)),
            _safe_float(anchor.get("strength", 0.0)),
            -abs(_safe_float(clip.get("duration", 0.0)) - TARGET_CLIP_IDEAL),
        )

    # Reserve the best available candidate for EACH uncovered decisive event.
    for event_id in sorted(injected_event_ids):
        covering = [
            clip
            for clip in all_valid
            if _clip_covers_visual_event_id(clip, event_id, money_moments)
        ]
        if not covering:
            continue
        covering.sort(key=_candidate_rank, reverse=True)
        chosen = covering[0]
        if not any(existing is chosen for existing in reserved):
            reserved.append(chosen)

    # If there are somehow more unique decisive stories than candidate slots,
    # keep the strongest reserved ones rather than silently dropping them by
    # ordinary score sorting.
    reserved.sort(key=_candidate_rank, reverse=True)
    final = reserved[:MAX_CLIPS]

    for clip in all_valid:
        if len(final) >= MAX_CLIPS:
            break
        if any(existing is clip for existing in final):
            continue
        final.append(clip)

    return final


def _judge_ignored_decisive_visual(
    judgement: dict[str, Any],
    clips: list[dict[str, Any]],
    money_moments: list[dict[str, Any]],
    injected_event_ids: set[str],
) -> set[str]:
    """Return decisive event IDs available in candidates but omitted by selection."""
    if not injected_event_ids or not clips:
        return set()

    try:
        selected_index = int(judgement.get("selected_candidate_index", 1)) - 1
    except (TypeError, ValueError):
        selected_index = 0
    if not 0 <= selected_index < len(clips):
        selected_index = 0

    selected = clips[selected_index]
    available_ids = {
        event_id
        for event_id in injected_event_ids
        if any(
            _clip_covers_visual_event_id(clip, event_id, money_moments)
            for clip in clips
        )
    }
    selected_ids = {
        event_id
        for event_id in available_ids
        if _clip_covers_visual_event_id(selected, event_id, money_moments)
    }
    return available_ids - selected_ids


# ============================================================
# FULL ANALYSIS
# ============================================================

def analyze_transcript(
    transcript: dict[str, Any],
    video_report: dict[str, Any] | None = None,
) -> dict[str, Any]:

    duration = _video_duration(
        transcript
    )

    if duration <= 0:
        raise RuntimeError(
            "Transcript source.duration içermiyor."
        )

    transcript_text = (
        build_timestamped_transcript(
            transcript
        )
    )

    if not transcript_text:
        raise RuntimeError(
            "Transcript içinde analiz edilecek metin yok."
        )

    visual_events = build_visual_events(
        video_report,
        video_duration=duration,
    )

    # V26 SOURCE PEAK ORIGIN GUARD: sparse frame sampling can miss the exact
    # physical state change even when a huge source transient clearly exists.
    # Add only very strong deterministic regions as required-review evidence.
    source_peak_guards = build_source_peak_guard_events(
        video_report,
        visual_events,
        video_duration=duration,
    )
    if source_peak_guards:
        visual_events.extend(source_peak_guards)
        visual_events.sort(
            key=lambda event: (
                _safe_float(event.get("start", 0.0)),
                _safe_float(event.get("end", 0.0)),
                str(event.get("event_id", "")),
            )
        )
        print(
            "   ⚡ V26 source peak guards: "
            + ", ".join(
                f"{event.get('event_id')}@{_safe_float(event.get('start', 0.0)):.2f}s"
                for event in source_peak_guards
            )
        )

    raw_pass = request_money_moments(
        transcript=transcript,
        visual_events=visual_events,
    )

    raw_moments = raw_pass.get(
        "moments",
        [],
    )

    if not isinstance(
        raw_moments,
        list,
    ):
        raw_moments = []

    moments = validate_money_moments(
        raw_moments,
        video_duration=duration,
    )

    audit = validate_visual_event_audit(
        raw_pass.get(
            "visual_event_audit",
            [],
        ),
        visual_events,
    )

    audited_ids = {
        item[
            "event_id"
        ]
        for item in audit
    }

    required_ids = required_visual_event_ids(
        visual_events
    )

    missing_ids = sorted(
        required_ids
        - audited_ids
    )

    # Important visual facts are not allowed to silently disappear.
    # One repair pass is allowed. If that pass itself errors or still omits rows,
    # finish the audit deterministically instead of killing an otherwise valid run.
    if missing_ids:
        try:
            repair = request_visual_audit_repair(
                transcript=transcript,
                visual_events=visual_events,
                current_moments=moments,
                current_audit=audit,
                missing_event_ids=missing_ids,
            )

            extra_raw = repair.get(
                "additional_moments",
                [],
            )
            if not isinstance(extra_raw, list):
                extra_raw = []

            extra = validate_money_moments(
                extra_raw,
                video_duration=duration,
            )
            moments = validate_money_moments(
                moments + extra,
                video_duration=duration,
            )

            repair_audit = validate_visual_event_audit(
                repair.get(
                    "visual_event_audit",
                    [],
                ),
                visual_events,
            )

            merged_audit: dict[str, dict[str, str]] = {
                item["event_id"]: item
                for item in audit
                if isinstance(item, dict) and item.get("event_id")
            }
            for item in repair_audit:
                merged_audit[item["event_id"]] = item

            audit = list(merged_audit.values())
            audited_ids = {
                item["event_id"]
                for item in audit
                if isinstance(item, dict) and item.get("event_id")
            }
            still_missing = sorted(required_ids - audited_ids)

            if still_missing:
                print(
                    "   ⚠️ Visual audit repair bazı event'leri yine atladı; "
                    "deterministic audit fallback uygulanıyor: "
                    + ", ".join(still_missing)
                )
                audit = _complete_visual_audit_fallback(
                    audit=audit,
                    visual_events=visual_events,
                    missing_event_ids=still_missing,
                )
        except Exception as error:
            print(
                "   ⚠️ Visual audit repair kullanılamadı; hard-fail yerine "
                "deterministic audit fallback uygulanıyor. "
                + str(error)[:220]
            )
            audit = _complete_visual_audit_fallback(
                audit=audit,
                visual_events=visual_events,
                missing_event_ids=missing_ids,
            )

        audited_ids = {
            item["event_id"]
            for item in audit
            if isinstance(item, dict) and item.get("event_id")
        }

    # V23 catastrophic visual-payoff guard.  A high-confidence object break,
    # destruction, crash, fall, impact, etc. is guaranteed to reach the clip
    # composer as a CANDIDATE even if the transcript/scout did not turn it into
    # a money moment. Terra still makes the final choice.
    moments, injected_visual_event_ids = _inject_decisive_visual_money_moments(
        money_moments=moments,
        visual_events=visual_events,
        video_duration=duration,
    )
    if injected_visual_event_ids:
        upgraded_audit: list[dict[str, str]] = []
        for row in audit:
            item = dict(row)
            event_id = normalize_spaces(str(item.get("event_id", "")))
            if event_id in injected_visual_event_ids and item.get("decision") == "reject":
                item["decision"] = "secondary"
                item["reason"] = (
                    "V23 decisive visual-event candidate guard: factual event must reach final Terra judgement; "
                    "this does not force final selection."
                )
            upgraded_audit.append(item)
        audit = upgraded_audit
        print(
            "   👁️ V23 decisive visual candidates: "
            + ", ".join(sorted(injected_visual_event_ids))
        )

    if not moments:
        raise RuntimeError(
            "Terra video içinde klip kurabileceği belirgin bir money moment bulamadı."
        )

    raw_clips = request_clips(
        transcript=transcript,
        money_moments=moments,
        visual_events=visual_events,
        visual_event_audit=audit,
    )

    clips = validate_analysis(
        raw_clips,
        video_duration=duration,
        money_moments=moments,
    )

    if not clips:
        print(
            "   ⚠️ Model clip sınırları geçersizdi; "
            "onaylı money moment zamanlarından güvenli sınırlar yeniden kuruluyor."
        )

        fallback_raw = _build_money_moment_fallback_clips(
            money_moments=moments,
            video_duration=duration,
        )

        clips = validate_analysis(
            fallback_raw,
            video_duration=duration,
            money_moments=moments,
        )

    if not clips:
        print(
            "   ⚠️ Boundary recovery ikinci katmana geçti; "
            "en güçlü money moment etrafında canonical clip kuruluyor."
        )
        emergency_clip = _build_emergency_money_moment_clip(
            money_moments=moments,
            video_duration=duration,
        )
        if emergency_clip is not None:
            clips = [emergency_clip]

    if not clips:
        moment_spans = ", ".join(
            f"{_safe_float(item.get('start', 0.0)):.2f}-{_safe_float(item.get('end', 0.0)):.2f}"
            for item in moments[:5]
            if isinstance(item, dict)
        )
        raise RuntimeError(
            "Money moment bulundu ancak hiçbir güvenli clip sınırı kurulamadı. "
            f"timeline_duration={duration:.3f}s, money_moments=[{moment_spans}]"
        )

    # If the model clip composer somehow omitted an injected decisive visual
    # event, add one deterministic valid candidate around that exact event.
    # This prevents a door break/destruction payoff from disappearing before
    # Terra ever has a chance to compare it with talking-only candidates.
    clips = _ensure_decisive_visual_clip_candidates(
        clips=clips,
        money_moments=moments,
        injected_event_ids=injected_visual_event_ids,
        video_duration=duration,
    )

    judgement = request_final_judge(
        transcript=transcript,
        visual_events=visual_events,
        money_moments=moments,
        clips=clips,
        reasoning_effort=CLIP_JUDGE_REASONING_EFFORT,
    )

    judge_margin = _final_judge_score_margin(judgement)
    high_escalated = False
    ignored_decisive_visual_ids = _judge_ignored_decisive_visual(
        judgement,
        clips,
        moments,
        injected_visual_event_ids,
    )
    if len(clips) > 1 and (
        judge_margin is None
        or judge_margin <= CLIP_HIGH_ESCALATION_MARGIN
        or ignored_decisive_visual_ids
    ):
        if ignored_decisive_visual_ids:
            print(
                "   ↗️ Terra High escalation: Medium seçim şu decisive visual payoff event'lerini "
                "kapsamıyor: "
                + ", ".join(sorted(ignored_decisive_visual_ids))
                + ". Final judgement bir kez High ile doğrulanıyor."
            )
        else:
            print(
                "   ↗️ Terra High escalation: Medium kararı çok yakın "
                f"(margin={judge_margin if judge_margin is not None else 'unknown'})."
            )
        judgement = request_final_judge(
            transcript=transcript,
            visual_events=visual_events,
            money_moments=moments,
            clips=clips,
            reasoning_effort=CLIP_JUDGE_ESCALATION_REASONING_EFFORT,
        )
        high_escalated = True
        judge_margin = _final_judge_score_margin(judgement)

    judgement["mimir_reasoning_effort"] = (
        CLIP_JUDGE_ESCALATION_REASONING_EFFORT if high_escalated else CLIP_JUDGE_REASONING_EFFORT
    )
    judgement["mimir_high_escalated"] = high_escalated
    judgement["mimir_top2_margin"] = judge_margin

    clips = apply_final_judgement(
        clips,
        judgement,
    )

    selected_position = 0

    for index, item in enumerate(
        clips
    ):
        if item.get(
            "terra_selected"
        ) is True:
            selected_position = index
            break

    selected_clip = clips[
        selected_position
    ]

    # V16: do not spend a Terra story-expansion call on clips that are already
    # clearly complete. Short/incomplete clips still get the existing Terra pass.
    if _story_expansion_needed(selected_clip, duration):
        try:
            story_expansion = request_story_expansion(
                transcript=transcript,
                visual_events=visual_events,
                money_moments=moments,
                selected_clip=selected_clip,
            )
            expanded_selected = apply_story_expansion(
                selected_clip=selected_clip,
                proposal=story_expansion,
                money_moments=moments,
                video_duration=duration,
            )
        except Exception as error:
            print(
                "   ⚠️ Terra story-coverage pass kullanılamadı; "
                "seçilmiş highlight korunup deterministic causal-flow guard uygulanıyor. "
                + str(error)[:260]
            )
            expanded_selected = dict(selected_clip)
            expanded_selected["story_expansion"] = {
                "applied": False,
                "fallback": "deterministic_causal_story_flow",
                "error": str(error)[:260],
            }
    else:
        expanded_selected = dict(selected_clip)
        expanded_selected["story_expansion"] = {
            "applied": False,
            "skipped": True,
            "reason": "Terra final judge already rated a >=22s clip as complete.",
        }

    expanded_selected = enforce_causal_story_flow(
        transcript=transcript,
        selected_clip=expanded_selected,
        money_moments=moments,
        video_duration=duration,
    )

    # Final local edge QA: never changes the chosen story/money moment. It can
    # only repair a mid-sentence start/end or a clearly expendable tiny edge.
    try:
        boundary_proposal = request_boundary_polish(
            transcript=transcript,
            selected_clip=expanded_selected,
            video_duration=duration,
        )
        expanded_selected = apply_boundary_polish(
            selected_clip=expanded_selected,
            proposal=boundary_proposal,
            money_moments=moments,
            video_duration=duration,
        )
    except Exception as error:
        print(
            "   ⚠️ Local boundary polish kullanılamadı; Terra'nın geçerli story sınırları korunuyor. "
            + str(error)[:260]
        )
        expanded_selected = dict(expanded_selected)
        expanded_selected["boundary_polish"] = {
            "applied": False,
            "reason": "boundary polish unavailable; preserved valid Terra bounds",
            "error": str(error)[:260],
        }

    clips[
        selected_position
    ] = expanded_selected

    # Keep the Terra-selected clip first for downstream compatibility.
    clips.sort(
        key=lambda item: (
            item.get(
                "terra_selected"
            ) is not True,
            int(
                item.get(
                    "rank",
                    999,
                )
            ),
        )
    )

    return {
        "version": ANALYZER_VERSION,
        "revision": ANALYZER_REVISION,
        "mode": "terra_visual_payoff_guard_v23_r12",
        "model": CLIP_JUDGE_MODEL,
        "scout_model": CLIP_SCOUT_MODEL,
        "score_policy": SCORE_POLICY,
        "duration_policy": {
            "target_min": TARGET_CLIP_MIN,
            "target_ideal": TARGET_CLIP_IDEAL,
            "target_max": TARGET_CLIP_MAX,
            "absolute_min": MIN_CLIP_DURATION,
            "absolute_max": MAX_CLIP_DURATION,
            "short_exception_below": SHORT_CLIP_EXCEPTION_BELOW,
        },
        "editorial_owner": "Terra",
        "visual_observer_role": "facts_only_no_clip_judgement",
        "visual_event_count": len(
            visual_events
        ),
        "required_visual_audit_count": len(
            required_ids
        ),
        "visual_events": visual_events,
        "visual_event_audit": audit,
        "money_moment_count": len(
            moments
        ),
        "money_moments": moments,
        "terra_final_judge": judgement,
        "terra_high_escalated": high_escalated,
        "clip_count": len(
            clips
        ),
        "clips": clips,
    }


# ============================================================
# SAVE
# ============================================================

def get_output_path(
    transcript_path: str | Path,
) -> Path:

    transcript_path = Path(
        transcript_path
    ).expanduser().resolve()

    return (
        ANALYSIS_DIR
        / f"{transcript_path.stem}_clips.json"
    ).resolve()


def save_analysis(
    transcript_path: str | Path,
    analysis: dict[str, Any],
    video_report_path: str | Path | None = None,
) -> Path:

    transcript_path = Path(
        transcript_path
    ).expanduser().resolve()

    output_path = get_output_path(
        transcript_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    package = dict(
        analysis
    )

    package[
        "inputs"
    ] = {
        "transcript": str(
            transcript_path
        ),

        "video_brain_report": (
            str(
                Path(
                    video_report_path
                ).expanduser().resolve()
            )
            if video_report_path
            else None
        ),

        "clip_selection_owner": (
            "Terra"
        ),

        "visual_observer_role": (
            "facts_only_no_editorial_judgement"
        ),
    }

    temp = output_path.with_suffix(
        output_path.suffix
        + ".tmp"
    )

    temp.write_text(
        json.dumps(
            package,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(
        temp,
        output_path,
    )

    return output_path


# ============================================================
# SUMMARY
# ============================================================

def print_analysis_summary(
    analysis: dict[str, Any],
) -> None:

    clips = analysis.get(
        "clips",
        [],
    )

    print()
    print(
        "=" * 72
    )

    print(
        "🔥 TERRA STORY-ARC ANALYZER SONUÇLARI"
    )

    print(
        "=" * 72
    )

    print(
        "👁️ Factual visual events: "
        f"{analysis.get('visual_event_count', 0)}"
    )

    print(
        "🎯 Money moments: "
        f"{analysis.get('money_moment_count', 0)}"
    )

    audit = analysis.get(
        "visual_event_audit",
        [],
    )

    if isinstance(
        audit,
        list,
    ) and audit:

        print()
        print(
            "🔎 TERRA VISUAL EVENT AUDIT"
        )

        event_map = {
            str(
                item.get(
                    "event_id",
                    "",
                )
            ): item
            for item in analysis.get(
                "visual_events",
                [],
            )
            if isinstance(
                item,
                dict,
            )
        }

        for row in audit:

            if not isinstance(
                row,
                dict,
            ):
                continue

            event = event_map.get(
                str(
                    row.get(
                        "event_id",
                        "",
                    )
                ),
                {},
            )

            print(
                "   "
                f"{row.get('event_id', '')} | "
                f"{row.get('decision', '')} | "
                f"{event.get('type', '')} | "
                f"{event.get('description', '')}"
            )

            print(
                "      Terra: "
                f"{row.get('reason', '')}"
            )

    for index, clip in enumerate(
        clips,
        start=1,
    ):

        anchor = clip.get(
            "strongest_anchor",
            {},
        )

        print()
        print(
            f"🎬 KLİP {index}"
        )

        print(
            "⭐ Rank score: "
            f"{float(clip.get('score', 0.0)):.1f}/10 "
            "(gate yok)"
        )

        print(
            "⏱️ "
            f"{float(clip['start']):.2f}"
            " → "
            f"{float(clip['end']):.2f}"
            " "
            f"({float(clip['duration']):.2f}s)"
        )

        print(
            f"📛 {clip.get('title', '')}"
        )

        print(
            "🧲 Hook: "
            f"{clip.get('hook_text', '')}"
        )

        print(
            "💎 En can alıcı nokta: "
            f"{float(anchor.get('start', 0.0)):.2f}"
            " → "
            f"{float(anchor.get('end', 0.0)):.2f}"
            " | "
            f"{float(anchor.get('strength', 0.0)):.1f}/10"
        )

        covered_ids = clip.get(
            "covered_moment_ids",
            [],
        )

        print(
            "🧩 Covered money moments: "
            f"{len(covered_ids) if isinstance(covered_ids, list) else 0}"
        )

        if clip.get(
            "terra_selected"
        ) is True:
            expansion = clip.get(
                "story_expansion",
                {},
            )

            if isinstance(
                expansion,
                dict,
            ):
                print(
                    "📏 Story expansion: "
                    f"{float(expansion.get('original_duration', clip['duration'])):.2f}s"
                    " → "
                    f"{float(expansion.get('final_duration', clip['duration'])):.2f}s"
                )

            print(
                "🎯 Target Short: "
                f"{TARGET_CLIP_MIN:.0f}-{TARGET_CLIP_MAX:.0f}s "
                f"(ideal ~{TARGET_CLIP_IDEAL:.0f}s)"
            )

        print(
            "🛡️ Protected range: "
            f"{len(clip.get('must_keep_ranges', []))}"
        )

        for protected in clip.get(
            "must_keep_ranges",
            [],
        ):

            print(
                "   🛡️ "
                f"{float(protected['start']):.2f}"
                " → "
                f"{float(protected['end']):.2f}"
                " | "
                f"{protected.get('reason', '')}"
            )


# ============================================================
# PUBLIC ENTRY
# ============================================================

def create_clip_analysis(
    transcript_path: str | Path,
    video_report_path: str | Path | None = None,
) -> dict[str, Any]:

    transcript_path = Path(
        transcript_path
    ).expanduser().resolve()

    print()
    print(
        "=" * 72
    )
    print(
        "🔥 MIMIR EDITOR / CLIP ANALYZER — TERRA V23"
    )
    print(
        "=" * 72
    )
    print(
        f"🤖 Scout: {CLIP_SCOUT_MODEL} [{CLIP_SCOUT_REASONING_EFFORT}] | Judge: {CLIP_JUDGE_MODEL} [{CLIP_JUDGE_REASONING_EFFORT}]"
    )
    print(
        f"🧠 Routing: Luna scout → Terra final judge"
    )
    print(
        "🎯 Can alıcılık + final clip kararı: SADECE TERRA"
    )
    print(
        "👁️ Visual observer: yalnızca ne olduğunu raporlar"
    )

    transcript = load_json(
        transcript_path
    )

    visual_report = load_visual_report(
        video_report_path
    )

    print(
        "👁️ Whole-VOD visual facts: "
        + (
            "VAR"
            if visual_report is not None
            else "YOK — transcript-only fallback"
        )
    )

    analysis = analyze_transcript(
        transcript,
        video_report=visual_report,
    )

    output_path = save_analysis(
        transcript_path=transcript_path,
        analysis=analysis,
        video_report_path=video_report_path,
    )

    print_analysis_summary(
        analysis
    )

    print()
    print(
        f"✅ Hazır:\n{output_path}"
    )

    return analysis



if __name__ == "__main__":

    transcript_path = input(
        "Transcript JSON yolunu gir: "
    ).strip().strip('"')

    try:

        create_clip_analysis(
            transcript_path
        )

    except Exception as error:

        print()
        print(
            "❌ TERRA CLIP ANALYZER HATASI:"
        )

        print(
            error
        )