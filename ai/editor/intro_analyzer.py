from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ai.openai_client import client
from ai.model_config import (
    INTRO_DRAFT_MODEL,
    INTRO_DRAFT_REASONING_EFFORT,
    INTRO_JUDGE_MODEL,
    INTRO_JUDGE_REASONING_EFFORT,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

VOD_OUTPUT_DIR = (
    PROJECT_ROOT
    / "vod_output"
)

INTRO_DIR = (
    VOD_OUTPUT_DIR
    / "intros"
)


# ============================================================
# CONFIG
# ============================================================

INTRO_ANALYZER_VERSION = 1
INTRO_ANALYZER_REVISION = 7

# No intro is better than a bad intro.
MIN_RECOMMENDED_SCORE = 8.0

# Terra tek çağrıda birkaç farklı hook üretir, sonra ayrı bir final-judge
# çağrısıyla aralarından seçim yapar.
CANDIDATE_COUNT = 4
MAX_AI_ATTEMPTS = 2
MAX_FINAL_REPAIR_ROUNDS = 1
MIN_GEMINI_EVENT_CONFIDENCE = 0.55

MIN_INTRO_WORDS = 3
MAX_INTRO_WORDS = 8

MAX_INTRO_CHARACTERS = 46

# Hook artık ayrı bir kart değil; seçilmiş moving teaser'ın üstünde duruyor.
# Süre yalnızca text hold hint'idir, videoya ekstra intro süresi eklemez.
MIN_INTRO_DURATION = 1.15
MAX_INTRO_DURATION = 2.10


# ============================================================
# GENERIC CLICKBAIT BLACKLIST
# ============================================================

BANNED_GENERIC_PHRASES = (
    "YOU WON'T BELIEVE",
    "YOU WONT BELIEVE",
    "WAIT FOR IT",
    "WATCH TILL THE END",
    "WATCH UNTIL THE END",
    "KEEP WATCHING",
    "THIS IS CRAZY",
)


# ============================================================
# STRUCTURED OUTPUT
# ============================================================

CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "minItems": CANDIDATE_COUNT,
            "maxItems": CANDIDATE_COUNT,
            "items": {
                "type": "object",
                "properties": {
                    "intro_text": {"type": "string"},
                    "tone": {
                        "type": "string",
                        "enum": [
                            "shock",
                            "weird",
                            "confusion",
                            "conflict",
                            "absurdity",
                            "hype",
                            "funny",
                            "awkward",
                            "reveal",
                            "curiosity",
                        ],
                    },
                    "copy_strategy": {
                        "type": "string",
                        "enum": [
                            "reaction_setup",
                            "disbelief",
                            "weirdness",
                            "conflict_setup",
                            "absurdity",
                            "curiosity_gap",
                            "creator_callout",
                            "unexpected_quote",
                            "stakes",
                        ],
                    },
                    "uses_specific_name": {"type": "boolean"},
                    "specific_name": {"type": "string"},
                    "reason": {"type": "string"},
                    "curiosity_target": {"type": "string"},
                },
                "required": [
                    "intro_text",
                    "tone",
                    "copy_strategy",
                    "uses_specific_name",
                    "specific_name",
                    "reason",
                    "curiosity_target",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "recommended": {"type": "boolean"},
        "selected_candidate_index": {
            "type": "integer",
            "minimum": 0,
            "maximum": CANDIDATE_COUNT,
        },
        "overall_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 10,
        },
        "reason": {"type": "string"},
        "rejection_reason": {"type": "string"},
        "gemini_support_used": {"type": "boolean"},
        "candidate_scores": {
            "type": "array",
            "minItems": 1,
            "maxItems": CANDIDATE_COUNT,
            "items": {
                "type": "object",
                "properties": {
                    "candidate_index": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": CANDIDATE_COUNT,
                    },
                    "overall_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "specificity_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "curiosity_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "teaser_pair_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "non_spoiler_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "visual_fit_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "reason": {"type": "string"},
                },
                "required": [
                    "candidate_index",
                    "overall_score",
                    "specificity_score",
                    "curiosity_score",
                    "teaser_pair_score",
                    "non_spoiler_score",
                    "visual_fit_score",
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "recommended",
        "selected_candidate_index",
        "overall_score",
        "reason",
        "rejection_reason",
        "gemini_support_used",
        "candidate_scores",
    ],
    "additionalProperties": False,
}


# ============================================================
# AI INSTRUCTIONS
# ============================================================

INTRO_INSTRUCTIONS = """
You are MIMIR's hook-copy candidate generator for a high-retention short-form editing system.

Generate excellent candidates, but a separate Terra final judge makes the FINAL intro-copy decision.

IMPORTANT:
The teaser scene has already been selected.
You are NOT selecting a new clip and you are NOT rewriting the teaser.

The real render structure is:

    [MOVING CLEAN PEAK FOOTAGE + REAL AUDIO + YOUR HOOK TEXT]
                    ↓
                HARD RESTART
                    ↓
        [MAIN CLIP FROM TRUE BEGINNING]

There is NO separate frozen intro card and there are no speech captions during
the cold open: your line is the only text on the moving peak.

TRUTH RULES (checked by software; violating candidates are discarded):
- every fact must be supported by the transcript or the visual evidence;
- never state a number that is not in the evidence;
- never name a person, channel or brand unless it is a VERIFIED name given to
  you; describe people by role ("HE", "HIS FRIEND", "THE STREAMER") instead;
- no generic clickbait ("YOU WON'T BELIEVE", "WAIT FOR IT", "WATCH WHAT HAPPENS").

Your job is to find the ONE strongest ATTENTION TARGET inside the evidence
and write one short hook line that points the viewer toward that target
without revealing the payoff.

============================================================
LOCKED PEAK REGION (MANDATORY)
============================================================

A LOCKED PEAK REGION has been determined by the peak-selection system.
This is the strongest audio/visual event in the clip and MUST appear
inside the intro window. The intro window is defined by the teaser
timestamps (teaser_start to teaser_end from the locked_peak data).

Your hook line MUST work with this locked peak region. You cannot
change the intro window - it is fixed to cover the locked peak.

The LOCKED PEAK REGION data includes:
- peak_id: the specific peak event
- peak_start, peak_end: the core event time range
- teaser_start, teaser_end: the full intro window that will be rendered
- combined_score: audio+visual strength
- signals: event types (explosion, crash, reaction, reveal, etc.)
- visual_description: what happens at the peak

Your curiosity_target and hook line must be compatible with the
actual peak event that will be shown in the intro.


============================================================
STEP 1 — FIND THE ATTENTION TARGET
============================================================

Before writing the line, decide what the viewer should become curious about.

The attention target must be SPECIFIC to this exact clip.

Strong attention targets include:

- an unanswered question
- a contradiction
- a strange claim
- a risky guess
- a conflict
- a specific misunderstanding
- an unexpected comparison
- the cause of a reaction
- a reveal that has not happened yet
- a statement whose consequence is still unresolved

Weak targets include:

- \"what happens next\"
- \"his reaction\"
- \"something crazy\"
- \"this moment\"
- generic hype with no clip-specific subject

Use the provided:
- selected teaser
- full clip transcript
- timeline hook
- payoff region
- caption highlights
- teaser reason
- viewer question

to decide the target.

The field curiosity_target MUST describe that specific unresolved target.
It should be concrete enough that another editor can understand what the
intro is trying to make the viewer wait for.
"""



FINAL_JUDGE_INSTRUCTIONS = """
You are TERRA acting as the editor-in-chief and FINAL quality gate.

You receive:
- the selected teaser
- transcript/timeline editorial-core evidence
- optional Gemini visual-support evidence
- one to four validated hook candidates

Your job is NOT to be polite to the candidate writer.
Your job is to protect the final Short.

Core rule:

    NO HEADLINE IS BETTER THAN A BAD HEADLINE.

Rejecting every candidate does NOT remove the cold open: the moving peak
footage still plays with its real audio, just without text. So never accept a
candidate merely to have a headline.

Reject any candidate that states something the transcript/visual evidence does
not show, contradicts what the peak footage shows, invents or guesses a name,
invents a number, or spoils the payoff.

Score every candidate from 0 to 10.

Evaluate:
1. specificity to this exact clip
2. curiosity gap strength
3. how well it pairs with the selected teaser
4. spoiler safety / honesty
5. visual fit with Gemini support when Gemini evidence exists

The overall score is your final editorial score, not a simple arithmetic
average.

A candidate below 8.0 is NOT good enough to render.

Set recommended=true ONLY when:
- one candidate scores at least 8.0
- it clearly improves the teaser
- it does not spoil the payoff
- it is not generic
- it is natural short-form English
- its target is supported by the evidence

If the best candidate is below 8.0:
- recommended=false
- selected_candidate_index=0
- rejection_reason must explain why
- do NOT force a winner

If Gemini support is unavailable, judge visual fit conservatively from the
teaser/timeline evidence and set gemini_support_used=false.

Gemini is advisory only.
Terra has the final decision.

When rejecting, rejection_reason must be actionable: say what is missing
(specificity, curiosity, teaser pairing, spoiler safety, or visual grounding)
so a single repair generation can fix it.
""".strip()


# ============================================================
# GENERIC HELPERS
# ============================================================

def load_json(
    path: str | Path,
) -> dict[str, Any]:

    path = Path(
        path
    ).resolve()

    if not path.exists():

        raise FileNotFoundError(
            f"JSON bulunamadı:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
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
        float(
            value
        ),
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
            value,
            maximum,
        ),
    )


def normalize_spaces(
    value: str,
) -> str:

    return " ".join(
        str(
            value
        )
        .strip()
        .split()
    )


def normalize_for_compare(
    value: str,
) -> str:

    value = normalize_spaces(
        value
    ).casefold()

    value = re.sub(
        r"[^a-z0-9']+",
        " ",
        value,
    )

    return " ".join(
        value.split()
    )


# ============================================================
# PACKAGE VALIDATION
# ============================================================

def validate_teaser_package(
    package: dict[str, Any],
) -> None:

    if package.get(
        "version"
    ) != 1:

        raise RuntimeError(
            "Intro Analyzer, Teaser Analyzer V1 çıktısı bekliyor."
        )

    teasers = package.get(
        "teasers"
    )

    if (
        not isinstance(
            teasers,
            list,
        )
        or not teasers
    ):

        raise RuntimeError(
            "Teaser JSON içinde 'teasers' listesi bulunamadı."
        )

    inputs = package.get(
        "inputs"
    )

    if not isinstance(
        inputs,
        dict,
    ):

        raise RuntimeError(
            "Teaser JSON içinde 'inputs' bulunamadı."
        )

    for key in (
        "timeline",
        "transcript",
    ):

        value = inputs.get(
            key
        )

        if (
            not isinstance(
                value,
                str,
            )
            or not value.strip()
        ):

            raise RuntimeError(
                f"Teaser JSON inputs.{key} eksik."
            )


def validate_timeline(
    timeline: dict[str, Any],
) -> None:

    if timeline.get(
        "version"
    ) != 3:

        raise RuntimeError(
            "Intro Analyzer yalnızca Timeline V3 ile çalışır."
        )

    timelines = timeline.get(
        "timelines"
    )

    if (
        not isinstance(
            timelines,
            list,
        )
        or not timelines
    ):

        raise RuntimeError(
            "Timeline V3 içinde 'timelines' listesi bulunamadı."
        )


def validate_transcript(
    transcript: dict[str, Any],
) -> None:

    segments = transcript.get(
        "segments"
    )

    if (
        not isinstance(
            segments,
            list,
        )
        or not segments
    ):

        raise RuntimeError(
            "Transcript segment içermiyor."
        )


# ============================================================
# INPUT PATHS
# ============================================================

def get_referenced_paths(
    package: dict[str, Any],
) -> tuple[
    Path,
    Path,
]:

    inputs = package[
        "inputs"
    ]

    timeline_path = Path(
        inputs[
            "timeline"
        ]
    ).resolve()

    transcript_path = Path(
        inputs[
            "transcript"
        ]
    ).resolve()

    if not timeline_path.exists():

        raise FileNotFoundError(
            "Teaser JSON'un referans verdiği Timeline bulunamadı:\n"
            f"{timeline_path}"
        )

    if not transcript_path.exists():

        raise FileNotFoundError(
            "Teaser JSON'un referans verdiği Transcript bulunamadı:\n"
            f"{transcript_path}"
        )

    return (
        timeline_path,
        transcript_path,
    )


# ============================================================
# GET TEASER
# ============================================================

def get_teaser(
    package: dict[str, Any],
    clip_index: int,
) -> dict[str, Any]:

    for position, teaser in enumerate(
        package[
            "teasers"
        ],
        start=1,
    ):

        if not isinstance(
            teaser,
            dict,
        ):

            continue

        try:

            current_index = int(
                teaser.get(
                    "clip_index",
                    position,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            current_index = (
                position
            )

        if (
            current_index
            == clip_index
        ):

            return teaser

    raise IndexError(
        f"Teaser JSON içinde "
        f"clip_index={clip_index} bulunamadı."
    )


# ============================================================
# GET TIMELINE CLIP
# ============================================================

def get_timeline_clip(
    timeline: dict[str, Any],
    clip_index: int,
) -> dict[str, Any]:

    for position, clip in enumerate(
        timeline[
            "timelines"
        ],
        start=1,
    ):

        if not isinstance(
            clip,
            dict,
        ):

            continue

        try:

            current_index = int(
                clip.get(
                    "clip_index",
                    position,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            current_index = (
                position
            )

        if (
            current_index
            == clip_index
        ):

            return clip

    raise IndexError(
        f"Timeline içinde "
        f"clip_index={clip_index} bulunamadı."
    )


# ============================================================
# VERIFIED CREATOR NAME
# ============================================================

def resolve_verified_creator_name(
    package: dict[str, Any],
    timeline: dict[str, Any],
    teaser: dict[str, Any],
    manual_creator_name: str | None = None,
) -> str | None:

    # CLI'dan kullanıcı açıkça isim girdiyse
    # en güvenilir kaynak bu.
    if manual_creator_name:

        value = normalize_spaces(
            manual_creator_name
        )

        return (
            value
            or None
        )

    # Gelecekte source metadata'ya creator_name
    # eklersek otomatik okuyabilsin.
    possible_containers = [
        package,
        teaser,
        timeline.get(
            "source",
            {},
        ),
    ]

    keys = (
        "creator_name",
        "streamer_name",
        "channel_name",
    )

    for container in possible_containers:

        if not isinstance(
            container,
            dict,
        ):

            continue

        for key in keys:

            value = container.get(
                key
            )

            if (
                isinstance(
                    value,
                    str,
                )
                and value.strip()
            ):

                return normalize_spaces(
                    value
                )

    return None


# ============================================================
# CLIP TRANSCRIPT
# ============================================================

def build_clip_transcript(
    transcript: dict[str, Any],
    clip: dict[str, Any],
) -> str:

    source = clip.get(
        "source",
        {},
    )

    try:

        clip_start = float(
            source[
                "absolute_start"
            ]
        )

        clip_end = float(
            source[
                "absolute_end"
            ]
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as error:

        raise RuntimeError(
            "Timeline clip source absolute timestamps eksik."
        ) from error

    lines: list[str] = []

    for segment in transcript.get(
        "segments",
        [],
    ):

        if not isinstance(
            segment,
            dict,
        ):

            continue

        try:

            start = float(
                segment.get(
                    "start",
                    0,
                )
            )

            end = float(
                segment.get(
                    "end",
                    start,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            continue

        text = normalize_spaces(
            str(
                segment.get(
                    "text",
                    "",
                )
            )
        )

        if not text:

            continue

        overlap_start = max(
            start,
            clip_start,
        )

        overlap_end = min(
            end,
            clip_end,
        )

        if (
            overlap_end
            <= overlap_start
        ):

            continue

        relative_start = max(
            0.0,
            start - clip_start,
        )

        relative_end = max(
            relative_start,
            end - clip_start,
        )

        lines.append(
            (
                f"[{relative_start:.2f}"
                f"-"
                f"{relative_end:.2f}] "
                f"{text}"
            )
        )

    if not lines:

        raise RuntimeError(
            "Clip için transcript context oluşturulamadı."
        )

    return "\n".join(
        lines
    )


# ============================================================
# GEMINI VIDEO BRAIN SUPPORT
# ============================================================

def load_video_support(
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

    if not isinstance(
        report,
        dict,
    ):
        return None

    return report


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


def build_gemini_support_context(
    report: dict[str, Any] | None,
    teaser: dict[str, Any],
    clip: dict[str, Any],
) -> str:
    """
    Gemini is factual support only.

    Unlike the old implementation, V8 does not look only at a tiny window
    around the teaser. Terra also sees the important factual events across the
    selected clean clip so the hook can point toward the actual payoff.
    """

    if not isinstance(
        report,
        dict,
    ):
        return (
            "GEMINI VISUAL SUPPORT:\n"
            "Unavailable. Terra must judge from transcript/timeline/teaser evidence only."
        )

    edited = teaser.get(
        "edited",
        {},
    )

    if not isinstance(
        edited,
        dict,
    ):
        edited = {}

    teaser_start = _safe_float(
        edited.get(
            "teaser_start",
            0.0,
        )
    )

    teaser_end = _safe_float(
        edited.get(
            "teaser_end",
            teaser_start,
        )
    )

    teaser_window_start = max(
        0.0,
        teaser_start - 0.70,
    )

    teaser_window_end = max(
        teaser_window_start,
        teaser_end + 0.70,
    )

    payoff = clip.get(
        "payoff",
        {},
    )

    if not isinstance(
        payoff,
        dict,
    ):
        payoff = {}

    payoff_start = _safe_float(
        payoff.get(
            "edited_start",
            payoff.get(
                "source_start",
                -1.0,
            ),
        ),
        -1.0,
    )

    payoff_end = _safe_float(
        payoff.get(
            "edited_end",
            payoff.get(
                "source_end",
                payoff_start,
            ),
        ),
        payoff_start,
    )

    lines: list[str] = [
        "GEMINI VISUAL SUPPORT:",
        "Facts only. Terra decides significance and final copy.",
        f"Teaser window: {teaser_start:.2f}-{teaser_end:.2f}s",
    ]

    if (
        payoff_start >= 0
        and payoff_end >= payoff_start
    ):
        lines.append(
            f"Known payoff window: {payoff_start:.2f}-{payoff_end:.2f}s"
        )

    summary = normalize_spaces(
        str(
            report.get(
                "summary",
                "",
            )
        )
    )

    if summary:
        lines.append(
            f"Visual summary: {summary}"
        )

    visual_events = report.get(
        "visual_events",
        [],
    )

    all_events: list[
        tuple[
            float,
            float,
            float,
            str,
            str,
            bool,
            bool,
        ]
    ] = []

    if isinstance(
        visual_events,
        list,
    ):
        for event in visual_events:
            if not isinstance(
                event,
                dict,
            ):
                continue

            confidence = _safe_float(
                event.get(
                    "confidence",
                    0.0,
                )
            )

            if confidence < MIN_GEMINI_EVENT_CONFIDENCE:
                continue

            start = _safe_float(
                event.get(
                    "start",
                    0.0,
                )
            )

            end = _safe_float(
                event.get(
                    "end",
                    start,
                )
            )

            event_type = normalize_spaces(
                str(
                    event.get(
                        "type",
                        "other",
                    )
                )
            )

            description = normalize_spaces(
                str(
                    event.get(
                        "description",
                        "",
                    )
                )
            )

            near_teaser = not (
                end < teaser_window_start
                or start > teaser_window_end
            )

            near_payoff = bool(
                payoff_start >= 0
                and not (
                    end < payoff_start - 0.70
                    or start > payoff_end + 0.70
                )
            )

            all_events.append(
                (
                    start,
                    end,
                    confidence,
                    event_type,
                    description,
                    near_teaser,
                    near_payoff,
                )
            )

    # Put payoff/teaser-near facts first, then confidence, then time.
    all_events.sort(
        key=lambda item: (
            not item[6],
            not item[5],
            -item[2],
            item[0],
        )
    )

    if all_events:
        lines.append(
            "High-confidence visual facts across selected clip:"
        )

        for (
            start,
            end,
            confidence,
            event_type,
            description,
            near_teaser,
            near_payoff,
        ) in all_events[:12]:

            tags: list[str] = []

            if near_teaser:
                tags.append(
                    "near_teaser"
                )

            if near_payoff:
                tags.append(
                    "near_payoff"
                )

            tag_text = (
                " [" + ",".join(tags) + "]"
                if tags
                else ""
            )

            lines.append(
                f"- {start:.2f}-{end:.2f}s | {event_type} | "
                f"conf={confidence:.2f}{tag_text} | {description}"
            )

    else:
        lines.append(
            "High-confidence visual facts: none."
        )

    intro_visual = report.get("intro_visual_support", {})
    if isinstance(intro_visual, dict):
        peak_regions = intro_visual.get("visual_peak_regions", [])
        if isinstance(peak_regions, list) and peak_regions:
            ranked_peaks: list[tuple[bool, float, float, dict[str, Any]]] = []
            for item in peak_regions:
                if not isinstance(item, dict):
                    continue
                start = _safe_float(item.get("start", 0.0))
                end = _safe_float(item.get("end", start))
                intensity = _safe_float(item.get("observable_intensity", 0.0))
                confidence = _safe_float(item.get("confidence", 0.0))
                near_teaser = not (
                    end < teaser_window_start
                    or start > teaser_window_end
                )
                ranked_peaks.append((near_teaser, intensity, confidence, item))

            ranked_peaks.sort(
                key=lambda x: (not x[0], -x[1], -x[2], _safe_float(x[3].get("start", 0.0)))
            )
            lines.append("Gemini visual peak regions:")
            for near_teaser, intensity, confidence, item in ranked_peaks[:8]:
                start = _safe_float(item.get("start", 0.0))
                end = _safe_float(item.get("end", start))
                signals = item.get("signals", [])
                signal_text = ",".join(str(x) for x in signals) if isinstance(signals, list) else ""
                description = normalize_spaces(str(item.get("description", "")))
                tag = " [near_teaser]" if near_teaser else ""
                lines.append(
                    f"- {start:.2f}-{end:.2f}s | intensity={intensity:.2f} "
                    f"conf={confidence:.2f}{tag} | {signal_text} | {description}"
                )

    support = report.get(
        "editing_support",
        {},
    )

    if isinstance(
        support,
        dict,
    ):
        for key, label in (
            (
                "reaction_times",
                "Observed reaction times",
            ),
            (
                "visual_payoff_times",
                "Observed visual-payoff times",
            ),
            (
                "scene_change_times",
                "Observed scene-change times",
            ),
        ):
            values = support.get(
                key,
                [],
            )

            if not isinstance(
                values,
                list,
            ):
                continue

            cleaned: list[str] = []

            for value in values[:20]:
                current = _safe_float(
                    value,
                    -1.0,
                )

                if current >= 0:
                    cleaned.append(
                        f"{current:.2f}"
                    )

            if cleaned:
                lines.append(
                    f"{label}: "
                    + ", ".join(
                        cleaned
                    )
                )

    quality = report.get(
        "quality",
        {},
    )

    if isinstance(
        quality,
        dict,
    ):
        confidence = _safe_float(
            quality.get(
                "analysis_confidence",
                0.0,
            )
        )

        lines.append(
            f"Gemini analysis confidence: {confidence:.2f}"
        )

    lines.append(
        "Do not invent beyond these facts. Gemini never chooses the hook."
    )

    return "\n".join(
        lines
    )


# ============================================================
# BUILD AI INPUT
# ============================================================

def build_ai_input(
    teaser: dict[str, Any],
    clip: dict[str, Any],
    clip_transcript: str,
    creator_name: str | None,
    retry_feedback: str | None = None,
) -> str:

    hook = clip.get(
        "hook",
        {},
    )

    payoff = clip.get(
        "payoff",
        {},
    )

    hook_text = ""
    hook_type = ""

    if isinstance(
        hook,
        dict,
    ):

        hook_text = normalize_spaces(
            str(
                hook.get(
                    "text",
                    "",
                )
            )
        )

        hook_type = normalize_spaces(
            str(
                hook.get(
                    "type",
                    "",
                )
            )
        )

    payoff_info = ""

    if isinstance(
        payoff,
        dict,
    ):

        payoff_info = (
            f"{payoff.get('source_start', '?')}"
            f" -> "
            f"{payoff.get('source_end', '?')}"
        )

    highlights = clip.get(
        "caption_highlights",
        [],
    )

    if isinstance(
        highlights,
        list,
    ):

        highlight_text = ", ".join(
            normalize_spaces(
                str(
                    item
                )
            )

            for item in highlights

            if normalize_spaces(
                str(
                    item
                )
            )
        )

    else:

        highlight_text = ""

    creator_text = (
        creator_name
        if creator_name
        else "NONE"
    )

    editorial = clip.get(
        "editorial",
        {},
    )

    if not isinstance(
        editorial,
        dict,
    ):
        editorial = {}

    editorial_text = json.dumps(
        editorial,
        ensure_ascii=False,
        indent=2,
    )

    multimodal_support = teaser.get("multimodal_support", {})
    if isinstance(multimodal_support, dict) and multimodal_support:
        multimodal_text = json.dumps(
            multimodal_support,
            ensure_ascii=False,
            indent=2,
        )
    else:
        multimodal_text = "NONE"

    # LOCKED PEAK - the intro window MUST include this peak
    locked_peak = teaser.get("locked_peak", {})
    if isinstance(locked_peak, dict) and locked_peak:
        locked_peak_text = json.dumps(
            locked_peak,
            ensure_ascii=False,
            indent=2,
        )
    else:
        locked_peak_text = "NONE"

    retry_section = ""

    if retry_feedback:

        retry_section = f"""

============================================================
PREVIOUS ATTEMPT WAS REJECTED
============================================================

{retry_feedback}

Generate a different, valid intro line.
""".rstrip()

    return f"""
VERIFIED CREATOR NAME:
{creator_text}

CLIP TITLE:
{normalize_spaces(str(clip.get('title', '')))}

CLIP EMOTION:
{normalize_spaces(str(clip.get('emotion', '')))}

OLD TIMELINE HOOK TYPE:
{hook_type}

OLD TIMELINE HOOK TEXT:
{hook_text}

KNOWN PAYOFF REGION (clip-relative source time):
{payoff_info}

CAPTION HIGHLIGHTS:
{highlight_text}

SELECTED REAL TEASER:
{normalize_spaces(str(teaser.get('teaser_text', '')))}

TEASER TYPE:
{normalize_spaces(str(teaser.get('teaser_type', '')))}

TEASER SCORE:
{teaser.get('score', '?')}

WHY THE TEASER WAS SELECTED:
{normalize_spaces(str(teaser.get('reason', '')))}

CURIOSITY ALREADY CREATED BY THE TEASER:
{normalize_spaces(str(teaser.get('viewer_question', '')))}

SPOILER RISK:
{normalize_spaces(str(teaser.get('spoiler_risk', '')))}

MULTIMODAL PEAK SUPPORT FOR THE SELECTED TEASER:
{multimodal_text}

LOCKED PEAK REGION (MUST BE INCLUDED IN INTRO WINDOW):
{locked_peak_text}

TERRA EDITORIAL CORE FROM CLIP SELECTION:
{editorial_text}

FULL CLIP TRANSCRIPT:
{clip_transcript}

First identify the single strongest SPECIFIC attention target.
Then write the best moving-teaser hook line for that target.

The curiosity_target field must name the unresolved thing the viewer should care about.
Do not repeat the teaser quote.
{retry_section}
""".strip()


# ============================================================
# AI CALL
# ============================================================

def _parse_structured_response(
    response: Any,
    label: str,
) -> dict[str, Any]:

    output_text = str(
        response.output_text
    ).strip()

    if not output_text:
        raise RuntimeError(
            f"AI boş {label} çıktısı döndürdü."
        )

    try:
        data = json.loads(
            output_text
        )
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"AI geçersiz {label} JSON döndürdü:\n\n"
            + output_text
        ) from error

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            f"AI {label} çıktısı object değil."
        )

    return data


def request_intro_candidates(
    teaser: dict[str, Any],
    clip: dict[str, Any],
    clip_transcript: str,
    creator_name: str | None,
    gemini_context: str,
    retry_feedback: str | None = None,
) -> list[dict[str, Any]]:

    input_text = build_ai_input(
        teaser=teaser,
        clip=clip,
        clip_transcript=clip_transcript,
        creator_name=creator_name,
        retry_feedback=retry_feedback,
    )

    input_text += (
        "\n\n"
        + gemini_context
        + "\n\n"
        + (
            f"Generate exactly {CANDIDATE_COUNT} DISTINCT hook candidates. "
            "They must use genuinely different attention angles, not tiny rewrites."
        )
    )

    response = client.responses.create(
        model=INTRO_DRAFT_MODEL,
        reasoning={
            "effort": INTRO_DRAFT_REASONING_EFFORT,
        },
        instructions=INTRO_INSTRUCTIONS,
        input=input_text,
        text={
            "format": {
                "type": "json_schema",
                "name": "viral_intro_candidates_v4",
                "strict": True,
                "schema": CANDIDATE_SCHEMA,
            }
        },
    )

    data = _parse_structured_response(
        response,
        "intro candidate",
    )

    candidates = data.get(
        "candidates",
        [],
    )

    if not isinstance(
        candidates,
        list,
    ):
        raise RuntimeError(
            "AI candidates listesi döndürmedi."
        )

    return [
        item
        for item in candidates
        if isinstance(
            item,
            dict,
        )
    ]


def request_final_judgement(
    teaser: dict[str, Any],
    clip: dict[str, Any],
    clip_transcript: str,
    creator_name: str | None,
    gemini_context: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:

    candidate_text = json.dumps(
        {
            "candidates": [
                {
                    "candidate_index": index,
                    **candidate,
                }
                for index, candidate in enumerate(
                    candidates,
                    start=1,
                )
            ]
        },
        ensure_ascii=False,
        indent=2,
    )

    evidence = build_ai_input(
        teaser=teaser,
        clip=clip,
        clip_transcript=clip_transcript,
        creator_name=creator_name,
        retry_feedback=None,
    )

    judge_input = (
        evidence
        + "\n\n"
        + gemini_context
        + "\n\nCANDIDATES TO JUDGE:\n"
        + candidate_text
        + "\n\n"
        + (
            f"HARD QUALITY THRESHOLD: {MIN_RECOMMENDED_SCORE:.1f}/10. "
            "If the best candidate is below this threshold, reject ALL intros."
        )
    )

    response = client.responses.create(
        model=INTRO_JUDGE_MODEL,
        reasoning={
            "effort": INTRO_JUDGE_REASONING_EFFORT,
        },
        instructions=FINAL_JUDGE_INSTRUCTIONS,
        input=judge_input,
        text={
            "format": {
                "type": "json_schema",
                "name": "viral_intro_final_judge_v4",
                "strict": True,
                "schema": JUDGE_SCHEMA,
            }
        },
    )

    return _parse_structured_response(
        response,
        "intro final-judge",
    )


# ============================================================
# COPY VALIDATION
# ============================================================

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "million": 1000000, "billion": 1000000000, "first": 1, "second": 2, "third": 3,
}
_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+|\]\s*)$")


def _numbers_in(text: str) -> set[str]:
    """Numeric claims of a text as canonical strings ("1,000"/"1000"/"thousand" -> "1000")."""
    values: set[str] = set()
    for token in re.findall(r"[\d][\d,.]*[kKmM]?", str(text)):
        raw = token.rstrip(".,")
        multiplier = 1
        if raw[-1:] in "kK":
            multiplier, raw = 1000, raw[:-1]
        elif raw[-1:] in "mM":
            multiplier, raw = 1000000, raw[:-1]
        try:
            number = float(raw.replace(",", "")) * multiplier
        except ValueError:
            continue
        values.add(str(int(number)) if number == int(number) else str(number))
    for word in re.findall(r"[A-Za-z]+", str(text)):
        value = _NUMBER_WORDS.get(word.casefold())
        if value is not None:
            values.add(str(value))
    return values


# Capitalized for grammar/devotion/address, not identity (generic English).
_CAPITALIZED_COMMON = frozenset({"god", "jesus", "christ", "lord", "chat", "bro", "sir", "mom", "dad", "okay"})


def evidence_proper_nouns(evidence_text: str) -> set[str]:
    """Words the evidence writes capitalized mid-sentence (names, brands, places).

    Sentence-initial words, the pronoun "I", capitalized interjections/address
    words and words the evidence also writes in lowercase are ignored: their
    capital is grammar, not identity."""
    nouns: set[str] = set()
    text = str(evidence_text)
    lowercase = {word for word in re.findall(r"\b[a-z][a-z'\-]+", text)}
    for match in re.finditer(r"[A-Za-z][A-Za-z'\-]+", text):
        word = match.group(0)
        if not word[0].isupper() or word.isupper() and len(word) <= 2:
            continue
        before = text[max(0, match.start() - 3):match.start()]
        if _SENTENCE_START.search(text[:match.start()]) or before.endswith(("\n", ": ", "] ")):
            continue
        key = word.casefold().strip("'-")
        if key in _CAPITALIZED_COMMON or key in lowercase:
            continue
        nouns.add(key)
    return nouns


def headline_grounding_problems(
    text: str,
    *,
    evidence_text: str,
    verified_names: list[str] | tuple[str, ...] = (),
) -> list[str]:
    """Deterministic truth checks for hook copy (the judge still scores quality).

    * every number the headline states must appear in the evidence;
    * a word the evidence uses as a proper noun (a person, channel, brand...)
      may appear only when it is a verified name - an ASR-heard name can be a
      mishearing, and a headline must never guess who someone is.
    """
    problems: list[str] = []
    if not str(evidence_text).strip():
        return problems
    missing_numbers = sorted(_numbers_in(text) - _numbers_in(evidence_text))
    if missing_numbers:
        problems.append("headline states number(s) not in the evidence: " + ", ".join(missing_numbers[:4]))
    verified = {
        part.casefold()
        for name in verified_names
        for part in re.findall(r"[A-Za-z][A-Za-z'\-]+", str(name))
    }
    nouns = evidence_proper_nouns(evidence_text)
    for word in re.findall(r"[A-Za-z][A-Za-z'\-]+", str(text)):
        key = word.casefold().strip("'-")
        if len(key) >= 3 and key in nouns and key not in verified:
            problems.append(f"headline uses unverified name/proper noun: {word}")
            break
    return problems


def validate_intro_choice(
    ai_result: dict[str, Any],
    teaser_text: str,
    creator_name: str | None,
    *,
    evidence_text: str = "",
    verified_names: list[str] | tuple[str, ...] = (),
) -> tuple[
    bool,
    list[str],
    str,
]:

    problems: list[str] = []

    text = normalize_spaces(
        str(
            ai_result.get(
                "intro_text",
                "",
            )
        )
    )

    if not text:

        problems.append(
            "intro_text boş"
        )

        return (
            False,
            problems,
            text,
        )

    words = text.split()

    if (
        len(
            words
        )
        < MIN_INTRO_WORDS
    ):

        problems.append(
            (
                f"intro çok kısa: "
                f"{len(words)} kelime"
            )
        )

    if (
        len(
            words
        )
        > MAX_INTRO_WORDS
    ):

        problems.append(
            (
                f"intro çok uzun: "
                f"{len(words)} kelime"
            )
        )

    if (
        len(
            text
        )
        > MAX_INTRO_CHARACTERS
    ):

        problems.append(
            (
                f"intro {len(text)} karakter; "
                f"limit {MAX_INTRO_CHARACTERS}"
            )
        )

    intro_compare = normalize_for_compare(
        text
    )

    teaser_compare = normalize_for_compare(
        teaser_text
    )

    # Aynı cümle olamaz.
    if (
        intro_compare
        and intro_compare
        == teaser_compare
    ):

        problems.append(
            "intro teaser cümlesini aynen tekrar ediyor"
        )

    # Büyük oranda teaser cümlesini yeniden yazıyorsa da reddet.
    if teaser_compare:

        intro_tokens = set(
            intro_compare.split()
        )

        teaser_tokens = set(
            teaser_compare.split()
        )

        if (
            intro_tokens
            and teaser_tokens
        ):

            overlap = (
                len(
                    intro_tokens
                    & teaser_tokens
                )
                / max(
                    1,
                    len(
                        intro_tokens
                    ),
                )
            )

            if (
                overlap
                >= 0.80
            ):

                problems.append(
                    "intro teaser metnine fazla benziyor"
                )

    curiosity_target = normalize_spaces(
        str(
            ai_result.get(
                "curiosity_target",
                "",
            )
        )
    )

    if len(curiosity_target.split()) < 3:
        problems.append(
            "curiosity_target yeterince spesifik değil"
        )

    generic_targets = (
        "what happens next",
        "what happens",
        "his reaction",
        "their reaction",
        "this moment",
        "something crazy",
    )

    if curiosity_target.casefold() in generic_targets:
        problems.append(
            "curiosity_target generic kalmış"
        )

    upper_text = (
        text.upper()
    )

    for phrase in BANNED_GENERIC_PHRASES:

        if (
            phrase
            in upper_text
        ):

            problems.append(
                (
                    "yasak generic phrase: "
                    f"{phrase}"
                )
            )

            break

    # Quote card istemiyoruz.
    if (
        '"'
        in text
        or "“"
        in text
        or "”"
        in text
    ):

        problems.append(
            "tırnak işareti kullanılmamalı"
        )

    if (
        "#"
        in text
    ):

        problems.append(
            "hashtag kullanılmamalı"
        )

    # --------------------------------------------------------
    # NAME SECURITY
    # --------------------------------------------------------

    uses_specific_name = bool(
        ai_result.get(
            "uses_specific_name",
            False,
        )
    )

    specific_name = normalize_spaces(
        str(
            ai_result.get(
                "specific_name",
                "",
            )
        )
    )

    if uses_specific_name:

        if not creator_name:

            problems.append(
                "doğrulanmış creator adı yokken özel isim kullanılmış"
            )

        elif (
            specific_name.casefold()
            != creator_name.casefold()
        ):

            problems.append(
                "specific_name doğrulanmış creator adıyla eşleşmiyor"
            )

        elif (
            creator_name.casefold()
            not in text.casefold()
        ):

            problems.append(
                "uses_specific_name=true ama isim intro_text içinde yok"
            )

    else:

        if specific_name:

            problems.append(
                "uses_specific_name=false iken specific_name boş olmalı"
            )

    problems.extend(
        headline_grounding_problems(
            text,
            evidence_text=evidence_text,
            verified_names=[name for name in (creator_name, *verified_names) if name],
        )
    )

    return (
        len(
            problems
        )
        == 0,

        problems,

        text,
    )


# ============================================================
# INTRO DURATION
# ============================================================

def calculate_intro_duration(
    text: str,
) -> float:

    text = normalize_spaces(
        text
    )

    word_count = len(
        text.split()
    )

    character_count = len(
        text
    )

    # Bu süre videoya ekstra bir kart eklemiyor.
    # Renderer hook'u moving teaser üstünde bu kadar tutmaya çalışıyor.
    if word_count <= 3:
        duration = 1.20
    elif word_count == 4:
        duration = 1.38
    elif word_count == 5:
        duration = 1.55
    elif word_count == 6:
        duration = 1.72
    elif word_count == 7:
        duration = 1.90
    else:
        duration = 2.05

    if character_count >= 34:
        duration += 0.08

    return round_time(
        clamp(
            duration,
            MIN_INTRO_DURATION,
            MAX_INTRO_DURATION,
        )
    )


# ============================================================
# FREEZE FRAME TIME
# ============================================================

def calculate_freeze_frame_time(
    teaser: dict[str, Any],
) -> float:

    edited = teaser.get(
        "edited",
        {},
    )

    if not isinstance(
        edited,
        dict,
    ):

        return 0.0

    try:

        speech_start = float(
            edited.get(
                "speech_start",
                edited.get(
                    "teaser_start",
                    0.0,
                ),
            )
        )

        speech_end = float(
            edited.get(
                "speech_end",
                edited.get(
                    "teaser_end",
                    speech_start,
                ),
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        return 0.0

    if (
        speech_end
        <= speech_start
    ):

        return round_time(
            max(
                0.0,
                speech_start,
            )
        )

    # Şimdilik teaser konuşmasının yaklaşık %42 noktasından
    # deterministic bir freeze frame alıyoruz.
    #
    # Görsel analiz olmadığı için "en iyi yüz ifadesi" gibi
    # sahte bir iddiada bulunmuyoruz.

    frame_time = (
        speech_start
        + (
            speech_end
            - speech_start
        )
        * 0.42
    )

    return round_time(
        max(
            0.0,
            frame_time,
        )
    )


# ============================================================
# ANALYZE ONE INTRO
# ============================================================

def _build_no_intro_result(
    *,
    clip: dict[str, Any],
    clip_index: int,
    creator_name: str | None,
    score: float,
    reason: str,
    gemini_support_used: bool,
    candidates: list[dict[str, Any]],
    candidate_scores: list[dict[str, Any]],
    repair_rounds_used: int,
) -> dict[str, Any]:
    """The cold open still runs (moving peak + real audio); it just carries no headline.

    A strong peak with no headline is preferable to a generic, ungrounded or
    low-scoring headline."""

    return {
        "clip_index": clip_index,
        "title": str(
            clip.get(
                "title",
                "",
            )
        ),
        "recommended": True,
        "ai_recommended": False,
        "headline_status": "none",
        "score": round(
            score,
            1,
        ),
        "intro_text": "",
        "tone": "curiosity",
        "copy_strategy": "curiosity_gap",
        "creator": {
            "verified_name": creator_name,
            "used_in_copy": False,
        },
        "reason": reason,
        "curiosity_target": "",
        "quality_gate": {
            "threshold": MIN_RECOMMENDED_SCORE,
            "accepted": True,
            "no_headline": True,
            "candidate_count": len(
                candidates
            ),
            "selected_candidate_index": 0,
            "gemini_support_used": (
                gemini_support_used
            ),
            "candidate_scores": (
                candidate_scores
            ),
            "repair_rounds_used": (
                repair_rounds_used
            ),
            "rejection_reason": reason,
        },
        "candidates": [
            {
                "candidate_index": index,
                **candidate,
            }
            for index, candidate in enumerate(
                candidates,
                start=1,
            )
        ],
        "intro": {
            "duration": 0.0,
            "word_count": 0,
            "character_count": 0,
        },
        "background": {
            "source": "edited_clip",
            "freeze_frame_time": 0.0,
            "style": "moving_teaser",
        },
        "sequence": [
            "moving_peak_without_headline",
            "hard_main_clip_restart",
        ],
        "render_plan": {
            "intro_background": "moving_teaser",
            "then_play": "main_clip_restart",
            "restart_main_clip_after_teaser": True,
            "transition": "hard_cut",
        },
    }


def _collect_valid_candidates(
    *,
    raw_candidates: list[dict[str, Any]],
    teaser_text: str,
    creator_name: str | None,
    existing: list[dict[str, Any]],
    seen_texts: set[str],
    evidence_text: str = "",
    verified_names: list[str] | tuple[str, ...] = (),
) -> tuple[
    list[dict[str, Any]],
    list[str],
]:

    added: list[
        dict[str, Any]
    ] = []

    rejection_notes: list[
        str
    ] = []

    for candidate in raw_candidates:

        valid, problems, cleaned_text = (
            validate_intro_choice(
                ai_result=candidate,
                teaser_text=teaser_text,
                creator_name=creator_name,
                evidence_text=evidence_text,
                verified_names=verified_names,
            )
        )

        key = normalize_for_compare(
            cleaned_text
        )

        if (
            valid
            and key
            and key not in seen_texts
        ):
            cleaned = dict(
                candidate
            )

            cleaned[
                "intro_text"
            ] = cleaned_text.upper()

            existing.append(
                cleaned
            )

            added.append(
                cleaned
            )

            seen_texts.add(
                key
            )

            continue

        if problems:
            rejection_notes.append(
                f"{cleaned_text or '<empty>'}: "
                + "; ".join(
                    problems
                )
            )

    return (
        added,
        rejection_notes,
    )


def _judgement_state(
    judgement: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[
    bool,
    int,
    float,
]:

    try:
        selected_index = int(
            judgement.get(
                "selected_candidate_index",
                0,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        selected_index = 0

    score = _safe_float(
        judgement.get(
            "overall_score",
            0.0,
        )
    )

    accepted = bool(
        judgement.get(
            "recommended",
            False,
        )
        and score
        >= MIN_RECOMMENDED_SCORE
        and 1
        <= selected_index
        <= len(
            candidates
        )
    )

    return (
        accepted,
        selected_index,
        score,
    )


def analyze_intro_for_clip(
    package: dict[str, Any],
    timeline: dict[str, Any],
    transcript: dict[str, Any],
    clip_index: int,
    manual_creator_name: str | None = None,
    video_support_report: dict[str, Any] | None = None,
    verified_names: list[str] | tuple[str, ...] = (),
    caption_text: str = "",
) -> dict[str, Any]:

    teaser = get_teaser(
        package=package,
        clip_index=clip_index,
    )

    clip = get_timeline_clip(
        timeline=timeline,
        clip_index=clip_index,
    )

    clip_transcript = build_clip_transcript(
        transcript=transcript,
        clip=clip,
    )

    creator_name = resolve_verified_creator_name(
        package=package,
        timeline=timeline,
        teaser=teaser,
        manual_creator_name=manual_creator_name,
    )

    gemini_context = build_gemini_support_context(
        video_support_report,
        teaser,
        clip,
    )

    gemini_support_available = isinstance(
        video_support_report,
        dict,
    )

    # Everything a headline may claim must be visible in this evidence.
    evidence_text = "\n".join(
        part for part in (
            clip_transcript,
            caption_text,
            str(teaser.get("teaser_text", "")),
            gemini_context,
        ) if part
    )

    print()
    print("=" * 72)
    print(
        f"🧲 INTRO ANALYZER V4 — CLIP {clip_index}"
    )
    print(
        f"🤖 Intro draft: {INTRO_DRAFT_MODEL} [{INTRO_DRAFT_REASONING_EFFORT}] | "
        f"final judge: {INTRO_JUDGE_MODEL} [{INTRO_JUDGE_REASONING_EFFORT}]"
    )
    print(
        "👁️ Gemini factual support: "
        + (
            "VAR"
            if gemini_support_available
            else "YOK"
        )
    )
    print("=" * 72)
    print(
        f"🎬 {clip.get('title', '')}"
    )
    print(
        f"🔥 Teaser: {teaser.get('teaser_text', '')}"
    )

    valid_candidates: list[
        dict[str, Any]
    ] = []

    seen_texts: set[
        str
    ] = set()

    retry_feedback: str | None = None

    for attempt in range(
        1,
        MAX_AI_ATTEMPTS + 1,
    ):

        print()
        print(
            "🧠 Luna hook adayları üretiyor "
            f"({attempt}/{MAX_AI_ATTEMPTS})..."
        )

        raw_candidates = request_intro_candidates(
            teaser=teaser,
            clip=clip,
            clip_transcript=clip_transcript,
            creator_name=creator_name,
            gemini_context=gemini_context,
            retry_feedback=retry_feedback,
        )

        (
            _,
            rejection_notes,
        ) = _collect_valid_candidates(
            raw_candidates=raw_candidates,
            teaser_text=str(
                teaser.get(
                    "teaser_text",
                    "",
                )
            ),
            creator_name=creator_name,
            existing=valid_candidates,
            seen_texts=seen_texts,
            evidence_text=evidence_text,
            verified_names=verified_names,
        )

        if len(
            valid_candidates
        ) >= CANDIDATE_COUNT:
            break

        retry_feedback = (
            "Produce fresh, specific hooks. The previous batch did not leave "
            "enough distinct valid choices."
        )

        if rejection_notes:
            retry_feedback += (
                "\nValidation rejections:\n- "
                + "\n- ".join(
                    rejection_notes[:8]
                )
            )

    if not valid_candidates:
        print(
            "   ⏭️ No valid, grounded hook candidate: the cold open runs without a headline."
        )
        return _build_no_intro_result(
            clip=clip,
            clip_index=clip_index,
            creator_name=creator_name,
            score=0.0,
            reason="no candidate passed validation/grounding",
            gemini_support_used=False,
            candidates=[],
            candidate_scores=[],
            repair_rounds_used=0,
        )

    candidates = valid_candidates[
        :CANDIDATE_COUNT
    ]

    repair_rounds_used = 0

    print()
    print(
        f"⚖️ Terra final judge: "
        f"{len(candidates)} aday karşılaştırılıyor..."
    )

    judgement = request_final_judgement(
        teaser=teaser,
        clip=clip,
        clip_transcript=clip_transcript,
        creator_name=creator_name,
        gemini_context=gemini_context,
        candidates=candidates,
    )

    (
        accepted,
        selected_index,
        score,
    ) = _judgement_state(
        judgement,
        candidates,
    )

    # Important reliability fix:
    # A first 7.x result is not immediately "no intro". Terra gets one focused
    # repair round based on its own judge feedback, then judges again.
    if (
        not accepted
        and MAX_FINAL_REPAIR_ROUNDS > 0
    ):

        rejection_reason = normalize_spaces(
            str(
                judgement.get(
                    "rejection_reason",
                    "",
                )
            )
        )

        judge_reason = normalize_spaces(
            str(
                judgement.get(
                    "reason",
                    "",
                )
            )
        )

        score_rows = judgement.get(
            "candidate_scores",
            [],
        )

        score_feedback: list[
            str
        ] = []

        if isinstance(
            score_rows,
            list,
        ):
            for row in score_rows[
                :CANDIDATE_COUNT
            ]:
                if not isinstance(
                    row,
                    dict,
                ):
                    continue

                score_feedback.append(
                    "#"
                    + str(
                        row.get(
                            "candidate_index",
                            "?",
                        )
                    )
                    + " "
                    + str(
                        row.get(
                            "overall_score",
                            "?",
                        )
                    )
                    + "/10: "
                    + normalize_spaces(
                        str(
                            row.get(
                                "reason",
                                "",
                            )
                        )
                    )
                )

        repair_feedback = (
            "FINAL JUDGE REJECTED THE FIRST SET.\n"
            f"Best score: {score:.1f}/10; required: "
            f"{MIN_RECOMMENDED_SCORE:.1f}/10.\n"
            f"Judge reason: {judge_reason or '<none>'}\n"
            f"Rejection: {rejection_reason or '<none>'}\n"
            "Fix the actual weakness. Do not merely rephrase the old hooks."
        )

        if score_feedback:
            repair_feedback += (
                "\nCandidate feedback:\n- "
                + "\n- ".join(
                    score_feedback
                )
            )

        print()
        print(
            "🛠️ Intro 8/10 repair pass: "
            "Luna, Terra judge feedback'iyle yeni aday üretiyor..."
        )

        repaired_raw = request_intro_candidates(
            teaser=teaser,
            clip=clip,
            clip_transcript=clip_transcript,
            creator_name=creator_name,
            gemini_context=gemini_context,
            retry_feedback=repair_feedback,
        )

        repaired_valid: list[
            dict[str, Any]
        ] = []

        repaired_seen = set(
            seen_texts
        )

        (
            _,
            _,
        ) = _collect_valid_candidates(
            raw_candidates=repaired_raw,
            teaser_text=str(
                teaser.get(
                    "teaser_text",
                    "",
                )
            ),
            creator_name=creator_name,
            existing=repaired_valid,
            seen_texts=repaired_seen,
            evidence_text=evidence_text,
            verified_names=verified_names,
        )

        if repaired_valid:

            # Fresh repaired ideas get priority. Fill any remaining judge slots
            # with the original valid set so Terra still has comparison context.
            combined: list[
                dict[str, Any]
            ] = []

            combined_seen: set[
                str
            ] = set()

            for candidate in (
                repaired_valid
                + candidates
            ):
                key = normalize_for_compare(
                    str(
                        candidate.get(
                            "intro_text",
                            "",
                        )
                    )
                )

                if (
                    not key
                    or key in combined_seen
                ):
                    continue

                combined_seen.add(
                    key
                )

                combined.append(
                    candidate
                )

                if len(
                    combined
                ) >= CANDIDATE_COUNT:
                    break

            candidates = combined

            repair_rounds_used = 1

            print()
            print(
                f"⚖️ Terra repair judge: "
                f"{len(candidates)} aday tekrar karşılaştırılıyor..."
            )

            judgement = request_final_judgement(
                teaser=teaser,
                clip=clip,
                clip_transcript=clip_transcript,
                creator_name=creator_name,
                gemini_context=gemini_context,
                candidates=candidates,
            )

            (
                accepted,
                selected_index,
                score,
            ) = _judgement_state(
                judgement,
                candidates,
            )

    candidate_scores = judgement.get(
        "candidate_scores",
        [],
    )

    if not isinstance(
        candidate_scores,
        list,
    ):
        candidate_scores = []

    judge_support_used = bool(
        judgement.get(
            "gemini_support_used",
            False,
        )
        and gemini_support_available
    )

    if not accepted:

        rejection_reason = normalize_spaces(
            str(
                judgement.get(
                    "rejection_reason",
                    "",
                )
            )
        )

        reason = (
            rejection_reason
            or normalize_spaces(
                str(
                    judgement.get(
                        "reason",
                        "",
                    )
                )
            )
            or (
                "Best intro did not clear the "
                f"{MIN_RECOMMENDED_SCORE:.1f}/10 quality gate."
            )
        )

        print()
        print(
            "🏁 Terra final decision:"
        )
        print(
            f"⏭️ Headline REJECTED {score:.1f}/10 → cold open runs without a headline."
        )

        return _build_no_intro_result(
            clip=clip,
            clip_index=clip_index,
            creator_name=creator_name,
            score=score,
            reason=reason,
            gemini_support_used=(
                judge_support_used
            ),
            candidates=candidates,
            candidate_scores=(
                candidate_scores
            ),
            repair_rounds_used=(
                repair_rounds_used
            ),
        )

    selected_candidate = candidates[
        selected_index - 1
    ]

    final_text = normalize_spaces(
        str(
            selected_candidate.get(
                "intro_text",
                "",
            )
        )
    ).upper()

    duration = calculate_intro_duration(
        final_text
    )

    freeze_frame_time = calculate_freeze_frame_time(
        teaser
    )

    reason = normalize_spaces(
        str(
            judgement.get(
                "reason",
                "",
            )
        )
    )

    tone = str(
        selected_candidate.get(
            "tone",
            "curiosity",
        )
    )

    copy_strategy = str(
        selected_candidate.get(
            "copy_strategy",
            "curiosity_gap",
        )
    )

    curiosity_target = normalize_spaces(
        str(
            selected_candidate.get(
                "curiosity_target",
                "",
            )
        )
    )

    used_in_copy = bool(
        selected_candidate.get(
            "uses_specific_name",
            False,
        )
    )

    result = {
        "clip_index": clip_index,
        "title": str(
            clip.get(
                "title",
                "",
            )
        ),
        "recommended": True,
        "ai_recommended": True,
        "headline_status": "approved",
        "score": round(
            score,
            1,
        ),
        "intro_text": final_text,
        "tone": tone,
        "copy_strategy": copy_strategy,
        "creator": {
            "verified_name": creator_name,
            "used_in_copy": used_in_copy,
        },
        "reason": reason,
        "curiosity_target": curiosity_target,
        "quality_gate": {
            "threshold": MIN_RECOMMENDED_SCORE,
            "accepted": True,
            "no_headline": False,
            "candidate_count": len(
                candidates
            ),
            "selected_candidate_index": (
                selected_index
            ),
            "gemini_support_used": (
                judge_support_used
            ),
            "candidate_scores": (
                candidate_scores
            ),
            "repair_rounds_used": (
                repair_rounds_used
            ),
            "rejection_reason": "",
        },
        "candidates": [
            {
                "candidate_index": index,
                **candidate,
            }
            for index, candidate in enumerate(
                candidates,
                start=1,
            )
        ],
        "intro": {
            "duration": duration,
            "word_count": len(
                final_text.split()
            ),
            "character_count": len(
                final_text
            ),
        },
        "background": {
            "source": "edited_clip",
            "freeze_frame_time": (
                freeze_frame_time
            ),
            "style": "moving_teaser",
        },
        "sequence": [
            "moving_teaser_with_hook",
            "hard_main_clip_restart",
        ],
        "render_plan": {
            "intro_background": "moving_teaser",
            "intro_duration": duration,
            "freeze_frame_time": freeze_frame_time,
            "then_play": "main_clip_restart",
            "restart_main_clip_after_teaser": True,
            "transition": "hard_cut",
        },
    }

    print()
    print(
        "🏁 Terra final decision:"
    )
    print(
        f"✅ ACCEPTED {score:.1f}/10 → {final_text}"
    )
    print(
        f"🎯 Hard threshold: "
        f"{MIN_RECOMMENDED_SCORE:.1f}/10"
    )

    if repair_rounds_used:
        print(
            "🛠️ 8/10 repair pass kullanıldı."
        )

    return result


# ============================================================
# OUTPUT PATH
# ============================================================

def get_output_path(
    teaser_json_path: str | Path,
) -> Path:

    teaser_json_path = Path(
        teaser_json_path
    )

    base = (
        teaser_json_path.stem
    )

    suffix = (
        "_teasers"
    )

    if base.endswith(
        suffix
    ):

        base = base[
            :-len(
                suffix
            )
        ]

    INTRO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        INTRO_DIR
        / f"{base}_intros.json"
    ).resolve()


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    teaser_json_path: str | Path,
    timeline_path: str | Path,
    transcript_path: str | Path,
    results: list[dict[str, Any]],
    video_report_path: str | Path | None = None,
) -> Path:

    output_path = (
        get_output_path(
            teaser_json_path
        )
    )

    package = {
        "version": (
            INTRO_ANALYZER_VERSION
        ),

        "mode": (
            "moving_teaser_hook_copy"
        ),

        "model": INTRO_JUDGE_MODEL,
        "draft_model": INTRO_DRAFT_MODEL,
        "revision": INTRO_ANALYZER_REVISION,
        "quality_threshold": MIN_RECOMMENDED_SCORE,

        "inputs": {
            "teaser": str(
                Path(
                    teaser_json_path
                ).resolve()
            ),

            "timeline": str(
                Path(
                    timeline_path
                ).resolve()
            ),

            "transcript": str(
                Path(
                    transcript_path
                ).resolve()
            ),

            "video_brain_report": (
                str(
                    Path(
                        video_report_path
                    ).resolve()
                )
                if video_report_path
                else None
            ),
        },

        "clip_count": len(
            results
        ),

        "intros": (
            results
        ),
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            package,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


# ============================================================
# MAIN ANALYZER
# ============================================================

def analyze_intros(
    teaser_json_path: str | Path,
    clip_index: int | None = None,
    manual_creator_name: str | None = None,
    video_report_path: str | Path | None = None,
    verified_names: list[str] | tuple[str, ...] = (),
    caption_text: str = "",
) -> dict[str, Any]:

    teaser_json_path = Path(
        teaser_json_path
    ).resolve()

    teaser_package = load_json(
        teaser_json_path
    )

    validate_teaser_package(
        teaser_package
    )

    (
        timeline_path,
        transcript_path,
    ) = get_referenced_paths(
        teaser_package
    )

    timeline = load_json(
        timeline_path
    )

    transcript = load_json(
        transcript_path
    )

    validate_timeline(
        timeline
    )

    validate_transcript(
        transcript
    )

    video_support_report = load_video_support(
        video_report_path
    )

    results: list[
        dict[str, Any]
    ] = []

    # --------------------------------------------------------
    # ONE CLIP
    # --------------------------------------------------------

    if (
        clip_index
        is not None
    ):

        results.append(
            analyze_intro_for_clip(
                package=teaser_package,
                timeline=timeline,
                transcript=transcript,
                clip_index=clip_index,
                manual_creator_name=manual_creator_name,
                video_support_report=video_support_report,
                verified_names=verified_names,
                caption_text=caption_text,
            )
        )

    # --------------------------------------------------------
    # ALL CLIPS
    # --------------------------------------------------------

    else:

        for position, teaser in enumerate(
            teaser_package[
                "teasers"
            ],
            start=1,
        ):

            if not isinstance(
                teaser,
                dict,
            ):

                continue

            try:

                current_index = int(
                    teaser.get(
                        "clip_index",
                        position,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                current_index = (
                    position
                )

            results.append(
                analyze_intro_for_clip(
                    package=teaser_package,
                    timeline=timeline,
                    transcript=transcript,
                    clip_index=current_index,
                    manual_creator_name=manual_creator_name,
                )
            )

    output_path = save_results(
        teaser_json_path=teaser_json_path,
        timeline_path=timeline_path,
        transcript_path=transcript_path,
        results=results,
        video_report_path=video_report_path,
    )

    print()
    print(
        "=" * 72
    )

    print(
        "✅ INTRO ANALYZER — MOVING TEASER HOOK TAMAMLANDI"
    )

    print(
        f"📂 {output_path}"
    )

    print(
        "=" * 72
    )

    return {
        "output_path": str(
            output_path
        ),

        "intros": (
            results
        ),
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "MIMIR Intro Analyzer — Terra Final Judge"
    )

    print(
        "Terra + Gemini support, hard 8/10 intro quality gate"
    )

    print()

    teaser_json_path = input(
        "Teaser JSON yolunu gir: "
    ).strip().strip('"')

    clip_input = input(
        "Clip index "
        "(boş = tüm klipler): "
    ).strip()

    creator_input = input(
        "Doğrulanmış creator/streamer adı "
        "(yoksa boş bırak): "
    ).strip()

    try:

        if clip_input:

            selected_clip = int(
                clip_input
            )

        else:

            selected_clip = None

        manual_creator_name = (
            creator_input
            if creator_input
            else None
        )

        analyze_intros(
            teaser_json_path=teaser_json_path,
            clip_index=selected_clip,
            manual_creator_name=manual_creator_name,
        )

    except Exception as error:

        print()
        print(
            "❌ INTRO ANALYZER HATASI:"
        )

        print(
            error
        )
