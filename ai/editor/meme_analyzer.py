from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from ai.openai_client import client
from ai.editor import meme_audio_support
from ai.model_config import (
    MEME_MODEL,
    MEME_REASONING_EFFORT,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VOD_OUTPUT_DIR = PROJECT_ROOT / "vod_output"
MEME_OUTPUT_DIR = VOD_OUTPUT_DIR / "memes"


# ============================================================
# VERSION / POLICY
# ============================================================

# Discovery V1 ile uyumluluk için output version=2 kalıyor.
MEME_ANALYZER_VERSION = 3
MODE = "single_best_unexpected_local_meme_slot"

# Kullanıcı politikası:
# Klip başına EN FAZLA 1 meme.
MAX_OUTPUT_SLOTS = 1

# Kalite kapıları — audio mikro-accent görsel meme kadar intrusif değildir.
# Bu nedenle audio için daha doğal/eşiği düşük, visual için daha seçici bir gate var.
AUDIO_MIN_FIT_SCORE = 7.35
AUDIO_MIN_CONFIDENCE = 0.68
VISUAL_MIN_FIT_SCORE = 7.90
VISUAL_MIN_CONFIDENCE = 0.74
EITHER_MIN_FIT_SCORE = 7.60
EITHER_MIN_CONFIDENCE = 0.70
MEDIUM_RISK_SCORE_BONUS = 0.40
MEDIUM_RISK_CONFIDENCE_BONUS = 0.04

# Restart sonrası ilk bölüm temiz kalsın.
MAIN_OPENING_PROTECTION = 0.70

# Payoff mümkün olduğunca temiz.
PAYOFF_PROTECTION_BEFORE = 0.30
PAYOFF_PROTECTION_AFTER = 0.40
PAYOFF_AUDIO_MIN_FIT = 8.40
PAYOFF_AUDIO_MIN_CONFIDENCE = 0.80
PAYOFF_VISUAL_MIN_FIT = 9.00
PAYOFF_VISUAL_MIN_CONFIDENCE = 0.86

# Meme süresi üst sınırları.
AUDIO_MAX_DURATION = 1.55
VISUAL_MAX_DURATION = 0.95
EITHER_MAX_DURATION = 1.10

# AI ilk seçimleri kalite kapısından dönerse 1 kez daha düşünür.
MAX_AI_ATTEMPTS = 1

# V22 unexpectedness-first editorial gate. A meme is allowed only when the beat
# genuinely breaks expectation; "funny" or "loud" alone is not enough.
UNEXPECTEDNESS_MIN = 0.80
EDITORIAL_MEME_SCORE_MIN = 0.75
SECONDARY_WEIRDNESS_MIN = 0.55

MEME_CATEGORIES = {
    "reversal",
    "disbelief",
    "confusion",
    "absurdity",
    "fail",
    "awkward",
    "hype",
    "fear_shock",
    "wholesome_ironic",
    "impact",
}


# ============================================================
# AI STRUCTURED OUTPUT
# ============================================================

MEME_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "use_meme": {
            "type": "boolean"
        },
        "decision_reason": {
            "type": "string"
        },
        "candidates": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "anchor_word_id": {
                        "type": "integer"
                    },
                    "placement": {
                        "type": "string",
                        "enum": [
                            "on_word",
                            "after_word"
                        ]
                    },
                    "intent": {
                        "type": "string",
                        "enum": [
                            "confusion_reaction",
                            "disbelief_reaction",
                            "absurdity_reaction",
                            "awkward_reaction",
                            "fail_reaction",
                            "hype_accent",
                            "tension_release",
                            "punchline_support"
                        ]
                    },
                    "meme_category": {
                        "type": "string",
                        "enum": [
                            "reversal",
                            "disbelief",
                            "confusion",
                            "absurdity",
                            "fail",
                            "awkward",
                            "hype",
                            "fear_shock",
                            "wholesome_ironic",
                            "impact"
                        ]
                    },
                    "unexpectedness": {
                        "type": "number"
                    },
                    "absurdity": {
                        "type": "number"
                    },
                    "reversal": {
                        "type": "number"
                    },
                    "preferred_media": {
                        "type": "string",
                        "enum": [
                            "audio",
                            "visual",
                            "either"
                        ]
                    },
                    "sound_function": {
                        "type": "string",
                        "enum": [
                            "none",
                            "confused_voice",
                            "disbelief_sting",
                            "record_scratch",
                            "impact_boom",
                            "error_buzzer",
                            "awkward_cricket",
                            "fail_sting",
                            "crowd_gasp",
                            "hype_sting",
                            "dramatic_hit",
                            "cartoon_pop",
                            "comedic_pause",
                            "other"
                        ]
                    },
                    "strength": {
                        "type": "string",
                        "enum": [
                            "subtle",
                            "medium",
                            "strong"
                        ]
                    },
                    "max_duration": {
                        "type": "number"
                    },
                    "fit_score": {
                        "type": "number"
                    },
                    "confidence": {
                        "type": "number"
                    },
                    "intrusion_risk": {
                        "type": "string",
                        "enum": [
                            "low",
                            "medium",
                            "high"
                        ]
                    },
                    "reason": {
                        "type": "string"
                    },
                },
                "required": [
                    "anchor_word_id",
                    "placement",
                    "intent",
                    "meme_category",
                    "unexpectedness",
                    "absurdity",
                    "reversal",
                    "preferred_media",
                    "sound_function",
                    "strength",
                    "max_duration",
                    "fit_score",
                    "confidence",
                    "intrusion_risk",
                    "reason"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": [
        "use_meme",
        "decision_reason",
        "candidates"
    ],
    "additionalProperties": False
}


# ============================================================
# AI INSTRUCTIONS
# ============================================================

MEME_INSTRUCTIONS = """
You are a senior short-form editor for English gaming / livestream clips.

Your job is NOT to choose a specific meme asset.
Your job is to decide whether the clip deserves ONE meme/SFX accent and, if so,
identify the single strongest beat and its semantic meme category. A later deterministic
module matches that category against the user's curated LOCAL reaction/SFX library.

============================================================
PRIMARY LAW
============================================================

NO MEME IS BETTER THAN A BAD MEME.
A meme exists to punctuate an EXPECTATION BREAK, not merely a funny or loud moment.
The strongest valid targets are: an outcome suddenly reversing, a bizarre/unexpected
line, an absurd physical result, an awkward dead stop, a shocking reveal, a confident
claim immediately failing, or a reaction that makes the viewer think “what just happened?”.

HARD RULE: unexpectedness must be at least 0.70. If the beat is predictable, routine,
only loud, only profane, or already perfectly self-contained, return use_meme=false.

But do NOT confuse “subtle audio punctuation” with a heavy visual meme.
A local SFX should improve rhythm without taking over the clip. Local assets are rendered to their FULL natural duration; they are never intentionally chopped short.

The final system allows AT MOST ONE meme/SFX accent in the entire clip.

============================================================
AUDIO-FIRST OPPORTUNITIES
============================================================

Actively consider AUDIO when there is:
- a surprising or bizarre sentence
- an unexpected word/claim that deserves a tiny “what?” punctuation
- a deadpan line followed by a pause
- an abrupt tonal shift
- a fail/mistake that needs a very short sting
- a disbelief/confusion beat
- a sudden hype beat that benefits from one impact accent
- tension that releases into a punchline

Audio does NOT need a visual overlay to be useful.
A tiny SFX/reaction sound is often the cleaner choice.

You receive REAL AUDIO-ENVELOPE SUPPORT. Use it.
High insertion_headroom_score + low speech_overlap_risk means an SFX can land
cleanly after that word. A sudden level drop or a real pause is especially useful.
Do not invent pauses or audio peaks that are not listed.

If the joke is already strong, a micro sound may still be valid when it functions
as punctuation rather than replacing the source reaction.

============================================================
BAD REASONS
============================================================

Do not add a meme merely because:
- profanity exists
- somebody is loud
- there is empty space
- the clip is already funny
- “memes improve retention”

Do not cover important dialogue with a loud sound.
If speech_overlap_risk is high, prefer a tiny/subtle accent or reject the idea.

============================================================
ONE-MEME RULE
============================================================

You may propose up to 3 alternatives, but software keeps at most ONE.
Find the strongest beat, not several average beats.

============================================================
TIMING
============================================================

You receive exact transcript WORD IDs from the restarted MAIN CLIP.
Never invent timestamps. Return anchor_word_id and placement only.

placement="on_word": tiny impact/sting begins with the selected word.
placement="after_word": reaction/SFX begins just after the selected word.

For surprising/odd sentences and disbelief, after_word is usually best.
For impact/hype punctuation, on_word may be best.
The deterministic audio-support layer will apply a small safe delay. Because local SFX play to completion, favor an anchor with clean headroom AFTER the beat; do not choose a cramped point just because the semantic joke is good.

============================================================
TEASER / MAIN RESTART
============================================================

Do not meme the teaser. Candidate applies only to restarted MAIN.
Avoid cluttering the first moment immediately after restart.

============================================================
PAYOFF
============================================================

Payoff is protected. Prefer leaving it clean.
A very short audio sting may be acceptable only when unusually precise.
Visual overlays need a stronger justification in payoff.

============================================================
AUDIO VS VISUAL
============================================================

Prefer AUDIO when:
- a sound alone can punctuate the beat
- the source image already tells the story
- a visual would distract
- important gameplay/face information may be on screen
- there is a clean post-line pause/headroom window

Prefer VISUAL only when a small reaction image/clip clearly adds a second layer
and audio would be less natural. Visual must remain brief, small, centered and
semi-transparent. Never full-screen.

Use "either" only when both genuinely fit.

============================================================
MEME CATEGORY + SURPRISE SCORES
============================================================

Choose exactly one semantic category for each candidate:
- reversal: confidence/expectation immediately flips
- disbelief: “no way / what?” reaction
- confusion: genuinely hard-to-process oddity
- absurdity: bizarre/cursed/illogical event or line
- fail: clear mistake or failure with comic consequence
- awkward: dead stop / social awkwardness / empty reaction
- hype: unexpectedly huge positive escalation
- fear_shock: sudden scare/shock/jump
- wholesome_ironic: unexpectedly cute/soft beat used ironically
- impact: very short hard punctuation, not a long reaction

Score every candidate from 0.0 to 1.0:
- unexpectedness: how strongly reality breaks the viewer's immediate expectation
- absurdity: how weird/cursed/strange the beat feels
- reversal: how strongly the before/after meaning flips

Do not inflate these values to force a meme. Software applies a hard unexpectedness gate.

============================================================
SOUND FUNCTION
============================================================

For audio/either choose a generic sound_function that guides web discovery:
- confused_voice: tiny “huh/what?”-style reaction
- disbelief_sting: disbelief punctuation
- record_scratch: abrupt “wait, what?” tonal break
- impact_boom: hard emphasis on one beat
- error_buzzer: obvious wrong/fail cue
- awkward_cricket: awkward/dead-silence punctuation
- fail_sting: short failure cue
- crowd_gasp: surprise/shock reaction
- hype_sting: positive/hype accent
- dramatic_hit: dramatic reveal/tension hit
- cartoon_pop: light comic punctuation
- comedic_pause: a tiny sound that works with a pause
- other: only if none fit

For a visual-only idea use sound_function="none".
Do not name or fabricate a specific asset.

============================================================
STRENGTH
============================================================

subtle = default, polished, barely intrusive
medium = noticeable but secondary
strong = rare; one exceptional beat only

============================================================
SCORING
============================================================

Audio micro-accents are intentionally judged on a friendlier scale:
7.3-7.7 = usable/good when low-risk and rhythmically clean
7.8-8.5 = strong
8.6+    = exceptional

Visual memes should generally be closer to 8/10 or better because they are
more intrusive.

confidence = chance a strong human editor would KEEP the idea.
intrusion_risk: low / medium / high. Do not intentionally propose high risk.

Deterministic ranking after your answer uses roughly:
35% unexpectedness + 25% absurdity + 20% reversal +
10% factual insertion headroom + 10% semantic fit.
Therefore choose the true expectation-break beat, not merely the cleanest pause.

============================================================
AVAILABLE EVIDENCE
============================================================

You have transcript/edit metadata plus a factual real-audio envelope.
You do NOT reliably see gameplay/facial reactions in this step. Never invent
visual events, faces, gestures or objects.

============================================================
FINAL TEST
============================================================

Before use_meme=true ask:
1. Did something genuinely unexpected / bizarre / reversed happen here?
2. Is there one specific beat worth punctuating?
3. Would AUDIO alone improve rhythm/comedy without hiding dialogue?
4. Is the semantic category clear enough to match a curated local asset?
5. Can the local SFX start naturally here and play to completion without feeling pasted on?
6. Would a human editor keep it?

If the answer is weak, use_meme=false.
""".strip()


# ============================================================
# GENERIC HELPERS
# ============================================================

def load_json(
    path: str | Path,
) -> dict[str, Any]:

    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"JSON bulunamadı:\n{path}"
        )

    raw = path.read_text(
        encoding="utf-8"
    ).strip()

    if not raw:
        raise RuntimeError(
            f"JSON dosyası tamamen boş:\n{path}"
        )

    try:
        data = json.loads(raw)

    except json.JSONDecodeError as error:
        raise RuntimeError(
            "JSON formatı bozuk:\n"
            f"{path}\n\n"
            f"Satır: {error.lineno}\n"
            f"Sütun: {error.colno}\n"
            f"Hata: {error.msg}"
        ) from error

    if not isinstance(data, dict):
        raise RuntimeError(
            f"JSON root object değil:\n{path}"
        )

    return data


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


def round_time(
    value: float,
) -> float:

    return round(
        float(value),
        3,
    )


def clean_query(
    value: Any,
) -> str:

    return " ".join(
        str(value)
        .strip()
        .split()
    )[:160]


# ============================================================
# PACKAGE VALIDATION
# ============================================================

def validate_intro_package(
    package: dict[str, Any],
) -> None:

    if package.get("version") != 1:
        raise RuntimeError(
            "Meme Analyzer, Intro Analyzer V1 çıktısı bekliyor."
        )

    intros = package.get("intros")

    if (
        not isinstance(intros, list)
        or not intros
    ):
        raise RuntimeError(
            "Intro JSON içinde intros listesi bulunamadı."
        )

    inputs = package.get("inputs")

    if not isinstance(inputs, dict):
        raise RuntimeError(
            "Intro JSON inputs eksik."
        )

    for key in (
        "teaser",
        "timeline",
        "transcript",
    ):

        value = inputs.get(key)

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise RuntimeError(
                f"Intro JSON inputs.{key} eksik."
            )


def validate_teaser_package(
    package: dict[str, Any],
) -> None:

    if package.get("version") != 1:
        raise RuntimeError(
            "Meme Analyzer, Teaser Analyzer V1 çıktısı bekliyor."
        )

    if not isinstance(
        package.get("teasers"),
        list,
    ):
        raise RuntimeError(
            "Teaser JSON teasers listesi eksik."
        )


def validate_timeline(
    timeline: dict[str, Any],
) -> None:

    if timeline.get("version") != 3:
        raise RuntimeError(
            "Meme Analyzer yalnızca Timeline V3 ile çalışır."
        )

    timelines = timeline.get("timelines")

    if (
        not isinstance(timelines, list)
        or not timelines
    ):
        raise RuntimeError(
            "Timeline V3 timelines listesi eksik."
        )


def validate_transcript(
    transcript: dict[str, Any],
) -> None:

    words = transcript.get("words")

    if (
        not isinstance(words, list)
        or not words
    ):
        raise RuntimeError(
            "Transcript word timestamps içermiyor."
        )


# ============================================================
# REFERENCED INPUT PATHS
# ============================================================

def get_referenced_paths(
    intro_package: dict[str, Any],
) -> tuple[
    Path,
    Path,
    Path,
]:

    inputs = intro_package["inputs"]

    teaser_path = Path(
        inputs["teaser"]
    ).resolve()

    timeline_path = Path(
        inputs["timeline"]
    ).resolve()

    transcript_path = Path(
        inputs["transcript"]
    ).resolve()

    for label, path in (
        ("Teaser", teaser_path),
        ("Timeline", timeline_path),
        ("Transcript", transcript_path),
    ):

        if not path.exists():
            raise FileNotFoundError(
                f"{label} bulunamadı:\n{path}"
            )

    return (
        teaser_path,
        timeline_path,
        transcript_path,
    )


# ============================================================
# CLIP LOOKUP
# ============================================================

def _lookup_by_clip_index(
    items: list[Any],
    clip_index: int,
    label: str,
) -> dict[str, Any]:

    for position, item in enumerate(
        items,
        start=1,
    ):

        if not isinstance(item, dict):
            continue

        try:
            current_index = int(
                item.get(
                    "clip_index",
                    position,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            current_index = position

        if current_index == clip_index:
            return item

    raise IndexError(
        f"{label} içinde clip_index={clip_index} bulunamadı."
    )


def get_timeline_clip(
    timeline: dict[str, Any],
    clip_index: int,
) -> dict[str, Any]:

    return _lookup_by_clip_index(
        timeline["timelines"],
        clip_index,
        "Timeline",
    )


def get_teaser(
    teaser_package: dict[str, Any],
    clip_index: int,
) -> dict[str, Any]:

    return _lookup_by_clip_index(
        teaser_package["teasers"],
        clip_index,
        "Teaser JSON",
    )


def get_intro(
    intro_package: dict[str, Any],
    clip_index: int,
) -> dict[str, Any]:

    return _lookup_by_clip_index(
        intro_package["intros"],
        clip_index,
        "Intro JSON",
    )


# ============================================================
# CUT RANGES / TIMING MAP
# ============================================================

def normalize_cut_ranges(
    clip: dict[str, Any],
) -> list[dict[str, float]]:

    raw_ranges = clip.get(
        "cut_ranges",
        [],
    )

    if not isinstance(raw_ranges, list):
        return []

    ranges: list[
        dict[str, float]
    ] = []

    for raw in raw_ranges:

        if not isinstance(raw, dict):
            continue

        try:
            start = float(
                raw.get(
                    "start",
                    0.0,
                )
            )

            end = float(
                raw.get(
                    "end",
                    start,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if end <= start:
            continue

        ranges.append(
            {
                "start": start,
                "end": end,
            }
        )

    ranges.sort(
        key=lambda item: item["start"]
    )

    merged: list[
        dict[str, float]
    ] = []

    for current in ranges:

        if not merged:
            merged.append(
                current.copy()
            )
            continue

        previous = merged[-1]

        if current["start"] <= previous["end"]:

            previous["end"] = max(
                previous["end"],
                current["end"],
            )

        else:
            merged.append(
                current.copy()
            )

    return merged


def interval_overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:

    return max(
        0.0,
        min(
            end_a,
            end_b,
        )
        - max(
            start_a,
            start_b,
        ),
    )


def word_is_removed(
    relative_start: float,
    relative_end: float,
    cut_ranges: list[dict[str, float]],
) -> bool:

    duration = max(
        0.001,
        relative_end - relative_start,
    )

    removed = 0.0

    for cut in cut_ranges:

        removed += interval_overlap(
            relative_start,
            relative_end,
            float(cut["start"]),
            float(cut["end"]),
        )

    return (
        removed / duration
    ) >= 0.50


def source_to_edited_time(
    source_time: float,
    cut_ranges: list[dict[str, float]],
) -> float:

    source_time = float(source_time)

    removed = 0.0

    for cut in cut_ranges:

        start = float(cut["start"])
        end = float(cut["end"])

        if source_time <= start:
            break

        if source_time >= end:
            removed += (
                end - start
            )
            continue

        removed += (
            source_time - start
        )
        break

    return max(
        0.0,
        source_time - removed,
    )


# ============================================================
# VISIBLE WORDS
# ============================================================

def build_visible_words(
    transcript: dict[str, Any],
    clip: dict[str, Any],
) -> list[dict[str, Any]]:

    source = clip.get(
        "source",
        {},
    )

    try:
        clip_absolute_start = float(
            source["absolute_start"]
        )

        clip_absolute_end = float(
            source["absolute_end"]
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as error:

        raise RuntimeError(
            "Timeline clip source.absolute_start/end eksik."
        ) from error

    cut_ranges = normalize_cut_ranges(
        clip
    )

    result: list[
        dict[str, Any]
    ] = []

    for raw_word in transcript.get(
        "words",
        [],
    ):

        if not isinstance(raw_word, dict):
            continue

        try:
            word_id = int(
                raw_word["id"]
            )

            absolute_start = float(
                raw_word["start"]
            )

            absolute_end = float(
                raw_word["end"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        text = str(
            raw_word.get(
                "word",
                "",
            )
        ).strip()

        if not text:
            continue

        midpoint = (
            absolute_start
            + absolute_end
        ) / 2.0

        if not (
            clip_absolute_start
            <= midpoint
            <= clip_absolute_end
        ):
            continue

        relative_start = (
            absolute_start
            - clip_absolute_start
        )

        relative_end = (
            absolute_end
            - clip_absolute_start
        )

        if word_is_removed(
            relative_start,
            relative_end,
            cut_ranges,
        ):
            continue

        edited_start = source_to_edited_time(
            relative_start,
            cut_ranges,
        )

        edited_end = source_to_edited_time(
            relative_end,
            cut_ranges,
        )

        result.append(
            {
                "id": word_id,
                "word": text,

                "absolute_start": absolute_start,
                "absolute_end": absolute_end,

                "source_relative_start": relative_start,
                "source_relative_end": relative_end,

                "edited_start": edited_start,
                "edited_end": edited_end,
            }
        )

    if not result:
        raise RuntimeError(
            "Clip için kullanılabilir transcript kelimesi bulunamadı."
        )

    return result


def format_words_for_ai(
    words: list[dict[str, Any]],
) -> str:

    return "\n".join(
        (
            f"ID {word['id']} | "
            f"{word['edited_start']:.2f}"
            f"-"
            f"{word['edited_end']:.2f}s | "
            f"{word['word']}"
        )
        for word in words
    )


# ============================================================
# CLIP METADATA
# ============================================================

def get_edited_duration(
    clip: dict[str, Any],
    words: list[dict[str, Any]],
) -> float:

    edited = clip.get(
        "edited",
        {},
    )

    if isinstance(edited, dict):

        try:
            duration = float(
                edited.get(
                    "estimated_duration",
                    0.0,
                )
            )

            if duration > 0:
                return duration

        except (
            TypeError,
            ValueError,
        ):
            pass

    return max(
        float(word["edited_end"])
        for word in words
    )


def get_teaser_duration(
    teaser: dict[str, Any],
) -> float:

    if teaser.get("recommended") is not True:
        return 0.0

    edited = teaser.get(
        "edited",
        {},
    )

    if not isinstance(edited, dict):
        return 0.0

    try:
        return max(
            0.0,
            float(
                edited.get(
                    "duration",
                    0.0,
                )
            ),
        )

    except (
        TypeError,
        ValueError,
    ):
        return 0.0


def get_payoff_edited_range(
    clip: dict[str, Any],
) -> tuple[
    float,
    float,
] | None:

    payoff = clip.get("payoff")

    if not isinstance(payoff, dict):
        return None

    # 1) Direkt edited_start / edited_end
    try:
        start = float(
            payoff["edited_start"]
        )

        end = float(
            payoff["edited_end"]
        )

        if end > start:
            return (
                start,
                end,
            )

    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        pass

    # 2) Nested edited {start,end}
    nested = payoff.get("edited")

    if isinstance(nested, dict):

        try:
            start = float(
                nested["start"]
            )

            end = float(
                nested["end"]
            )

            if end > start:
                return (
                    start,
                    end,
                )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            pass

    # 3) Source-relative payoff -> edited map
    relative_pairs = (
        (
            "relative_start",
            "relative_end",
        ),
        (
            "source_relative_start",
            "source_relative_end",
        ),
    )

    cut_ranges = normalize_cut_ranges(
        clip
    )

    for start_key, end_key in relative_pairs:

        try:
            source_start = float(
                payoff[start_key]
            )

            source_end = float(
                payoff[end_key]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        if source_end <= source_start:
            continue

        return (
            source_to_edited_time(
                source_start,
                cut_ranges,
            ),
            source_to_edited_time(
                source_end,
                cut_ranges,
            ),
        )

    return None


def build_clip_context(
    clip: dict[str, Any],
    teaser: dict[str, Any],
    intro: dict[str, Any],
    words: list[dict[str, Any]],
    audio_support: dict[str, Any] | None = None,
) -> dict[str, Any]:

    return {
        "title": str(
            clip.get(
                "title",
                "",
            )
        ),

        "emotion": str(
            clip.get(
                "emotion",
                "",
            )
        ),

        "hook": clip.get(
            "hook",
            {}
        ),

        "caption_highlights": clip.get(
            "caption_highlights",
            []
        ),

        "editor_notes": clip.get(
            "editor_notes",
            []
        ),

        "teaser_text": str(
            teaser.get(
                "teaser_text",
                "",
            )
        ),

        "teaser_type": str(
            teaser.get(
                "teaser_type",
                "",
            )
        ),

        "intro_text": str(
            intro.get(
                "intro_text",
                "",
            )
        ),

        "payoff_edited_range": get_payoff_edited_range(
            clip
        ),

        "main_duration": get_edited_duration(
            clip,
            words,
        ),

        "audio_support": audio_support or {
            "available": False,
            "anchor_hints": [],
            "pause_windows": [],
            "audio_peaks": [],
        },
    }


# ============================================================
# LOCAL SFX CATALOG — FACTUAL, NO AI CALL
# ============================================================

def get_local_sfx_catalog() -> list[dict[str, Any]]:
    local_dir = PROJECT_ROOT / "meme_library" / "local_sfx"
    if not local_dir.is_dir():
        return []

    supported = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".webm"}
    catalog: list[dict[str, Any]] = []
    for path in sorted(local_dir.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.casefold() not in supported:
            continue
        duration = 0.0
        try:
            completed = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                check=False, timeout=6,
            )
            if completed.returncode == 0:
                duration = max(0.0, float((completed.stdout or "0").strip() or 0.0))
        except Exception:
            duration = 0.0
        catalog.append({"name": path.stem, "duration": round(duration, 3)})
    return catalog


def format_local_sfx_catalog() -> str:
    catalog = get_local_sfx_catalog()
    if not catalog:
        return "No usable local SFX were found."
    return "\n".join(
        f"- {item['name']} | full_duration={float(item['duration']):.3f}s"
        for item in catalog
    )


# ============================================================
# AI INPUT
# ============================================================

def build_ai_input(
    context: dict[str, Any],
    words: list[dict[str, Any]],
    retry_feedback: str | None = None,
) -> str:

    payoff = context[
        "payoff_edited_range"
    ]

    if payoff is None:
        payoff_text = (
            "UNKNOWN / NOT PROVIDED"
        )
    else:
        payoff_text = (
            f"{payoff[0]:.2f}"
            f" -> "
            f"{payoff[1]:.2f}s"
        )

    retry_text = ""

    if retry_feedback:

        retry_text = f"""

============================================================
PREVIOUS ATTEMPT FAILED QUALITY GATES
============================================================

{retry_feedback}

Reconsider the clip.
Return a different exceptional candidate or use_meme=false.
""".rstrip()

    return f"""
CLIP TITLE:
{context['title']}

CLIP EMOTION:
{context['emotion']}

MAIN CLIP DURATION:
{context['main_duration']:.2f}s

OPENING TEASER:
{context['teaser_text']}

TEASER TYPE:
{context['teaser_type']}

INTRO / HOOK COPY:
{context['intro_text']}

PAYOFF REGION:
{payoff_text}

CAPTION HIGHLIGHTS:
{json.dumps(context['caption_highlights'], ensure_ascii=False)}

EDITOR NOTES:
{json.dumps(context['editor_notes'], ensure_ascii=False)}

============================================================
AVAILABLE LOCAL SFX — PLAYED TO COMPLETION
============================================================

{format_local_sfx_catalog()}

Use these durations when choosing the beat. A long local reaction sound needs a
clean onset and enough remaining clip time; do not assume it will be truncated.

============================================================
REAL AUDIO SUPPORT FOR SFX TIMING
============================================================

{meme_audio_support.format_audio_support_for_ai(context.get('audio_support'))}

============================================================
VISIBLE RESTARTED-MAIN WORDS
============================================================

{format_words_for_ai(words)}

Analyze the entire main clip and decide whether ONE meme accent is genuinely
worth discovering.

You may return up to 3 candidate ideas internally, but they are alternatives.
The software will output at most ONE.

Never create a teaser meme.
Never invent timestamps.
Use only word IDs listed above.
{retry_text}
""".strip()


# ============================================================
# AI CALL
# ============================================================

def request_analysis(
    context: dict[str, Any],
    words: list[dict[str, Any]],
    retry_feedback: str | None = None,
) -> dict[str, Any]:

    print(
        f"🤖 Meme judge: {MEME_MODEL} "
        f"[{MEME_REASONING_EFFORT}]"
    )

    response = client.responses.create(
        model=MEME_MODEL,
        reasoning={
            "effort": MEME_REASONING_EFFORT,
        },

        instructions=MEME_INSTRUCTIONS,

        input=build_ai_input(
            context=context,
            words=words,
            retry_feedback=retry_feedback,
        ),

        text={
            "format": {
                "type": "json_schema",
                "name": "single_meme_slot_analysis_v2",
                "strict": True,
                "schema": MEME_ANALYSIS_SCHEMA,
            }
        },
    )

    output_text = str(
        response.output_text
    ).strip()

    if not output_text:
        raise RuntimeError(
            "AI boş meme analizi döndürdü."
        )

    try:
        data = json.loads(
            output_text
        )

    except json.JSONDecodeError as error:
        raise RuntimeError(
            "AI geçersiz meme JSON döndürdü:\n\n"
            + output_text
        ) from error

    if not isinstance(data, dict):
        raise RuntimeError(
            "AI meme çıktısı object değil."
        )

    return data


# ============================================================
# VALIDATION
# ============================================================

def candidate_max_duration(
    preferred_media: str,
    requested: float,
) -> float:

    if preferred_media == "audio":
        return clamp(
            requested,
            0.22,
            AUDIO_MAX_DURATION,
        )

    if preferred_media == "visual":
        return clamp(
            requested,
            0.35,
            VISUAL_MAX_DURATION,
        )

    return clamp(
        requested,
        0.25,
        EITHER_MAX_DURATION,
    )


def candidate_quality_gate(
    preferred_media: str,
    intrusion_risk: str,
) -> tuple[float, float]:
    if preferred_media == "audio":
        min_fit = AUDIO_MIN_FIT_SCORE
        min_confidence = AUDIO_MIN_CONFIDENCE
    elif preferred_media == "visual":
        min_fit = VISUAL_MIN_FIT_SCORE
        min_confidence = VISUAL_MIN_CONFIDENCE
    else:
        min_fit = EITHER_MIN_FIT_SCORE
        min_confidence = EITHER_MIN_CONFIDENCE

    if intrusion_risk == "medium":
        min_fit += MEDIUM_RISK_SCORE_BONUS
        min_confidence += MEDIUM_RISK_CONFIDENCE_BONUS

    return min_fit, min_confidence


def payoff_quality_gate(preferred_media: str) -> tuple[float, float]:
    if preferred_media == "audio":
        return PAYOFF_AUDIO_MIN_FIT, PAYOFF_AUDIO_MIN_CONFIDENCE
    return PAYOFF_VISUAL_MIN_FIT, PAYOFF_VISUAL_MIN_CONFIDENCE


def overlaps_payoff(
    start: float,
    end: float,
    payoff: tuple[
        float,
        float,
    ] | None,
) -> bool:

    if payoff is None:
        return False

    protected_start = max(
        0.0,
        payoff[0]
        - PAYOFF_PROTECTION_BEFORE,
    )

    protected_end = (
        payoff[1]
        + PAYOFF_PROTECTION_AFTER
    )

    return (
        interval_overlap(
            start,
            end,
            protected_start,
            protected_end,
        )
        > 0
    )


def validate_candidates(
    ai_result: dict[str, Any],
    words: list[dict[str, Any]],
    main_duration: float,
    teaser_duration: float,
    payoff: tuple[
        float,
        float,
    ] | None,
    audio_support: dict[str, Any] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[str],
]:

    raw_candidates = ai_result.get(
        "candidates",
        [],
    )

    if not isinstance(raw_candidates, list):
        raw_candidates = []

    word_lookup = {
        int(word["id"]): word
        for word in words
    }

    accepted: list[
        dict[str, Any]
    ] = []

    rejected: list[
        str
    ] = []

    for position, raw in enumerate(
        raw_candidates,
        start=1,
    ):

        if not isinstance(raw, dict):
            rejected.append(
                f"candidate {position}: object değil"
            )
            continue

        try:
            word_id = int(
                raw["anchor_word_id"]
            )

            fit_score = float(
                raw["fit_score"]
            )

            confidence = float(
                raw["confidence"]
            )

            requested_duration = float(
                raw["max_duration"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            rejected.append(
                f"candidate {position}: temel alanlar geçersiz"
            )
            continue

        word = word_lookup.get(
            word_id
        )

        if word is None:
            rejected.append(
                f"candidate {position}: word id {word_id} görünür değil"
            )
            continue

        placement = str(
            raw.get(
                "placement",
                "",
            )
        ).strip()

        if placement not in {
            "on_word",
            "after_word",
        }:
            rejected.append(
                f"candidate {position}: placement geçersiz"
            )
            continue

        preferred_media = str(
            raw.get(
                "preferred_media",
                "",
            )
        ).strip().lower()

        if preferred_media not in {
            "audio",
            "visual",
            "either",
        }:
            rejected.append(
                f"candidate {position}: preferred_media geçersiz"
            )
            continue

        intrusion_risk = str(
            raw.get(
                "intrusion_risk",
                "high",
            )
        ).strip().lower()

        fit_score = clamp(
            fit_score,
            0.0,
            10.0,
        )

        confidence = clamp(
            confidence,
            0.0,
            1.0,
        )

        meme_category = str(raw.get("meme_category", "")).strip().lower()
        if meme_category not in MEME_CATEGORIES:
            rejected.append(f"candidate {position}: meme_category geçersiz")
            continue

        try:
            unexpectedness = clamp(float(raw.get("unexpectedness", 0.0)), 0.0, 1.0)
            absurdity = clamp(float(raw.get("absurdity", 0.0)), 0.0, 1.0)
            reversal = clamp(float(raw.get("reversal", 0.0)), 0.0, 1.0)
        except (TypeError, ValueError):
            rejected.append(f"candidate {position}: surprise skorları geçersiz")
            continue

        if unexpectedness < UNEXPECTEDNESS_MIN:
            rejected.append(
                f"candidate {position}: unexpectedness {unexpectedness:.2f} < {UNEXPECTEDNESS_MIN:.2f}"
            )
            continue

        # High precision over recall: ordinary funny/loud moments are not meme beats.
        # Except for a truly sharp impact/fear event, the candidate must also carry
        # a second independent weirdness signal (absurdity or expectation reversal).
        if (
            max(absurdity, reversal) < SECONDARY_WEIRDNESS_MIN
            and not (meme_category in {"impact", "fear_shock"} and unexpectedness >= 0.90)
        ):
            rejected.append(
                f"candidate {position}: weak secondary weirdness "
                f"(absurdity={absurdity:.2f}, reversal={reversal:.2f})"
            )
            continue

        if intrusion_risk == "high":
            rejected.append(
                f"candidate {position}: intrusion_risk=high"
            )
            continue

        min_fit, min_confidence = candidate_quality_gate(
            preferred_media,
            intrusion_risk,
        )

        if fit_score < min_fit:
            rejected.append(
                (
                    f"candidate {position}: "
                    f"fit {fit_score:.2f} < {min_fit:.2f} "
                    f"({preferred_media}/{intrusion_risk})"
                )
            )
            continue

        if confidence < min_confidence:
            rejected.append(
                (
                    f"candidate {position}: "
                    f"confidence {confidence:.2f} < {min_confidence:.2f} "
                    f"({preferred_media}/{intrusion_risk})"
                )
            )
            continue

        duration = candidate_max_duration(
            preferred_media,
            requested_duration,
        )

        audio_hint = meme_audio_support.get_anchor_hint(
            audio_support,
            word_id,
        )

        if placement == "on_word":
            main_start = float(
                word["edited_start"]
            )
        else:
            safe_delay = 0.045
            if preferred_media in {"audio", "either"} and audio_hint:
                safe_delay = clamp(
                    float(audio_hint.get("recommended_delay_ms", 45)) / 1000.0,
                    0.015,
                    0.140,
                )
            main_start = float(
                word["edited_end"]
            ) + safe_delay

        if preferred_media in {"audio", "either"} and audio_hint:
            suggested_max = float(
                audio_hint.get("recommended_max_duration", duration)
            )
            if suggested_max > 0:
                duration = min(duration, max(0.22, suggested_max))

        if main_start < MAIN_OPENING_PROTECTION:
            rejected.append(
                (
                    f"candidate {position}: main restart'e "
                    f"fazla yakın ({main_start:.2f}s)"
                )
            )
            continue

        if main_start >= main_duration:
            rejected.append(
                f"candidate {position}: main clip dışında"
            )
            continue

        main_end = min(
            main_duration,
            main_start + duration,
        )

        duration = (
            main_end - main_start
        )

        if duration <= 0:
            rejected.append(
                f"candidate {position}: duration 0"
            )
            continue

        inside_payoff = overlaps_payoff(
            main_start,
            main_end,
            payoff,
        )

        payoff_min_fit, payoff_min_confidence = payoff_quality_gate(
            preferred_media
        )

        if inside_payoff and (
            fit_score < payoff_min_fit
            or confidence < payoff_min_confidence
        ):
            rejected.append(
                (
                    f"candidate {position}: payoff protection "
                    f"(fit={fit_score:.2f}, confidence={confidence:.2f}, "
                    f"needs={payoff_min_fit:.2f}/{payoff_min_confidence:.2f})"
                )
            )
            continue

        queries: list[str] = []

        final_start = (
            teaser_duration
            + main_start
        )

        final_end = (
            teaser_duration
            + main_end
        )

        placement_support_score = 0.50
        if preferred_media in {"audio", "either"} and audio_hint:
            support_score = clamp(float(audio_hint.get("support_score", 0.0)), 0.0, 1.0)
            headroom_score = clamp(float(audio_hint.get("insertion_headroom_score", 0.0)), 0.0, 1.0)
            punctuation_score = clamp(float(audio_hint.get("punctuation_score", 0.0)), 0.0, 1.0)
            placement_support_score = clamp(
                0.46 * support_score + 0.34 * headroom_score + 0.20 * punctuation_score,
                0.0,
                1.0,
            )
            if str(audio_hint.get("speech_overlap_risk", "high")) == "high" and placement_support_score < 0.28:
                rejected.append(
                    f"candidate {position}: audio insertion point too cramped "
                    f"(support={placement_support_score:.2f})"
                )
                continue

        semantic_fit = clamp(fit_score / 10.0, 0.0, 1.0)
        editorial_meme_score = clamp(
            0.35 * unexpectedness
            + 0.25 * absurdity
            + 0.20 * reversal
            + 0.10 * placement_support_score
            + 0.10 * semantic_fit,
            0.0,
            1.0,
        )
        if editorial_meme_score < EDITORIAL_MEME_SCORE_MIN:
            rejected.append(
                f"candidate {position}: editorial meme score {editorial_meme_score:.3f} "
                f"< {EDITORIAL_MEME_SCORE_MIN:.3f}"
            )
            continue

        accepted.append(
            {
                "slot_index": 1,

                "intent": str(
                    raw.get(
                        "intent",
                        "",
                    )
                ).strip(),

                "meme_category": meme_category,
                "unexpectedness": round(unexpectedness, 3),
                "absurdity": round(absurdity, 3),
                "reversal": round(reversal, 3),
                "editorial_meme_score": round(editorial_meme_score, 3),

                "preferred_media": (
                    preferred_media
                ),

                "sound_function": str(
                    raw.get(
                        "sound_function",
                        "none" if preferred_media == "visual" else "other",
                    )
                ).strip().lower(),

                "audio_mix": (
                    {
                        "support_available": True,
                        "insertion_headroom_score": float(audio_hint.get("insertion_headroom_score", 0.0)),
                        "punctuation_score": float(audio_hint.get("punctuation_score", 0.0)),
                        "speech_overlap_risk": str(audio_hint.get("speech_overlap_risk", "medium")),
                        "gap_after_seconds": float(audio_hint.get("gap_after_seconds", 0.0)),
                        "recommended_delay_ms": int(float(audio_hint.get("recommended_delay_ms", 45))),
                        "recommended_max_duration": float(audio_hint.get("recommended_max_duration", duration)),
                    }
                    if preferred_media in {"audio", "either"} and audio_hint
                    else {
                        "support_available": False,
                        "speech_overlap_risk": "unknown",
                    }
                ),

                "strength": str(
                    raw.get(
                        "strength",
                        "subtle",
                    )
                ).strip(),

                "fit_score": round(
                    fit_score,
                    2,
                ),

                "confidence": round(
                    confidence,
                    2,
                ),

                "intrusion_risk": (
                    intrusion_risk
                ),

                "placement_support_score": round(placement_support_score, 3),

                "reason": str(
                    raw.get(
                        "reason",
                        "",
                    )
                ).strip(),

                "anchor": {
                    "word_id": word_id,
                    "word": str(
                        word["word"]
                    ),
                    "placement": placement,
                },

                "timing": {
                    "main_edited_start": round_time(
                        main_start
                    ),

                    "main_edited_end": round_time(
                        main_end
                    ),

                    "final_start": round_time(
                        final_start
                    ),

                    "final_end": round_time(
                        final_end
                    ),

                    "max_duration": round_time(
                        duration
                    ),
                },

                "inside_payoff": (
                    inside_payoff
                ),

                "search_queries": (
                    queries[:4]
                ),

                "discovery_policy": {
                    "single_asset_only": True,
                    "local_only": True,
                    "category_first": True,
                    "no_asset_is_better_than_bad_asset": True,

                    "allow_audio": (
                        preferred_media
                        in {
                            "audio",
                            "either",
                        }
                    ),

                    "allow_visual": (
                        preferred_media
                        in {
                            "visual",
                            "either",
                        }
                    ),

                    "visual_style": {
                        "position": "center",
                        "small": True,
                        "semi_transparent": True,
                        "fullscreen_forbidden": True,
                    },
                },
            }
        )

    # En iyi tek fikir:
    # fit > confidence > düşük intrusion risk.
    risk_rank = {
        "low": 1,
        "medium": 0,
        "high": -1,
    }

    accepted.sort(
        key=lambda item: (
            float(item.get("editorial_meme_score", 0.0)),
            float(item.get("unexpectedness", 0.0)),
            float(item.get("placement_support_score", 0.50)),
            float(item["confidence"]),
            risk_rank.get(item["intrusion_risk"], -1),
        ),
        reverse=True,
    )

    return (
        accepted[:MAX_OUTPUT_SLOTS],
        rejected,
    )


# ============================================================
# ANALYZE ONE CLIP
# ============================================================

def analyze_clip(
    intro_package: dict[str, Any],
    teaser_package: dict[str, Any],
    timeline: dict[str, Any],
    transcript: dict[str, Any],
    clip_index: int,
    base_video_path: str | Path | None = None,
) -> dict[str, Any]:

    clip = get_timeline_clip(
        timeline,
        clip_index,
    )

    teaser = get_teaser(
        teaser_package,
        clip_index,
    )

    intro = get_intro(
        intro_package,
        clip_index,
    )

    words = build_visible_words(
        transcript,
        clip,
    )

    main_duration = float(
        get_edited_duration(
            clip,
            words,
        )
    )

    teaser_duration = get_teaser_duration(
        teaser
    )

    audio_support = meme_audio_support.analyze_meme_audio_support(
        base_video_path,
        words=words,
        teaser_duration=teaser_duration,
        main_duration=main_duration,
    )

    context = build_clip_context(
        clip,
        teaser,
        intro,
        words,
        audio_support=audio_support,
    )

    payoff = context[
        "payoff_edited_range"
    ]

    print()
    print(
        "=" * 74
    )

    print(
        f"🧠 SINGLE MEME ANALYZER — CLIP {clip_index}"
    )

    print(
        "=" * 74
    )

    print(
        f"🎬 {context['title']}"
    )

    print(
        f"⏱️ Main: {main_duration:.2f}s"
    )

    print(
        f"🔥 Protected teaser: {teaser_duration:.2f}s"
    )

    print(
        "🎯 Max output meme slots: 1"
    )

    retry_feedback: str | None = None

    final_slots: list[
        dict[str, Any]
    ] = []

    all_rejections: list[
        str
    ] = []

    last_ai_result: dict[
        str,
        Any
    ] | None = None

    for attempt in range(
        1,
        MAX_AI_ATTEMPTS + 1,
    ):

        print()
        print(
            f"🤖 AI değerlendirmesi "
            f"({attempt}/{MAX_AI_ATTEMPTS})..."
        )

        ai_result = request_analysis(
            context=context,
            words=words,
            retry_feedback=retry_feedback,
        )

        last_ai_result = ai_result

        # AI doğrudan "meme yok" diyorsa bitti.
        if ai_result.get(
            "use_meme"
        ) is not True:

            final_slots = []
            break

        slots, rejected = validate_candidates(
            ai_result=ai_result,
            words=words,
            main_duration=main_duration,
            teaser_duration=teaser_duration,
            payoff=payoff,
            audio_support=audio_support,
        )

        all_rejections.extend(
            rejected
        )

        if slots:

            final_slots = (
                slots[:1]
            )

            break

        retry_feedback = (
            "The previous meme ideas were rejected by deterministic "
            "quality gates:\n- "
            + "\n- ".join(
                rejected[:10]
            )
        )

    if last_ai_result is None:
        raise RuntimeError(
            "AI meme analizi üretilemedi."
        )

    recommended = bool(
        final_slots
    )

    print()
    print(
        f"✅ Final slots: {len(final_slots)}"
    )

    if final_slots:

        slot = final_slots[0]

        print(
            (
                f"🏆 {slot['intent']} "
                f"@ main "
                f"{slot['timing']['main_edited_start']:.2f}s "
                f"| {slot['preferred_media']} "
                f"| {slot.get('meme_category', 'unknown')} "
                f"| unexpected={slot.get('unexpectedness', 0.0):.2f} "
                f"| score={slot.get('editorial_meme_score', 0.0):.2f} "
                f"| confidence={slot['confidence']}"
            )
        )

    else:

        print(
            "👌 Meme yok. Kaynak klip temiz bırakıldı."
        )

    return {
        "clip_index": (
            clip_index
        ),

        "title": (
            context[
                "title"
            ]
        ),

        "recommended": (
            recommended
        ),

        "main_clip_duration": round_time(
            main_duration
        ),

        "opening_teaser_duration": round_time(
            teaser_duration
        ),

        "payoff_edited_range": (
            [round_time(payoff[0]), round_time(payoff[1])]
            if payoff is not None
            else None
        ),

        "max_slots_allowed": (
            MAX_OUTPUT_SLOTS
        ),

        "slots": (
            final_slots
        ),

        "audio_support": audio_support,

        "ai_summary": {
            "use_meme": bool(
                last_ai_result.get(
                    "use_meme",
                    False,
                )
            ),

            "decision_reason": str(
                last_ai_result.get(
                    "decision_reason",
                    "",
                )
            ).strip(),
        },

        "quality_gate": {
            "motto": (
                "No meme is better than a bad meme."
            ),

            "single_meme_only": True,
            "unexpectedness_min": UNEXPECTEDNESS_MIN,
            "editorial_meme_score_min": EDITORIAL_MEME_SCORE_MIN,

            "audio_min_fit_score": AUDIO_MIN_FIT_SCORE,
            "audio_min_confidence": AUDIO_MIN_CONFIDENCE,
            "visual_min_fit_score": VISUAL_MIN_FIT_SCORE,
            "visual_min_confidence": VISUAL_MIN_CONFIDENCE,

            "payoff_protection": True,

            "opening_teaser_protection": True,

            "rejections": (
                all_rejections
            ),
        },
    }


# ============================================================
# OUTPUT
# ============================================================

def get_output_path(
    intro_json_path: str | Path,
) -> Path:

    intro_json_path = Path(
        intro_json_path
    )

    base = (
        intro_json_path.stem
    )

    suffix = (
        "_intros"
    )

    if base.endswith(
        suffix
    ):
        base = base[
            :-len(suffix)
        ]

    MEME_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        MEME_OUTPUT_DIR
        / f"{base}_meme_slots.json"
    ).resolve()


def save_results(
    intro_json_path: str | Path,
    teaser_path: str | Path,
    timeline_path: str | Path,
    transcript_path: str | Path,
    clips: list[dict[str, Any]],
) -> Path:

    output_path = get_output_path(
        intro_json_path
    )

    package = {
        "version": (
            MEME_ANALYZER_VERSION
        ),

        "mode": (
            MODE
        ),

        "model": MEME_MODEL,
        "reasoning_effort": MEME_REASONING_EFFORT,

        "inputs": {
            "intro": str(
                Path(
                    intro_json_path
                ).resolve()
            ),

            "teaser": str(
                Path(
                    teaser_path
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
        },

        "clip_count": len(
            clips
        ),

        "clips": (
            clips
        ),
    }

    save_json = json.dumps(
        package,
        ensure_ascii=False,
        indent=2,
    )

    output_path.write_text(
        save_json,
        encoding="utf-8",
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def analyze_memes(
    intro_json_path: str | Path,
    clip_index: int | None = None,
    base_video_path: str | Path | None = None,
) -> dict[str, Any]:

    intro_json_path = Path(
        intro_json_path
    ).resolve()

    intro_package = load_json(
        intro_json_path
    )

    validate_intro_package(
        intro_package
    )

    (
        teaser_path,
        timeline_path,
        transcript_path,
    ) = get_referenced_paths(
        intro_package
    )

    teaser_package = load_json(
        teaser_path
    )

    timeline = load_json(
        timeline_path
    )

    transcript = load_json(
        transcript_path
    )

    validate_teaser_package(
        teaser_package
    )

    validate_timeline(
        timeline
    )

    validate_transcript(
        transcript
    )

    results: list[
        dict[str, Any]
    ] = []

    if clip_index is not None:

        results.append(
            analyze_clip(
                intro_package=intro_package,
                teaser_package=teaser_package,
                timeline=timeline,
                transcript=transcript,
                clip_index=clip_index,
                base_video_path=base_video_path,
            )
        )

    else:

        for position, intro in enumerate(
            intro_package["intros"],
            start=1,
        ):

            if not isinstance(
                intro,
                dict,
            ):
                continue

            try:
                current_index = int(
                    intro.get(
                        "clip_index",
                        position,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):
                current_index = position

            results.append(
                analyze_clip(
                    intro_package=intro_package,
                    teaser_package=teaser_package,
                    timeline=timeline,
                    transcript=transcript,
                    clip_index=current_index,
                    base_video_path=base_video_path,
                )
            )

    output_path = save_results(
        intro_json_path=intro_json_path,
        teaser_path=teaser_path,
        timeline_path=timeline_path,
        transcript_path=transcript_path,
        clips=results,
    )

    print()
    print(
        "=" * 74
    )

    print(
        "✅ SINGLE MEME ANALYZER TAMAMLANDI"
    )

    print(
        f"📂 {output_path}"
    )

    print(
        "=" * 74
    )

    return {
        "output_path": str(
            output_path
        ),

        "clips": (
            results
        ),
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "MIMIR Meme Analyzer V2 — Clean Single-Meme Edition"
    )

    print(
        "One clip. Zero or one meme. Only the best opportunity."
    )

    print(
        "No meme is better than a bad meme."
    )

    print()

    intro_json_path = input(
        "Intro JSON yolunu gir: "
    ).strip().strip('"')

    clip_input = input(
        "Clip index "
        "(boş = tüm klipler): "
    ).strip()

    try:

        selected_clip = (
            int(
                clip_input
            )
            if clip_input
            else None
        )

        analyze_memes(
            intro_json_path=(
                intro_json_path
            ),
            clip_index=(
                selected_clip
            ),
        )

    except Exception as error:

        print()
        print(
            "❌ MEME ANALYZER HATASI:"
        )

        print(
            error
        )
