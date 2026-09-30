from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai.openai_client import client
from ai.editor import intro_bounds, intro_peak_support
from ai.model_config import (
    TEASER_MODEL,
    TEASER_REASONING_EFFORT,
    TEASER_REVIEW_REASONING_EFFORT,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

VOD_OUTPUT_DIR = (
    PROJECT_ROOT
    / "vod_output"
)

TEASER_DIR = (
    VOD_OUTPUT_DIR
    / "teasers"
)


# ============================================================
# CONFIG
# ============================================================

TEASER_ANALYZER_VERSION = 1
TEASER_ANALYZER_REVISION = 7

# Generic safety bounds only. The cold-open length itself is measured from the
# event (ai/editor/intro_bounds.py): onset, reaction decay, whole phrases and
# shot boundaries. There is no per-strength or per-type duration.
MIN_TEASER_DURATION = intro_bounds.MIN_INTRO_S
MAX_TEASER_DURATION = intro_bounds.MAX_INTRO_S

# Güçlü sayılması için.
RECOMMENDED_SCORE = 7.0

# A spoken phrase and a nearby multimodal peak are one event when this close.
PHRASE_PEAK_UNION_GAP = 1.10


# ============================================================
# AI OUTPUT SCHEMA
# ============================================================

TEASER_SCHEMA = {
    "type": "object",

    "properties": {
        "recommended": {
            "type": "boolean"
        },

        "score": {
            "type": "number"
        },

        "teaser_type": {
            "type": "string",

            "enum": [
                "shock",
                "confusion",
                "reaction",
                "conflict",
                "absurdity",
                "reveal",
                "rage",
                "hype",
                "funny",
                "quote",
            ],
        },

        "selection_mode": {
            "type": "string",
            "enum": ["spoken_phrase", "peak_window"]
        },

        "peak_id": {
            "type": "integer",
            "minimum": -1
        },

        "peak_alignment_score": {
            "type": "number"
        },

        "start_word_id": {
            "type": "integer",
            "minimum": -1
        },

        "end_word_id": {
            "type": "integer",
            "minimum": -1
        },

        "reason": {
            "type": "string"
        },

        "viewer_question": {
            "type": "string"
        },

        "spoiler_risk": {
            "type": "string",

            "enum": [
                "low",
                "medium",
                "high",
            ],
        },
    },

    "required": [
        "recommended",
        "score",
        "teaser_type",
        "selection_mode",
        "peak_id",
        "peak_alignment_score",
        "start_word_id",
        "end_word_id",
        "reason",
        "viewer_question",
        "spoiler_risk",
    ],

    "additionalProperties": False,
}


# ============================================================
# AI INSTRUCTIONS
# ============================================================

TEASER_INSTRUCTIONS = """
You are a world-class short-form video editor specializing in viral
English-language gaming, livestream, Twitch, YouTube and reaction clips.

You are selecting a COLD OPEN / TEASER from an already selected clip.

The final short will be edited like this:

    [TEASER FROM A STRONG LATER MOMENT]
                    ↓
              hard restart
                    ↓
    [THE MAIN CLIP FROM ITS REAL BEGINNING]

The teaser scene will therefore appear once at the beginning and then
naturally appear again later when the main clip reaches that moment.

This repetition is intentional.


============================================================
YOUR MAIN JOB
============================================================

Decide WHICH real moment is the cold open. You are NOT writing a hook sentence.

The cold open is either:

- a MULTIMODAL PEAK from the provided candidates (selection_mode="peak_window",
  peak_id = that candidate), optionally together with the exact spoken words
  that belong to it (start_word_id/end_word_id, else -1), or
- a SPOKEN LINE (selection_mode="spoken_phrase", start_word_id..end_word_id
  from AVAILABLE WORDS), optionally linked to the peak it belongs to (peak_id,
  else -1).

Each peak candidate has: peak_id, core time range, the window MIMIR measured
for it, combined/audio/visual scores, signals, visual_description and
nearby_text. If a silent physical event is the strongest moment, select it. If
a spoken reaction aligned with audio+visual evidence is strongest, select it.
Do not select a weak spoken phrase just because it has "clean" words.

You never return timestamps. MIMIR measures where the chosen event starts and
finishes (sound onset, reaction decay, complete phrases, shot changes) and
cuts the cold open from that evidence.


============================================================
WHAT MAKES A GREAT TEASER
============================================================

Prioritize moments such as:

- "What the fuck is this?"
- extreme confusion
- disbelief
- sudden anger
- absurd statements
- accusations
- shocking reactions
- hilarious misunderstandings
- quotable insults
- emotional escalation
- a question that demands context
- a sentence that makes the viewer wonder:
  "Why did he say that?"
  "What happened?"
  "Who is he talking about?"
  "How did they get here?"


============================================================
THE TEASER MUST WORK WITHOUT CONTEXT
============================================================

A viewer should be able to hear the teaser without knowing what happened
before it and still feel curiosity.

Bad teaser:

"Yeah, that's what I said."

Why?
It needs previous context.

Good teaser:

"What the fuck is this?"

Why?
It instantly creates a question.


============================================================
DO NOT JUST PICK THE LOUDEST LINE
============================================================

Profanity alone does not make a good teaser.

The line needs at least one of:

- curiosity
- emotional intensity
- absurdity
- conflict
- surprise
- a strong unanswered question


============================================================
IMPORTANT: PREFER A LATER MOMENT
============================================================

This teaser is meant to preview something the viewer will encounter
later in the clip.

Therefore, when two moments are similarly strong, strongly prefer the
one that happens later rather than the opening line.

Do NOT select the first line merely because it is energetic.

The purpose is:

    strong future moment
        ↓
    restart
        ↓
    viewer watches to understand how we reach it


============================================================
DO NOT DESTROY THE PAYOFF
============================================================

A teaser may come from the payoff region, but avoid revealing so much
that there is no reason left to watch the main clip.

Prefer:

reaction
question
confusion
partial revelation

over:

complete explanation of the entire joke/outcome

when both choices are similarly strong.


============================================================
NATURAL SPEECH BOUNDARIES
============================================================

Choose the complete natural spoken phrase.

Do NOT cut:

"What the..."

when the real useful line is:

"What the fuck is this?"

Do NOT include unrelated dialogue before or after it just to make the
teaser longer.

Do not think about duration. Choose the words that form the complete line;
MIMIR keeps the immediately adjacent scream, visible reaction, impact or payoff
beat that belongs to the same event and never pads with unrelated dead air.


============================================================
WORD IDS
============================================================

You will receive AVAILABLE WORDS.

Example:

ID 51 | WHAT
ID 52 | THE
ID 53 | FUCK
ID 54 | IS
ID 55 | THIS?

A valid response would be:

start_word_id = 51
end_word_id   = 55

The chosen words must exist in AVAILABLE WORDS.

Do NOT invent word IDs.

The selected range must follow the order of the provided words.


============================================================
SCORE
============================================================

Score teaser strength from 0 to 10.

9-10:
Immediately arresting. Excellent cold open.

8:
Strong and clearly useful.

7:
Good but not exceptional.

Below 7:
Probably better to let the clip start naturally.

Set recommended=false when a teaser would make the clip worse or when
there is no sufficiently strong self-contained spoken moment.


============================================================
VIEWER QUESTION
============================================================

viewer_question should describe the curiosity created in the viewer.

Examples:

"Why is he so confused?"

"What did this guy do?"

"Why does he think this player is terrible?"

Do not use clickbait language.


============================================================
SPOILER RISK
============================================================

low:
Creates curiosity without giving away the main answer.

medium:
Reveals part of the payoff but still creates meaningful curiosity.

high:
Basically gives away the entire joke/outcome.

Prefer low or medium when possible.


============================================================
MULTIMODAL PEAK SUPPORT
============================================================

You may receive INTRO PEAK SUPPORT generated from:
- measured audio-energy/suddenness peaks
- Gemini factual visual peak regions
- their time alignment

This support exists specifically so you do NOT miss a scream, shout, visible
reaction, impact, reveal, fall, chaos spike, or other high-energy moment merely
because the transcript wording looks ordinary.

Prefer a strong AUDIO+VISUAL aligned peak when it also creates curiosity or
emotional stopping power. Do not blindly choose the numerically highest peak;
Terra remains the editor and must consider context, spoiler risk and story.

Use selection_mode="peak_window" when the strongest cold open is primarily
a non-verbal or multimodal moment and word IDs alone would cut away the scream,
reaction, impact, reveal, or visual peak. Set peak_id to the supplied PEAK ID.
If there is no useful spoken anchor inside that peak, start_word_id=end_word_id=-1.

Use selection_mode="spoken_phrase" when the spoken phrase itself is the main
reason the teaser works. When a nearby peak strengthens that phrase, set peak_id
to that PEAK ID so the renderer can preserve the surrounding reaction. Otherwise
set peak_id=-1.

Never invent visual facts beyond the supplied support.


============================================================
FINAL RULE
============================================================

The best teaser is NOT necessarily the most important sentence.

The best teaser is the sentence that makes someone stop scrolling
and want to see how the clip reaches that moment.
""".strip()


# ============================================================
# JSON
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


# ============================================================
# GENERIC
# ============================================================

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
            value,
            maximum,
        ),
    )


# ============================================================
# VALIDATE TIMELINE
# ============================================================

def validate_timeline(
    timeline: dict[str, Any],
) -> None:

    if timeline.get(
        "version"
    ) != 3:

        raise RuntimeError(
            "Teaser Analyzer yalnızca Timeline V3 ile çalışır."
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

    source = timeline.get(
        "source"
    )

    if not isinstance(
        source,
        dict,
    ):

        raise RuntimeError(
            "Timeline source bilgisi bulunamadı."
        )


# ============================================================
# VALIDATE TRANSCRIPT
# ============================================================

def validate_transcript(
    transcript: dict[str, Any],
) -> None:

    words = transcript.get(
        "words"
    )

    if (
        not isinstance(
            words,
            list,
        )
        or not words
    ):

        raise RuntimeError(
            "Transcript word timestamps içermiyor."
        )

    source = transcript.get(
        "source"
    )

    if not isinstance(
        source,
        dict,
    ):

        raise RuntimeError(
            "Transcript source bilgisi bulunamadı."
        )


# ============================================================
# GET CLIP
# ============================================================

def get_clip(
    timeline: dict[str, Any],
    clip_index: int,
) -> dict[str, Any]:

    timelines = timeline[
        "timelines"
    ]

    for position, clip in enumerate(
        timelines,
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
        f"clip_index={clip_index} bulunamadı."
    )


# ============================================================
# CUT RANGES
# ============================================================

def normalize_cut_ranges(
    clip: dict[str, Any],
) -> list[dict[str, float]]:

    raw_ranges = clip.get(
        "cut_ranges",
        [],
    )

    ranges: list[
        dict[str, float]
    ] = []

    for item in raw_ranges:

        if not isinstance(
            item,
            dict,
        ):
            continue

        try:

            start = float(
                item.get(
                    "start",
                    0,
                )
            )

            end = float(
                item.get(
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
        key=lambda item: item[
            "start"
        ]
    )

    # Merge overlapping cuts.
    merged: list[
        dict[str, float]
    ] = []

    for current in ranges:

        if not merged:

            merged.append(
                current.copy()
            )

            continue

        previous = merged[
            -1
        ]

        if (
            current[
                "start"
            ]
            <= previous[
                "end"
            ]
        ):

            previous[
                "end"
            ] = max(
                previous[
                    "end"
                ],
                current[
                    "end"
                ],
            )

        else:

            merged.append(
                current.copy()
            )

    return merged


# ============================================================
# SOURCE -> EDITED TIME
# ============================================================

def source_to_edited_time(
    source_time: float,
    cut_ranges: list[dict[str, float]],
) -> float:

    source_time = float(
        source_time
    )

    removed = 0.0

    for cut in cut_ranges:

        cut_start = float(
            cut[
                "start"
            ]
        )

        cut_end = float(
            cut[
                "end"
            ]
        )

        # Timestamp cuttan önce.
        if (
            source_time
            <= cut_start
        ):
            break

        # Timestamp cuttan tamamen sonra.
        if (
            source_time
            >= cut_end
        ):

            removed += (
                cut_end
                - cut_start
            )

            continue

        # Timestamp cut'ın içinde.
        removed += (
            source_time
            - cut_start
        )

        break

    return max(
        0.0,
        source_time - removed,
    )


# ============================================================
# EDITED -> SOURCE TIME
# ============================================================

def edited_to_source_time(
    edited_time: float,
    cut_ranges: list[dict[str, float]],
) -> float:
    edited_time = max(0.0, float(edited_time))
    source_cursor = 0.0
    edited_cursor = 0.0

    for cut in cut_ranges:
        cut_start = float(cut["start"])
        cut_end = float(cut["end"])
        kept_length = max(0.0, cut_start - source_cursor)

        if edited_time <= edited_cursor + kept_length:
            return source_cursor + (edited_time - edited_cursor)

        edited_cursor += kept_length
        source_cursor = max(source_cursor, cut_end)

    return source_cursor + max(0.0, edited_time - edited_cursor)


def get_peak_candidate(
    peak_support: dict[str, Any] | None,
    peak_id: int,
) -> dict[str, Any] | None:
    if peak_id < 0 or not isinstance(peak_support, dict):
        return None

    candidates = peak_support.get("candidates", [])
    if not isinstance(candidates, list):
        return None

    for item in candidates:
        if not isinstance(item, dict):
            continue
        try:
            current = int(item.get("peak_id", -1))
        except (TypeError, ValueError):
            continue
        if current == peak_id:
            return item

    return None


def words_overlapping_window(
    available_words: list[dict[str, Any]],
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    return [
        word
        for word in available_words
        if float(word.get("edited_end", 0.0)) >= start
        and float(word.get("edited_start", 0.0)) <= end
    ]


# ============================================================
# WORD REMOVED?
# ============================================================

def word_is_removed(
    relative_start: float,
    relative_end: float,
    cut_ranges: list[dict[str, float]],
) -> bool:

    midpoint = (
        relative_start
        + relative_end
    ) / 2.0

    for cut in cut_ranges:

        if (
            float(
                cut[
                    "start"
                ]
            )
            < midpoint
            < float(
                cut[
                    "end"
                ]
            )
        ):

            return True

    return False


# ============================================================
# BUILD AVAILABLE WORDS
# ============================================================

def build_available_words(
    transcript: dict[str, Any],
    clip: dict[str, Any],
) -> list[dict[str, Any]]:

    source = clip.get(
        "source",
        {}
    )

    try:

        absolute_clip_start = float(
            source[
                "absolute_start"
            ]
        )

        absolute_clip_end = float(
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

    cut_ranges = normalize_cut_ranges(
        clip
    )

    result: list[
        dict[str, Any]
    ] = []

    for word in transcript.get(
        "words",
        []
    ):

        if not isinstance(
            word,
            dict,
        ):
            continue

        try:

            word_id = int(
                word[
                    "id"
                ]
            )

            absolute_start = float(
                word[
                    "start"
                ]
            )

            absolute_end = float(
                word[
                    "end"
                ]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        text = str(
            word.get(
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

        # Sadece bu clip içindeki kelimeler.
        if not (
            absolute_clip_start
            <= midpoint
            <= absolute_clip_end
        ):

            continue

        relative_start = (
            absolute_start
            - absolute_clip_start
        )

        relative_end = (
            absolute_end
            - absolute_clip_start
        )

        # Otomatik pacing cut ile silinmiş kelimeyi
        # teaser adayı yapma.
        if word_is_removed(
            relative_start=relative_start,
            relative_end=relative_end,
            cut_ranges=cut_ranges,
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

                "absolute_start": (
                    absolute_start
                ),

                "absolute_end": (
                    absolute_end
                ),

                "relative_start": (
                    relative_start
                ),

                "relative_end": (
                    relative_end
                ),

                "edited_start": (
                    edited_start
                ),

                "edited_end": (
                    edited_end
                ),
            }
        )

    if not result:

        raise RuntimeError(
            "Bu clip için kullanılabilir transcript kelimesi bulunamadı."
        )

    return result


# ============================================================
# FORMAT WORDS FOR AI
# ============================================================

def format_words_for_ai(
    words: list[dict[str, Any]],
) -> str:

    lines = []

    for item in words:

        lines.append(
            (
                f"ID {item['id']} | "
                f"clip {item['relative_start']:.2f}"
                f"-{item['relative_end']:.2f}s | "
                f"{item['word']}"
            )
        )

    return "\n".join(
        lines
    )


# ============================================================
# CLIP CONTEXT FOR AI
# ============================================================

def build_clip_context(
    clip: dict[str, Any],
) -> str:

    title = str(
        clip.get(
            "title",
            ""
        )
    )

    emotion = str(
        clip.get(
            "emotion",
            ""
        )
    )

    hook = clip.get(
        "hook",
        {}
    )

    hook_text = ""

    if isinstance(
        hook,
        dict,
    ):

        hook_text = str(
            hook.get(
                "text",
                "",
            )
        )

    payoff = clip.get(
        "payoff",
        {}
    )

    payoff_info = ""

    if isinstance(
        payoff,
        dict,
    ):

        payoff_start = payoff.get(
            "source_start"
        )

        payoff_end = payoff.get(
            "source_end"
        )

        payoff_info = (
            f"{payoff_start} -> {payoff_end}"
        )

    highlights = clip.get(
        "caption_highlights",
        [],
    )

    if not isinstance(
        highlights,
        list,
    ):

        highlights = []

    highlight_text = ", ".join(
        str(
            value
        )
        for value in highlights
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

    return f"""
CLIP TITLE:
{title}

PRIMARY EMOTION:
{emotion}

OLD TEXT HOOK:
{hook_text}

KNOWN PAYOFF REGION IN CLIP-RELATIVE SOURCE TIME:
{payoff_info}

IMPORTANT CAPTION PHRASES:
{highlight_text}

TERRA EDITORIAL CORE:
{editorial_text}
""".strip()


# ============================================================
# AI CALL
# ============================================================

def request_teaser_choice(
    clip: dict[str, Any],
    available_words: list[dict[str, Any]],
    peak_support: dict[str, Any] | None = None,
    review_note: str | None = None,
) -> dict[str, Any]:

    words_text = format_words_for_ai(
        available_words
    )

    context = build_clip_context(
        clip
    )

    peak_text = intro_peak_support.format_for_ai(
        peak_support or {}
    )

    review_text = ""
    if review_note:
        review_text = f"""

============================================================
PEAK REVIEW REQUIRED
============================================================

{review_note}

Re-evaluate the teaser from scratch. You MAY keep the original editorial idea
only if it is genuinely stronger than the highlighted multimodal peak.
""".rstrip()

    input_text = f"""
{context}

============================================================
MULTIMODAL INTRO PEAKS
============================================================

{peak_text}

============================================================
AVAILABLE WORDS
============================================================

{words_text}

Choose the strongest cold-open teaser using the real words AND supplied peak windows.
{review_text}

Remember:

- If the best stopping-power moment is a supplied multimodal/non-verbal peak,
  use selection_mode=peak_window and its peak_id.
- If speech is the main hook, use selection_mode=spoken_phrase.
- Prefer a strong later moment.
- Preserve screams, shouting, visible reactions and impacts that sit immediately
  beside the selected words instead of trimming them away.
- If a high-confidence audio+visual peak is clearly stronger than ordinary dialogue,
  do not ignore it just because its transcript is simple or because part of the
  reaction is non-verbal.
- Prefer audio+visual aligned peaks when they are genuinely strong and curious.
- Do not create text or timestamps.
- Do not unnecessarily reveal the whole outcome.
- If no reliable peak exists, fall back to the best real spoken phrase.
- Even when the teaser is not strong enough to recommend on its own, still
  choose the single best technically usable moment. The downstream intro judge
  may pair it with a strong hook.
""".strip()

    print()
    print(
        "🧠 Luna teaser anını seçiyor..."
    )

    reasoning_effort = (
        TEASER_REVIEW_REASONING_EFFORT
        if review_note
        else TEASER_REASONING_EFFORT
    )

    print(
        f"🤖 Model: {TEASER_MODEL} "
        f"[{reasoning_effort}]"
    )

    response = client.responses.create(
        model=TEASER_MODEL,
        reasoning={
            "effort": reasoning_effort,
        },

        instructions=(
            TEASER_INSTRUCTIONS
        ),

        input=input_text,

        text={
            "format": {
                "type": "json_schema",

                "name": (
                    "viral_teaser_selection"
                ),

                "strict": True,

                "schema": (
                    TEASER_SCHEMA
                ),
            }
        },
    )

    output_text = str(
        response.output_text
    ).strip()

    if not output_text:

        raise RuntimeError(
            "AI boş teaser analizi döndürdü."
        )

    try:

        data = json.loads(
            output_text
        )

    except json.JSONDecodeError as error:

        raise RuntimeError(
            "AI geçersiz teaser JSON döndürdü:\n\n"
            + output_text
        ) from error

    return data


def _choice_peak_strength(
    ai_result: dict[str, Any],
    peak_support: dict[str, Any],
    available_words: list[dict[str, Any]],
) -> float:
    try:
        peak_id = int(ai_result.get("peak_id", -1))
    except (TypeError, ValueError):
        peak_id = -1

    peak = get_peak_candidate(peak_support, peak_id)
    if peak is not None:
        return float(peak.get("combined_score", 0.0))

    try:
        start_id = int(ai_result.get("start_word_id", -1))
        end_id = int(ai_result.get("end_word_id", -1))
    except (TypeError, ValueError):
        return 0.0

    if start_id < 0 or end_id < 0:
        return 0.0

    try:
        words = get_selected_words(available_words, start_id, end_id)
    except Exception:
        return 0.0

    start = float(words[0]["edited_start"])
    end = float(words[-1]["edited_end"])
    best = 0.0
    for item in peak_support.get("candidates", []):
        if not isinstance(item, dict):
            continue
        p_start = float(item.get("teaser_start", item.get("start", 0.0)))
        p_end = float(item.get("teaser_end", item.get("end", p_start)))
        if not (end < p_start or start > p_end):
            best = max(best, float(item.get("combined_score", 0.0)))
    return best


def peak_review_note(
    ai_result: dict[str, Any],
    peak_support: dict[str, Any],
    available_words: list[dict[str, Any]],
) -> str | None:
    candidates = peak_support.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return None

    multimodal = [
        item for item in candidates
        if isinstance(item, dict) and bool(item.get("multimodal"))
    ]
    if not multimodal:
        return None

    top = max(multimodal, key=lambda item: float(item.get("combined_score", 0.0)))
    top_score = float(top.get("combined_score", 0.0))
    if top_score < 0.72:
        return None

    chosen_score = _choice_peak_strength(ai_result, peak_support, available_words)
    try:
        alignment = float(ai_result.get("peak_alignment_score", 0.0)) / 10.0
    except (TypeError, ValueError):
        alignment = 0.0

    # Only spend a second Terra call when the first choice materially underuses
    # a strong audio+visual region.
    if chosen_score >= top_score - 0.16 and alignment >= 0.60:
        return None

    signals = ", ".join(str(x) for x in top.get("signals", []))
    visual = str(top.get("visual_description", "")).strip()
    nearby = str(top.get("nearby_text", "")).strip()
    return (
        f"The first choice appears weakly aligned with the strongest multimodal evidence.\n"
        f"Re-check PEAK {top.get('peak_id')} ({float(top.get('teaser_start',0)):.2f}-"
        f"{float(top.get('teaser_end',0)):.2f}s): combined={top_score:.2f}, "
        f"audio={float(top.get('audio_score',0)):.2f}, visual={float(top.get('visual_score',0)):.2f}.\n"
        f"Signals: {signals or 'audio+visual peak'}.\n"
        f"Visual fact: {visual or 'n/a'}.\n"
        f"Nearby speech: {nearby or 'none / mostly non-verbal'}.\n"
        "Do not automatically pick it, but explicitly compare its stopping power, curiosity, "
        "spoiler risk and story value against your first choice."
    )


# ============================================================
# GET SELECTED WORD RANGE
# ============================================================

def get_selected_words(
    available_words: list[dict[str, Any]],
    start_word_id: int,
    end_word_id: int,
) -> list[dict[str, Any]]:

    start_position = None
    end_position = None

    for position, word in enumerate(
        available_words
    ):

        word_id = int(
            word[
                "id"
            ]
        )

        if (
            word_id
            == start_word_id
        ):

            start_position = (
                position
            )

        if (
            word_id
            == end_word_id
        ):

            end_position = (
                position
            )

    if start_position is None:

        raise RuntimeError(
            f"AI geçersiz start_word_id seçti: "
            f"{start_word_id}"
        )

    if end_position is None:

        raise RuntimeError(
            f"AI geçersiz end_word_id seçti: "
            f"{end_word_id}"
        )

    if (
        end_position
        < start_position
    ):

        raise RuntimeError(
            "AI teaser word range ters."
        )

    selected = available_words[
        start_position:
        end_position + 1
    ]

    if not selected:

        raise RuntimeError(
            "Teaser word range boş."
        )

    return selected


# ============================================================
# JOIN SELECTED TEXT
# ============================================================

def join_selected_words(
    words: list[dict[str, Any]],
) -> str:

    return " ".join(
        str(
            item[
                "word"
            ]
        ).strip()

        for item in words
    ).strip()


# ============================================================
# BUILD FINAL RESULT
# ============================================================

def boundary_words_for(
    available_words: list[dict[str, Any]],
    caption_profile: dict[str, Any] | None,
) -> list[intro_bounds.Word]:
    """Word clock used for cold-open boundaries.

    The final caption profile is measured on the exact paced-clip audio (the
    same clock the captions use); the whole-VOD transcript mapped through cuts
    is only the fallback when that profile is unavailable."""
    words = intro_bounds.words_from_profile(caption_profile)
    return words or intro_bounds.words_from_rows(available_words)


def build_teaser_result(
    clip: dict[str, Any],
    available_words: list[dict[str, Any]],
    ai_result: dict[str, Any],
    peak_support: dict[str, Any] | None = None,
    *,
    boundary_words: list[intro_bounds.Word] | None = None,
    envelope: intro_bounds.Envelope | None = None,
    media_path: str | Path | None = None,
) -> dict[str, Any]:
    """Editor's choice (WHICH event) -> evidence-measured cold-open window."""

    selection_mode = str(
        ai_result.get("selection_mode", "spoken_phrase")
    ).strip().lower()

    if selection_mode not in {"spoken_phrase", "peak_window"}:
        selection_mode = "spoken_phrase"

    try:
        peak_id = int(ai_result.get("peak_id", -1))
    except (TypeError, ValueError):
        peak_id = -1

    selected_peak = get_peak_candidate(
        peak_support,
        peak_id,
    )

    edited_info = clip.get("edited", {})
    try:
        clip_edited_duration = float(edited_info["estimated_duration"])
    except (KeyError, TypeError, ValueError):
        clip_edited_duration = max(
            float(word["edited_end"])
            for word in available_words
        )

    cut_ranges = normalize_cut_ranges(clip)

    try:
        start_word_id = int(ai_result.get("start_word_id", -1))
        end_word_id = int(ai_result.get("end_word_id", -1))
    except (TypeError, ValueError):
        start_word_id = end_word_id = -1
    selected_words: list[dict[str, Any]] = []
    if start_word_id >= 0 and end_word_id >= 0:
        selected_words = get_selected_words(
            available_words=available_words,
            start_word_id=start_word_id,
            end_word_id=end_word_id,
        )

    if selection_mode == "peak_window" and selected_peak is None:
        # The model referred to a peak that does not exist: its words are the event.
        selection_mode = "spoken_phrase"
    if selection_mode == "spoken_phrase" and not selected_words:
        raise RuntimeError("spoken_phrase seçildi ama geçerli word ID verilmedi.")

    # ---- event core (WHAT happened), then evidence bounds (HOW LONG it lasts)
    lead_in: float | None = None
    if selection_mode == "peak_window":
        core_start = float(selected_peak.get("start", 0.0))
        core_end = float(selected_peak.get("end", core_start))
        if selected_words:
            speech_start = float(selected_words[0]["edited_start"])
            speech_end = float(selected_words[-1]["edited_end"])
            if speech_start < core_start:
                lead_in = speech_start
            core_end = max(core_end, min(speech_end, core_start + MAX_TEASER_DURATION))
    else:
        core_start = float(selected_words[0]["edited_start"])
        core_end = float(selected_words[-1]["edited_end"])
        if selected_peak is not None:
            peak_start = float(selected_peak.get("start", 0.0))
            peak_end = float(selected_peak.get("end", peak_start))
            gap = max(0.0, peak_start - core_end, core_start - peak_end)
            if gap <= PHRASE_PEAK_UNION_GAP:
                core_start, core_end = min(core_start, peak_start), max(core_end, peak_end)

    words = boundary_words if boundary_words is not None else intro_bounds.words_from_rows(available_words)
    event = intro_bounds.EventEvidence(core_start, core_end, lead_in_start=lead_in, source=selection_mode)
    first_pass = intro_bounds.compute_intro_bounds(
        event, clip_duration=clip_edited_duration, words=words, envelope=envelope)
    cuts = intro_bounds.scene_cuts(media_path, first_pass.start - 1.0, first_pass.end + 1.0) if media_path else []
    bounds = intro_bounds.compute_intro_bounds(
        event, clip_duration=clip_edited_duration, words=words, envelope=envelope, cuts=cuts)
    teaser_edited_start = bounds.start
    teaser_edited_end = bounds.end
    teaser_duration = teaser_edited_end - teaser_edited_start

    if not selected_words:
        selected_words = words_overlapping_window(
            available_words,
            teaser_edited_start,
            teaser_edited_end,
        )

    # Word metadata may be absent for a purely non-verbal peak.
    if selected_words:
        teaser_text = join_selected_words(selected_words)
        start_word_id = int(selected_words[0]["id"])
        end_word_id = int(selected_words[-1]["id"])
        speech_edited_start = float(selected_words[0]["edited_start"])
        speech_edited_end = float(selected_words[-1]["edited_end"])
    else:
        teaser_text = ""
        start_word_id = -1
        end_word_id = -1
        speech_edited_start = teaser_edited_start
        speech_edited_end = teaser_edited_end

    source = clip["source"]
    absolute_clip_start = float(source["absolute_start"])

    source_relative_start = edited_to_source_time(
        teaser_edited_start,
        cut_ranges,
    )
    source_relative_end = edited_to_source_time(
        teaser_edited_end,
        cut_ranges,
    )
    source_absolute_start = absolute_clip_start + source_relative_start
    source_absolute_end = absolute_clip_start + source_relative_end

    score = float(ai_result.get("score", 0))
    recommended = bool(ai_result.get("recommended", False))
    duration_valid = (
        MIN_TEASER_DURATION - 1e-6 <= teaser_duration <= MAX_TEASER_DURATION + 1e-6
    )
    final_recommended = (
        recommended
        and score >= RECOMMENDED_SCORE
        and duration_valid
    )

    warnings: list[str] = []
    if not duration_valid:
        warnings.append(
            "Teaser duration güvenli aralık dışında: "
            f"{teaser_duration:.2f}s"
        )
    if source_relative_start < 0.75:
        warnings.append(
            "Teaser ana clip'in başlangıcına çok yakın. Cold-open tekrar etkisi zayıf olabilir."
        )
    if selection_mode == "peak_window" and selected_peak is not None:
        if not bool(selected_peak.get("multimodal")):
            warnings.append(
                "Peak window yalnız tek modaliteyle destekleniyor; Terra gerekçe ile seçti."
            )

    peak_alignment_score = clamp(
        float(ai_result.get("peak_alignment_score", 0.0)),
        0.0,
        10.0,
    )

    return {
        "clip_index": int(clip.get("clip_index", 1)),
        "title": str(clip.get("title", "")),
        "recommended": final_recommended,
        "ai_recommended": recommended,
        "score": round(score, 1),
        "teaser_type": str(ai_result.get("teaser_type", "reaction")),
        "selection_mode": selection_mode,
        "peak_id": peak_id if selected_peak is not None else -1,
        "peak_alignment_score": round(peak_alignment_score, 1),
        "teaser_text": teaser_text,
        "word_range": {
            "start_id": start_word_id,
            "end_id": end_word_id,
            "word_count": len(selected_words),
        },
        "source": {
            "absolute_start": round_time(source_absolute_start),
            "absolute_end": round_time(source_absolute_end),
            "relative_start": round_time(source_relative_start),
            "relative_end": round_time(source_relative_end),
        },
        "edited": {
            "speech_start": round_time(speech_edited_start),
            "speech_end": round_time(speech_edited_end),
            "teaser_start": round_time(teaser_edited_start),
            "teaser_end": round_time(teaser_edited_end),
            "duration": round_time(teaser_duration),
            "duration_policy": "event_evidence",
            "bounds": bounds.to_dict(),
        },
        "multimodal_support": selected_peak or {},
        "locked_peak": {
            "peak_id": peak_id,
            "peak_start": round_time(float(selected_peak.get("start", 0.0))),
            "peak_end": round_time(float(selected_peak.get("end", 0.0))),
            "teaser_start": round_time(teaser_edited_start),
            "teaser_end": round_time(teaser_edited_end),
            "combined_score": round(float(selected_peak.get("combined_score", 0.0)), 3),
            "audio_score": round(float(selected_peak.get("audio_score", 0.0)), 3),
            "visual_score": round(float(selected_peak.get("visual_score", 0.0)), 3),
            "multimodal": bool(selected_peak.get("multimodal")),
            "signals": selected_peak.get("signals", []),
            "visual_description": str(selected_peak.get("visual_description", "")),
        } if selected_peak is not None else {},
        "reason": str(ai_result.get("reason", "")).strip(),
        "viewer_question": str(ai_result.get("viewer_question", "")).strip(),
        "spoiler_risk": str(ai_result.get("spoiler_risk", "medium")),
        "warnings": warnings,
        "render_plan": {
            "input": "clean_paced_clip",
            "prepend_teaser": True,
            "restart_main_clip": True,
            "remove_teaser_from_main_clip": False,
            "normal_captions_in_intro": False,
            "transition": "hard_cut",
        },
    }


# ============================================================
# ANALYZE ONE CLIP
# ============================================================

def analyze_clip(
    timeline: dict[str, Any],
    transcript: dict[str, Any],
    clip_index: int,
    edited_video_path: str | Path | None = None,
    video_report_path: str | Path | None = None,
    caption_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:

    clip = get_clip(
        timeline=timeline,
        clip_index=clip_index,
    )

    available_words = (
        build_available_words(
            transcript=transcript,
            clip=clip,
        )
    )

    print()
    print(
        "=" * 68
    )

    print(
        f"🎬 CLIP {clip_index}"
    )

    print(
        f"📛 {clip.get('title', '')}"
    )

    print(
        f"📝 Available words: "
        f"{len(available_words)}"
    )

    edited_info = clip.get("edited", {})
    try:
        clip_duration = float(edited_info.get("estimated_duration", 0.0))
    except (TypeError, ValueError):
        clip_duration = 0.0
    if clip_duration <= 0:
        clip_duration = max(float(word["edited_end"]) for word in available_words)

    envelope = intro_bounds.audio_envelope(edited_video_path)
    boundary_words = boundary_words_for(available_words, caption_profile)
    peak_support = intro_peak_support.build_peak_support(
        edited_video_path=edited_video_path,
        video_report_path=video_report_path,
        clip_duration=clip_duration,
        envelope=envelope,
        words=boundary_words,
    )
    intro_peak_support.attach_nearby_words(
        peak_support,
        available_words,
    )

    peak_candidates = peak_support.get("candidates", [])
    print(
        f"⚡ Intro peak candidates: {len(peak_candidates) if isinstance(peak_candidates, list) else 0} "
        f"(audio={peak_support.get('audio_available')}, visual={peak_support.get('visual_available')})"
    )

    ai_result = (
        request_teaser_choice(
            clip=clip,
            available_words=available_words,
            peak_support=peak_support,
        )
    )

    review_note = peak_review_note(
        ai_result,
        peak_support,
        available_words,
    )
    if review_note:
        print("🔁 Luna multimodal peak review yapıyor...")
        ai_result = request_teaser_choice(
            clip=clip,
            available_words=available_words,
            peak_support=peak_support,
            review_note=review_note,
        )

    result = build_teaser_result(
        clip=clip,
        available_words=available_words,
        ai_result=ai_result,
        peak_support=peak_support,
        boundary_words=boundary_words,
        envelope=envelope,
        media_path=edited_video_path,
    )

    print()
    print(
        f"🔥 Teaser: "
        f"{result['teaser_text']}"
    )

    print(
        f"⭐ Score: "
        f"{result['score']}/10"
    )

    print(
        f"⚡ Mode: {result.get('selection_mode')} | peak={result.get('peak_id')} | "
        f"alignment={result.get('peak_alignment_score')}/10"
    )

    print(
        f"✅ Recommended: "
        f"{result['recommended']}"
    )

    print(
        f"🎞️ Edited preview: "
        f"{result['edited']['teaser_start']:.2f}"
        f" → "
        f"{result['edited']['teaser_end']:.2f}"
        f" "
        f"({result['edited']['duration']:.2f}s)"
    )

    print(
        f"❓ Viewer question: "
        f"{result['viewer_question']}"
    )

    if result[
        "warnings"
    ]:

        print(
            "⚠️ Warnings:"
        )

        for warning in result[
            "warnings"
        ]:

            print(
                f"   - {warning}"
            )

    return result


# ============================================================
# SAVE
# ============================================================

def get_output_path(
    timeline_path: str | Path,
) -> Path:

    timeline_path = Path(
        timeline_path
    )

    base = (
        timeline_path.stem
    )

    suffix = (
        "_timeline_v3"
    )

    if base.endswith(
        suffix
    ):

        base = base[
            :-len(
                suffix
            )
        ]

    TEASER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        TEASER_DIR
        / f"{base}_teasers.json"
    ).resolve()


def save_results(
    timeline_path: str | Path,
    transcript_path: str | Path,
    results: list[dict[str, Any]],
    edited_video_path: str | Path | None = None,
    video_report_path: str | Path | None = None,
) -> Path:

    output_path = get_output_path(
        timeline_path
    )

    package = {
        "version": (
            TEASER_ANALYZER_VERSION
        ),

        "mode": (
            "cold_open_payoff_teaser"
        ),

        "model": TEASER_MODEL,
        "reasoning_effort": TEASER_REASONING_EFFORT,
        "review_reasoning_effort": TEASER_REVIEW_REASONING_EFFORT,
        "revision": TEASER_ANALYZER_REVISION,

        "inputs": {
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
            "edited_video": (
                str(Path(edited_video_path).resolve())
                if edited_video_path
                else None
            ),
            "video_report": (
                str(Path(video_report_path).resolve())
                if video_report_path
                else None
            ),
        },

        "clip_count": len(
            results
        ),

        "teasers": (
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
# MAIN
# ============================================================

def analyze_teasers(
    timeline_path: str | Path,
    transcript_path: str | Path,
    clip_index: int | None = None,
    edited_video_path: str | Path | None = None,
    video_report_path: str | Path | None = None,
    caption_profile_path: str | Path | None = None,
) -> dict[str, Any]:

    timeline_path = Path(
        timeline_path
    ).resolve()

    transcript_path = Path(
        transcript_path
    ).resolve()

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

    caption_profile: dict[str, Any] | None = None
    if caption_profile_path and Path(caption_profile_path).is_file():
        try:
            caption_profile = load_json(caption_profile_path)
        except Exception:
            caption_profile = None

    timelines = timeline[
        "timelines"
    ]

    results: list[
        dict[str, Any]
    ] = []

    if clip_index is not None:

        results.append(
            analyze_clip(
                timeline=timeline,
                transcript=transcript,
                clip_index=clip_index,
                edited_video_path=edited_video_path,
                video_report_path=video_report_path,
                caption_profile=caption_profile,
            )
        )

    else:

        for position, clip in enumerate(
            timelines,
            start=1,
        ):

            if isinstance(
                clip,
                dict,
            ):

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

            else:

                current_index = (
                    position
                )

            results.append(
                analyze_clip(
                    timeline=timeline,
                    transcript=transcript,
                    clip_index=current_index,
                    edited_video_path=edited_video_path,
                    video_report_path=video_report_path,
                )
            )

    output_path = save_results(
        timeline_path=timeline_path,
        transcript_path=transcript_path,
        results=results,
        edited_video_path=edited_video_path,
        video_report_path=video_report_path,
    )

    print()
    print(
        "=" * 68
    )

    print(
        "✅ TEASER ANALYZER TAMAMLANDI"
    )

    print(
        f"📂 {output_path}"
    )

    print(
        "=" * 68
    )

    return {
        "output_path": str(
            output_path
        ),

        "teasers": results,
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "MIMIR Teaser Analyzer V1"
    )

    print(
        "Cold-open / payoff teaser selection"
    )

    print()

    timeline_path = input(
        "Timeline V3 JSON yolunu gir: "
    ).strip().strip('"')

    transcript_path = input(
        "Transcript JSON yolunu gir: "
    ).strip().strip('"')

    clip_input = input(
        "Clip index "
        "(boş = tüm klipler): "
    ).strip()

    try:

        if clip_input:

            selected_clip = int(
                clip_input
            )

        else:

            selected_clip = None

        analyze_teasers(
            timeline_path=timeline_path,
            transcript_path=transcript_path,
            clip_index=selected_clip,
        )

    except Exception as error:

        print()
        print(
            "❌ TEASER ANALYZER HATASI:"
        )

        print(
            error
        )

