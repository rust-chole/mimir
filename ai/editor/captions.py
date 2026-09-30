from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CAPTION_OUTPUT_DIR = (
    PROJECT_ROOT
    / "vod_output"
    / "captions"
)


# ============================================================
# CONFIG
# ============================================================

CAPTION_VERSION = 24

PLAY_RES_X = 1080
PLAY_RES_Y = 1920

FONT_NAME = "Arial"
FONT_SIZE = 74

WORDS_PER_GROUP = 4
MAX_GROUP_CHARACTERS = 30

# Bu kadar uzun boşluk varsa yeni caption grubu başlat.
GROUP_BREAK_GAP = 0.42

# Kelime bittikten sonra sıradaki kelime çok yakınsa
# caption yeni kelime başlayana kadar ekranda kalır.
MAX_HOLD_GAP = 0.18

MIN_EVENT_DURATION = 0.07

# Acoustic word timestamps remain untouched. This floor applies only to the
# visible ASS event of the LAST word in a caption group, preventing 50-100 ms
# single-word flashes such as "Amy." while preserving the measured word onset.
MIN_TERMINAL_DISPLAY_DURATION = 0.32

# V23 two-lane stale-caption guard.
# This NEVER changes Whisper/acoustic timestamps. It only prevents one speaker's
# rendered word from visually hanging for seconds after that speaker stopped.
# Normal word timestamps in MIMIR are far shorter than this; >1.15 s is treated
# as a display anomaly in two-lane mode only.
MAX_TWO_LANE_EVENT_DURATION = 1.15
CROSS_LANE_HANDOFF_EPSILON = 0.025

# V24 adaptive two-lane policy.
# A second visual lane is NOT a permanent speaker lane. It exists only around
# measured simultaneous/interrupting speech. Ordinary A->B conversation stays
# on the same main caption position.
MIN_TRUE_SPEECH_OVERLAP = 0.045
OVERLAP_WINDOW_PAD_BEFORE = 0.08
OVERLAP_WINDOW_PAD_AFTER = 0.14
WORD_OVERLAP_FALLBACK_PAD = 0.12
OVERLAP_WINDOW_MERGE_GAP = 0.12

# Caption'ın ekrandaki yüksekliği.
CAPTION_MARGIN_V = 530


# ASS renkleri BGR formatında.
BASE_TEXT_COLOR = "&H00FFFFFF"
INACTIVE_TEXT_COLOR = "&H00F2F2F2"

# Aktif kelime: sarı
ACTIVE_TEXT_COLOR = "&H0000D7FF"

# AI tarafından özellikle önemli seçilen kelime: turuncu
HIGHLIGHT_TEXT_COLOR = "&H000080FF"

# Secondary speaker palette (cyan/blue family).
SECONDARY_BASE_TEXT_COLOR = "&H00FFF7E8"
SECONDARY_ACTIVE_TEXT_COLOR = "&H00FFE054"
SECONDARY_HIGHLIGHT_TEXT_COLOR = "&H00FF6BA9"
SECONDARY_MARGIN_V = 650

OUTLINE_COLOR = "&H00000000"
SHADOW_COLOR = "&HFF000000"


# ============================================================
# HELPERS
# ============================================================

def load_json(
    path: str | Path,
) -> dict[str, Any]:

    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"JSON bulunamadı: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


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


def normalize_word(
    text: str,
) -> str:

    text = str(
        text
    ).strip().casefold()

    return re.sub(
        r"[^\w']+",
        "",
        text,
    )


# ============================================================
# ASS TIME
# ============================================================

def ass_time(
    seconds: float,
) -> str:

    seconds = max(
        0.0,
        float(seconds),
    )

    total_centiseconds = int(
        round(
            seconds * 100
        )
    )

    hours = (
        total_centiseconds
        // 360000
    )

    total_centiseconds %= 360000

    minutes = (
        total_centiseconds
        // 6000
    )

    total_centiseconds %= 6000

    secs = (
        total_centiseconds
        // 100
    )

    centiseconds = (
        total_centiseconds
        % 100
    )

    return (
        f"{hours}:"
        f"{minutes:02d}:"
        f"{secs:02d}."
        f"{centiseconds:02d}"
    )


def escape_ass_text(
    text: str,
) -> str:

    text = str(text)

    text = text.replace(
        "\\",
        r"\\",
    )

    text = text.replace(
        "{",
        r"\{",
    )

    text = text.replace(
        "}",
        r"\}",
    )

    text = text.replace(
        "\n",
        " ",
    )

    return text.strip()


# ============================================================
# TIMELINE
# ============================================================

def get_clip_timeline(
    timeline_data: dict[str, Any],
    clip_index: int,
) -> dict[str, Any]:

    for clip in timeline_data.get(
        "timelines",
        [],
    ):

        if int(
            clip.get(
                "clip_index",
                -1,
            )
        ) == int(
            clip_index
        ):

            return clip

    raise RuntimeError(
        f"Timeline içinde clip_index={clip_index} bulunamadı."
    )


# ============================================================
# CUT RANGES
# ============================================================

def get_cut_ranges(
    clip_timeline: dict[str, Any],
) -> list[dict[str, float]]:

    ranges = []

    for item in clip_timeline.get(
        "cut_ranges",
        [],
    ):

        start = float(
            item.get(
                "start",
                0,
            )
        )

        end = float(
            item.get(
                "end",
                0,
            )
        )

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

    return ranges


def time_is_removed(
    source_time: float,
    cut_ranges: list[dict[str, float]],
) -> bool:

    for cut in cut_ranges:

        if (
            float(
                cut["start"]
            )
            < source_time
            < float(
                cut["end"]
            )
        ):

            return True

    return False


def interval_is_removed(
    start: float,
    end: float,
    cut_ranges: list[dict[str, float]],
) -> bool:

    midpoint = (
        float(start)
        + float(end)
    ) / 2.0

    return time_is_removed(
        midpoint,
        cut_ranges,
    )


def map_source_to_edited_time(
    source_time: float,
    cut_ranges: list[dict[str, float]],
) -> float:

    source_time = max(
        0.0,
        float(source_time),
    )

    removed_before = 0.0

    for cut in cut_ranges:

        start = float(
            cut["start"]
        )

        end = float(
            cut["end"]
        )

        if source_time >= end:

            removed_before += (
                end
                - start
            )

            continue

        if source_time > start:

            return round_time(
                start
                - removed_before
            )

        break

    return round_time(
        source_time
        - removed_before
    )


# ============================================================
# SOURCE WORDS
# ============================================================

def get_source_words(
    transcript: dict[str, Any],
    clip_timeline: dict[str, Any],
) -> list[dict[str, Any]]:

    source = clip_timeline.get(
        "source",
        {},
    )

    clip_start = float(
        source.get(
            "absolute_start",
            0,
        )
    )

    clip_end = float(
        source.get(
            "absolute_end",
            clip_start,
        )
    )

    if clip_end <= clip_start:

        raise RuntimeError(
            "Timeline içindeki clip source aralığı geçersiz."
        )

    result: list[dict[str, Any]] = []

    for item in transcript.get(
        "words",
        [],
    ):

        absolute_start = float(
            item.get(
                "start",
                0,
            )
        )

        absolute_end = float(
            item.get(
                "end",
                0,
            )
        )

        word = str(
            item.get(
                "word",
                "",
            )
        ).strip()

        if not word:
            continue

        if absolute_end <= clip_start:
            continue

        if absolute_start >= clip_end:
            break

        source_start = max(
            0.0,
            absolute_start
            - clip_start,
        )

        source_end = min(
            clip_end
            - clip_start,

            absolute_end
            - clip_start,
        )

        if source_end <= source_start:

            source_end = (
                source_start
                + MIN_EVENT_DURATION
            )

        result.append(
            {
                "word": word,

                "normalized": normalize_word(
                    word
                ),

                "source_start": round_time(
                    source_start
                ),

                "source_end": round_time(
                    source_end
                ),
            }
        )

    return result


# ============================================================
# SPEAKER POLICY
# ============================================================

# Caption Sync V5 no longer remaps or clamps the final-clip word clock here.
# Speaker assignment is performed in speaker_caption_support.py, and the
# timestamp authority remains the edited clip transcription.


# ============================================================
# EDITED WORDS
# ============================================================

def build_edited_words(
    source_words: list[dict[str, Any]],
    cut_ranges: list[dict[str, float]],
    edited_duration: float,
) -> list[dict[str, Any]]:

    edited_words: list[dict[str, Any]] = []

    for item in source_words:

        source_start = float(
            item[
                "source_start"
            ]
        )

        source_end = float(
            item[
                "source_end"
            ]
        )

        if interval_is_removed(
            source_start,
            source_end,
            cut_ranges,
        ):

            continue

        edited_start = (
            map_source_to_edited_time(
                source_start,
                cut_ranges,
            )
        )

        edited_end = (
            map_source_to_edited_time(
                source_end,
                cut_ranges,
            )
        )

        edited_start = clamp(
            edited_start,
            0.0,
            edited_duration,
        )

        edited_end = clamp(
            edited_end,
            0.0,
            edited_duration,
        )

        if (
            edited_end
            - edited_start
            < MIN_EVENT_DURATION
        ):

            edited_end = min(
                edited_duration,
                edited_start
                + MIN_EVENT_DURATION,
            )

        if edited_end <= edited_start:
            continue

        edited_words.append(
            {
                **item,

                "edited_start": round_time(
                    edited_start
                ),

                "edited_end": round_time(
                    edited_end
                ),
            }
        )

    return edited_words


# ============================================================
# EMPHASIS
# ============================================================

def get_emphasis_markers(
    clip_timeline: dict[str, Any],
) -> list[dict[str, Any]]:

    markers = []

    for event in clip_timeline.get(
        "events",
        [],
    ):

        if event.get(
            "type"
        ) != "caption_emphasis":

            continue

        markers.append(
            {
                "word": normalize_word(
                    event.get(
                        "word",
                        "",
                    )
                ),

                "source_time": float(
                    event.get(
                        "source_time",
                        -999,
                    )
                ),
            }
        )

    return markers


def word_is_emphasized(
    word: dict[str, Any],
    markers: list[dict[str, Any]],
) -> bool:

    normalized = word.get(
        "normalized",
        "",
    )

    source_start = float(
        word.get(
            "source_start",
            0,
        )
    )

    for marker in markers:

        if marker[
            "word"
        ] != normalized:

            continue

        if abs(
            marker[
                "source_time"
            ]
            - source_start
        ) <= 0.08:

            return True

    return False


# ============================================================
# CAPTION GROUPS
# ============================================================

def should_break_after_word(
    text: str,
) -> bool:

    text = str(
        text
    ).strip()

    return text.endswith(
        (
            ".",
            "!",
            "?",
            ";",
            ":",
        )
    )


def split_caption_groups(
    words: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:

    if not words:
        return []

    groups: list[
        list[
            dict[str, Any]
        ]
    ] = []

    current: list[
        dict[str, Any]
    ] = []

    character_count = 0

    for word in words:

        text = str(
            word[
                "word"
            ]
        )

        if current:

            previous = (
                current[-1]
            )

            gap = (
                float(
                    word[
                        "edited_start"
                    ]
                )
                - float(
                    previous[
                        "edited_end"
                    ]
                )
            )

            projected_characters = (
                character_count
                + 1
                + len(text)
            )

            previous_label = str(previous.get("speaker_label", "")).strip()
            current_label = str(word.get("speaker_label", "")).strip()

            # V24: raw A/B diarization is metadata, not a caption-layout command.
            # Only a visible HUMAN-confirmed name transition may force a group
            # boundary. If both identities are blank, dialogue flows naturally.
            speaker_changed = (
                (previous_label or current_label)
                and previous_label.casefold() != current_label.casefold()
            )

            if (
                speaker_changed
                or gap
                >= GROUP_BREAK_GAP

                or len(
                    current
                )
                >= WORDS_PER_GROUP

                or projected_characters
                > MAX_GROUP_CHARACTERS
            ):

                groups.append(
                    current
                )

                current = []
                character_count = 0

        current.append(
            word
        )

        character_count += (
            len(text)
            + (
                1
                if len(
                    current
                ) > 1
                else 0
            )
        )

        if should_break_after_word(
            text
        ):

            groups.append(
                current
            )

            current = []
            character_count = 0

    if current:

        groups.append(
            current
        )

    return groups


# ============================================================
# WORD STYLE
# ============================================================

def _speaker_palette(role: str) -> tuple[str, str, str, str]:
    if role == "secondary":
        return (
            SECONDARY_BASE_TEXT_COLOR,
            SECONDARY_ACTIVE_TEXT_COLOR,
            SECONDARY_HIGHLIGHT_TEXT_COLOR,
            "ViralSecondary",
        )
    return (
        INACTIVE_TEXT_COLOR,
        ACTIVE_TEXT_COLOR,
        HIGHLIGHT_TEXT_COLOR,
        "ViralMain",
    )


def render_word(
    word: dict[str, Any],
    *,
    active: bool,
    emphasized: bool,
    role: str,
) -> str:
    text = escape_ass_text(word["word"])
    inactive_color, active_color, highlight_color, style_name = _speaker_palette(role)

    if active and emphasized:
        return (
            "{"
            "\\b1"
            f"\\c{highlight_color}"
            "\\fscx94"
            "\\fscy94"
            "\\t(0,70,\\fscx116\\fscy116)"
            "}"
            f"{text}"
            f"{{\\r{style_name}}}"
        )

    if active:
        return (
            "{"
            "\\b1"
            f"\\c{active_color}"
            "\\fscx96"
            "\\fscy96"
            "\\t(0,70,\\fscx108\\fscy108)"
            "}"
            f"{text}"
            f"{{\\r{style_name}}}"
        )

    if emphasized:
        return (
            "{"
            "\\b1"
            f"\\c{highlight_color}"
            "}"
            f"{text}"
            f"{{\\r{style_name}}}"
        )

    return (
        "{"
        f"\\c{inactive_color}"
        "}"
        f"{text}"
        f"{{\\r{style_name}}}"
    )


def build_caption_text(
    group: list[dict[str, Any]],
    active_index: int,
    emphasis_markers: list[dict[str, Any]],
) -> str:
    rendered = []
    role = str(group[0].get("speaker_role", "main")) if group else "main"
    label = str(group[0].get("speaker_label", "")).strip() if group else ""
    _, _, _, style_name = _speaker_palette(role)

    for index, word in enumerate(group):
        # V21: keep the full group's geometry stable, but do NOT reveal future
        # words before they are spoken. Transparent text still occupies layout
        # width in libass, so the caption does not jump horizontally as words
        # become visible. Previous words remain visible, current word is active.
        if index > active_index:
            text = escape_ass_text(word["word"])
            rendered.append(f"{{\\alpha&HFF&}}{text}{{\\r{style_name}}}")
            continue
        rendered.append(
            render_word(
                word,
                active=(index == active_index),
                emphasized=word_is_emphasized(word, emphasis_markers),
                role=role,
            )
        )

    body = " ".join(rendered)
    if label:
        _, active_color, _, style_name = _speaker_palette(role)
        prefix = f"{{\\b1\\c{active_color}}}{escape_ass_text(label)}:{{\\r{style_name}}}"
        return f"{prefix} {body}"
    return body


# ============================================================
# EVENT TIMING
# ============================================================

def get_event_end(
    group: list[dict[str, Any]],
    active_index: int,
    edited_duration: float,
    next_group_start: float | None = None,
    strict_no_overlap: bool = False,
) -> float:
    """Return DISPLAY end without changing the acoustic word clock.

    Intermediate words still hand off exactly at the next measured onset. The
    terminal word of a group receives a small readability floor, capped at the
    next group's onset when one exists. This fixes unreadable 50-100 ms flashes
    without pushing any following caption later.
    """
    current = group[active_index]
    start = float(current["edited_start"])
    end = float(current["edited_end"])
    terminal = active_index + 1 >= len(group)

    if not terminal:
        next_word = group[active_index + 1]
        next_start = float(next_word["edited_start"])
        gap = next_start - end
        if 0.0 <= gap <= MAX_HOLD_GAP:
            end = next_start
        # Never let an active word event run through the next measured onset.
        end = min(end, next_start)
        if strict_no_overlap and next_start > start:
            # In diarization hard-failure mode, readability floors must never
            # create two simultaneous ASS events on the single fail-safe lane.
            return min(max(end, start), next_start, edited_duration)
        end = max(end, min(next_start, start + MIN_EVENT_DURATION))
    else:
        # Display-only hold. Acoustic edited_end is preserved in the word data.
        desired = max(end, start + MIN_TERMINAL_DISPLAY_DURATION)
        if next_group_start is not None and next_group_start > start:
            if strict_no_overlap:
                # Hard-failure lane: keep a tiny terminal word readable, but do
                # not grant the normal 320 ms hold when another phrase begins
                # immediately.  The next group's first event may be delayed by
                # at most this small display-only floor; acoustic timestamps in
                # the profile remain untouched.
                minimum_readable_end = max(end, start + MIN_EVENT_DURATION)
                if next_group_start >= minimum_readable_end:
                    return min(next_group_start, edited_duration)
                return min(minimum_readable_end, edited_duration)
            desired = min(desired, next_group_start)
            # Normal trusted-speaker path keeps the existing readability rule.
            if 0.0 <= next_group_start - end <= MAX_HOLD_GAP:
                desired = max(desired, next_group_start)
        end = desired

    return min(max(end, start + MIN_EVENT_DURATION), edited_duration)


# ============================================================
# ASS HEADER
# ============================================================

def build_ass_header() -> str:

    return f"""[Script Info]
Title: MIMIR Unified Clean Captions V15
ScriptType: v4.00+
PlayResX: {PLAY_RES_X}
PlayResY: {PLAY_RES_Y}
ScaledBorderAndShadow: yes
WrapStyle: 2
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ViralMain,{FONT_NAME},{FONT_SIZE},{BASE_TEXT_COLOR},{BASE_TEXT_COLOR},{OUTLINE_COLOR},{SHADOW_COLOR},-1,0,0,0,100,100,0,0,1,6,2,2,70,70,{CAPTION_MARGIN_V},1
Style: ViralSecondary,{FONT_NAME},{FONT_SIZE},{SECONDARY_BASE_TEXT_COLOR},{SECONDARY_BASE_TEXT_COLOR},{OUTLINE_COLOR},{SHADOW_COLOR},-1,0,0,0,100,100,0,0,1,6,2,2,70,70,{SECONDARY_MARGIN_V},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""



# ============================================================
# FINAL-CLIP WORD CLOCK
# ============================================================

def _profile_edited_words(
    speaker_profile: dict[str, Any] | None,
    edited_duration: float,
) -> list[dict[str, Any]]:
    """Read words transcribed directly from the final edited clip.

    Speaker metadata is WHO-only. Word timing is the caption stack's single
    word-alignment clock on the exact final edited short and is never clamped
    to diarization segment boundaries. The current `exact_final_short_audio`
    marker and the earlier `edited_clip` / `exact_final_48k_audio` markers are
    accepted.
    """
    if not speaker_profile or str(speaker_profile.get("status", "")) != "ok":
        return []
    timing_basis = str(speaker_profile.get("timing_basis", "")).strip()
    if timing_basis not in {"edited_clip", "exact_final_48k_audio", "exact_final_short_audio"}:
        return []

    result: list[dict[str, Any]] = []
    for raw in speaker_profile.get("words", []) or []:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("word", "")).strip()
        if not text:
            continue
        try:
            start = clamp(float(raw.get("edited_start", 0.0)), 0.0, edited_duration)
            end = clamp(float(raw.get("edited_end", start)), 0.0, edited_duration)
        except (TypeError, ValueError):
            continue
        if end - start < MIN_EVENT_DURATION:
            end = min(edited_duration, start + MIN_EVENT_DURATION)
        if end <= start:
            continue
        result.append(
            {
                "word": text,
                "normalized": normalize_word(text),
                # source_start/source_end are retained only for compatibility
                # with styling helpers.  They are NOT remapped through cuts.
                "source_start": round_time(start),
                "source_end": round_time(end),
                "edited_start": round_time(start),
                "edited_end": round_time(end),
                "speaker_raw": str(raw.get("speaker_raw", "")),
                "speaker_role": str(raw.get("speaker_role", "main")) or "main",
                "speaker_label": str(raw.get("speaker_label", "")),
                "speaker_confidence": float(raw.get("speaker_confidence", 0.0) or 0.0),
                "timing_source": str(raw.get("timing_source", "edited_clip")),
            }
        )

    # Keep chronological order but allow real A/B overlap in dual speech.
    result.sort(key=lambda item: (float(item["edited_start"]), float(item["edited_end"])))
    return result


# ============================================================
# SPEAKER RENDER FAIL-SAFE
# ============================================================

def _speaker_render_hard_failure(
    speaker_profile: dict[str, Any] | None,
) -> bool:
    """Return True only when the diarization boundary audit hard-failed.

    This is intentionally render-only.  The speaker profile, identities and
    Whisper word clock stay untouched; unreliable WHO metadata simply loses
    permission to split/recolor captions for this render.
    """
    if not speaker_profile or str(speaker_profile.get("status", "")) != "ok":
        return False
    audit = speaker_profile.get("speaker_boundary_audit", {}) or {}
    return bool(audit.get("hard_failure", False))


def _trusted_human_display_map(
    speaker_profile: dict[str, Any] | None,
) -> dict[str, str]:
    """Return only user-confirmed/raw->display mappings safe for fail-safe render.

    Naming already writes display_labels after the human voice checkpoint.
    In a diarization hard-failure we do NOT trust raw boundaries for grouping,
    but a name explicitly confirmed by the user remains valid display metadata.
    """
    if not isinstance(speaker_profile, dict):
        return {}

    display = speaker_profile.get("display_labels", {})
    if not isinstance(display, dict):
        return {}

    source = str(
        (speaker_profile.get("speaker_names") or {}).get("source", "")
        if isinstance(speaker_profile.get("speaker_names"), dict)
        else ""
    ).casefold()

    calibration = speaker_profile.get("identity_calibration", {})
    human_raws: set[str] = set()

    if isinstance(calibration, dict):
        for item in calibration.values():
            if not isinstance(item, dict):
                continue
            if not bool(item.get("human_verified", False)):
                continue
            raw = str(item.get("raw_speaker", "")).strip()
            if raw:
                human_raws.add(raw)

    manual_source = (
        "manual" in source
        or "human" in source
        or "voice_calibrated" in source
    )

    trusted: dict[str, str] = {}
    for raw, name in display.items():
        raw_key = str(raw).strip()
        clean_name = str(name).strip()
        if not raw_key or not clean_name:
            continue
        if manual_source or raw_key in human_raws:
            trusted[raw_key] = clean_name

    return trusted


def _neutralize_speaker_render_metadata(
    words: list[dict[str, Any]],
    *,
    trusted_display_map: dict[str, str] | None = None,
    lane_raw_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Prepare hard-failure speaker metadata for safe rendering.

    V22 keeps at most two defensible speaker lanes alive. This is the important
    difference from V17-V20: a hard boundary failure no longer forces every
    word into one visual lane when we still know there are two real speakers.

    Human-confirmed names stay authoritative. Untrusted extra clusters lose
    their identity and fall back to the main lane. Text/timestamps are untouched.
    """
    trusted_display_map = trusted_display_map or {}
    lane_raw_map = lane_raw_map or {}
    output: list[dict[str, Any]] = []

    for word in words:
        item = dict(word)
        raw_speaker = str(item.get("speaker_raw", "")).strip()
        label = str(item.get("speaker_label", "")).strip()

        if raw_speaker in trusted_display_map:
            label = trusted_display_map[raw_speaker]

        if raw_speaker in lane_raw_map:
            # Keep this raw id only as a render-lane key. It cannot alter the
            # acoustic word clock or wording.
            item["speaker_raw"] = raw_speaker
            item["speaker_role"] = lane_raw_map[raw_speaker]
        else:
            # Any third/unstable cluster loses permission to create a third
            # caption lane.
            item["speaker_raw"] = ""
            item["speaker_role"] = "main"

        item["speaker_label"] = label
        item["speaker_confidence"] = 0.0
        output.append(item)

    return output


def _fail_safe_group_display_label(
    group: list[dict[str, Any]],
) -> str:
    """Choose a human label without letting uncertain words erase it.

    Important V19 rule:
    - blank/unresolved words do NOT cancel a confirmed human name;
    - exactly one distinct non-empty human name in the group -> show it;
    - two different non-empty names in the same visual group -> show no name.

    Thus broken diarization may make a group unlabeled, but it cannot invent
    the wrong person's name and cannot erase a clean human identity merely
    because one neighbouring word was unresolved.
    """
    if not group:
        return ""

    labels = [
        str(word.get("speaker_label", "")).strip()
        for word in group
        if str(word.get("speaker_label", "")).strip()
    ]

    if not labels:
        return ""

    unique: list[str] = []
    seen: set[str] = set()
    for label in labels:
        folded = label.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        unique.append(label)

    if len(unique) != 1:
        return ""

    return unique[0]




def _fail_safe_two_lane_raw_map(
    speaker_profile: dict[str, Any] | None,
    trusted_display_map: dict[str, str],
) -> dict[str, str]:
    """Choose at most two raw speakers that may own independent render lanes.

    This helper is used only when the diarization boundary audit hard-failed.
    We do not blindly trust every A/B/C/D/E cluster. Two-lane rendering is
    enabled only when there is a compact, defensible two-speaker mapping:

    1) explicit primary+secondary from the profile, or
    2) exactly two participant speakers, or
    3) two human-confirmed raw speaker identities.

    The result maps raw speaker id -> ``main`` / ``secondary``. Word text and
    acoustic timestamps are never changed.
    """
    if not isinstance(speaker_profile, dict):
        return {}

    primary = str(speaker_profile.get("primary_speaker") or "").strip()
    secondary = str(speaker_profile.get("secondary_speaker") or "").strip()
    if primary and secondary and primary != secondary:
        return {primary: "main", secondary: "secondary"}

    participants = [
        str(value).strip()
        for value in (speaker_profile.get("participant_speakers") or [])
        if str(value).strip()
    ]
    participants = list(dict.fromkeys(participants))
    if len(participants) == 2:
        return {participants[0]: "main", participants[1]: "secondary"}

    trusted_raws = [
        raw for raw in trusted_display_map
        if str(raw).strip()
    ]
    trusted_raws = list(dict.fromkeys(trusted_raws))
    if len(trusted_raws) == 2:
        return {trusted_raws[0]: "main", trusted_raws[1]: "secondary"}

    return {}



def _render_two_speaker_raw_map(
    speaker_profile: dict[str, Any] | None,
    trusted_display_map: dict[str, str],
) -> dict[str, str]:
    """Return at most two defensible raw speaker ids for layout purposes only.

    This does NOT decide what name is printed and never changes word timing.
    """
    if not isinstance(speaker_profile, dict):
        return {}

    primary = str(speaker_profile.get("primary_speaker") or "").strip()
    secondary = str(speaker_profile.get("secondary_speaker") or "").strip()
    if primary and secondary and primary != secondary:
        return {primary: "main", secondary: "secondary"}

    participants = [
        str(value).strip()
        for value in (speaker_profile.get("participant_speakers") or [])
        if str(value).strip()
    ]
    participants = list(dict.fromkeys(participants))
    if len(participants) == 2:
        return {participants[0]: "main", participants[1]: "secondary"}

    trusted_raws = [str(raw).strip() for raw in trusted_display_map if str(raw).strip()]
    trusted_raws = list(dict.fromkeys(trusted_raws))
    if len(trusted_raws) == 2:
        return {trusted_raws[0]: "main", trusted_raws[1]: "secondary"}

    return {}


def _merge_time_windows(
    windows: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    clean = sorted(
        (max(0.0, float(start)), max(0.0, float(end)))
        for start, end in windows
        if float(end) > float(start)
    )
    if not clean:
        return []

    merged: list[list[float]] = [[clean[0][0], clean[0][1]]]
    for start, end in clean[1:]:
        previous = merged[-1]
        if start <= previous[1] + OVERLAP_WINDOW_MERGE_GAP:
            previous[1] = max(previous[1], end)
        else:
            merged.append([start, end])

    return [(round_time(start), round_time(end)) for start, end in merged]


def _adaptive_overlap_windows(
    speaker_profile: dict[str, Any] | None,
    words: list[dict[str, Any]],
    raw_map: dict[str, str],
) -> list[tuple[float, float]]:
    """Find only real two-person collision windows.

    Priority:
    1) diarization segments that physically overlap;
    2) exact final-word clocks that physically overlap.

    Sequential turn-taking is deliberately NOT a collision.
    """
    if len(raw_map) != 2:
        return []

    main_raw = next((raw for raw, role in raw_map.items() if role == "main"), "")
    secondary_raw = next((raw for raw, role in raw_map.items() if role == "secondary"), "")
    if not main_raw or not secondary_raw:
        return []

    windows: list[tuple[float, float]] = []

    # Full speaker-turn evidence first. If two diarized turns overlap, keep the
    # whole secondary turn on lane 2 so a sentence is never split mid-phrase.
    if isinstance(speaker_profile, dict):
        segments = [
            item
            for item in (speaker_profile.get("segments") or [])
            if isinstance(item, dict)
        ]
        main_segments = [
            item for item in segments
            if str(item.get("speaker", "")).strip() == main_raw
        ]
        secondary_segments = [
            item for item in segments
            if str(item.get("speaker", "")).strip() == secondary_raw
        ]

        for secondary in secondary_segments:
            try:
                s_start = float(secondary.get("start", 0.0))
                s_end = float(secondary.get("end", s_start))
            except (TypeError, ValueError):
                continue
            if s_end <= s_start:
                continue

            for main in main_segments:
                try:
                    m_start = float(main.get("start", 0.0))
                    m_end = float(main.get("end", m_start))
                except (TypeError, ValueError):
                    continue
                overlap = min(s_end, m_end) - max(s_start, m_start)
                if overlap >= MIN_TRUE_SPEECH_OVERLAP:
                    windows.append(
                        (
                            max(0.0, s_start - OVERLAP_WINDOW_PAD_BEFORE),
                            s_end + OVERLAP_WINDOW_PAD_AFTER,
                        )
                    )
                    break

    # Word-clock fallback catches overlap that segment normalization missed.
    main_words = [
        word for word in words
        if str(word.get("speaker_raw", "")).strip() == main_raw
    ]
    secondary_words = [
        word for word in words
        if str(word.get("speaker_raw", "")).strip() == secondary_raw
    ]

    for secondary in secondary_words:
        s_start = float(secondary.get("edited_start", 0.0))
        s_end = float(secondary.get("edited_end", s_start))
        for main in main_words:
            m_start = float(main.get("edited_start", 0.0))
            m_end = float(main.get("edited_end", m_start))
            overlap = min(s_end, m_end) - max(s_start, m_start)
            if overlap >= MIN_TRUE_SPEECH_OVERLAP:
                windows.append(
                    (
                        max(0.0, s_start - WORD_OVERLAP_FALLBACK_PAD),
                        s_end + WORD_OVERLAP_FALLBACK_PAD,
                    )
                )
                break

    return _merge_time_windows(windows)


def _interval_hits_windows(
    start: float,
    end: float,
    windows: list[tuple[float, float]],
) -> bool:
    for win_start, win_end in windows:
        if min(end, win_end) - max(start, win_start) > 0.0:
            return True
    return False


def _prepare_adaptive_render_words(
    words: list[dict[str, Any]],
    *,
    speaker_profile: dict[str, Any] | None,
    trusted_display_map: dict[str, str],
) -> tuple[list[dict[str, Any]], list[tuple[float, float]]]:
    """Apply the final V24 render policy without touching text or timestamps.

    - Human names are sticky and authoritative.
    - Blank naming means BLANK on screen: no automatic A/B/X placeholders.
    - Ordinary sequential dialogue uses one main visual lane.
    - Secondary lane appears only inside real measured overlap windows.
    """
    raw_map = _render_two_speaker_raw_map(
        speaker_profile,
        trusted_display_map,
    )

    collision_windows = _adaptive_overlap_windows(
        speaker_profile,
        words,
        raw_map,
    )

    secondary_raw = next(
        (raw for raw, role in raw_map.items() if role == "secondary"),
        "",
    )

    prepared: list[dict[str, Any]] = []
    for word in words:
        item = dict(word)
        raw = str(item.get("speaker_raw", "")).strip()

        # Never leak automatic placeholder identity into published captions.
        # A label is printable only if it came from the human naming checkpoint.
        item["speaker_label"] = trusted_display_map.get(raw, "")

        role = "main"
        if raw and raw == secondary_raw and collision_windows:
            start = float(item.get("edited_start", 0.0))
            end = float(item.get("edited_end", start))
            if _interval_hits_windows(start, end, collision_windows):
                role = "secondary"

        item["speaker_role"] = role
        prepared.append(item)

    return prepared, collision_windows


def _has_two_render_lanes(words: list[dict[str, Any]]) -> bool:
    roles = {
        str(word.get("speaker_role", "main"))
        for word in words
        if isinstance(word, dict)
    }
    return "main" in roles and "secondary" in roles


def _lane_words(
    words: list[dict[str, Any]],
    role: str,
) -> list[dict[str, Any]]:
    if role == "secondary":
        return [
            word for word in words
            if str(word.get("speaker_role", "main")) == "secondary"
        ]
    return [
        word for word in words
        if str(word.get("speaker_role", "main")) != "secondary"
    ]

# ============================================================
# CREATE ASS
# ============================================================

def create_ass_for_clip(
    transcript: dict[str, Any],
    clip_timeline: dict[str, Any],
    output_path: str | Path,
    speaker_profile: dict[str, Any] | None = None,
) -> Path:

    output_path = Path(
        output_path
    ).resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    timeline_duration = float(
        clip_timeline.get(
            "edited",
            {},
        ).get(
            "estimated_duration",
            0,
        )
    )

    profile_duration = 0.0
    if speaker_profile and str(speaker_profile.get("status", "")) == "ok":
        try:
            profile_duration = float(speaker_profile.get("clip_duration", 0.0) or 0.0)
        except (TypeError, ValueError):
            profile_duration = 0.0

    # When V5 has a final-clip transcription, the real rendered-video duration
    # is the caption clock authority. Timeline duration remains the fallback.
    edited_duration = profile_duration if profile_duration > 0 else timeline_duration

    if edited_duration <= 0:
        raise RuntimeError(
            "Caption için edited duration geçersiz."
        )

    speaker_render_fail_safe = _speaker_render_hard_failure(speaker_profile)

    edited_words = _profile_edited_words(
        speaker_profile=speaker_profile,
        edited_duration=edited_duration,
    )

    if not edited_words:
        # Safe compatibility fallback.  Old VOD timestamps are used only when
        # final-clip transcription was unavailable; no speaker filter is
        # allowed to erase words in this path.
        cut_ranges = get_cut_ranges(
            clip_timeline
        )
        source_words = get_source_words(
            transcript=transcript,
            clip_timeline=clip_timeline,
        )
        source_words = [
            {
                **dict(item),
                "speaker_raw": "",
                "speaker_role": "main",
                "speaker_label": "",
            }
            for item in source_words
        ]
        edited_words = build_edited_words(
            source_words=source_words,
            cut_ranges=cut_ranges,
            edited_duration=edited_duration,
        )

    if not edited_words:

        raise RuntimeError(
            "Bu klip için caption üretilecek kelime bulunamadı."
        )

    # V24 final render policy:
    # identity metadata may help detect real overlap, but it may NEVER fragment
    # ordinary blank-named dialogue into permanent A/B lanes.
    trusted_display_map = _trusted_human_display_map(speaker_profile)
    edited_words, collision_windows = _prepare_adaptive_render_words(
        edited_words,
        speaker_profile=speaker_profile,
        trusted_display_map=trusted_display_map,
    )

    emphasis_markers = (
        get_emphasis_markers(
            clip_timeline
        )
    )

    two_lane_render = _has_two_render_lanes(edited_words)

    # V24: lane 2 exists only inside measured collision windows. Normal A->B
    # turn-taking remains one stable main caption stream. During true overlap,
    # each lane owns its own grouping/hold clock.
    lane_specs: list[tuple[str, list[dict[str, Any]]]]
    if two_lane_render:
        lane_specs = [
            ("main", _lane_words(edited_words, "main")),
            ("secondary", _lane_words(edited_words, "secondary")),
        ]
    else:
        lane_specs = [("main", edited_words)]

    event_rows: list[tuple[float, int, str]] = []
    lane_free_at = {"main": 0.0, "secondary": 0.0}
    lane_words_by_role = {
        role: words_for_lane
        for role, words_for_lane in lane_specs
    }

    for lane_role, lane_word_list in lane_specs:
        if not lane_word_list:
            continue

        groups = split_caption_groups(lane_word_list)
        lane_index = 1 if lane_role == "secondary" else 0

        for group_index, group in enumerate(groups):
            next_group_start = None
            if group_index + 1 < len(groups) and groups[group_index + 1]:
                next_group_start = float(groups[group_index + 1][0]["edited_start"])

            for active_index, word in enumerate(group):
                start = float(word["edited_start"])

                # In hard-failure mode prevent collisions only WITHIN THE SAME
                # lane. Main and Secondary are intentionally allowed to overlap.
                if (
                    speaker_render_fail_safe
                    and active_index == 0
                    and start < lane_free_at[lane_role]
                    and lane_free_at[lane_role] - start <= 0.12
                ):
                    start = lane_free_at[lane_role]

                end = get_event_end(
                    group=group,
                    active_index=active_index,
                    edited_duration=edited_duration,
                    next_group_start=next_group_start,
                    strict_no_overlap=speaker_render_fail_safe,
                )

                # V23: two independent lanes may overlap ONLY when the measured
                # speech itself overlaps. A finished caption must not hang merely
                # because its own lane has no next phrase for several seconds.
                if two_lane_render:
                    acoustic_end = float(word.get("edited_end", end))
                    end = min(
                        end,
                        start + MAX_TWO_LANE_EVENT_DURATION,
                    )

                    other_role = (
                        "secondary"
                        if lane_role == "main"
                        else "main"
                    )
                    other_words = lane_words_by_role.get(other_role, [])

                    # Find the first onset from the other speaker that happens
                    # after this word acoustically ended. If it begins during a
                    # display-only hold, hand the screen over immediately.
                    for other_word in other_words:
                        other_start = float(other_word.get("edited_start", 0.0))
                        if other_start < acoustic_end - CROSS_LANE_HANDOFF_EPSILON:
                            continue
                        if other_start >= end:
                            break
                        if other_start > start:
                            end = max(
                                start + MIN_EVENT_DURATION,
                                other_start,
                            )
                            break

                if end <= start:
                    continue

                if speaker_render_fail_safe:
                    lane_free_at[lane_role] = max(lane_free_at[lane_role], end)

                render_group = group
                if speaker_render_fail_safe:
                    safe_label = _fail_safe_group_display_label(group)
                    render_group = [
                        {
                            **dict(item),
                            "speaker_label": safe_label,
                        }
                        for item in group
                    ]

                caption_text = build_caption_text(
                    group=render_group,
                    active_index=active_index,
                    emphasis_markers=emphasis_markers,
                )

                style_name = (
                    "ViralSecondary"
                    if lane_role == "secondary"
                    else "ViralMain"
                )

                event_text = (
                    "Dialogue: "
                    "0,"
                    f"{ass_time(start)},"
                    f"{ass_time(end)},"
                    f"{style_name},"
                    ","
                    "0,"
                    "0,"
                    "0,"
                    ","
                    f"{caption_text}"
                )
                event_rows.append((start, lane_index, event_text))

    # Keep the ASS deterministic while allowing simultaneous rows on two lanes.
    event_rows.sort(key=lambda row: (row[0], row[1]))
    events = [row[2] for row in event_rows]

    with output_path.open(
        "w",
        encoding="utf-8-sig",
    ) as file:

        file.write(
            build_ass_header()
        )

        file.write(
            "\n".join(
                events
            )
        )

        file.write(
            "\n"
        )

    return output_path


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

def build_output_directory(
    timeline_path: str | Path,
) -> Path:

    timeline_path = Path(
        timeline_path
    ).resolve()

    base_name = (
        timeline_path.stem
        .replace(
            "_timeline_v3",
            "",
        )
    )

    output_dir = (
        CAPTION_OUTPUT_DIR
        / base_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


# ============================================================
# SINGLE CLIP
# ============================================================

def create_clip_captions(
    transcript_path: str | Path,
    timeline_path: str | Path,
    clip_index: int,
    speaker_profile_path: str | Path | None = None,
) -> Path:

    transcript = load_json(
        transcript_path
    )

    timeline_data = load_json(
        timeline_path
    )

    speaker_profile = None
    if speaker_profile_path:
        try:
            speaker_profile = load_json(speaker_profile_path)
        except Exception:
            speaker_profile = None

    if int(
        timeline_data.get(
            "version",
            -1,
        )
    ) != 3:

        raise RuntimeError(
            "Bu captions.py sürümü Timeline V3 bekliyor."
        )

    clip_timeline = get_clip_timeline(
        timeline_data=timeline_data,
        clip_index=clip_index,
    )

    output_dir = build_output_directory(
        timeline_path
    )

    output_path = (
        output_dir
        / f"clip_{clip_index:02d}_captions_v{CAPTION_VERSION}.ass"
    )

    result = create_ass_for_clip(
        transcript=transcript,
        clip_timeline=clip_timeline,
        output_path=output_path,
        speaker_profile=speaker_profile,
    )

    print()
    print(
        f"✅ Clip {clip_index} caption hazır:"
    )

    print(
        result
    )

    return result


# ============================================================
# ALL CLIPS
# ============================================================

def create_all_captions(
    transcript_path: str | Path,
    timeline_path: str | Path,
) -> list[Path]:

    transcript = load_json(
        transcript_path
    )

    timeline_data = load_json(
        timeline_path
    )

    if int(
        timeline_data.get(
            "version",
            -1,
        )
    ) != 3:

        raise RuntimeError(
            "Bu captions.py sürümü Timeline V3 bekliyor."
        )

    timelines = timeline_data.get(
        "timelines",
        [],
    )

    if not timelines:

        raise RuntimeError(
            "Timeline V3 içinde klip bulunamadı."
        )

    output_dir = build_output_directory(
        timeline_path
    )

    created: list[Path] = []

    for clip_timeline in timelines:

        clip_index = int(
            clip_timeline[
                "clip_index"
            ]
        )

        output_path = (
            output_dir
            / f"clip_{clip_index:02d}_captions_v{CAPTION_VERSION}.ass"
        )

        result = create_ass_for_clip(
            transcript=transcript,
            clip_timeline=clip_timeline,
            output_path=output_path,
        )

        created.append(
            result
        )

        print(
            f"✅ Clip {clip_index}: "
            f"{output_path.name}"
        )

    return created


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=" * 65
    )

    print(
        "📝 MIMIR VIRAL CAPTIONS V3"
    )

    print(
        "=" * 65
    )

    transcript_path = input(
        "Transcript JSON yolunu gir: "
    ).strip().strip('"')

    timeline_path = input(
        "Timeline V3 JSON yolunu gir: "
    ).strip().strip('"')

    clip_index_text = input(
        "Clip index "
        "(boş = tüm klipler): "
    ).strip()

    try:

        if clip_index_text:

            create_clip_captions(
                transcript_path=transcript_path,
                timeline_path=timeline_path,
                clip_index=int(
                    clip_index_text
                ),
            )

        else:

            created = create_all_captions(
                transcript_path=transcript_path,
                timeline_path=timeline_path,
            )

            print()
            print(
                f"✅ Toplam {len(created)} caption dosyası oluşturuldu."
            )

    except Exception as error:

        print()
        print(
            f"❌ CAPTIONS V3 HATASI:\n{error}"
        )