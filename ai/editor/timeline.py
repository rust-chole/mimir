from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TIMELINE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "vod_output"
    / "timelines"
)


# ============================================================
# CONFIG
# ============================================================

TIMELINE_VERSION = 3
TIMELINE_REVISION = 7

# Terra money-moment protection. Automatic pacing cuts may never overlap
# these ranges; they are downgraded to REVIEW instead.
PROTECTED_RANGE_PADDING = 0.04

# Emergency pacing guard, not a duration target.
# Terra-protected beats are the main safety mechanism; obvious dead air should
# still be removable. These limits only stop catastrophic over-cutting.
MAX_AUTOMATIC_CUT_RATIO = 0.45
MAX_AUTOMATIC_CUT_SECONDS = 16.0
MIN_RETAINED_RATIO = 0.55
MIN_EDITED_STORY_DURATION = 14.0

# V19 post-pacing story-integrity guard. Protected causal handles are already
# inviolable; this second deterministic pass protects the *bridges between*
# them so pacing cannot leave the origin, peak and reaction as disconnected
# islands. It adds no AI call and only rolls back cuts when the local bridge
# would become visibly fragmented.
STORY_INTEGRITY_REASON_MARKERS = (
    "causal origin",
    "visual causal origin",
    "causal escalation",
    "reaction/consequence",
    "reaction",
    "consequence",
    "payoff",
)
STORY_INTEGRITY_SHORT_BRIDGE = 3.0
STORY_INTEGRITY_MEDIUM_BRIDGE = 7.0
STORY_INTEGRITY_SHORT_MAX_SINGLE_CUT = 0.45
STORY_INTEGRITY_MEDIUM_MAX_SINGLE_CUT = 0.85
STORY_INTEGRITY_LONG_MAX_SINGLE_CUT = 1.20
STORY_INTEGRITY_SHORT_MAX_REMOVAL_RATIO = 0.18
STORY_INTEGRITY_MEDIUM_MAX_REMOVAL_RATIO = 0.24
STORY_INTEGRITY_LONG_MAX_REMOVAL_RATIO = 0.32

HOOK_TEXT_DURATION = 2.2

MIN_CAPTION_EVENT_GAP = 0.18
MAX_OCCURRENCES_PER_HIGHLIGHT_WORD = 3

MAX_SFX_PER_CLIP = 3
MIN_SFX_GAP = 0.90

STOP_WORDS = {
    "a",
    "an",
    "the",
    "he",
    "she",
    "it",
    "its",
    "it's",
    "i",
    "you",
    "we",
    "they",
    "is",
    "are",
    "was",
    "were",
    "am",
    "be",
    "been",
    "being",
    "to",
    "of",
    "for",
    "in",
    "on",
    "at",
    "and",
    "or",
    "but",
    "so",
    "this",
    "that",
    "these",
    "those",
    "with",
    "as",
}


# ============================================================
# GENERIC HELPERS
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


def is_meaningful_word(
    text: str,
) -> bool:

    normalized = normalize_word(
        text
    )

    if not normalized:
        return False

    if normalized in STOP_WORDS:
        return False

    if len(normalized) <= 1:
        return False

    return True


def ranges_overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> bool:

    return (
        start_a < end_b
        and end_a > start_b
    )


# ============================================================
# PACING PATH
# ============================================================

def guess_pacing_path(
    analysis_path: str | Path,
) -> Path:

    analysis_path = Path(
        analysis_path
    ).resolve()

    base_name = (
        analysis_path.stem
        .replace(
            "_clips",
            "",
        )
    )

    return (
        PROJECT_ROOT
        / "vod_output"
        / "pacing"
        / f"{base_name}_pacing.json"
    )


# ============================================================
# CLIP WORDS
# ============================================================

def get_clip_words(
    transcript: dict[str, Any],
    clip_start: float,
    clip_end: float,
) -> list[dict[str, Any]]:

    words = transcript.get(
        "words",
        [],
    )

    result: list[dict[str, Any]] = []

    for item in words:

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

        relative_start = max(
            0.0,
            absolute_start - clip_start,
        )

        relative_end = min(
            clip_end - clip_start,
            absolute_end - clip_start,
        )

        if relative_end <= relative_start:
            relative_end = (
                relative_start
                + 0.05
            )

        result.append(
            {
                "word": word,

                "normalized": normalize_word(
                    word
                ),

                "source_start": round_time(
                    relative_start
                ),

                "source_end": round_time(
                    relative_end
                ),

                "absolute_start": round_time(
                    absolute_start
                ),

                "absolute_end": round_time(
                    absolute_end
                ),
            }
        )

    return result


# ============================================================
# PACING / CUTS
# ============================================================

def get_pacing_clip(
    pacing_data: dict[str, Any],
    clip_index: int,
) -> dict[str, Any] | None:

    for clip in pacing_data.get(
        "clips",
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

    return None


def merge_ranges(
    ranges: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    if not ranges:
        return []

    ordered = sorted(
        ranges,
        key=lambda item: float(
            item["start"]
        ),
    )

    merged: list[dict[str, Any]] = []

    current = dict(
        ordered[0]
    )

    for item in ordered[1:]:

        item_start = float(
            item["start"]
        )

        item_end = float(
            item["end"]
        )

        current_end = float(
            current["end"]
        )

        if item_start <= current_end + 0.01:

            current["end"] = max(
                current_end,
                item_end,
            )

            current["priority"] = max(
                int(
                    current.get(
                        "priority",
                        1,
                    )
                ),
                int(
                    item.get(
                        "priority",
                        1,
                    )
                ),
            )

            current["types"] = sorted(
                set(
                    current.get(
                        "types",
                        [],
                    )
                    + item.get(
                        "types",
                        [],
                    )
                )
            )

            continue

        merged.append(
            current
        )

        current = dict(
            item
        )

    merged.append(
        current
    )

    for item in merged:

        item["start"] = round_time(
            item["start"]
        )

        item["end"] = round_time(
            item["end"]
        )

        item["duration"] = round_time(
            float(
                item["end"]
            )
            - float(
                item["start"]
            )
        )

    return merged



# ============================================================
# TERRA MONEY-MOMENT PROTECTION
# ============================================================

def normalize_protected_ranges(
    clip: dict[str, Any],
    clip_start: float,
    clip_duration: float,
) -> list[dict[str, Any]]:
    """
    Analyzer stores must_keep_ranges as ORIGINAL VOD timestamps.
    Timeline converts them to clip-relative ranges.
    """

    clip_end = (
        clip_start
        + clip_duration
    )

    raw_ranges = clip.get(
        "must_keep_ranges",
        [],
    )

    result: list[
        dict[str, Any]
    ] = []

    if isinstance(
        raw_ranges,
        list,
    ):

        for item in raw_ranges:

            if not isinstance(
                item,
                dict,
            ):
                continue

            absolute_start = clamp(
                float(
                    item.get(
                        "start",
                        clip_start,
                    )
                ),
                clip_start,
                clip_end,
            )

            absolute_end = clamp(
                float(
                    item.get(
                        "end",
                        absolute_start,
                    )
                ),
                absolute_start,
                clip_end,
            )

            if absolute_end <= absolute_start:
                continue

            relative_start = max(
                0.0,
                absolute_start
                - clip_start
                - PROTECTED_RANGE_PADDING,
            )

            relative_end = min(
                clip_duration,
                absolute_end
                - clip_start
                + PROTECTED_RANGE_PADDING,
            )

            if relative_end <= relative_start:
                continue

            result.append(
                {
                    "start": round_time(
                        relative_start
                    ),

                    "end": round_time(
                        relative_end
                    ),

                    "duration": round_time(
                        relative_end
                        - relative_start
                    ),

                    "reason": str(
                        item.get(
                            "reason",
                            "Terra protected money moment.",
                        )
                    ),
                }
            )

    # Backward safety: payoff is always protected.
    payoff_absolute_start = clamp(
        float(
            clip.get(
                "payoff_start",
                clip_start,
            )
        ),
        clip_start,
        clip_end,
    )

    payoff_absolute_end = clamp(
        float(
            clip.get(
                "payoff_end",
                payoff_absolute_start,
            )
        ),
        payoff_absolute_start,
        clip_end,
    )

    if payoff_absolute_end > payoff_absolute_start:

        result.append(
            {
                "start": round_time(
                    max(
                        0.0,
                        payoff_absolute_start
                        - clip_start
                        - PROTECTED_RANGE_PADDING,
                    )
                ),

                "end": round_time(
                    min(
                        clip_duration,
                        payoff_absolute_end
                        - clip_start
                        + PROTECTED_RANGE_PADDING,
                    )
                ),

                "reason": (
                    "Payoff safety protection."
                ),
            }
        )

    if not result:
        return []

    result.sort(
        key=lambda item: (
            float(
                item["start"]
            ),
            float(
                item["end"]
            ),
        )
    )

    merged: list[
        dict[str, Any]
    ] = []

    current = dict(
        result[0]
    )

    reasons = [
        str(
            current.get(
                "reason",
                "",
            )
        )
    ]

    for item in result[
        1:
    ]:

        if (
            float(
                item["start"]
            )
            <= float(
                current["end"]
            )
            + 0.001
        ):

            current["end"] = max(
                float(
                    current["end"]
                ),
                float(
                    item["end"]
                ),
            )

            reason = str(
                item.get(
                    "reason",
                    "",
                )
            )

            if (
                reason
                and reason not in reasons
            ):
                reasons.append(
                    reason
                )

            current["reason"] = (
                " | ".join(
                    reasons
                )
            )

            continue

        current["start"] = round_time(
            current["start"]
        )

        current["end"] = round_time(
            current["end"]
        )

        current["duration"] = round_time(
            float(
                current["end"]
            )
            - float(
                current["start"]
            )
        )

        merged.append(
            current
        )

        current = dict(
            item
        )

        reasons = [
            str(
                current.get(
                    "reason",
                    "",
                )
            )
        ]

    current["start"] = round_time(
        current["start"]
    )

    current["end"] = round_time(
        current["end"]
    )

    current["duration"] = round_time(
        float(
            current["end"]
        )
        - float(
            current["start"]
        )
    )

    merged.append(
        current
    )

    return merged


def overlaps_protected_range(
    start: float,
    end: float,
    protected_ranges: list[dict[str, Any]],
) -> dict[str, Any] | None:

    for protected in protected_ranges:

        if ranges_overlap(
            start,
            end,
            float(
                protected["start"]
            ),
            float(
                protected["end"]
            ),
        ):

            return protected

    return None


def minimum_allowed_edited_duration(
    clip_duration: float,
) -> float:
    """
    Emergency floor only.

    Do not use this as a target duration. The actual editor should remove
    obvious unprotected dead air even when that makes a dense Short shorter.
    """

    clip_duration = max(
        0.0,
        float(
            clip_duration
        ),
    )

    if clip_duration <= 0:
        return 0.0

    if clip_duration >= 30.0:
        return round_time(
            max(
                18.0,
                clip_duration
                * MIN_RETAINED_RATIO,
            )
        )

    if clip_duration >= 20.0:
        return round_time(
            max(
                MIN_EDITED_STORY_DURATION,
                clip_duration
                * 0.60,
            )
        )

    if clip_duration >= 14.0:
        return round_time(
            max(
                11.0,
                clip_duration
                * 0.70,
            )
        )

    return round_time(
        clip_duration
        * 0.80
    )


def apply_automatic_cut_budget(
    automatic_ranges: list[dict[str, Any]],
    review_ranges: list[dict[str, Any]],
    clip_duration: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    float,
]:
    """
    Pick only the highest-value automatic cuts that fit the story-preservation
    budget. Everything else becomes REVIEW instead of silently shortening the
    final Short.
    """

    merged_automatic = merge_ranges(
        automatic_ranges
    )

    merged_review = merge_ranges(
        review_ranges
    )

    minimum_duration = (
        minimum_allowed_edited_duration(
            clip_duration
        )
    )

    ratio_budget = max(
        0.0,
        clip_duration
        * MAX_AUTOMATIC_CUT_RATIO,
    )

    duration_budget = max(
        0.0,
        clip_duration
        - minimum_duration,
    )

    budget = min(
        ratio_budget,
        duration_budget,
        MAX_AUTOMATIC_CUT_SECONDS,
    )

    # Highest priority first; for equal priority take the larger obvious dead
    # air. Output is sorted chronologically again afterwards.
    ordered = sorted(
        merged_automatic,
        key=lambda item: (
            -int(
                item.get(
                    "priority",
                    1,
                )
            ),
            -float(
                item.get(
                    "duration",
                    0.0,
                )
            ),
            float(
                item.get(
                    "start",
                    0.0,
                )
            ),
        ),
    )

    accepted: list[
        dict[str, Any]
    ] = []

    spent = 0.0

    for item in ordered:

        duration = max(
            0.0,
            float(
                item.get(
                    "duration",
                    0.0,
                )
            ),
        )

        if (
            duration > 0
            and spent + duration
            <= budget + 1e-9
        ):
            accepted.append(
                item
            )
            spent += duration
            continue

        review_item = dict(
            item
        )

        review_item[
            "blocked_by_cut_budget"
        ] = True

        review_item[
            "reason"
        ] = (
            "Automatic cut yalnızca emergency minimum-duration guard'ını "
            "aşacağı için REVIEW'e çevrildi. Protected beat yoksa diğer "
            "dead-air cut'ları uygulanmaya devam eder."
        )

        merged_review.append(
            review_item
        )

    accepted.sort(
        key=lambda item: float(
            item[
                "start"
            ]
        )
    )

    merged_review = merge_ranges(
        merged_review
    )

    return (
        accepted,
        merged_review,
        round_time(
            budget
        ),
    )


def build_cut_ranges(
    pacing_clip: dict[str, Any] | None,
    clip_duration: float,
    payoff_start: float,
    payoff_end: float,
    protected_ranges: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:

    if pacing_clip is None:
        return [], []

    automatic_ranges: list[dict[str, Any]] = []
    review_ranges: list[dict[str, Any]] = []

    for suggestion in pacing_clip.get(
        "suggestions",
        [],
    ):

        action = str(
            suggestion.get(
                "action",
                "",
            )
        ).casefold()

        relative = suggestion.get(
            "relative",
            {},
        )

        start = clamp(
            float(
                relative.get(
                    "start",
                    0,
                )
            ),
            0.0,
            clip_duration,
        )

        end = clamp(
            float(
                relative.get(
                    "end",
                    0,
                )
            ),
            0.0,
            clip_duration,
        )

        if end <= start:
            continue

        item = {
            "start": round_time(
                start
            ),

            "end": round_time(
                end
            ),

            "duration": round_time(
                end - start
            ),

            "priority": int(
                suggestion.get(
                    "priority",
                    1,
                )
            ),

            "types": [
                str(
                    suggestion.get(
                        "type",
                        "dead_air",
                    )
                )
            ],

            "reason": str(
                suggestion.get(
                    "reason",
                    "",
                )
            ),
        }

        if action in {
            "cut",
            "trim",
        }:

            protected_hit = (
                overlaps_protected_range(
                    start,
                    end,
                    protected_ranges,
                )
            )

            if protected_hit is not None:

                item["reason"] = (
                    "Pacing kesimi Terra'nın protected money-moment "
                    "aralığıyla çakıştığı için otomatik kesimden çıkarıldı. "
                    "Protected: "
                    + str(
                        protected_hit.get(
                            "reason",
                            "",
                        )
                    )
                )

                item[
                    "blocked_by_protected_range"
                ] = True

                review_ranges.append(
                    item
                )

                continue

            if ranges_overlap(
                start,
                end,
                payoff_start,
                payoff_end,
            ):

                item["reason"] = (
                    "Pacing kesimi payoff ile çakıştığı için "
                    "otomatik kesim yerine REVIEW'e çevrildi."
                )

                item[
                    "blocked_by_payoff"
                ] = True

                review_ranges.append(
                    item
                )

                continue

            automatic_ranges.append(
                item
            )

        elif action == "review":

            review_ranges.append(
                item
            )

    (
        automatic_ranges,
        review_ranges,
        _,
    ) = apply_automatic_cut_budget(
        automatic_ranges=automatic_ranges,
        review_ranges=review_ranges,
        clip_duration=clip_duration,
    )

    return (
        automatic_ranges,
        review_ranges,
    )


def _is_causal_story_handle(item: dict[str, Any]) -> bool:
    reason = str(item.get("reason", "")).casefold()
    return any(marker in reason for marker in STORY_INTEGRITY_REASON_MARKERS)


def _overlap_duration(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def enforce_post_pacing_story_integrity(
    cut_ranges: list[dict[str, Any]],
    review_ranges: list[dict[str, Any]],
    protected_ranges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Rollback only pacing cuts that would break a causal story bridge.

    Protected ranges guarantee that the important beats themselves survive.
    This guard checks the gaps between consecutive causal handles after pacing
    has proposed its cuts. Small dead-air trims remain allowed; large jumps or
    too much aggregate removal inside a short causal bridge are downgraded to
    REVIEW so the main video still reads as cause -> escalation -> payoff ->
    reaction. No model call and no extra render is involved.
    """
    if not cut_ranges or not protected_ranges:
        return cut_ranges, review_ranges, {
            "triggered": False,
            "blocked_cut_count": 0,
            "bridge_count": 0,
        }

    handles = [dict(item) for item in protected_ranges if _is_causal_story_handle(item)]
    handles.sort(key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))

    # Merge overlapping/touching causal handles; otherwise the same beat can
    # create an artificial zero-length bridge.
    merged_handles: list[dict[str, Any]] = []
    for item in handles:
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        if end <= start:
            continue
        if merged_handles and start <= float(merged_handles[-1]["end"]) + 0.08:
            merged_handles[-1]["end"] = max(float(merged_handles[-1]["end"]), end)
            merged_handles[-1]["reason"] = (
                str(merged_handles[-1].get("reason", "")) + " | " + str(item.get("reason", ""))
            ).strip(" |")
        else:
            merged_handles.append({"start": start, "end": end, "reason": str(item.get("reason", ""))})

    if len(merged_handles) < 2:
        return cut_ranges, review_ranges, {
            "triggered": False,
            "blocked_cut_count": 0,
            "bridge_count": 0,
        }

    blocked_ids: set[int] = set()
    bridge_reports: list[dict[str, Any]] = []

    for left, right in zip(merged_handles, merged_handles[1:]):
        bridge_start = float(left["end"])
        bridge_end = float(right["start"])
        bridge_duration = max(0.0, bridge_end - bridge_start)
        if bridge_duration <= 0.08:
            continue

        candidates: list[tuple[int, dict[str, Any], float]] = []
        for index, cut in enumerate(cut_ranges):
            overlap = _overlap_duration(
                float(cut.get("start", 0.0)),
                float(cut.get("end", 0.0)),
                bridge_start,
                bridge_end,
            )
            if overlap > 0.001:
                candidates.append((index, cut, overlap))

        if not candidates:
            continue

        if bridge_duration <= STORY_INTEGRITY_SHORT_BRIDGE:
            max_single = STORY_INTEGRITY_SHORT_MAX_SINGLE_CUT
            max_ratio = STORY_INTEGRITY_SHORT_MAX_REMOVAL_RATIO
        elif bridge_duration <= STORY_INTEGRITY_MEDIUM_BRIDGE:
            max_single = STORY_INTEGRITY_MEDIUM_MAX_SINGLE_CUT
            max_ratio = STORY_INTEGRITY_MEDIUM_MAX_REMOVAL_RATIO
        else:
            max_single = STORY_INTEGRITY_LONG_MAX_SINGLE_CUT
            max_ratio = STORY_INTEGRITY_LONG_MAX_REMOVAL_RATIO

        removal_budget = bridge_duration * max_ratio
        # Preserve the least disruptive, highest-priority trims first. Anything
        # that creates a jump larger than max_single is never accepted inside
        # the causal bridge.
        ordered = sorted(
            candidates,
            key=lambda row: (
                -int(row[1].get("priority", 1)),
                row[2],
                float(row[1].get("start", 0.0)),
            ),
        )
        kept_removal = 0.0
        bridge_blocked = 0
        for index, cut, overlap in ordered:
            if index in blocked_ids:
                continue
            if overlap > max_single + 1e-9 or kept_removal + overlap > removal_budget + 1e-9:
                blocked_ids.add(index)
                bridge_blocked += 1
            else:
                kept_removal += overlap

        bridge_reports.append({
            "start": round_time(bridge_start),
            "end": round_time(bridge_end),
            "duration": round_time(bridge_duration),
            "allowed_removal": round_time(removal_budget),
            "kept_removal": round_time(kept_removal),
            "blocked_cut_count": bridge_blocked,
        })

    if not blocked_ids:
        return cut_ranges, review_ranges, {
            "triggered": False,
            "blocked_cut_count": 0,
            "bridge_count": len(bridge_reports),
            "bridges": bridge_reports,
        }

    safe_cuts: list[dict[str, Any]] = []
    updated_review = list(review_ranges)
    for index, cut in enumerate(cut_ranges):
        if index not in blocked_ids:
            safe_cuts.append(cut)
            continue
        review_item = dict(cut)
        review_item["blocked_by_story_integrity"] = True
        review_item["reason"] = (
            "V19 post-pacing story-integrity guard: bu cut causal beat'ler arasındaki "
            "okunabilir akışı fazla kısalttığı için otomatik kesimden çıkarıldı."
        )
        updated_review.append(review_item)

    safe_cuts.sort(key=lambda item: float(item.get("start", 0.0)))
    updated_review = merge_ranges(updated_review)
    return safe_cuts, updated_review, {
        "triggered": True,
        "blocked_cut_count": len(blocked_ids),
        "bridge_count": len(bridge_reports),
        "bridges": bridge_reports,
    }


def time_is_removed(
    source_time: float,
    cut_ranges: list[dict[str, Any]],
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


def map_source_to_edited_time(
    source_time: float,
    cut_ranges: list[dict[str, Any]],
) -> float:

    source_time = max(
        0.0,
        float(
            source_time
        ),
    )

    removed_before = 0.0

    for cut in cut_ranges:

        start = float(
            cut["start"]
        )

        end = float(
            cut["end"]
        )

        duration = (
            end
            - start
        )

        if source_time >= end:

            removed_before += (
                duration
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


def build_cut_events(
    cut_ranges: list[dict[str, Any]],
    clip_start_absolute: float,
) -> list[dict[str, Any]]:

    events = []

    removed_before = 0.0

    for cut in cut_ranges:

        source_start = float(
            cut["start"]
        )

        source_end = float(
            cut["end"]
        )

        duration = (
            source_end
            - source_start
        )

        splice_time = (
            source_start
            - removed_before
        )

        events.append(
            {
                "type": "cut",

                "source_start": round_time(
                    source_start
                ),

                "source_end": round_time(
                    source_end
                ),

                "absolute_start": round_time(
                    clip_start_absolute
                    + source_start
                ),

                "absolute_end": round_time(
                    clip_start_absolute
                    + source_end
                ),

                "edited_time": round_time(
                    splice_time
                ),

                "remove_duration": round_time(
                    duration
                ),

                "priority": int(
                    cut.get(
                        "priority",
                        1,
                    )
                ),

                "reason": str(
                    cut.get(
                        "reason",
                        "",
                    )
                ),
            }
        )

        removed_before += (
            duration
        )

    return events


def build_review_events(
    review_ranges: list[dict[str, Any]],
    cut_ranges: list[dict[str, Any]],
    clip_start_absolute: float,
) -> list[dict[str, Any]]:

    events = []

    for review in review_ranges:

        source_start = float(
            review["start"]
        )

        source_end = float(
            review["end"]
        )

        events.append(
            {
                "type": "review_cut",

                "source_start": round_time(
                    source_start
                ),

                "source_end": round_time(
                    source_end
                ),

                "absolute_start": round_time(
                    clip_start_absolute
                    + source_start
                ),

                "absolute_end": round_time(
                    clip_start_absolute
                    + source_end
                ),

                "edited_start": map_source_to_edited_time(
                    source_start,
                    cut_ranges,
                ),

                "edited_end": map_source_to_edited_time(
                    source_end,
                    cut_ranges,
                ),

                "possible_saving": round_time(
                    source_end
                    - source_start
                ),

                "priority": int(
                    review.get(
                        "priority",
                        1,
                    )
                ),

                "reason": str(
                    review.get(
                        "reason",
                        "",
                    )
                ),
            }
        )

    return events


# ============================================================
# CAPTION EVENTS
# ============================================================

def build_highlight_word_set(
    caption_highlights: list[str],
) -> set[str]:

    result: set[str] = set()

    for phrase in caption_highlights:

        for raw_word in str(
            phrase
        ).split():

            if not is_meaningful_word(
                raw_word
            ):
                continue

            normalized = normalize_word(
                raw_word
            )

            if normalized:
                result.add(
                    normalized
                )

    return result


def build_caption_events(
    clip_words: list[dict[str, Any]],
    caption_highlights: list[str],
    cut_ranges: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    highlight_words = (
        build_highlight_word_set(
            caption_highlights
        )
    )

    if not highlight_words:
        return []

    events: list[dict[str, Any]] = []

    occurrence_count: dict[str, int] = {}
    last_event_time = -999.0

    for word in clip_words:

        normalized = word[
            "normalized"
        ]

        if normalized not in highlight_words:
            continue

        source_start = float(
            word["source_start"]
        )

        source_end = float(
            word["source_end"]
        )

        if time_is_removed(
            source_start,
            cut_ranges,
        ):
            continue

        current_count = occurrence_count.get(
            normalized,
            0,
        )

        if (
            current_count
            >= MAX_OCCURRENCES_PER_HIGHLIGHT_WORD
        ):
            continue

        edited_time = map_source_to_edited_time(
            source_start,
            cut_ranges,
        )

        if (
            edited_time
            - last_event_time
            < MIN_CAPTION_EVENT_GAP
        ):
            continue

        edited_end = map_source_to_edited_time(
            source_end,
            cut_ranges,
        )

        events.append(
            {
                "type": "caption_emphasis",

                "word": word[
                    "word"
                ],

                "source_time": round_time(
                    source_start
                ),

                "absolute_time": round_time(
                    word["absolute_start"]
                ),

                "edited_time": edited_time,

                "edited_end": edited_end,

                "strength": 1.0,
            }
        )

        occurrence_count[
            normalized
        ] = (
            current_count
            + 1
        )

        last_event_time = (
            edited_time
        )

    return events


# ============================================================
# HOOK
# ============================================================

def build_hook_event(
    clip: dict[str, Any],
) -> dict[str, Any] | None:

    hook_text = str(
        clip.get(
            "hook_text",
            "",
        )
    ).strip()

    if not hook_text:
        return None

    return {
        "type": "hook_text",

        "source_time": 0.0,

        "edited_time": 0.0,

        "duration": HOOK_TEXT_DURATION,

        "hook_type": str(
            clip.get(
                "hook_type",
                "curiosity",
            )
        ),

        "text": hook_text,
    }


# ============================================================
# PAYOFF
# ============================================================

def build_payoff_event(
    clip: dict[str, Any],
    clip_start: float,
    cut_ranges: list[dict[str, Any]],
) -> dict[str, Any]:

    payoff_absolute_start = float(
        clip.get(
            "payoff_start",
            clip_start,
        )
    )

    payoff_absolute_end = float(
        clip.get(
            "payoff_end",
            clip.get(
                "end",
                payoff_absolute_start,
            ),
        )
    )

    source_start = max(
        0.0,
        payoff_absolute_start
        - clip_start,
    )

    source_end = max(
        source_start,
        payoff_absolute_end
        - clip_start,
    )

    return {
        "type": "payoff",

        "emotion": str(
            clip.get(
                "emotion",
                "",
            )
        ),

        "source_start": round_time(
            source_start
        ),

        "source_end": round_time(
            source_end
        ),

        "absolute_start": round_time(
            payoff_absolute_start
        ),

        "absolute_end": round_time(
            payoff_absolute_end
        ),

        "edited_start": map_source_to_edited_time(
            source_start,
            cut_ranges,
        ),

        "edited_end": map_source_to_edited_time(
            source_end,
            cut_ranges,
        ),
    }


# ============================================================
# SFX SUGGESTIONS
# ============================================================

def choose_hook_sfx(
    hook_type: str,
) -> str:

    hook_type = str(
        hook_type
    ).casefold()

    if hook_type in {
        "shock",
        "reaction",
        "conflict",
    }:
        return "impact"

    if hook_type == "payoff_teaser":
        return "reverse_whoosh"

    return "whoosh"


def choose_payoff_sfx(
    emotion: str,
) -> str:

    emotion = str(
        emotion
    ).casefold()

    if emotion in {
        "shock",
        "rage",
        "conflict",
        "surprise",
    }:
        return "impact"

    if emotion in {
        "funny",
        "awkward",
    }:
        return "pop"

    if emotion == "hype":
        return "bass_hit"

    return "impact"


def build_sfx_events(
    clip: dict[str, Any],
    payoff_event: dict[str, Any],
    caption_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    candidates: list[dict[str, Any]] = []

    candidates.append(
        {
            "type": "sfx_suggestion",

            "edited_time": 0.0,

            "source_time": 0.0,

            "sfx": choose_hook_sfx(
                clip.get(
                    "hook_type",
                    "curiosity",
                )
            ),

            "priority": 4,

            "reason": (
                "İlk saniyedeki hook/pattern interrupt için."
            ),
        }
    )

    candidates.append(
        {
            "type": "sfx_suggestion",

            "edited_time": float(
                payoff_event[
                    "edited_start"
                ]
            ),

            "source_time": float(
                payoff_event[
                    "source_start"
                ]
            ),

            "sfx": choose_payoff_sfx(
                clip.get(
                    "emotion",
                    "",
                )
            ),

            "priority": 5,

            "reason": (
                "Ana payoff/reaksiyon anını güçlendirmek için."
            ),
        }
    )

    if caption_events:

        strongest_caption = max(
            caption_events,
            key=lambda event: len(
                normalize_word(
                    event.get(
                        "word",
                        "",
                    )
                )
            ),
        )

        candidates.append(
            {
                "type": "sfx_suggestion",

                "edited_time": float(
                    strongest_caption[
                        "edited_time"
                    ]
                ),

                "source_time": float(
                    strongest_caption[
                        "source_time"
                    ]
                ),

                "sfx": "pop",

                "priority": 2,

                "reason": (
                    f"Önemli caption kelimesi: "
                    f"{strongest_caption['word']}"
                ),
            }
        )

    candidates.sort(
        key=lambda event: (
            -int(
                event[
                    "priority"
                ]
            ),
            float(
                event[
                    "edited_time"
                ]
            ),
        )
    )

    selected: list[dict[str, Any]] = []

    for candidate in candidates:

        time = float(
            candidate[
                "edited_time"
            ]
        )

        too_close = any(
            abs(
                time
                - float(
                    existing[
                        "edited_time"
                    ]
                )
            )
            < MIN_SFX_GAP

            for existing in selected
        )

        if too_close:
            continue

        selected.append(
            candidate
        )

        if (
            len(
                selected
            )
            >= MAX_SFX_PER_CLIP
        ):
            break

    selected.sort(
        key=lambda event: float(
            event[
                "edited_time"
            ]
        )
    )

    return selected


# ============================================================
# EVENT ORDER
# ============================================================

def get_event_sort_time(
    event: dict[str, Any],
) -> float:

    if "edited_time" in event:
        return float(
            event["edited_time"]
        )

    if "edited_start" in event:
        return float(
            event["edited_start"]
        )

    return 0.0


def event_priority(
    event_type: str,
) -> int:

    priorities = {
        "cut": 0,
        "hook_text": 1,
        "sfx_suggestion": 2,
        "caption_emphasis": 3,
        "review_cut": 4,
        "payoff": 5,
    }

    return priorities.get(
        event_type,
        99,
    )


def sort_events(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    return sorted(
        events,
        key=lambda event: (
            get_event_sort_time(
                event
            ),
            event_priority(
                str(
                    event.get(
                        "type",
                        "",
                    )
                )
            ),
        ),
    )


# ============================================================
# SINGLE CLIP TIMELINE
# ============================================================

def build_clip_timeline(
    clip: dict[str, Any],
    transcript: dict[str, Any],
    pacing_clip: dict[str, Any] | None,
    clip_index: int,
) -> dict[str, Any]:

    clip_start = float(
        clip["start"]
    )

    clip_end = float(
        clip["end"]
    )

    clip_duration = (
        clip_end
        - clip_start
    )

    if clip_duration <= 0:
        raise RuntimeError(
            f"{clip_index}. klibin süresi geçersiz."
        )

    payoff_source_start = max(
        0.0,
        float(
            clip.get(
                "payoff_start",
                clip_start,
            )
        )
        - clip_start,
    )

    payoff_source_end = min(
        clip_duration,
        max(
            payoff_source_start,
            float(
                clip.get(
                    "payoff_end",
                    clip_end,
                )
            )
            - clip_start,
        ),
    )

    clip_words = get_clip_words(
        transcript=transcript,
        clip_start=clip_start,
        clip_end=clip_end,
    )

    protected_ranges = (
        normalize_protected_ranges(
            clip=clip,
            clip_start=clip_start,
            clip_duration=clip_duration,
        )
    )

    cut_ranges, review_ranges = (
        build_cut_ranges(
            pacing_clip=pacing_clip,
            clip_duration=clip_duration,
            payoff_start=payoff_source_start,
            payoff_end=payoff_source_end,
            protected_ranges=protected_ranges,
        )
    )

    # V19: check the already-generated pacing cuts as a complete edit. The
    # protected beats themselves are already safe; this only rolls back cuts
    # that would make the bridges between cause/escalation/payoff/reaction read
    # like disconnected jumps.
    cut_ranges, review_ranges, story_integrity = enforce_post_pacing_story_integrity(
        cut_ranges=cut_ranges,
        review_ranges=review_ranges,
        protected_ranges=protected_ranges,
    )

    automatic_cut_budget = round_time(
        min(
            clip_duration
            * MAX_AUTOMATIC_CUT_RATIO,
            max(
                0.0,
                clip_duration
                - minimum_allowed_edited_duration(
                    clip_duration
                ),
            ),
            MAX_AUTOMATIC_CUT_SECONDS,
        )
    )

    minimum_edited_duration = (
        minimum_allowed_edited_duration(
            clip_duration
        )
    )

    removed_duration = sum(
        float(
            cut[
                "duration"
            ]
        )
        for cut in cut_ranges
    )

    edited_duration = max(
        0.0,
        clip_duration
        - removed_duration,
    )

    cut_events = build_cut_events(
        cut_ranges=cut_ranges,
        clip_start_absolute=clip_start,
    )

    review_events = build_review_events(
        review_ranges=review_ranges,
        cut_ranges=cut_ranges,
        clip_start_absolute=clip_start,
    )

    caption_events = build_caption_events(
        clip_words=clip_words,
        caption_highlights=clip.get(
            "caption_highlights",
            [],
        ),
        cut_ranges=cut_ranges,
    )

    hook_event = build_hook_event(
        clip
    )

    payoff_event = build_payoff_event(
        clip=clip,
        clip_start=clip_start,
        cut_ranges=cut_ranges,
    )

    sfx_events = build_sfx_events(
        clip=clip,
        payoff_event=payoff_event,
        caption_events=caption_events,
    )

    events: list[dict[str, Any]] = []

    events.extend(
        cut_events
    )

    events.extend(
        review_events
    )

    events.extend(
        caption_events
    )

    events.extend(
        sfx_events
    )

    events.append(
        payoff_event
    )

    if hook_event is not None:
        events.append(
            hook_event
        )

    events = sort_events(
        events
    )

    return {
        "clip_index": clip_index,

        "title": str(
            clip.get(
                "title",
                f"clip_{clip_index}",
            )
        ),

        "score": float(
            clip.get(
                "score",
                0,
            )
        ),

        "emotion": str(
            clip.get(
                "emotion",
                "",
            )
        ),

        "source": {
            "absolute_start": round_time(
                clip_start
            ),

            "absolute_end": round_time(
                clip_end
            ),

            "duration": round_time(
                clip_duration
            ),
        },

        "edited": {
            "estimated_duration": round_time(
                edited_duration
            ),

            "removed_duration": round_time(
                removed_duration
            ),

            "minimum_story_duration": (
                minimum_edited_duration
            ),

            "automatic_cut_budget": (
                automatic_cut_budget
            ),

            "retained_ratio": round(
                (
                    edited_duration
                    / clip_duration
                )
                if clip_duration > 0
                else 1.0,
                3,
            ),

            "automatic_cut_count": len(
                cut_ranges
            ),

            "review_cut_count": len(
                review_ranges
            ),

            "story_integrity_guard": story_integrity,
        },

        "hook": {
            "type": str(
                clip.get(
                    "hook_type",
                    "",
                )
            ),

            "text": str(
                clip.get(
                    "hook_text",
                    "",
                )
            ),
        },

        "payoff": payoff_event,

        "editorial": {
            "primary_moment_id": str(
                clip.get(
                    "primary_moment_id",
                    "",
                )
            ),
            "covered_moment_ids": (
                clip.get(
                    "covered_moment_ids",
                    [],
                )
                if isinstance(
                    clip.get(
                        "covered_moment_ids",
                        [],
                    ),
                    list,
                )
                else []
            ),
            "strongest_anchor": (
                clip.get(
                    "strongest_anchor",
                    {},
                )
                if isinstance(
                    clip.get(
                        "strongest_anchor",
                        {},
                    ),
                    dict,
                )
                else {}
            ),
            "anchor_moments": (
                clip.get(
                    "anchor_moments",
                    [],
                )
                if isinstance(
                    clip.get(
                        "anchor_moments",
                        [],
                    ),
                    list,
                )
                else []
            ),
            "coverage_reason": str(
                clip.get(
                    "coverage_reason",
                    "",
                )
            ),
            "terra_selection_reason": str(
                clip.get(
                    "terra_selection_reason",
                    "",
                )
            ),
        },

        "caption_highlights": clip.get(
            "caption_highlights",
            [],
        ),

        "protected_ranges": (
            protected_ranges
        ),

        "cut_ranges": cut_ranges,

        "review_ranges": review_ranges,

        "events": events,
    }


# ============================================================
# FULL TIMELINE
# ============================================================

def build_timelines(
    analysis_path: str | Path,
    transcript_path: str | Path,
    pacing_path: str | Path | None = None,
) -> dict[str, Any]:

    analysis_path = Path(
        analysis_path
    ).resolve()

    transcript_path = Path(
        transcript_path
    ).resolve()

    if pacing_path is None:

        pacing_path = guess_pacing_path(
            analysis_path
        )

    pacing_path = Path(
        pacing_path
    ).resolve()

    analysis = load_json(
        analysis_path
    )

    transcript = load_json(
        transcript_path
    )

    if pacing_path.exists():

        pacing_data = load_json(
            pacing_path
        )

    else:

        print()
        print(
            "⚠️ Pacing JSON bulunamadı."
        )

        print(
            "Timeline cutsız oluşturulacak."
        )

        pacing_data = {
            "clips": []
        }

    clips = analysis.get(
        "clips",
        [],
    )

    if not clips:
        raise RuntimeError(
            "Clip analysis dosyasında klip bulunamadı."
        )

    timelines = []

    for index, clip in enumerate(
        clips,
        start=1,
    ):

        pacing_clip = get_pacing_clip(
            pacing_data,
            index,
        )

        timeline = build_clip_timeline(
            clip=clip,
            transcript=transcript,
            pacing_clip=pacing_clip,
            clip_index=index,
        )

        timelines.append(
            timeline
        )

    return {
        "version": TIMELINE_VERSION,

        "revision": TIMELINE_REVISION,

        "mode": "premiere_edit_recipe",

        "source": transcript.get(
            "source",
            {},
        ),

        "clip_count": len(
            timelines
        ),

        "inputs": {
            "analysis": str(
                analysis_path
            ),

            "transcript": str(
                transcript_path
            ),

            "pacing": (
                str(
                    pacing_path
                )
                if pacing_path.exists()
                else None
            ),
        },

        "timelines": timelines,
    }


# ============================================================
# SAVE
# ============================================================

def save_timelines(
    analysis_path: str | Path,
    timeline_data: dict[str, Any],
) -> Path:

    analysis_path = Path(
        analysis_path
    ).resolve()

    TIMELINE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_name = (
        analysis_path.stem
        .replace(
            "_clips",
            "",
        )
    )

    output_path = (
        TIMELINE_OUTPUT_DIR
        / f"{base_name}_timeline_v3.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            timeline_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


# ============================================================
# SUMMARY
# ============================================================

def print_timeline_summary(
    timeline_data: dict[str, Any],
) -> None:

    for timeline in timeline_data.get(
        "timelines",
        [],
    ):

        print()
        print(
            "=" * 62
        )

        print(
            f"🎬 KLİP {timeline['clip_index']}"
        )

        print(
            f"📛 {timeline['title']}"
        )

        print(
            f"⭐ {timeline['score']}/10"
        )

        print(
            f"⏱️ Kaynak: "
            f"{timeline['source']['duration']:.2f}s"
        )

        print(
            f"⚡ Tahmini edit: "
            f"{timeline['edited']['estimated_duration']:.2f}s"
        )

        print(
            f"✂️ Otomatik kesim: "
            f"{timeline['edited']['automatic_cut_count']}"
        )

        print(
            f"⚠️ Review: "
            f"{timeline['edited']['review_cut_count']}"
        )

        print(
            f"🧲 Hook: "
            f"{timeline['hook']['text']}"
        )

        print(
            f"💥 Payoff: "
            f"{timeline['payoff']['edited_start']:.2f}"
            f" → "
            f"{timeline['payoff']['edited_end']:.2f}"
        )

        sfx_count = sum(
            1
            for event in timeline[
                "events"
            ]
            if event.get(
                "type"
            ) == "sfx_suggestion"
        )

        caption_count = sum(
            1
            for event in timeline[
                "events"
            ]
            if event.get(
                "type"
            ) == "caption_emphasis"
        )

        print(
            f"🔊 SFX marker: "
            f"{sfx_count}"
        )

        print(
            f"📝 Caption vurgusu: "
            f"{caption_count}"
        )

        print()

        for event in timeline[
            "events"
        ]:

            event_type = str(
                event.get(
                    "type",
                    "",
                )
            )

            if event_type == "cut":

                print(
                    f"   ✂️ CUT "
                    f"{event['source_start']:.2f}"
                    f" → "
                    f"{event['source_end']:.2f}"
                    f"  (-{event['remove_duration']:.2f}s)"
                )

            elif event_type == "review_cut":

                print(
                    f"   ⚠️ REVIEW "
                    f"{event['source_start']:.2f}"
                    f" → "
                    f"{event['source_end']:.2f}"
                )

            elif event_type == "hook_text":

                print(
                    f"   🧲 0.00s "
                    f"HOOK [{event['text']}]"
                )

            elif event_type == "caption_emphasis":

                print(
                    f"   📝 "
                    f"{event['edited_time']:.2f}s "
                    f"[{event['word']}]"
                )

            elif event_type == "sfx_suggestion":

                print(
                    f"   🔊 "
                    f"{event['edited_time']:.2f}s "
                    f"{event['sfx']}"
                )

            elif event_type == "payoff":

                print(
                    f"   💥 "
                    f"{event['edited_start']:.2f}s "
                    f"PAYOFF"
                )


# ============================================================
# ENTRY
# ============================================================

def create_edit_timeline(
    analysis_path: str | Path,
    transcript_path: str | Path,
    pacing_path: str | Path | None = None,
) -> dict[str, Any]:

    print()
    print(
        "=" * 65
    )

    print(
        "🎞️ MIMIR PREMIERE TIMELINE V3"
    )

    print(
        "=" * 65
    )

    timeline_data = build_timelines(
        analysis_path=analysis_path,
        transcript_path=transcript_path,
        pacing_path=pacing_path,
    )

    output_path = save_timelines(
        analysis_path=analysis_path,
        timeline_data=timeline_data,
    )

    print_timeline_summary(
        timeline_data
    )

    print()
    print(
        "=" * 65
    )

    print(
        "✅ TIMELINE V3 HAZIR"
    )

    print(
        f"📂 {output_path}"
    )

    print(
        "=" * 65
    )

    return timeline_data


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    analysis = input(
        "Clip analysis JSON yolunu gir: "
    ).strip().strip('"')

    transcript = input(
        "Transcript JSON yolunu gir: "
    ).strip().strip('"')

    pacing = input(
        "Pacing JSON yolu "
        "(boş bırakırsan otomatik bulur): "
    ).strip().strip('"')

    try:

        create_edit_timeline(
            analysis_path=analysis,
            transcript_path=transcript,
            pacing_path=(
                pacing
                if pacing
                else None
            ),
        )

    except Exception as error:

        print()
        print(
            f"❌ TIMELINE V3 HATASI:\n{error}"
        )