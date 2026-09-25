from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PACING_OUTPUT_DIR = (
    PROJECT_ROOT
    / "vod_output"
    / "pacing"
)


# ============================================================
# CONFIG
# ============================================================

# Pacing V3 is balanced: obvious dead air is compressed aggressively, while
# Terra-protected story beats/reaction timing are preserved downstream.
PACING_REVISION = 3

# Below this gap we do not even suggest a change.
MIN_GAP = 0.60

# Only truly long internal gaps become automatic cuts. Medium pauses are
# REVIEW-only so comedy/tension/reaction timing survives.
AUTO_CUT_GAP = 1.05
AUTO_CUT_MIN_REMOVABLE = 0.52

# Leave more natural breathing room around spoken words.
KEEP_AFTER_PREVIOUS_WORD = 0.18
KEEP_BEFORE_NEXT_WORD = 0.18

# Suggestions smaller than this are not worth touching.
MIN_REMOVABLE_DURATION = 0.22

# Payoff çevresinde sessizlik bazen komedi / gerilim için değerlidir.
PAYOFF_PROTECTION_BEFORE = 0.40
PAYOFF_PROTECTION_AFTER = 0.50

# Edge trimming is also conservative.
LEADING_SILENCE_THRESHOLD = 0.45
TRAILING_SILENCE_THRESHOLD = 0.60
AUTO_LEADING_TRIM_THRESHOLD = 0.70
AUTO_TRAILING_TRIM_THRESHOLD = 0.90

LEADING_PADDING = 0.08
TRAILING_PADDING = 0.18


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
        encoding="utf-8"
    ) as file:

        return json.load(file)


def round_time(
    value: float,
) -> float:

    return round(
        float(value),
        3
    )


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
# CLIP WORDS
# ============================================================

def get_clip_words(
    transcript: dict[str, Any],
    clip_start: float,
    clip_end: float,
) -> list[dict[str, Any]]:

    words = transcript.get(
        "words",
        []
    )

    result: list[dict[str, Any]] = []

    for item in words:

        absolute_start = float(
            item.get(
                "start",
                0
            )
        )

        absolute_end = float(
            item.get(
                "end",
                0
            )
        )

        word = str(
            item.get(
                "word",
                ""
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
            absolute_start - clip_start
        )

        relative_end = min(
            clip_end - clip_start,
            absolute_end - clip_start
        )

        if relative_end <= relative_start:
            relative_end = (
                relative_start
                + 0.05
            )

        result.append(
            {
                "word": word,

                "relative_start": round_time(
                    relative_start
                ),

                "relative_end": round_time(
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
# GAP CLASSIFICATION
# ============================================================

def classify_gap(
    gap_duration: float,
) -> str:

    if gap_duration >= 1.60:
        return "very_long"

    if gap_duration >= 1.00:
        return "long"

    if gap_duration >= 0.70:
        return "medium"

    return "short"


def calculate_priority(
    removable_duration: float,
) -> int:
    """
    Premiere edit önceliği.

    5 = kesin bak
    1 = çok düşük
    """

    if removable_duration >= 1.50:
        return 5

    if removable_duration >= 1.00:
        return 4

    if removable_duration >= 0.65:
        return 3

    if removable_duration >= 0.35:
        return 2

    return 1


# ============================================================
# PAYOFF PROTECTION
# ============================================================

def is_near_payoff(
    silence_start: float,
    silence_end: float,
    payoff_relative_start: float,
    payoff_relative_end: float,
) -> bool:

    protected_start = max(
        0.0,
        payoff_relative_start
        - PAYOFF_PROTECTION_BEFORE
    )

    protected_end = (
        payoff_relative_end
        + PAYOFF_PROTECTION_AFTER
    )

    return ranges_overlap(
        silence_start,
        silence_end,
        protected_start,
        protected_end,
    )


# ============================================================
# INTERNAL SILENCE DETECTION
# ============================================================

def detect_internal_silences(
    words: list[dict[str, Any]],
    clip_start: float,
    payoff_relative_start: float,
    payoff_relative_end: float,
) -> list[dict[str, Any]]:

    if len(words) < 2:
        return []

    suggestions: list[dict[str, Any]] = []

    for index in range(
        len(words) - 1
    ):

        previous_word = words[index]
        next_word = words[index + 1]

        silence_start = float(
            previous_word["relative_end"]
        )

        silence_end = float(
            next_word["relative_start"]
        )

        raw_gap = (
            silence_end
            - silence_start
        )

        if raw_gap < MIN_GAP:
            continue

        cut_start = (
            silence_start
            + KEEP_AFTER_PREVIOUS_WORD
        )

        cut_end = (
            silence_end
            - KEEP_BEFORE_NEXT_WORD
        )

        removable_duration = (
            cut_end
            - cut_start
        )

        if removable_duration < MIN_REMOVABLE_DURATION:
            continue

        near_payoff = is_near_payoff(
            silence_start=silence_start,
            silence_end=silence_end,
            payoff_relative_start=payoff_relative_start,
            payoff_relative_end=payoff_relative_end,
        )

        if (
            near_payoff
            and raw_gap < 1.35
        ):
            action = "review"

            reason = (
                "Pause payoff çevresinde ve kısa/orta uzunlukta. "
                "Timing değeri olabilir; REVIEW."
            )

        elif (
            raw_gap >= AUTO_CUT_GAP
            and removable_duration >= AUTO_CUT_MIN_REMOVABLE
        ):
            action = "cut"

            reason = (
                "Belirgin dead air. Konuşmanın iki yanında doğal nefes "
                "bırakılarak orta bölüm otomatik sıkıştırılabilir."
            )

        else:
            action = "review"

            reason = (
                "Kısa/orta pause. Otomatik kesim yerine REVIEW."
            )

        absolute_cut_start = (
            clip_start
            + cut_start
        )

        absolute_cut_end = (
            clip_start
            + cut_end
        )

        suggestions.append(
            {
                "type": "dead_air",

                "action": action,

                "priority": calculate_priority(
                    removable_duration
                ),

                "gap_class": classify_gap(
                    raw_gap
                ),

                "previous_word": previous_word[
                    "word"
                ],

                "next_word": next_word[
                    "word"
                ],

                "raw_gap_duration": round_time(
                    raw_gap
                ),

                "remove_duration": round_time(
                    removable_duration
                ),

                "relative": {
                    "start": round_time(
                        cut_start
                    ),

                    "end": round_time(
                        cut_end
                    ),
                },

                "absolute": {
                    "start": round_time(
                        absolute_cut_start
                    ),

                    "end": round_time(
                        absolute_cut_end
                    ),
                },

                "near_payoff": near_payoff,

                "reason": reason,
            }
        )

    return suggestions


# ============================================================
# LEADING / TRAILING SILENCE
# ============================================================

def detect_edge_trims(
    words: list[dict[str, Any]],
    clip_start: float,
    clip_duration: float,
) -> list[dict[str, Any]]:

    if not words:
        return []

    suggestions = []

    first_word_start = float(
        words[0]["relative_start"]
    )

    # --------------------------------------------------------
    # LEADING
    # --------------------------------------------------------

    if (
        first_word_start
        >= LEADING_SILENCE_THRESHOLD
    ):

        trim_end = max(
            0.0,
            first_word_start
            - LEADING_PADDING
        )

        if trim_end >= MIN_REMOVABLE_DURATION:

            leading_action = (
                "trim"
                if first_word_start >= AUTO_LEADING_TRIM_THRESHOLD
                else "review"
            )

            suggestions.append(
                {
                    "type": "leading_dead_air",

                    "action": leading_action,

                    "priority": 4,

                    "remove_duration": round_time(
                        trim_end
                    ),

                    "relative": {
                        "start": 0.0,
                        "end": round_time(
                            trim_end
                        ),
                    },

                    "absolute": {
                        "start": round_time(
                            clip_start
                        ),

                        "end": round_time(
                            clip_start
                            + trim_end
                        ),
                    },

                    "reason": (
                        "Klip başındaki belirgin boşluk. İlk söze kısa doğal handle bırakılır."
                    ),
                }
            )

    # --------------------------------------------------------
    # TRAILING
    # --------------------------------------------------------

    last_word_end = float(
        words[-1]["relative_end"]
    )

    trailing_gap = (
        clip_duration
        - last_word_end
    )

    if (
        trailing_gap
        >= TRAILING_SILENCE_THRESHOLD
    ):

        trim_start = min(
            clip_duration,
            last_word_end
            + TRAILING_PADDING
        )

        removable = (
            clip_duration
            - trim_start
        )

        if removable >= MIN_REMOVABLE_DURATION:

            trailing_action = (
                "trim"
                if trailing_gap >= AUTO_TRAILING_TRIM_THRESHOLD
                else "review"
            )

            suggestions.append(
                {
                    "type": "trailing_dead_air",

                    "action": trailing_action,

                    "priority": 4,

                    "remove_duration": round_time(
                        removable
                    ),

                    "relative": {
                        "start": round_time(
                            trim_start
                        ),

                        "end": round_time(
                            clip_duration
                        ),
                    },

                    "absolute": {
                        "start": round_time(
                            clip_start
                            + trim_start
                        ),

                        "end": round_time(
                            clip_start
                            + clip_duration
                        ),
                    },

                    "reason": (
                        "Klip sonundaki belirgin boşluk. Son söze kısa doğal handle bırakılır."
                    ),
                }
            )

    return suggestions


# ============================================================
# SINGLE CLIP
# ============================================================

def analyze_clip_pacing(
    clip: dict[str, Any],
    transcript: dict[str, Any],
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

    payoff_start = float(
        clip.get(
            "payoff_start",
            clip_start
        )
    )

    payoff_end = float(
        clip.get(
            "payoff_end",
            clip_end
        )
    )

    payoff_relative_start = max(
        0.0,
        payoff_start
        - clip_start
    )

    payoff_relative_end = min(
        clip_duration,
        payoff_end
        - clip_start
    )

    words = get_clip_words(
        transcript=transcript,
        clip_start=clip_start,
        clip_end=clip_end,
    )

    internal_silences = (
        detect_internal_silences(
            words=words,
            clip_start=clip_start,
            payoff_relative_start=payoff_relative_start,
            payoff_relative_end=payoff_relative_end,
        )
    )

    edge_trims = detect_edge_trims(
        words=words,
        clip_start=clip_start,
        clip_duration=clip_duration,
    )

    suggestions = (
        edge_trims
        + internal_silences
    )

    suggestions.sort(
        key=lambda item: (
            float(
                item["relative"]["start"]
            ),
            -int(
                item["priority"]
            ),
        )
    )

    cut_savings = sum(
        float(
            item.get(
                "remove_duration",
                0
            )
        )
        for item in suggestions
        if item.get("action") in {
            "cut",
            "trim"
        }
    )

    review_savings = sum(
        float(
            item.get(
                "remove_duration",
                0
            )
        )
        for item in suggestions
        if item.get("action") == "review"
    )

    estimated_duration = max(
        0.0,
        clip_duration
        - cut_savings
    )

    return {
        "clip_index": clip_index,

        "title": str(
            clip.get(
                "title",
                f"clip_{clip_index}"
            )
        ),

        "score": float(
            clip.get(
                "score",
                0
            )
        ),

        "source": {
            "start": round_time(
                clip_start
            ),

            "end": round_time(
                clip_end
            ),

            "duration": round_time(
                clip_duration
            ),
        },

        "payoff": {
            "start": round_time(
                payoff_relative_start
            ),

            "end": round_time(
                payoff_relative_end
            ),
        },

        "word_count": len(
            words
        ),

        "suggestion_count": len(
            suggestions
        ),

        "automatic_time_saving": round_time(
            cut_savings
        ),

        "possible_extra_saving": round_time(
            review_savings
        ),

        "estimated_duration_after_cuts": round_time(
            estimated_duration
        ),

        "suggestions": suggestions,
    }


# ============================================================
# FULL ANALYSIS
# ============================================================

def analyze_pacing(
    analysis_path: str | Path,
    transcript_path: str | Path,
) -> dict[str, Any]:

    analysis = load_json(
        analysis_path
    )

    transcript = load_json(
        transcript_path
    )

    clips = analysis.get(
        "clips",
        []
    )

    if not clips:
        raise RuntimeError(
            "Clip analysis dosyasında klip bulunamadı."
        )

    results = []

    for index, clip in enumerate(
        clips,
        start=1
    ):

        results.append(
            analyze_clip_pacing(
                clip=clip,
                transcript=transcript,
                clip_index=index,
            )
        )

    return {
        "version": 1,
        "revision": PACING_REVISION,
        "mode": "conservative_story_pacing",

        "source": transcript.get(
            "source",
            {}
        ),

        "clip_count": len(
            results
        ),

        "clips": results,
    }


# ============================================================
# SAVE
# ============================================================

def save_pacing_analysis(
    analysis_path: str | Path,
    pacing_data: dict[str, Any],
) -> Path:

    analysis_path = Path(
        analysis_path
    ).resolve()

    PACING_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    base_name = (
        analysis_path.stem
        .replace(
            "_clips",
            ""
        )
    )

    output_path = (
        PACING_OUTPUT_DIR
        / f"{base_name}_pacing.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            pacing_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_pacing_summary(
    pacing_data: dict[str, Any],
) -> None:

    for clip in pacing_data.get(
        "clips",
        []
    ):

        print()
        print(
            "=" * 60
        )

        print(
            f"🎬 KLİP {clip['clip_index']}"
        )

        print(
            f"📛 {clip['title']}"
        )

        print(
            f"⏱️ Orijinal: "
            f"{clip['source']['duration']:.2f} sn"
        )

        print(
            f"✂️ Kesilebilir: "
            f"{clip['automatic_time_saving']:.2f} sn"
        )

        print(
            f"⚠️ İncelenecek: "
            f"{clip['possible_extra_saving']:.2f} sn"
        )

        print(
            f"⚡ Tahmini yeni süre: "
            f"{clip['estimated_duration_after_cuts']:.2f} sn"
        )

        suggestions = clip.get(
            "suggestions",
            []
        )

        if not suggestions:

            print()
            print(
                "✅ Pacing zaten temiz."
            )

            continue

        print()

        for suggestion in suggestions:

            start = suggestion[
                "relative"
            ]["start"]

            end = suggestion[
                "relative"
            ]["end"]

            action = suggestion[
                "action"
            ].upper()

            duration = suggestion[
                "remove_duration"
            ]

            priority = suggestion[
                "priority"
            ]

            print(
                f"{action:7} "
                f"{start:6.2f} → {end:6.2f} "
                f"| -{duration:.2f}s "
                f"| P{priority}"
            )


# ============================================================
# ENTRY
# ============================================================

def create_pacing_analysis(
    analysis_path: str | Path,
    transcript_path: str | Path,
) -> dict[str, Any]:

    print()
    print(
        "=" * 65
    )

    print(
        "⚡ MIMIR PACING ANALYZER — CONSERVATIVE V2"
    )

    print(
        "=" * 65
    )

    pacing_data = analyze_pacing(
        analysis_path=analysis_path,
        transcript_path=transcript_path,
    )

    output_path = save_pacing_analysis(
        analysis_path=analysis_path,
        pacing_data=pacing_data,
    )

    print_pacing_summary(
        pacing_data
    )

    print()
    print(
        "=" * 65
    )

    print(
        "✅ PACING ANALİZİ HAZIR"
    )

    print(
        f"📂 {output_path}"
    )

    print(
        "=" * 65
    )

    return pacing_data


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

    try:

        create_pacing_analysis(
            analysis_path=analysis,
            transcript_path=transcript,
        )

    except Exception as error:

        print()
        print(
            f"❌ PACING HATASI:\n{error}"
        )