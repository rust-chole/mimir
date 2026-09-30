from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ai.openai_client import client


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VOD_OUTPUT_DIR = PROJECT_ROOT / "vod_output"

AUDIO_DIR = (
    VOD_OUTPUT_DIR
    / "audio"
)

TRANSCRIPT_DIR = (
    VOD_OUTPUT_DIR
    / "transcripts"
)

TEMP_DIR = (
    VOD_OUTPUT_DIR
    / "temp"
    / "transcription"
)


# ============================================================
# TRANSCRIPTION CONFIG
# ============================================================

ACCURATE_MODEL = os.getenv("MIMIR_VOD_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe").strip() or "gpt-4o-mini-transcribe"

# whisper-1 word timestamps are the whole-VOD scouting clock (and, isolated in
# ai.caption_stack.legacy_whisper, the final-caption migration fallback). Final
# caption wording and timing are owned by ai.caption_stack.
TIMING_MODEL = "whisper-1"

TRANSCRIPTION_LANGUAGE = os.getenv("CAPTION_LANGUAGE", os.getenv("MIMIR_TRANSCRIPTION_LANGUAGE", "en")).strip() or "en"
CAPTION_MASTER_SAMPLE_RATE = max(16000, int(os.getenv("MIMIR_CAPTION_MASTER_SAMPLE_RATE", "48000") or 48000))

CHUNK_SECONDS = 600.0

AUDIO_SAMPLE_RATE = 16000
AUDIO_BITRATE = "64k"


# ============================================================
# ALIGNMENT V3.2 CONFIG
# ============================================================

# Caption için 40ms gibi saçma kelimeler istemiyoruz.
MIN_WORD_DURATION = 0.08

# Tek bir kelimeyi 1 saniye boyunca açık tutma.
MAX_WORD_DURATION = 0.70

# Kelime merkezlerinin üst üste binmesini engeller.
MIN_CENTER_GAP = 0.04

# Alignment bunun altındaysa terminalde uyar.
ALIGNMENT_WARNING_THRESHOLD = 0.78


# ============================================================
# NUMBER NORMALIZATION
# ============================================================

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


# ============================================================
# TRANSCRIPTION PROMPT
# ============================================================

TRANSCRIPTION_PROMPT = """
This is English gaming, livestream, streamer, YouTube or Twitch content.

Transcribe exactly what is spoken.

Do not summarize.
Do not rewrite grammar.
Do not censor profanity.
Do not remove repetitions.
Do not invent speech.

Pay special attention to gaming and livestream vocabulary.

Common vocabulary may include:
chat, Twitch, YouTube, Discord, stream, streamer, viewers,
bro, dude, chat, W, L, cooked, cap, no cap, no way,
game, gameplay, clip, donation, dono, sub, subscriber,
mods, moderator, IRL, rage, reaction, arena, ranked.

The word "chat" often refers to the livestream audience.
Do not change "chat" into the name "Chad" unless the audio
clearly indicates that the speaker is referring to a person named Chad.

Preserve:
- slang
- profanity
- repeated words
- brand names
- usernames
- game names
- streamer vocabulary
- unfinished sentences

Only transcribe speech that is actually audible.
""".strip()

# ============================================================
# GENERIC HELPERS
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


def load_response_dict(
    response: Any,
) -> dict[str, Any]:

    if isinstance(
        response,
        dict,
    ):
        return response

    if hasattr(
        response,
        "model_dump",
    ):
        return response.model_dump()

    if hasattr(
        response,
        "dict",
    ):
        return response.dict()

    return {}


def get_response_text(
    response: Any,
) -> str:

    text = getattr(
        response,
        "text",
        None,
    )

    if text:
        return str(
            text
        ).strip()

    data = load_response_dict(
        response
    )

    return str(
        data.get(
            "text",
            "",
        )
    ).strip()


# ============================================================
# WORD NORMALIZATION
# ============================================================

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


def canonical_word(
    text: str,
) -> str:
    """
    SequenceMatcher için biraz daha akıllı normalize eder.

    twelve ↔ 12
    """

    normalized = normalize_word(
        text
    )

    if normalized in NUMBER_WORDS:

        return NUMBER_WORDS[
            normalized
        ]

    return normalized


def join_word_text(
    words: list[dict[str, Any]],
) -> str:

    return " ".join(
        str(
            item.get(
                "word",
                "",
            )
        ).strip()

        for item in words

        if str(
            item.get(
                "word",
                "",
            )
        ).strip()
    ).strip()


# ============================================================
# VIDEO INFO
# ============================================================

def get_video_duration(
    video_path: str | Path,
) -> float:

    video_path = Path(
        video_path
    ).resolve()

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video bulunamadı:\n{video_path}"
        )

    command = [
        "ffprobe",
        "-v",
        "error",

        "-show_entries",
        "format=duration",

        "-of",
        "default=noprint_wrappers=1:nokey=1",

        str(video_path),
    ]

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )

    except FileNotFoundError as error:

        raise RuntimeError(
            "ffprobe bulunamadı."
        ) from error

    except subprocess.CalledProcessError as error:

        raise RuntimeError(
            "Video süresi okunamadı:\n"
            + error.stderr
        ) from error

    try:

        duration = float(
            result.stdout.strip()
        )

    except ValueError as error:

        raise RuntimeError(
            "ffprobe geçersiz duration döndürdü."
        ) from error

    if duration <= 0:

        raise RuntimeError(
            "Video süresi geçersiz."
        )

    return duration


# ============================================================
# AUDIO EXTRACTION
# ============================================================

def extract_audio(
    video_path: str | Path,
) -> Path:

    video_path = Path(
        video_path
    ).resolve()

    AUDIO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        AUDIO_DIR
        / f"{video_path.stem}.mp3"
    )

    command = [
        "ffmpeg",
        "-y",

        "-i",
        str(video_path),

        "-vn",

        "-ac",
        "1",

        "-ar",
        str(
            AUDIO_SAMPLE_RATE
        ),

        "-b:a",
        AUDIO_BITRATE,

        str(output_path),
    ]

    print()
    print(
        "🎵 Audio çıkarılıyor..."
    )

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    except FileNotFoundError as error:

        raise RuntimeError(
            "FFmpeg bulunamadı."
        ) from error

    if result.returncode != 0:

        raise RuntimeError(
            "FFmpeg audio extraction hatası:\n\n"
            + result.stderr
        )

    if not output_path.exists():

        raise RuntimeError(
            "Audio dosyası oluşmadı."
        )

    return output_path


# ============================================================
# AUDIO CHUNKS
# ============================================================

def create_audio_chunks(
    audio_path: str | Path,
    duration: float,
) -> list[dict[str, Any]]:

    audio_path = Path(
        audio_path
    ).resolve()

    if duration <= CHUNK_SECONDS:

        return [
            {
                "path": audio_path,
                "offset": 0.0,
                "duration": duration,
                "temporary": False,
            }
        ]

    chunk_directory = (
        TEMP_DIR
        / audio_path.stem
    )

    if chunk_directory.exists():

        shutil.rmtree(
            chunk_directory
        )

    chunk_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks: list[
        dict[str, Any]
    ] = []

    offset = 0.0
    index = 1

    while offset < duration:

        chunk_duration = min(
            CHUNK_SECONDS,
            duration - offset,
        )

        chunk_path = (
            chunk_directory
            / f"chunk_{index:03d}.mp3"
        )

        command = [
            "ffmpeg",
            "-y",

            "-ss",
            f"{offset:.3f}",

            "-i",
            str(audio_path),

            "-t",
            f"{chunk_duration:.3f}",

            "-ac",
            "1",

            "-ar",
            str(
                AUDIO_SAMPLE_RATE
            ),

            "-b:a",
            AUDIO_BITRATE,

            str(chunk_path),
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:

            raise RuntimeError(
                "Audio chunk oluşturulamadı:\n\n"
                + result.stderr
            )

        if not chunk_path.exists():

            raise RuntimeError(
                f"Chunk oluşmadı:\n{chunk_path}"
            )

        chunks.append(
            {
                "path": chunk_path,
                "offset": offset,
                "duration": chunk_duration,
                "temporary": True,
            }
        )

        offset += (
            chunk_duration
        )

        index += 1

    return chunks


# ============================================================
# GPT TRANSCRIPTION
# ============================================================

def transcribe_accurate_text(
    audio_path: str | Path,
) -> str:
    """Primary semantic pass with one retry; empty becomes recoverable."""

    audio_path = Path(
        audio_path
    ).resolve()

    attempts = [
        (
            TRANSCRIPTION_PROMPT,
            "prompted",
        ),
        (
            None,
            "plain retry",
        ),
    ]

    for attempt_index, (
        prompt,
        label,
    ) in enumerate(
        attempts,
        start=1,
    ):

        print(
            f"   🧠 {ACCURATE_MODEL}: "
            f"accurate transcript "
            f"({attempt_index}/{len(attempts)}, {label})..."
        )

        with audio_path.open(
            "rb"
        ) as audio_file:

            kwargs: dict[str, Any] = {
                "model": ACCURATE_MODEL,
                "file": audio_file,
                "language": TRANSCRIPTION_LANGUAGE,
            }

            if prompt:
                kwargs[
                    "prompt"
                ] = prompt

            response = client.audio.transcriptions.create(
                **kwargs
            )

        text = get_response_text(
            response
        )

        if text:
            return text

        print(
            "   ⚠️ Accurate model boş döndü."
        )

    print(
        "   ↪️ whisper-1 üzerinden transcript recovery denenecek."
    )

    return ""


# ============================================================
# WHISPER TIMING PASS
# ============================================================

def transcribe_word_timing(
    audio_path: str | Path,
    *,
    known_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Raw whisper-1 native word timestamps (no prompt, no calibration).

    The whole-VOD scouting clock, and the input of the legacy final-caption
    timing fallback (``ai.caption_stack.legacy_whisper``). It never owns caption
    wording. Names are never sent: ``known_names`` is accepted for call-site
    compatibility and deliberately ignored.
    """
    audio_path = Path(audio_path).resolve()

    print(f"   ⏱️ {TIMING_MODEL}: RAW native word timestamp clock (zero calibration)...")
    with audio_path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=TIMING_MODEL,
            file=audio_file,
            language=TRANSCRIPTION_LANGUAGE,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )

    data = load_response_dict(response)
    if not data.get("words"):
        print("   ⚠️ Whisper timing modeli word timestamp döndürmedi.")
    data["timing_model"] = TIMING_MODEL
    data["timing_mode"] = "raw_whisper_native_word_clock_v29"
    data["post_calibration"] = False
    return data


# ============================================================
# WHISPER TEXT FALLBACK
# ============================================================

def recover_text_from_timing(
    timing_data: dict[str, Any],
) -> str:
    """Recover text from whisper-1 verbose_json if gpt-transcribe is empty."""

    direct_text = str(
        timing_data.get(
            "text",
            "",
        )
    ).strip()

    if direct_text:
        return direct_text

    raw_words = timing_data.get(
        "words",
        [],
    )

    if not isinstance(
        raw_words,
        list,
    ):
        return ""

    words: list[str] = []

    for item in raw_words:

        if not isinstance(
            item,
            dict,
        ):
            continue

        word = str(
            item.get(
                "word",
                "",
            )
        ).strip()

        if word:
            words.append(
                word
            )

    return " ".join(
        words
    ).strip()


# ============================================================
# GPT TOKENIZATION
# ============================================================

def tokenize_accurate_text(
    text: str,
) -> list[str]:

    raw_tokens = re.findall(
        r"\S+",
        text,
    )

    result: list[str] = []

    for token in raw_tokens:

        token = token.strip()

        if not token:
            continue

        # Sadece punctuation ise önceki kelimeye ekle.
        if (
            not normalize_word(
                token
            )
            and result
        ):

            result[-1] += (
                token
            )

            continue

        if normalize_word(
            token
        ):

            result.append(
                token
            )

    return result


# ============================================================
# WHISPER WORD EXTRACTION
# ============================================================

def extract_timing_words(
    timing_data: dict[str, Any],
) -> list[dict[str, Any]]:

    result: list[
        dict[str, Any]
    ] = []

    for item in timing_data.get(
        "words",
        [],
    ):

        word = str(
            item.get(
                "word",
                "",
            )
        ).strip()

        if not word:
            continue

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

        normalized = canonical_word(
            word
        )

        if not normalized:
            continue

        if end < start:

            end = start

        result.append(
            {
                "word": word,

                "canonical": normalized,

                "start": start,

                "end": end,
            }
        )

    return result


# ============================================================
# DISTRIBUTE GPT WORDS OVER A TIME SPAN
# ============================================================

def distribute_words_over_span(
    words: list[str],
    start: float,
    end: float,
) -> list[dict[str, Any]]:

    if not words:
        return []

    start = float(
        start
    )

    end = float(
        end
    )

    if end <= start:

        end = (
            start
            + MIN_WORD_DURATION
            * len(
                words
            )
        )

    weights = [
        max(
            1,
            len(
                normalize_word(
                    word
                )
            ),
        )

        for word in words
    ]

    total_weight = sum(
        weights
    )

    span_duration = (
        end
        - start
    )

    result: list[
        dict[str, Any]
    ] = []

    cursor = start
    used_weight = 0

    for index, word in enumerate(
        words
    ):

        weight = weights[
            index
        ]

        if index == len(
            words
        ) - 1:

            word_end = end

        else:

            used_weight += (
                weight
            )

            fraction = (
                used_weight
                / total_weight
            )

            word_end = (
                start
                + span_duration
                * fraction
            )

        result.append(
            {
                "word": word,

                "start": cursor,

                "end": word_end,

                "alignment_source": "interpolated",
            }
        )

        cursor = word_end

    return result


# ============================================================
# PROVISIONAL ALIGNMENT
# ============================================================

def build_provisional_alignment(
    accurate_text: str,
    timing_data: dict[str, Any],
    chunk_duration: float,
) -> tuple[
    list[dict[str, Any]],
    float,
]:

    accurate_words = tokenize_accurate_text(
        accurate_text
    )

    timing_words = extract_timing_words(
        timing_data
    )

    if not accurate_words:

        raise RuntimeError(
            "GPT transcript'te kelime bulunamadı."
        )

    if not timing_words:

        raise RuntimeError(
            "Timing transcript'te kelime bulunamadı."
        )

    accurate_canonical = [
        canonical_word(
            word
        )

        for word in accurate_words
    ]

    timing_canonical = [
        item[
            "canonical"
        ]

        for item in timing_words
    ]

    matcher = SequenceMatcher(
        None,
        timing_canonical,
        accurate_canonical,
        autojunk=False,
    )

    alignment_ratio = (
        matcher.ratio()
    )

    aligned: list[
        dict[str, Any]
    ] = []

    for (
        tag,
        timing_start_index,
        timing_end_index,
        accurate_start_index,
        accurate_end_index,
    ) in matcher.get_opcodes():

        # ====================================================
        # EXACTLY MATCHED TOKENS
        # ====================================================

        if tag == "equal":

            count = (
                accurate_end_index
                - accurate_start_index
            )

            for offset in range(
                count
            ):

                timing_word = timing_words[
                    timing_start_index
                    + offset
                ]

                accurate_word = accurate_words[
                    accurate_start_index
                    + offset
                ]

                aligned.append(
                    {
                        "word": accurate_word,

                        "start": float(
                            timing_word[
                                "start"
                            ]
                        ),

                        "end": float(
                            timing_word[
                                "end"
                            ]
                        ),

                        "alignment_source": "anchor",
                    }
                )

            continue

        # ====================================================
        # WHISPER HAS EXTRA WORDS
        # ====================================================

        if tag == "delete":

            # GPT metni asıl kaynak.
            continue

        replacement_words = accurate_words[
            accurate_start_index:
            accurate_end_index
        ]

        if not replacement_words:
            continue

        # ====================================================
        # REPLACEMENT
        # ====================================================

        if (
            timing_end_index
            > timing_start_index
        ):

            span_start = float(
                timing_words[
                    timing_start_index
                ][
                    "start"
                ]
            )

            span_end = float(
                timing_words[
                    timing_end_index - 1
                ][
                    "end"
                ]
            )

        # ====================================================
        # GPT INSERTION
        # ====================================================

        else:

            if timing_start_index > 0:

                span_start = float(
                    timing_words[
                        timing_start_index - 1
                    ][
                        "end"
                    ]
                )

            else:

                span_start = 0.0

            if (
                timing_start_index
                < len(
                    timing_words
                )
            ):

                span_end = float(
                    timing_words[
                        timing_start_index
                    ][
                        "start"
                    ]
                )

            else:

                span_end = (
                    chunk_duration
                )

            minimum_needed = (
                MIN_WORD_DURATION
                * len(
                    replacement_words
                )
            )

            if (
                span_end
                - span_start
                < minimum_needed
            ):

                center = (
                    span_start
                    + span_end
                ) / 2

                span_start = max(
                    0.0,
                    center
                    - minimum_needed / 2,
                )

                span_end = min(
                    chunk_duration,
                    span_start
                    + minimum_needed,
                )

        aligned.extend(
            distribute_words_over_span(
                words=replacement_words,
                start=span_start,
                end=span_end,
            )
        )

    if not aligned:

        raise RuntimeError(
            "Transcript alignment başarısız."
        )

    return (
        aligned,
        alignment_ratio,
    )


# ============================================================
# V3.1 TIMESTAMP REPAIR ENGINE
# ============================================================

def repair_word_timeline(
    words: list[dict[str, Any]],
    chunk_duration: float,
) -> list[dict[str, Any]]:
    """
    Provisional timestamp'lerden güvenli caption timestamp'leri üretir.

    Ana prensip:

        Kelimelerin HAM start/end'ine körü körüne güvenmek yerine
        önce her kelimenin zaman merkezini buluyoruz.

        Sonra bu merkezlerden kelime sınırları oluşturuyoruz.

    Böylece:
        - overlap olmaz
        - 0.00 / 0.00 gibi başlangıçlar düzelir
        - 40ms kelimeler azaltılır
        - 0.9s boyunca kalan sıradan kelimeler azaltılır
    """

    if not words:
        return []

    chunk_duration = float(
        chunk_duration
    )

    word_count = len(
        words
    )

    # Çok kısa videolarda minimum center gap'i adaptif küçült.
    dynamic_center_gap = min(
        MIN_CENTER_GAP,

        (
            chunk_duration
            / max(
                word_count + 1,
                1,
            )
        )
        * 0.50,
    )

    dynamic_center_gap = max(
        0.005,
        dynamic_center_gap,
    )

    # --------------------------------------------------------
    # RAW CENTERS
    # --------------------------------------------------------

    raw_centers: list[
        float
    ] = []

    raw_durations: list[
        float
    ] = []

    for item in words:

        raw_start = clamp(
            float(
                item.get(
                    "start",
                    0,
                )
            ),
            0.0,
            chunk_duration,
        )

        raw_end = clamp(
            float(
                item.get(
                    "end",
                    raw_start,
                )
            ),
            0.0,
            chunk_duration,
        )

        if raw_end < raw_start:

            raw_end = (
                raw_start
            )

        raw_center = (
            raw_start
            + raw_end
        ) / 2.0

        raw_centers.append(
            raw_center
        )

        raw_durations.append(
            max(
                0.0,
                raw_end - raw_start,
            )
        )

    # --------------------------------------------------------
    # FORWARD CENTER REPAIR
    # --------------------------------------------------------

    centers: list[
        float
    ] = []

    for index, center in enumerate(
        raw_centers
    ):

        if index == 0:

            repaired = clamp(
                center,
                0.0,
                chunk_duration,
            )

        else:

            repaired = max(
                center,
                centers[-1]
                + dynamic_center_gap,
            )

        centers.append(
            repaired
        )

    # --------------------------------------------------------
    # MAKE SURE LAST CENTER FITS INSIDE VIDEO
    # --------------------------------------------------------

    maximum_last_center = max(
        0.0,
        chunk_duration
        - dynamic_center_gap / 2,
    )

    if (
        centers[-1]
        > maximum_last_center
    ):

        centers[-1] = (
            maximum_last_center
        )

        # Backward pass.
        for index in range(
            len(
                centers
            ) - 2,
            -1,
            -1,
        ):

            centers[
                index
            ] = min(
                centers[
                    index
                ],

                centers[
                    index + 1
                ]
                - dynamic_center_gap,
            )

    # If backwards pushed first center negative, shift all.
    if centers[0] < 0:

        shift = (
            -centers[0]
        )

        centers = [
            center
            + shift

            for center in centers
        ]

    # --------------------------------------------------------
    # CREATE NON-OVERLAPPING BOUNDARIES
    # --------------------------------------------------------

    repaired_words: list[
        dict[str, Any]
    ] = []

    for index, item in enumerate(
        words
    ):

        center = centers[
            index
        ]

        if index == 0:

            left_boundary = 0.0

        else:

            left_boundary = (
                centers[
                    index - 1
                ]
                + center
            ) / 2.0

        if index == len(
            words
        ) - 1:

            right_boundary = (
                chunk_duration
            )

        else:

            right_boundary = (
                center
                + centers[
                    index + 1
                ]
            ) / 2.0

        left_boundary = clamp(
            left_boundary,
            0.0,
            chunk_duration,
        )

        right_boundary = clamp(
            right_boundary,
            left_boundary,
            chunk_duration,
        )

        available_duration = (
            right_boundary
            - left_boundary
        )

        if available_duration <= 0:
            continue

        raw_duration = raw_durations[
            index
        ]

        target_duration = clamp(
            raw_duration,
            MIN_WORD_DURATION,
            MAX_WORD_DURATION,
        )

        # Bu kelimenin etrafındaki alan target süreden
        # küçükse mevcut alanın tamamını kullan.
        target_duration = min(
            target_duration,
            available_duration,
        )

        start = (
            center
            - target_duration / 2
        )

        start = clamp(
            start,
            left_boundary,
            max(
                left_boundary,
                right_boundary
                - target_duration,
            ),
        )

        end = (
            start
            + target_duration
        )

        end = min(
            end,
            right_boundary,
        )

        # Çok kısa kaldıysa mümkün olduğu kadar büyüt.
        if (
            end - start
            < MIN_WORD_DURATION
            and available_duration
            >= MIN_WORD_DURATION
        ):

            target_duration = (
                MIN_WORD_DURATION
            )

            start = clamp(
                center
                - target_duration / 2,

                left_boundary,

                right_boundary
                - target_duration,
            )

            end = (
                start
                + target_duration
            )

        if end <= start:
            continue

        repaired_words.append(
            {
                "word": str(
                    item[
                        "word"
                    ]
                ),

                "start": round_time(
                    start
                ),

                "end": round_time(
                    end
                ),

                "alignment_source": str(
                    item.get(
                        "alignment_source",
                        "unknown",
                    )
                ),
            }
        )

    # --------------------------------------------------------
    # FINAL ROUNDING SAFETY PASS
    # --------------------------------------------------------

    final_words: list[
        dict[str, Any]
    ] = []

    previous_end = 0.0

    for item in repaired_words:

        start = max(
            previous_end,
            float(
                item[
                    "start"
                ]
            ),
        )

        end = max(
            start,
            float(
                item[
                    "end"
                ]
            ),
        )

        end = min(
            end,
            chunk_duration,
        )

        if end <= start:
            continue

        final_words.append(
            {
                "word": item[
                    "word"
                ],

                "start": round_time(
                    start
                ),

                "end": round_time(
                    end
                ),

                "duration": round_time(
                    end - start
                ),

                "alignment_source": item[
                    "alignment_source"
                ],
            }
        )

        previous_end = (
            end
        )

    return final_words


# ============================================================
# ALIGN TRANSCRIPT
# ============================================================

def align_transcript_words(
    accurate_text: str,
    timing_data: dict[str, Any],
    chunk_duration: float,
) -> tuple[
    list[dict[str, Any]],
    float,
]:

    provisional_words, alignment_ratio = (
        build_provisional_alignment(
            accurate_text=accurate_text,
            timing_data=timing_data,
            chunk_duration=chunk_duration,
        )
    )

    repaired_words = (
        repair_word_timeline(
            words=provisional_words,
            chunk_duration=chunk_duration,
        )
    )

    if not repaired_words:

        raise RuntimeError(
            "V3.1 timestamp repair kelime üretemedi."
        )

    return (
        repaired_words,
        alignment_ratio,
    )


# ============================================================
# ADD CHUNK OFFSET
# ============================================================

def add_time_offset(
    words: list[dict[str, Any]],
    offset: float,
) -> list[dict[str, Any]]:

    result = []

    for item in words:

        start = (
            float(
                item[
                    "start"
                ]
            )
            + offset
        )

        end = (
            float(
                item[
                    "end"
                ]
            )
            + offset
        )

        result.append(
            {
                "word": item[
                    "word"
                ],

                "start": round_time(
                    start
                ),

                "end": round_time(
                    end
                ),

                "duration": round_time(
                    end - start
                ),

                "alignment_source": item.get(
                    "alignment_source",
                    "unknown",
                ),
            }
        )

    return result


# ============================================================
# SEGMENTS
# ============================================================

def should_end_segment(
    segment_words: list[dict[str, Any]],
    next_word: dict[str, Any] | None,
) -> bool:

    if not segment_words:
        return False

    first = segment_words[
        0
    ]

    last = segment_words[
        -1
    ]

    segment_duration = (
        float(
            last[
                "end"
            ]
        )
        - float(
            first[
                "start"
            ]
        )
    )

    last_text = str(
        last[
            "word"
        ]
    ).strip()

    if (
        len(
            segment_words
        )
        >= 35
    ):

        return True

    if (
        segment_duration
        >= 10.0
    ):

        return True

    if (
        len(
            segment_words
        )
        >= 3
        and last_text.endswith(
            (
                ".",
                "!",
                "?",
            )
        )
    ):

        return True

    if next_word is not None:

        gap = (
            float(
                next_word[
                    "start"
                ]
            )
            - float(
                last[
                    "end"
                ]
            )
        )

        if gap >= 0.70:

            return True

    return False


def build_segments(
    words: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    if not words:
        return []

    segments: list[
        dict[str, Any]
    ] = []

    current: list[
        dict[str, Any]
    ] = []

    for index, word in enumerate(
        words
    ):

        current.append(
            word
        )

        next_word = (
            words[
                index + 1
            ]

            if (
                index + 1
                < len(
                    words
                )
            )

            else None
        )

        if not should_end_segment(
            current,
            next_word,
        ):

            continue

        start = float(
            current[
                0
            ][
                "start"
            ]
        )

        end = float(
            current[
                -1
            ][
                "end"
            ]
        )

        segments.append(
            {
                "id": len(
                    segments
                ),

                "start": round_time(
                    start
                ),

                "end": round_time(
                    end
                ),

                "duration": round_time(
                    end - start
                ),

                "text": join_word_text(
                    current
                ),
            }
        )

        current = []

    if current:

        start = float(
            current[
                0
            ][
                "start"
            ]
        )

        end = float(
            current[
                -1
            ][
                "end"
            ]
        )

        segments.append(
            {
                "id": len(
                    segments
                ),

                "start": round_time(
                    start
                ),

                "end": round_time(
                    end
                ),

                "duration": round_time(
                    end - start
                ),

                "text": join_word_text(
                    current
                ),
            }
        )

    return segments


# ============================================================
# PROCESS ONE CHUNK
# ============================================================

def process_chunk(
    chunk: dict[str, Any],
    chunk_index: int,
    chunk_count: int,
) -> dict[str, Any]:

    path = Path(
        chunk[
            "path"
        ]
    ).resolve()

    offset = float(
        chunk[
            "offset"
        ]
    )

    duration = float(
        chunk[
            "duration"
        ]
    )

    print()
    print(
        f"📦 Chunk {chunk_index}/{chunk_count}"
    )

    print(
        f"   {offset:.2f}s"
        f" → "
        f"{offset + duration:.2f}s"
    )

    accurate_text = transcribe_accurate_text(
        path
    )

    timing_data = transcribe_word_timing(
        path
    )

    timing_words = extract_timing_words(
        timing_data
    )

    fallback_used = False
    text_source = ACCURATE_MODEL

    if not accurate_text:

        recovered_text = recover_text_from_timing(
            timing_data
        )

        if (
            recovered_text
            and timing_words
        ):

            accurate_text = recovered_text
            fallback_used = True
            text_source = TIMING_MODEL

            print(
                "   ✅ Recovery başarılı: "
                "metin whisper-1 üzerinden alındı."
            )

        else:

            raise RuntimeError(
                "Bu chunk'ta iki transcription modeli de "
                "kullanılabilir konuşma bulamadı.\n"
                "Muhtemel nedenler:\n"
                "- videoda konuşma yok / audio sessiz,\n"
                "- konuşma aşırı düşük seviyede,\n"
                "- source videoda yanlış audio track seçiliyor,\n"
                "- language='en' bu video için uygun değil."
            )

    if not timing_words:

        raise RuntimeError(
            "Konuşma metni bulundu fakat whisper-1 word timestamp "
            "üretemedi. Caption/edit timing güvenilir olmadığı için durduruldu."
        )

    (
        aligned_words,
        alignment_ratio,
    ) = align_transcript_words(
        accurate_text=accurate_text,
        timing_data=timing_data,
        chunk_duration=duration,
    )

    aligned_words = add_time_offset(
        words=aligned_words,
        offset=offset,
    )

    print(
        f"   🎯 Alignment: "
        f"{alignment_ratio * 100:.1f}%"
    )

    if (
        alignment_ratio
        < ALIGNMENT_WARNING_THRESHOLD
    ):

        print(
            "   ⚠️ Alignment düşük. "
            "Bu chunk manuel kontrol edilmeli."
        )

    print(
        f"   📝 Words: "
        f"{len(aligned_words)}"
    )

    if fallback_used:
        print(
            "   🛟 Fallback: whisper-1 text + timestamps"
        )

    return {
        "text": accurate_text,
        "words": aligned_words,
        "alignment_ratio": alignment_ratio,
        "fallback_used": fallback_used,
        "text_source": text_source,
    }

# ============================================================
# FULL TRANSCRIPTION
# ============================================================

def transcribe_audio(
    audio_path: str | Path,
    video_duration: float,
) -> dict[str, Any]:

    chunks = create_audio_chunks(
        audio_path=audio_path,
        duration=video_duration,
    )

    all_words: list[
        dict[str, Any]
    ] = []

    texts: list[str] = []

    alignment_scores: list[
        float
    ] = []

    fallback_chunk_count = 0

    text_sources: list[
        str
    ] = []

    print()
    print(
        "=" * 68
    )

    print(
        "🧠 HYBRID TRANSCRIPTION V3.2"
    )

    print(
        f"Accurate model : {ACCURATE_MODEL}"
    )

    print(
        f"Timing model   : {TIMING_MODEL}"
    )

    print(
        f"Chunk count    : {len(chunks)}"
    )

    print(
        "=" * 68
    )

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        result = process_chunk(
            chunk=chunk,
            chunk_index=index,
            chunk_count=len(
                chunks
            ),
        )

        texts.append(
            result[
                "text"
            ]
        )

        all_words.extend(
            result[
                "words"
            ]
        )

        alignment_scores.append(
            result[
                "alignment_ratio"
            ]
        )

        if result.get(
            "fallback_used"
        ) is True:
            fallback_chunk_count += 1

        text_sources.append(
            str(
                result.get(
                    "text_source",
                    ACCURATE_MODEL,
                )
            )
        )

    full_text = " ".join(
        text.strip()

        for text in texts

        if text.strip()
    ).strip()

    segments = build_segments(
        all_words
    )

    average_alignment = (
        sum(
            alignment_scores
        )
        / len(
            alignment_scores
        )

        if alignment_scores

        else 0.0
    )

    return {
        "text": full_text,

        "words": all_words,

        "segments": segments,

        "alignment_ratio": (
            average_alignment
        ),

        "chunk_count": len(
            chunks
        ),
        "fallback_chunk_count": fallback_chunk_count,
        "text_sources": text_sources,
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_transcript(
    transcription: dict[str, Any],
    video_duration: float,
) -> None:

    text = str(
        transcription.get(
            "text",
            "",
        )
    ).strip()

    words = transcription.get(
        "words",
        [],
    )

    segments = transcription.get(
        "segments",
        [],
    )

    if not text:

        raise RuntimeError(
            "Transcript text boş."
        )

    if not words:

        raise RuntimeError(
            "Transcript word timestamp içermiyor."
        )

    if not segments:

        raise RuntimeError(
            "Transcript segment içermiyor."
        )

    previous_start = -1.0
    previous_end = -1.0

    for index, word in enumerate(
        words
    ):

        start = float(
            word[
                "start"
            ]
        )

        end = float(
            word[
                "end"
            ]
        )

        if start < previous_start:

            raise RuntimeError(
                f"Word timestamp sırası bozuk: "
                f"index={index}"
            )

        # V3.1 için kritik.
        if (
            previous_end >= 0
            and start < previous_end - 0.002
        ):

            raise RuntimeError(
                "Word timestamp overlap bulundu: "
                f"index={index}, "
                f"start={start}, "
                f"previous_end={previous_end}"
            )

        if end <= start:

            raise RuntimeError(
                f"Word duration geçersiz: "
                f"index={index}"
            )

        if start > video_duration + 1.0:

            raise RuntimeError(
                f"Word timestamp video dışına çıktı: "
                f"index={index}"
            )

        previous_start = (
            start
        )

        previous_end = (
            end
        )


# ============================================================
# PACKAGE
# ============================================================

def build_transcript_package(
    video_path: str | Path,
    audio_path: str | Path,
    video_duration: float,
    transcription: dict[str, Any],
) -> dict[str, Any]:

    video_path = Path(
        video_path
    ).resolve()

    audio_path = Path(
        audio_path
    ).resolve()

    words = []

    for index, word in enumerate(
        transcription[
            "words"
        ]
    ):

        words.append(
            {
                "id": index,

                "start": round_time(
                    word[
                        "start"
                    ]
                ),

                "end": round_time(
                    word[
                        "end"
                    ]
                ),

                "duration": round_time(
                    float(
                        word[
                            "end"
                        ]
                    )
                    - float(
                        word[
                            "start"
                        ]
                    )
                ),

                "word": str(
                    word[
                        "word"
                    ]
                ),
            }
        )

    segments = []

    for index, segment in enumerate(
        transcription[
            "segments"
        ]
    ):

        segments.append(
            {
                "id": index,

                "start": round_time(
                    segment[
                        "start"
                    ]
                ),

                "end": round_time(
                    segment[
                        "end"
                    ]
                ),

                "duration": round_time(
                    segment[
                        "duration"
                    ]
                ),

                "text": str(
                    segment[
                        "text"
                    ]
                ),
            }
        )

    # Version 2 kalıyor.
    # Analyzer / pacing / timeline / captions bozulmuyor.
    return {
        "version": 2,

        "pipeline_version": "3.2",

        "source": {
            "video_path": str(
                video_path
            ),

            "audio_path": str(
                audio_path
            ),

            "video_name": (
                video_path.name
            ),

            "video_stem": (
                video_path.stem
            ),

            "duration": round_time(
                video_duration
            ),
        },

        "transcription": {
            "model": ACCURATE_MODEL,

            "timing_model": TIMING_MODEL,

            "language": (
                TRANSCRIPTION_LANGUAGE
            ),

            "method": (
                "gpt_transcribe_plus_"
                "repaired_word_alignment_v32_with_fallback"
            ),

            "alignment_score": round(
                float(
                    transcription[
                        "alignment_ratio"
                    ]
                ),
                4,
            ),

            "chunk_count": int(
                transcription[
                    "chunk_count"
                ]
            ),

            "fallback_chunk_count": int(
                transcription.get(
                    "fallback_chunk_count",
                    0,
                )
            ),

            "text_sources": list(
                transcription.get(
                    "text_sources",
                    [],
                )
            ),

            "timestamp_rules": {
                "min_word_duration": (
                    MIN_WORD_DURATION
                ),

                "max_word_duration": (
                    MAX_WORD_DURATION
                ),

                "min_center_gap": (
                    MIN_CENTER_GAP
                ),

                "overlap_allowed": False,
            },

            "text": str(
                transcription[
                    "text"
                ]
            ),
        },

        "segments": segments,

        "words": words,
    }


# ============================================================
# SAVE
# ============================================================

def save_transcript(
    video_path: str | Path,
    transcript_package: dict[str, Any],
) -> Path:

    video_path = Path(
        video_path
    ).resolve()

    TRANSCRIPT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        TRANSCRIPT_DIR
        / f"{video_path.stem}.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            transcript_package,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


# ============================================================
# CLEAN TEMP
# ============================================================

def cleanup_temp_chunks(
    audio_path: str | Path,
) -> None:

    audio_path = Path(
        audio_path
    ).resolve()

    directory = (
        TEMP_DIR
        / audio_path.stem
    )

    if directory.exists():

        shutil.rmtree(
            directory,
            ignore_errors=True,
        )


# ============================================================
# MAIN PROCESS
# ============================================================

def process_vod(
    video_path: str | Path,
) -> dict[str, Any]:

    video_path = Path(
        video_path
    ).resolve()

    if not video_path.exists():

        raise FileNotFoundError(
            f"VOD bulunamadı:\n{video_path}"
        )

    print()
    print(
        "=" * 68
    )

    print(
        "🎬 MIMIR VOD PROCESSOR V3.2"
    )

    print(
        "=" * 68
    )

    print(
        f"📹 {video_path.name}"
    )

    video_duration = get_video_duration(
        video_path
    )

    print(
        f"⏱️ Süre: "
        f"{video_duration:.2f}s"
    )

    audio_path = extract_audio(
        video_path
    )

    transcription = None
    package = None
    transcript_path = None

    try:

        transcription = transcribe_audio(
            audio_path=audio_path,
            video_duration=video_duration,
        )

        validate_transcript(
            transcription=transcription,
            video_duration=video_duration,
        )

        package = build_transcript_package(
            video_path=video_path,
            audio_path=audio_path,
            video_duration=video_duration,
            transcription=transcription,
        )

        transcript_path = save_transcript(
            video_path=video_path,
            transcript_package=package,
        )

    finally:

        cleanup_temp_chunks(
            audio_path
        )

    if (
        transcription is None
        or package is None
        or transcript_path is None
    ):

        raise RuntimeError(
            "Transcription tamamlanamadı."
        )

    print()
    print(
        "=" * 68
    )

    print(
        "✅ VOD TRANSCRIPTION V3.2 HAZIR"
    )

    print(
        f"🧠 Metin modeli: "
        f"{ACCURATE_MODEL}"
    )

    print(
        f"⏱️ Timing modeli: "
        f"{TIMING_MODEL}"
    )

    print(
        f"🎯 Alignment: "
        f"{transcription['alignment_ratio'] * 100:.1f}%"
    )

    print(
        f"📝 Word count: "
        f"{len(package['words'])}"
    )

    fallback_count = int(
        transcription.get(
            "fallback_chunk_count",
            0,
        )
    )

    if fallback_count:
        print(
            f"🛟 Whisper fallback kullanılan chunk: "
            f"{fallback_count}/{transcription['chunk_count']}"
        )

    print(
        "✅ Word overlap kontrolü geçti."
    )

    print(
        f"📂 Transcript:\n"
        f"{transcript_path}"
    )

    print(
        "=" * 68
    )

    return package


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "Yeni VOD dosyasını terminale sürükleyebilirsin."
    )

    video = input(
        "VOD dosyasının yolunu gir: "
    ).strip().strip('"')

    try:

        process_vod(
            video
        )

    except Exception as error:

        print()
        print(
            "❌ VOD PROCESSOR V3.2 HATASI:"
        )

        print(
            error
        )