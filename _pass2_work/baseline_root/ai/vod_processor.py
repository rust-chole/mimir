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

# Final-visible captions default to MAX accuracy.  In this mode stale .env
# overrides from older MIMIR builds cannot silently downgrade the STT stack.
# Set MIMIR_CAPTION_ACCURACY_MODE=custom only if you intentionally want to use
# the legacy model override variables below.
CAPTION_ACCURACY_MODE = os.getenv("MIMIR_CAPTION_ACCURACY_MODE", "max").strip().lower() or "max"
if CAPTION_ACCURACY_MODE == "max":
    CAPTION_ACCURATE_MODEL = "gpt-transcribe"
    CAPTION_CROSSCHECK_MODEL = "gpt-4o-transcribe"
else:
    CAPTION_ACCURATE_MODEL = os.getenv("MIMIR_CAPTION_TRANSCRIBE_MODEL", "gpt-transcribe").strip() or "gpt-transcribe"
    CAPTION_CROSSCHECK_MODEL = os.getenv("MIMIR_CAPTION_CROSSCHECK_MODEL", "gpt-4o-transcribe").strip() or "gpt-4o-transcribe"

# Whisper remains timing-only because the current transcription endpoint exposes
# its word timestamps directly; it never owns final caption wording.
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
bro, dude, chat, W, L, cooked, cap, no cap, no way, shawty,
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

# High-accuracy caption hinting for gpt-transcribe.  The primary pass stays
# mostly acoustic; a separate precision pass receives verified participant
# names and domain keywords.  This keeps one unbiased ear while still using
# the newer gpt-transcribe keyword interface where it helps most.
CAPTION_DEFAULT_KEYWORDS = tuple(
    item.strip()
    for item in os.getenv(
        "MIMIR_CAPTION_KEYWORDS",
        "chat,Twitch,YouTube,Discord,stream,streamer,IRL,dono,no cap,shawty",
    ).split(",")
    if item.strip()
)


def _caption_verified_names(known_names: list[str] | tuple[str, ...] | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in known_names or []:
        value = " ".join(str(raw).strip().split())
        key = value.casefold()
        if (
            not value
            or key in seen
            or key in {"a", "b", "c", "speaker a", "speaker b", "speaker c", "main", "secondary", "unknown"}
        ):
            continue
        seen.add(key)
        result.append(value)
    return result[:12]


def _caption_keywords(known_names: list[str] | tuple[str, ...] | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in [*CAPTION_DEFAULT_KEYWORDS, *_caption_verified_names(known_names)]:
        value = " ".join(str(raw).strip().split())
        key = value.casefold()
        if not value or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result[:32]


def _caption_name_context(known_names: list[str] | tuple[str, ...] | None = None) -> str:
    names = _caption_verified_names(known_names)
    if not names:
        return ""
    return (
        "\n\nVERIFIED PARTICIPANT SPELLINGS (reference only): "
        + ", ".join(names)
        + ". If the audio clearly says one of these names, use this spelling exactly. "
          "Do not force a name when the audio says something else."
    )


def _create_caption_transcription(
    audio_path: str | Path,
    *,
    model: str,
    prompt: str | None = None,
    keywords: list[str] | tuple[str, ...] | None = None,
) -> Any:
    """Create one caption transcription with model-correct hint fields.

    Current OpenAI file-transcription semantics are deliberately kept simple:
    ``gpt-transcribe`` receives ``keywords`` and ``languages`` through
    ``extra_body``; older models keep the singular ``language`` field.  This
    avoids silently losing verified-name hints on modern SDKs while retaining a
    conservative legacy fallback for older installs.
    """
    audio_path = Path(audio_path).resolve()
    selected_model = str(model)
    with audio_path.open("rb") as audio_file:
        kwargs: dict[str, Any] = {
            "model": selected_model,
            "file": audio_file,
        }
        if prompt:
            kwargs["prompt"] = prompt

        if selected_model == "gpt-transcribe":
            extra_body: dict[str, Any] = {}
            clean_keywords = [
                " ".join(str(item).strip().split())
                for item in (keywords or [])
                if " ".join(str(item).strip().split())
            ]
            if clean_keywords:
                extra_body["keywords"] = clean_keywords
            if TRANSCRIPTION_LANGUAGE:
                extra_body["languages"] = [TRANSCRIPTION_LANGUAGE]
            if extra_body:
                kwargs["extra_body"] = extra_body
        elif TRANSCRIPTION_LANGUAGE:
            kwargs["language"] = TRANSCRIPTION_LANGUAGE

        try:
            return client.audio.transcriptions.create(**kwargs)
        except TypeError as error:
            # Very old OpenAI Python SDKs may not expose ``extra_body``.  Do not
            # break the pipeline: retry once using the older request shape.  The
            # prompt still carries verified spellings even if literal keyword
            # hints cannot be sent by that SDK.
            if selected_model != "gpt-transcribe" or "extra_body" not in kwargs:
                raise
            kwargs.pop("extra_body", None)
            if TRANSCRIPTION_LANGUAGE:
                kwargs["language"] = TRANSCRIPTION_LANGUAGE
            audio_file.seek(0)
            print(
                "   ⚠️ OpenAI SDK gpt-transcribe extra_body hintlerini desteklemiyor; "
                f"legacy uyumla devam: {error}"
            )
            return client.audio.transcriptions.create(**kwargs)


# ============================================================
# FIXED WHISPER WORD CLOCK (CLEAN CAPTION CLOCK)
# ============================================================

def align_text_to_fixed_whisper_clock(
    accurate_text: str,
    timing_data: dict[str, Any],
    chunk_duration: float,
) -> tuple[list[dict[str, Any]], float]:
    """Map GPT wording onto one immutable Whisper word clock.

    Authority contract:
    - GPT owns visible wording.
    - Whisper owns measured word onsets.
    - Exact Whisper anchors are never shifted to make GPT fit.
    - GPT-only words are placed only inside the local gap before the next
      Whisper anchor; if that gap is too small, time is borrowed from the TAIL
      of the previous token. A later measured onset is never pushed.
    - No global offset, silence snap, segment interpolation or cumulative shift.
    """
    target = tokenize_accurate_text(accurate_text)
    clock = extract_timing_words(timing_data)
    duration = max(0.0, float(chunk_duration))
    if not target or not clock:
        raise RuntimeError("Fixed Whisper caption clock için text/word timestamp eksik.")

    target_canon = [canonical_word(item) for item in target]
    clock_canon = [str(item.get("canonical", "")) for item in clock]
    matcher = SequenceMatcher(None, clock_canon, target_canon, autojunk=False)
    alignment_ratio = float(matcher.ratio())
    rows: list[dict[str, Any]] = []

    MIN_WORD = 0.035
    MIN_GAP = 0.005

    def append_row(word: str, start: float, end: float, source: str) -> None:
        word = str(word).strip()
        if not word:
            return
        start = max(0.0, min(duration, float(start)))
        end = max(start + 0.025, min(duration, float(end)))
        if rows and float(rows[-1]["end"]) > start:
            # Never move the current/later onset. Only shorten the previous tail.
            #
            # V3 used a 25 ms minimum on the PREVIOUS token while trimming an
            # overlap. On dense Whisper clocks that floor can itself remain to
            # the right of the next immutable onset, making the final audit fail
            # even though the later anchor was correct. Acoustic duration and
            # DISPLAY duration are separate contracts: a microscopic acoustic
            # tail is legal here; captions.py owns readability extension later.
            previous = rows[-1]
            previous_start = float(previous["start"])
            previous_end = float(previous["end"])

            # Coincident/near-coincident measured onsets cannot support two
            # independently ordered word intervals at millisecond precision.
            # Preserve every visible word by grouping only this unresolved pair;
            # no later onset is moved and no text is dropped.
            if start <= previous_start + 0.001:
                previous["word"] = (str(previous.get("word", "")).strip() + " " + word).strip()
                previous["end"] = round_time(max(previous_end, end))
                previous["alignment_source"] = (
                    str(previous.get("alignment_source", source)) + "+coincident_anchor_group"
                )
                return

            safe_end = min(previous_end, start - MIN_GAP)
            if safe_end <= previous_start:
                # There is less than MIN_GAP available. Ending exactly at the
                # next measured onset is still monotonic and keeps that onset
                # byte-for-byte intact. The renderer can hold the prior word
                # visually without changing this acoustic clock.
                safe_end = start
            previous["end"] = round_time(min(start, max(previous_start + 0.001, safe_end)))

        rows.append({
            "word": word,
            "start": round_time(start),
            "end": round_time(end),
            "alignment_source": source,
        })

    def distribute(words: list[str], start: float, end: float, source: str) -> None:
        words = [str(word).strip() for word in words if str(word).strip()]
        if not words:
            return
        start = max(0.0, min(duration, float(start)))
        end = max(start + 0.025, min(duration, float(end)))
        span = max(0.0, end - start)
        if len(words) == 1:
            append_row(words[0], start, end, source)
            return
        if span < MIN_WORD * len(words):
            # Preserve wording without borrowing from any later clock anchor.
            append_row(" ".join(words), start, end, source + "_group")
            return
        weights = [max(1, len(canonical_word(word))) for word in words]
        total = float(sum(weights) or len(words))
        cursor = start
        remaining = span
        remaining_weight = total
        for index, (word, weight) in enumerate(zip(words, weights)):
            if index == len(words) - 1:
                token_end = end
            else:
                minimum_for_rest = MIN_WORD * (len(words) - index - 1)
                proportional = remaining * (float(weight) / max(1.0, remaining_weight))
                allocation = max(MIN_WORD, min(proportional, remaining - minimum_for_rest))
                token_end = min(end, cursor + allocation)
            append_row(word, cursor, token_end, source)
            remaining = max(0.0, end - token_end)
            remaining_weight = max(0.0, remaining_weight - float(weight))
            cursor = token_end

    def source_region(source_words: list[dict[str, Any]]) -> tuple[float, float] | None:
        if not source_words:
            return None
        start = float(source_words[0].get("start", 0.0))
        end = float(source_words[-1].get("end", start + 0.04))
        return max(0.0, start), min(duration, max(start + 0.025, end))

    def place_insertion(words: list[str], source_position: int) -> None:
        """Place GPT-only words locally without touching the next Whisper onset."""
        words = [str(word).strip() for word in words if str(word).strip()]
        if not words:
            return
        prev_clock = clock[source_position - 1] if source_position > 0 else None
        next_clock = clock[source_position] if source_position < len(clock) else None
        next_start = (
            float(next_clock.get("start", 0.0))
            if next_clock is not None else duration
        )
        previous_end = (
            float(prev_clock.get("end", 0.0))
            if prev_clock is not None else 0.0
        )
        required = max(0.055, MIN_WORD * len(words))

        # Best case: Whisper left a real acoustic gap between anchors.
        gap_start = max(0.0, previous_end)
        gap_end = min(duration, next_start)
        if gap_end - gap_start >= required:
            distribute(words, gap_start + MIN_GAP, gap_end - MIN_GAP, "whisper_local_insert_gap")
            return

        # Middle/trailing insertion: borrow only from the previous token's tail.
        # The next measured onset remains byte-for-byte unchanged.
        if rows and prev_clock is not None:
            previous = rows[-1]
            previous_start = float(previous.get("start", 0.0))
            hard_right = gap_end if next_clock is not None else min(duration, max(gap_end, previous_end + required))
            insertion_start = max(previous_start + MIN_WORD, hard_right - required)
            if hard_right - insertion_start >= 0.025:
                previous["end"] = round_time(max(previous_start + 0.025, insertion_start - MIN_GAP))
                distribute(words, insertion_start, hard_right - (MIN_GAP if next_clock is not None else 0.0), "whisper_local_insert_tail")
                return

        # Leading insertion has no earlier anchor. Keep it local immediately
        # before the first measured Whisper onset. This may estimate only the
        # missing leading words; it never moves the first/later Whisper anchor.
        if next_clock is not None:
            end = max(0.025, next_start - MIN_GAP)
            start = max(0.0, end - required)
            if end > start + 0.02:
                distribute(words, start, end, "whisper_local_insert_leading")
                return

        # Last-resort local grouping. No later onset exists to corrupt.
        if rows:
            previous = rows[-1]
            start = float(previous.get("end", previous.get("start", 0.0)))
            end = min(duration, max(start + 0.04, start + required))
            if end > start:
                distribute(words, start, end, "whisper_local_insert_trailing")
                return

        raise RuntimeError("GPT-only caption kelimeleri için güvenli lokal clock alanı bulunamadı.")

    for tag, source_start, source_end, target_start, target_end in matcher.get_opcodes():
        source_slice = clock[source_start:source_end]
        target_slice = target[target_start:target_end]

        if tag == "equal":
            for timed, token in zip(source_slice, target_slice):
                append_row(
                    token,
                    float(timed.get("start", 0.0)),
                    float(timed.get("end", timed.get("start", 0.0))),
                    "whisper_exact_anchor",
                )
            continue

        if tag == "replace":
            region = source_region(source_slice)
            if region is not None:
                distribute(target_slice, region[0], region[1], "whisper_local_replace")
            elif target_slice:
                place_insertion(target_slice, source_start)
            continue

        if tag == "delete":
            # Whisper heard an extra lexical token. GPT wording authority omits
            # it; later Whisper anchors stay untouched.
            continue

        if tag == "insert":
            place_insertion(target_slice, source_start)
            continue

    if not rows:
        raise RuntimeError("Fixed Whisper caption alignment kelime üretmedi.")

    # Defensive monotonic audit. It is illegal to repair a violation by moving a
    # later onset; fail closed so the caller can surface the problem instead.
    previous_start = -1.0
    previous_end = -1.0
    for row in rows:
        start = float(row["start"])
        end = float(row["end"])
        if start + 1e-9 < previous_start or start + 1e-9 < previous_end - 0.010:
            raise RuntimeError("Caption clock monotonicity ihlali; later onset taşınmadı.")
        if end <= start:
            raise RuntimeError("Caption clock sıfır/negatif kelime süresi üretti.")
        previous_start = start
        previous_end = end

    rows[0]["fixed_clock_dropped_insertions"] = 0
    return rows, alignment_ratio



# Backward-compatible alias for the short-lived experimental helper name.  Any
# caller that picked it up still gets the clean fixed Whisper behavior.


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


def transcribe_caption_accurate_text(
    audio_path: str | Path,
    *,
    known_names: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Primary FINAL-caption listen using the strongest file STT model.

    Pass 1 is intentionally acoustic-first and receives only generic domain
    keyword spellings.  Verified participant names are reserved for the separate
    precision pass so one evidence channel remains independent of identity hints.
    """
    audio_path = Path(audio_path).resolve()
    attempts: list[tuple[str | None, list[str], str]] = [
        (None, _caption_keywords(None), "acoustic + domain keywords"),
        (TRANSCRIPTION_PROMPT, _caption_keywords(None), "empty-response recovery"),
    ]
    for attempt_index, (prompt, keywords, label) in enumerate(attempts, start=1):
        print(
            f"   📝 {CAPTION_ACCURATE_MODEL}: final caption transcript "
            f"({attempt_index}/{len(attempts)}, {label})..."
        )
        response = _create_caption_transcription(
            audio_path,
            model=CAPTION_ACCURATE_MODEL,
            prompt=prompt,
            keywords=keywords,
        )
        text = get_response_text(response)
        if text:
            return text
    return ""

def _transcribe_caption_pass(
    audio_path: str | Path,
    *,
    model: str,
    label: str,
    prompt: str | None,
    keywords: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Independent ASR pass for the short that will actually be published."""
    audio_path = Path(audio_path).resolve()
    print(f"   🎯 {model}: {label}...")
    response = _create_caption_transcription(
        audio_path,
        model=model,
        prompt=prompt,
        keywords=keywords,
    )
    return get_response_text(response)

def transcribe_caption_independent_text(
    audio_path: str | Path,
    *,
    known_names: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Second gpt-transcribe ear with verified-name and domain hinting.

    Names are spelling references only.  The instructions explicitly forbid
    forcing them into audio that does not contain them.
    """
    prompt = (
        TRANSCRIPTION_PROMPT
        + _caption_name_context(known_names)
        + "\n\nPRECISION VERIFICATION: Listen from scratch. Preserve contractions, negations, "
          "names, numbers, slang, profanity, clipped words and repetitions exactly as audible. "
          "Do not make the sentence more grammatical and do not copy wording from another transcript."
    )
    return _transcribe_caption_pass(
        audio_path,
        model=CAPTION_ACCURATE_MODEL,
        label="verified-name/context precision listen",
        prompt=prompt,
        keywords=_caption_keywords(known_names),
    )

def transcribe_caption_crosscheck_text(
    audio_path: str | Path,
) -> str:
    """Model-diverse acoustic cross-check. Used only when more evidence is useful."""
    return _transcribe_caption_pass(
        audio_path,
        model=CAPTION_CROSSCHECK_MODEL,
        label="model-diverse acoustic cross-check",
        prompt=None,
        keywords=None,
    )






def extract_caption_micro_audio(
    audio_path: str | Path,
    *,
    start: float,
    end: float,
    label: str = "suspect",
    enhanced: bool = False,
) -> Path:
    """Cut one suspicious window from the lossless final-caption master.

    ``enhanced=False`` keeps a raw PCM view. ``enhanced=True`` creates a second,
    deterministic speech-oriented view using only gentle band-limiting,
    compression and loudness normalization. No aggressive denoiser is used,
    because denoisers can erase consonants that matter to transcription.
    """
    audio_path = Path(audio_path).resolve()
    start = max(0.0, float(start))
    end = max(start + 0.20, float(end))
    duration = end - start

    micro_dir = TEMP_DIR / "caption_micro"
    micro_dir.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", str(label)).strip("_") or "suspect"
    view = "enh" if enhanced else "raw"
    output = micro_dir / (
        f"{audio_path.stem}_{int(start * 1000):07d}_{int(end * 1000):07d}_{safe_label}_{view}.wav"
    )

    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", str(audio_path),
        "-t", f"{duration:.3f}", "-vn",
    ]
    if enhanced:
        command += [
            "-af",
            "highpass=f=70,lowpass=f=7800,acompressor=threshold=-20dB:ratio=2.2:attack=8:release=90,loudnorm=I=-16:TP=-1.5:LRA=9",
        ]
    command += [
        "-ac", "1", "-ar", str(CAPTION_MASTER_SAMPLE_RATE), "-c:a", "pcm_s16le",
        str(output),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0 or not output.is_file() or output.stat().st_size < 512:
        raise RuntimeError(
            "Caption micro-audio oluşturulamadı: "
            + (result.stderr.strip() or f"{start:.3f}-{end:.3f}")
        )
    return output


def transcribe_caption_micro_pass(
    audio_path: str | Path,
    *,
    model: str,
    label: str,
    context_before: str = "",
    context_after: str = "",
    extra_instruction: str = "",
    use_context: bool = True,
    known_names: list[str] | tuple[str, ...] | None = None,
) -> str:
    """High-precision ASR over a small acoustic window.

    ``use_context=False`` is the acoustic-only evidence channel. Context-guided
    passes are separate evidence so context can help without silently biasing all
    listeners toward the same guess.
    """
    context_before = " ".join(str(context_before).split())[-360:]
    context_after = " ".join(str(context_after).split())[:360]
    prompt: str | None = None
    if use_context or extra_instruction:
        prompt = (
            TRANSCRIPTION_PROMPT
            + "\n\nMICRO ACCURACY PASS: This file is a short excerpt around a suspicious word or phrase. "
              "Transcribe the ENTIRE audible excerpt exactly from scratch. Never repair grammar. "
              "Negations, contractions, names, numbers, slang, clipped syllables and repetitions are critical."
            + (f"\nContext immediately before (reference only): {context_before}" if use_context and context_before else "")
            + (f"\nContext immediately after (reference only): {context_after}" if use_context and context_after else "")
            + (f"\n{extra_instruction}" if extra_instruction else "")
        )
    return _transcribe_caption_pass(
        audio_path,
        model=str(model),
        label=str(label),
        prompt=prompt,
        keywords=_caption_keywords(known_names) if str(model) == "gpt-transcribe" else None,
    )


# ============================================================
# MODERN SEGMENT-ANCHORED CAPTION TIMING
# ============================================================





# ============================================================
# WHISPER TIMING PASS (FALLBACK ONLY)
# ============================================================

def transcribe_word_timing(
    audio_path: str | Path,
    *,
    known_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """The ONE immutable final-caption clock: raw Whisper word timestamps.

    V29 intentionally sends NO name/context prompt to the timing ear.  Names and
    wording belong to gpt-transcribe; speaker identity belongs to diarization.
    Whisper receives only the exact final 48 kHz PCM plus the language hint, so
    its acoustic word boundaries cannot be biased by a participant-name prompt.
    ``known_names`` remains in the signature only for drop-in compatibility and
    is deliberately ignored.
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