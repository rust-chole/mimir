from __future__ import annotations

import base64
import json
import os
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ai.openai_client import client
from ai import vod_processor, model_config
from ai.editor import speaker_role_judge
from ai.caption_stack import final_captions


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SPEAKER_OUTPUT_DIR = PROJECT_ROOT / "vod_output" / "speaker_captions"
TEMP_DIR = PROJECT_ROOT / "vod_output" / "temp" / "speaker_captions"

SPEAKER_PROFILE_VERSION = 27
DIARIZATION_MODEL = os.getenv(
    "CAPTION_DIARIZATION_MODEL",
    "gpt-4o-transcribe-diarize",
).strip() or "gpt-4o-transcribe-diarize"
CAPTION_LANGUAGE = os.getenv("CAPTION_LANGUAGE", "en").strip()

# Speaker discovery must hear quieter secondary voices before the local noise/crowd
# filters decide whether they are real participants. OpenAI server VAD defaults to
# 0.50; 0.35 is deliberately a modest sensitivity increase, not an aggressive
# open-mic setting. These remain env-overridable.
DIARIZATION_VAD_THRESHOLD = max(0.0, min(1.0, float(os.getenv("MIMIR_DIARIZATION_VAD_THRESHOLD", "0.35") or 0.35)))
DIARIZATION_VAD_PREFIX_PADDING_MS = max(0, int(os.getenv("MIMIR_DIARIZATION_VAD_PREFIX_PADDING_MS", "300") or 300))
DIARIZATION_VAD_SILENCE_MS = max(50, int(os.getenv("MIMIR_DIARIZATION_VAD_SILENCE_MS", "220") or 220))

# V11 speaker-recall ensemble. The normal path stays one request. Only a
# single/uncertain first pass opens the sensitive retry. An anchored third pass
# is allowed only after a plausible 2/3-speaker pass has produced reference
# samples, so extra cost is conditional rather than unconditional.
DIARIZATION_RETRY_VAD_THRESHOLD = max(0.0, min(1.0, float(os.getenv("MIMIR_DIARIZATION_RETRY_VAD_THRESHOLD", "0.22") or 0.22)))
DIARIZATION_RETRY_PREFIX_PADDING_MS = max(0, int(os.getenv("MIMIR_DIARIZATION_RETRY_PREFIX_PADDING_MS", "450") or 450))
DIARIZATION_RETRY_SILENCE_MS = max(50, int(os.getenv("MIMIR_DIARIZATION_RETRY_SILENCE_MS", "140") or 140))
DIARIZATION_REFERENCE_MIN_SECONDS = max(2.0, float(os.getenv("MIMIR_DIARIZATION_REFERENCE_MIN_SECONDS", "2.0") or 2.0))
DIARIZATION_REFERENCE_MAX_SECONDS = max(DIARIZATION_REFERENCE_MIN_SECONDS, min(10.0, float(os.getenv("MIMIR_DIARIZATION_REFERENCE_MAX_SECONDS", "5.0") or 5.0)))
SPEAKER_COUNT_CONFIRM_CONFIDENCE = max(0.0, min(1.0, float(os.getenv("MIMIR_SPEAKER_COUNT_CONFIRM_CONFIDENCE", "0.96") or 0.96)))

# Final-caption WORDS and their CLOCK are produced by ai.caption_stack from the
# exact final edited clip (Qwen ears -> frozen transcript -> forced alignment).
# This module owns WHO: diarization, word -> speaker assignment and the caption
# colour of each VOICE (never a person's identity). Speaker metadata can never
# rewrite a caption word or its timing.

# Real participants must have a meaningful amount of speech.  Tiny speaker
# fragments are usually cross-talk, game audio, crowd noise or diarization
# instability and must not become their own subtitle lane.
MIN_MEANINGFUL_WORDS = 2
MIN_DUAL_SECONDS = 0.85
MIN_DUAL_SHARE = 0.075

# Word-to-speaker assignment.  These are intentionally permissive because a
# missed diarization boundary must never erase a real caption word.
NEAR_SEGMENT_TOLERANCE = 0.34

# V4 speaker/meaning integrity.  Diarization is evidence for WHO spoke; it is
# never allowed to rewrite the high-accuracy transcript or its word clock.
TEXT_MATCH_CONFIDENCE = 0.92
TIMING_ONLY_MAX_CONFIDENCE = 0.72
MIN_STABLE_TURN_WORDS = 2
MIN_STABLE_TURN_SECONDS = 0.42
MIN_STABLE_TURN_CONFIDENCE = 0.70
MAX_SINGLE_WORD_FLIP_SECONDS = 0.48
# V5: continuity may bridge only a genuinely local diarization boundary. A
# caption run seconds away from the last measured speaker turn is UNKNOWN, not
# automatically the previous person. This prevents human names from leaking
# across VAD/diarization holes.
MAX_CONTINUITY_FILL_GAP_SECONDS = max(0.12, min(1.00, float(os.getenv("MIMIR_MAX_CONTINUITY_FILL_GAP_SECONDS", "0.48") or 0.48)))
MIN_DUAL_SEPARATION_QUALITY = 0.58
MIN_DUAL_TEXT_ALIGNMENT = 0.50
MAX_DUAL_SWITCHES_PER_20_WORDS = 6.0
MIN_DUAL_MEANING_COVERAGE = 0.94
TEXT_ALIGNMENT_SOFT_TIME_DRIFT = 0.55
TEXT_ALIGNMENT_MAX_TIME_DRIFT = 1.25

# A diarizer can split cheers, laughter, game/room audio or a one-off shout into
# extra speaker IDs.  These IDs must not become a third human participant.
NOISE_TEXT_MARKERS = (
    "[music]", "[applause]", "[cheering]", "[crowd]", "[laughter]",
    "music", "applause", "cheering", "crowd noise", "laughter",
    "laughing", "screaming", "scream", "shouting", "yelling",
)
MIN_REAL_TURN_SECONDS = 0.55
MIN_REAL_TURN_WORDS = 2
MIN_THIRD_REAL_TURNS = 2
MIN_THIRD_STRONG_SECONDS = 2.20
MIN_THIRD_STRONG_WORDS = 5


def _write_json(path: str | Path, data: dict[str, Any]) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)
    return path


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(value, "dict"):
        dumped = value.dict()
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _segment_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return {
        "start": getattr(value, "start", 0.0),
        "end": getattr(value, "end", 0.0),
        "speaker": getattr(value, "speaker", ""),
        "text": getattr(value, "text", ""),
    }


def _safe_name(text: str) -> str:
    value = str(text).strip()
    for char in '<>:"/\\|?*':
        value = value.replace(char, "_")
    return " ".join(value.split()).strip(" ._") or "clip"


def _probe_duration(path: str | Path) -> float:
    path = Path(path).resolve()
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffprobe duration okunamadı")
    return max(0.0, float(completed.stdout.strip()))


def _output_path(edited_clip_path: str | Path, clip_index: int) -> Path:
    edited_clip_path = Path(edited_clip_path).resolve()
    directory = SPEAKER_OUTPUT_DIR / _safe_name(edited_clip_path.parent.name)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"clip_{int(clip_index):02d}_speakers_v{SPEAKER_PROFILE_VERSION}.json"


def _extract_audio(edited_clip_path: str | Path, clip_index: int) -> Path:
    edited_clip_path = Path(edited_clip_path).resolve()
    if not edited_clip_path.is_file():
        raise FileNotFoundError(f"Edited clip bulunamadı: {edited_clip_path}")

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    output = TEMP_DIR / f"clip_{int(clip_index):02d}_{abs(hash(str(edited_clip_path))) & 0xfffffff:x}.wav"
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(edited_clip_path),
        "-vn", "-ac", "1", "-ar", str(getattr(vod_processor, "CAPTION_MASTER_SAMPLE_RATE", 48000)), "-c:a", "pcm_s16le",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size < 1024:
        raise RuntimeError(
            "Edited clip caption audio'su çıkarılamadı. "
            + (completed.stderr.strip() or "ffmpeg başarısız")
        )
    return output


def _run_diarization(
    audio_path: Path,
    *,
    chunking: str = "auto",
    vad_threshold: float | None = None,
    vad_prefix_padding_ms: int | None = None,
    vad_silence_ms: int | None = None,
    known_speaker_names: list[str] | None = None,
    known_speaker_references: list[str] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": DIARIZATION_MODEL,
        "response_format": "diarized_json",
    }

    if chunking == "auto":
        # OpenAI auto chunking normalizes loudness first, then chooses VAD
        # boundaries. This is the highest-recall normal path and avoids a
        # hand-tuned threshold on every clean clip.
        kwargs["chunking_strategy"] = "auto"
    else:
        kwargs["chunking_strategy"] = {
            "type": "server_vad",
            "threshold": float(
                DIARIZATION_VAD_THRESHOLD if vad_threshold is None else vad_threshold
            ),
            "prefix_padding_ms": int(
                DIARIZATION_VAD_PREFIX_PADDING_MS
                if vad_prefix_padding_ms is None else vad_prefix_padding_ms
            ),
            "silence_duration_ms": int(
                DIARIZATION_VAD_SILENCE_MS
                if vad_silence_ms is None else vad_silence_ms
            ),
        }

    if CAPTION_LANGUAGE:
        kwargs["language"] = CAPTION_LANGUAGE

    if known_speaker_names and known_speaker_references:
        if len(known_speaker_names) == len(known_speaker_references):
            kwargs["known_speaker_names"] = list(known_speaker_names)[:4]
            kwargs["known_speaker_references"] = list(known_speaker_references)[:4]

    with audio_path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(file=audio_file, **kwargs)

    data = _as_dict(response)
    if data:
        return data
    raw_segments = getattr(response, "segments", [])
    return {
        "text": str(getattr(response, "text", "")),
        "duration": float(getattr(response, "duration", 0.0) or 0.0),
        "segments": [_segment_dict(item) for item in raw_segments],
    }


def _normalize_segments(data: dict[str, Any], duration: float) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(data.get("segments", []) or []):
        item = _segment_dict(raw)
        speaker = str(item.get("speaker", "")).strip() or "UNKNOWN"
        text = str(item.get("text", "")).strip()
        try:
            start = max(0.0, min(duration, float(item.get("start", 0.0))))
            end = max(start, min(duration, float(item.get("end", start))))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        result.append(
            {
                "id": str(item.get("id", f"seg_{index:03d}")),
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "speaker": speaker,
                "text": text,
                "word_count": len(text.split()),
            }
        )
    result.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return result


def _looks_like_noise_text(text: str) -> bool:
    value = " ".join(str(text).casefold().split())
    if not value:
        return True
    return any(marker in value for marker in NOISE_TEXT_MARKERS)


def _speaker_stats(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw: dict[str, dict[str, Any]] = {}
    total = 0.0
    for segment in segments:
        speaker = str(segment["speaker"])
        seconds = max(0.0, float(segment["duration"]))
        words = int(segment.get("word_count", 0))
        text = str(segment.get("text", ""))
        total += seconds
        entry = raw.setdefault(
            speaker,
            {
                "speaker": speaker,
                "speaking_seconds": 0.0,
                "segment_count": 0,
                "word_count": 0,
                "substantial_segment_count": 0,
                "noise_segment_count": 0,
                "max_segment_seconds": 0.0,
            },
        )
        entry["speaking_seconds"] += seconds
        entry["segment_count"] += 1
        entry["word_count"] += words
        entry["max_segment_seconds"] = max(float(entry["max_segment_seconds"]), seconds)
        if _looks_like_noise_text(text):
            entry["noise_segment_count"] += 1
        if (
            seconds >= MIN_REAL_TURN_SECONDS
            and words >= MIN_REAL_TURN_WORDS
            and not _looks_like_noise_text(text)
        ):
            entry["substantial_segment_count"] += 1

    denominator = max(total, 1e-6)
    result: list[dict[str, Any]] = []
    for entry in raw.values():
        seconds = float(entry["speaking_seconds"])
        segment_count = max(1, int(entry["segment_count"]))
        words = int(entry["word_count"])
        noise_count = int(entry["noise_segment_count"])
        result.append(
            {
                **entry,
                "speaking_seconds": round(seconds, 3),
                "share": round(seconds / denominator, 4),
                "avg_segment_seconds": round(seconds / segment_count, 3),
                "avg_words_per_segment": round(words / segment_count, 3),
                "noise_segment_ratio": round(noise_count / segment_count, 3),
            }
        )
    result.sort(
        key=lambda item: (
            float(item["speaking_seconds"]),
            int(item["word_count"]),
            int(item.get("substantial_segment_count", 0)),
        ),
        reverse=True,
    )
    return result


def _participant_like(item: dict[str, Any]) -> bool:
    seconds = float(item.get("speaking_seconds", 0.0))
    words = int(item.get("word_count", 0))
    substantial = int(item.get("substantial_segment_count", 0))
    noise_ratio = float(item.get("noise_segment_ratio", 0.0))
    avg_seconds = float(item.get("avg_segment_seconds", 0.0))
    avg_words = float(item.get("avg_words_per_segment", 0.0))

    # One coherent spoken turn is enough for a genuine second participant.
    if substantial >= 1 and seconds >= 0.55 and words >= 2 and noise_ratio < 0.67:
        return True

    # Repeated coherent fragments can also be a real person even when no
    # individual turn crosses the substantial-turn threshold.
    if seconds >= 0.95 and words >= 3 and avg_seconds >= 0.32 and avg_words >= 1.2 and noise_ratio < 0.50:
        return True

    return False


def _strong_third_participant(item: dict[str, Any]) -> bool:
    """A third speaker needs stronger evidence than A/B.

    This is the crowd-noise guard: a cheer, laugh or one-off shout may be
    diarized as SPEAKER_02, but it cannot flip a two-person clip into crowd.
    """
    if not _participant_like(item):
        return False
    substantial = int(item.get("substantial_segment_count", 0))
    seconds = float(item.get("speaking_seconds", 0.0))
    words = int(item.get("word_count", 0))
    return substantial >= MIN_THIRD_REAL_TURNS or (
        seconds >= MIN_THIRD_STRONG_SECONDS and words >= MIN_THIRD_STRONG_WORDS
    )


def _classify(
    stats: list[dict[str, Any]],
) -> tuple[str, str | None, str | None, list[str], list[str], list[str]]:
    if not stats:
        return "single", None, None, [], [], []

    participants = [item for item in stats if _participant_like(item)]
    participant_ids = [str(item["speaker"]) for item in participants]
    background_ids = [
        str(item["speaker"]) for item in stats
        if str(item["speaker"]) not in set(participant_ids)
    ]

    # If diarization is weak, preserve the top cluster as a single speaker
    # instead of deleting captions. Identity naming is skipped if primary is
    # missing elsewhere in the pipeline.
    if not participants:
        primary = str(stats[0]["speaker"])
        return "single", primary, None, [primary], [primary], background_ids

    primary = str(participants[0]["speaker"])
    if len(participants) == 1:
        return "single", primary, None, [primary], participant_ids, background_ids

    second_item = participants[1]
    secondary_ok = (
        float(second_item.get("speaking_seconds", 0.0)) >= MIN_DUAL_SECONDS
        and int(second_item.get("word_count", 0)) >= MIN_MEANINGFUL_WORDS
        and (
            float(second_item.get("share", 0.0)) >= MIN_DUAL_SHARE
            or float(second_item.get("speaking_seconds", 0.0)) >= 2.0
        )
    )
    if not secondary_ok:
        secondary_id = str(second_item["speaker"])
        if secondary_id not in background_ids:
            background_ids.append(secondary_id)
        return "single", primary, None, [primary], [primary], background_ids

    secondary = str(second_item["speaker"])

    # A convincingly real third human is supported as a named participant.
    # Background cheers/room/game audio still never become participant #3.
    real_thirds = [item for item in participants[2:] if _strong_third_participant(item)]
    if len(real_thirds) == 1:
        third = str(real_thirds[0]["speaker"])
        triple = [primary, secondary, third]
        return "triple", primary, secondary, triple, triple, [str(item["speaker"]) for item in stats if str(item["speaker"]) not in set(triple)]
    if len(real_thirds) > 1:
        return "crowd", primary, None, [primary], participant_ids, background_ids

    return "dual", primary, secondary, [primary, secondary], participant_ids[:2], background_ids



def _coherent_speaker_ids(
    stats: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> list[str]:
    """Very permissive safety census used only to decide whether to ASK.

    This census preserves plausible speaker IDs, but the human prompt is separately confidence-gated.
    Final caption assignment remains protected by the proven V7 turn-quality
    gate, so this does not allow weak speaker evidence to corrupt timestamps.
    """
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for segment in segments:
        speaker = str(segment.get("speaker", ""))
        if speaker:
            by_speaker.setdefault(speaker, []).append(segment)

    result: list[str] = []
    for item in stats:
        speaker = str(item.get("speaker", ""))
        if not speaker:
            continue
        seconds = float(item.get("speaking_seconds", 0.0) or 0.0)
        words = int(item.get("word_count", 0) or 0)
        noise_ratio = float(item.get("noise_segment_ratio", 0.0) or 0.0)
        coherent_turn = False
        for seg in by_speaker.get(speaker, []):
            text = str(seg.get("text", ""))
            duration = float(seg.get("duration", 0.0) or 0.0)
            word_count = int(seg.get("word_count", 0) or 0)
            if not _looks_like_noise_text(text) and (
                (word_count >= 2 and duration >= 0.34)
                or (word_count >= 3 and duration >= 0.24)
            ):
                coherent_turn = True
                break
        if coherent_turn or (seconds >= 0.48 and words >= 2 and noise_ratio < 0.75):
            result.append(speaker)
    return result


def _classify_with_luna(
    segments: list[dict[str, Any]],
    stats: list[dict[str, Any]],
) -> tuple[str, str | None, str | None, list[str], list[str], list[str], dict[str, Any]]:
    """Luna-low role judge with an ask-biased deterministic safety net."""
    local_mode, local_primary, local_secondary, local_kept, local_participants, local_background = _classify(stats)
    known = [str(item.get("speaker", "")) for item in stats if str(item.get("speaker", ""))]
    known_set = set(known)
    coherent = [speaker for speaker in _coherent_speaker_ids(stats, segments) if speaker in known_set]

    try:
        decision = speaker_role_judge.classify_speaker_roles(segments, stats)
    except Exception as error:
        decision = {
            "status": "fallback",
            "mode": local_mode,
            "primary_speaker": local_primary,
            "secondary_speaker": local_secondary,
            "human_speakers": local_participants,
            "background_speakers": local_background,
            "confidence": 0.0,
            "reason": f"Luna role judge failed: {error}",
        }

    if str(decision.get("status", "")) == "ok":
        mode = str(decision.get("mode", local_mode))
        primary = str(decision.get("primary_speaker") or "") or None
        secondary = str(decision.get("secondary_speaker") or "") or None
        humans = [str(x) for x in decision.get("human_speakers", []) if str(x) in known_set]
        background = [str(x) for x in decision.get("background_speakers", []) if str(x) in known_set]
        confidence = float(decision.get("confidence", 0.0) or 0.0)

        # Preserve two coherent lexical voices internally; caption colours are
        # still earned per turn later (_assign_speaker_colors).
        if len(coherent) == 2 and mode != "dual":
            explicitly_background = set(background)
            rejected = [x for x in coherent if x in explicitly_background]
            if not (len(rejected) == 1 and confidence >= 0.94):
                mode = "dual"
                primary = coherent[0]
                secondary = coherent[1]
                humans = coherent[:2]
                background = [x for x in known if x not in set(humans)]
                decision = {
                    **decision,
                    "safety_override": "force_dual_two_coherent_voices",
                }

        if mode == "dual":
            if primary not in known_set:
                primary = next((x for x in humans if x in known_set), None)
            if secondary not in known_set or secondary == primary:
                secondary = next((x for x in humans if x in known_set and x != primary), None)
            if primary and secondary:
                kept = [primary, secondary]
                participants = [primary, secondary]
                background = [x for x in known if x not in set(kept)]
                return mode, primary, secondary, kept, participants, background, decision

        if mode == "triple":
            ordered = []
            for value in [primary, secondary, *humans]:
                if value in known_set and value not in ordered:
                    ordered.append(value)
            if len(ordered) >= 3:
                primary, secondary = ordered[0], ordered[1]
                kept = ordered[:3]
                participants = kept[:]
                background = [x for x in known if x not in set(kept)]
                return "triple", primary, secondary, kept, participants, background, decision

        if mode == "single" and primary in known_set:
            return "single", primary, None, [primary], [primary], [x for x in known if x != primary], decision

        if mode == "crowd" and primary in known_set:
            return "crowd", primary, None, [primary], humans or [primary], [x for x in known if x not in set(humans)], decision

    # Luna failure/invalid output: exact Gold V7 local classification.
    return (
        local_mode, local_primary, local_secondary, local_kept,
        local_participants, local_background, decision,
    )



def _make_speaker_enhanced_audio(audio_path: Path, clip_index: int) -> Path:
    """Create one local speech-focused view for a conditional diarization retry."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    output = TEMP_DIR / f"clip_{int(clip_index):02d}_speaker_retry_enhanced.wav"
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(audio_path),
        "-vn",
        "-af", "highpass=f=70,lowpass=f=12000,dynaudnorm=f=150:g=15:p=0.95:m=12",
        "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le",
        str(output),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size < 1024:
        raise RuntimeError(
            "Speaker retry audio hazırlanamadı. "
            + (completed.stderr.strip() or "ffmpeg başarısız")
        )
    return output


def _speaker_reference_ranges(
    segments: list[dict[str, Any]],
    speaker: str,
) -> list[tuple[float, float]]:
    """Pick clean 2-10s reference speech, trimming cross-speaker boundaries."""
    candidates: list[tuple[float, float, float]] = []
    for index, seg in enumerate(segments):
        if str(seg.get("speaker", "")) != speaker:
            continue
        text = str(seg.get("text", ""))
        if _looks_like_noise_text(text):
            continue
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start) or start)
        words = int(seg.get("word_count", len(text.split())) or 0)
        if words < 2:
            continue

        previous = segments[index - 1] if index > 0 else None
        following = segments[index + 1] if index + 1 < len(segments) else None
        if previous and str(previous.get("speaker", "")) != speaker:
            start += 0.12
        if following and str(following.get("speaker", "")) != speaker:
            end -= 0.12
        duration = max(0.0, end - start)
        if duration < 0.80:
            continue
        candidates.append((start, end, duration))

    candidates.sort(key=lambda row: row[2], reverse=True)
    picked: list[tuple[float, float]] = []
    total = 0.0
    for start, end, duration in candidates:
        remaining = DIARIZATION_REFERENCE_MAX_SECONDS - total
        if remaining <= 0.05:
            break
        take = min(duration, remaining)
        if duration > take:
            center = (start + end) / 2.0
            start = max(start, center - take / 2.0)
            end = start + take
        picked.append((start, end))
        total += max(0.0, end - start)
        if total >= DIARIZATION_REFERENCE_MIN_SECONDS:
            break
    if total + 1e-6 < DIARIZATION_REFERENCE_MIN_SECONDS:
        return []
    return picked


def _render_speaker_reference(
    audio_path: Path,
    segments: list[dict[str, Any]],
    speaker: str,
    clip_index: int,
    label: str,
) -> Path | None:
    ranges = _speaker_reference_ranges(segments, speaker)
    if not ranges:
        return None
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    output = TEMP_DIR / f"clip_{int(clip_index):02d}_speaker_ref_{label}.wav"
    filters: list[str] = []
    labels: list[str] = []
    for idx, (start, end) in enumerate(ranges):
        tag = f"sr{idx}"
        filters.append(
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},"
            f"asetpts=PTS-STARTPTS,aresample=48000,"
            f"aformat=sample_fmts=s16:channel_layouts=mono[{tag}]"
        )
        labels.append(f"[{tag}]")
    if len(labels) == 1:
        filters.append(f"{labels[0]}anull[aout]")
    else:
        filters.append("".join(labels) + f"concat=n={len(labels)}:v=0:a=1[aout]")
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(audio_path),
        "-filter_complex", ";".join(filters),
        "-map", "[aout]",
        "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le",
        str(output),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size < 1024:
        return None
    duration = _probe_duration(output)
    if duration < DIARIZATION_REFERENCE_MIN_SECONDS - 0.05 or duration > 10.05:
        return None
    return output


def _audio_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def _scan_pass(
    audio_path: Path,
    duration: float,
    *,
    chunking: str,
    vad_threshold: float | None = None,
    vad_prefix_padding_ms: int | None = None,
    vad_silence_ms: int | None = None,
    known_speaker_names: list[str] | None = None,
    known_speaker_references: list[str] | None = None,
) -> dict[str, Any]:
    diarized = _run_diarization(
        audio_path,
        chunking=chunking,
        vad_threshold=vad_threshold,
        vad_prefix_padding_ms=vad_prefix_padding_ms,
        vad_silence_ms=vad_silence_ms,
        known_speaker_names=known_speaker_names,
        known_speaker_references=known_speaker_references,
    )
    segments = _normalize_segments(diarized, duration)
    stats = _speaker_stats(segments)
    if segments and stats:
        # A single raw speaker ID gives Luna no additional evidence to discover
        # a second voice, so do not waste a model call. Multi-ID evidence is the
        # only case where the participant-vs-crowd role judge adds value.
        distinct_ids = [str(row.get("speaker", "")) for row in stats if str(row.get("speaker", ""))]
        if len(distinct_ids) <= 1:
            mode, primary, secondary, kept, participants, background = _classify(stats)
            role = {
                "status": "deterministic_single_id",
                "mode": mode,
                "primary_speaker": primary,
                "secondary_speaker": secondary,
                "human_speakers": participants,
                "background_speakers": background,
                "confidence": 0.0,
                "reason": "Only one diarization speaker ID; role model skipped until multi-speaker evidence exists.",
            }
        else:
            mode, primary, secondary, kept, participants, background, role = _classify_with_luna(segments, stats)
    else:
        mode, primary, secondary, kept, participants, background, role = (
            "single", None, None, [], [], [], {}
        )
    return {
        "mode": mode,
        "primary": primary,
        "secondary": secondary,
        "kept": kept,
        "participants": participants,
        "background": background,
        "role": role,
        "segments": segments,
        "stats": stats,
    }


def _mode_count(mode: str) -> int:
    if mode == "dual":
        return 2
    if mode == "triple":
        return 3
    return 1


def _role_confidence(scan: dict[str, Any]) -> float:
    role = scan.get("role", {}) if isinstance(scan.get("role"), dict) else {}
    return max(0.0, min(1.0, float(role.get("confidence", 0.0) or 0.0)))


# Speaker count confidence and boundary quality are different signals. A model
# can be 99% sure that TWO people are talking while still chopping one sentence
# into A/B/A/B 50 ms fragments. High count confidence must never skip a retry
# when the raw turn geometry is physically implausible.
BOUNDARY_EARLY_ACCEPT_MIN = 0.78
BOUNDARY_NAMING_MIN = 0.68
BOUNDARY_TINY_SECONDS = 0.12
BOUNDARY_SHORT_SECONDS = 0.28
BOUNDARY_PINGPONG_WINDOW_SECONDS = 1.10
BOUNDARY_MAX_WORDS_PER_SECOND = 14.0


def _speaker_boundary_audit(scan: dict[str, Any]) -> dict[str, Any]:
    segments = [dict(x) for x in (scan.get("segments") or []) if isinstance(x, dict)]
    if not segments:
        return {
            "quality": 0.0, "suspicious": True, "hard_failure": True,
            "tiny_segments": 0, "short_segments": 0, "impossible_rates": 0,
            "rapid_switches": 0, "ping_pong": 0, "reasons": ["no_segments"],
        }

    tiny = 0
    short = 0
    impossible = 0
    rapid_switches = 0
    ping_pong = 0
    reasons: list[str] = []

    for index, seg in enumerate(segments):
        duration = max(0.0, float(seg.get("duration", 0.0) or 0.0))
        words = max(0, int(seg.get("word_count", 0) or 0))
        if words and duration <= BOUNDARY_TINY_SECONDS:
            tiny += 1
        elif words and duration <= BOUNDARY_SHORT_SECONDS:
            short += 1
        if words and duration > 0.0 and words / max(duration, 0.02) > BOUNDARY_MAX_WORDS_PER_SECOND:
            impossible += 1

        if index > 0:
            previous = segments[index - 1]
            if str(previous.get("speaker", "")) != str(seg.get("speaker", "")):
                previous_duration = max(0.0, float(previous.get("duration", 0.0) or 0.0))
                if min(previous_duration, duration) <= 0.35:
                    rapid_switches += 1

        if index >= 2:
            a = segments[index - 2]
            b = segments[index - 1]
            c = seg
            sa = str(a.get("speaker", "")); sb = str(b.get("speaker", "")); sc = str(c.get("speaker", ""))
            if sa and sb and sa == sc and sa != sb:
                window = float(c.get("end", 0.0) or 0.0) - float(a.get("start", 0.0) or 0.0)
                if 0.0 < window <= BOUNDARY_PINGPONG_WINDOW_SECONDS:
                    ping_pong += 1

    if tiny:
        reasons.append(f"tiny_segments={tiny}")
    if impossible:
        reasons.append(f"impossible_speech_rate={impossible}")
    if rapid_switches >= 2:
        reasons.append(f"rapid_switches={rapid_switches}")
    if ping_pong:
        reasons.append(f"ping_pong={ping_pong}")

    # Conservative deterministic score. It is only a retry/naming gate; it never
    # changes words or timestamps.
    penalty = (0.24 * tiny) + (0.09 * short) + (0.22 * impossible) + (0.08 * rapid_switches) + (0.20 * ping_pong)
    quality = max(0.0, min(1.0, 1.0 - penalty))
    hard_failure = bool(tiny or impossible or ping_pong)
    suspicious = hard_failure or rapid_switches >= 2 or quality < BOUNDARY_EARLY_ACCEPT_MIN
    return {
        "quality": round(quality, 4),
        "suspicious": bool(suspicious),
        "hard_failure": bool(hard_failure),
        "tiny_segments": tiny,
        "short_segments": short,
        "impossible_rates": impossible,
        "rapid_switches": rapid_switches,
        "ping_pong": ping_pong,
        "reasons": reasons,
    }


def _attach_boundary_audit(scan: dict[str, Any]) -> dict[str, Any]:
    audit = _speaker_boundary_audit(scan)
    scan["speaker_boundary_quality"] = float(audit["quality"])
    scan["speaker_boundary_audit"] = audit
    return audit


def _speaker_ensemble_scan(
    audio_path: Path,
    duration: float,
    clip_index: int,
) -> dict[str, Any]:
    """Recall-first speaker scan with conditional evidence escalation.

    Cost policy:
    - pass 1 always: raw 48 kHz + OpenAI auto chunking/loudness normalization
    - pass 2 only if pass 1 is single/uncertain: speech-focused audio + sensitive VAD
    - anchored pass only if either pass plausibly finds 2/3 speakers and can provide
      2-10 second clean references for every participant
    """
    audit: list[dict[str, Any]] = []

    primary = _scan_pass(audio_path, duration, chunking="auto")
    primary_boundary = _attach_boundary_audit(primary)
    audit.append({
        "name": "primary_auto_raw48k",
        "mode": primary["mode"],
        "participant_count": len(primary.get("participants", []) or []),
        "role_confidence": round(_role_confidence(primary), 4),
        "boundary_quality": primary_boundary["quality"],
        "boundary_suspicious": primary_boundary["suspicious"],
        "boundary_reasons": primary_boundary["reasons"],
        "speaker_ids": [str(x.get("speaker", "")) for x in primary.get("stats", [])],
    })

    # Count confidence alone is NOT enough. Only a high-confidence multi-speaker
    # result with physically sane turn boundaries may stop after pass 1.
    if (
        primary["mode"] in {"dual", "triple"}
        and _role_confidence(primary) >= 0.90
        and not bool(primary_boundary["suspicious"])
        and float(primary_boundary["quality"]) >= BOUNDARY_EARLY_ACCEPT_MIN
    ):
        primary["speaker_count_confidence"] = _role_confidence(primary)
        primary["ensemble_audit"] = audit
        primary["selection_reason"] = "primary_high_confidence_multi_clean_boundaries"
        return primary

    retry: dict[str, Any] | None = None
    enhanced: Path | None = None
    try:
        enhanced = _make_speaker_enhanced_audio(audio_path, clip_index)
        retry = _scan_pass(
            enhanced,
            duration,
            chunking="server_vad",
            vad_threshold=DIARIZATION_RETRY_VAD_THRESHOLD,
            vad_prefix_padding_ms=DIARIZATION_RETRY_PREFIX_PADDING_MS,
            vad_silence_ms=DIARIZATION_RETRY_SILENCE_MS,
        )
        retry_boundary = _attach_boundary_audit(retry)
        audit.append({
            "name": "retry_sensitive_enhanced48k",
            "mode": retry["mode"],
            "participant_count": len(retry.get("participants", []) or []),
            "role_confidence": round(_role_confidence(retry), 4),
            "boundary_quality": retry_boundary["quality"],
            "boundary_suspicious": retry_boundary["suspicious"],
            "boundary_reasons": retry_boundary["reasons"],
            "speaker_ids": [str(x.get("speaker", "")) for x in retry.get("stats", [])],
        })
    except Exception as error:
        audit.append({"name": "retry_sensitive_enhanced48k", "error": str(error)[:240]})

    candidates = [scan for scan in (primary, retry) if scan and scan.get("mode") in {"dual", "triple"}]
    if not candidates:
        primary["speaker_count_confidence"] = 0.0
        primary["ensemble_audit"] = audit
        primary["selection_reason"] = "both_passes_single_or_unresolved"
        return primary

    # Prefer the multi-speaker pass with stronger role confidence, then more
    # participant speech. This pass supplies candidate reference windows.
    def _candidate_score(scan: dict[str, Any]) -> tuple[float, float, float]:
        participant_set = set(scan.get("participants", []) or [])
        speech = sum(
            float(row.get("speaking_seconds", 0.0) or 0.0)
            for row in scan.get("stats", [])
            if str(row.get("speaker", "")) in participant_set
        )
        # Prefer clean boundaries before tiny differences in model confidence.
        return (
            float(scan.get("speaker_boundary_quality", 0.0) or 0.0),
            _role_confidence(scan),
            speech,
        )

    candidate = max(candidates, key=_candidate_score)
    participants = [str(x) for x in candidate.get("participants", []) if str(x)]
    expected_count = _mode_count(str(candidate.get("mode", "single")))
    participants = participants[:expected_count]

    names: list[str] = []
    references: list[str] = []
    reference_paths: list[str] = []
    if len(participants) == expected_count and expected_count in {2, 3}:
        for idx, speaker in enumerate(participants):
            label = chr(ord("A") + idx)
            ref_path = _render_speaker_reference(
                audio_path,
                candidate.get("segments", []),
                speaker,
                clip_index,
                label,
            )
            if ref_path is None:
                names = []
                references = []
                reference_paths = []
                break
            names.append(label)
            references.append(_audio_data_url(ref_path))
            reference_paths.append(str(ref_path))

    if names and len(names) == expected_count:
        try:
            anchored = _scan_pass(
                audio_path,
                duration,
                chunking="auto",
                known_speaker_names=names,
                known_speaker_references=references,
            )
            anchored_count = _mode_count(str(anchored.get("mode", "single")))
            anchored_boundary = _attach_boundary_audit(anchored)
            audit.append({
                "name": "anchored_known_speaker_refs",
                "mode": anchored["mode"],
                "participant_count": len(anchored.get("participants", []) or []),
                "role_confidence": round(_role_confidence(anchored), 4),
                "boundary_quality": anchored_boundary["quality"],
                "boundary_suspicious": anchored_boundary["suspicious"],
                "boundary_reasons": anchored_boundary["reasons"],
                "reference_paths": reference_paths,
                "speaker_ids": [str(x.get("speaker", "")) for x in anchored.get("stats", [])],
            })
            if (
                anchored_count == expected_count
                and anchored.get("mode") in {"dual", "triple"}
                and float(anchored_boundary["quality"]) >= BOUNDARY_NAMING_MIN
            ):
                anchored["speaker_count_confidence"] = max(
                    SPEAKER_COUNT_CONFIRM_CONFIDENCE,
                    _role_confidence(anchored),
                )
                anchored["ensemble_audit"] = audit
                anchored["selection_reason"] = "anchored_reference_confirmed_multi"
                return anchored
        except Exception as error:
            audit.append({"name": "anchored_known_speaker_refs", "error": str(error)[:240]})

    # If two independent unanchored passes agree on the same multi-speaker
    # count, that count itself is strong evidence even if reference extraction
    # was impossible (e.g. a short but obvious second turn).
    if retry and primary.get("mode") == retry.get("mode") and primary.get("mode") in {"dual", "triple"}:
        candidate["speaker_count_confidence"] = max(
            SPEAKER_COUNT_CONFIRM_CONFIDENCE,
            _role_confidence(candidate),
        )
        candidate["ensemble_audit"] = audit
        candidate["selection_reason"] = "two_pass_count_agreement_boundary_ranked"
        return candidate

    # A single alternate multi-speaker hit is not enough to force a prompt.
    # Keep it only if Luna itself is already highly confident; otherwise plain
    # captions are safer than a false A/B split.
    if _role_confidence(candidate) >= 0.90:
        candidate["speaker_count_confidence"] = _role_confidence(candidate)
        candidate["ensemble_audit"] = audit
        candidate["selection_reason"] = "single_multi_pass_but_role_high_confidence"
        return candidate

    primary["speaker_count_confidence"] = 0.0
    primary["ensemble_audit"] = audit
    primary["selection_reason"] = "multi_not_confirmed_plain"
    return primary

def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _segment_distance(midpoint: float, segment: dict[str, Any]) -> float:
    start = float(segment["start"])
    end = float(segment["end"])
    if start <= midpoint <= end:
        return 0.0
    return min(abs(midpoint - start), abs(midpoint - end))


def _best_segment(word: dict[str, Any], segments: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    """Timing-only evidence.  Never changes word start/end."""
    start = float(word["edited_start"])
    end = float(word["edited_end"])
    duration = max(0.04, end - start)
    midpoint = (start + end) / 2.0

    best: dict[str, Any] | None = None
    best_overlap = 0.0
    for segment in segments:
        amount = _overlap(start, end, float(segment["start"]), float(segment["end"]))
        if amount > best_overlap:
            best_overlap = amount
            best = segment

    if best is not None and best_overlap > 0:
        return best, min(TIMING_ONLY_MAX_CONFIDENCE, best_overlap / duration)

    if not segments:
        return None, 0.0
    nearest = min(segments, key=lambda item: _segment_distance(midpoint, item))
    distance = _segment_distance(midpoint, nearest)
    if distance <= NEAR_SEGMENT_TOLERANCE:
        confidence = max(0.10, 1.0 - (distance / NEAR_SEGMENT_TOLERANCE)) * 0.42
        return nearest, confidence
    return None, 0.0


def _norm_token(value: str) -> str:
    text = str(value).casefold().replace("’", "'")
    text = re.sub(r"[^\w']+", "", text, flags=re.UNICODE)
    return text.strip("'")


def _diarized_token_stream(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand diarized segment text into a monotonic token stream.

    The diarizer owns only speaker labels.  Per-token times are *estimates* used
    as secondary evidence; final caption timing always comes from Whisper words.
    """
    output: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments):
        raw_tokens = [token for token in str(segment.get("text", "")).split() if _norm_token(token)]
        if not raw_tokens:
            continue
        start = float(segment["start"])
        end = float(segment["end"])
        span = max(0.06, end - start)
        count = len(raw_tokens)
        for token_index, raw in enumerate(raw_tokens):
            frac0 = token_index / count
            frac1 = (token_index + 1) / count
            output.append(
                {
                    "token": _norm_token(raw),
                    "speaker": str(segment.get("speaker", "")),
                    "segment_index": segment_index,
                    "segment_id": str(segment.get("id", f"seg_{segment_index:03d}")),
                    "segment_start": start,
                    "segment_end": end,
                    "approx_start": start + span * frac0,
                    "approx_end": start + span * frac1,
                }
            )
    return output


def _text_speaker_evidence(
    words: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Align accurate transcript words to diarized transcript text.

    Sequence alignment fixes the classic failure where diarization boundaries
    drift by a few hundred milliseconds and a word is assigned to the wrong
    person.  We use only exact normalized token blocks here: if text evidence is
    not clean, timing/continuity decides instead of guessing.
    """
    diar_tokens = _diarized_token_stream(segments)
    accurate_tokens = [_norm_token(item.get("word", "")) for item in words]
    diar_values = [str(item["token"]) for item in diar_tokens]

    evidence: dict[int, dict[str, Any]] = {}
    matcher = SequenceMatcher(a=accurate_tokens, b=diar_values, autojunk=False)
    matched = 0
    for block in matcher.get_matching_blocks():
        if block.size <= 0:
            continue
        for offset in range(block.size):
            wi = block.a + offset
            di = block.b + offset
            if wi >= len(words) or di >= len(diar_tokens):
                continue
            token = diar_tokens[di]
            word_mid = (
                float(words[wi].get("edited_start", 0.0))
                + float(words[wi].get("edited_end", 0.0))
            ) / 2.0
            token_mid = (float(token["approx_start"]) + float(token["approx_end"])) / 2.0
            drift = abs(word_mid - token_mid)
            # Repeated words ("bro", "no", "yeah") can make pure text
            # sequence alignment choose the wrong occurrence.  Final-audio time
            # is a guardrail: a lexical match far away in time is rejected.
            if drift > TEXT_ALIGNMENT_MAX_TIME_DRIFT:
                continue
            text_conf = TEXT_MATCH_CONFIDENCE
            if drift > TEXT_ALIGNMENT_SOFT_TIME_DRIFT:
                text_conf = 0.80
            evidence[wi] = {
                "speaker": str(token["speaker"]),
                "confidence": text_conf,
                "segment_index": int(token["segment_index"]),
                "segment_id": str(token["segment_id"]),
                "source": "diarized_text_alignment",
                "time_drift": round(drift, 3),
            }
            matched += 1

    meaningful = sum(1 for value in accurate_tokens if value)
    ratio = matched / max(1, meaningful)
    return evidence, {
        "matched_words": matched,
        "accurate_word_count": meaningful,
        "diarized_token_count": len(diar_tokens),
        "text_alignment_ratio": round(ratio, 4),
    }


def _provisional_assignments(
    words: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    text_map, alignment = _text_speaker_evidence(words, segments)
    provisional: list[dict[str, Any]] = []
    agreement = 0
    disagreement = 0

    for index, raw in enumerate(words):
        word = dict(raw)
        text_ev = text_map.get(index)
        timing_segment, timing_conf = _best_segment(word, segments)
        timing_speaker = str(timing_segment.get("speaker", "")) if timing_segment else ""
        text_speaker = str(text_ev.get("speaker", "")) if text_ev else ""

        if text_speaker and timing_speaker and text_speaker == timing_speaker:
            speaker = text_speaker
            confidence = min(0.99, max(float(text_ev["confidence"]), timing_conf) + 0.05)
            source = "text+timing_agree"
            agreement += 1
        elif text_speaker:
            # Text sequence is monotonic and substantially more robust around
            # diarization boundary drift than raw timestamp overlap.
            speaker = text_speaker
            confidence = float(text_ev["confidence"])
            source = "text_alignment"
            if timing_speaker and timing_speaker != text_speaker:
                disagreement += 1
        elif timing_speaker:
            speaker = timing_speaker
            confidence = float(timing_conf)
            source = "timing_only"
        else:
            speaker = ""
            confidence = 0.0
            source = "unresolved"

        provisional.append(
            {
                **word,
                "speaker_candidate": speaker,
                "speaker_candidate_confidence": round(confidence, 3),
                "speaker_evidence_source": source,
            }
        )

    alignment["text_timing_agreements"] = agreement
    alignment["text_timing_disagreements"] = disagreement
    return provisional, alignment




def _run_metrics(words: list[dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    subset = words[start:end]
    if not subset:
        return {"words": 0, "duration": 0.0, "avg_confidence": 0.0, "speaker": ""}
    begin = float(subset[0]["edited_start"])
    finish = float(subset[-1]["edited_end"])
    confidences = [float(item.get("speaker_candidate_confidence", 0.0)) for item in subset]
    return {
        "words": len(subset),
        "duration": max(0.0, finish - begin),
        "avg_confidence": sum(confidences) / max(1, len(confidences)),
        "speaker": str(subset[0].get("speaker_candidate", "")),
    }


def _speaker_runs(words: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    if not words:
        return []
    runs: list[tuple[int, int, str]] = []
    start = 0
    current = str(words[0].get("speaker_candidate", ""))
    for index in range(1, len(words)):
        speaker = str(words[index].get("speaker_candidate", ""))
        if speaker != current:
            runs.append((start, index, current))
            start = index
            current = speaker
    runs.append((start, len(words), current))
    return runs


def _stabilize_speaker_turns(words: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove impossible word-by-word speaker ping-pong without changing text/times."""
    result = [dict(item) for item in words]
    absorbed = 0

    # Unknown words may inherit a speaker only across a LOCAL boundary.  V4
    # used the nearest non-empty neighbor regardless of temporal distance; a
    # multi-second VAD hole therefore leaked the previous person's real name
    # onto quiet speech from someone else.  Preserve UNKNOWN across large gaps.
    for index, item in enumerate(result):
        if str(item.get("speaker_candidate", "")):
            continue

        left_index = next(
            (j for j in range(index - 1, -1, -1)
             if str(result[j].get("speaker_candidate", ""))),
            None,
        )
        right_index = next(
            (j for j in range(index + 1, len(result))
             if str(result[j].get("speaker_candidate", ""))),
            None,
        )

        left = str(result[left_index].get("speaker_candidate", "")) if left_index is not None else ""
        right = str(result[right_index].get("speaker_candidate", "")) if right_index is not None else ""
        start = float(item.get("edited_start", 0.0) or 0.0)
        end = float(item.get("edited_end", start) or start)
        left_gap = (
            max(0.0, start - float(result[left_index].get("edited_end", start) or start))
            if left_index is not None else float("inf")
        )
        right_gap = (
            max(0.0, float(result[right_index].get("edited_start", end) or end) - end)
            if right_index is not None else float("inf")
        )

        chosen = ""
        evidence_gap = float("inf")
        if left and right and left == right:
            # Same speaker on both sides: bridge only if BOTH edges are local.
            if max(left_gap, right_gap) <= MAX_CONTINUITY_FILL_GAP_SECONDS:
                chosen = left
                evidence_gap = max(left_gap, right_gap)
        else:
            # One-sided continuation is valid only when the measured neighbor
            # is close enough to be the same utterance/turn.
            if left and left_gap <= MAX_CONTINUITY_FILL_GAP_SECONDS and left_gap <= right_gap:
                chosen = left
                evidence_gap = left_gap
            elif right and right_gap <= MAX_CONTINUITY_FILL_GAP_SECONDS:
                chosen = right
                evidence_gap = right_gap

        if chosen:
            item["speaker_candidate"] = chosen
            item["speaker_candidate_confidence"] = min(
                0.56, float(item.get("speaker_candidate_confidence", 0.0)) + 0.34
            )
            item["speaker_evidence_source"] = "continuity_fill_local"
            item["speaker_evidence_gap"] = round(float(evidence_gap), 3)
        else:
            item["speaker_evidence_source"] = "unresolved_diarization_gap"
            item["speaker_candidate_confidence"] = 0.0

    # Repeatedly absorb tiny one-word speaker flips.  A real speaker switch must
    # form a coherent turn, not one timing-jittered token inside a sentence.
    for _ in range(3):
        changed = False
        runs = _speaker_runs(result)
        for run_index, (start, end, speaker) in enumerate(runs):
            metrics = _run_metrics(result, start, end)
            if not speaker:
                # A real evidence gap is not a speaker flip. Never invent a
                # human identity merely to make the turn sequence prettier.
                continue
            if metrics["words"] != 1:
                continue
            if metrics["duration"] > MAX_SINGLE_WORD_FLIP_SECONDS:
                continue
            left_speaker = runs[run_index - 1][2] if run_index > 0 else ""
            right_speaker = runs[run_index + 1][2] if run_index + 1 < len(runs) else ""
            replacement = ""
            if left_speaker and right_speaker and left_speaker == right_speaker:
                replacement = left_speaker
            # V5: one-sided continuity is not enough to rewrite an observed
            # speaker turn, especially beside an explicit UNKNOWN evidence gap.
            # Keep the measured speaker unless BOTH neighboring turns agree.
            if replacement and replacement != speaker:
                result[start]["speaker_candidate"] = replacement
                result[start]["speaker_candidate_confidence"] = min(
                    0.66, float(result[start].get("speaker_candidate_confidence", 0.0))
                )
                result[start]["speaker_evidence_source"] = "turn_hysteresis"
                absorbed += 1
                changed = True
        if not changed:
            break

    # A remaining very short/weak run is not allowed to split a phrase unless
    # it has enough lexical evidence to be a real interjection/turn.
    runs = _speaker_runs(result)
    for run_index, (start, end, speaker) in enumerate(runs):
        metrics = _run_metrics(result, start, end)
        if not speaker:
            continue
        if metrics["words"] >= MIN_STABLE_TURN_WORDS:
            continue
        if metrics["duration"] >= MIN_STABLE_TURN_SECONDS and metrics["avg_confidence"] >= 0.84:
            continue
        if metrics["avg_confidence"] >= 0.90:
            continue
        left_speaker = runs[run_index - 1][2] if run_index > 0 else ""
        right_speaker = runs[run_index + 1][2] if run_index + 1 < len(runs) else ""
        replacement = ""
        if left_speaker and right_speaker and left_speaker == right_speaker:
            replacement = left_speaker
        # Never absorb a measured weak turn into only one neighbor.  A missing
        # neighbor may be a diarization hole, not evidence of continuity.
        if replacement and replacement != speaker:
            for index in range(start, end):
                result[index]["speaker_candidate"] = replacement
                result[index]["speaker_candidate_confidence"] = min(
                    0.68, float(result[index].get("speaker_candidate_confidence", 0.0))
                )
                result[index]["speaker_evidence_source"] = "weak_turn_absorbed"
            absorbed += end - start

    runs = _speaker_runs(result)
    switches = max(0, len(runs) - 1)
    return result, {
        "absorbed_unstable_words": absorbed,
        "stable_turn_count": len(runs),
        "speaker_switch_count": switches,
        "switches_per_20_words": round(switches * 20.0 / max(1, len(result)), 3),
    }


def _turns_from_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for start, end, speaker in _speaker_runs(words):
        metrics = _run_metrics(words, start, end)
        turns.append(
            {
                "start_index": start,
                "end_index": end,
                "speaker": speaker,
                "word_count": int(metrics["words"]),
                "start": round(float(words[start]["edited_start"]), 3),
                "end": round(float(words[end - 1]["edited_end"]), 3),
                "duration": round(float(metrics["duration"]), 3),
                "avg_confidence": round(float(metrics["avg_confidence"]), 3),
            }
        )
    return turns


def _assign_words(
    words: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    mode: str,
    primary: str | None,
    secondary: str | None,
    background_speakers: list[str] | None = None,
    participant_speakers: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Speaker-aware caption assignment with semantic integrity.

    Rules:
    - Never rewrite/delete transcript words one-by-one because diarization moved.
    - Speaker switches happen at stable turns, not isolated words.
    - Dual/triple mode keeps the complete dialogue; only strongly proven ambient
      turns can be omitted.
    - If speaker separation is not trustworthy, fall back to one unlabeled lane
      rather than publishing wrong names on mixed speakers.
    """
    primary = str(primary or "")
    secondary = str(secondary or "")
    participants = []
    for value in (participant_speakers or []):
        speaker = str(value or "")
        if speaker and speaker not in participants:
            participants.append(speaker)
    for value in (primary, secondary):
        if value and value not in participants:
            participants.append(value)
    if mode == "triple":
        participants = participants[:3]
    elif mode == "dual":
        participants = participants[:2]
    background = {str(value) for value in (background_speakers or []) if str(value)}

    provisional, alignment = _provisional_assignments(words, segments)
    stabilized, stability = _stabilize_speaker_turns(provisional)
    turns = _turns_from_words(stabilized)

    # Separation quality is intentionally conservative.  A wrong A/B label is
    # worse than one unlabeled but perfectly synchronized caption lane.
    aligned_ratio = float(alignment.get("text_alignment_ratio", 0.0))
    nonempty = [item for item in stabilized if str(item.get("speaker_candidate", ""))]
    confident = [
        item for item in nonempty
        if float(item.get("speaker_candidate_confidence", 0.0)) >= 0.70
    ]
    confidence_coverage = len(confident) / max(1, len(stabilized))
    switch_rate = float(stability.get("switches_per_20_words", 0.0))
    switch_factor = max(0.0, 1.0 - max(0.0, switch_rate - 2.0) / 8.0)
    separation_quality = (
        0.52 * aligned_ratio
        + 0.33 * confidence_coverage
        + 0.15 * switch_factor
    )

    effective_mode = mode
    low_confidence_dual = False
    stable_counts: dict[str, int] = {}
    for item in stabilized:
        speaker = str(item.get("speaker_candidate", ""))
        if speaker:
            stable_counts[speaker] = stable_counts.get(speaker, 0) + 1
    dual_has_both_speakers = (
        bool(primary)
        and bool(secondary)
        and stable_counts.get(primary, 0) >= 2
        and stable_counts.get(secondary, 0) >= 2
    )
    triple_has_all_speakers = (
        len(participants) == 3
        and all(stable_counts.get(speaker, 0) >= 2 for speaker in participants)
    )
    if mode in {"dual", "triple"} and (
        (mode == "dual" and not dual_has_both_speakers)
        or (mode == "triple" and not triple_has_all_speakers)
        or separation_quality < MIN_DUAL_SEPARATION_QUALITY
        or aligned_ratio < MIN_DUAL_TEXT_ALIGNMENT
        or switch_rate > MAX_DUAL_SWITCHES_PER_20_WORDS
    ):
        effective_mode = "unresolved"
        low_confidence_dual = True

    output: list[dict[str, Any]] = []
    suppressed = 0
    uncertain = 0

    # Decide suppression at TURN level so the transcript cannot lose random
    # middle words and become semantically broken.
    suppress_indexes: set[int] = set()
    for turn in turns:
        speaker = str(turn["speaker"])
        avg_conf = float(turn["avg_confidence"])
        word_count = int(turn["word_count"])
        duration = float(turn["duration"])
        start_index = int(turn["start_index"])
        end_index = int(turn["end_index"])

        strong_turn = (
            word_count >= 2
            and duration >= 0.42
            and avg_conf >= 0.76
        ) or (
            word_count >= 3 and avg_conf >= 0.70
        )

        if effective_mode in {"dual", "triple"}:
            if speaker in background and strong_turn:
                suppress_indexes.update(range(start_index, end_index))
        elif effective_mode in {"single", "crowd"}:
            if primary and speaker and speaker != primary and strong_turn:
                suppress_indexes.update(range(start_index, end_index))
        # unresolved: suppress nothing. Meaning wins over identity guessing.

    # Dual dialogue must preserve nearly all lexical content.  If ambient
    # filtering would remove too much, cancel suppression instead of producing
    # a broken sentence.
    if effective_mode in {"dual", "triple"}:
        projected_coverage = (len(stabilized) - len(suppress_indexes)) / max(1, len(stabilized))
        if projected_coverage < MIN_DUAL_MEANING_COVERAGE:
            suppress_indexes.clear()

    previous_role = "main"
    previous_speaker = primary
    for index, raw in enumerate(stabilized):
        if index in suppress_indexes:
            suppressed += 1
            continue
        word = dict(raw)
        detected = str(word.get("speaker_candidate", ""))
        confidence = float(word.get("speaker_candidate_confidence", 0.0))

        if effective_mode in {"dual", "triple"}:
            label_map = {speaker: chr(ord("A") + idx) for idx, speaker in enumerate(participants)}
            if detected in label_map:
                label = label_map[detected]
                role = "main" if label == "A" else ("secondary" if label == "B" else "tertiary")
            else:
                # V5 fail-closed identity: unresolved speaker evidence remains
                # unresolved.  Do NOT copy the previous raw speaker/name across
                # a diarization hole. Styling may keep the previous role, but
                # no human name is emitted without local WHO evidence.
                detected = ""
                label = ""
                role = previous_role if previous_role in {"main", "secondary", "tertiary"} else "main"
                confidence = 0.0
                uncertain += 1
        elif effective_mode == "unresolved":
            role = "main"
            label = ""
            if not detected:
                uncertain += 1
        else:
            role = "main"
            label = ""
            if detected != primary:
                uncertain += 1
            detected = primary or detected

        word.update(
            {
                "speaker_raw": detected,
                "speaker_role": role,
                "speaker_label": label,
                "speaker_confidence": round(confidence, 3),
            }
        )
        output.append(word)
        previous_role = role
        previous_speaker = detected or previous_speaker

    original_count = max(1, len(words))
    coverage_ratio = len(output) / original_count

    # Absolute semantic safety valve.  A diarization decision may reduce output
    # only by coherent alternate turns.  Anything pathological becomes a full,
    # single unlabeled transcript instead of silently deleting meaning.
    safety_fallback = False
    if len(words) >= 6 and (
        not output
        or (effective_mode in {"dual", "triple"} and coverage_ratio < MIN_DUAL_MEANING_COVERAGE)
        or coverage_ratio < 0.52
    ):
        safety_fallback = True
        effective_mode = "unresolved"
        output = [
            {
                **dict(item),
                "speaker_raw": "",
                "speaker_role": "main",
                "speaker_label": "",
                "speaker_confidence": 0.0,
            }
            for item in words
        ]
        suppressed = 0
        coverage_ratio = 1.0

    return output, {
        "word_count_before": len(words),
        "word_count_after": len(output),
        "coverage_ratio": round(coverage_ratio, 4),
        "meaning_preservation_ratio": round(coverage_ratio, 4),
        "suppressed_words": suppressed,
        "uncertain_words_kept": uncertain,
        "effective_mode": effective_mode,
        "low_confidence_dual_fallback": low_confidence_dual,
        "separation_quality": round(separation_quality, 4),
        "stable_speaker_word_counts": stable_counts,
        "dual_has_both_speakers": dual_has_both_speakers,
        "triple_has_all_speakers": triple_has_all_speakers,
        "text_alignment": alignment,
        "turn_stability": stability,
        "turns": turns,
        "safety_fallback_to_unresolved": safety_fallback,
    }

# ============================================================
# SPEAKER COLOURS (a colour names a VOICE, never a person)
# ============================================================

SPEAKER_COLOR_VERSION = 1
SPEAKER_COLOR_SLOTS = ("A", "B", "C")
SPEAKER_COLOR_NEUTRAL = "neutral"
# A turn earns its voice's colour only with the turn stabilizer's own keep rule
# (a real turn, or a short turn with strong evidence) and a confidence floor.
MIN_COLOR_RUN_CONFIDENCE = 0.60
# Whole-clip reliability: too much uncertain speech or too many colour changes
# means the colours would flicker rather than follow turns -> no speaker colours.
MAX_NEUTRAL_COLOR_SHARE = 0.34
MAX_COLOR_CHANGES_PER_20_WORDS = MAX_DUAL_SWITCHES_PER_20_WORDS


def _stable_color_run(words: int, duration: float, avg_confidence: float) -> bool:
    if avg_confidence < MIN_COLOR_RUN_CONFIDENCE:
        return False
    return (
        words >= MIN_STABLE_TURN_WORDS
        or (duration >= MIN_STABLE_TURN_SECONDS and avg_confidence >= 0.84)
        or avg_confidence >= 0.90
    )


def _assign_speaker_colors(
    words: list[dict[str, Any]],
    *,
    mode: str,
    participants: list[str],
    hard_failure: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Caption colour of every word from the acoustic speaker turns.

    The first voice with a stable turn is colour ``A``, the next ``B``, then
    ``C``, and a voice keeps its colour for the whole clip (A -> B -> A returns
    to A). Words without speaker evidence, and turns too short or weak to
    trust, are ``neutral``: no colour is guessed and one jittered word cannot
    flash another voice's colour. A clip without two reliable voices (single
    speaker, unresolved, boundary hard failure, too much uncertainty or colour
    flicker) gets no speaker colours at all (``""``, the plain caption look).

    Only ``speaker_color`` is written; text, times and speaker ids are untouched.
    """
    output = [dict(word) for word in words]
    voices = list(dict.fromkeys(str(x) for x in (participants or []) if str(x)))[:len(SPEAKER_COLOR_SLOTS)]
    summary: dict[str, Any] = {
        "version": SPEAKER_COLOR_VERSION,
        "engaged": False,
        "reason": "",
        "slots": {},
        "words": len(output),
        "neutral_words": 0,
        "color_changes": 0,
        "demoted_turns": 0,
        "policy": "colour per voice in order of first stable turn; uncertain or weak turns neutral; "
                  "fewer than two reliable voices -> no speaker colours; lanes are layout only",
    }

    def plain(reason: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        for word in output:
            word["speaker_color"] = ""
        summary["reason"] = reason
        return output, summary

    if hard_failure:
        return plain("speaker_boundary_hard_failure")
    if mode not in {"dual", "triple"}:
        return plain(f"mode_{mode or 'unknown'}")
    if len(voices) < 2:
        return plain("fewer_than_two_voices")
    if not output:
        return plain("no_words")

    runs: list[list[Any]] = []          # [start, end, voice]
    for index, word in enumerate(output):
        voice = str(word.get("speaker_raw") or "")
        voice = voice if voice in voices else ""
        if runs and runs[-1][2] == voice:
            runs[-1][1] = index + 1
        else:
            runs.append([index, index + 1, voice])

    trusted: list[str] = [""] * len(output)
    for start, end, voice in runs:
        if not voice:
            continue
        try:
            duration = float(output[end - 1].get("edited_end", 0.0)) - float(output[start].get("edited_start", 0.0))
            confidences = [float(output[i].get("speaker_confidence", 0.0) or 0.0) for i in range(start, end)]
        except (TypeError, ValueError):
            summary["demoted_turns"] += 1
            continue
        if _stable_color_run(end - start, max(0.0, duration), sum(confidences) / len(confidences)):
            for index in range(start, end):
                trusted[index] = voice
        else:
            summary["demoted_turns"] += 1

    slots: dict[str, str] = {}
    for voice in trusted:
        if voice and voice not in slots:
            slots[voice] = SPEAKER_COLOR_SLOTS[len(slots)]
    colors = [slots[voice] if voice else SPEAKER_COLOR_NEUTRAL for voice in trusted]
    neutral = colors.count(SPEAKER_COLOR_NEUTRAL)
    changes = sum(1 for previous, current in zip(colors, colors[1:]) if previous != current)
    summary.update(slots=dict(slots), neutral_words=neutral, color_changes=changes)

    if len(slots) < 2:
        return plain("fewer_than_two_reliable_voices")
    if neutral / len(colors) > MAX_NEUTRAL_COLOR_SHARE:
        return plain("too_much_uncertain_speaker_evidence")
    if changes * 20.0 / len(colors) > MAX_COLOR_CHANGES_PER_20_WORDS:
        return plain("speaker_color_flicker")

    for word, color in zip(output, colors):
        word["speaker_color"] = color
    summary["engaged"] = True
    summary["reason"] = "two_or_more_reliable_voices"
    return output, summary


def _scan_output_path(edited_clip_path: str | Path, clip_index: int) -> Path:
    edited_clip_path = Path(edited_clip_path).resolve()
    directory = SPEAKER_OUTPUT_DIR / _safe_name(edited_clip_path.parent.name)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"clip_{int(clip_index):02d}_speaker_scan_v{SPEAKER_PROFILE_VERSION}.json"


def get_speaker_scan_path(audio_or_clip_path: str | Path, clip_index: int) -> Path:
    return _scan_output_path(audio_or_clip_path, clip_index)



def create_speaker_scan_from_audio(
    audio_path: str | Path,
    clip_index: int,
    reference_path: str | Path | None = None,
) -> Path:
    """Speaker scan from the exact pre-render pacing audio.

    No video render is required. The full selected clip audio is diarized first,
    then Luna-low classifies real participants versus crowd/noise. The acoustic
    turns feed caption colours only; nobody is asked who is speaking.
    """
    audio_path = Path(audio_path).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Pre-render speaker audio bulunamadı: {audio_path}")
    output = _scan_output_path(audio_path, clip_index)
    duration = _probe_duration(audio_path)
    if duration <= 0:
        raise RuntimeError("Pre-render speaker audio süresi geçersiz.")

    diarization_error = ""
    ensemble_audit: list[dict[str, Any]] = []
    speaker_count_confidence = 0.0
    selection_reason = ""
    scan: dict[str, Any] = {}
    try:
        scan = _speaker_ensemble_scan(audio_path, duration, clip_index)
        segments = list(scan.get("segments", []) or [])
        stats = list(scan.get("stats", []) or [])
        mode = str(scan.get("mode", "single"))
        primary = scan.get("primary")
        secondary = scan.get("secondary")
        kept = list(scan.get("kept", []) or [])
        participants = list(scan.get("participants", []) or [])
        background_speakers = list(scan.get("background", []) or [])
        role_decision = dict(scan.get("role", {}) or {})
        ensemble_audit = list(scan.get("ensemble_audit", []) or [])
        speaker_count_confidence = float(scan.get("speaker_count_confidence", 0.0) or 0.0)
        selection_reason = str(scan.get("selection_reason", ""))
    except Exception as error:
        diarization_error = str(error)
        segments, stats, role_decision = [], [], {}
        mode, primary, secondary, kept, participants, background_speakers = "single", None, None, [], [], []

    profile = {
        "version": SPEAKER_PROFILE_VERSION,
        "status": "ok",
        "phase": "pre_render_speaker_scan",
        "timing_basis": "exact_pacing_audio_before_video_encode",
        "diarization_model": DIARIZATION_MODEL,
        "edited_clip_path": str(Path(reference_path).resolve()) if reference_path else "",
        "speaker_audio_path": str(audio_path),
        "clip_index": int(clip_index),
        "clip_duration": round(duration, 3),
        "mode": mode,
        "primary_speaker": primary,
        "secondary_speaker": secondary,
        "kept_speakers": kept,
        "participant_speakers": participants,
        "background_speakers": background_speakers,
        "classification_source": "speaker_ensemble_v11",
        "role_judge": role_decision,
        "speaker_count_confidence": round(speaker_count_confidence, 4),
        "speaker_ensemble": ensemble_audit,
        "speaker_selection_reason": selection_reason,
        "speaker_boundary_quality": float(scan.get("speaker_boundary_quality", 0.0) or 0.0),
        "speaker_boundary_audit": dict(scan.get("speaker_boundary_audit", {}) or {}),
        "display_labels": {},
        "speaker_stats": stats,
        "segments": segments,
        "words": [],
        "assignment": {},
        "diarization_status": "ok" if segments else "fallback_single",
        "diarization_error": diarization_error,
        "policy": {
            "speaker_algorithm": "48k raw auto diarization -> conditional sensitive retry -> conditional known-speaker anchored confirmation -> Gold V7 assignment",
            "render_order": "audio scan -> video render (no identity prompt)",
            "speaker_colors": "acoustic turns colour voices on the final profile; no person identity",
            "primary_chunking": "auto",
            "retry_vad_threshold": DIARIZATION_RETRY_VAD_THRESHOLD,
            "known_speaker_anchor": "2-10s references only after plausible multi-speaker evidence",
            "dual_seconds_gate": MIN_DUAL_SECONDS,
            "dual_share_gate": MIN_DUAL_SHARE,
            "crowd": "ambient crowd/laughter/cheering is not participant #3",
        },
    }
    _write_json(output, profile)

    return output


def create_speaker_scan(
    edited_clip_path: str | Path,
    clip_index: int,
) -> Path:
    """Fast whole-clip speaker inventory before any caption text/timing work.

    This is the correctness boundary: one diarization request gets the entire
    final edited clip by itself.  No other API request runs concurrently with
    it. Uncertain speaker structure never blocks plain captions.
    """
    edited_clip_path = Path(edited_clip_path).resolve()
    output = _scan_output_path(edited_clip_path, clip_index)
    audio_path: Path | None = None
    try:
        duration = _probe_duration(edited_clip_path)
        if duration <= 0:
            raise RuntimeError("Edited clip süresi geçersiz.")
        audio_path = _extract_audio(edited_clip_path, clip_index)

        diarization_error = ""
        ensemble_audit: list[dict[str, Any]] = []
        speaker_count_confidence = 0.0
        selection_reason = ""
        scan: dict[str, Any] = {}
        try:
            scan = _speaker_ensemble_scan(audio_path, duration, clip_index)
            segments = list(scan.get("segments", []) or [])
            stats = list(scan.get("stats", []) or [])
            mode = str(scan.get("mode", "single"))
            primary = scan.get("primary")
            secondary = scan.get("secondary")
            kept = list(scan.get("kept", []) or [])
            participants = list(scan.get("participants", []) or [])
            background_speakers = list(scan.get("background", []) or [])
            role_decision = dict(scan.get("role", {}) or {})
            ensemble_audit = list(scan.get("ensemble_audit", []) or [])
            speaker_count_confidence = float(scan.get("speaker_count_confidence", 0.0) or 0.0)
            selection_reason = str(scan.get("selection_reason", ""))
        except Exception as error:
            diarization_error = str(error)
            segments, stats, role_decision = [], [], {}
            mode, primary, secondary, kept, participants, background_speakers = "single", None, None, [], [], []

        profile = {
            "version": SPEAKER_PROFILE_VERSION,
            "status": "ok",
            "phase": "speaker_scan",
            "timing_basis": "pending_final_clip_words",
            "diarization_model": DIARIZATION_MODEL,
            "edited_clip_path": str(edited_clip_path),
            "speaker_audio_path": str(audio_path),
            "clip_index": int(clip_index),
            "clip_duration": round(duration, 3),
            "mode": mode,
            "primary_speaker": primary,
            "secondary_speaker": secondary,
            "kept_speakers": kept,
            "participant_speakers": participants,
            "background_speakers": background_speakers,
            "classification_source": "speaker_ensemble_v11",
            "role_judge": role_decision,
            "speaker_count_confidence": round(speaker_count_confidence, 4),
            "speaker_ensemble": ensemble_audit,
            "speaker_selection_reason": selection_reason,
            "speaker_boundary_quality": float(scan.get("speaker_boundary_quality", 0.0) or 0.0),
            "speaker_boundary_audit": dict(scan.get("speaker_boundary_audit", {}) or {}),
            "display_labels": {},
            "speaker_stats": stats,
            "segments": segments,
            "words": [],
            "assignment": {},
            "diarization_status": "ok" if segments else "fallback_single",
            "diarization_error": diarization_error,
            "policy": {
                "speaker_algorithm": "Gold V7 assignment + Luna-low participant/crowd role judge",
                "api_order": "diarization alone first; Luna-low role judge; video render later",
                "crowd": "ambient crowd/laughter/cheering is not participant #3",
            },
        }
        _write_json(output, profile)

        return output
    except Exception as error:
        if audio_path is not None:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass
        return _write_json(output, {
            "version": SPEAKER_PROFILE_VERSION,
            "status": "fallback",
            "phase": "speaker_scan",
            "timing_basis": "source_timeline",
            "clip_index": int(clip_index),
            "edited_clip_path": str(edited_clip_path),
            "words": [], "segments": [], "speaker_stats": [],
            "mode": "fallback", "primary_speaker": None,
            "secondary_speaker": None, "kept_speakers": [],
            "participant_speakers": [], "background_speakers": [],
            "display_labels": {}, "error": str(error),
        })


def create_speaker_profile(
    edited_clip_path: str | Path,
    clip_index: int,
    transcript_path: str | Path | None = None,
    timeline_path: str | Path | None = None,
    speaker_scan_path: str | Path | None = None,
    verified_terms: list[str] | tuple[str, ...] | None = None,
) -> Path:
    """Build the final caption profile without letting speakers recalibrate words.

    1. ai.caption_stack produces the WORDS (frozen transcript) and their CLOCK
       (one word-alignment provider) from the exact final edited clip.
    2. Pre-render diarization segments feed the proven Gold V7 turn assignment.
    3. The acoustic turns give every word its voice's caption colour
       (``_assign_speaker_colors``). No person is identified or named; speaker
       metadata never touches text or timestamps.

    ``verified_terms``: user-verified names/terms (creator, configured entities)
    given to the precision ear as spelling references only.
    """
    edited_clip_path = Path(edited_clip_path).resolve()
    output = _output_path(edited_clip_path, clip_index)
    audio_path: Path | None = None
    # Keep diagnostic fallback construction safe even if an early probe/read
    # fails before the scan metadata is loaded.
    scan: dict[str, Any] = {}

    try:
        duration = _probe_duration(edited_clip_path)
        if duration <= 0:
            raise RuntimeError("Edited clip süresi geçersiz.")

        scan = {}
        if speaker_scan_path:
            scan_path = Path(speaker_scan_path).resolve()
            if scan_path.is_file():
                scan = json.loads(scan_path.read_text(encoding="utf-8"))

        audio_path = _extract_audio(edited_clip_path, clip_index)
        scan_segments = [dict(x) for x in (scan.get("segments") or []) if isinstance(x, dict)]

        words, alignment_ratio, text_source, caption_quality = final_captions.transcribe_final_short(
            edited_clip_path,
            duration,
            verified_terms=list(verified_terms or ()),
        )

        segments = scan_segments
        stats = [dict(x) for x in (scan.get("speaker_stats") or []) if isinstance(x, dict)]
        diarization_error = str(scan.get("diarization_error") or "")

        # Compatibility only when no pre-render scan exists. Do not re-diarize
        # after a known scan failure; accurate plain captions are safer.
        if not segments and not speaker_scan_path:
            try:
                diarized = _run_diarization(audio_path)
                segments = _normalize_segments(diarized, duration)
                stats = _speaker_stats(segments)
                diarization_error = ""
            except Exception as error:
                diarization_error = str(error)

        scan_classification = str(scan.get("classification_source", ""))
        scan_mode = str(scan.get("mode", ""))
        if (
            segments
            and stats
            and scan_classification in {"luna_low_over_full_diarization", "speaker_ensemble_v11"}
            and scan_mode in {"single", "dual", "triple", "crowd", "unresolved"}
        ):
            mode = scan_mode
            primary = str(scan.get("primary_speaker") or "") or None
            secondary = str(scan.get("secondary_speaker") or "") or None
            kept = [str(x) for x in scan.get("kept_speakers", []) if str(x)]
            participants = [str(x) for x in scan.get("participant_speakers", []) if str(x)]
            background_speakers = [str(x) for x in scan.get("background_speakers", []) if str(x)]
        elif segments and stats:
            mode, primary, secondary, kept, participants, background_speakers = _classify(stats)
        else:
            mode, primary, secondary, kept, participants, background_speakers = (
                "unresolved", None, None, [], [], []
            )

        # Gold V7 owns word -> raw-speaker assignment for EVERY clip.
        assigned_words, assignment = _assign_words(
            words=words,
            segments=segments,
            mode=mode,
            primary=primary,
            secondary=secondary,
            background_speakers=background_speakers,
            participant_speakers=participants,
        )
        effective_mode = str(assignment.get("effective_mode", mode))
        if effective_mode != mode:
            mode = effective_mode
            if mode not in {"dual", "triple"}:
                secondary = None
                kept = [primary] if primary and mode != "unresolved" else []

        boundary_audit = dict(scan.get("speaker_boundary_audit", {}) or {})
        assigned_words, speaker_colors = _assign_speaker_colors(
            assigned_words,
            mode=mode,
            participants=participants,
            hard_failure=bool(boundary_audit.get("hard_failure", False)),
        )
        for raw in assigned_words:
            if isinstance(raw, dict):
                # A colour names a voice; no caption ever prints a speaker name.
                raw["speaker_label"] = ""

        if isinstance(caption_quality, dict):
            caption_quality["speaker_mutates_text"] = False
            caption_quality["speaker_mutates_timestamps"] = False
            caption_quality["speaker_word_assignment"] = "gold_v7_acoustic_turns"
            caption_quality["speaker_colors"] = {
                key: speaker_colors[key] for key in ("engaged", "reason", "slots", "neutral_words")
            }

        profile = {
            "version": SPEAKER_PROFILE_VERSION,
            "status": "ok",
            "phase": "final_profile",
            "timing_basis": final_captions.TIMING_BASIS,
            "timing_model": str((caption_quality or {}).get("timing_model", "")),
            "text_model": text_source,
            "diarization_model": DIARIZATION_MODEL,
            "edited_clip_path": str(edited_clip_path),
            "clip_index": int(clip_index),
            "clip_duration": round(duration, 3),
            "alignment_ratio": round(alignment_ratio, 4),
            "caption_quality": caption_quality,
            "mode": mode,
            "primary_speaker": primary,
            "secondary_speaker": secondary,
            "kept_speakers": kept,
            "participant_speakers": participants,
            "background_speakers": background_speakers,
            "display_labels": {},
            "speaker_colors": speaker_colors,
            "speaker_stats": stats,
            "speaker_boundary_quality": float(scan.get("speaker_boundary_quality", 0.0) or 0.0),
            "speaker_boundary_audit": boundary_audit,
            "segments": segments,
            "words": assigned_words,
            "assignment": assignment,
            "diarization_status": "ok" if segments else "fallback_single",
            "diarization_error": diarization_error,
            "policy": {
                "word_truth": "caption_stack frozen transcript (Qwen ears, agreement / caption judge); "
                              "speaker layer cannot rewrite text",
                "clock_truth": "one word-alignment provider over the frozen words; no post calibration",
                "speaker_truth": "Gold V7 acoustic turns (diarization + text/timing evidence); no person identity",
                "speaker_color": "a colour names a voice: first reliable voice A, next B, then C, for the whole "
                                 "clip; uncertain/weak turns neutral; fewer than two reliable voices -> plain",
            },
            "compatibility": {
                "transcript_path": str(transcript_path) if transcript_path else None,
                "timeline_path": str(timeline_path) if timeline_path else None,
                "speaker_scan_path": str(speaker_scan_path) if speaker_scan_path else None,
            },
        }
    except Exception as error:
        # Diagnostic fallback only: the pipeline refuses to publish legacy-timed
        # captions from this profile (see shorts_pipeline mandatory final-clock guard).
        profile = {
            "version": SPEAKER_PROFILE_VERSION,
            "status": "fallback",
            "timing_basis": "failed_final_profile",
            "clip_index": int(clip_index),
            "edited_clip_path": str(edited_clip_path),
            "words": [],
            "segments": [dict(x) for x in (scan.get("segments") or []) if isinstance(x, dict)],
            "speaker_stats": [dict(x) for x in (scan.get("speaker_stats") or []) if isinstance(x, dict)],
            "mode": str(scan.get("mode") or "fallback"),
            "primary_speaker": scan.get("primary_speaker"),
            "secondary_speaker": scan.get("secondary_speaker"),
            "kept_speakers": [str(x) for x in (scan.get("kept_speakers") or []) if str(x)],
            "participant_speakers": [str(x) for x in (scan.get("participant_speakers") or []) if str(x)],
            "background_speakers": [str(x) for x in (scan.get("background_speakers") or []) if str(x)],
            "display_labels": {},
            "error": str(error),
        }
    finally:
        if audio_path is not None:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass

    return _write_json(output, profile)
