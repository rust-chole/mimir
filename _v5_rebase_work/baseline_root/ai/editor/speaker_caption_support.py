from __future__ import annotations

import array
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import re
import statistics
import subprocess
import sys
import wave
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ai.openai_client import client
from ai import vod_processor, model_config
from ai.editor import speaker_role_judge


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SPEAKER_OUTPUT_DIR = PROJECT_ROOT / "vod_output" / "speaker_captions"
TEMP_DIR = PROJECT_ROOT / "vod_output" / "temp" / "speaker_captions"

SPEAKER_PROFILE_VERSION = 25
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

# Final-visible captions are quality-gated independently from whole-VOD scouting.
# 0.97 is an ASR-consensus confidence target, not a claim of measured ground-truth
# accuracy. Wording is decided only by multiple high-quality ASR passes. Modern
# final-clip diarized SEGMENTS own phrase timing; Whisper is fallback only. Speaker
# preflight diarization remains identity evidence only. Unresolved local wording conflicts
# may trigger one precision round (two extra selected-clip-only ASR calls).
CAPTION_WORD_ACCURACY_TARGET = max(0.0, min(1.0, float(os.getenv("MIMIR_CAPTION_WORD_ACCURACY_TARGET", "0.97") or 0.97)))
CAPTION_QUALITY_RETRY_LIMIT = 1

# V9.6 accuracy-first local repair. The semantic model can only FLAG suspicious
# word spans; it can never rewrite captions. Every flagged span must be resolved
# acoustically by independent micro-audio ASR majority.
CAPTION_MICRO_MAX_SPANS = max(1, int(os.getenv("MIMIR_CAPTION_MICRO_MAX_SPANS", "10") or 10))
CAPTION_MICRO_CONTEXT_WORDS = max(2, int(os.getenv("MIMIR_CAPTION_MICRO_CONTEXT_WORDS", "8") or 8))
CAPTION_MICRO_PAD_SECONDS = max(0.25, float(os.getenv("MIMIR_CAPTION_MICRO_PAD_SECONDS", "1.25") or 1.25))
CAPTION_MICRO_MIN_SECONDS = max(2.0, float(os.getenv("MIMIR_CAPTION_MICRO_MIN_SECONDS", "6.0") or 6.0))
CAPTION_MICRO_MAX_SECONDS = max(CAPTION_MICRO_MIN_SECONDS, float(os.getenv("MIMIR_CAPTION_MICRO_MAX_SECONDS", "10.0") or 10.0))
CAPTION_PARALLEL_WORKERS = max(1, min(5, int(os.getenv("MIMIR_CAPTION_PARALLEL_WORKERS", "5") or 5)))

# V25 truth-over-guessing. A visibly uncertain phrase is better rendered as ???
# than as fluent nonsense. The mask is conditional: it is allowed only after the
# existing multi-ASR micro evidence still cannot form a strong majority.
# V25.1 safety: never collapse a long disagreement run into one visible ??? token.
CAPTION_UNKNOWN_MAX_WORDS = max(1, min(3, int(os.getenv("MIMIR_CAPTION_UNKNOWN_MAX_WORDS", "2") or 2)))
CAPTION_SUSPECT_MAX_WORDS = max(CAPTION_UNKNOWN_MAX_WORDS, min(8, int(os.getenv("MIMIR_CAPTION_SUSPECT_MAX_WORDS", "6") or 6)))
CAPTION_REFINEMENT_MIN_LEXICAL_COVERAGE = max(0.60, min(0.95, float(os.getenv("MIMIR_CAPTION_REFINEMENT_MIN_LEXICAL_COVERAGE", "0.78") or 0.78)))

# V3 local acoustic clock guard. Whisper remains the primary clock, but an
# isolated phrase-start anchor is no longer treated as infallible when the exact
# final 48 kHz PCM proves that the anchor sits in a quiet region and a nearby
# earlier speech onset is much stronger. This is deliberately backward-only:
# it fixes visible caption DELAY without creating speculative early captions.
CAPTION_CLOCK_GUARD_ENABLED = str(os.getenv("MIMIR_CAPTION_CLOCK_GUARD_ENABLED", "1")).strip().lower() not in {"0", "false", "no", "off"}
CAPTION_CLOCK_GUARD_MIN_PRE_GAP = max(0.35, float(os.getenv("MIMIR_CAPTION_CLOCK_GUARD_MIN_PRE_GAP", "0.50") or 0.50))
CAPTION_CLOCK_GUARD_MAX_BACKSHIFT = max(0.30, min(1.50, float(os.getenv("MIMIR_CAPTION_CLOCK_GUARD_MAX_BACKSHIFT", "1.20") or 1.20)))
CAPTION_CLOCK_GUARD_MIN_SHIFT = max(0.10, min(0.50, float(os.getenv("MIMIR_CAPTION_CLOCK_GUARD_MIN_SHIFT", "0.18") or 0.18)))
CAPTION_CLOCK_GUARD_ORIGINAL_MAX_SCORE = max(0.02, min(0.40, float(os.getenv("MIMIR_CAPTION_CLOCK_GUARD_ORIGINAL_MAX_SCORE", "0.18") or 0.18)))
CAPTION_CLOCK_GUARD_MIN_IMPROVEMENT = max(0.10, min(0.80, float(os.getenv("MIMIR_CAPTION_CLOCK_GUARD_MIN_IMPROVEMENT", "0.30") or 0.30)))
CAPTION_CLOCK_GUARD_MIN_TARGET_SCORE = max(0.20, min(0.90, float(os.getenv("MIMIR_CAPTION_CLOCK_GUARD_MIN_TARGET_SCORE", "0.34") or 0.34)))
CAPTION_CLOCK_GUARD_FRAME_SECONDS = 0.020
CAPTION_CLOCK_GUARD_MAX_GROUP_SECONDS = max(0.40, min(3.0, float(os.getenv("MIMIR_CAPTION_CLOCK_GUARD_MAX_GROUP_SECONDS", "2.20") or 2.20)))


# V25 word-clock truth. Modern diarized segments prevent cross-phrase drift; a
# parallel Whisper timing ear supplies actual word onsets INSIDE each segment.
# Whisper never owns wording and can never rewrite caption text.

# V20 caption sync/name integrity. Local-only checks: no unconditional extra AI call.
CAPTION_KNOWN_NAME_SIMILARITY = max(0.55, min(0.95, float(os.getenv("MIMIR_CAPTION_KNOWN_NAME_SIMILARITY", "0.64") or 0.64)))

# V21 phrase-level sync. A caption word/run is never allowed to begin inside a
# locally PROVEN silence and then wait hundreds of milliseconds for real speech.
# This is separate from V20's small global clock calibration. It uses the same
# single cached FFmpeg silencedetect pass, so there is no extra media scan.

# Accuracy-first scheduling: do NOT start a semantic caption transcription on
# the pre-render speaker WAV. Speaker audio is for WHO; final caption text
# is produced later from a fresh 48 kHz PCM master extracted from the rendered
# edited clip. This removes a redundant background API call and avoids letting a
# downsampled speaker asset become the final wording source.

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

# V27 human-verified identity lock. Once the user has listened to A/B/C and
# supplied real names, those names are not merely cosmetic labels anymore.
# We rerun diarization on the exact final 48 kHz audio with OpenAI
# known_speaker_names + 2-10s voice references, using three independent
# segmentation views. A visible name is emitted only when the anchored
# evidence agrees (or one pass has very strong uncontested overlap).
IDENTITY_LOCK_ENABLED = str(os.getenv("MIMIR_IDENTITY_LOCK_ENABLED", "1")).strip().lower() not in {"0", "false", "no", "off"}
IDENTITY_LOCK_VAD_THRESHOLD = max(0.0, min(1.0, float(os.getenv("MIMIR_IDENTITY_LOCK_VAD_THRESHOLD", "0.30") or 0.30)))
IDENTITY_LOCK_VAD_PREFIX_PADDING_MS = max(0, int(os.getenv("MIMIR_IDENTITY_LOCK_VAD_PREFIX_PADDING_MS", "320") or 320))
IDENTITY_LOCK_VAD_SILENCE_MS = max(50, int(os.getenv("MIMIR_IDENTITY_LOCK_VAD_SILENCE_MS", "180") or 180))

# V30 speaker-boundary calibration. Known-speaker segment timestamps can lag a
# real turn by a few hundred milliseconds even when the diarized TEXT belongs to
# the correct speaker. Never let pure segment overlap decide a word that sits on
# a known-speaker transition. At those boundaries, exact monotonic lexical
# alignment across the anchored diarization passes is the tie-breaker; without
# lexical majority the word stays unlabeled rather than being assigned wrongly.

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


def _text_agreement(left: str, right: str) -> float:
    left_words = [vod_processor.canonical_word(x) for x in re.findall(r"\S+", str(left))]
    right_words = [vod_processor.canonical_word(x) for x in re.findall(r"\S+", str(right))]
    left_words = [x for x in left_words if x]
    right_words = [x for x in right_words if x]
    if not left_words or not right_words:
        return 0.0
    return float(SequenceMatcher(None, left_words, right_words, autojunk=False).ratio())


def _caption_tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"\S+", str(text)) if vod_processor.canonical_word(token)]


def _canon_phrase(tokens: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        value for value in (vod_processor.canonical_word(token) for token in tokens) if value
    )


def _canon_text(text: str) -> tuple[str, ...]:
    """Canonical token tuple used only for transcript-evidence deduplication."""
    return _canon_phrase(_caption_tokens(text))


def _edit_proposals(base_text: str, reference_text: str) -> dict[tuple[int, int], tuple[str, ...]]:
    """Return local edits that reference ASR proposes against the primary transcript."""
    base_tokens = _caption_tokens(base_text)
    ref_tokens = _caption_tokens(reference_text)
    base_values = [vod_processor.canonical_word(token) for token in base_tokens]
    ref_values = [vod_processor.canonical_word(token) for token in ref_tokens]
    matcher = SequenceMatcher(None, base_values, ref_values, autojunk=False)
    result: dict[tuple[int, int], tuple[str, ...]] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        result[(int(i1), int(i2))] = tuple(ref_tokens[j1:j2])
    return result









def _known_name_tokens(names: list[str] | None) -> list[str]:
    result: list[str] = []
    for raw in names or []:
        for token in re.findall(r"[A-Za-z0-9']+", str(raw)):
            value = vod_processor.canonical_word(token)
            if len(value) < 3 or value in {"speaker", "unknown", "main", "secondary"}:
                continue
            if value not in result:
                result.append(value)
    return result






def _known_name_display_map(names: list[str] | None) -> dict[str, str]:
    """Return canonical verified participant name -> human supplied spelling."""
    result: dict[str, str] = {}
    for raw in names or []:
        display = " ".join(str(raw).strip().split())
        tokens = re.findall(r"[A-Za-z0-9']+", display)
        if len(tokens) != 1:
            continue
        key = vod_processor.canonical_word(tokens[0])
        if len(key) < 3 or key in {"speaker", "unknown", "main", "secondary"}:
            continue
        result.setdefault(key, tokens[0])
    return result


def _apply_verified_vocative_name_orthography(
    text: str,
    names: list[str] | None,
) -> tuple[str, dict[str, Any]]:
    """Normalize a phonetic near-miss only when it is clearly a direct address.

    This is deliberately *orthographic*, not semantic ASR correction.  Human-
    verified participant names are authoritative for how that participant's name
    is spelled, but never prove that every similar-sounding word is that name.
    Therefore a near-match is rewritten only when the token behaves like a
    vocative: it is comma/colon-delimited and nearby words address ``you``.

    Example: ``Tyler, would you ...`` with verified participant ``TYLA`` becomes
    ``Tyla, would you ...``.  ``I watched Tyler yesterday`` is untouched.
    Token count and timing are never changed.
    """
    tokens = _caption_tokens(text)
    display_map = _known_name_display_map(names)
    if not tokens or not display_map:
        return text, {"status": "not_needed", "corrections": []}

    corrections: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        current = vod_processor.canonical_word(token)
        if not current or current in display_map or len(current) < 3:
            continue

        ranked: list[tuple[float, str, str]] = []
        for target, display in display_map.items():
            if current[:2] != target[:2]:
                continue
            if abs(len(current) - len(target)) > 2:
                continue
            similarity = float(SequenceMatcher(None, current, target, autojunk=False).ratio())
            if similarity < CAPTION_KNOWN_NAME_SIMILARITY:
                continue
            ranked.append((similarity, target, display))
        ranked.sort(reverse=True)
        if not ranked:
            continue
        if len(ranked) > 1 and ranked[0][0] < ranked[1][0] + 0.10:
            continue

        # Conservative direct-address gate.  A comma/colon plus second-person
        # language distinguishes a vocative name from mentioning a third party.
        has_vocative_punct = token.rstrip().endswith((",", ":"))
        following = [vod_processor.canonical_word(x) for x in tokens[index + 1:index + 5]]
        following = [x for x in following if x]
        second_person = "you" in following or "your" in following
        if not (has_vocative_punct and second_person):
            continue

        similarity, target, display = ranked[0]
        match = re.match(r"^([^A-Za-z0-9']*)([A-Za-z0-9']+)(.*)$", token)
        if not match:
            continue
        prefix, core, suffix = match.groups()
        if core.isupper():
            replacement_core = display.upper()
        elif core[:1].isupper():
            replacement_core = display[:1].upper() + display[1:].lower()
        else:
            replacement_core = display.lower()
        replacement = prefix + replacement_core + suffix
        if replacement == token:
            continue
        tokens[index] = replacement
        corrections.append({
            "index": index,
            "from": token,
            "to": replacement,
            "target_name": target,
            "similarity": round(similarity, 4),
            "reason": "human_verified_name+vocative_second_person",
        })

    return " ".join(tokens).strip(), {
        "status": "corrected" if corrections else "clean",
        "corrections": corrections,
        "policy": "verified participant spelling only in conservative direct-address vocatives; token count/timing unchanged",
    }


def _known_name_suspicions(text: str, names: list[str] | None) -> list[dict[str, Any]]:
    """Flag near-miss spellings of names already known from speaker identity.

    This never rewrites text. It only sends the tiny span through the existing
    micro acoustic evidence path.
    """
    known = _known_name_tokens(names)
    if not known:
        return []
    rows: list[dict[str, Any]] = []
    for index, token in enumerate(_caption_tokens(text)):
        value = vod_processor.canonical_word(token)
        if not value or len(value) < 3 or value in known:
            continue
        for name in known:
            if value[:2] != name[:2]:
                continue
            if abs(len(value) - len(name)) > 2:
                continue
            similarity = float(SequenceMatcher(None, value, name, autojunk=False).ratio())
            if similarity < CAPTION_KNOWN_NAME_SIMILARITY:
                continue
            rows.append({
                "start": index,
                "end": index + 1,
                "severity": min(1.0, 0.82 + 0.18 * similarity),
                "reason": f"possible known-name spelling mismatch: {token} vs {name}",
                "source": "known_name",
            })
            break
    return rows









def _asr_disagreement_spans(text: str, refs: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    token_count = len(_caption_tokens(text))
    for ref_index, ref in enumerate(refs, start=1):
        for (start, end), replacement in _edit_proposals(text, ref).items():
            if start == end:
                start = max(0, min(token_count - 1, start - 1)) if token_count else 0
                end = min(token_count, start + 2)
            if end <= start or start >= token_count:
                continue
            rows.append({
                "start": max(0, start),
                "end": min(token_count, max(start + 1, end)),
                "severity": 1.0,
                "reason": f"full-clip ASR disagreement pass {ref_index}",
                "source": "asr_disagreement",
                "alternative": " ".join(replacement)[:120],
            })
    return rows


def _merge_suspect_spans(rows: list[dict[str, Any]], token_count: int) -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    for row in rows:
        start = max(0, min(token_count, int(row.get("start", 0) or 0)))
        end = max(start + 1, min(token_count, int(row.get("end", start + 1) or (start + 1))))
        if end <= start:
            continue
        clean.append({
            "start": start,
            "end": end,
            "severity": float(row.get("severity", 0.0) or 0.0),
            "sources": [str(row.get("source", "unknown"))],
            "reasons": [str(row.get("reason", ""))[:220]],
        })
    clean.sort(key=lambda item: (item["start"], item["end"]))
    merged: list[dict[str, Any]] = []
    for row in clean:
        # V25.1: merge only genuinely overlapping local doubts, and never let
        # chained ASR edits grow into a transcript-sized suspect span. The old
        # ``end + 1`` rule could chain adjacent disagreements across most of a
        # Short; V25 uncertainty masking would then replace that whole run with
        # a single sentinel, producing one visible ??? and deleting the rest.
        can_merge = False
        if merged and row["start"] < merged[-1]["end"]:
            prospective_end = max(merged[-1]["end"], row["end"])
            prospective_words = prospective_end - merged[-1]["start"]
            can_merge = prospective_words <= CAPTION_SUSPECT_MAX_WORDS
        if can_merge:
            merged[-1]["end"] = max(merged[-1]["end"], row["end"])
            merged[-1]["severity"] = max(merged[-1]["severity"], row["severity"])
            merged[-1]["sources"] = list(dict.fromkeys(merged[-1]["sources"] + row["sources"]))
            merged[-1]["reasons"] = list(dict.fromkeys(merged[-1]["reasons"] + row["reasons"]))[:4]
        else:
            merged.append(dict(row))

    # ASR disagreements are acoustically observed and therefore win priority;
    # semantic-only suspects follow by severity. Keep the final list in timeline order.
    ranked = sorted(
        merged,
        key=lambda item: (
            "asr_disagreement" not in item["sources"],
            -float(item["severity"]),
            item["start"],
        ),
    )[:CAPTION_MICRO_MAX_SPANS]
    return sorted(ranked, key=lambda item: item["start"])


def _aligned_index_for_token(token_index: int, token_count: int, aligned_count: int) -> int:
    if aligned_count <= 1 or token_count <= 1:
        return 0
    if token_count == aligned_count:
        return max(0, min(aligned_count - 1, token_index))
    ratio = max(0.0, min(1.0, token_index / max(1, token_count - 1)))
    return max(0, min(aligned_count - 1, int(round(ratio * (aligned_count - 1)))))


def _reference_phrase_for_core(
    base_text: str,
    reference_text: str,
    core_start: int,
    core_end: int,
) -> tuple[str, ...]:
    """Map one base-token span into one ASR reference without inventing text."""
    base_tokens = _caption_tokens(base_text)
    ref_tokens = _caption_tokens(reference_text)
    base_values = [vod_processor.canonical_word(x) for x in base_tokens]
    ref_values = [vod_processor.canonical_word(x) for x in ref_tokens]
    if not base_tokens or not ref_tokens:
        return tuple()
    core_start = max(0, min(len(base_tokens) - 1, int(core_start)))
    core_end = max(core_start + 1, min(len(base_tokens), int(core_end)))
    matcher = SequenceMatcher(None, base_values, ref_values, autojunk=False)
    picked: list[int] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            if core_start <= i1 <= core_end:
                picked.extend(range(j1, j2))
            continue
        overlap_start = max(core_start, i1)
        overlap_end = min(core_end, i2)
        if overlap_start >= overlap_end:
            continue
        if tag == "equal":
            picked.extend(range(j1 + (overlap_start - i1), j1 + (overlap_end - i1)))
        elif tag == "replace":
            picked.extend(range(j1, j2))
        elif tag == "delete":
            continue
    if not picked:
        return tuple()
    lo = max(0, min(picked))
    hi = min(len(ref_tokens), max(picked) + 1)
    return tuple(ref_tokens[lo:hi])


def _candidate_votes(
    *,
    window_text: str,
    transcripts: list[str],
    core_rel: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert actual ASR outputs into candidate phrases for only the suspect core."""
    buckets: dict[tuple[str, ...], dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    for index, transcript in enumerate(transcripts, start=1):
        text = str(transcript or "").strip()
        if not text:
            continue
        phrase = _reference_phrase_for_core(window_text, text, core_rel[0], core_rel[1])
        canon = _canon_phrase(phrase)
        # Empty means this ASR effectively deleted the core. Keep it as evidence,
        # but deletion needs a stronger 3-vote threshold before automatic use.
        slot = buckets.setdefault(canon, {"phrase": phrase, "votes": 0, "sources": []})
        slot["votes"] += 1
        slot["sources"].append(f"asr_{index}")
        evidence.append({"source": f"asr_{index}", "phrase": " ".join(phrase), "transcript": text[:400]})
    ranked = sorted(
        buckets.values(),
        key=lambda item: (int(item["votes"]), len(_canon_phrase(tuple(item["phrase"])))),
        reverse=True,
    )
    return ranked, evidence


def _expand_micro_window(
    *,
    time_start: float,
    time_end: float,
    duration: float,
) -> tuple[float, float]:
    """Keep enough neighboring phrase context without re-transcribing the whole Short."""
    time_start = max(0.0, float(time_start))
    time_end = min(duration, max(time_start + 0.20, float(time_end)))
    length = time_end - time_start
    if length < CAPTION_MICRO_MIN_SECONDS:
        missing = CAPTION_MICRO_MIN_SECONDS - length
        time_start = max(0.0, time_start - missing / 2.0)
        time_end = min(duration, time_end + missing / 2.0)
        if time_end - time_start < CAPTION_MICRO_MIN_SECONDS:
            if time_start <= 0.001:
                time_end = min(duration, CAPTION_MICRO_MIN_SECONDS)
            elif time_end >= duration - 0.001:
                time_start = max(0.0, duration - CAPTION_MICRO_MIN_SECONDS)
    if time_end - time_start > CAPTION_MICRO_MAX_SECONDS:
        center = (time_start + time_end) / 2.0
        time_start = max(0.0, center - CAPTION_MICRO_MAX_SECONDS / 2.0)
        time_end = min(duration, time_start + CAPTION_MICRO_MAX_SECONDS)
        if time_end - time_start < CAPTION_MICRO_MAX_SECONDS:
            time_start = max(0.0, time_end - CAPTION_MICRO_MAX_SECONDS)
    return time_start, time_end







def _preserve_token_case_and_punctuation(source: str, replacement_word: str) -> str:
    """Replace only the lexical core while preserving visible punctuation/case."""
    source = str(source)
    match = re.match(r"^(\W*)([A-Za-z']+)(\W*)$", source)
    if not match:
        return replacement_word
    prefix, core, suffix = match.groups()
    replacement = replacement_word
    if core.isupper():
        replacement = replacement.upper()
    elif core[:1].isupper():
        replacement = replacement[:1].upper() + replacement[1:]
    return f"{prefix}{replacement}{suffix}"


def _streamer_slang_evidence_resolution(
    *,
    current_tokens: list[str],
    candidate_rows: list[dict[str, Any]],
    context_text: str,
) -> tuple[list[str] | None, dict[str, Any]]:
    """Resolve ONLY evidence-backed orthographic streamer-slang ambiguity.

    This is not a generic spellchecker. A preferred spelling is allowed only
    when an independent micro-ASR candidate actually emitted that spelling and
    the local wording looks like direct-address livestream speech. The current
    audio-derived word count/order stays unchanged.
    """
    aliases = {
        "shorty": "shawty",
    }
    context_canon = {vod_processor.canonical_word(x) for x in _caption_tokens(context_text)}
    direct_address_cues = {"you", "your", "hey", "girl", "bro", "look", "good", "damn"}
    if not (context_canon & direct_address_cues):
        return None, {"status": "no_direct_address_context"}

    candidate_canons: list[tuple[set[str], int]] = []
    for row in candidate_rows:
        phrase = tuple(row.get("phrase", ()))
        canon = {vod_processor.canonical_word(x) for x in phrase if vod_processor.canonical_word(x)}
        candidate_canons.append((canon, int(row.get("votes", 0) or 0)))

    updated = list(current_tokens)
    changes: list[dict[str, Any]] = []
    for index, token in enumerate(current_tokens):
        canon = vod_processor.canonical_word(token)
        preferred = aliases.get(canon)
        if not preferred:
            continue
        evidence_votes = sum(votes for canon_set, votes in candidate_canons if preferred in canon_set)
        if evidence_votes < 1:
            continue
        updated[index] = _preserve_token_case_and_punctuation(token, preferred)
        changes.append({
            "from": token,
            "to": updated[index],
            "preferred": preferred,
            "independent_evidence_votes": evidence_votes,
        })

    if not changes:
        return None, {"status": "no_evidence_backed_alias"}
    return updated, {"status": "resolved", "changes": changes}


def _micro_refine_caption(
    *,
    audio_path: Path,
    base_text: str,
    aligned_words: list[dict[str, Any]],
    suspect_spans: list[dict[str, Any]],
    duration: float,
    known_names: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    tokens = _caption_tokens(base_text)
    if not tokens or not aligned_words or not suspect_spans:
        return base_text, {"checked_spans": 0, "corrected_spans": 0, "unresolved_spans": 0, "human_reviews": 0, "details": []}

    original_token_count = len(tokens)
    original_aligned_count = len(aligned_words)
    details: list[dict[str, Any]] = []
    corrected = 0
    unresolved = 0
    human_reviews = 0
    unknown_masks = 0

    # Right-to-left keeps earlier token indices stable when one local phrase changes length.
    for span in sorted(suspect_spans, key=lambda item: item["start"], reverse=True):
        core_start = max(0, min(original_token_count - 1, int(span["start"])))
        core_end = max(core_start + 1, min(original_token_count, int(span["end"])))
        window_start = max(0, core_start - CAPTION_MICRO_CONTEXT_WORDS)
        window_end = min(original_token_count, core_end + CAPTION_MICRO_CONTEXT_WORDS)
        window_end = min(len(tokens), window_end)
        core_end_now = min(len(tokens), core_end)
        if window_start >= window_end or core_start >= core_end_now:
            unresolved += 1
            continue

        aligned_start_index = _aligned_index_for_token(window_start, original_token_count, original_aligned_count)
        aligned_end_index = _aligned_index_for_token(max(window_start, window_end - 1), original_token_count, original_aligned_count)
        time_start = max(0.0, float(aligned_words[aligned_start_index].get("start", 0.0)) - CAPTION_MICRO_PAD_SECONDS)
        time_end = min(duration, float(aligned_words[aligned_end_index].get("end", time_start + 0.5)) + CAPTION_MICRO_PAD_SECONDS)
        time_start, time_end = _expand_micro_window(time_start=time_start, time_end=time_end, duration=duration)
        if time_end <= time_start + 0.20:
            unresolved += 1
            continue

        window_text = " ".join(tokens[window_start:window_end]).strip()
        before = " ".join(tokens[max(0, window_start - 12):window_start])
        after = " ".join(tokens[window_end:min(len(tokens), window_end + 12)])
        core_rel = (core_start - window_start, core_end_now - window_start)
        current_phrase = " ".join(tokens[core_start:core_end_now]).strip()
        raw_path: Path | None = None
        enhanced_path: Path | None = None
        errors: list[str] = []
        transcripts: list[str] = []
        candidate_rows: list[dict[str, Any]] = []
        semantic_meta: dict[str, Any] = {"status": "not_needed"}
        human_meta: dict[str, Any] = {"status": "not_needed"}
        selected_phrase: str | None = None
        selected_source = ""
        known_name_span = "known_name" in set(str(value) for value in span.get("sources", []))
        glossary_tokens = _known_name_tokens(known_names)
        glossary_instruction = (
            "Known on-screen participant names (reference only): " + ", ".join(glossary_tokens)
            + ". If the AUDIO clearly says one of these names, spell that name exactly. "
              "Never force a name when the audio says something else."
        ) if glossary_tokens else ""
        try:
            raw_path = vod_processor.extract_caption_micro_audio(
                audio_path, start=time_start, end=time_end,
                label=f"span_{core_start}_{core_end}", enhanced=False,
            )
            enhanced_path = vod_processor.extract_caption_micro_audio(
                audio_path, start=time_start, end=time_end,
                label=f"span_{core_start}_{core_end}", enhanced=True,
            )
            first_three = [
                (raw_path, vod_processor.CAPTION_ACCURATE_MODEL, "raw acoustic", False, "Acoustic-only listen; no contextual guessing."),
                (enhanced_path, vod_processor.CAPTION_ACCURATE_MODEL, "enhanced acoustic", False, "Independent enhanced-audio listen; preserve tiny function words."),
                (raw_path, vod_processor.CAPTION_CROSSCHECK_MODEL, "context diverse", True, "Context may disambiguate, but audio always wins. " + glossary_instruction),
            ]

            # V13 speed-only optimization: these are independent evidence ears.
            # Run them concurrently, but consume results in the original fixed
            # order so voting/output semantics stay deterministic.
            with ThreadPoolExecutor(max_workers=min(CAPTION_PARALLEL_WORKERS, len(first_three)), thread_name_prefix="mimir-caption-micro") as executor:
                futures = [
                    executor.submit(
                        vod_processor.transcribe_caption_micro_pass,
                        view,
                        model=model,
                        label=label,
                        context_before=before,
                        context_after=after,
                        extra_instruction=instruction,
                        use_context=use_context,
                        known_names=known_names,
                    )
                    for view, model, label, use_context, instruction in first_three
                ]
                for future, (_view, _model, label, _use_context, _instruction) in zip(futures, first_three):
                    try:
                        text = str(future.result() or "").strip()
                        if text:
                            transcripts.append(text)
                    except Exception as error:
                        errors.append(f"{label}: {error}")

            candidate_rows, evidence = _candidate_votes(window_text=window_text, transcripts=transcripts, core_rel=core_rel)
            if candidate_rows:
                top = candidate_rows[0]
                # V31: the primary gpt-transcribe text is authoritative. A local
                # rewrite after only 2/3 ears was too eager and could create
                # fluent-but-wrong captions. First round must be unanimous 3/3.
                phrase = tuple(top.get("phrase", ()))
                required = 3
                if int(top.get("votes", 0)) >= required and phrase:
                    selected_phrase = " ".join(phrase)
                    selected_source = f"micro_strict_{int(top.get('votes', 0))}_of_{len(transcripts)}"

            # No majority: two more DIFFERENT evidence views on this same tiny window only.
            if selected_phrase is None and CAPTION_QUALITY_RETRY_LIMIT > 0:
                extra = [
                    (enhanced_path, vod_processor.CAPTION_CROSSCHECK_MODEL, "enhanced diverse", True, "Independent model on enhanced audio. " + glossary_instruction),
                    (raw_path, vod_processor.CAPTION_ACCURATE_MODEL, "context precision", True, "Final context-guided precision listen; never repair grammar. " + glossary_instruction),
                ]
                with ThreadPoolExecutor(max_workers=min(CAPTION_PARALLEL_WORKERS, len(extra)), thread_name_prefix="mimir-caption-micro-extra") as executor:
                    futures = [
                        executor.submit(
                            vod_processor.transcribe_caption_micro_pass,
                            view,
                            model=model,
                            label=label,
                            context_before=before,
                            context_after=after,
                            extra_instruction=instruction,
                            use_context=use_context,
                            known_names=known_names,
                        )
                        for view, model, label, use_context, instruction in extra
                    ]
                    for future, (_view, _model, label, _use_context, _instruction) in zip(futures, extra):
                        try:
                            text = str(future.result() or "").strip()
                            if text:
                                transcripts.append(text)
                        except Exception as error:
                            errors.append(f"{label}: {error}")
                candidate_rows, evidence = _candidate_votes(window_text=window_text, transcripts=transcripts, core_rel=core_rel)
                if candidate_rows:
                    top = candidate_rows[0]
                    phrase = tuple(top.get("phrase", ()))
                    # After five independent local ears, require 4/5 to rewrite.
                    # Deletion is never automatic in V31; preserving the primary
                    # transcript is safer than silently dropping spoken words.
                    if phrase and int(top.get("votes", 0)) >= 4:
                        selected_phrase = " ".join(phrase)
                        selected_source = f"micro_strict_{int(top.get('votes', 0))}_of_{len(transcripts)}"

            # V3 lexical policy: orthographic streamer slang may be normalized
            # only when an independent micro-ASR ear explicitly emitted the
            # preferred spelling. This never invents a word and never touches
            # timestamps. Example: direct-address shorty/shawty ambiguity.
            slang_meta: dict[str, Any] = {"status": "not_needed"}
            if selected_phrase is None and candidate_rows:
                slang_tokens, slang_meta = _streamer_slang_evidence_resolution(
                    current_tokens=list(tokens[core_start:core_end_now]),
                    candidate_rows=candidate_rows,
                    context_text=" ".join(tokens[max(0, core_start - 6):min(len(tokens), core_end_now + 6)]),
                )
                if slang_tokens is not None:
                    selected_phrase = " ".join(slang_tokens)
                    selected_source = "streamer_slang_evidence"

            # V31/V3: no general semantic tie-break can rewrite wording and no ??? mask can
            # replace real primary text. If the acoustic evidence is not strong
            # enough, preserve the high-accuracy gpt-transcribe phrase verbatim.
            semantic_meta = {"status": "disabled_v31_primary_text_authority"}
            uncertainty_meta: dict[str, Any] = {"score": 0.0, "eligible": False, "status": "disabled_v31_preserve_primary"}
            if selected_phrase is None:
                human_meta = {"status": "disabled_non_blocking"}

            if selected_phrase is not None:
                new_words = _caption_tokens(selected_phrase)
                old_canon = _canon_phrase(tuple(tokens[core_start:core_end_now]))
                new_canon = _canon_phrase(tuple(new_words))
                if old_canon != new_canon:
                    tokens[core_start:core_end_now] = new_words
                    corrected += 1
            else:
                unresolved += 1

            details.append({
                "span": [core_start, core_end],
                "sources": span.get("sources", []),
                "reasons": span.get("reasons", []),
                "audio_window": [round(time_start, 3), round(time_end, 3)],
                "micro_passes": len(transcripts),
                "candidate_votes": [
                    {"phrase": " ".join(item.get("phrase", ())), "votes": int(item.get("votes", 0)), "sources": item.get("sources", [])}
                    for item in candidate_rows[:8]
                ],
                "selected_source": selected_source,
                "selected_phrase": selected_phrase,
                "semantic_tiebreak": semantic_meta,
                "streamer_slang_resolution": slang_meta,
                "human_review": human_meta,
                "uncertainty": uncertainty_meta,
                "masked_unknown": selected_source == "uncertainty_mask",
                "resolved": selected_phrase is not None,
                "errors": errors[:6],
            })
        finally:
            for path in (raw_path, enhanced_path):
                if path is not None:
                    try:
                        path.unlink(missing_ok=True)
                    except Exception:
                        pass

    return " ".join(tokens).strip(), {
        "checked_spans": len(suspect_spans),
        "corrected_spans": corrected,
        "unresolved_spans": unresolved,
        "human_reviews": human_reviews,
        "unknown_masks": unknown_masks,
        "details": list(reversed(details)),
    }




def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(x) for x in values)
    index = int(round(max(0.0, min(1.0, fraction)) * max(0, len(ordered) - 1)))
    return ordered[index]


def _read_pcm_feature_frames(audio_path: Path) -> list[dict[str, float]]:
    """Read exact mono PCM16 WAV and calculate 20 ms acoustic features."""
    with wave.open(str(audio_path), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
            raise RuntimeError("acoustic guard requires mono PCM16 final-caption WAV")
        sample_rate = int(wav.getframerate())
        raw = wav.readframes(wav.getnframes())
    samples = array.array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    frame_samples = max(1, int(round(sample_rate * CAPTION_CLOCK_GUARD_FRAME_SECONDS)))
    result: list[dict[str, float]] = []
    for offset in range(0, max(0, len(samples) - frame_samples + 1), frame_samples):
        frame = samples[offset:offset + frame_samples]
        if not frame:
            continue
        square_sum = sum(float(value) * float(value) for value in frame)
        rms = math.sqrt(square_sum / max(1, len(frame)))
        dbfs = 20.0 * math.log10(max(rms, 1.0) / 32768.0)
        crossings = sum(
            1 for left, right in zip(frame, frame[1:])
            if (left < 0 <= right) or (left >= 0 > right)
        )
        zcr = crossings / max(1, len(frame) - 1)
        result.append({
            "start": offset / float(sample_rate),
            "dbfs": dbfs,
            "zcr": zcr,
        })
    return result


def _acoustic_likelihood_frames(
    frames: list[dict[str, float]],
    *,
    window_start: float,
    window_end: float,
) -> list[dict[str, float]]:
    local = [row for row in frames if window_start <= float(row["start"]) < window_end]
    if not local:
        return []
    noise_db = _percentile([float(row["dbfs"]) for row in local], 0.25)
    zcr_base = statistics.median(float(row["zcr"]) for row in local)
    result: list[dict[str, float]] = []
    for row in local:
        dbfs = float(row["dbfs"]); zcr = float(row["zcr"])
        energy = max(0.0, min(1.0, (dbfs - noise_db - 2.0) / 14.0))
        zcr_threshold = max(0.070, zcr_base * 1.65)
        fricative = max(0.0, min(1.0, (zcr - zcr_threshold) / 0.090))
        fricative *= max(0.15, min(1.0, (dbfs - noise_db + 4.0) / 7.0))
        likelihood = max(energy, 0.95 * fricative)
        result.append({**row, "likelihood": likelihood, "noise_db": noise_db, "zcr_base": zcr_base})
    return result


def _window_acoustic_score(frames: list[dict[str, float]], start: float, end: float) -> float:
    values = [float(row.get("likelihood", 0.0)) for row in frames if start <= float(row["start"]) < end]
    if not values:
        return 0.0
    ordered = sorted(values, reverse=True)
    top_count = max(1, int(len(ordered) * 0.70))
    core = sum(ordered[:top_count]) / top_count
    onset_end = start + max(0.06, (end - start) * 0.35)
    onset_values = [float(row.get("likelihood", 0.0)) for row in frames if start <= float(row["start"]) < onset_end]
    onset = sum(onset_values) / max(1, len(onset_values))
    return 0.70 * core + 0.30 * onset


def _find_previous_speech_onset(
    frames: list[dict[str, float]],
    *,
    search_start: float,
    candidate_start: float,
) -> tuple[float | None, dict[str, float]]:
    local = [dict(row) for row in frames if search_start <= float(row["start"]) < candidate_start + 0.04]
    if not local:
        return None, {}
    active = [float(row.get("likelihood", 0.0)) >= 0.15 for row in local]
    active_indices = [i for i, value in enumerate(active) if value]
    # Bridge <=120 ms holes so low-energy consonants stay attached to the same
    # spoken burst instead of snapping the onset to the following loud vowel.
    max_bridge = max(1, int(round(0.12 / CAPTION_CLOCK_GUARD_FRAME_SECONDS)))
    for left, right in zip(active_indices, active_indices[1:]):
        if 1 < right - left <= max_bridge + 1:
            for index in range(left + 1, right):
                active[index] = True

    regions: list[tuple[int, int]] = []
    region_start: int | None = None
    for index, is_active in enumerate(active + [False]):
        if is_active and region_start is None:
            region_start = index
        elif not is_active and region_start is not None:
            region_end = index
            if (region_end - region_start) * CAPTION_CLOCK_GUARD_FRAME_SECONDS >= 0.12:
                vals = [float(local[i].get("likelihood", 0.0)) for i in range(region_start, region_end)]
                if vals and max(vals) >= 0.42:
                    regions.append((region_start, region_end))
            region_start = None
    if not regions:
        return None, {}

    # Nearest credible burst before the bad Whisper anchor.
    start_index, end_index = regions[-1]
    region = local[start_index:end_index]
    zcr_base = float(region[0].get("zcr_base", 0.04)) if region else 0.04
    noise_db = float(region[0].get("noise_db", -40.0)) if region else -40.0
    onset = float(region[0]["start"])
    # Refine to first actual speech-like frame: either voiced energy or an
    # unvoiced/fricative consonant burst. This catches /s/ in "sixteen".
    for row in region:
        voiced = float(row["dbfs"]) >= noise_db + 6.0
        fricative = float(row["zcr"]) >= max(0.090, zcr_base * 2.0) and float(row["dbfs"]) >= noise_db - 2.0
        if voiced or fricative:
            onset = float(row["start"]); break
    score = _window_acoustic_score(frames, float(region[0]["start"]), float(region[-1]["start"]) + CAPTION_CLOCK_GUARD_FRAME_SECONDS)
    return onset, {
        "region_start": round(float(region[0]["start"]), 3),
        "region_end": round(float(region[-1]["start"]) + CAPTION_CLOCK_GUARD_FRAME_SECONDS, 3),
        "region_score": round(score, 4),
        "noise_db": round(noise_db, 2),
    }


def _apply_local_acoustic_clock_guard(
    *,
    audio_path: Path,
    words: list[dict[str, Any]],
    duration: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Correct only proven LATE phrase-start Whisper outliers.

    No global shift, no speaker timing, no semantic timing. Existing anchors are
    untouched unless a long-pause phrase begins in acoustically weak audio and
    the exact PCM contains a much stronger earlier speech onset nearby.
    """
    if not CAPTION_CLOCK_GUARD_ENABLED or not words:
        return words, {"status": "disabled" if not CAPTION_CLOCK_GUARD_ENABLED else "empty", "corrected_groups": 0, "details": []}
    try:
        raw_frames = _read_pcm_feature_frames(audio_path)
    except Exception as error:
        return words, {"status": "unavailable", "corrected_groups": 0, "details": [], "error": str(error)}

    output = [dict(row) for row in words]
    # Phrase groups use the same 420 ms semantic pause boundary as ASS grouping.
    groups: list[list[int]] = []
    current: list[int] = []
    for index, row in enumerate(output):
        if current:
            previous = output[current[-1]]
            if float(row.get("edited_start", 0.0)) - float(previous.get("edited_end", 0.0)) >= 0.42:
                groups.append(current); current = []
        current.append(index)
    if current:
        groups.append(current)

    details: list[dict[str, Any]] = []
    corrected = 0
    for group_index, indices in enumerate(groups):
        first = output[indices[0]]; last = output[indices[-1]]
        start = float(first.get("edited_start", 0.0)); end = float(last.get("edited_end", start))
        group_duration = max(0.0, end - start)
        previous_end = float(output[groups[group_index - 1][-1]].get("edited_end", 0.0)) if group_index > 0 else 0.0
        pre_gap = start - previous_end
        if group_index == 0 or pre_gap < CAPTION_CLOCK_GUARD_MIN_PRE_GAP or group_duration <= 0.0 or group_duration > CAPTION_CLOCK_GUARD_MAX_GROUP_SECONDS:
            continue

        search_start = max(previous_end + 0.05, start - CAPTION_CLOCK_GUARD_MAX_BACKSHIFT)
        feature_frames = _acoustic_likelihood_frames(
            raw_frames,
            window_start=max(0.0, search_start - 0.08),
            window_end=min(duration, end + 0.10),
        )
        if not feature_frames:
            continue
        original_score = _window_acoustic_score(feature_frames, start, end)
        if original_score > CAPTION_CLOCK_GUARD_ORIGINAL_MAX_SCORE:
            continue
        onset, region_meta = _find_previous_speech_onset(
            feature_frames,
            search_start=search_start,
            candidate_start=start,
        )
        if onset is None:
            continue
        shift = onset - start
        if shift >= -CAPTION_CLOCK_GUARD_MIN_SHIFT or abs(shift) > CAPTION_CLOCK_GUARD_MAX_BACKSHIFT + 0.02:
            continue
        shifted_end = end + shift
        if shifted_end <= previous_end + 0.03:
            continue
        target_score = _window_acoustic_score(feature_frames, onset, shifted_end)
        improvement = target_score - original_score
        if target_score < CAPTION_CLOCK_GUARD_MIN_TARGET_SCORE or improvement < CAPTION_CLOCK_GUARD_MIN_IMPROVEMENT:
            continue

        next_start = float(output[groups[group_index + 1][0]].get("edited_start", duration)) if group_index + 1 < len(groups) else duration
        if shifted_end >= next_start - 0.02:
            continue

        for index in indices:
            row = output[index]
            old_start = float(row.get("edited_start", 0.0)); old_end = float(row.get("edited_end", old_start))
            row["original_edited_start"] = round(old_start, 3)
            row["original_edited_end"] = round(old_end, 3)
            row["edited_start"] = round(max(0.0, old_start + shift), 3)
            row["edited_end"] = round(min(duration, max(old_start + shift + 0.025, old_end + shift)), 3)
            row["timing_source"] = str(row.get("timing_source", vod_processor.TIMING_MODEL)) + "_acoustic_guard"
            row["acoustic_guard_delta"] = round(shift, 3)
        corrected += 1
        details.append({
            "words": [str(output[index].get("word", "")) for index in indices],
            "original_start": round(start, 3),
            "corrected_start": round(onset, 3),
            "delta": round(shift, 3),
            "original_score": round(original_score, 4),
            "target_score": round(target_score, 4),
            "improvement": round(improvement, 4),
            **region_meta,
        })

    return output, {
        "status": "corrected" if corrected else "clean",
        "corrected_groups": corrected,
        "details": details,
        "policy": "backward-only exact-PCM phrase-start outlier correction; no global shift",
    }


def _transcribe_edited_words(
    audio_path: Path,
    duration: float,
    *,
    accurate_text: str | None = None,
    diarization_text: str = "",
    known_names: list[str] | None = None,
) -> tuple[list[dict[str, Any]], float, str, dict[str, Any]]:
    """Final-caption truth path with three strictly separated authorities.

    WORDS   -> gpt-transcribe (primary, immutable unless a local acoustic vote is decisive)
    CLOCK   -> whisper-1 native word timestamps on the exact final 48 kHz PCM
    SPEAKER -> diarization elsewhere; never allowed to move or rewrite a word

    A single model-diverse full-clip cross-check only LOCATES suspicious spans.
    It cannot replace the primary transcript. Local correction is allowed only
    through the existing tiny-window acoustic vote (3/3, or 4/5 after retry).
    There is no semantic rewriting, global clock calibration, silence snap,
    diarized timing, hybrid timing or transcript-wide majority replacement.
    """
    verification_errors: list[str] = []
    primary_text = str(accurate_text or "").strip()
    crosscheck_text = ""
    timing_data: dict[str, Any] = {}

    with ThreadPoolExecutor(
        max_workers=min(CAPTION_PARALLEL_WORKERS, 3),
        thread_name_prefix="mimir-caption-unified",
    ) as executor:
        primary_future = None
        if not primary_text:
            primary_future = executor.submit(
                vod_processor.transcribe_caption_independent_text,
                audio_path,
                known_names=known_names,
            )
        crosscheck_future = executor.submit(
            vod_processor.transcribe_caption_crosscheck_text,
            audio_path,
        )
        timing_future = executor.submit(
            vod_processor.transcribe_word_timing,
            audio_path,
            known_names=None,
        )

        if primary_future is not None:
            try:
                primary_text = str(primary_future.result() or "").strip()
            except Exception as error:
                verification_errors.append(f"primary gpt-transcribe: {error}")
        try:
            crosscheck_text = str(crosscheck_future.result() or "").strip()
        except Exception as error:
            verification_errors.append(f"crosscheck gpt-4o-transcribe: {error}")
        try:
            timing_data = dict(timing_future.result() or {})
        except Exception as error:
            verification_errors.append(f"whisper native word clock: {error}")

    if not primary_text:
        # One recovery listen, only because the primary produced no wording at all.
        try:
            primary_text = str(
                vod_processor.transcribe_caption_accurate_text(audio_path, known_names=None) or ""
            ).strip()
            if primary_text:
                verification_errors.append("primary empty; acoustic gpt-transcribe recovery used")
        except Exception as error:
            verification_errors.append(f"primary recovery: {error}")

    if not primary_text:
        raise RuntimeError("Final-caption gpt-transcribe metin üretemedi.")

    timing_words = vod_processor.extract_timing_words(timing_data or {})
    if not timing_words:
        raise RuntimeError(
            "Final caption için whisper-1 native word timestamp üretilemedi. "
            "Sentetik/global timing fallback bilinçli olarak kapalı."
        )

    provisional_aligned, provisional_alignment_ratio = (
        vod_processor.align_text_to_fixed_whisper_clock(
            accurate_text=primary_text,
            timing_data=timing_data,
            chunk_duration=duration,
        )
    )

    refs = [crosscheck_text] if crosscheck_text and _canon_text(crosscheck_text) != _canon_text(primary_text) else []
    disagreement_rows = _asr_disagreement_spans(primary_text, refs)
    known_name_rows = _known_name_suspicions(primary_text, known_names)
    suspect_spans = _merge_suspect_spans(
        disagreement_rows + known_name_rows,
        len(_caption_tokens(primary_text)),
    )

    refined_text = primary_text
    micro_meta: dict[str, Any] = {
        "checked_spans": 0,
        "corrected_spans": 0,
        "unresolved_spans": 0,
        "human_reviews": 0,
        "unknown_masks": 0,
        "details": [],
    }
    if suspect_spans:
        refined_text, micro_meta = _micro_refine_caption(
            audio_path=audio_path,
            base_text=primary_text,
            aligned_words=provisional_aligned,
            suspect_spans=suspect_spans,
            duration=duration,
            known_names=known_names,
        )

    # V7 verified-name orthography lock.  Acoustic ASR often cannot distinguish
    # homophonic spellings such as Tyla/Tyler.  When a human-verified participant
    # name appears as a conservative direct-address near-match, normalize only
    # its spelling.  This runs after acoustic micro evidence and never changes
    # token count, timestamps, speaker assignment, or surrounding wording.
    refined_text, known_name_orthography_meta = _apply_verified_vocative_name_orthography(
        refined_text, known_names
    )

    # A local repair may fix a tiny span, never collapse or rewrite the passage.
    base_token_count = len(_caption_tokens(primary_text))
    refined_token_count = len(_caption_tokens(refined_text))
    lexical_coverage = refined_token_count / max(1, base_token_count)
    if base_token_count >= 6 and lexical_coverage < CAPTION_REFINEMENT_MIN_LEXICAL_COVERAGE:
        refined_text = primary_text
        micro_meta = dict(micro_meta or {})
        micro_meta["safety_reverted"] = True
        micro_meta["safety_reason"] = "lexical_coverage"
        micro_meta["pre_revert_lexical_coverage"] = round(float(lexical_coverage), 4)

    words, alignment_ratio = vod_processor.align_text_to_fixed_whisper_clock(
        accurate_text=refined_text,
        timing_data=timing_data,
        chunk_duration=duration,
    )

    normalized: list[dict[str, Any]] = []
    for item in words:
        text = str(item.get("word", "")).strip()
        if not text:
            continue
        start = max(0.0, min(duration, float(item.get("start", 0.0))))
        end = max(start + 0.025, min(duration, float(item.get("end", start + 0.04))))
        if end <= start:
            continue
        normalized.append({
            "word": text,
            "edited_start": round(start, 3),
            "edited_end": round(end, 3),
            "timing_source": str(item.get("alignment_source", vod_processor.TIMING_MODEL)),
        })
    if not normalized:
        raise RuntimeError("Edited clip transcription kelime üretmedi.")

    normalized, clock_guard_meta = _apply_local_acoustic_clock_guard(
        audio_path=audio_path,
        words=normalized,
        duration=duration,
    )

    evidence_agreement = (
        _text_agreement(refined_text, crosscheck_text)
        if crosscheck_text else 1.0
    )
    quality = {
        "target": round(CAPTION_WORD_ACCURACY_TARGET, 3),
        "alignment_with_fixed_whisper_clock": round(float(alignment_ratio), 4),
        "provisional_alignment_with_fixed_whisper_clock": round(float(provisional_alignment_ratio), 4),
        "quality_score": round(float(evidence_agreement), 4),
        "text_model": vod_processor.CAPTION_ACCURATE_MODEL,
        "crosscheck_model": vod_processor.CAPTION_CROSSCHECK_MODEL,
        "timing_model": vod_processor.TIMING_MODEL,
        "timing_mode": "whisper_native_word_clock_plus_local_acoustic_outlier_guard",
        "timing_authorities": 1,
        "speaker_timing_authority": False,
        "post_clock_calibration": False,
        "local_acoustic_clock_guard": clock_guard_meta,
        "clock_corrections": int(clock_guard_meta.get("corrected_groups", 0) or 0),
        "global_clock_calibration": {"status": "removed"},
        "phrase_start_silence_guard": {"status": "removed"},
        "hybrid_word_timing": {"status": "removed"},
        "full_asr_passes": 2,
        "timing_passes": 1,
        "suspect_spans": len(suspect_spans),
        "local_corrections": int(micro_meta.get("corrected_spans", 0) or 0),
        "unresolved_local_conflicts": int(micro_meta.get("unresolved_spans", 0) or 0),
        "known_name_flagged_spans": known_name_rows,
        "known_names": _known_name_tokens(known_names),
        "known_name_orthography": known_name_orthography_meta,
        "micro_accuracy": micro_meta,
        "verification_errors": verification_errors,
        "policy": (
            "gpt-transcribe wording authority + gpt-4o-transcribe disagreement locator; "
            "strict local wording correction only + human-verified direct-address name orthography; "
            "whisper-1 native word clock with backward-only exact-PCM outlier guard; "
            "zero global calibration/diarized timing; speaker metadata cannot mutate words or time"
        ),
    }
    return normalized, float(alignment_ratio), "caption_truth_unified_clean_v3", quality



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

        # Preserve two coherent lexical voices internally, but do not force a
        # human prompt here. speaker_naming separately requires high confidence.
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


def _manual_identity_map(scan: dict[str, Any]) -> dict[str, str]:
    """Return raw diarizer-id -> human-confirmed display name.

    V28 accepts both the legacy all-or-nothing preview source and the new
    per-speaker calibrated source. Partial human identity is valid: one trusted
    voice may be named while every other voice stays unlabeled.
    """
    if not IDENTITY_LOCK_ENABLED or not isinstance(scan, dict):
        return {}
    names_meta = scan.get("speaker_names") if isinstance(scan.get("speaker_names"), dict) else {}
    source = str(names_meta.get("source", ""))
    if source not in {"manual_after_audio_preview", "manual_voice_calibrated_v28"}:
        return {}
    labels = scan.get("display_labels") if isinstance(scan.get("display_labels"), dict) else {}
    participants = [str(x) for x in scan.get("participant_speakers", []) if str(x)]
    ordered: dict[str, str] = {}
    for raw in participants:
        value = " ".join(str(labels.get(raw, "")).replace("\n", " ").split()).strip()
        if not value or value.upper() in {"A", "B", "C"}:
            continue
        ordered[raw] = value[:32]
    folded = [value.casefold() for value in ordered.values()]
    if not ordered or len(ordered) > 3 or len(set(folded)) != len(folded):
        return {}
    return ordered



def _confirmed_identity_ranges(row: dict[str, Any]) -> list[tuple[float, float]]:
    """Return only ranges explicitly stored under a human-verified identity."""
    if not isinstance(row, dict) or row.get("human_verified") is not True:
        return []
    ranges: list[tuple[float, float]] = []
    for key in ("reference_ranges", "verification_ranges"):
        for pair in row.get(key, []) or []:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            try:
                start = float(pair[0]); end = float(pair[1])
            except (TypeError, ValueError):
                continue
            if end <= start + 0.08:
                continue
            ranges.append((max(0.0, start), max(0.0, end)))
    # Keep the originally confirmed order but remove exact duplicates.
    unique: list[tuple[float, float]] = []
    for pair in ranges:
        if pair not in unique:
            unique.append(pair)
    return unique


def _render_confirmed_identity_reference(
    audio_path: Path,
    ranges: list[tuple[float, float]],
    *,
    clip_index: int,
    raw_speaker: str,
) -> Path | None:
    """Stitch human-confirmed snippets into OpenAI's 2-10s reference contract.

    V5 regenerated a too-short human reference from generic raw A/B diarization.
    That can contaminate the identity sample with the wrong person—the exact
    failure we are trying to prevent.  V6 instead stitches only ranges that the
    user already confirmed as the same voice.
    """
    if not ranges:
        return None
    picked: list[tuple[float, float]] = []
    total = 0.0
    for start, end in ranges:
        if total >= 9.8:
            break
        length = max(0.0, end - start)
        if length <= 0.08:
            continue
        take = min(length, 9.8 - total)
        picked.append((start, start + take))
        total += take
        if total >= DIARIZATION_REFERENCE_MIN_SECONDS:
            break
    if total + 1e-6 < DIARIZATION_REFERENCE_MIN_SECONDS:
        return None

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    safe_raw = re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw_speaker))[:16] or "speaker"
    output = TEMP_DIR / f"clip_{int(clip_index):02d}_human_confirmed_ref_{safe_raw}.wav"
    filters: list[str] = []
    labels: list[str] = []
    for idx, (start, end) in enumerate(picked):
        tag = f"hcr{idx}"
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
        "-map", "[aout]", "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le",
        str(output),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
    )
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size < 1024:
        return None
    try:
        duration = _probe_duration(output)
    except Exception:
        return None
    if duration < DIARIZATION_REFERENCE_MIN_SECONDS - 0.05 or duration > 10.05:
        return None
    return output

def _identity_reference_bundle(
    *,
    audio_path: Path,
    scan: dict[str, Any],
    scan_segments: list[dict[str, Any]],
    raw_to_name: dict[str, str],
    clip_index: int,
) -> tuple[list[str], list[str], list[str]]:
    """Use HUMAN-confirmed reference WAVs first; regenerate only as fallback.

    The naming checkpoint stores 2-10s, 48 kHz voice references that the human
    actually listened to and verified on a disjoint utterance. Reusing those
    exact bytes is safer than rebuilding identity anchors from a later heuristic
    speaker map. This function has no caption timing authority.
    """
    calibration = scan.get("identity_calibration") if isinstance(scan.get("identity_calibration"), dict) else {}
    by_raw: dict[str, dict[str, Any]] = {}
    for row in calibration.values():
        if not isinstance(row, dict) or row.get("human_verified") is not True:
            continue
        raw = str(row.get("raw_speaker", "")).strip()
        name = " ".join(str(row.get("name", "")).split()).strip()
        if raw and name:
            by_raw[raw] = row

    names: list[str] = []
    references: list[str] = []
    paths: list[str] = []
    for idx, (raw_speaker, display_name) in enumerate(raw_to_name.items()):
        confirmed = by_raw.get(raw_speaker)
        ref_path: Path | None = None
        if confirmed is not None:
            candidate = Path(str(confirmed.get("reference_path", ""))).resolve()
            confirmed_name = " ".join(str(confirmed.get("name", "")).split()).strip()
            if candidate.is_file() and confirmed_name.casefold() == display_name.casefold():
                try:
                    ref_duration = _probe_duration(candidate)
                except Exception:
                    ref_duration = 0.0
                if 1.95 <= ref_duration <= 10.10:
                    ref_path = candidate

        if ref_path is None and confirmed is not None:
            ref_path = _render_confirmed_identity_reference(
                audio_path,
                _confirmed_identity_ranges(confirmed),
                clip_index=clip_index,
                raw_speaker=raw_speaker,
            )

        if ref_path is None:
            # Last resort only.  Prefer human-confirmed material above; generic
            # raw A/B speaker ranges can themselves contain the identity error.
            ref_path = _render_speaker_reference(
                audio_path,
                scan_segments,
                raw_speaker,
                clip_index,
                f"named_{idx + 1}",
            )
        if ref_path is None:
            # Partial identity is valid. Skip only the unavailable identity.
            continue
        names.append(display_name)
        references.append(_audio_data_url(ref_path))
        paths.append(str(ref_path))
    return names, references, paths


def _verify_anchored_names_on_heldout(
    *,
    scan: dict[str, Any],
    segment_sets: list[list[dict[str, Any]]],
    raw_to_name: dict[str, str],
    auto_check_names: set[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Cross-check known-speaker labels on human-verified HELD-OUT utterances.

    The reference ranges themselves were sent to OpenAI, so validating on those
    same bytes would be circular. V28 stores a second disjoint utterance that the
    user confirmed as the same person. A name survives only if a majority of the
    anchored diarization passes also finds that name on the held-out interval.
    When no held-out interval exists, the explicit human confirmation is retained
    but marked as not automatically cross-checkable.
    """
    calibration = scan.get("identity_calibration") if isinstance(scan.get("identity_calibration"), dict) else {}
    by_raw = {
        str(row.get("raw_speaker", "")).strip(): row
        for row in calibration.values()
        if isinstance(row, dict) and row.get("human_verified") is True
    }
    trusted: list[str] = []
    details: dict[str, Any] = {}
    required = 2 if len(segment_sets) >= 2 else 1

    auto_check_folded = {str(x).casefold() for x in (auto_check_names or set()) if str(x).strip()}

    for raw, name in raw_to_name.items():
        row = by_raw.get(raw) or {}
        # Short human-preview identities may be too short for OpenAI's 2-10s
        # known-speaker contract. If that name was never submitted as an anchor,
        # do not falsely "verify" it against generic speaker IDs. The human's
        # explicit reference+verification confirmation remains authoritative.
        if name.casefold() not in auto_check_folded:
            trusted.append(name)
            details[name] = {
                "status": "human_confirmed_short_preview_no_model_anchor",
                "votes": None,
                "required": None,
            }
            continue

        ranges: list[tuple[float, float]] = []
        for pair in row.get("verification_ranges", []) or []:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            try:
                start = float(pair[0]); end = float(pair[1])
            except (TypeError, ValueError):
                continue
            if end > start:
                ranges.append((start, end))

        if not ranges:
            trusted.append(name)
            details[name] = {
                "status": "human_confirmed_no_heldout_auto_check",
                "votes": None,
                "required": None,
            }
            continue

        ratios: list[float] = []
        votes = 0
        total_duration = sum(end - start for start, end in ranges)
        for segments in segment_sets:
            matched = 0.0
            for start, end in ranges:
                for seg in segments:
                    if str(seg.get("speaker", "")).strip().casefold() != name.casefold():
                        continue
                    matched += _overlap(
                        start,
                        end,
                        float(seg.get("start", 0.0) or 0.0),
                        float(seg.get("end", 0.0) or 0.0),
                    )
            ratio = min(1.0, matched / max(0.04, total_duration))
            ratios.append(ratio)
            if ratio >= 0.55:
                votes += 1

        passed = votes >= required
        if passed:
            trusted.append(name)
        details[name] = {
            "status": "passed" if passed else "failed",
            "votes": votes,
            "required": required,
            "pass_overlap_ratios": [round(value, 4) for value in ratios],
            "heldout_ranges": [[round(a, 3), round(b, 3)] for a, b in ranges],
        }

    return trusted, details


def _human_verified_identity_map(
    scan: dict[str, Any],
    raw_to_name: dict[str, str],
) -> dict[str, str]:
    """Return only raw->name pairs the human explicitly confirmed by voice.

    The interactive preview is the authoritative identity action. Model-based
    known-speaker passes may add diagnostics, but they must never silently erase
    a human-confirmed name from the final ASS. Speaker WORD ownership remains
    Gold V7 and is not changed here.
    """
    calibration = scan.get("identity_calibration") if isinstance(scan.get("identity_calibration"), dict) else {}
    verified: dict[str, str] = {}
    for row in calibration.values():
        if not isinstance(row, dict) or row.get("human_verified") is not True:
            continue
        raw = str(row.get("raw_speaker", "")).strip()
        name = " ".join(str(row.get("name", "")).split()).strip()
        expected = " ".join(str(raw_to_name.get(raw, "")).split()).strip()
        if raw and name and expected and name.casefold() == expected.casefold():
            verified[raw] = expected
    return verified


def _disjoint_holdout_checkable_names(
    scan: dict[str, Any],
    raw_to_name: dict[str, str],
) -> set[str]:
    """Names whose verification utterance was not needed to build the reference.

    A short human reference may be stitched with its verification range to meet
    the 2-second known-speaker contract. In that case the same verification
    bytes cannot honestly be counted again as held-out model evidence.
    """
    calibration = scan.get("identity_calibration") if isinstance(scan.get("identity_calibration"), dict) else {}
    result: set[str] = set()
    for row in calibration.values():
        if not isinstance(row, dict) or row.get("human_verified") is not True:
            continue
        raw = str(row.get("raw_speaker", "")).strip()
        name = " ".join(str(row.get("name", "")).split()).strip()
        expected = " ".join(str(raw_to_name.get(raw, "")).split()).strip()
        if not raw or not name or not expected or name.casefold() != expected.casefold():
            continue
        # This flag is set only when the original human reference itself already
        # satisfies the 2-10s known-speaker contract. Therefore its verification
        # utterance remains truly disjoint and may be used as a held-out check.
        if row.get("known_speaker_reference_eligible") is True and (row.get("verification_ranges") or []):
            result.add(expected)
    return result


def _run_human_identity_anchor(
    *,
    audio_path: Path,
    duration: float,
    scan: dict[str, Any],
    clip_index: int,
) -> dict[str, Any]:
    """Whole-clip known-speaker anchor after the human A/B identity check.

    One automatic-chunking pass supplies broad named-speaker evidence. Extra
    identity work is reserved for short risky phrases later in the pipeline,
    where two independently padded local windows must agree. This changes WHO
    metadata only; caption text and Whisper timestamps remain untouched.
    """
    raw_to_name = _manual_identity_map(scan)
    if not raw_to_name:
        return {"status": "not_applicable", "segment_sets": []}

    scan_segments = [dict(x) for x in (scan.get("segments") or []) if isinstance(x, dict)]
    names, references, reference_paths = _identity_reference_bundle(
        audio_path=audio_path,
        scan=scan,
        scan_segments=scan_segments,
        raw_to_name=raw_to_name,
        clip_index=clip_index,
    )
    if not names:
        return {
            "status": "reference_unavailable",
            "segment_sets": [],
            "raw_to_name": raw_to_name,
        }

    segment_sets: list[list[dict[str, Any]]] = []
    errors: list[str] = []
    # One whole-clip auto pass gives broad named-speaker coverage.  Previous
    # explicit server_vad passes produced `chunking_strategy is required` 400s
    # in real runs and contributed no evidence.  V6 spends any extra identity
    # calls only on short risky phrases, where chunking is not required.
    specs = [("auto", None, None, None)]
    for label, threshold, prefix_ms, silence_ms in specs:
        try:
            data = _run_diarization(
                audio_path,
                chunking=label,
                vad_threshold=threshold,
                vad_prefix_padding_ms=prefix_ms,
                vad_silence_ms=silence_ms,
                known_speaker_names=names,
                known_speaker_references=references,
            )
            normalized = _normalize_segments(data, duration)
            if normalized:
                segment_sets.append(normalized)
            else:
                errors.append(f"{label}: no segments")
        except Exception as error:
            errors.append(f"{label}: {error}")

    trusted_names, heldout_verification = _verify_anchored_names_on_heldout(
        scan=scan,
        segment_sets=segment_sets,
        raw_to_name=raw_to_name,
        auto_check_names=_disjoint_holdout_checkable_names(scan, raw_to_name),
    ) if segment_sets else (
        list(raw_to_name.values()) if raw_to_name else [],
        {name: {"status": "human_confirmed_no_model_anchor_available"} for name in raw_to_name.values()},
    )

    return {
        "status": (
            "ok" if segment_sets and trusted_names
            else ("heldout_verification_failed" if segment_sets else ("human_only" if trusted_names else "failed"))
        ),
        "segment_sets": segment_sets,
        "known_names": trusted_names,
        "requested_known_names": names,
        "raw_to_name": raw_to_name,
        "reference_paths": reference_paths,
        "passes_completed": len(segment_sets),
        "heldout_verification": heldout_verification,
        "errors": errors,
        "model": DIARIZATION_MODEL,
        "policy": (
            "human-confirmed 2-10s voice refs -> one whole-clip known-speaker anchor -> "
            "disjoint held-out check when the held-out audio was not consumed by reference stitching"
        ),
    }







# V6 human-known-voice ownership lock.  The existing identity anchor already
# performs a known-speaker diarization pass using the exact human-confirmed
# reference WAVs.  V5 used that pass only to validate the cosmetic A/B -> name
# map, while word ownership still came exclusively from generic raw A/B turns.
# That allowed a swallowed speaker boundary to render TYLA as KAI (or vice
# versa).  V6 allows the already-paid known-speaker evidence to correct WHO
# metadata only.  Wording and Whisper timestamps remain immutable.
IDENTITY_PHRASE_BREAK_GAP_SECONDS = max(
    0.35,
    min(1.20, float(os.getenv("MIMIR_IDENTITY_PHRASE_BREAK_GAP_SECONDS", "0.62") or 0.62)),
)
IDENTITY_DIRECT_MIN_CONFIDENCE = max(
    0.70,
    min(0.95, float(os.getenv("MIMIR_IDENTITY_DIRECT_MIN_CONFIDENCE", "0.78") or 0.78)),
)
IDENTITY_MICRO_ENABLED = str(os.getenv("MIMIR_IDENTITY_MICRO_ENABLED", "1")).strip().lower() not in {"0", "false", "no", "off"}
IDENTITY_MICRO_RISK_GAP_SECONDS = max(0.55, min(2.5, float(os.getenv("MIMIR_IDENTITY_MICRO_RISK_GAP_SECONDS", "0.90") or 0.90)))
IDENTITY_MICRO_MAX_PHRASES = max(1, min(8, int(os.getenv("MIMIR_IDENTITY_MICRO_MAX_PHRASES", "5") or 5)))
IDENTITY_MICRO_MIN_TARGET_COVERAGE = max(0.30, min(0.90, float(os.getenv("MIMIR_IDENTITY_MICRO_MIN_TARGET_COVERAGE", "0.48") or 0.48)))


def _known_name_pass_candidates(
    words: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    allowed_names: list[str],
) -> list[dict[str, Any]]:
    """Map immutable caption words onto direct KNOWN-SPEAKER segment names."""
    allowed = {
        " ".join(str(name).split()).strip().casefold():
        " ".join(str(name).split()).strip()
        for name in allowed_names
        if " ".join(str(name).split()).strip()
    }
    provisional, _ = _provisional_assignments(words, segments)
    result: list[dict[str, Any]] = []
    for row in provisional:
        candidate = " ".join(str(row.get("speaker_candidate", "")).split()).strip()
        canonical = allowed.get(candidate.casefold(), "")
        result.append(
            {
                "name": canonical,
                "confidence": (
                    float(row.get("speaker_candidate_confidence", 0.0) or 0.0)
                    if canonical else 0.0
                ),
                "source": str(row.get("speaker_evidence_source", "")),
            }
        )
    return result


def _apply_human_known_voice_overlay(
    *,
    words: list[dict[str, Any]],
    segment_sets: list[list[dict[str, Any]]],
    raw_to_name: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Correct WHO from human-confirmed known-speaker evidence, fail closed.

    Generic A/B diarization remains useful discovery evidence, but once the
    human has named two voices and a known-speaker anchor pass exists, strong
    direct-name lexical evidence outranks generic raw A/B ownership.  A weak
    timing-only raw guess is never allowed to print a human name merely because
    an A->KAI/B->TYLA mapping exists.
    """
    result = [dict(row) for row in words]
    audit: dict[str, Any] = {
        "status": "not_needed",
        "passes": 0,
        "resolved_words": 0,
        "remapped_words": 0,
        "cleared_weak_words": 0,
        "conflicts": 0,
        "policy": (
            "human-confirmed known-speaker evidence may change WHO metadata only; "
            "text and edited_start/edited_end are immutable"
        ),
    }
    if not result or len(raw_to_name) < 2 or not segment_sets:
        return result, audit

    allowed_names = [
        " ".join(str(name).split()).strip()
        for name in raw_to_name.values()
        if " ".join(str(name).split()).strip()
    ]
    pass_candidates: list[list[dict[str, Any]]] = []
    for segments in segment_sets:
        if not isinstance(segments, list) or not segments:
            continue
        candidates = _known_name_pass_candidates(result, segments, allowed_names)
        if len(candidates) == len(result):
            pass_candidates.append(candidates)
    if not pass_candidates:
        audit["status"] = "no_usable_anchor_pass"
        return result, audit

    audit["passes"] = len(pass_candidates)
    inverse = {
        " ".join(str(name).split()).strip().casefold(): str(raw)
        for raw, name in raw_to_name.items()
        if " ".join(str(name).split()).strip()
    }
    raw_order = list(raw_to_name.keys())
    raw_role = {
        raw: ("main" if index == 0 else ("secondary" if index == 1 else "tertiary"))
        for index, raw in enumerate(raw_order)
    }
    raw_display = {
        str(raw): " ".join(str(name).split()).strip()
        for raw, name in raw_to_name.items()
    }

    for index, word in enumerate(result):
        votes: dict[str, list[tuple[float, str]]] = {}
        for view in pass_candidates:
            row = view[index]
            name = " ".join(str(row.get("name", "")).split()).strip()
            confidence = float(row.get("confidence", 0.0) or 0.0)
            source = str(row.get("source", ""))
            if name:
                votes.setdefault(name.casefold(), []).append((confidence, source))

        chosen_name = ""
        chosen_conf = 0.0
        if len(pass_candidates) >= 2:
            ranked = sorted(
                votes.items(),
                key=lambda item: (len(item[1]), sum(v[0] for v in item[1])),
                reverse=True,
            )
            if ranked:
                folded, evidence = ranked[0]
                # At least two independent anchored views must agree.
                if len(evidence) >= 2:
                    avg_conf = sum(v[0] for v in evidence) / len(evidence)
                    if avg_conf >= IDENTITY_DIRECT_MIN_CONFIDENCE:
                        chosen_name = next(
                            (name for name in allowed_names if name.casefold() == folded),
                            "",
                        )
                        chosen_conf = min(v[0] for v in evidence)
        elif votes:
            # The current API occasionally gives us only the auto-chunking
            # anchored view.  One pass may override raw A/B only with lexical
            # evidence, never timing-only overlap.
            folded, evidence = max(
                votes.items(),
                key=lambda item: max((v[0] for v in item[1]), default=0.0),
            )
            conf, source = max(evidence, key=lambda item: item[0])
            strong_single = (
                (source == "text+timing_agree" and conf >= 0.84)
                or (source == "text_alignment" and conf >= 0.80)
            )
            if strong_single:
                chosen_name = next(
                    (name for name in allowed_names if name.casefold() == folded),
                    "",
                )
                chosen_conf = conf

        original_raw = str(word.get("speaker_raw", ""))
        original_label = str(word.get("speaker_label", ""))
        if chosen_name:
            canonical_raw = inverse.get(chosen_name.casefold(), "")
            if canonical_raw:
                word.setdefault("speaker_raw_before_known_voice", original_raw)
                word.setdefault("speaker_label_before_known_voice", original_label)
                word["speaker_raw"] = canonical_raw
                word["speaker_role"] = raw_role.get(canonical_raw, "main")
                word["speaker_label"] = chosen_name
                word["speaker_confidence"] = round(float(chosen_conf), 3)
                word["speaker_evidence_source"] = "human_known_voice_anchor"
                word["identity_anchor_name"] = chosen_name
                audit["resolved_words"] += 1
                if original_raw != canonical_raw:
                    audit["remapped_words"] += 1
                    audit["conflicts"] += 1
                continue

        raw_conf = float(word.get("speaker_confidence", 0.0) or 0.0)
        raw_source = str(word.get("speaker_evidence_source", ""))
        weak_raw = (
            not original_raw
            or raw_conf < 0.70
            or raw_source in {
                "timing_only",
                "continuity_fill",
                "continuity_fill_local",
                "unresolved_diarization_gap",
                "turn_hysteresis",
            }
        )
        if weak_raw:
            word.setdefault("speaker_raw_before_known_voice", original_raw)
            word.setdefault("speaker_label_before_known_voice", original_label)
            word["speaker_raw"] = ""
            word["speaker_role"] = "main"
            word["speaker_label"] = ""
            word["speaker_confidence"] = 0.0
            word["speaker_evidence_source"] = "human_known_voice_unresolved"
            if original_raw or raw_display.get(original_raw, ""):
                audit["cleared_weak_words"] += 1

    if audit["remapped_words"]:
        audit["status"] = "resolved_with_remaps"
    elif audit["resolved_words"]:
        audit["status"] = "resolved"
    else:
        audit["status"] = "no_strong_direct_identity"
    return result, audit


def _identity_phrase_ranges(words: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Build short spoken utterances without changing caption geometry."""
    if not words:
        return []
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(words)):
        previous = words[index - 1]
        current = words[index]
        previous_text = str(previous.get("word", "")).strip()
        gap = max(
            0.0,
            float(current.get("edited_start", 0.0) or 0.0)
            - float(previous.get("edited_end", 0.0) or 0.0),
        )
        sentence_end = bool(re.search(r"[.!?][\"')\]]*$", previous_text))
        too_long = (index - start) >= 12
        if sentence_end or gap >= IDENTITY_PHRASE_BREAK_GAP_SECONDS or too_long:
            ranges.append((start, index))
            start = index
    ranges.append((start, len(words)))
    return ranges


def _lock_human_identity_by_phrase(
    *,
    words: list[dict[str, Any]],
    raw_to_name: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Never render two human identities inside one ordinary spoken phrase.

    The winner is chosen only from direct human-known-voice anchor evidence.
    If strong anchored voices conflict inside the same punctuation/gap-delimited
    phrase and neither clearly dominates, the WHO label for the entire phrase is
    blank.  This is intentionally fail-closed: an unlabeled caption is better
    than KAI/TYLA changing halfway through one utterance.
    """
    result = [dict(row) for row in words]
    audit: dict[str, Any] = {
        "status": "not_needed",
        "locked_phrases": 0,
        "cleared_conflict_phrases": 0,
        "words_reassigned": 0,
        "words_cleared": 0,
        "policy": "one punctuation/gap-delimited utterance -> one verified human identity or UNKNOWN",
    }
    if not result or len(raw_to_name) < 2:
        return result, audit

    inverse = {
        " ".join(str(name).split()).strip().casefold(): str(raw)
        for raw, name in raw_to_name.items()
        if " ".join(str(name).split()).strip()
    }
    raw_order = list(raw_to_name.keys())
    raw_role = {
        raw: ("main" if index == 0 else ("secondary" if index == 1 else "tertiary"))
        for index, raw in enumerate(raw_order)
    }

    for start, end in _identity_phrase_ranges(result):
        phrase = result[start:end]
        if not phrase:
            continue
        direct: dict[str, list[float]] = {}
        for row in phrase:
            name = " ".join(str(row.get("identity_anchor_name", "")).split()).strip()
            confidence = float(row.get("speaker_confidence", 0.0) or 0.0)
            if name and confidence >= IDENTITY_DIRECT_MIN_CONFIDENCE:
                direct.setdefault(name.casefold(), []).append(confidence)
        if not direct:
            continue

        ranked = sorted(
            direct.items(),
            key=lambda item: (len(item[1]), sum(item[1])),
            reverse=True,
        )
        winner_folded, winner_evidence = ranked[0]
        runner_evidence = ranked[1][1] if len(ranked) > 1 else []
        winner_count = len(winner_evidence)
        runner_count = len(runner_evidence)
        winner_score = sum(winner_evidence)
        runner_score = sum(runner_evidence)
        conflict = bool(runner_evidence)

        # A phrase with conflicting direct voice evidence is published with a
        # name only when one identity clearly dominates both count and weight.
        dominant = (
            not conflict
            or (
                winner_count >= max(2, runner_count + 1)
                and winner_score >= max(1.35 * runner_score, runner_score + 0.55)
            )
        )
        winner_name = next(
            (name for name in raw_to_name.values() if str(name).casefold() == winner_folded),
            "",
        )
        canonical_raw = inverse.get(winner_folded, "") if dominant else ""

        if dominant and canonical_raw and winner_name:
            phrase_conf = round(min(winner_evidence), 3)
            for row in phrase:
                previous_raw = str(row.get("speaker_raw", ""))
                if previous_raw != canonical_raw:
                    audit["words_reassigned"] += 1
                row.setdefault("speaker_raw_before_phrase_lock", previous_raw)
                row["speaker_raw"] = canonical_raw
                row["speaker_role"] = raw_role.get(canonical_raw, "main")
                row["speaker_label"] = str(winner_name)
                row["speaker_confidence"] = phrase_conf
                row["speaker_evidence_source"] = "human_known_voice_phrase_lock"
                row["identity_phrase_name"] = str(winner_name)
            audit["locked_phrases"] += 1
        else:
            for row in phrase:
                if str(row.get("speaker_raw", "")) or str(row.get("speaker_label", "")):
                    audit["words_cleared"] += 1
                row.setdefault("speaker_raw_before_phrase_lock", str(row.get("speaker_raw", "")))
                row["speaker_raw"] = ""
                row["speaker_role"] = "main"
                row["speaker_label"] = ""
                row["speaker_confidence"] = 0.0
                row["speaker_evidence_source"] = "human_known_voice_phrase_conflict"
            audit["cleared_conflict_phrases"] += 1

    if audit["cleared_conflict_phrases"]:
        audit["status"] = "locked_with_fail_closed_conflicts"
    elif audit["locked_phrases"]:
        audit["status"] = "locked"
    else:
        audit["status"] = "no_direct_phrase_evidence"
    return result, audit


def _enforce_one_identity_per_phrase(
    *,
    words: list[dict[str, Any]],
    raw_to_name: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Final fail-closed invariant: one spoken phrase gets one name or none.

    Even after all identity evidence, renderer-visible KAI/TYLA layers must never
    alternate inside one phrase. A mixed or partially unresolved phrase is safer
    unlabeled than confidently assigned to the wrong person. Text and timing are
    never touched.
    """
    result = [dict(row) for row in words]
    known_raw = {str(raw) for raw, name in raw_to_name.items() if str(raw) and str(name).strip()}
    audit: dict[str, Any] = {
        "status": "clean",
        "phrases_checked": 0,
        "phrases_cleared": 0,
        "words_cleared": 0,
        "policy": "one phrase = one human identity or UNKNOWN; mixed/partial identity fails closed",
    }
    if not result or len(known_raw) < 2:
        audit["status"] = "not_applicable"
        return result, audit

    for start, end in _identity_phrase_ranges(result):
        phrase = result[start:end]
        if not phrase:
            continue
        audit["phrases_checked"] += 1
        raws = [str(row.get("speaker_raw", "")).strip() for row in phrase]
        named = {raw for raw in raws if raw in known_raw}
        unresolved = any(raw not in known_raw for raw in raws)
        if len(named) <= 1 and not (named and unresolved):
            continue
        for row in phrase:
            if str(row.get("speaker_raw", "")).strip() or str(row.get("speaker_label", "")).strip():
                audit["words_cleared"] += 1
            row["speaker_raw"] = ""
            row["speaker_role"] = "main"
            row["speaker_label"] = ""
            row["speaker_confidence"] = 0.0
            row["speaker_evidence_source"] = "identity_phrase_final_fail_closed"
        audit["phrases_cleared"] += 1

    if audit["phrases_cleared"]:
        audit["status"] = "cleared_mixed_or_partial_phrases"
    return result, audit


def _render_identity_micro_window(
    audio_path: Path,
    *,
    start: float,
    end: float,
    clip_index: int,
    phrase_index: int,
    view_index: int,
) -> Path:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    output = TEMP_DIR / (
        f"clip_{int(clip_index):02d}_identity_phrase_{int(phrase_index):02d}_"
        f"view_{int(view_index):02d}.wav"
    )
    length = max(0.0, float(end) - float(start))
    if length < 0.50:
        raise RuntimeError("Identity micro window too short")
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{float(start):.3f}", "-t", f"{length:.3f}",
        "-i", str(audio_path), "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le",
        str(output),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
    )
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size < 1024:
        raise RuntimeError(completed.stderr.strip() or "identity micro ffmpeg failed")
    return output


def _identity_micro_window_bounds(
    *,
    phrase_start: float,
    phrase_end: float,
    duration: float,
    pad: float,
) -> tuple[float, float]:
    start = max(0.0, phrase_start - pad)
    end = min(duration, phrase_end + pad)
    minimum = 2.40
    if end - start < minimum:
        shortage = minimum - (end - start)
        left = min(start, shortage / 2.0)
        start -= left
        shortage -= left
        end = min(duration, end + shortage)
        if end - start < minimum and start > 0:
            start = max(0.0, end - minimum)
    return round(start, 3), round(end, 3)


def _micro_identity_vote(
    segments: list[dict[str, Any]],
    *,
    phrase_start_local: float,
    phrase_end_local: float,
    allowed_names: list[str],
) -> tuple[str, float, dict[str, float]]:
    duration = max(0.10, phrase_end_local - phrase_start_local)
    allowed = {name.casefold(): name for name in allowed_names}
    scores: dict[str, float] = {name: 0.0 for name in allowed_names}
    for segment in segments:
        raw_name = " ".join(str(segment.get("speaker", "")).split()).strip()
        canonical = allowed.get(raw_name.casefold(), "")
        if not canonical:
            continue
        amount = _overlap(
            phrase_start_local,
            phrase_end_local,
            float(segment.get("start", 0.0) or 0.0),
            float(segment.get("end", 0.0) or 0.0),
        )
        scores[canonical] = scores.get(canonical, 0.0) + amount
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] <= 0.0:
        return "", 0.0, scores
    top_name, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    coverage = min(1.0, top / duration)
    if coverage < IDENTITY_MICRO_MIN_TARGET_COVERAGE:
        return "", coverage, scores
    if second > 0.0 and not (top >= second * 1.35 and top - second >= 0.10):
        return "", coverage, scores
    return top_name, coverage, scores



def _identity_phrase_is_filler(phrase: list[dict[str, Any]]) -> bool:
    filler = {"uh", "um", "erm", "hmm", "hm", "mm", "mhm"}
    tokens: list[str] = []
    for row in phrase:
        value = _norm_token(str(row.get("word", "")))
        if value:
            tokens.append(value)
    return bool(tokens) and all(token in filler for token in tokens)

def _micro_verify_risky_identity_phrases(
    *,
    audio_path: Path,
    duration: float,
    scan: dict[str, Any],
    scan_segments: list[dict[str, Any]],
    words: list[dict[str, Any]],
    raw_to_name: dict[str, str],
    clip_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Two local known-voice ears decide only risky phrase identities.

    A phrase is risky when it follows a substantial gap, contains unresolved or
    conflicting identity evidence, or still relies on weak WHO confidence.  Two
    short (<30s) windows must agree on the same human-confirmed voice; otherwise
    the entire phrase is left unlabeled.  This prevents a confident-looking
    whole-clip diarization mistake from becoming a visible KAI/TYLA swap.
    """
    result = [dict(row) for row in words]
    audit: dict[str, Any] = {
        "status": "not_needed",
        "checked_phrases": 0,
        "resolved_phrases": 0,
        "unresolved_phrases": 0,
        "api_calls_attempted": 0,
        "details": [],
        "policy": "two short known-speaker views agree -> one phrase identity; disagreement -> UNKNOWN",
    }
    if not IDENTITY_MICRO_ENABLED or len(raw_to_name) < 2 or not result:
        return result, audit

    names, references, reference_paths = _identity_reference_bundle(
        audio_path=audio_path,
        scan=scan,
        scan_segments=scan_segments,
        raw_to_name=raw_to_name,
        clip_index=clip_index,
    )
    if len(names) < 2 or len(names) != len(references):
        audit["status"] = "reference_unavailable"
        audit["reference_paths"] = reference_paths
        return result, audit
    audit["reference_paths"] = reference_paths

    inverse = {str(name).casefold(): str(raw) for raw, name in raw_to_name.items()}
    raw_order = list(raw_to_name.keys())
    raw_role = {
        raw: ("main" if index == 0 else ("secondary" if index == 1 else "tertiary"))
        for index, raw in enumerate(raw_order)
    }

    phrase_ranges = _identity_phrase_ranges(result)
    # A pure long-gap recheck is useful only after the clip has already shown a
    # real speaker alternation. Before the first verified switch, a setup pause
    # does not imply an A/B identity-namespace reset and is not worth an API call.
    first_verified_switch_time = float("inf")
    previous_raw = ""
    for row in result:
        current_raw = str(row.get("speaker_raw", ""))
        if not current_raw:
            continue
        if previous_raw and current_raw != previous_raw:
            first_verified_switch_time = float(row.get("edited_start", 0.0) or 0.0)
            break
        previous_raw = current_raw

    risky_scored: list[tuple[float, int, int, list[str]]] = []
    for phrase_index, (start_i, end_i) in enumerate(phrase_ranges):
        phrase = result[start_i:end_i]
        if not phrase:
            continue
        if _identity_phrase_is_filler(phrase):
            # Filler is cheap to leave unlabeled; do not spend identity API
            # budget proving who said a standalone "uh/um".
            continue
        phrase_start = float(phrase[0].get("edited_start", 0.0) or 0.0)
        previous_end = (
            float(result[start_i - 1].get("edited_end", phrase_start) or phrase_start)
            if start_i > 0 else phrase_start
        )
        pre_gap = max(0.0, phrase_start - previous_end)
        sources = {str(row.get("speaker_evidence_source", "")) for row in phrase}
        labels = {str(row.get("speaker_label", "")).strip() for row in phrase if str(row.get("speaker_label", "")).strip()}
        min_conf = min(float(row.get("speaker_confidence", 0.0) or 0.0) for row in phrase)
        reasons: list[str] = []
        score = 0.0
        if (
            pre_gap >= IDENTITY_MICRO_RISK_GAP_SECONDS
            and phrase_start >= first_verified_switch_time
        ):
            reasons.append(f"pre_gap={pre_gap:.3f}")
            score += 10.0 * pre_gap
        if any("conflict" in source or "unresolved" in source for source in sources):
            reasons.append("identity_conflict_or_unresolved")
            score += 100.0
        if len(labels) > 1:
            reasons.append("mixed_visible_labels")
            score += 120.0
        if min_conf < 0.70:
            reasons.append(f"weak_identity_conf={min_conf:.3f}")
            score += 30.0 * (0.70 - min_conf)
        if reasons:
            # Slightly favor later ties: post-gap identity resets often surface
            # near the tail of a short, exactly like the Kaityla regression.
            score += min(2.0, phrase_start / max(1.0, duration))
            risky_scored.append((score, start_i, end_i, reasons))

    risky_scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    risky = [(start_i, end_i, reasons) for _, start_i, end_i, reasons in risky_scored[:IDENTITY_MICRO_MAX_PHRASES]]
    risky.sort(key=lambda item: item[0])

    if not risky:
        audit["status"] = "clean"
        return result, audit

    for phrase_number, (start_i, end_i, reasons) in enumerate(risky, 1):
        phrase = result[start_i:end_i]
        phrase_start = float(phrase[0].get("edited_start", 0.0) or 0.0)
        phrase_end = float(phrase[-1].get("edited_end", phrase_start) or phrase_start)
        votes: list[tuple[str, float]] = []
        view_details: list[dict[str, Any]] = []
        # Strict 2/2 local consensus. The whole-clip identity anchor is NOT a
        # vote here; it is precisely the source we are auditing. This prevents a
        # confident but wrong global KAI/TYLA label from self-confirming.
        for view_index, pad in enumerate((0.75, 1.35), 1):
            window_start, window_end = _identity_micro_window_bounds(
                phrase_start=phrase_start,
                phrase_end=phrase_end,
                duration=duration,
                pad=pad,
            )
            try:
                micro_path = _render_identity_micro_window(
                    audio_path,
                    start=window_start,
                    end=window_end,
                    clip_index=clip_index,
                    phrase_index=phrase_number,
                    view_index=view_index,
                )
                audit["api_calls_attempted"] += 1
                data = _run_diarization(
                    micro_path,
                    chunking="auto",
                    known_speaker_names=names,
                    known_speaker_references=references,
                )
                micro_duration = max(0.0, window_end - window_start)
                segments = _normalize_segments(data, micro_duration)
                name, coverage, scores = _micro_identity_vote(
                    segments,
                    phrase_start_local=max(0.0, phrase_start - window_start),
                    phrase_end_local=max(0.0, phrase_end - window_start),
                    allowed_names=names,
                )
                if name:
                    votes.append((name, coverage))
                view_details.append(
                    {
                        "view": view_index,
                        "window": [window_start, window_end],
                        "vote": name,
                        "coverage": round(coverage, 4),
                        "scores": {key: round(value, 4) for key, value in scores.items()},
                    }
                )
            except Exception as error:
                view_details.append({"view": view_index, "error": str(error)})

        resolved_name = ""
        if len(votes) >= 2 and votes[0][0].casefold() == votes[1][0].casefold():
            resolved_name = votes[0][0]
        canonical_raw = inverse.get(resolved_name.casefold(), "") if resolved_name else ""
        if canonical_raw:
            confidence = round(min(v[1] for v in votes), 3)
            for row in phrase:
                row["speaker_raw"] = canonical_raw
                row["speaker_role"] = raw_role.get(canonical_raw, "main")
                row["speaker_label"] = str(raw_to_name.get(canonical_raw, resolved_name))
                row["speaker_confidence"] = confidence
                row["speaker_evidence_source"] = "human_known_voice_micro_consensus"
                row["identity_micro_name"] = resolved_name
            audit["resolved_phrases"] += 1
        else:
            for row in phrase:
                row["speaker_raw"] = ""
                row["speaker_role"] = "main"
                row["speaker_label"] = ""
                row["speaker_confidence"] = 0.0
                row["speaker_evidence_source"] = "human_known_voice_micro_unresolved"
            audit["unresolved_phrases"] += 1

        audit["checked_phrases"] += 1
        audit["details"].append(
            {
                "phrase_words": [str(row.get("word", "")) for row in phrase],
                "range": [round(phrase_start, 3), round(phrase_end, 3)],
                "reasons": reasons,
                "views": view_details,
                "resolved_name": resolved_name,
            }
        )

    audit["status"] = "checked"
    return result, audit






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
    Luna-low classifies real participants versus crowd/noise, then the human
    naming checkpoint can run before the heavy video encode.
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
        "display_labels": ({primary: "A", secondary: "B"} if mode == "dual" and primary and secondary else {}),
        "speaker_stats": stats,
        "segments": segments,
        "words": [],
        "assignment": {},
        "diarization_status": "ok" if segments else "fallback_single",
        "diarization_error": diarization_error,
        "policy": {
            "speaker_algorithm": "48k raw auto diarization -> conditional sensitive retry -> conditional known-speaker anchored confirmation -> Gold V7 assignment",
            "render_order": "audio scan -> optional high-confidence 2/3-person audio identity -> video render",
            "identity_prompt": "only high-confidence 2/3 real speakers; otherwise plain captions without a teaser",
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
    it. The returned profile is sufficient for an optional 2/3-speaker audio
    identity checkpoint; uncertain identity never blocks plain captions.
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
            "display_labels": (
                {primary: "A", secondary: "B"}
                if mode == "dual" and primary and secondary else {}
            ),
            "speaker_stats": stats,
            "segments": segments,
            "words": [],
            "assignment": {},
            "diarization_status": "ok" if segments else "fallback_single",
            "diarization_error": diarization_error,
            "policy": {
                "speaker_algorithm": "Gold V7 assignment + Luna-low participant/crowd role judge",
                "api_order": "diarization alone first; Luna-low role judge; human naming; video render later",
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
) -> Path:
    """Build the final caption profile without letting identity recalibrate words.

    Gold rule restored:
    1. Final 48 kHz audio produces WORDS + immutable Whisper CLOCK.
    2. Pre-render diarization segments feed the proven Gold V7 turn assignment.
    3. Human voice references only validate raw-speaker -> display-name mapping.
       They never re-segment caption words and never touch timestamps/text.
    """
    edited_clip_path = Path(edited_clip_path).resolve()
    output = _output_path(edited_clip_path, clip_index)
    audio_path: Path | None = None
    # Keep diagnostic fallback construction safe even if an early probe/read
    # fails before the scan metadata is loaded.
    scan: dict[str, Any] = {}
    inherited_names: dict[str, Any] = {}
    inherited_labels: dict[str, Any] = {}

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
        inherited_names = scan.get("speaker_names") if isinstance(scan.get("speaker_names"), dict) else {}
        inherited_labels = scan.get("display_labels") if isinstance(scan.get("display_labels"), dict) else {}

        caption_known_names: list[str] = []
        for value in [
            *(inherited_names.get(key, "") for key in ("A", "B", "C")),
            *inherited_labels.values(),
        ]:
            clean = " ".join(str(value).split()).strip()
            if clean and clean.upper() not in {"A", "B", "C"} and clean not in caption_known_names:
                caption_known_names.append(clean)

        words, alignment_ratio, text_source, caption_quality = _transcribe_edited_words(
            audio_path,
            duration,
            known_names=caption_known_names,
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

        force_plain_captions = bool(scan.get("force_plain_captions", False))
        if force_plain_captions:
            mode, primary, secondary, kept, background_speakers = "unresolved", None, None, [], []

        # IMPORTANT: Gold V7 owns word->raw-speaker assignment for EVERY clip,
        # including clips whose real names were human-calibrated.
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

        # Human identity anchor: once the user has confirmed A/B voices, the
        # existing known-speaker pass is allowed to correct WHO metadata only.
        # Text and Whisper timing remain immutable.
        identity_anchor: dict[str, Any] = {"status": "not_applicable", "segment_sets": []}
        identity_overlay: dict[str, Any] = {"status": "not_applicable"}
        identity_phrase_lock: dict[str, Any] = {"status": "not_applicable"}
        identity_micro_verify: dict[str, Any] = {"status": "not_applicable"}
        identity_final_guard: dict[str, Any] = {"status": "not_applicable"}
        manual_map = _manual_identity_map(scan)
        validated_identity_map: dict[str, str] = {}
        if manual_map and not force_plain_captions:
            # A name entered after the user listened to that raw speaker's
            # reference is already human-verified identity evidence. Preserve
            # that raw-id -> name map as the final DISPLAY mapping. Optional
            # model anchors remain diagnostic only and can never erase it.
            validated_identity_map.update(_human_verified_identity_map(scan, manual_map))
            try:
                identity_anchor = _run_human_identity_anchor(
                    audio_path=audio_path,
                    duration=duration,
                    scan=scan,
                    clip_index=clip_index,
                )
            except Exception as error:
                identity_anchor = {"status": "failed", "segment_sets": [], "errors": [str(error)]}

            trusted = {
                " ".join(str(name).split()).strip().casefold()
                for name in (identity_anchor.get("known_names") or [])
                if " ".join(str(name).split()).strip()
            }
            for raw_speaker, display_name in manual_map.items():
                clean_name = " ".join(str(display_name).split()).strip()
                if clean_name and clean_name.casefold() in trusted:
                    validated_identity_map[str(raw_speaker)] = clean_name
            if isinstance(identity_anchor, dict):
                identity_anchor["human_verified_display_map"] = dict(validated_identity_map)
                identity_anchor["model_anchor_can_erase_human_name"] = False

        display_labels: dict[str, str] = {}
        if manual_map:
            # Partial validation is intentional: a trusted Kai can be labeled
            # while an uncertain second voice remains unlabeled.
            display_labels = dict(validated_identity_map)
        elif not force_plain_captions:
            if mode in {"dual", "triple"}:
                for index, speaker in enumerate(participants[:3]):
                    key = chr(ord("A") + index)
                    label = str(inherited_labels.get(speaker) or inherited_names.get(key) or key).strip()
                    if label:
                        display_labels[speaker] = label
            elif mode == "single" and primary:
                label = str(inherited_labels.get(primary) or inherited_names.get("A") or "").strip()
                if label:
                    display_labels[primary] = label

        # V6: do not skip the rescue merely because generic Gold V7 degraded
        # effective_mode to `unresolved`.  The whole point of the human-known
        # voice anchor is to rescue WHO when raw A/B ownership is weak.
        if manual_map and not force_plain_captions and len(manual_map) >= 2:
            segment_sets = [
                [dict(row) for row in segment_set if isinstance(row, dict)]
                for segment_set in (identity_anchor.get("segment_sets") or [])
                if isinstance(segment_set, list)
            ]
            assigned_words, identity_overlay = _apply_human_known_voice_overlay(
                words=assigned_words,
                segment_sets=segment_sets,
                raw_to_name=manual_map,
            )
            assigned_words, identity_phrase_lock = _lock_human_identity_by_phrase(
                words=assigned_words,
                raw_to_name=manual_map,
            )
            assigned_words, identity_micro_verify = _micro_verify_risky_identity_phrases(
                audio_path=audio_path,
                duration=duration,
                scan=scan,
                scan_segments=segments,
                words=assigned_words,
                raw_to_name=manual_map,
                clip_index=clip_index,
            )
            assigned_words, identity_final_guard = _enforce_one_identity_per_phrase(
                words=assigned_words,
                raw_to_name=manual_map,
            )

        for raw in assigned_words:
            if not isinstance(raw, dict):
                continue
            speaker = str(raw.get("speaker_raw") or "")
            # One final label projection prevents stale KAI/TYLA text after a
            # raw-speaker remap. UNKNOWN always renders without a human name.
            raw["speaker_label"] = str(display_labels.get(speaker, ""))

        if isinstance(caption_quality, dict):
            caption_quality["speaker_mutates_text"] = False
            caption_quality["speaker_mutates_timestamps"] = False
            caption_quality["identity_word_assignment"] = "gold_v7_plus_human_known_voice_phrase_lock_plus_risky_micro_consensus"
            caption_quality["identity_label_validation"] = str(identity_anchor.get("status", "not_applicable"))
            caption_quality["identity_overlay"] = dict(identity_overlay)
            caption_quality["identity_phrase_lock"] = dict(identity_phrase_lock)
            caption_quality["identity_micro_verify"] = dict(identity_micro_verify)
            caption_quality["identity_final_guard"] = dict(identity_final_guard)

        profile = {
            "version": SPEAKER_PROFILE_VERSION,
            "status": "ok",
            "phase": "final_profile",
            "timing_basis": "exact_final_48k_audio",
            "timing_model": str((caption_quality or {}).get("timing_model", vod_processor.TIMING_MODEL)),
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
            "display_labels": display_labels,
            "speaker_names": inherited_names,
            "speaker_stats": stats,
            "speaker_boundary_quality": float(scan.get("speaker_boundary_quality", 0.0) or 0.0),
            "speaker_boundary_audit": dict(scan.get("speaker_boundary_audit", {}) or {}),
            "segments": segments,
            "words": assigned_words,
            "assignment": assignment,
            "identity_anchor": identity_anchor,
            "identity_overlay": identity_overlay,
            "identity_phrase_lock": identity_phrase_lock,
            "identity_micro_verify": identity_micro_verify,
            "identity_final_guard": identity_final_guard,
            "validated_identity_map": validated_identity_map,
            "diarization_status": "ok" if segments else "fallback_single",
            "diarization_error": diarization_error,
            "policy": {
                "word_truth": "gpt-transcribe; speaker layer cannot rewrite text",
                "clock_truth": "whisper-1 raw native word clock; no post calibration",
                "speaker_truth": "Gold V7 discovery + human-confirmed known-speaker phrase lock + risky local two-view consensus",
                "identity_truth": "human voice references may remap speaker metadata only; text/time immutable",
                "uncertain_identity": "risky phrase needs two local known-voice votes; otherwise no identity label",
            },
            "compatibility": {
                "transcript_path": str(transcript_path) if transcript_path else None,
                "timeline_path": str(timeline_path) if timeline_path else None,
                "speaker_scan_path": str(speaker_scan_path) if speaker_scan_path else None,
            },
        }
    except Exception as error:
        # Diagnostic fallback only. Preserve the PRE-RENDER human identity
        # metadata so a clock failure can never look like "the user never named
        # the speakers". The pipeline refuses to publish legacy-timed captions
        # from this profile (see shorts_pipeline mandatory final-clock guard).
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
            "display_labels": dict(inherited_labels),
            "speaker_names": dict(inherited_names),
            "force_plain_captions": bool(scan.get("force_plain_captions", False)),
            "error": str(error),
        }
    finally:
        if audio_path is not None:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass

    return _write_json(output, profile)

