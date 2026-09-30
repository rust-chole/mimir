"""MIGRATION FALLBACK word-timing provider: the legacy Whisper word clock, isolated.

This is NOT the normal final-caption path. It runs only when the configured
primary alignment provider (Qwen3-ForcedAligner) cannot produce a valid clock
for the frozen transcript, and then it re-times the WHOLE frozen transcript
by itself: one timing authority per successful run, never mixed with, and
never mutating, another provider's output.

Contents (moved unchanged from ``vod_processor`` / ``speaker_caption_support``):

- ``align_text_to_fixed_whisper_clock``: maps FROZEN wording onto whisper-1
  native word onsets (exact anchors, local inserts/replacements inside the
  gap before the next anchor, previous-tail borrowing). The frozen words are
  never changed; coincident onsets may group two words into one timed unit.
- ``apply_local_acoustic_clock_guard``: backward-only exact-PCM correction of
  a proven late phrase-start Whisper anchor.

Whole-VOD scouting keeps its own Whisper timing in ``vod_processor``.
"""
from __future__ import annotations

import array
import math
import os
import statistics
import sys
import wave
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Sequence

from ai.vod_processor import (
    TIMING_MODEL,
    canonical_word,
    extract_timing_words,
    round_time,
    tokenize_accurate_text,
)


# ============================================================
# FIXED WHISPER WORD CLOCK
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


# ============================================================
# EXACT-PCM PHRASE-START GUARD (legacy clock only)
# ============================================================

CAPTION_CLOCK_GUARD_ENABLED = str(os.getenv("MIMIR_CAPTION_CLOCK_GUARD_ENABLED", "1")).strip().lower() not in {"0", "false", "no", "off"}
CAPTION_CLOCK_GUARD_MIN_PRE_GAP = max(0.35, float(os.getenv("MIMIR_CAPTION_CLOCK_GUARD_MIN_PRE_GAP", "0.50") or 0.50))
CAPTION_CLOCK_GUARD_MAX_BACKSHIFT = max(0.30, min(1.50, float(os.getenv("MIMIR_CAPTION_CLOCK_GUARD_MAX_BACKSHIFT", "1.20") or 1.20)))
CAPTION_CLOCK_GUARD_MIN_SHIFT = max(0.10, min(0.50, float(os.getenv("MIMIR_CAPTION_CLOCK_GUARD_MIN_SHIFT", "0.18") or 0.18)))
CAPTION_CLOCK_GUARD_ORIGINAL_MAX_SCORE = max(0.02, min(0.40, float(os.getenv("MIMIR_CAPTION_CLOCK_GUARD_ORIGINAL_MAX_SCORE", "0.18") or 0.18)))
CAPTION_CLOCK_GUARD_MIN_IMPROVEMENT = max(0.10, min(0.80, float(os.getenv("MIMIR_CAPTION_CLOCK_GUARD_MIN_IMPROVEMENT", "0.30") or 0.30)))
CAPTION_CLOCK_GUARD_MIN_TARGET_SCORE = max(0.20, min(0.90, float(os.getenv("MIMIR_CAPTION_CLOCK_GUARD_MIN_TARGET_SCORE", "0.34") or 0.34)))
CAPTION_CLOCK_GUARD_FRAME_SECONDS = 0.020
CAPTION_CLOCK_GUARD_MAX_GROUP_SECONDS = max(0.40, min(3.0, float(os.getenv("MIMIR_CAPTION_CLOCK_GUARD_MAX_GROUP_SECONDS", "2.20") or 2.20)))


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


def apply_local_acoustic_clock_guard(
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
            row["timing_source"] = str(row.get("timing_source", TIMING_MODEL)) + "_acoustic_guard"
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


# ============================================================
# PROVIDER
# ============================================================

class LegacyWhisperAlignmentFallback:
    """``WordAlignmentProvider`` over whisper-1 word timestamps of the analysis audio."""

    name = "legacy_whisper"
    max_input_seconds = 600.0

    def __init__(self, transcribe: Any = None) -> None:
        self._transcribe = transcribe

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": TIMING_MODEL, "role": "migration_fallback",
                "clock_guard": "backward-only exact-PCM phrase-start guard" if CAPTION_CLOCK_GUARD_ENABLED else "off"}

    def align(self, audio: Any, window: tuple[float, float], tokens: Sequence[str], language: str,
              workdir: Path) -> list[Any]:
        from ai import vod_processor
        from ai.caption_stack import audio as analysis_audio
        from ai.caption_stack.alignment import AlignedUnit, AlignmentError

        del language   # whisper-1 uses the pipeline transcription language (vod_processor)
        start, end = float(window[0]), float(window[1])
        if start <= 1e-6 and end >= audio.duration - 1e-6:
            path, offset = audio.path, 0.0
            end = audio.duration
        else:
            path, offset, end = analysis_audio.cut_window(audio, start, end, Path(workdir) / (
                f"whisper_{int(start * 1000):08d}_{int(end * 1000):08d}.wav"))
        length = end - offset
        transcribe = self._transcribe or vod_processor.transcribe_word_timing
        timing = transcribe(path, known_names=None)
        if not extract_timing_words(timing or {}):
            raise AlignmentError("whisper-1 returned no word timestamps")
        rows, _ratio = align_text_to_fixed_whisper_clock(" ".join(tokens), timing, length)
        edited = [{"word": r["word"], "edited_start": r["start"], "edited_end": r["end"],
                   "timing_source": r.get("alignment_source", TIMING_MODEL)} for r in rows]
        edited, _guard = apply_local_acoustic_clock_guard(audio_path=Path(path), words=edited, duration=length)
        units: list[Any] = []
        cursor = 0
        for row in edited:
            count = len(tokenize_accurate_text(str(row["word"])))
            if count <= 0 or " ".join(tokens[cursor:cursor + count]) != " ".join(str(row["word"]).split()):
                raise AlignmentError("legacy Whisper clock changed the frozen wording")
            units.append(AlignedUnit(cursor, cursor + count, round(offset + float(row["edited_start"]), 3),
                                     round(offset + float(row["edited_end"]), 3),
                                     f"{self.name}:{row.get('timing_source', TIMING_MODEL)}"))
            cursor += count
        if cursor != len(tokens):
            raise AlignmentError("legacy Whisper clock lost frozen words")
        return units
