from __future__ import annotations

import array
import math
import shutil
import subprocess
from pathlib import Path
from statistics import median
from typing import Any


MEME_AUDIO_SUPPORT_VERSION = 1

SAMPLE_RATE = 16000
FRAME_SECONDS = 0.080
HOP_SECONDS = 0.040
MAX_ANCHOR_HINTS = 18
MAX_PAUSE_WINDOWS = 10
MAX_AUDIO_PEAKS = 8


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dbfs_from_samples(samples: array.array) -> float:
    if not samples:
        return -96.0
    square_sum = 0.0
    for sample in samples:
        value = int(sample)
        square_sum += value * value
    rms = math.sqrt(square_sum / max(1, len(samples)))
    return 20.0 * math.log10(max(rms, 1.0) / 32768.0)


def _decode_pcm(video_path: str | Path | None, duration_hint: float) -> array.array:
    if not video_path or shutil.which("ffmpeg") is None:
        return array.array("h")

    target = Path(video_path).expanduser().resolve()
    if not target.is_file():
        return array.array("h")

    command = [
        "ffmpeg", "-v", "error", "-i", str(target),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
    ]

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=max(30.0, min(150.0, duration_hint * 2.5 + 20.0)),
        )
    except Exception:
        return array.array("h")

    if completed.returncode != 0 or not completed.stdout:
        return array.array("h")

    result = array.array("h")
    try:
        result.frombytes(completed.stdout)
    except Exception:
        return array.array("h")
    return result


def _build_frames(samples: array.array) -> list[dict[str, float]]:
    if not samples:
        return []

    frame_size = max(1, int(FRAME_SECONDS * SAMPLE_RATE))
    hop_size = max(1, int(HOP_SECONDS * SAMPLE_RATE))
    frames: list[dict[str, float]] = []

    for start in range(0, max(1, len(samples) - frame_size + 1), hop_size):
        end = min(len(samples), start + frame_size)
        if end <= start:
            continue
        chunk = samples[start:end]
        dbfs = _dbfs_from_samples(chunk)
        center = ((start + end) / 2.0) / SAMPLE_RATE
        frames.append({"time": center, "dbfs": dbfs})

    return frames


def _window_db(frames: list[dict[str, float]], start: float, end: float, fallback: float) -> float:
    if end <= start:
        return fallback
    values = [f["dbfs"] for f in frames if start <= f["time"] <= end]
    return median(values) if values else fallback


def _window_peak_db(frames: list[dict[str, float]], start: float, end: float, fallback: float) -> float:
    values = [f["dbfs"] for f in frames if start <= f["time"] <= end]
    return max(values) if values else fallback


def _level_score(dbfs: float, floor: float, ceiling: float) -> float:
    return _clamp((dbfs - floor) / max(4.0, ceiling - floor), 0.0, 1.0)


def _quiet_score(dbfs: float, floor: float, ceiling: float) -> float:
    return 1.0 - _level_score(dbfs, floor, ceiling)


def _overlap_risk(gap_after: float) -> str:
    if gap_after >= 0.42:
        return "low"
    if gap_after >= 0.18:
        return "medium"
    return "high"


def _recommended_duration(gap_after: float, overlap_risk: str) -> float:
    if overlap_risk == "low":
        return _clamp(gap_after + 0.10, 0.34, 1.25)
    if overlap_risk == "medium":
        return _clamp(gap_after + 0.08, 0.26, 0.68)
    return 0.30


def _recommended_delay_ms(gap_after: float, post_quiet: float) -> int:
    if gap_after >= 0.35 and post_quiet >= 0.55:
        return 70
    if gap_after >= 0.18:
        return 45
    return 20


def _anchor_hints(
    frames: list[dict[str, float]],
    words: list[dict[str, Any]],
    teaser_duration: float,
    main_duration: float,
    floor: float,
    ceiling: float,
) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []

    for index, word in enumerate(words):
        try:
            word_id = int(word["id"])
            main_start = float(word["edited_start"])
            main_end = float(word["edited_end"])
        except (KeyError, TypeError, ValueError):
            continue

        if index + 1 < len(words):
            try:
                next_start = float(words[index + 1]["edited_start"])
            except (KeyError, TypeError, ValueError):
                next_start = main_duration
        else:
            next_start = main_duration

        gap_after = max(0.0, next_start - main_end)
        final_end = teaser_duration + main_end

        pre_db = _window_db(frames, max(0.0, final_end - 0.28), final_end + 0.02, floor)
        post_db = _window_db(frames, final_end + 0.05, final_end + 0.42, floor)
        local_peak_db = _window_peak_db(frames, max(0.0, final_end - 0.32), final_end + 0.24, floor)

        post_quiet = _quiet_score(post_db, floor, ceiling)
        pre_level = _level_score(pre_db, floor, ceiling)
        peak_level = _level_score(local_peak_db, floor, ceiling)
        drop_db = pre_db - post_db
        drop_score = _clamp(drop_db / 9.0, 0.0, 1.0)
        gap_score = _clamp(gap_after / 0.65, 0.0, 1.0)

        risk = _overlap_risk(gap_after)
        headroom = _clamp(0.48 * post_quiet + 0.30 * gap_score + 0.22 * drop_score, 0.0, 1.0)
        punctuation = _clamp(0.34 * peak_level + 0.30 * drop_score + 0.22 * gap_score + 0.14 * post_quiet, 0.0, 1.0)
        score = _clamp(0.62 * headroom + 0.38 * punctuation, 0.0, 1.0)

        hints.append({
            "word_id": word_id,
            "word": str(word.get("word", "")).strip(),
            "main_word_start": round(main_start, 3),
            "main_word_end": round(main_end, 3),
            "final_word_end": round(final_end, 3),
            "gap_after_seconds": round(gap_after, 3),
            "pre_dbfs": round(pre_db, 2),
            "post_dbfs": round(post_db, 2),
            "drop_db": round(drop_db, 2),
            "post_quiet_score": round(post_quiet, 3),
            "local_peak_score": round(peak_level, 3),
            "insertion_headroom_score": round(headroom, 3),
            "punctuation_score": round(punctuation, 3),
            "support_score": round(score, 3),
            "speech_overlap_risk": risk,
            "recommended_delay_ms": _recommended_delay_ms(gap_after, post_quiet),
            "recommended_max_duration": round(_recommended_duration(gap_after, risk), 3),
        })

    hints.sort(
        key=lambda item: (
            -float(item["support_score"]),
            -float(item["insertion_headroom_score"]),
            int(item["word_id"]),
        )
    )
    return hints[:MAX_ANCHOR_HINTS]


def _pause_windows(
    words: list[dict[str, Any]],
    frames: list[dict[str, float]],
    teaser_duration: float,
    floor: float,
    ceiling: float,
) -> list[dict[str, Any]]:
    pauses: list[dict[str, Any]] = []

    for index in range(len(words) - 1):
        current = words[index]
        nxt = words[index + 1]
        try:
            start = float(current["edited_end"])
            end = float(nxt["edited_start"])
            word_id = int(current["id"])
        except (KeyError, TypeError, ValueError):
            continue

        duration = end - start
        if duration < 0.16:
            continue

        final_start = teaser_duration + start
        final_end = teaser_duration + end
        dbfs = _window_db(frames, final_start, final_end, floor)
        quiet = _quiet_score(dbfs, floor, ceiling)
        score = _clamp(0.58 * _clamp(duration / 0.75, 0.0, 1.0) + 0.42 * quiet, 0.0, 1.0)

        pauses.append({
            "after_word_id": word_id,
            "after_word": str(current.get("word", "")).strip(),
            "main_start": round(start, 3),
            "main_end": round(end, 3),
            "duration": round(duration, 3),
            "dbfs": round(dbfs, 2),
            "quiet_score": round(quiet, 3),
            "pause_score": round(score, 3),
        })

    pauses.sort(key=lambda item: (-float(item["pause_score"]), -float(item["duration"])))
    return pauses[:MAX_PAUSE_WINDOWS]


def _audio_peaks(
    frames: list[dict[str, float]],
    teaser_duration: float,
    main_duration: float,
    floor: float,
    ceiling: float,
) -> list[dict[str, Any]]:
    main_start = teaser_duration
    main_end = teaser_duration + main_duration
    candidates: list[dict[str, Any]] = []

    for index, frame in enumerate(frames):
        t = frame["time"]
        if not (main_start <= t <= main_end):
            continue

        radius = max(1, int(0.22 / HOP_SECONDS))
        left = max(0, index - radius)
        right = min(len(frames), index + radius + 1)
        if frame["dbfs"] < max(x["dbfs"] for x in frames[left:right]):
            continue

        level = _level_score(frame["dbfs"], floor, ceiling)
        if level < 0.58:
            continue

        candidates.append({
            "main_time": round(max(0.0, t - teaser_duration), 3),
            "final_time": round(t, 3),
            "dbfs": round(frame["dbfs"], 2),
            "peak_score": round(level, 3),
        })

    candidates.sort(key=lambda item: (-float(item["peak_score"]), float(item["main_time"])))
    selected: list[dict[str, Any]] = []
    for item in candidates:
        if any(abs(float(item["main_time"]) - float(other["main_time"])) < 0.65 for other in selected):
            continue
        selected.append(item)
        if len(selected) >= MAX_AUDIO_PEAKS:
            break
    return selected


def analyze_meme_audio_support(
    video_path: str | Path | None,
    *,
    words: list[dict[str, Any]],
    teaser_duration: float,
    main_duration: float,
) -> dict[str, Any]:
    """Analyze final-preview audio for safe, punchy SFX insertion opportunities.

    This is factual support only. It does not decide whether a meme is funny.
    Terra/meme_analyzer still makes the editorial decision.
    """
    duration_hint = max(0.1, teaser_duration + main_duration)
    samples = _decode_pcm(video_path, duration_hint)
    frames = _build_frames(samples)

    if not frames:
        return {
            "version": MEME_AUDIO_SUPPORT_VERSION,
            "available": False,
            "source": str(video_path or ""),
            "summary": "Audio envelope unavailable; use transcript-only judgment.",
            "anchor_hints": [],
            "pause_windows": [],
            "audio_peaks": [],
        }

    values = sorted(frame["dbfs"] for frame in frames)
    floor = median(values)
    p90 = values[min(len(values) - 1, int(len(values) * 0.90))]
    ceiling = max(p90, floor + 6.0)

    hints = _anchor_hints(
        frames,
        words,
        teaser_duration,
        main_duration,
        floor,
        ceiling,
    )
    pauses = _pause_windows(words, frames, teaser_duration, floor, ceiling)
    peaks = _audio_peaks(frames, teaser_duration, main_duration, floor, ceiling)

    return {
        "version": MEME_AUDIO_SUPPORT_VERSION,
        "available": True,
        "source": str(Path(video_path).expanduser().resolve()) if video_path else "",
        "global_floor_dbfs": round(floor, 2),
        "global_ceiling_dbfs": round(ceiling, 2),
        "summary": (
            "Factual audio support for meme timing. High insertion_headroom_score + "
            "low speech_overlap_risk means a short SFX can land cleanly after the word."
        ),
        "anchor_hints": hints,
        "pause_windows": pauses,
        "audio_peaks": peaks,
    }


def get_anchor_hint(audio_support: dict[str, Any] | None, word_id: int) -> dict[str, Any] | None:
    if not isinstance(audio_support, dict):
        return None
    raw = audio_support.get("anchor_hints", [])
    if not isinstance(raw, list):
        return None
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            current = int(item.get("word_id"))
        except (TypeError, ValueError):
            continue
        if current == int(word_id):
            return item
    return None


def format_audio_support_for_ai(audio_support: dict[str, Any] | None) -> str:
    if not isinstance(audio_support, dict) or not audio_support.get("available"):
        return "Audio envelope unavailable. Do not invent audio peaks or pauses."

    lines = [
        "Real audio-envelope support (factual, not an editorial decision):",
        "Anchor hints:",
    ]

    for item in audio_support.get("anchor_hints", [])[:14]:
        if not isinstance(item, dict):
            continue
        lines.append(
            "ID {word_id} '{word}' | gap={gap:.2f}s | headroom={head:.2f} | "
            "punctuation={punct:.2f} | overlap={risk} | suggested_delay={delay}ms | "
            "suggested_max={dur:.2f}s".format(
                word_id=item.get("word_id"),
                word=item.get("word", ""),
                gap=_safe_float(item.get("gap_after_seconds")),
                head=_safe_float(item.get("insertion_headroom_score")),
                punct=_safe_float(item.get("punctuation_score")),
                risk=item.get("speech_overlap_risk", "unknown"),
                delay=int(_safe_float(item.get("recommended_delay_ms"))),
                dur=_safe_float(item.get("recommended_max_duration")),
            )
        )

    pauses = audio_support.get("pause_windows", [])
    if isinstance(pauses, list) and pauses:
        lines.append("Best post-line pause windows:")
        for item in pauses[:6]:
            if not isinstance(item, dict):
                continue
            lines.append(
                "after ID {word_id} '{word}' | pause={duration:.2f}s | quiet={quiet:.2f}".format(
                    word_id=item.get("after_word_id"),
                    word=item.get("after_word", ""),
                    duration=_safe_float(item.get("duration")),
                    quiet=_safe_float(item.get("quiet_score")),
                )
            )

    return "\n".join(lines)
