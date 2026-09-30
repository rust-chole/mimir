from __future__ import annotations

import array
import json
import math
import shutil
import subprocess
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from ai.editor import intro_bounds


PEAK_SUPPORT_VERSION = 2

AUDIO_SAMPLE_RATE = 16000
AUDIO_WINDOW_SECONDS = 0.32
AUDIO_HOP_SECONDS = 0.10
MAX_AUDIO_PEAKS = 10
MAX_VISUAL_PEAKS = 10
MAX_COMBINED_PEAKS = 12
MIN_PEAK_SEPARATION_SECONDS = 0.75
MERGE_DISTANCE_SECONDS = 0.65

VISUAL_EVENT_WEIGHTS: dict[str, float] = {
    "reaction": 1.00,
    "visual_payoff": 1.00,
    "explosion": 1.00,
    "crash": 0.98,
    "impact": 0.96,
    "destruction": 0.94,
    "object_break": 0.93,
    "fall": 0.90,
    "reveal": 0.90,
    "sudden_change": 0.88,
    "gameplay_event": 0.82,
    "physical_action": 0.74,
    "entrance_exit": 0.70,
    "scene_change": 0.62,
    "popup": 0.50,
    "other": 0.45,
}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_report(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        return None
    try:
        package = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(package, dict):
        return None
    report = package.get("report")
    return report if isinstance(report, dict) else None


def analyze_audio_peaks(
    video_path: str | Path | None,
    *,
    clip_duration: float,
) -> list[dict[str, Any]]:
    """Decode mono PCM with ffmpeg and find short-term loudness/suddenness peaks."""
    if not video_path:
        return []

    path = Path(video_path).expanduser().resolve()
    if not path.is_file() or shutil.which("ffmpeg") is None:
        return []

    command = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE),
        "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=max(30.0, min(120.0, clip_duration * 2.5 + 15.0)),
        )
    except Exception:
        return []

    if result.returncode != 0 or not result.stdout:
        return []

    samples = array.array("h")
    try:
        samples.frombytes(result.stdout)
    except Exception:
        return []

    if not samples:
        return []

    window = max(1, int(AUDIO_WINDOW_SECONDS * AUDIO_SAMPLE_RATE))
    hop = max(1, int(AUDIO_HOP_SECONDS * AUDIO_SAMPLE_RATE))
    frames: list[dict[str, float]] = []

    for start_i in range(0, max(1, len(samples) - window + 1), hop):
        end_i = min(len(samples), start_i + window)
        if end_i <= start_i:
            continue
        chunk = samples[start_i:end_i]
        square_sum = 0.0
        peak_abs = 0
        for sample in chunk:
            value = int(sample)
            square_sum += value * value
            peak_abs = max(peak_abs, abs(value))
        rms = math.sqrt(square_sum / max(1, len(chunk)))
        dbfs = 20.0 * math.log10(max(rms, 1.0) / 32768.0)
        peak_dbfs = 20.0 * math.log10(max(float(peak_abs), 1.0) / 32768.0)
        center = ((start_i + end_i) / 2.0) / AUDIO_SAMPLE_RATE
        frames.append({"time": center, "dbfs": dbfs, "peak_dbfs": peak_dbfs})

    if not frames:
        return []

    db_values = [frame["dbfs"] for frame in frames]
    floor = median(db_values)
    sorted_db = sorted(db_values)
    p90 = sorted_db[min(len(sorted_db) - 1, int(len(sorted_db) * 0.90))]
    ceiling = max(p90, max(db_values) - 3.0, floor + 4.0)
    dynamic = max(3.0, ceiling - floor)

    for index, frame in enumerate(frames):
        level = _clamp((frame["dbfs"] - floor) / dynamic, 0.0, 1.0)
        back_start = max(0, index - max(1, int(0.65 / AUDIO_HOP_SECONDS)))
        previous = frames[back_start:index]
        previous_db = median([x["dbfs"] for x in previous]) if previous else floor
        jump = _clamp((frame["dbfs"] - previous_db) / 10.0, 0.0, 1.0)
        near_clip = _clamp((frame["peak_dbfs"] + 6.0) / 6.0, 0.0, 1.0)
        frame["score"] = round(0.62 * level + 0.26 * jump + 0.12 * near_clip, 4)

    local: list[dict[str, float]] = []
    radius = max(1, int(0.40 / AUDIO_HOP_SECONDS))
    for index, frame in enumerate(frames):
        left = max(0, index - radius)
        right = min(len(frames), index + radius + 1)
        if frame["score"] >= max(x["score"] for x in frames[left:right]):
            local.append(frame)

    local.sort(key=lambda x: (-x["score"], -x["dbfs"], x["time"]))
    chosen: list[dict[str, Any]] = []
    for frame in local:
        if frame["score"] < 0.30:
            continue
        if any(abs(frame["time"] - item["center"]) < MIN_PEAK_SEPARATION_SECONDS for item in chosen):
            continue
        center = _clamp(frame["time"], 0.0, clip_duration)
        chosen.append({
            "center": round(center, 3),
            "start": round(max(0.0, center - 0.38), 3),
            "end": round(min(clip_duration, center + 0.48), 3),
            "audio_score": round(float(frame["score"]), 3),
            "dbfs": round(float(frame["dbfs"]), 2),
            "peak_dbfs": round(float(frame["peak_dbfs"]), 2),
            "signals": ["audio_energy_peak"],
        })
        if len(chosen) >= MAX_AUDIO_PEAKS:
            break

    return chosen


def extract_visual_peaks(
    video_report_path: str | Path | None,
    *,
    clip_duration: float,
) -> list[dict[str, Any]]:
    report = _load_report(video_report_path)
    if report is None:
        return []

    dedicated = report.get("intro_visual_support", {})
    dedicated_peaks = dedicated.get("visual_peak_regions", []) if isinstance(dedicated, dict) else []
    result: list[dict[str, Any]] = []

    if isinstance(dedicated_peaks, list):
        for item in dedicated_peaks[:30]:
            if not isinstance(item, dict):
                continue
            start = _clamp(_safe_float(item.get("start")), 0.0, clip_duration)
            end = _clamp(_safe_float(item.get("end"), start), start, clip_duration)
            confidence = _clamp(_safe_float(item.get("confidence")), 0.0, 1.0)
            intensity = _clamp(_safe_float(item.get("observable_intensity")), 0.0, 1.0)
            if confidence < 0.35:
                continue
            signals_raw = item.get("signals", [])
            signals = [str(x).strip() for x in signals_raw if str(x).strip()] if isinstance(signals_raw, list) else []
            result.append({
                "start": round(start, 3),
                "end": round(max(end, min(clip_duration, start + 0.20)), 3),
                "visual_score": round((0.70 * intensity + 0.30 * confidence), 3),
                "confidence": round(confidence, 3),
                "description": str(item.get("description", "")).strip(),
                "signals": signals or ["visual_peak"],
            })

    if not result:
        events = report.get("visual_events", [])
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                event_type = str(event.get("type", "other")).strip().lower()
                confidence = _clamp(_safe_float(event.get("confidence")), 0.0, 1.0)
                weight = VISUAL_EVENT_WEIGHTS.get(event_type, VISUAL_EVENT_WEIGHTS["other"])
                score = weight * confidence
                if score < 0.35:
                    continue
                start = _clamp(_safe_float(event.get("start")), 0.0, clip_duration)
                end = _clamp(_safe_float(event.get("end"), start), start, clip_duration)
                if end - start < 0.20:
                    end = min(clip_duration, start + 0.35)
                result.append({
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "visual_score": round(score, 3),
                    "confidence": round(confidence, 3),
                    "description": str(event.get("description", "")).strip(),
                    "signals": [event_type],
                })

    result.sort(key=lambda x: (-x["visual_score"], x["start"]))
    kept: list[dict[str, Any]] = []
    for item in result:
        center = (item["start"] + item["end"]) / 2.0
        if any(abs(center - ((other["start"] + other["end"]) / 2.0)) < 0.45 for other in kept):
            continue
        kept.append(item)
        if len(kept) >= MAX_VISUAL_PEAKS:
            break
    return kept


def _distance(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    if a_end < b_start:
        return b_start - a_end
    if b_end < a_start:
        return a_start - b_end
    return 0.0


def _window_around(
    start: float,
    end: float,
    clip_duration: float,
    *,
    envelope: intro_bounds.Envelope | None = None,
    words: Sequence[intro_bounds.Word] = (),
) -> tuple[float, float]:
    """The cold-open window the evidence supports for this candidate (no fixed length)."""
    bounds = intro_bounds.compute_intro_bounds(
        intro_bounds.EventEvidence(start, end),
        clip_duration=clip_duration,
        words=words,
        envelope=envelope,
    )
    return round(bounds.start, 3), round(bounds.end, 3)


def build_peak_support(
    *,
    edited_video_path: str | Path | None,
    video_report_path: str | Path | None,
    clip_duration: float,
    envelope: intro_bounds.Envelope | None = None,
    words: Sequence[intro_bounds.Word] = (),
) -> dict[str, Any]:
    clip_duration = max(0.01, float(clip_duration))
    if envelope is None:
        envelope = intro_bounds.audio_envelope(edited_video_path)
    audio = analyze_audio_peaks(edited_video_path, clip_duration=clip_duration)
    visual = extract_visual_peaks(video_report_path, clip_duration=clip_duration)

    candidates: list[dict[str, Any]] = []
    used_visual: set[int] = set()

    # Anchor around audio peaks first and merge nearby visual evidence.
    for audio_peak in audio:
        start = float(audio_peak["start"])
        end = float(audio_peak["end"])
        best_visual_index: int | None = None
        best_distance = 999.0
        for index, visual_peak in enumerate(visual):
            distance = _distance(start, end, float(visual_peak["start"]), float(visual_peak["end"]))
            if distance <= MERGE_DISTANCE_SECONDS and distance < best_distance:
                best_visual_index = index
                best_distance = distance

        visual_score = 0.0
        description = ""
        signals = list(audio_peak.get("signals", []))
        if best_visual_index is not None:
            item = visual[best_visual_index]
            used_visual.add(best_visual_index)
            start = min(start, float(item["start"]))
            end = max(end, float(item["end"]))
            visual_score = float(item["visual_score"])
            description = str(item.get("description", ""))
            signals.extend(item.get("signals", []))

        audio_score = float(audio_peak["audio_score"])
        dual_bonus = 0.16 if visual_score > 0 else 0.0
        combined = _clamp(0.58 * audio_score + 0.42 * visual_score + dual_bonus, 0.0, 1.0)
        teaser_start, teaser_end = _window_around(start, end, clip_duration, envelope=envelope, words=words)
        candidates.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "teaser_start": teaser_start,
            "teaser_end": teaser_end,
            "audio_score": round(audio_score, 3),
            "visual_score": round(visual_score, 3),
            "combined_score": round(combined, 3),
            "signals": sorted(set(str(x) for x in signals if str(x).strip())),
            "visual_description": description,
            "audio_dbfs": audio_peak.get("dbfs"),
        })

    # Add strong visual-only peaks so silent reactions/reveals remain selectable.
    for index, item in enumerate(visual):
        if index in used_visual:
            continue
        visual_score = float(item["visual_score"])
        teaser_start, teaser_end = _window_around(float(item["start"]), float(item["end"]), clip_duration,
                                                  envelope=envelope, words=words)
        candidates.append({
            "start": float(item["start"]),
            "end": float(item["end"]),
            "teaser_start": teaser_start,
            "teaser_end": teaser_end,
            "audio_score": 0.0,
            "visual_score": round(visual_score, 3),
            "combined_score": round(_clamp(0.82 * visual_score, 0.0, 1.0), 3),
            "signals": sorted(set(str(x) for x in item.get("signals", []) if str(x).strip())),
            "visual_description": str(item.get("description", "")),
            "audio_dbfs": None,
        })

    candidates.sort(key=lambda x: (-x["combined_score"], -x["visual_score"], -x["audio_score"], x["start"]))
    deduped: list[dict[str, Any]] = []
    for item in candidates:
        center = (float(item["start"]) + float(item["end"])) / 2.0
        if any(abs(center - ((float(other["start"]) + float(other["end"])) / 2.0)) < 0.55 for other in deduped):
            continue
        deduped.append(item)
        if len(deduped) >= MAX_COMBINED_PEAKS:
            break

    for index, item in enumerate(deduped, start=1):
        item["peak_id"] = index
        item["rank"] = index
        item["multimodal"] = bool(item["audio_score"] > 0 and item["visual_score"] > 0)

    return {
        "version": PEAK_SUPPORT_VERSION,
        "audio_available": bool(audio),
        "visual_available": bool(visual),
        "audio_peaks": audio,
        "visual_peaks": visual,
        "candidates": deduped,
    }


def attach_nearby_words(
    support: dict[str, Any],
    available_words: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = support.get("candidates", [])
    if not isinstance(candidates, list):
        return support

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        start = float(candidate.get("teaser_start", candidate.get("start", 0.0)))
        end = float(candidate.get("teaser_end", candidate.get("end", start)))
        nearby = [
            word for word in available_words
            if float(word.get("edited_end", 0.0)) >= start
            and float(word.get("edited_start", 0.0)) <= end
        ]
        candidate["nearby_word_ids"] = [int(word["id"]) for word in nearby]
        candidate["nearby_text"] = " ".join(str(word.get("word", "")).strip() for word in nearby).strip()
    return support


def format_for_ai(support: dict[str, Any]) -> str:
    candidates = support.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return (
            "INTRO PEAK SUPPORT:\n"
            "No reliable audio/visual peak candidates were available. "
            "Use transcript/editorial evidence only."
        )

    lines = [
        "INTRO PEAK SUPPORT:",
        "These are mechanical/factual support candidates, NOT the final editorial choice.",
        "combined_score rewards audio+visual alignment; Terra still decides the teaser.",
    ]
    for item in candidates:
        signals = ",".join(item.get("signals", [])) or "none"
        text = str(item.get("nearby_text", "")).strip()
        description = str(item.get("visual_description", "")).strip()
        lines.append(
            f"PEAK {item.get('peak_id')} | core {float(item.get('start',0)):.2f}-{float(item.get('end',0)):.2f}s | "
            f"measured_window {float(item.get('teaser_start',0)):.2f}-{float(item.get('teaser_end',0)):.2f}s | "
            f"combined={float(item.get('combined_score',0)):.2f} audio={float(item.get('audio_score',0)):.2f} "
            f"visual={float(item.get('visual_score',0)):.2f} | multimodal={bool(item.get('multimodal'))} | signals={signals}"
        )
        if description:
            lines.append(f"  visual: {description}")
        if text:
            lines.append(f"  nearby speech: {text}")
    return "\n".join(lines)
