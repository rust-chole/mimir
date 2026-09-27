"""Diarization segments -> participant census (single / dual / triple / crowd)."""
from __future__ import annotations

import math
import re
from typing import Any, Sequence

from mimir.models.provider import DiarizedSegment

MIN_REAL_TURN_SECONDS = 0.55
MIN_REAL_TURN_WORDS = 2
MIN_DUAL_SECONDS = 0.85
MIN_DUAL_SHARE = 0.075
MIN_MEANINGFUL_WORDS = 2
MIN_THIRD_REAL_TURNS = 2
MIN_THIRD_STRONG_SECONDS = 2.20
MIN_THIRD_STRONG_WORDS = 5
NOISE_MARKERS = ("[", "(", "music", "applause", "laughter", "cheering", "noise", "inaudible")


def looks_like_noise(text: str) -> bool:
    value = str(text).strip().casefold()
    if not value:
        return True
    return any(marker in value for marker in NOISE_MARKERS) or not re.search(r"[a-z0-9]", value)


def normalize_segments(segments: Sequence[DiarizedSegment], offset: float, limit: float) -> list[dict[str, Any]]:
    """Usable segments only: finite times with positive duration inside the window and a speaker label."""
    rows = []
    for index, segment in enumerate(segments):
        try:
            raw_start, raw_end = float(segment.start), float(segment.end)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(raw_start) and math.isfinite(raw_end)) or not str(segment.speaker or "").strip():
            continue  # checked before max()/min(), which would silently turn NaN into a bound
        start = max(0.0, raw_start) + offset
        end = min(limit, raw_end + offset)
        if end <= start:
            continue
        text = " ".join(str(segment.text).split())
        rows.append({"index": index, "raw_speaker": str(segment.speaker), "start": round(start, 3),
                     "end": round(end, 3), "duration": round(end - start, 3), "text": text,
                     "word_count": len([t for t in text.split() if re.search(r"\w", t)])})
    rows.sort(key=lambda r: (r["start"], r["end"]))
    return rows


def speaker_stats(segments: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    raw: dict[str, dict[str, Any]] = {}
    total = 0.0
    for segment in segments:
        entry = raw.setdefault(segment["raw_speaker"], {
            "raw_speaker": segment["raw_speaker"], "speaking_seconds": 0.0, "segment_count": 0, "word_count": 0,
            "substantial_segment_count": 0, "noise_segment_count": 0})
        seconds = float(segment["duration"])
        total += seconds
        entry["speaking_seconds"] += seconds
        entry["segment_count"] += 1
        entry["word_count"] += int(segment["word_count"])
        noise = looks_like_noise(segment["text"])
        entry["noise_segment_count"] += int(noise)
        if seconds >= MIN_REAL_TURN_SECONDS and segment["word_count"] >= MIN_REAL_TURN_WORDS and not noise:
            entry["substantial_segment_count"] += 1
    rows = []
    for entry in raw.values():
        count = max(1, entry["segment_count"])
        rows.append({**entry, "speaking_seconds": round(entry["speaking_seconds"], 3),
                     "share": round(entry["speaking_seconds"] / max(total, 1e-6), 4),
                     "avg_segment_seconds": round(entry["speaking_seconds"] / count, 3),
                     "avg_words_per_segment": round(entry["word_count"] / count, 3),
                     "noise_segment_ratio": round(entry["noise_segment_count"] / count, 3)})
    rows.sort(key=lambda r: (r["speaking_seconds"], r["word_count"], r["substantial_segment_count"]), reverse=True)
    return rows


def participant_like(item: dict[str, Any]) -> bool:
    if item["substantial_segment_count"] >= 1 and item["speaking_seconds"] >= 0.55 and item["word_count"] >= 2 \
            and item["noise_segment_ratio"] < 0.67:
        return True
    return (item["speaking_seconds"] >= 0.95 and item["word_count"] >= 3 and item["avg_segment_seconds"] >= 0.32
            and item["avg_words_per_segment"] >= 1.2 and item["noise_segment_ratio"] < 0.50)


def strong_third(item: dict[str, Any]) -> bool:
    """A third speaker needs stronger evidence (crowd cheers / one-off shouts are not participants)."""
    if not participant_like(item):
        return False
    return item["substantial_segment_count"] >= MIN_THIRD_REAL_TURNS or (
        item["speaking_seconds"] >= MIN_THIRD_STRONG_SECONDS and item["word_count"] >= MIN_THIRD_STRONG_WORDS)


def classify(stats: Sequence[dict[str, Any]]) -> tuple[str, list[str], list[str]]:
    """(mode, participant raw ids in order, background raw ids).

    ``unresolved``: the diarizer produced segments but none of them is participant-like speech
    (noise, one-off fragments). That is no evidence of a single speaker.
    """
    if not stats:
        return "silent", [], []
    participants = [s for s in stats if participant_like(s)]
    background = [s["raw_speaker"] for s in stats if s not in participants]
    if not participants:
        return "unresolved", [], [s["raw_speaker"] for s in stats]
    primary = participants[0]["raw_speaker"]
    if len(participants) == 1:
        return "single", [primary], background
    second = participants[1]
    if not (second["speaking_seconds"] >= MIN_DUAL_SECONDS and second["word_count"] >= MIN_MEANINGFUL_WORDS
            and (second["share"] >= MIN_DUAL_SHARE or second["speaking_seconds"] >= 2.0)):
        return "single", [primary], background + [p["raw_speaker"] for p in participants[1:]]
    thirds = [p for p in participants[2:] if strong_third(p)]
    if len(thirds) == 1:
        chosen = [primary, second["raw_speaker"], thirds[0]["raw_speaker"]]
        return "triple", chosen, [s["raw_speaker"] for s in stats if s["raw_speaker"] not in chosen]
    if len(thirds) > 1:
        chosen = [p["raw_speaker"] for p in participants]
        return "crowd", chosen, background
    chosen = [primary, second["raw_speaker"]]
    return "dual", chosen, [s["raw_speaker"] for s in stats if s["raw_speaker"] not in chosen]
