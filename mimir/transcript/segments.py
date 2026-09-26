"""Sentence-like segments over timed words (for editorial prompts and pacing context)."""
from __future__ import annotations

from typing import Any, Sequence

MAX_SEGMENT_WORDS = 35
MAX_SEGMENT_SECONDS = 10.0
SEGMENT_GAP = 0.70


def build_segments(words: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        if current:
            start, end = float(current[0]["start"]), float(current[-1]["end"])
            segments.append({
                "id": len(segments),
                "start": round(start, 3),
                "end": round(end, 3),
                "text": " ".join(str(w["text"]) for w in current),
                "word_ids": [w["id"] for w in current],
            })
            current.clear()

    for index, word in enumerate(words):
        current.append(word)
        nxt = words[index + 1] if index + 1 < len(words) else None
        duration = float(current[-1]["end"]) - float(current[0]["start"])
        text = str(word["text"]).strip()
        end_here = (
            len(current) >= MAX_SEGMENT_WORDS
            or duration >= MAX_SEGMENT_SECONDS
            or (len(current) >= 3 and text.endswith((".", "!", "?")))
            or (nxt is not None and float(nxt["start"]) - float(word["end"]) >= SEGMENT_GAP)
        )
        if end_here:
            flush()
    flush()
    return segments


def timestamped_lines(segments: Sequence[dict[str, Any]], *, speakers: dict[str, str] | None = None) -> str:
    lines = []
    for segment in segments:
        text = " ".join(str(segment.get("text", "")).split())
        if not text:
            continue
        prefix = ""
        if speakers and segment.get("speaker"):
            prefix = f"{speakers.get(segment['speaker'], segment['speaker'])}: "
        lines.append(f"[{float(segment['start']):.2f} - {float(segment['end']):.2f}] {prefix}{text}")
    return "\n".join(lines)
