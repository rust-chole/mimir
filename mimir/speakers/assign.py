"""Assign caption words to speakers without touching text or times.

Evidence per word:
* text evidence  - the diarizer's own transcript aligned to the caption words
                   (robust against boundary drift), rejected when far in time;
* timing evidence - overlap with a diarized segment (weaker, capped).
Then impossible single-word ping-pong is absorbed; unresolved words stay
UNRESOLVED across real gaps (no identity is ever invented to look tidy).
A word that two voices' own diarized text both claim at that time, or that only
timing places inside two voices' simultaneous turns, is AMBIGUOUS: it stays
unresolved (source ``ambiguous_overlap``) instead of getting a confident owner.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Sequence

NEAR_SEGMENT_TOLERANCE = 0.34
TEXT_MATCH_CONFIDENCE = 0.92
TIMING_ONLY_MAX_CONFIDENCE = 0.72
TEXT_SOFT_DRIFT = 0.55
TEXT_MAX_DRIFT = 1.25
MAX_FILL_GAP = 0.48
MAX_SINGLE_WORD_FLIP = 0.48
MIN_STABLE_TURN_WORDS = 2
MIN_STABLE_TURN_SECONDS = 0.42
AMBIGUOUS_TIMING_SHARE = 0.5   # a word this much inside two voices' turns has no timing owner


def _norm(value: str) -> str:
    return re.sub(r"[^\w']+", "", str(value).casefold().replace("’", "'")).strip("'")


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _distance(t: float, segment: dict[str, Any]) -> float:
    if segment["start"] <= t <= segment["end"]:
        return 0.0
    return min(abs(t - segment["start"]), abs(t - segment["end"]))


def _timing_evidence(word: dict[str, Any], segments: Sequence[dict[str, Any]]) -> tuple[str, float]:
    start, end = float(word["start"]), float(word["end"])
    duration = max(0.04, end - start)
    best, amount = None, 0.0
    for segment in segments:
        value = _overlap(start, end, segment["start"], segment["end"])
        if value > amount:
            best, amount = segment, value
    if best is not None:
        return best["speaker"], min(TIMING_ONLY_MAX_CONFIDENCE, amount / duration)
    if not segments:
        return "", 0.0
    mid = (start + end) / 2
    nearest = min(segments, key=lambda s: _distance(mid, s))
    distance = _distance(mid, nearest)
    if distance <= NEAR_SEGMENT_TOLERANCE:
        return nearest["speaker"], max(0.10, 1.0 - distance / NEAR_SEGMENT_TOLERANCE) * 0.42
    return "", 0.0


def _timing_ambiguous(word: dict[str, Any], segments: Sequence[dict[str, Any]]) -> bool:
    start, end = float(word["start"]), float(word["end"])
    duration = max(0.04, end - start)
    inside: dict[str, float] = {}
    for segment in segments:
        share = _overlap(start, end, segment["start"], segment["end"]) / duration
        inside[segment["speaker"]] = max(inside.get(segment["speaker"], 0.0), share)
    return sum(1 for share in inside.values() if share >= AMBIGUOUS_TIMING_SHARE) >= 2


def _text_evidence(words: Sequence[dict[str, Any]], segments: Sequence[dict[str, Any]]
                   ) -> tuple[dict[int, tuple[str, float]], set[int]]:
    """Align each diarized segment's own text to the caption words in its time range.

    Local (per segment) alignment keeps interleaved speech from two voices apart;
    a word claimed by segments of one speaker goes to the best time fit. A word claimed
    within normal drift by two DIFFERENT speakers is returned as ambiguous: the position
    estimate inside a segment is too weak to decide who said it.
    """
    claims: dict[int, tuple[float, str, float]] = {}
    firm: dict[int, set[str]] = {}
    norms = [_norm(w["text"]) for w in words]
    for segment in segments:
        tokens = [_norm(t) for t in segment["text"].split() if _norm(t)]
        if not tokens:
            continue
        a, b = segment["start"] - TEXT_MAX_DRIFT, segment["end"] + TEXT_MAX_DRIFT
        indices = [i for i, w in enumerate(words) if float(w["end"]) > a and float(w["start"]) < b]
        if not indices:
            continue
        span = max(0.06, segment["end"] - segment["start"])
        matcher = SequenceMatcher(a=[norms[i] for i in indices], b=tokens, autojunk=False)
        for block in matcher.get_matching_blocks():
            for k in range(block.size):
                wi = indices[block.a + k]
                approx = segment["start"] + span * (block.b + k + 0.5) / len(tokens)
                mid = (float(words[wi]["start"]) + float(words[wi]["end"])) / 2
                inside = segment["start"] - 0.05 <= mid <= segment["end"] + 0.05
                drift = 0.0 if inside else min(abs(mid - segment["start"]), abs(mid - segment["end"]))
                if drift > TEXT_MAX_DRIFT:
                    continue
                fit = drift + 0.25 * abs(mid - approx) / span
                confidence = TEXT_MATCH_CONFIDENCE if drift <= TEXT_SOFT_DRIFT else 0.80
                if drift <= TEXT_SOFT_DRIFT:
                    firm.setdefault(wi, set()).add(segment["speaker"])
                if wi not in claims or fit < claims[wi][0]:
                    claims[wi] = (fit, segment["speaker"], confidence)
    ambiguous = {wi for wi, speakers in firm.items() if len(speakers) >= 2}
    return ({wi: (speaker, confidence) for wi, (_, speaker, confidence) in claims.items() if wi not in ambiguous},
            ambiguous)


def _runs(rows: Sequence[dict[str, Any]]) -> list[tuple[int, int, str]]:
    runs: list[tuple[int, int, str]] = []
    start = 0
    for index in range(1, len(rows) + 1):
        if index == len(rows) or rows[index]["speaker"] != rows[start]["speaker"]:
            runs.append((start, index, rows[start]["speaker"]))
            start = index
    return runs


def assign_speakers(words: Sequence[dict[str, Any]], segments: Sequence[dict[str, Any]]
                    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``segments`` already carry anonymous participant ids in ``speaker`` (background removed).

    Returns one row per word: {"speaker", "confidence", "source"} plus metrics.
    """
    text_map, text_ambiguous = _text_evidence(words, segments)
    rows: list[dict[str, Any]] = []
    agree = disagree = 0
    for index, word in enumerate(words):
        timing_speaker, timing_conf = _timing_evidence(word, segments)
        text_speaker, text_conf = text_map.get(index, ("", 0.0))
        if index in text_ambiguous or (not text_speaker and _timing_ambiguous(word, segments)):
            rows.append({"speaker": "", "confidence": 0.0, "source": "ambiguous_overlap"})
        elif text_speaker and text_speaker == timing_speaker:
            rows.append({"speaker": text_speaker, "confidence": min(0.99, max(text_conf, timing_conf) + 0.05),
                         "source": "text+timing"})
            agree += 1
        elif text_speaker:
            rows.append({"speaker": text_speaker, "confidence": text_conf, "source": "text_alignment"})
            disagree += int(bool(timing_speaker))
        elif timing_speaker:
            rows.append({"speaker": timing_speaker, "confidence": timing_conf, "source": "timing_only"})
        else:
            rows.append({"speaker": "", "confidence": 0.0, "source": "unresolved"})

    # local continuity fill across tiny gaps only
    for index, row in enumerate(rows):
        if row["speaker"] or row["source"] == "ambiguous_overlap":
            continue  # ambiguity is not a gap to paper over
        left = next((j for j in range(index - 1, -1, -1) if rows[j]["speaker"]), None)
        right = next((j for j in range(index + 1, len(rows)) if rows[j]["speaker"]), None)
        start, end = float(words[index]["start"]), float(words[index]["end"])
        left_gap = start - float(words[left]["end"]) if left is not None else float("inf")
        right_gap = float(words[right]["start"]) - end if right is not None else float("inf")
        left_speaker = rows[left]["speaker"] if left is not None else ""
        right_speaker = rows[right]["speaker"] if right is not None else ""
        chosen = ""
        if left_speaker and left_speaker == right_speaker and max(left_gap, right_gap) <= MAX_FILL_GAP:
            chosen = left_speaker
        elif left_speaker != right_speaker:
            if left_speaker and left_gap <= MAX_FILL_GAP and left_gap <= right_gap:
                chosen = left_speaker
            elif right_speaker and right_gap <= MAX_FILL_GAP:
                chosen = right_speaker
        if chosen:
            rows[index] = {"speaker": chosen, "confidence": 0.56, "source": "continuity_fill"}

    absorbed = 0
    for _ in range(3):  # absorb one-word flips sandwiched by the same speaker
        changed = False
        runs = _runs(rows)
        for k, (a, b, speaker) in enumerate(runs):
            if not speaker or b - a != 1:
                continue
            if float(words[a]["end"]) - float(words[a]["start"]) > MAX_SINGLE_WORD_FLIP:
                continue
            if rows[a]["confidence"] >= 0.90:
                continue  # the word was found in that speaker's own diarized text: a real interjection
            left = runs[k - 1][2] if k > 0 else ""
            right = runs[k + 1][2] if k + 1 < len(runs) else ""
            if left and left == right and left != speaker:
                rows[a] = {"speaker": left, "confidence": min(0.66, rows[a]["confidence"]), "source": "turn_hysteresis"}
                absorbed += 1
                changed = True
        if not changed:
            break
    runs = _runs(rows)
    for k, (a, b, speaker) in enumerate(runs):
        if not speaker or b - a >= MIN_STABLE_TURN_WORDS:
            continue
        seconds = float(words[b - 1]["end"]) - float(words[a]["start"])
        confidence = sum(r["confidence"] for r in rows[a:b]) / (b - a)
        if (seconds >= MIN_STABLE_TURN_SECONDS and confidence >= 0.84) or confidence >= 0.90:
            continue
        left = runs[k - 1][2] if k > 0 else ""
        right = runs[k + 1][2] if k + 1 < len(runs) else ""
        if left and left == right and left != speaker:
            for i in range(a, b):
                rows[i] = {"speaker": left, "confidence": min(0.68, rows[i]["confidence"]), "source": "weak_turn_absorbed"}
            absorbed += b - a
    runs = _runs(rows)
    return rows, {"text_timing_agreements": agree, "text_timing_disagreements": disagree,
                  "absorbed_words": absorbed, "turns": len(runs),
                  "unresolved_words": sum(1 for r in rows if not r["speaker"]),
                  "ambiguous_words": sum(1 for r in rows if r["source"] == "ambiguous_overlap")}


MIN_MEASURED_OVERLAP = 0.35  # diarization boundary jitter stays below this


def measured_overlaps(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Simultaneous speech measured by diarization: two participants' turns overlapping for at
    least MIN_MEASURED_OVERLAP. The speaker who started later is the interrupter; their whole
    overlapping turn is recorded so captions never switch lanes mid-sentence."""
    ordered = sorted(segments, key=lambda s: (s["start"], s["end"]))
    out: list[dict[str, Any]] = []
    for i, first in enumerate(ordered):
        for second in ordered[i + 1:]:
            if second["start"] >= first["end"]:
                break
            if second["speaker"] == first["speaker"]:
                continue
            both = min(first["end"], second["end"]) - second["start"]
            if both >= MIN_MEASURED_OVERLAP:
                out.append({"start": round(second["start"], 3), "end": round(min(first["end"], second["end"]), 3),
                            "held_by": first["speaker"], "interrupter": second["speaker"],
                            "turn": [round(second["start"], 3), round(second["end"], 3)]})
    return out
