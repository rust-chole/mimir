"""Measured caption clock: structural health check + bounded, measured recovery.

The whisper-1 native word clock on the EXACT final 48 kHz audio stays the
normal timing authority. This module only asks whether that clock is
structurally usable, and repairs it with MEASURED evidence when it is not:

    health    multiple symptoms, never one brittle number:
              FATAL  no lexical words | transcript-sized collapse into one instant |
                     most words outside the audio | most durations zero/negative
              SOFT   some words outside the audio | many zero/negative durations |
                     onset regressions beyond jitter | clock covers too few of the words
              healthy = no fatal symptom and at most one soft symptom

    recovery  1. re-time the exact final audio in overlapping chunks (~10 s, small
                 overlap), merge the chunk clocks by overlap ownership, dedupe
                 conservatively, and SPLICE only the regions where the primary
                 clock is locally damaged (healthy primary anchors never move)
              2. an already-existing independent measured clock (the whole-VOD
                 whisper words mapped through the timeline cut map), used only if
                 it is structurally healthier, spliced the same way
              every candidate is re-checked; diarization is never a word clock

    decision  a healthy clock is used; otherwise the structurally best measured
              clock is used with a disclosed degradation when it has no FATAL
              symptom; if every measured clock is still fatally broken the
              caption stage BLOCKS (ClockUnavailable). No synthetic, global or
              interpolated timing exists here.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

CLOCK_VERSION = 1
# Generic media/algorithm parameters (not tuned to any video).
CHUNK_S = float(os.getenv("MIMIR_CLOCK_CHUNK_SECONDS", "10") or 10)
OVERLAP_S = float(os.getenv("MIMIR_CLOCK_CHUNK_OVERLAP", "1.5") or 1.5)
JITTER_S = 0.25            # an onset step back smaller than this is jitter / simultaneous speech
SAME_INSTANT_S = 0.02
COLLAPSE_MIN_WORDS = 8     # a pile-up this large at one instant is a collapsed clock ...
COLLAPSE_SHARE = 0.25      # ... when it is also this share of all words
RANGE_TOL_S = 0.25
DEDUPE_S = 0.15            # same word from both sides of a chunk seam within this -> one anchor
SEAM_DISAGREE_S = 0.30     # reported only: overlap anchors disagreeing more than this
LOCAL_MISSING_SHARE = 0.6  # a primary region with < this share of the re-timed words lost words
WORKERS = 3


class ClockUnavailable(RuntimeError):
    """No measured word clock survived recovery (the caption stage must block)."""


def _canon(word: str) -> str:
    from ai import vod_processor

    return vod_processor.canonical_word(word)


def clock_rows(timing_data: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Lexical measured words: word/start/end (non-lexical tokens dropped, order kept)."""
    rows: list[dict[str, Any]] = []
    for item in (timing_data or {}).get("words", []) or []:
        if not isinstance(item, Mapping):
            continue
        word = str(item.get("word", "")).strip()
        if not word or not _canon(word):
            continue
        try:
            start = float(item.get("start", 0.0))
            end = float(item.get("end", start))
        except (TypeError, ValueError):
            continue
        rows.append({"word": word, "start": start, "end": end})
    return rows


# ============================================================
# HEALTH
# ============================================================

@dataclass
class ClockHealth:
    fatal: list[str] = field(default_factory=list)
    soft: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        return not self.fatal and len(self.soft) <= 1

    @property
    def rank(self) -> tuple[int, int]:
        return len(self.fatal), len(self.soft)

    def to_dict(self) -> dict[str, Any]:
        return {"healthy": self.healthy, "fatal": list(self.fatal), "soft": list(self.soft),
                "metrics": dict(self.metrics)}


def _largest_pileup(rows: Sequence[Mapping[str, Any]]) -> int:
    best = run = 0
    previous: float | None = None
    for row in sorted(rows, key=lambda r: float(r["start"])):
        start = float(row["start"])
        run = run + 1 if previous is not None and start - previous <= SAME_INSTANT_S else 1
        best = max(best, run)
        previous = start
    return best


def assess(rows: Sequence[Mapping[str, Any]], *, duration: float, expected_words: int = 0) -> ClockHealth:
    health = ClockHealth()
    n = len(rows)
    health.metrics["words"] = n
    health.metrics["expected_words"] = int(expected_words)
    if n == 0:
        health.fatal.append("no_words")
        return health
    starts = [float(r["start"]) for r in rows]
    outside = sum(1 for r in rows if float(r["start"]) < -RANGE_TOL_S
                  or float(r["end"]) > duration + RANGE_TOL_S) if duration > 0 else 0
    nonpositive = sum(1 for r in rows if float(r["end"]) <= float(r["start"]))
    regressions = sum(1 for a, b in zip(starts, starts[1:]) if b < a - JITTER_S)
    pileup = _largest_pileup(rows)
    coverage = n / float(expected_words) if expected_words > 0 else 1.0
    health.metrics.update({"outside_audio": outside, "nonpositive_durations": nonpositive,
                           "onset_regressions": regressions, "largest_same_instant": pileup,
                           "coverage": round(coverage, 3)})
    if pileup >= max(COLLAPSE_MIN_WORDS, COLLAPSE_SHARE * n):
        health.fatal.append("collapsed_to_one_instant")
    if outside > 0.5 * n:
        health.fatal.append("mostly_outside_audio")
    elif outside:
        health.soft.append("words_outside_audio")
    if nonpositive > 0.5 * n:
        health.fatal.append("mostly_nonpositive_durations")
    elif nonpositive > 0.2 * n:
        health.soft.append("many_nonpositive_durations")
    if regressions > max(1, 0.05 * n):
        health.soft.append("onset_regressions")
    if expected_words > 0 and coverage < 0.5:
        health.soft.append("low_coverage")
    return health


def _region_damaged(primary: Sequence[Mapping[str, Any]], recovered: Sequence[Mapping[str, Any]], *,
                    start: float, end: float) -> str:
    """Why the primary clock is locally damaged in [start, end) ("" = locally fine: keep it)."""
    if not recovered:
        return ""                      # no measured replacement here: the primary stays
    if len(primary) < LOCAL_MISSING_SHARE * len(recovered) and len(recovered) >= 3:
        return "missing_words"
    if len(primary) >= 4 and _largest_pileup(primary) >= max(4, 0.5 * len(primary)):
        return "collapsed"
    if primary and sum(1 for r in primary if float(r["end"]) <= float(r["start"])) > 0.5 * len(primary):
        return "nonpositive_durations"
    starts = [float(r["start"]) for r in primary]
    if sum(1 for a, b in zip(starts, starts[1:]) if b < a - JITTER_S) > max(1, 0.2 * len(primary)):
        return "onset_regressions"
    return ""


# ============================================================
# RECOVERY 1: OVERLAPPING CHUNKS OF THE EXACT FINAL AUDIO
# ============================================================

def chunk_windows(duration: float, chunk: float = CHUNK_S, overlap: float = OVERLAP_S) -> list[tuple[float, float]]:
    if duration <= 0:
        return []
    step = max(1.0, chunk - overlap)
    windows: list[tuple[float, float]] = []
    start = 0.0
    while True:
        end = min(duration, start + chunk)
        windows.append((round(start, 3), round(end, 3)))
        if end >= duration - 1e-6:
            return windows
        start += step


def ownership(windows: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Each chunk owns up to the middle of its overlaps: no seam word is counted twice and
    every owned word sits at least half an overlap away from its chunk's cut edge."""
    owned: list[tuple[float, float]] = []
    for index, (start, end) in enumerate(windows):
        left = start if index == 0 else (start + windows[index - 1][1]) / 2.0
        right = end if index == len(windows) - 1 else (windows[index + 1][0] + end) / 2.0
        owned.append((left, right))
    return owned


def merge_chunks(chunks: Sequence[tuple[tuple[float, float], Sequence[Mapping[str, Any]] | None]]
                 ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Chunk clocks (absolute times) -> one clock by overlap ownership + conservative dedupe."""
    windows = [window for window, _rows in chunks]
    owned = ownership(windows)
    merged: list[dict[str, Any]] = []
    seams: list[dict[str, Any]] = []
    for index, ((window, rows), (left, right)) in enumerate(zip(chunks, owned)):
        for row in rows or []:
            start = float(row["start"])
            last = index == len(chunks) - 1
            if left - 1e-6 <= start and (start < right or (last and start <= right + RANGE_TOL_S)):
                merged.append({**row, "chunk": index})
        if index + 1 < len(chunks) and rows and chunks[index + 1][1]:
            a0, a1 = chunks[index + 1][0][0], window[1]           # the measured overlap
            here = [r for r in rows if a0 <= float(r["start"]) < a1]
            there = [r for r in chunks[index + 1][1] or [] if a0 <= float(r["start"]) < a1]
            deltas = [abs(float(x["start"]) - float(y["start"])) for x in here for y in there
                      if _canon(x["word"]) == _canon(y["word"]) and abs(float(x["start"]) - float(y["start"])) < 1.0]
            if deltas:
                deltas.sort()
                seams.append({"seam": index, "matched_anchors": len(deltas),
                              "median_delta": round(deltas[len(deltas) // 2], 3), "max_delta": round(deltas[-1], 3),
                              "disagrees": deltas[len(deltas) // 2] > SEAM_DISAGREE_S})
    merged.sort(key=lambda r: (float(r["start"]), float(r["end"])))
    deduped: list[dict[str, Any]] = []
    dropped = 0
    for row in merged:
        previous = deduped[-1] if deduped else None
        if (previous is not None and previous["chunk"] != row["chunk"] and _canon(previous["word"]) == _canon(row["word"])
                and abs(float(row["start"]) - float(previous["start"])) < DEDUPE_S):
            dropped += 1
            continue
        deduped.append(row)
    return deduped, {"chunks": len(chunks), "seams": seams, "deduped": dropped}


def splice(primary: Sequence[Mapping[str, Any]], recovered: Sequence[Mapping[str, Any]],
           regions: Sequence[tuple[float, float]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Primary words where the primary is locally fine, recovered words only where it is damaged.

    Kept primary words keep their ORIGINAL order (nothing is re-sorted to look healthier);
    a region with no re-measured words always keeps the primary."""

    def region_of(row: Mapping[str, Any]) -> int | None:
        t = float(row["start"])
        for index, (start, end) in enumerate(regions):
            last = index == len(regions) - 1
            if start - 1e-6 <= t and (t < end or (last and t <= end + RANGE_TOL_S)):
                return index
        return None

    by_region_p: list[list[dict[str, Any]]] = [[] for _ in regions]
    by_region_r: list[list[dict[str, Any]]] = [[] for _ in regions]
    for row in primary:
        index = region_of(row)
        if index is not None:          # a primary word beyond the audio has no region
            by_region_p[index].append(dict(row))
    for row in recovered:
        index = region_of(row)
        if index is not None:
            by_region_r[index].append(dict(row))
    out: list[dict[str, Any]] = []
    replaced: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(regions):
        p, r = by_region_p[index], sorted(by_region_r[index], key=lambda x: (float(x["start"]), float(x["end"])))
        why = _region_damaged(p, r, start=start, end=end)
        if why:
            out.extend({**row, "timing_source": row.get("timing_source", "clock_recovery")} for row in r)
            replaced.append({"region": [round(start, 3), round(end, 3)], "reason": why,
                             "primary_words": len(p), "recovered_words": len(r)})
        else:
            out.extend(p)
    return out, replaced


def chunked_recovery(audio_path: Path, duration: float, *, timer: Callable[[Path], Mapping[str, Any]],
                     cutter: Callable[..., Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    windows = chunk_windows(duration)

    def run(item: tuple[int, tuple[float, float]]) -> tuple[tuple[float, float], list[dict[str, Any]] | None, str]:
        index, (start, end) = item
        path: Path | None = None
        try:
            path = cutter(audio_path, start=start, end=end, label=f"clock_chunk_{index:03d}", enhanced=False)
            rows = [{**row, "start": round(float(row["start"]) + start, 3), "end": round(float(row["end"]) + start, 3)}
                    for row in clock_rows(timer(path))]
            return (start, end), rows, ""
        except Exception as error:  # a failed chunk keeps the primary in its region
            return (start, end), None, f"chunk {index}: {type(error).__name__}: {str(error)[:160]}"
        finally:
            if path is not None:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass

    with ThreadPoolExecutor(max_workers=min(WORKERS, max(1, len(windows))), thread_name_prefix="mimir-clock") as pool:
        results = list(pool.map(run, enumerate(windows)))
    merged, report = merge_chunks([(window, rows) for window, rows, _error in results])
    report["errors"] = [error for _w, _r, error in results if error]
    report["regions"] = [list(r) for r in ownership(windows)]
    return merged, report


# ============================================================
# DECISION
# ============================================================

def recover_clock(primary_data: Mapping[str, Any] | None, *, audio_path: Path, duration: float, expected_words: int,
                  timer: Callable[[Path], Mapping[str, Any]] | None = None, cutter: Callable[..., Path] | None = None,
                  reference: Callable[[], Sequence[Mapping[str, Any]]] | None = None
                  ) -> tuple[dict[str, Any], dict[str, Any]]:
    """(timing_data for the fixed-clock aligner, clock report). Raises ClockUnavailable."""
    from ai import vod_processor

    timer = timer or (lambda path: vod_processor.transcribe_word_timing(path))
    cutter = cutter or vod_processor.extract_caption_micro_audio
    primary = clock_rows(primary_data)
    primary_health = assess(primary, duration=duration, expected_words=expected_words)
    report: dict[str, Any] = {"version": CLOCK_VERSION, "source": "whisper_primary",
                              "primary_health": primary_health.to_dict(), "recovery_attempted": False,
                              "attempts": [], "selected": "whisper_primary", "status": "healthy",
                              "reason": "primary measured clock is structurally healthy"}
    if primary_health.healthy:
        return {**dict(primary_data or {}), "words": primary}, report

    report["recovery_attempted"] = True
    candidates: list[tuple[ClockHealth, str, list[dict[str, Any]]]] = [(primary_health, "whisper_primary", primary)]

    windows = chunk_windows(duration)
    regions = ownership(windows)
    try:
        chunked, chunk_report = chunked_recovery(audio_path, duration, timer=timer, cutter=cutter)
        chunked = [{**row, "timing_source": "whisper_chunk_recovery"} for row in chunked]
        spliced, replaced = splice(primary, chunked, regions)
        health = assess(spliced, duration=duration, expected_words=expected_words)
        report["attempts"].append({"method": "overlapping_chunks_exact_final_audio", "health": health.to_dict(),
                                   "replaced_regions": replaced, **chunk_report, "selected": False,
                                   **({} if replaced else {"note": "no damaged region could be re-measured"})})
        if replaced:
            candidates.append((health, "whisper_chunk_recovery", spliced))
            if health.healthy:
                return _select(report, candidates[-1], primary_data)
    except Exception as error:
        report["attempts"].append({"method": "overlapping_chunks_exact_final_audio", "selected": False,
                                   "error": f"{type(error).__name__}: {str(error)[:200]}"})

    if reference is not None:
        try:
            ref_rows = [{**dict(r), "timing_source": "vod_reference_clock"} for r in reference() or []]
            spliced, replaced = splice(primary, ref_rows, regions)
            health = assess(spliced, duration=duration, expected_words=expected_words)
            healthier = bool(replaced) and health.rank < primary_health.rank
            report["attempts"].append({"method": "independent_whole_vod_clock", "health": health.to_dict(),
                                       "replaced_regions": replaced, "healthier_than_primary": healthier,
                                       "selected": False})
            if healthier:
                candidates.append((health, "vod_reference_clock", spliced))
                if health.healthy:
                    return _select(report, candidates[-1], primary_data)
        except Exception as error:
            report["attempts"].append({"method": "independent_whole_vod_clock", "selected": False,
                                       "error": f"{type(error).__name__}: {str(error)[:200]}"})

    best = min(candidates, key=lambda row: row[0].rank)
    if best[0].fatal:
        report.update(status="unavailable", selected=None,
                      reason="no measured clock survived recovery: " + ", ".join(best[0].fatal))
        raise ClockUnavailable(report["reason"])
    return _select(report, best, primary_data, degraded=True)


def _select(report: dict[str, Any], candidate: tuple[ClockHealth, str, list[dict[str, Any]]],
            primary_data: Mapping[str, Any] | None, *, degraded: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    health, source, rows = candidate
    for attempt in report["attempts"]:
        attempt["selected"] = (attempt.get("method") == "overlapping_chunks_exact_final_audio"
                               and source == "whisper_chunk_recovery") or (
            attempt.get("method") == "independent_whole_vod_clock" and source == "vod_reference_clock")
    report.update(source=source, selected=source, selected_health=health.to_dict(),
                  status="degraded" if degraded else ("recovered" if source != "whisper_primary" else "healthy"),
                  reason=("best measured clock has soft symptoms: " + ", ".join(health.soft)) if degraded
                  else f"{source} is structurally healthy")
    clean = [{k: v for k, v in row.items() if k != "chunk"} for row in rows]
    return {**dict(primary_data or {}), "words": clean, "clock_source": source}, report
