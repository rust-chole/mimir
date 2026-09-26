"""Backward-only correction of proven LATE phrase-start onsets (exact PCM evidence).

No global shift, no speaker timing, no semantic timing. A phrase that starts
after a real pause is moved EARLIER only when its current window is
acoustically weak and a much stronger speech onset exists just before it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from mimir.media.audio import speech_frames

FRAME_SECONDS = 0.020
MIN_PRE_GAP = 0.50
MAX_BACKSHIFT = 1.20
MIN_SHIFT = 0.18
ORIGINAL_MAX_SCORE = 0.18
MIN_IMPROVEMENT = 0.30
MIN_TARGET_SCORE = 0.34
MAX_GROUP_SECONDS = 2.20
PHRASE_GAP = 0.42


@dataclass
class _Frames:
    start: np.ndarray
    dbfs: np.ndarray
    zcr: np.ndarray
    likelihood: np.ndarray
    noise_db: float
    zcr_base: float


def _local_frames(features: dict[str, np.ndarray], window_start: float, window_end: float) -> _Frames | None:
    mask = (features["start"] >= window_start) & (features["start"] < window_end)
    if not mask.any():
        return None
    start, dbfs, zcr = features["start"][mask], features["dbfs"][mask], features["zcr"][mask]
    noise_db = float(np.percentile(dbfs, 25))
    zcr_base = float(np.median(zcr))
    energy = np.clip((dbfs - noise_db - 2.0) / 14.0, 0.0, 1.0)
    fricative = np.clip((zcr - max(0.070, zcr_base * 1.65)) / 0.090, 0.0, 1.0)
    fricative *= np.clip((dbfs - noise_db + 4.0) / 7.0, 0.15, 1.0)
    likelihood = np.maximum(energy, 0.95 * fricative)
    return _Frames(start, dbfs, zcr, likelihood, noise_db, zcr_base)


def _window_score(frames: _Frames, start: float, end: float) -> float:
    mask = (frames.start >= start) & (frames.start < end)
    values = np.sort(frames.likelihood[mask])[::-1]
    if len(values) == 0:
        return 0.0
    core = float(values[:max(1, int(len(values) * 0.70))].mean())
    onset_end = start + max(0.06, (end - start) * 0.35)
    onset_mask = (frames.start >= start) & (frames.start < onset_end)
    onset = float(frames.likelihood[onset_mask].mean()) if onset_mask.any() else 0.0
    return 0.70 * core + 0.30 * onset


def _previous_onset(frames: _Frames, search_start: float, candidate_start: float) -> float | None:
    mask = (frames.start >= search_start) & (frames.start < candidate_start + 0.04)
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return None
    active = frames.likelihood[idx] >= 0.15
    bridge = max(1, int(round(0.12 / FRAME_SECONDS)))
    on = np.nonzero(active)[0]
    for left, right in zip(on, on[1:]):
        if 1 < right - left <= bridge + 1:
            active[left + 1:right] = True
    regions: list[tuple[int, int]] = []
    region_start = None
    for i, flag in enumerate(list(active) + [False]):
        if flag and region_start is None:
            region_start = i
        elif not flag and region_start is not None:
            if (i - region_start) * FRAME_SECONDS >= 0.12 and frames.likelihood[idx[region_start:i]].max() >= 0.42:
                regions.append((region_start, i))
            region_start = None
    if not regions:
        return None
    a, b = regions[-1]
    onset = float(frames.start[idx[a]])
    for j in idx[a:b]:
        voiced = frames.dbfs[j] >= frames.noise_db + 6.0
        fricative = frames.zcr[j] >= max(0.090, frames.zcr_base * 2.0) and frames.dbfs[j] >= frames.noise_db - 2.0
        if voiced or fricative:
            onset = float(frames.start[j])
            break
    return onset


def apply_clock_guard(words: Sequence[dict[str, Any]], samples: np.ndarray, rate: int, *, offset: float,
                      duration: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``words`` carry absolute ``start``/``end``; ``samples`` start at ``offset`` (source seconds)."""
    output = [dict(word) for word in words]
    if not output:
        return output, {"status": "empty", "corrected_groups": 0, "details": []}
    features = speech_frames(samples, rate, FRAME_SECONDS)
    features = {**features, "start": features["start"] + offset}
    groups: list[list[int]] = []
    current: list[int] = []
    for index, word in enumerate(output):
        if current and float(word["start"]) - float(output[current[-1]]["end"]) >= PHRASE_GAP:
            groups.append(current)
            current = []
        current.append(index)
    if current:
        groups.append(current)
    details: list[dict[str, Any]] = []
    end_limit = offset + duration
    for gi, indices in enumerate(groups):
        start = float(output[indices[0]]["start"])
        end = float(output[indices[-1]]["end"])
        previous_end = float(output[groups[gi - 1][-1]]["end"]) if gi > 0 else offset
        if gi == 0 or start - previous_end < MIN_PRE_GAP or not (0.0 < end - start <= MAX_GROUP_SECONDS):
            continue
        search_start = max(previous_end + 0.05, start - MAX_BACKSHIFT)
        frames = _local_frames(features, max(offset, search_start - 0.08), min(end_limit, end + 0.10))
        if frames is None:
            continue
        original = _window_score(frames, start, end)
        if original > ORIGINAL_MAX_SCORE:
            continue
        onset = _previous_onset(frames, search_start, start)
        if onset is None:
            continue
        shift = onset - start
        if shift >= -MIN_SHIFT or abs(shift) > MAX_BACKSHIFT + 0.02:
            continue
        shifted_end = end + shift
        if shifted_end <= previous_end + 0.03:
            continue
        target = _window_score(frames, onset, shifted_end)
        if target < MIN_TARGET_SCORE or target - original < MIN_IMPROVEMENT:
            continue
        next_start = float(output[groups[gi + 1][0]]["start"]) if gi + 1 < len(groups) else end_limit
        if shifted_end >= next_start - 0.02:
            continue
        for index in indices:
            word = output[index]
            old_start, old_end = float(word["start"]), float(word["end"])
            word["start"] = round(old_start + shift, 3)
            word["end"] = round(max(old_start + shift + 0.025, old_end + shift), 3)
            word["timing_source"] = f"{word.get('timing_source', 'clock')}+pcm_guard"
            word["timing_guard_delta"] = round(shift, 3)
        details.append({"words": [output[i]["text"] for i in indices], "original_start": round(start, 3),
                        "corrected_start": round(onset, 3), "delta": round(shift, 3),
                        "original_score": round(original, 4), "target_score": round(target, 4)})
    return output, {"status": "corrected" if details else "clean", "corrected_groups": len(details),
                    "details": details, "policy": "backward-only exact-PCM phrase-start correction"}
