"""Audio speaker <-> face track association (evidence only, never assumed).

A face track is linked to a diarized speaker only when its mouth-region
activity rises while that speaker talks and not while others talk: a
significant positive point-biserial correlation (Fisher z, sample-size aware)
with a clear margin over every rival pairing, one-to-one. Otherwise the track
stays unlinked and the camera treats it conservatively.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from mimir.vision.tracking import Sample

MIN_ACTIVITY_SAMPLES = 20
MIN_SPEAKING_SAMPLES = 6
MIN_CORRELATION = 0.15
MIN_Z = 3.0
MIN_MARGIN = 0.10
SPEECH_PAD = 0.10


@dataclass(frozen=True)
class LinkStat:
    r: float
    n: int
    z: float


def _activity_rows(samples: Sequence[Sample]) -> list[tuple[float, float]]:
    rows = [(s.t, s.activity) for s in samples if s.activity is not None]
    if not rows:
        return []
    stretches: list[list[tuple[float, float]]] = [[rows[0]]]
    for row in rows[1:]:
        if row[0] - stretches[-1][-1][0] > 0.5:
            stretches.append([row])
        else:
            stretches[-1].append(row)
    smoothed: list[tuple[float, float]] = []
    for stretch in stretches:  # smooth within continuous stretches only
        values = np.array([a for _, a in stretch], dtype=np.float64)
        kernel = np.ones(5) / 5.0
        padded = np.pad(values, 2, mode="edge")
        smooth = np.convolve(padded, kernel, mode="valid")
        smoothed.extend((t, float(v)) for (t, _), v in zip(stretch, smooth))
    return smoothed


def speaking_intervals(segments: Sequence[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    result: dict[str, list[tuple[float, float]]] = {}
    for segment in segments:
        result.setdefault(segment["speaker"], []).append((segment["start"] - SPEECH_PAD, segment["end"] + SPEECH_PAD))
    return result


def link_stats(tracks: dict[str, Sequence[Sample]], intervals: dict[str, list[tuple[float, float]]]
               ) -> dict[tuple[str, str], LinkStat]:
    stats: dict[tuple[str, str], LinkStat] = {}
    for track_id, samples in tracks.items():
        rows = _activity_rows(samples)
        if len(rows) < MIN_ACTIVITY_SAMPLES:
            continue
        times = np.array([t for t, _ in rows])
        activity = np.array([a for _, a in rows])
        for speaker, spans in intervals.items():
            talk = np.zeros(len(times))
            for a, b in spans:
                talk[(times >= a) & (times <= b)] = 1.0
            count = talk.sum()
            if count < MIN_SPEAKING_SAMPLES or count > len(talk) - MIN_SPEAKING_SAMPLES:
                continue
            if activity.std() < 1e-12:
                continue
            r = float(np.corrcoef(activity, talk)[0, 1])
            if math.isnan(r):
                continue
            clipped = max(-0.999, min(0.999, r))
            stats[(track_id, speaker)] = LinkStat(r, len(rows), math.atanh(clipped) * math.sqrt(max(1, len(rows) - 3)))
    return stats


def associate(tracks: dict[str, Sequence[Sample]], segments: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """track_id -> {"speaker", "confidence", "evidence"} for clearly supported links only."""
    stats = link_stats(tracks, speaking_intervals(segments))
    assigned: dict[str, dict[str, Any]] = {}
    used: set[str] = set()
    for (track_id, speaker), stat in sorted(stats.items(), key=lambda item: (-item[1].z, item[0])):
        if track_id in assigned or speaker in used or stat.r < MIN_CORRELATION or stat.z < MIN_Z:
            continue
        rivals = [s.r for (tid, spk), s in stats.items()
                  if (tid == track_id and spk != speaker) or (spk == speaker and tid != track_id)]
        margin = stat.r - max(rivals, default=-1.0)
        if margin < MIN_MARGIN:
            continue
        strength = min(1.0, (stat.z - MIN_Z) / 4.0) * 0.6 + min(1.0, margin / 0.4) * 0.4
        assigned[track_id] = {"speaker": speaker, "confidence": round(0.5 + 0.45 * max(0.0, strength), 3),
                              "evidence": f"mouth_activity r={stat.r:.2f} z={stat.z:.1f} margin={margin:.2f} n={stat.n}"}
        used.add(speaker)
    return assigned
