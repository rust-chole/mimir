"""Speaker (audio diarization) <-> visual track association.

Who is VISIBLE (face tracks) is not who is SPEAKING (MIMIR diarization).
This layer links them only when evidence supports it: the mouth-region motion
energy of a face track must rise while that speaker talks (and not while the
others talk), clearly better than for any other track/speaker pairing.
Otherwise the track stays unassociated (speaker_id None) - no pretended
certainty, and no face recognition is claimed.

Evidence per (track, speaker): point-biserial correlation between the
temporally smoothed activity signal (0.5 s window at the tracker's 10 fps)
and the speaking indicator, with its Fisher z statistic (sample-size aware:
a long stable track with a modest but consistent correlation is strong
evidence; a few noisy samples are not). A link needs a significant, positive
correlation and a margin over every rival pairing; assignment is one-to-one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

from ai.editor.pro_edit.caption_guard import CaptionWordRef
from ai.editor.pro_edit.subjects import SubjectTrack

SPEAKER_LINK_VERSION = 2
MIN_ACTIVITY_SAMPLES = 20
MIN_SPEAKING_SAMPLES = 6
MIN_CORRELATION = 0.15
MIN_Z = 3.0
MIN_MARGIN = 0.10
SMOOTH_HALF_WINDOW = 2
SPEECH_PAD_S = 0.10


@dataclass(frozen=True)
class LinkStat:
    r: float
    n: int
    z: float


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _smooth(values: Sequence[float], half: int = SMOOTH_HALF_WINDOW) -> list[float]:
    out: list[float] = []
    for index in range(len(values)):
        lo, hi = max(0, index - half), min(len(values), index + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def speaking_intervals(words: Iterable[CaptionWordRef]) -> dict[str, list[tuple[float, float]]]:
    result: dict[str, list[tuple[float, float]]] = {}
    for word in words:
        if word.speaker_id:
            result.setdefault(word.speaker_id, []).append((word.start - SPEECH_PAD_S, word.end + SPEECH_PAD_S))
    return result


def _speaking(intervals: Sequence[tuple[float, float]], t: float) -> float:
    return 1.0 if any(a <= t <= b for a, b in intervals) else 0.0


def _activity_rows(track: SubjectTrack) -> list[tuple[float, float]]:
    rows = [(s.t, s.activity) for s in track.samples if s.activity is not None]
    if not rows:
        return []
    # Smooth within continuous stretches only (a stitched gap is not motion).
    stretches: list[list[tuple[float, float]]] = [[rows[0]]]
    for row in rows[1:]:
        if row[0] - stretches[-1][-1][0] > 0.5:
            stretches.append([row])
        else:
            stretches[-1].append(row)
    smoothed: list[tuple[float, float]] = []
    for stretch in stretches:
        values = _smooth([float(a) for _, a in stretch])
        smoothed.extend((t, v) for (t, _), v in zip(stretch, values))
    return smoothed


def association_stats(tracks: Sequence[SubjectTrack], words: Sequence[CaptionWordRef]
                      ) -> dict[tuple[str, str], LinkStat]:
    speakers = speaking_intervals(words)
    stats: dict[tuple[str, str], LinkStat] = {}
    for track in tracks:
        rows = _activity_rows(track)
        if len(rows) < MIN_ACTIVITY_SAMPLES:
            continue
        activity = [a for _, a in rows]
        for speaker, intervals in speakers.items():
            talk = [_speaking(intervals, t) for t, _ in rows]
            if sum(talk) < MIN_SPEAKING_SAMPLES or sum(talk) > len(talk) - MIN_SPEAKING_SAMPLES:
                continue  # constant indicator: correlation undefined -> no evidence
            corr = _pearson(activity, talk)
            if corr is None:
                continue
            clipped = max(-0.999, min(0.999, corr))
            stats[(track.subject_id, speaker)] = LinkStat(corr, len(rows),
                                                          math.atanh(clipped) * math.sqrt(max(1, len(rows) - 3)))
    return stats


def association_scores(tracks: Sequence[SubjectTrack], words: Sequence[CaptionWordRef]) -> dict[tuple[str, str], float]:
    """(track, speaker) -> correlation (kept for diagnostics / callers of V1)."""
    return {key: stat.r for key, stat in association_stats(tracks, words).items()}


def associate_speakers(tracks: Sequence[SubjectTrack], words: Sequence[CaptionWordRef]) -> tuple[SubjectTrack, ...]:
    """Fill ``speaker_id`` for face tracks with clear audiovisual evidence.

    Sidecar-provided associations are kept untouched.
    """
    stats = association_stats([t for t in tracks if t.speaker_id is None and t.kind == "face"], words)
    assigned_tracks: dict[str, tuple[str, LinkStat, float]] = {}
    used_speakers: set[str] = set()
    for (track_id, speaker), stat in sorted(stats.items(), key=lambda item: (-item[1].z, item[0])):
        if track_id in assigned_tracks or speaker in used_speakers:
            continue
        if stat.r < MIN_CORRELATION or stat.z < MIN_Z:
            continue
        others = [s.r for (tid, spk), s in stats.items() if tid == track_id and spk != speaker]
        rivals = [s.r for (tid, spk), s in stats.items() if spk == speaker and tid != track_id]
        margin = stat.r - max(others + rivals, default=-1.0)
        if margin < MIN_MARGIN:
            continue
        assigned_tracks[track_id] = (speaker, stat, margin)
        used_speakers.add(speaker)
    result = []
    for track in tracks:
        link = assigned_tracks.get(track.subject_id)
        if link is None:
            result.append(track)
            continue
        speaker, stat, margin = link
        strength = min(1.0, (stat.z - MIN_Z) / 4.0) * 0.6 + min(1.0, margin / 0.4) * 0.4
        confidence = round(0.5 + 0.45 * max(0.0, strength), 3)
        result.append(replace(track, speaker_id=speaker, speaker_confidence=confidence,
                              speaker_evidence=f"mouth_activity_corr={stat.r:.2f},z={stat.z:.1f},"
                                               f"margin={margin:.2f},n={stat.n}"))
    return tuple(result)
