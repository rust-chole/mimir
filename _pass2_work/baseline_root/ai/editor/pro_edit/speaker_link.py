"""Speaker (audio diarization) <-> visual track association.

Who is VISIBLE (face tracks) is not who is SPEAKING (MIMIR diarization).
This layer links them only when evidence supports it: per-sample mouth-region
motion energy of a face track must correlate with that speaker's talking
timeline, clearly better than with any other speaker. Otherwise the track
stays unassociated (speaker_id None) - no pretended certainty.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterable, Sequence

from ai.editor.pro_edit.caption_guard import CaptionWordRef
from ai.editor.pro_edit.subjects import SubjectTrack

MIN_ACTIVITY_SAMPLES = 12
MIN_SPEAKING_SAMPLES = 4
MIN_CORRELATION = 0.30
MIN_MARGIN = 0.12
SPEECH_PAD_S = 0.10


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


def speaking_intervals(words: Iterable[CaptionWordRef]) -> dict[str, list[tuple[float, float]]]:
    result: dict[str, list[tuple[float, float]]] = {}
    for word in words:
        if word.speaker_id:
            result.setdefault(word.speaker_id, []).append((word.start - SPEECH_PAD_S, word.end + SPEECH_PAD_S))
    return result


def _speaking(intervals: Sequence[tuple[float, float]], t: float) -> float:
    return 1.0 if any(a <= t <= b for a, b in intervals) else 0.0


def association_scores(tracks: Sequence[SubjectTrack], words: Sequence[CaptionWordRef]) -> dict[tuple[str, str], float]:
    speakers = speaking_intervals(words)
    scores: dict[tuple[str, str], float] = {}
    for track in tracks:
        rows = [(s.t, s.activity) for s in track.samples if s.activity is not None]
        if len(rows) < MIN_ACTIVITY_SAMPLES:
            continue
        activity = [a for _, a in rows]
        for speaker, intervals in speakers.items():
            talk = [_speaking(intervals, t) for t, _ in rows]
            if sum(talk) < MIN_SPEAKING_SAMPLES or sum(talk) > len(talk) - MIN_SPEAKING_SAMPLES:
                continue  # constant indicator: correlation undefined -> no evidence
            corr = _pearson(activity, talk)
            if corr is not None:
                scores[(track.subject_id, speaker)] = corr
    return scores


def associate_speakers(tracks: Sequence[SubjectTrack], words: Sequence[CaptionWordRef]) -> tuple[SubjectTrack, ...]:
    """Fill ``speaker_id`` for face tracks with clear audiovisual evidence.

    Sidecar-provided associations are kept untouched.
    """
    scores = association_scores([t for t in tracks if t.speaker_id is None and t.kind == "face"], words)
    assigned_tracks: dict[str, tuple[str, float, float]] = {}
    used_speakers: set[str] = set()
    for (track_id, speaker), corr in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
        if track_id in assigned_tracks or speaker in used_speakers or corr < MIN_CORRELATION:
            continue
        others = [c for (tid, spk), c in scores.items() if tid == track_id and spk != speaker]
        rivals = [c for (tid, spk), c in scores.items() if spk == speaker and tid != track_id]
        margin = corr - max(others + rivals, default=-1.0)
        if margin < MIN_MARGIN:
            continue
        assigned_tracks[track_id] = (speaker, corr, margin)
        used_speakers.add(speaker)
    result = []
    for track in tracks:
        link = assigned_tracks.get(track.subject_id)
        if link is None:
            result.append(track)
            continue
        speaker, corr, margin = link
        confidence = 0.5 + 0.5 * max(0.0, min(1.0, (corr - MIN_CORRELATION) / (0.7 - MIN_CORRELATION)))
        result.append(replace(track, speaker_id=speaker, speaker_confidence=round(confidence, 3),
                              speaker_evidence=f"mouth_activity_corr={corr:.2f},margin={margin:.2f}"))
    return tuple(result)
