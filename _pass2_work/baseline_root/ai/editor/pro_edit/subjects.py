"""Tracker-independent subject evidence + deterministic camera-target logic.

V1 ships no ML tracker. Tracks come from a provider; the default provider
returns no tracks, so every target resolves to CENTER_SAFE. A future face or
object tracker only has to write the sidecar format read by
``JsonSidecarSubjectProvider``.

Sidecar format (PACED_CLIP seconds, normalized coordinates)::

    {"schema_version": 1, "timeline_domain": "paced_clip",
     "tracks": [{"subject_id": "face_0", "kind": "face", "speaker_id": "A",
                 "samples": [{"t": 0.0, "cx": 0.5, "cy": 0.4,
                              "width": 0.2, "height": 0.3, "confidence": 0.9}]}]}
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Protocol, Sequence

from ai.editor.pro_edit.errors import SubjectResolutionError
from ai.editor.pro_edit.schema import EditTarget, TargetType
from ai.editor.pro_edit.style import CameraMotionLimits
from ai.editor.pro_edit.timebase import TimelineDomain

SubjectKind = Literal["face", "person", "object", "unknown"]
_KINDS = {"face", "person", "object", "unknown"}
SUBJECT_SIDECAR_SCHEMA_VERSION = 1
CENTER = (0.5, 0.5)


@dataclass(frozen=True)
class SubjectSample:
    t: float
    cx: float
    cy: float
    width: float
    height: float
    confidence: float
    # Mouth-region motion energy (speaker association evidence); None if unknown.
    activity: float | None = None
    # "detected" | "tracked" (optical flow) | "interpolated" | "sidecar"
    source: str = "sidecar"


@dataclass(frozen=True)
class SubjectTrack:
    subject_id: str
    kind: SubjectKind
    samples: tuple[SubjectSample, ...]
    speaker_id: str | None = None
    # Speaker link is evidence-based; confidence 0 means "not associated".
    speaker_confidence: float = 0.0
    speaker_evidence: str = ""

    def window(self, start: float, end: float) -> tuple[SubjectSample, ...]:
        return tuple(s for s in self.samples if start <= s.t <= end)

    def summary(self) -> dict[str, object]:
        conf = [s.confidence for s in self.samples]
        return {
            "subject_id": self.subject_id,
            "kind": self.kind,
            "speaker_id": self.speaker_id,
            "speaker_confidence": round(self.speaker_confidence, 3),
            "speaker_evidence": self.speaker_evidence,
            "sample_count": len(self.samples),
            "mean_confidence": round(sum(conf) / len(conf), 3) if conf else 0.0,
            "t_range": [round(self.samples[0].t, 3), round(self.samples[-1].t, 3)] if self.samples else [],
        }


def _finite(value: object) -> float:
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise ValueError("non-finite")
    return number


def sanitize_sample(raw: dict[str, object]) -> SubjectSample | None:
    """Accept only finite, normalized, positive-size samples; never invent any."""
    try:
        t = _finite(raw.get("t"))
        cx = _finite(raw.get("cx"))
        cy = _finite(raw.get("cy"))
        width = _finite(raw.get("width"))
        height = _finite(raw.get("height"))
        confidence = _finite(raw.get("confidence"))
    except (TypeError, ValueError):
        return None
    if t < 0 or not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return None
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        return None
    activity_raw = raw.get("activity")
    try:
        activity = _finite(activity_raw) if activity_raw is not None else None
    except (TypeError, ValueError):
        activity = None
    source = str(raw.get("source", "sidecar"))
    return SubjectSample(t, cx, cy, width, height, max(0.0, min(1.0, confidence)), activity,
                         source if source in {"detected", "tracked", "interpolated", "sidecar"} else "sidecar")


class SubjectTrackProvider(Protocol):
    name: str

    def load(self, duration_s: float) -> tuple[SubjectTrack, ...]:
        ...


class NullSubjectProvider:
    """No tracker evidence: all targets fall back to CENTER_SAFE."""

    name = "none"

    def load(self, duration_s: float) -> tuple[SubjectTrack, ...]:
        del duration_s
        return ()


class JsonSidecarSubjectProvider:
    name = "json_sidecar"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self, duration_s: float) -> tuple[SubjectTrack, ...]:
        if not self.path.is_file():
            return ()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
        except (OSError, ValueError) as error:
            raise SubjectResolutionError(f"subject sidecar unreadable: {self.path}: {error}") from error
        if not isinstance(data, dict) or int(data.get("schema_version", -1)) != SUBJECT_SIDECAR_SCHEMA_VERSION:
            raise SubjectResolutionError("subject sidecar schema_version unsupported")
        if str(data.get("timeline_domain", "")) != TimelineDomain.PACED_CLIP.value:
            raise SubjectResolutionError("subject sidecar must be in paced_clip timeline domain")
        tracks: list[SubjectTrack] = []
        seen: set[str] = set()
        for raw in data.get("tracks", []) or []:
            if not isinstance(raw, dict):
                continue
            subject_id = str(raw.get("subject_id", "")).strip()
            if not subject_id or subject_id in seen:
                continue
            kind = str(raw.get("kind", "unknown"))
            samples = sorted(
                (s for s in (sanitize_sample(x) for x in raw.get("samples", []) or [] if isinstance(x, dict))
                 if s is not None and s.t <= duration_s + 0.5),
                key=lambda s: s.t,
            )
            if not samples:
                continue
            seen.add(subject_id)
            speaker = raw.get("speaker_id")
            try:
                speaker_conf = max(0.0, min(1.0, _finite(raw.get("speaker_confidence", 1.0 if speaker else 0.0))))
            except (TypeError, ValueError):
                speaker_conf = 0.0
            tracks.append(SubjectTrack(
                subject_id=subject_id,
                kind=kind if kind in _KINDS else "unknown",  # type: ignore[arg-type]
                samples=tuple(samples),
                speaker_id=str(speaker) if speaker not in (None, "") else None,
                speaker_confidence=speaker_conf if speaker else 0.0,
                speaker_evidence=str(raw.get("speaker_evidence", "sidecar" if speaker else "")),
            ))
        return tuple(tracks)

    def load_regions(self, duration_s: float) -> tuple[tuple[float, float, tuple[float, float, float, float], str], ...]:
        """Optional important regions: [{"start","end","box":[x0,y0,x1,y1],"label"}]."""
        if not self.path.is_file():
            return ()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
        except (OSError, ValueError):
            return ()
        rows = []
        for raw in (data.get("regions", []) if isinstance(data, dict) else []) or []:
            try:
                start, end = _finite(raw["start"]), _finite(raw["end"])
                box = tuple(_finite(v) for v in raw["box"])
            except (KeyError, TypeError, ValueError):
                continue
            if len(box) != 4 or end <= start or start > duration_s + 0.5:
                continue
            x0, y0, x1, y1 = box
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                continue
            rows.append((start, end, (x0, y0, x1, y1), str(raw.get("label", "region"))))
        return tuple(rows)


def write_sidecar(path: str | Path, tracks: Sequence[SubjectTrack], *, meta: dict[str, object] | None = None,
                  regions: Sequence[dict[str, object]] = ()) -> Path:
    """Serialize tracks in the sidecar format (shared by all trackers)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": SUBJECT_SIDECAR_SCHEMA_VERSION,
        "timeline_domain": TimelineDomain.PACED_CLIP.value,
        "meta": dict(meta or {}),
        "tracks": [{
            "subject_id": t.subject_id, "kind": t.kind, "speaker_id": t.speaker_id,
            "speaker_confidence": round(t.speaker_confidence, 4), "speaker_evidence": t.speaker_evidence,
            "samples": [{"t": round(x.t, 4), "cx": round(x.cx, 5), "cy": round(x.cy, 5),
                         "width": round(x.width, 5), "height": round(x.height, 5),
                         "confidence": round(x.confidence, 4),
                         "activity": None if x.activity is None else round(x.activity, 5), "source": x.source}
                        for x in t.samples],
        } for t in tracks],
        "regions": list(regions),
    }
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    import os
    os.replace(temp, target)
    return target


def _reject_constant(value: str) -> float:
    raise ValueError(f"non-standard JSON constant {value!r} rejected")


# ============================================================
# TARGET RESOLUTION (requested -> active speaker -> dominant -> center)
# ============================================================

@dataclass(frozen=True)
class ResolvedTarget:
    source: str  # "subject" | "active_speaker" | "dominant_subject" | "center_safe"
    subject_id: str | None
    reliable: bool
    mean_confidence: float
    coverage: float
    reason: str

    @property
    def is_center_safe(self) -> bool:
        return self.subject_id is None


CENTER_SAFE_TARGET = ResolvedTarget("center_safe", None, False, 0.0, 0.0, "no subject evidence")


def track_reliability(track: SubjectTrack, start: float, end: float, limits: CameraMotionLimits) -> tuple[float, float]:
    """(mean confidence, time coverage) of a track inside [start, end]."""
    samples = track.window(start, end)
    duration = max(1e-6, end - start)
    if not samples:
        return 0.0, 0.0
    mean_conf = sum(s.confidence for s in samples) / len(samples)
    good = [s for s in samples if s.confidence >= limits.subject_min_confidence]
    if len(good) >= 2:
        covered = min(duration, good[-1].t - good[0].t + limits.keyframe_interval_s)
    elif good:
        covered = min(duration, limits.keyframe_interval_s)
    else:
        covered = 0.0
    return mean_conf, max(0.0, min(1.0, covered / duration))


def resolve_target(
    requested: EditTarget,
    tracks: Sequence[SubjectTrack],
    start: float,
    end: float,
    limits: CameraMotionLimits,
    active_speaker: str | None = None,
) -> ResolvedTarget:
    """Deterministic fallback chain; only evidence actually available is used."""
    if not tracks:
        return CENTER_SAFE_TARGET
    by_id = {track.subject_id: track for track in tracks}

    def judge(track: SubjectTrack, source: str) -> ResolvedTarget:
        conf, coverage = track_reliability(track, start, end, limits)
        reliable = conf >= limits.subject_min_confidence and coverage >= limits.subject_min_coverage
        return ResolvedTarget(source, track.subject_id if reliable else None, reliable, round(conf, 3),
                              round(coverage, 3), "reliable" if reliable else "subject confidence/coverage too low")

    candidates: list[tuple[SubjectTrack, str]] = []
    if requested.type is TargetType.SUBJECT and requested.id in by_id:
        candidates.append((by_id[str(requested.id)], "subject"))
    speaker = requested.id if requested.type is TargetType.ACTIVE_SPEAKER and requested.id else active_speaker
    if speaker:
        candidates.extend((t, "active_speaker") for t in tracks
                          if t.speaker_id == speaker and t.speaker_confidence >= limits.speaker_link_min_confidence)
    ranked = sorted(tracks, key=lambda t: (-track_reliability(t, start, end, limits)[1],
                                          -track_reliability(t, start, end, limits)[0], t.subject_id))
    candidates.extend((t, "dominant_subject") for t in ranked)

    last: ResolvedTarget | None = None
    for track, source in candidates:
        verdict = judge(track, source)
        if verdict.reliable:
            return verdict
        last = verdict
    if last is not None:
        return ResolvedTarget("center_safe", None, False, last.mean_confidence, last.coverage,
                              f"fallback to center_safe: {last.reason}")
    return CENTER_SAFE_TARGET


# ============================================================
# SMOOTHED CAMERA CENTER PATH
# ============================================================

def _interpolate(samples: Sequence[SubjectSample], t: float, max_gap: float) -> SubjectSample | None:
    if not samples:
        return None
    if t <= samples[0].t:
        return samples[0] if samples[0].t - t <= max_gap else None
    if t >= samples[-1].t:
        return samples[-1] if t - samples[-1].t <= max_gap else None
    lo, hi = 0, len(samples) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if samples[mid].t <= t:
            lo = mid
        else:
            hi = mid
    a, b = samples[lo], samples[hi]
    if b.t - a.t > 2 * max_gap:
        nearest = a if t - a.t <= b.t - t else b
        return nearest if abs(nearest.t - t) <= max_gap else None
    u = (t - a.t) / max(1e-9, b.t - a.t)
    return SubjectSample(
        t,
        a.cx + (b.cx - a.cx) * u,
        a.cy + (b.cy - a.cy) * u,
        a.width + (b.width - a.width) * u,
        a.height + (b.height - a.height) * u,
        min(a.confidence, b.confidence),
    )


def smooth_center_path(
    samples: Sequence[SubjectSample],
    times: Sequence[float],
    limits: CameraMotionLimits,
    initial: tuple[float, float] = CENTER,
) -> list[tuple[float, float]]:
    """Confidence-gated dead-zone camera with exponential smoothing and a velocity clamp.

    * low-confidence measurement -> hold previous target (not jump to center)
    * sustained low confidence beyond timeout -> drift to center, speed-limited
    * movement inside the dead zone -> hold
    * first-order smoothing with time constant ``smoothing_tau_s`` (low lag)
    * per-step displacement <= ``max_speed_norm_per_s * dt``
    """
    if not times:
        return []
    ordered = sorted(samples, key=lambda s: s.t)
    max_gap = max(limits.keyframe_interval_s * 1.5, 0.25)
    cx, cy = initial
    target_x, target_y = initial
    low_since: float | None = None
    previous_t = float(times[0])
    result: list[tuple[float, float]] = []
    for raw_t in times:
        t = float(raw_t)
        dt = max(0.0, t - previous_t)
        previous_t = t
        measured = _interpolate(ordered, t, max_gap)
        if measured is not None and measured.confidence >= limits.subject_min_confidence:
            low_since = None
            dx = measured.cx - cx
            dy = measured.cy - cy
            # Dead zone: only the excess beyond the zone moves the camera.
            target_x = cx + (math.copysign(abs(dx) - limits.dead_zone_x, dx) if abs(dx) > limits.dead_zone_x else 0.0)
            target_y = cy + (math.copysign(abs(dy) - limits.dead_zone_y, dy) if abs(dy) > limits.dead_zone_y else 0.0)
        else:
            if low_since is None:
                low_since = t
            if t - low_since >= limits.low_confidence_hold_timeout_s:
                target_x, target_y = CENTER
        if dt > 0:
            alpha = 1.0 - math.exp(-dt / max(1e-3, limits.smoothing_tau_s))
            step_x = (target_x - cx) * alpha
            step_y = (target_y - cy) * alpha
            max_step = limits.max_speed_norm_per_s * dt
            norm = math.hypot(step_x, step_y)
            if norm > max_step > 0:
                step_x *= max_step / norm
                step_y *= max_step / norm
            cx += step_x
            cy += step_y
        result.append((min(1.0, max(0.0, cx)), min(1.0, max(0.0, cy))))
    return result


def stable_target_segments(
    observations: Iterable[tuple[float, str | None, float]],
    limits: CameraMotionLimits,
) -> list[tuple[float, str | None]]:
    """Camera hysteresis over (time, candidate_id, confidence) observations.

    A candidate becomes the camera target only when its confidence is at least
    ``speaker_switch_confidence`` AND it stays dominant for at least
    ``speaker_switch_min_hold_s``. Returns (switch_time, target_id) change points;
    this prevents A->B->A->B jitter on interjections/overlap speech.
    """
    rows = sorted(observations, key=lambda row: row[0])
    if not rows:
        return []
    current: str | None = None
    changes: list[tuple[float, str | None]] = []
    pending: str | None = None
    pending_since = 0.0
    for t, candidate, confidence in rows:
        strong = candidate is not None and confidence >= limits.speaker_switch_confidence
        if current is None and strong:
            current = candidate
            changes.append((t, current))
            pending = None
            continue
        if not strong or candidate == current:
            pending = None
            continue
        if candidate != pending:
            pending, pending_since = candidate, t
        if t - pending_since >= limits.speaker_switch_min_hold_s:
            current = pending
            changes.append((pending_since, current))
            pending = None
    return changes
