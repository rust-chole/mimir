"""Deterministic preset resolution: semantic event -> bounded camera geometry.

    semantic preset + intensity + target + context  ->  ResolvedCameraOp

The planner's intensity is never trusted alone: every physical value is
clamped by the style pack, the protected-range/center-safe caps, the subject
safety box and the hard limits in ``camera``. Identical inputs always give
identical output (no randomness anywhere).
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from ai.editor.pro_edit.camera import (
    HARD_MAX_ZOOM,
    IDENTITY,
    CameraPath,
    CameraSegment,
    CameraState,
    OutputProfile,
    base_window,
    clamp,
    lerp,
    verify_path,
)
from ai.editor.pro_edit.context import EditContext
from ai.editor.pro_edit.framing import Box, Framing, FramingRequest, SubjectBox, crop_contains, solve_framing
from ai.editor.pro_edit.story import RequiredRegion, StorySpan
from ai.editor.pro_edit.errors import PresetResolutionError, SubjectResolutionError
from ai.editor.pro_edit.policy import center_safe_zoom_cap
from ai.editor.pro_edit.schema import (
    CameraMode,
    Channel,
    EditEvent,
    EditPlan,
    IntroDirective,
    MotionPreset,
    StoryRole,
    SfxCue,
    SupportVisual,
    TargetType,
)
from ai.editor.pro_edit.style import StylePack
from ai.editor.pro_edit.subjects import (
    ResolvedTarget,
    SubjectSample,
    SubjectTrack,
    resolve_target,
    smooth_center_path,
    stable_target_segments,
    track_reliability,
)
from ai.editor.pro_edit.timebase import FrameRate

PRESET_ENGINE_VERSION = 2
MIN_OP_FRAMES = 4
MIN_RAMP_FRAMES = 2


@dataclass(frozen=True)
class ResolvedParams:
    preset: str
    camera: str
    intensity: float
    scale_start: float
    scale_peak: float
    attack_s: float
    hold_s: float
    release_s: float
    anchor_x: float
    anchor_y: float
    easing_in: str
    easing_out: str
    zoom_cap: float
    target_source: str
    target_subject: str | None
    target_reliable: bool
    subject_confidence: float
    merged_with_next: bool = False
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale_start": round(self.scale_start, 4),
            "scale_peak": round(self.scale_peak, 4),
            "attack_s": round(self.attack_s, 3),
            "hold_s": round(self.hold_s, 3),
            "release_s": round(self.release_s, 3),
            "anchor_x": round(self.anchor_x, 4),
            "anchor_y": round(self.anchor_y, 4),
            "easing_in": self.easing_in,
            "easing_out": self.easing_out,
            "zoom_cap": round(self.zoom_cap, 4),
            "target": {"source": self.target_source, "subject_id": self.target_subject,
                       "reliable": self.target_reliable, "confidence": round(self.subject_confidence, 3)},
            "merged_with_next": self.merged_with_next,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ResolvedCameraOp:
    source_event_id: str
    start_frame: int
    end_frame: int
    attack_frames: int
    hold_frames: int
    release_frames: int
    params: ResolvedParams
    segments: tuple[CameraSegment, ...]
    channel: Channel = Channel.CAMERA


@dataclass(frozen=True)
class ResolvedPlan:
    engine_version: int
    clip_identity: str
    style_name: str
    style_version: int
    fps: FrameRate
    frame_count: int
    source_width: int
    source_height: int
    output_profile: OutputProfile
    base: tuple[int, int, int, int]
    ops: tuple[ResolvedCameraOp, ...]
    path: CameraPath
    dropped: tuple[tuple[str, str], ...] = ()
    recommendations: tuple[dict[str, Any], ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def output_size(self) -> tuple[int, int]:
        return self.base[2], self.base[3]

    @property
    def is_identity(self) -> bool:
        return self.path.is_identity and self.output_profile is OutputProfile.PRESERVE

    def trace(self) -> list[dict[str, Any]]:
        return [
            {
                "event_id": op.source_event_id,
                "preset": op.params.preset,
                "camera": op.params.camera,
                "frames": [op.start_frame, op.end_frame],
                "resolved": op.params.to_dict(),
            }
            for op in self.ops
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_version": self.engine_version,
            "clip_identity": self.clip_identity,
            "style": {"name": self.style_name, "version": self.style_version},
            "fps": str(self.fps),
            "frame_count": self.frame_count,
            "source_size": [self.source_width, self.source_height],
            "output_profile": self.output_profile.value,
            "base_window": list(self.base),
            "active_frame_ranges": [list(r) for r in self.path.active_ranges()],
            "ops": self.trace(),
            "dropped": [{"event_id": e, "reason": r} for e, r in self.dropped],
            "recommendations_not_executed_v1": [dict(r) for r in self.recommendations],
            "metrics": dict(self.metrics),
        }


# ============================================================
# SUBJECT GEOMETRY (source-normalized -> base-window-normalized)
# ============================================================

def _to_base(sample: SubjectSample, source_w: int, source_h: int, base: tuple[int, int, int, int]) -> SubjectSample:
    bx, by, bw, bh = base
    return replace(
        sample,
        cx=clamp((sample.cx * source_w - bx) / bw, 0.0, 1.0),
        cy=clamp((sample.cy * source_h - by) / bh, 0.0, 1.0),
        width=min(1.0, sample.width * source_w / bw),
        height=min(1.0, sample.height * source_h / bh),
    )


def _box_to_base(region: RequiredRegion, source_w: int, source_h: int, base: tuple[int, int, int, int]) -> Box:
    bx, by, bw, bh = base
    return Box((region.x0 * source_w - bx) / bw, (region.y0 * source_h - by) / bh,
               (region.x1 * source_w - bx) / bw, (region.y1 * source_h - by) / bh).clipped()


def track_window(track: SubjectTrack, start: float, end: float, min_conf: float, source_w: int, source_h: int,
                 base: tuple[int, int, int, int]) -> tuple[Box, tuple[SubjectSample, ...]]:
    """Median subject box + reliable samples inside [start, end] (base coords)."""
    samples = tuple(_to_base(s, source_w, source_h, base) for s in track.window(start, end)
                    if s.confidence >= min_conf)
    if not samples:
        raise SubjectResolutionError(f"subject {track.subject_id} has no reliable samples in window")
    box = Box.from_center(statistics.median(s.cx for s in samples), statistics.median(s.cy for s in samples),
                          statistics.median(s.width for s in samples), statistics.median(s.height for s in samples))
    return box, samples


@dataclass(frozen=True)
class StoryConstraints:
    required: tuple[Box, ...] = ()
    min_visible_fraction: float = 0.0
    span_ids: tuple[str, ...] = ()
    caption_top: float | None = None
    avoid_bands: tuple[tuple[float, float], ...] = ()


def story_constraints(context: EditContext, spans: Sequence[StorySpan], start: float, end: float,
                      base: tuple[int, int, int, int], style: StylePack, *, captions: bool,
                      hook_text: bool = False) -> StoryConstraints:
    """Geometry the crop must respect inside [start, end] (story + captions +
    burned intro hook text)."""
    required: list[Box] = []
    min_visible = 0.0
    ids: list[str] = []
    W, H = context.clip.width, context.clip.height
    by_id = {t.subject_id: t for t in context.subject_tracks}
    for span in spans:
        if not span.overlaps(start, end):
            continue
        ids.append(span.span_id)
        min_visible = max(min_visible, span.min_visible_fraction)
        required.extend(_box_to_base(r, W, H, base) for r in span.required_regions)
        for subject_id in span.required_subject_ids:
            track = by_id.get(subject_id)
            if track is None:
                continue
            try:
                box, _ = track_window(track, max(start, span.start), min(end, span.end),
                                      style.camera.subject_min_confidence, W, H, base)
            except SubjectResolutionError:
                continue
            required.append(box)
    caption_top = None
    avoid: list[tuple[float, float]] = []
    if captions:
        bands = context.caption_region.active_bands(start, end)
        lower = [b for b in bands if (b.y0 + b.y1) / 2.0 >= 0.5]
        caption_top = min((b.y0 for b in lower), default=None)
        # Captions placed in the upper half: keep the face fully above or below them.
        avoid.extend((b.y0, b.y1) for b in bands if (b.y0 + b.y1) / 2.0 < 0.5)
    if hook_text and context.hook_band is not None and context.hook_band.active(start, end):
        avoid.append((context.hook_band.y0, context.hook_band.y1))
    return StoryConstraints(tuple(required), min_visible, tuple(ids), caption_top, tuple(avoid))


def active_speaker_follow_samples(
    context: EditContext,
    event: EditEvent,
    style: StylePack,
    base: tuple[int, int, int, int],
) -> tuple[tuple[SubjectSample, ...], tuple[str, ...]]:
    """Multi-speaker speaker_focus with hysteresis.

    Word-level speaker evidence -> ``stable_target_segments`` (confidence +
    minimum dominance hold) -> follow samples taken from the currently stable
    speaker's track. Only evidence-linked tracks (speaker_confidence) qualify.
    """
    limits = style.camera
    linked: dict[str, SubjectTrack] = {}
    for track in context.subject_tracks:
        if track.speaker_id is None or track.speaker_confidence < limits.speaker_link_min_confidence:
            continue
        conf, coverage = track_reliability(track, event.start, event.end, limits)
        if conf >= limits.subject_min_confidence and coverage >= limits.subject_min_coverage:
            linked.setdefault(track.speaker_id, track)
    if len(linked) < 2:
        return (), ()
    observations = [((w.start + w.end) / 2.0, w.speaker_id, 0.9) for w in context.words
                    if w.speaker_id in linked and event.start - 1.0 <= w.start <= event.end]
    changes = stable_target_segments(observations, limits)
    if not changes:
        return (), ()
    samples: list[SubjectSample] = []
    order: list[str] = []
    W, H = context.clip.width, context.clip.height
    for index, (switch_time, speaker) in enumerate(changes):
        if speaker is None:
            continue
        until = changes[index + 1][0] if index + 1 < len(changes) else event.end + 1.0
        lo = event.start if index == 0 else max(event.start, switch_time)
        if until <= event.start or lo >= event.end:
            continue
        _, window = track_window(linked[speaker], lo, min(until, event.end), limits.subject_min_confidence, W, H, base)
        samples.extend(window)
        order.append(speaker)
    return tuple(sorted(samples, key=lambda s: s.t)), tuple(order)


# ============================================================
# PER-EVENT PREPARATION
# ============================================================

@dataclass(frozen=True)
class _Prepared:
    event: EditEvent
    start_frame: int
    end_frame: int
    attack_frames: int
    release_frames: int
    hold_min_frames: int
    zoom: float
    zoom_cap: float
    anchor: tuple[float, float]
    target: ResolvedTarget
    follow_samples: tuple[SubjectSample, ...]
    follow_size: tuple[float, float]
    subject_kind: str
    constraints: StoryConstraints
    easing_in: str
    easing_out: str
    notes: tuple[str, ...]
    caption_clear: bool | None = None


def _frames(seconds: float, fps: FrameRate) -> int:
    return max(0, int(round(seconds * fps.fps)))


def allocate_ramps(total: int, attack: int, release: int, min_hold: int) -> tuple[int, int, int]:
    """Split ``total`` frames into attack/hold/release (all ramps >= 2 frames)."""
    if total < MIN_OP_FRAMES:
        raise PresetResolutionError(f"event spans {total} frames (< {MIN_OP_FRAMES})")
    attack = max(MIN_RAMP_FRAMES, attack)
    release = max(0, release)
    if release:
        release = max(MIN_RAMP_FRAMES, release)
    min_hold = max(0, min(min_hold, total - MIN_RAMP_FRAMES - (MIN_RAMP_FRAMES if release else 0)))
    if attack + release + min_hold > total:
        available = total - min_hold
        ramps = attack + release
        attack = max(MIN_RAMP_FRAMES, int(math.floor(attack * available / ramps)))
        if release:
            release = max(MIN_RAMP_FRAMES, available - attack)
        if attack + release > available:
            attack = max(MIN_RAMP_FRAMES, available - release)
    hold = total - attack - release
    if hold < 0 or attack < MIN_RAMP_FRAMES:
        raise PresetResolutionError("event too short for preset ramps")
    return attack, hold, release


def _dual_subjects(context: EditContext, event: EditEvent, style: StylePack,
                   base: tuple[int, int, int, int]) -> list[SubjectBox]:
    limits = style.camera
    ranked = []
    for track in context.subject_tracks:
        conf, coverage = track_reliability(track, event.start, event.end, limits)
        if conf >= limits.subject_min_confidence and coverage >= limits.subject_min_coverage:
            ranked.append((-coverage, -conf, track.subject_id, track))
    boxes = []
    for _, _, _, track in sorted(ranked)[:2]:
        box, _ = track_window(track, event.start, event.end, limits.subject_min_confidence,
                              context.clip.width, context.clip.height, base)
        boxes.append(SubjectBox(box, track.kind, track.subject_id))
    return boxes if len(boxes) == 2 else []


def _prepare(event: EditEvent, context: EditContext, style: StylePack, base: tuple[int, int, int, int], *,
             spans: Sequence[StorySpan] | None = None, captions: bool = True,
             hook_text: bool = False) -> _Prepared:
    fps = context.clip.fps
    frame_count = context.clip.frame_count
    start_frame = int(clamp(fps.frame_index(event.start), 0, frame_count))
    end_frame = int(clamp(fps.frame_index(event.end), 0, frame_count))
    total = end_frame - start_frame
    if total < MIN_OP_FRAMES:
        raise PresetResolutionError(f"{event.event_id}: {total} frames is too short")
    motion = event.motion
    if motion not in style.presets:
        raise PresetResolutionError(f"{event.event_id}: preset {motion.value} not in style {style.name}")
    bounds = style.presets[motion]
    intensity = clamp(float(event.intensity), 0.0, 1.0)
    limits = style.camera
    notes: list[str] = []
    W, H = context.clip.width, context.clip.height

    active_speaker = context.active_speaker(event.start, event.end)
    target = resolve_target(event.target, context.subject_tracks, event.start, event.end, limits, active_speaker)
    subjects: list[SubjectBox] = []
    follow: tuple[SubjectSample, ...] = ()
    kind = "unknown"
    if event.camera is CameraMode.DUAL_SUBJECT:
        subjects = _dual_subjects(context, event, style, base)
        if subjects:
            notes.append("dual_subject:" + "+".join(s.subject_id for s in subjects))
    if not subjects and target.subject_id is not None:
        track = next(t for t in context.subject_tracks if t.subject_id == target.subject_id)
        box, follow = track_window(track, event.start, event.end, limits.subject_min_confidence, W, H, base)
        subjects = [SubjectBox(box, track.kind, track.subject_id)]
        kind = track.kind
        notes.append(f"target:{target.source}:{track.subject_id}")
    if motion is MotionPreset.SPEAKER_FOCUS and event.target.type is TargetType.ACTIVE_SPEAKER:
        speaker_samples, order = active_speaker_follow_samples(context, event, style, base)
        if len(order) >= 2:
            follow = speaker_samples
            kind = "face"
            first = speaker_samples[0]
            subjects = [SubjectBox(Box.from_center(first.cx, first.cy,
                                                   statistics.median(s.width for s in speaker_samples),
                                                   statistics.median(s.height for s in speaker_samples)),
                                   "face", "active_speaker")]
            notes.append("active_speaker_hysteresis:" + ">".join(order))
    if subjects:
        max_zoom = min(style.zoom.max_zoom, HARD_MAX_ZOOM)
        kind = subjects[0].kind
    else:
        max_zoom = min(style.zoom.max_zoom, center_safe_zoom_cap(context, style, event.start, event.end))
        notes.append(f"center_safe:{target.reason}")

    camera_band = style.framing.get(event.camera)
    framing_zoom = camera_band.at(intensity) if camera_band is not None else 1.0
    if motion is MotionPreset.SPEAKER_FOCUS:
        desired = max(framing_zoom, bounds.scale.at(intensity), style.zoom.pan_min_zoom)
    elif motion is MotionPreset.SUBTLE_PULL_OUT:
        desired = 1.0
    else:
        desired = max(bounds.scale.at(intensity), framing_zoom)

    constraints = story_constraints(context, spans if spans is not None else context.spans, event.start,
                                    event.end, base, style, captions=captions, hook_text=hook_text)
    framing = solve_framing(FramingRequest(
        desired_zoom=desired, camera=event.camera, max_zoom=max_zoom, subjects=tuple(subjects),
        required=constraints.required, min_visible_fraction=constraints.min_visible_fraction,
        caption_top=constraints.caption_top, avoid_bands=constraints.avoid_bands))
    notes.extend(framing.notes)
    zoom = framing.zoom
    if motion is not MotionPreset.SUBTLE_PULL_OUT and zoom <= 1.0 + 1e-4:
        raise PresetResolutionError(f"{event.event_id}: no visible camera change after bounds")

    release_s = bounds.release_s.at(intensity)
    attack_s = bounds.attack_s.at(intensity)
    # A preset is only active for its bounded window; the rest of a long event
    # keeps the original framing (a punch-in never becomes an 8 s zoom hold).
    max_frames = max(MIN_OP_FRAMES, _frames(bounds.max_active_s + release_s, fps))
    if total > max_frames:
        end_frame = start_frame + max_frames
        total = max_frames
        notes.append("capped_to_preset_max_active")
    active_s = total / fps.fps
    if motion is MotionPreset.SLOW_PUSH:
        attack_s = min(attack_s, max(0.0, active_s - release_s))
    min_hold_frames = _frames(style.role(event.role).min_hold_s, fps)
    # Pre-validate both ramp layouts so segment construction cannot fail later.
    allocate_ramps(total, _frames(attack_s, fps), _frames(release_s, fps), min_hold_frames)
    allocate_ramps(total, _frames(attack_s, fps), 0, min_hold_frames)
    size = (subjects[0].box.w, subjects[0].box.h) if subjects else (0.0, 0.0)
    return _Prepared(
        event=event, start_frame=start_frame, end_frame=end_frame,
        attack_frames=_frames(attack_s, fps), release_frames=_frames(release_s, fps),
        hold_min_frames=min_hold_frames, zoom=zoom, zoom_cap=max_zoom, anchor=framing.anchor, target=target,
        follow_samples=follow if motion is MotionPreset.SPEAKER_FOCUS else (), follow_size=size,
        subject_kind=kind, constraints=constraints, easing_in=bounds.easing_in, easing_out=bounds.easing_out,
        notes=tuple(notes), caption_clear=framing.caption_clear,
    )


def verify_story_geometry(segments: Sequence[CameraSegment], constraints: StoryConstraints) -> None:
    """Payoff visibility is geometric: every frame of an op must keep the
    required regions inside the crop and the minimum visible fraction."""
    if not constraints.required and constraints.min_visible_fraction <= 0:
        return
    for segment in segments:
        step = max(1, segment.length // 8)
        for frame in list(range(segment.start_frame, segment.end_frame, step)) + [segment.end_frame - 1]:
            state = segment.state_at(frame)
            if constraints.min_visible_fraction > 0 and 1.0 / state.zoom < constraints.min_visible_fraction - 1e-6:
                raise PresetResolutionError(
                    f"story geometry: frame {frame} keeps {1.0 / state.zoom:.3f} < "
                    f"{constraints.min_visible_fraction} of the frame ({','.join(constraints.span_ids)})")
            for box in constraints.required:
                if not crop_contains(state.zoom, (state.cx, state.cy), box):
                    raise PresetResolutionError(
                        f"story geometry: frame {frame} crops a required region {box.to_list()} "
                        f"({','.join(constraints.span_ids)})")


# ============================================================
# SEGMENT CONSTRUCTION
# ============================================================

def _follow_keyframes(prep: _Prepared, entry: CameraState, frames: list[int], fps: FrameRate,
                      style: StylePack) -> list[tuple[float, float]]:
    """Smooth the SUBJECT (camera-operator target), then frame each keyframe.

    Raw detections never drive the crop: dead zone + EMA + velocity clamp +
    confidence hold act on the subject center; the framing solver then maps
    the smoothed subject to a crop at the op's fixed zoom. The caption rule
    is decided once for the whole op so it can never toggle mid-shot.
    """
    if not prep.follow_samples:
        return [prep.anchor for _ in frames]
    times = [f / fps.fps for f in frames]
    measured = [(s.cx, s.cy) for s in prep.follow_samples]
    initial = measured[0] if entry.close_to(IDENTITY) else (entry.cx, entry.cy)
    centers = smooth_center_path(prep.follow_samples, times, style.camera, initial=initial)
    w, h = prep.follow_size

    def frame_all(caption_top: float | None, avoid: tuple[tuple[float, float], ...]) -> list[Framing]:
        return [solve_framing(FramingRequest(
            desired_zoom=prep.zoom, camera=prep.event.camera, max_zoom=prep.zoom,
            subjects=(SubjectBox(Box.from_center(cx, cy, w, h), prep.subject_kind),),
            required=prep.constraints.required, min_visible_fraction=prep.constraints.min_visible_fraction,
            caption_top=caption_top, avoid_bands=avoid)) for cx, cy in centers]

    caption_top, avoid = prep.constraints.caption_top, prep.constraints.avoid_bands
    framings = frame_all(caption_top, avoid)
    if avoid and any(f.text_clear is False for f in framings):
        avoid = ()
        framings = frame_all(caption_top, avoid)
    if caption_top is not None and any(f.caption_clear is False for f in framings):
        framings = frame_all(None, avoid)
    return [f.anchor for f in framings]


def _build_segments(prep: _Prepared, entry: CameraState, merge_next: bool, style: StylePack,
                    fps: FrameRate) -> tuple[list[CameraSegment], CameraState, ResolvedParams, tuple[int, int, int]]:
    event = prep.event
    eid = event.event_id
    total = prep.end_frame - prep.start_frame
    release = 0 if merge_next or event.motion is MotionPreset.SUBTLE_PULL_OUT else prep.release_frames
    attack, hold, release = allocate_ramps(total, prep.attack_frames, release, prep.hold_min_frames)
    sf = prep.start_frame
    a_end = sf + attack
    h_end = a_end + hold
    segments: list[CameraSegment] = []

    if event.motion is MotionPreset.SUBTLE_PULL_OUT:
        if entry.close_to(IDENTITY):
            raise PresetResolutionError(f"{eid}: subtle_pull_out needs a zoomed entry state")
        segments.append(CameraSegment(sf, a_end, entry, IDENTITY, prep.easing_in, eid, "attack"))
        if hold:
            segments.append(CameraSegment(a_end, h_end, IDENTITY, IDENTITY, "linear", eid, "hold"))
        exit_state = IDENTITY
        peak = IDENTITY
    else:
        if event.motion in (MotionPreset.SPEAKER_FOCUS,):
            step = max(1, _frames(style.camera.keyframe_interval_s, fps))
            key_frames = [a_end - 1] + list(range(a_end - 1 + step, h_end - 1, step)) + [h_end - 1]
            key_frames = sorted(set(max(sf, k) for k in key_frames))
            centers = _follow_keyframes(prep, entry, key_frames, fps, style)
            peak = CameraState(prep.zoom, *centers[0])
            segments.append(CameraSegment(sf, a_end, entry, peak, prep.easing_in, eid, "attack"))
            previous_frame, previous_state = a_end - 1, peak
            for frame, center in zip(key_frames[1:], centers[1:]):
                state = CameraState(prep.zoom, *center)
                if frame <= previous_frame:
                    continue
                # Segment covers (previous_frame, frame]; last frame lands exactly on ``state``.
                segments.append(CameraSegment(previous_frame + 1, frame + 1, _advance(previous_state, state,
                                              previous_frame, frame), state, "linear", eid, "follow"))
                previous_frame, previous_state = frame, state
            if previous_frame < h_end - 1:
                segments.append(CameraSegment(previous_frame + 1, h_end, previous_state, previous_state, "linear",
                                              eid, "hold"))
            exit_state = previous_state
        else:
            target_center = prep.anchor
            if event.motion is MotionPreset.SNAP_REFRAME:
                dx, dy = target_center[0] - entry.cx, target_center[1] - entry.cy
                distance = math.hypot(dx, dy)
                limit = style.camera.max_reframe_distance_norm
                if distance > limit > 0:
                    target_center = (entry.cx + dx * limit / distance, entry.cy + dy * limit / distance)
            zoom = prep.zoom
            if event.motion is MotionPreset.SNAP_REFRAME:
                zoom = min(max(prep.zoom, entry.zoom), max(1.0, prep.zoom_cap))
            peak = CameraState(clamp(zoom, 1.0, HARD_MAX_ZOOM), *target_center)
            segments.append(CameraSegment(sf, a_end, entry, peak, prep.easing_in, eid, "attack"))
            if hold:
                segments.append(CameraSegment(a_end, h_end, peak, peak, "linear", eid, "hold"))
            exit_state = peak
        if release:
            segments.append(CameraSegment(h_end, prep.end_frame, exit_state, IDENTITY, prep.easing_out, eid,
                                          "release"))
            exit_state = IDENTITY
    verify_story_geometry(segments, prep.constraints)

    params = ResolvedParams(
        preset=event.motion.value,
        camera=event.camera.value,
        intensity=event.intensity,
        scale_start=entry.zoom,
        scale_peak=peak.zoom,
        attack_s=attack / fps.fps,
        hold_s=hold / fps.fps,
        release_s=release / fps.fps,
        anchor_x=peak.cx,
        anchor_y=peak.cy,
        easing_in=prep.easing_in,
        easing_out=prep.easing_out,
        zoom_cap=prep.zoom_cap,
        target_source=prep.target.source,
        target_subject=prep.target.subject_id,
        target_reliable=prep.target.reliable,
        subject_confidence=prep.target.mean_confidence,
        merged_with_next=merge_next,
        notes=prep.notes,
    )
    return segments, exit_state, params, (attack, hold, release)


def _advance(a: CameraState, b: CameraState, frame_a: int, frame_b: int) -> CameraState:
    """State one frame after ``frame_a`` on the straight line a->b (keeps constant velocity)."""
    span = max(1, frame_b - frame_a)
    t = 1.0 / span
    return CameraState(lerp(a.zoom, b.zoom, t), lerp(a.cx, b.cx, t), lerp(a.cy, b.cy, t))


def _recommendations(plan: EditPlan, fps: FrameRate) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for event in plan.events:
        frame = fps.frame_index(event.start)
        if event.sfx is not SfxCue.NONE:
            rows.append({"channel": Channel.SFX.value, "event_id": event.event_id, "frame": frame,
                         "value": event.sfx.value, "executed": False})
        if event.support_visual is not SupportVisual.NONE:
            rows.append({"channel": Channel.SUPPORT_VISUAL.value, "event_id": event.event_id, "frame": frame,
                         "value": event.support_visual.value, "executed": False})
        if event.emphasis_word_ids or event.caption_style.value != "default":
            rows.append({"channel": Channel.CAPTION_STYLE.value, "event_id": event.event_id, "frame": frame,
                         "value": event.caption_style.value, "word_ids": list(event.emphasis_word_ids),
                         "executed": False})
    return tuple(rows)


def resolve_plan(
    plan: EditPlan,
    context: EditContext,
    style: StylePack,
    *,
    output_profile: OutputProfile = OutputProfile.PRESERVE,
) -> ResolvedPlan:
    """Resolve a VALIDATED plan. Per-event failures drop only that event
    (its frames keep the original pixels); structural failures raise."""
    fps = context.clip.fps
    frame_count = context.clip.frame_count
    base = base_window(context.clip.width, context.clip.height, output_profile)
    dropped: list[tuple[str, str]] = []
    prepared: list[_Prepared] = []
    for event in sorted(plan.camera_events, key=lambda e: (e.start, e.event_id)):
        try:
            prepared.append(_prepare(event, context, style, base))
        except (PresetResolutionError, SubjectResolutionError) as error:
            dropped.append((event.event_id, str(error)))

    merge_gap = _frames(style.durations.merge_gap_s, fps)
    segments: list[CameraSegment] = []
    ops: list[ResolvedCameraOp] = []
    state = IDENTITY
    cursor = 0
    for index, prep in enumerate(prepared):
        if prep.start_frame < cursor:
            # Frame rounding made two adjacent events touch: trim, never overlap.
            if prep.end_frame - cursor < MIN_OP_FRAMES:
                dropped.append((prep.event.event_id, "no frames left after previous camera op"))
                continue
            prep = replace(prep, start_frame=cursor)
        if prep.event.motion is MotionPreset.SUBTLE_PULL_OUT and state.close_to(IDENTITY):
            dropped.append((prep.event.event_id, "subtle_pull_out needs a zoomed entry state"))
            continue
        following = prepared[index + 1] if index + 1 < len(prepared) else None
        merge = bool(following and 0 <= following.start_frame - prep.end_frame <= merge_gap)
        if prep.start_frame > cursor and not state.close_to(IDENTITY):
            segments.append(CameraSegment(cursor, prep.start_frame, state, state, "linear", "", "bridge_hold"))
        try:
            op_segments, exit_state, params, (attack, hold, release) = _build_segments(prep, state, merge, style, fps)
        except (PresetResolutionError, SubjectResolutionError) as error:
            if not state.close_to(IDENTITY):
                raise PresetResolutionError(
                    f"{prep.event.event_id}: cannot continue a merged camera move: {error}") from error
            dropped.append((prep.event.event_id, str(error)))
            continue
        segments.extend(op_segments)
        ops.append(ResolvedCameraOp(prep.event.event_id, prep.start_frame, prep.end_frame, attack, hold, release,
                                    params, tuple(op_segments)))
        state = exit_state
        cursor = prep.end_frame
    if not state.close_to(IDENTITY):
        # Only reachable if the op a camera move merged into was dropped.
        release_end = min(frame_count, cursor + max(MIN_RAMP_FRAMES, _frames(0.3, fps)))
        if release_end - cursor < MIN_RAMP_FRAMES:
            raise PresetResolutionError("camera cannot return to identity before clip end")
        segments.append(CameraSegment(cursor, release_end, state, IDENTITY, "smoothstep", "", "final_release"))

    path = CameraPath(frame_count, tuple(segments))
    metrics = verify_path(path, max_zoom=min(style.zoom.max_zoom, HARD_MAX_ZOOM), fps=fps.fps,
                          max_follow_speed=style.camera.max_speed_norm_per_s)
    metrics["active_frames"] = float(sum(b - a for a, b in path.active_ranges()))
    return ResolvedPlan(
        engine_version=PRESET_ENGINE_VERSION,
        clip_identity=context.clip_identity,
        style_name=style.name,
        style_version=style.version,
        fps=fps,
        frame_count=frame_count,
        source_width=context.clip.width,
        source_height=context.clip.height,
        output_profile=output_profile,
        base=base,
        ops=tuple(ops),
        path=path,
        dropped=tuple(dropped),
        recommendations=_recommendations(plan, fps),
        metrics=metrics,
    )


def resolved_ops_by_event(resolved: ResolvedPlan) -> dict[str, ResolvedCameraOp]:
    return {op.source_event_id: op for op in resolved.ops}


def sample_zoom_curve(resolved: ResolvedPlan, frames: Sequence[int]) -> list[float]:
    return [resolved.path.state_at(f).zoom for f in frames]


def resolve_intro_plan(
    directive: IntroDirective | None,
    context: EditContext,
    style: StylePack,
    *,
    output_profile: OutputProfile = OutputProfile.PRESERVE,
) -> ResolvedPlan | None:
    """Camera for the EXISTING cold-open, in the clean paced clip's frame domain.

    The op covers exactly the selected teaser source range (MIMIR intro
    selection is read-only) and is constrained by the intro span: peak action
    visibility, required regions. Frames outside the teaser stay untouched,
    so the intro renderer's trim/handoff math is unchanged. No captions exist
    on the teaser (only the hook ASS, burned by the intro renderer on top).
    """
    if directive is None or not directive.is_camera_active or context.intro is None or context.intro_span is None:
        return None
    if output_profile is not OutputProfile.PRESERVE:
        raise PresetResolutionError("intro camera supports only the preserve output profile")
    fps = context.clip.fps
    event = EditEvent(
        event_id="intro", start=context.intro.teaser_start, end=context.intro.teaser_end,
        role=StoryRole.PAYOFF, camera=directive.camera, motion=directive.motion, target=directive.target,
        intensity=directive.intensity, confidence=directive.confidence, reason_code=directive.reason_code,
        story_span_id=context.intro_span.span_id,
    )
    base = base_window(context.clip.width, context.clip.height, OutputProfile.PRESERVE)
    prep = _prepare(event, context, style, base, spans=(context.intro_span,), captions=False, hook_text=True)
    segments, _exit, params, (attack, hold, release) = _build_segments(prep, IDENTITY, False, style, fps)
    if segments and segments[-1].end_frame > fps.frame_index(context.intro.teaser_end) + 1:
        raise PresetResolutionError("intro camera would leave the selected teaser range")
    path = CameraPath(context.clip.frame_count, tuple(segments))
    metrics = verify_path(path, max_zoom=min(style.zoom.max_zoom, HARD_MAX_ZOOM), fps=fps.fps,
                          max_follow_speed=style.camera.max_speed_norm_per_s)
    metrics["active_frames"] = float(sum(b - a for a, b in path.active_ranges()))
    op = ResolvedCameraOp("intro", prep.start_frame, prep.end_frame, attack, hold, release, params, tuple(segments))
    return ResolvedPlan(
        engine_version=PRESET_ENGINE_VERSION, clip_identity=context.clip_identity, style_name=style.name,
        style_version=style.version, fps=fps, frame_count=context.clip.frame_count,
        source_width=context.clip.width, source_height=context.clip.height,
        output_profile=OutputProfile.PRESERVE, base=base, ops=(op,), path=path, metrics=metrics,
    )
