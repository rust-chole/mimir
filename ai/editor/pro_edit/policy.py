"""Editorial semantics (separate from renderer mathematics).

Pure functions over validated EditEvents. Every rule is deterministic and
biased toward LESS editing: uncertain -> weaker/static, conflict -> the
higher story priority wins, budgets exhausted -> downgrade, never invent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

from ai.editor.pro_edit.context import EditContext
from ai.editor.pro_edit.schema import (
    ROLE_PRIORITY,
    STRONG_MOTIONS,
    CameraMode,
    CaptionStyle,
    EditEvent,
    EditTarget,
    MotionPreset,
    SfxCue,
    StoryRole,
    SupportVisual,
    TargetType,
)
from ai.editor.pro_edit.style import StylePack

OVERLAY_LAYOUT_CONTENT_TYPES = frozenset({"gameplay", "streamer_gameplay"})


@dataclass(frozen=True)
class PolicyNote:
    code: str
    message: str
    event_id: str | None = None


# ============================================================
# BUDGET / METRICS
# ============================================================

def effect_budget(duration_s: float, style: StylePack) -> dict[str, int]:
    """Effect budget scales with clip duration (at least 1 of each)."""
    tens = max(0.0, float(duration_s)) / 10.0
    b = style.budget

    def scaled(per_10s: float, cap: int) -> int:
        return int(max(1, min(cap, math.floor(per_10s * tens + 0.5))))

    return {
        "max_strong_motion_events": scaled(b.strong_motion_per_10s, b.max_strong_motion_events),
        "max_support_visual_events": scaled(b.support_visual_per_10s, b.max_support_visual_events),
        "max_snap_reframes": scaled(b.snap_reframe_per_10s, b.max_snap_reframes),
        "max_sfx_events": scaled(b.sfx_per_10s, b.max_sfx_events),
        "max_camera_events": scaled(b.camera_events_per_10s, 48),
    }


def density_metrics(events: Sequence[EditEvent], duration_s: float) -> dict[str, float]:
    duration = max(1e-6, float(duration_s))
    camera = sorted((e for e in events if e.is_camera_active), key=lambda e: e.start)
    strong = [e for e in camera if e.is_strong]
    repeated = 0
    run = 0
    previous: MotionPreset | None = None
    for event in camera:
        run = run + 1 if event.motion is previous else 1
        previous = event.motion
        repeated = max(repeated, run)
    return {
        "strong_motion_seconds_ratio": round(sum(e.duration for e in strong) / duration, 4),
        "strong_events_per_10s": round(len(strong) * 10.0 / duration, 4),
        "camera_changes_per_10s": round(len(camera) * 10.0 / duration, 4),
        "max_repeated_same_preset": float(repeated),
        "camera_events": float(len(camera)),
        "strong_events": float(len(strong)),
    }


def priority_key(event: EditEvent) -> tuple[int, float, float, float, str]:
    """Higher sorts first under reverse=True: story priority > confidence > intensity > earlier."""
    return (ROLE_PRIORITY[event.role], round(event.confidence, 6), round(event.intensity, 6),
            -round(event.start, 6), event.event_id)


# ============================================================
# PER-EVENT RULES
# ============================================================

def gate_confidence(event: EditEvent, style: StylePack) -> tuple[EditEvent, PolicyNote | None]:
    """Editorial confidence gate: low -> static, medium -> mild only."""
    c = style.confidence
    if event.confidence < c.low and event.is_camera_active:
        return (replace(event, motion=MotionPreset.STATIC_CLEAN, camera=CameraMode.PRESERVE),
                PolicyNote("confidence_low_static", f"confidence {event.confidence:.2f} < {c.low}", event.event_id))
    if event.confidence < c.medium:
        changed = event
        if event.motion in STRONG_MOTIONS:
            changed = replace(changed, motion=MotionPreset.SLOW_PUSH)
        if changed.intensity > c.mild_intensity_cap:
            changed = replace(changed, intensity=c.mild_intensity_cap)
        if changed != event:
            return changed, PolicyNote("confidence_medium_mild_only",
                                       f"confidence {event.confidence:.2f} < {c.medium}: mild motion only",
                                       event.event_id)
        return event, None
    if event.confidence < c.high and event.intensity > c.medium_intensity_cap:
        return (replace(event, intensity=c.medium_intensity_cap),
                PolicyNote("confidence_intensity_cap", f"confidence {event.confidence:.2f} < {c.high}",
                           event.event_id))
    return event, None


def _has_reaction_evidence(context: EditContext, start: float, end: float) -> bool:
    if any(r.start < end and start < r.end for r in context.story.reaction_ranges):
        return True
    return any(v.type == "reaction" and v.start < end and start < v.end for v in context.visual_events)


def enforce_role_policy(event: EditEvent, context: EditContext, style: StylePack) -> tuple[EditEvent, list[PolicyNote]]:
    notes: list[PolicyNote] = []
    eid = event.event_id

    # Story roles/spans are owned by the story system, not by the planner.
    owned_span = context.span(event.story_span_id) or context.dominant_span(event.start, event.end)
    owned_role = owned_span.role if owned_span is not None else context.dominant_role(event.start, event.end)
    if owned_span is not None and event.story_span_id != owned_span.span_id:
        notes.append(PolicyNote("story_span_derived", f"linked to {owned_span.span_id}", eid))
        event = replace(event, story_span_id=owned_span.span_id)
    if owned_role is not event.role:
        notes.append(PolicyNote("role_owned_by_story_system",
                                f"planner role {event.role.value} -> story role {owned_role.value}", eid))
        event = replace(event, role=owned_role)
    policy = style.role(event.role)
    touched_spans = context.spans_overlapping(event.start, event.end)
    if event.is_camera_active and any(not span.camera_allowed for span in touched_spans):
        notes.append(PolicyNote("story_span_camera_locked",
                                "a touched story span does not allow camera edits", eid))
        event = replace(event, motion=MotionPreset.STATIC_CLEAN, camera=CameraMode.PRESERVE)
    if event.support_visual is not SupportVisual.NONE and any(not s.support_visual_allowed for s in touched_spans):
        notes.append(PolicyNote("story_span_source_required",
                                "source footage required in a touched story span", eid))
        event = replace(event, support_visual=SupportVisual.NONE)

    if event.motion is MotionPreset.STATIC_CLEAN and event.camera is not CameraMode.PRESERVE:
        notes.append(PolicyNote("static_clean_preserves_frame", "static_clean implies scale 1.0", eid))
        event = replace(event, camera=CameraMode.PRESERVE)

    if event.motion not in policy.allowed_motion or event.motion not in style.allowed_motion:
        notes.append(PolicyNote("motion_not_allowed_for_role",
                                f"{event.motion.value} not allowed in {event.role.value}", eid))
        event = replace(event, motion=policy.fallback_motion)
    if event.motion in STRONG_MOTIONS and not policy.strong_motion_allowed:
        notes.append(PolicyNote("strong_motion_suppressed", f"{event.role.value} suppresses strong motion", eid))
        event = replace(event, motion=policy.fallback_motion)

    if event.camera not in policy.allowed_camera:
        replacement = next((c for c in policy.preferred_camera if c in policy.allowed_camera), CameraMode.PRESERVE)
        notes.append(PolicyNote("camera_not_allowed_for_role",
                                f"{event.camera.value} -> {replacement.value} in {event.role.value}", eid))
        event = replace(event, camera=replacement)

    # Evidence requirements for subject-dependent framings.
    if event.camera is CameraMode.REACTION_CLOSE and not _has_reaction_evidence(context, event.start, event.end):
        notes.append(PolicyNote("reaction_framing_without_evidence",
                                "reaction_close requires reaction evidence", eid))
        event = replace(event, camera=CameraMode.PRESERVE)
    if event.camera is CameraMode.DUAL_SUBJECT and len(context.subject_tracks) < 2:
        notes.append(PolicyNote("dual_subject_without_two_subjects", "needs two subject tracks", eid))
        event = replace(event, camera=CameraMode.PRESERVE)
    if event.camera is CameraMode.OBJECT_FOCUS and not any(t.kind == "object" for t in context.subject_tracks):
        notes.append(PolicyNote("object_focus_without_object_track", "needs an object track", eid))
        event = replace(event, camera=CameraMode.PRESERVE)
    if event.motion is MotionPreset.SNAP_REFRAME and not context.subject_tracks:
        notes.append(PolicyNote("snap_reframe_without_subjects", "no stable anchors to reframe between", eid))
        event = replace(event, motion=policy.fallback_motion if policy.fallback_motion is not MotionPreset.SNAP_REFRAME
                        else MotionPreset.STATIC_CLEAN)
    if event.motion is MotionPreset.STATIC_CLEAN and event.camera is not CameraMode.PRESERVE:
        event = replace(event, camera=CameraMode.PRESERVE)

    lo, hi = policy.min_intensity, policy.max_intensity
    if not (lo <= event.intensity <= hi):
        clamped = max(lo, min(hi, event.intensity))
        notes.append(PolicyNote("intensity_role_clamp", f"{event.intensity:.3f} -> {clamped:.3f}", eid))
        event = replace(event, intensity=clamped)

    # Targets: only evidence that exists.
    target = event.target
    if target.type is TargetType.SUBJECT and target.id not in context.subject_ids():
        notes.append(PolicyNote("unknown_subject_center_safe", f"subject {target.id!r} unknown", eid))
        target = EditTarget(TargetType.CENTER_SAFE, None)
    elif target.type is TargetType.ACTIVE_SPEAKER and target.id is not None and target.id not in context.speaker_ids():
        notes.append(PolicyNote("unknown_speaker_center_safe", f"speaker {target.id!r} unknown", eid))
        target = EditTarget(TargetType.CENTER_SAFE, None)
    elif target.type is not TargetType.CENTER_SAFE and not context.subject_tracks:
        notes.append(PolicyNote("no_subject_evidence_center_safe", "no subject tracks available", eid))
        target = EditTarget(TargetType.CENTER_SAFE, None)
    elif target.type is TargetType.CENTER_SAFE and target.id is not None:
        target = EditTarget(TargetType.CENTER_SAFE, None)
    if target != event.target:
        event = replace(event, target=target)

    # Payoff/must-keep visibility beats decoration.
    if event.support_visual is not SupportVisual.NONE:
        if event.support_visual not in policy.support_visual_allowed:
            notes.append(PolicyNote("support_visual_not_allowed_for_role", event.role.value, eid))
            event = replace(event, support_visual=SupportVisual.NONE)
        elif style.support_visual.forbid_over_protected and context.overlaps_protected(event.start, event.end):
            notes.append(PolicyNote("payoff_source_visibility_protected",
                                    "support visual may not cover payoff/must-keep footage", eid))
            event = replace(event, support_visual=SupportVisual.NONE)
    if event.sfx is not SfxCue.NONE and (not policy.sfx_allowed or event.sfx not in style.sfx.allowed):
        notes.append(PolicyNote("sfx_not_allowed_for_role", event.role.value, eid))
        event = replace(event, sfx=SfxCue.NONE)
    return event, notes


def center_safe_zoom_cap(context: EditContext, style: StylePack, start: float, end: float) -> float:
    """Max zoom when the target is CENTER_SAFE (action location unknown)."""
    cap = style.zoom.max_zoom
    if context.overlaps_protected(start, end):
        cap = min(cap, style.zoom.protected_center_safe_max)
    layout = str(context.layout_content_type or "").casefold()
    if layout in OVERLAY_LAYOUT_CONTENT_TYPES:
        # Gameplay layouts usually carry an edge facecam overlay; a centered
        # crop could cut it. Stay conservative without evidence.
        cap = min(cap, style.zoom.overlay_layout_center_safe_max)
    return cap


# ============================================================
# MULTI-EVENT RULES
# ============================================================

def _subtract(start: float, end: float, taken: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    pieces = [(start, end)]
    for t_start, t_end in sorted(taken):
        next_pieces: list[tuple[float, float]] = []
        for p_start, p_end in pieces:
            if t_end <= p_start or t_start >= p_end:
                next_pieces.append((p_start, p_end))
                continue
            if t_start > p_start:
                next_pieces.append((p_start, t_start))
            if t_end < p_end:
                next_pieces.append((t_end, p_end))
        pieces = next_pieces
    return pieces


def _has_other_channels(event: EditEvent) -> bool:
    return bool(event.sfx is not SfxCue.NONE or event.support_visual is not SupportVisual.NONE
                or event.emphasis_word_ids or event.caption_style is not CaptionStyle.DEFAULT)


def _deactivate_camera(event: EditEvent) -> EditEvent:
    return replace(event, motion=MotionPreset.STATIC_CLEAN, camera=CameraMode.PRESERVE)


def resolve_camera_overlaps(events: Sequence[EditEvent], style: StylePack) -> tuple[list[EditEvent], list[PolicyNote], set[str]]:
    """One camera owner per frame. Priority winner keeps its window; losers are
    trimmed to their largest free piece or lose the camera channel.
    Returns (events, notes, rejected_ids)."""
    notes: list[PolicyNote] = []
    rejected: set[str] = set()
    taken: list[tuple[float, float]] = []
    decided: dict[str, EditEvent] = {}
    for event in sorted(events, key=priority_key, reverse=True):
        if not event.is_camera_active:
            decided[event.event_id] = event
            continue
        pieces = _subtract(event.start, event.end, taken)
        if len(pieces) == 1 and pieces[0] == (event.start, event.end):
            decided[event.event_id] = event
            taken.append((event.start, event.end))
            continue
        best = max(pieces, key=lambda p: (p[1] - p[0], -p[0]), default=None)
        if best is not None and best[1] - best[0] >= style.durations.min_event_s:
            trimmed = replace(event, start=best[0], end=best[1])
            notes.append(PolicyNote("camera_overlap_trimmed",
                                    f"trimmed to {best[0]:.3f}-{best[1]:.3f} (higher-priority camera owner)",
                                    event.event_id))
            decided[event.event_id] = trimmed
            taken.append(best)
        elif _has_other_channels(event):
            notes.append(PolicyNote("camera_overlap_channel_dropped", "camera channel lost to higher priority",
                                    event.event_id))
            decided[event.event_id] = _deactivate_camera(event)
        else:
            notes.append(PolicyNote("camera_overlap_rejected", "fully covered by higher-priority camera event",
                                    event.event_id))
            rejected.add(event.event_id)
    ordered = [decided[e.event_id] for e in events if e.event_id in decided]
    return ordered, notes, rejected


def _downgrade_motion(event: EditEvent, style: StylePack) -> EditEvent:
    policy = style.role(event.role)
    if event.motion in STRONG_MOTIONS and MotionPreset.SLOW_PUSH in policy.allowed_motion:
        return replace(event, motion=MotionPreset.SLOW_PUSH, intensity=min(event.intensity, 0.6))
    return _deactivate_camera(event)


def enforce_density(
    events: Sequence[EditEvent],
    duration_s: float,
    style: StylePack,
    budget: dict[str, int],
) -> tuple[list[EditEvent], list[PolicyNote]]:
    """Anti-over-editing: budgets, sliding-window density, gaps, repetition."""
    notes: list[PolicyNote] = []
    current = {e.event_id: e for e in events}
    d = style.density
    duration = max(1e-6, float(duration_s))

    # 1. Strong motions (priority order).
    accepted: list[EditEvent] = []
    strong_seconds = 0.0
    snaps = 0
    for event in sorted((e for e in events if e.is_camera_active and e.is_strong), key=priority_key, reverse=True):
        reason = None
        in_window = sum(1 for a in accepted if abs(a.start - event.start) < 10.0)
        if len(accepted) >= budget["max_strong_motion_events"]:
            reason = "strong_budget_exhausted"
        elif in_window + 1 > d.strong_events_per_10s:
            reason = "strong_density_per_10s"
        elif any(max(a.start - event.end, event.start - a.end) < d.min_gap_between_strong_s for a in accepted):
            reason = "strong_min_gap"
        elif (strong_seconds + event.duration) / duration > d.max_strong_motion_seconds_ratio:
            reason = "strong_seconds_ratio"
        elif event.motion is MotionPreset.SNAP_REFRAME and snaps >= budget["max_snap_reframes"]:
            reason = "snap_reframe_budget"
        if reason is None:
            accepted.append(event)
            strong_seconds += event.duration
            snaps += event.motion is MotionPreset.SNAP_REFRAME
        else:
            current[event.event_id] = _downgrade_motion(event, style)
            notes.append(PolicyNote("density_downgrade", reason, event.event_id))

    # 2. Camera events per budget / per 10 s.
    camera = sorted((current[e.event_id] for e in events if current[e.event_id].is_camera_active),
                    key=priority_key, reverse=True)
    kept: list[EditEvent] = []
    for event in camera:
        window = sum(1 for k in kept if abs(k.start - event.start) < 10.0)
        if len(kept) >= budget["max_camera_events"] or window + 1 > d.camera_changes_per_10s:
            current[event.event_id] = _deactivate_camera(event)
            notes.append(PolicyNote("camera_change_density", "too many camera changes", event.event_id))
        else:
            kept.append(event)

    # 3. Consecutive same preset limit (chronological).
    run_motion: MotionPreset | None = None
    run = 0
    for event in sorted((current[e.event_id] for e in events), key=lambda e: (e.start, e.event_id)):
        if not event.is_camera_active:
            continue
        run = run + 1 if event.motion is run_motion else 1
        run_motion = event.motion
        if run > d.consecutive_same_preset_limit:
            downgraded = _downgrade_motion(event, style)
            if downgraded.motion is event.motion:
                downgraded = _deactivate_camera(event)
            current[event.event_id] = downgraded
            notes.append(PolicyNote("repeated_preset_limit", f"{event.motion.value} x{run}", event.event_id))
            run_motion = downgraded.motion if downgraded.is_camera_active else None
            run = 1 if downgraded.is_camera_active else 0

    # 4. SFX: budget + min gap; 5. support visuals: budget.
    sfx_kept: list[EditEvent] = []
    for event in sorted((current[e.event_id] for e in events if current[e.event_id].sfx is not SfxCue.NONE),
                        key=priority_key, reverse=True):
        if (len(sfx_kept) >= budget["max_sfx_events"]
                or any(abs(k.start - event.start) < style.sfx.min_gap_s for k in sfx_kept)):
            current[event.event_id] = replace(event, sfx=SfxCue.NONE)
            notes.append(PolicyNote("sfx_density", "sfx budget/min gap", event.event_id))
        else:
            sfx_kept.append(event)
    visuals = 0
    for event in sorted((current[e.event_id] for e in events
                         if current[e.event_id].support_visual is not SupportVisual.NONE),
                        key=priority_key, reverse=True):
        if visuals >= budget["max_support_visual_events"]:
            current[event.event_id] = replace(event, support_visual=SupportVisual.NONE)
            notes.append(PolicyNote("support_visual_budget", "support visual budget exhausted", event.event_id))
        else:
            visuals += 1
    return [current[e.event_id] for e in events], notes


def role_counts(events: Iterable[EditEvent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        counts[event.role.value] = counts.get(event.role.value, 0) + 1
    return counts
