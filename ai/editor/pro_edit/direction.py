"""Camera V6 direction: WHAT MUST THE VIEWER SEE RIGHT NOW?

The planner (model, rules or cache) chooses editorial INTENT per story span.
This deterministic pass decides WHO/WHAT that intent may frame, from evidence
MIMIR already has - it never invents coordinates and never renders:

    story span (role, action visibility, required regions/subjects)
  + visible camera subjects (persistent face tracks, speaker link confidence)
  + active speaker (caption words) and conversational exchanges
  + layout (gameplay / screen content owns the frame, face-cam overlays)
  -> a small shot vocabulary:
     HOLD, WIDE_CONTEXT, TWO_SHOT, ACTIVE_SPEAKER_MEDIUM, ACTIVE_SPEAKER_PUNCH,
     REACTION, ACTION_REGION, GAMEPLAY_PRIORITY, SCREEN_PRIORITY
  -> the existing EditEvent enums (camera mode, motion, target subject)

The resolver/framing solver then computes crop, zoom, centre and timing and
verifies story geometry on every frame; if requirements conflict it widens.
Rules are general (no person, layout or clip specifics):

* the story outranks the voice: payoff/reaction moments with several visible
  participants keep them all (two-shot) instead of a single-speaker crop, and
  a group of three or more visible people stays in a wide shot;
* an action span keeps its required regions / subjects; when the action
  location is unknown the frame stays >= 85 % visible (two visible people ->
  two-shot, otherwise wide);
* gameplay / screen content is never cropped away for a face: only the story
  peaks (hook / payoff / reaction) get a gentle whole-frame push;
* a speaker close-up needs a speaker-face link; weak links get medium framing,
  no link -> context framing (conservative fallback);
* long static regions are either given one restrained, evidence-backed
  emphasis or recorded as a HOLD with a concrete reason.
"""
from __future__ import annotations

import dataclasses
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from ai.editor.pro_edit.context import EditContext
from ai.editor.pro_edit.schema import (
    CameraMode,
    EditEvent,
    EditPlan,
    EditTarget,
    MotionPreset,
    ReasonCode,
    StoryRole,
    TargetType,
)
from ai.editor.pro_edit.story import StorySpan
from ai.editor.pro_edit.style import StylePack
from ai.editor.pro_edit.subjects import SubjectTrack, track_reliability

DIRECTION_VERSION = 2
LONG_STATIC_S = 6.0                   # an unexplained static stretch longer than this needs a decision
MIN_ADDED_TURN_S = 3.0                # added emphasis only on a sustained single-speaker turn
ADDED_EVENT_MAX_S = 4.0
MIN_ADDED_SPACING_S = 8.0             # never more than one added emphasis per 8 s (no rhythmic zooms)
CAMERA_SUBJECT_MIN_HEIGHT = 0.07      # face height / frame height: smaller faces are background
CAMERA_SUBJECT_MIN_SECONDS = 3.0
CAMERA_SUBJECT_MIN_SHARE = 0.25       # ... or this share of the visible main
CAMERA_SUBJECT_MIN_CONFIDENCE = 0.5
CAMERA_SUBJECT_EDGE = 0.03
# A detection that never moves in the frame and never shows mouth motion is a static
# pattern (poster, photo, logo, overlay art the detector mistakes for a face): real
# faces sway and animate, even a still face-cam streamer (measured activity >= 0.005).
STATIC_PATTERN_MAX_STD = 0.004
STATIC_PATTERN_MAX_ACTIVITY = 0.003
WINDOW_COVERAGE = 0.5                 # a subject is "visible" in a window when tracked for half of it
STRONG_LINK = 0.7                     # speaker-face link needed for a speaker close-up
DOMINANT_SHARE = 0.8                  # one speaker owns the window
FACECAM_MAX_HEIGHT = 0.12
FACECAM_CORNER = 0.32
SCREEN_CONTENT_TYPES = frozenset({"gameplay", "streamer_gameplay"})
TWO_SHOT_ROLES = frozenset({StoryRole.ESCALATION, StoryRole.PAYOFF, StoryRole.REACTION})
SCREEN_PEAK_ROLES = frozenset({StoryRole.HOOK, StoryRole.PAYOFF, StoryRole.REACTION})
GROUP_MIN_SUBJECTS = 3


class ShotIntent(str, Enum):
    HOLD = "HOLD"
    WIDE_CONTEXT = "WIDE_CONTEXT"
    TWO_SHOT = "TWO_SHOT"
    ACTIVE_SPEAKER_MEDIUM = "ACTIVE_SPEAKER_MEDIUM"
    ACTIVE_SPEAKER_PUNCH = "ACTIVE_SPEAKER_PUNCH"
    REACTION = "REACTION"
    ACTION_REGION = "ACTION_REGION"
    GAMEPLAY_PRIORITY = "GAMEPLAY_PRIORITY"
    SCREEN_PRIORITY = "SCREEN_PRIORITY"


@dataclass(frozen=True)
class CameraSubject:
    track_id: str
    speaker_id: str | None
    link_confidence: float
    t0: float
    t1: float
    cx: float
    cy: float
    height: float

    def to_dict(self) -> dict[str, Any]:
        return {"track_id": self.track_id, "speaker_id": self.speaker_id,
                "link_confidence": round(self.link_confidence, 3), "t": [round(self.t0, 2), round(self.t1, 2)],
                "center": [round(self.cx, 3), round(self.cy, 3)], "height": round(self.height, 3)}


@dataclass
class DirectionReport:
    version: int = DIRECTION_VERSION
    layout_mode: str = "subjects"
    layout_reason: str = ""
    subjects: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    additions: list[dict[str, Any]] = field(default_factory=list)
    holds: list[dict[str, Any]] = field(default_factory=list)
    # Directed camera events the validator's restraint policy switched off (with its codes).
    restraint: list[dict[str, Any]] = field(default_factory=list)
    # Persistent face tracks that may never carry a shot (with the reason).
    excluded_subjects: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "layout_mode": self.layout_mode, "layout_reason": self.layout_reason,
                "subjects": list(self.subjects), "excluded_subjects": list(self.excluded_subjects),
                "decisions": list(self.decisions), "additions": list(self.additions),
                "restraint": list(self.restraint), "holds": list(self.holds)}

    def summary(self) -> dict[str, Any]:
        intents: dict[str, int] = {}
        for row in self.decisions + self.additions:
            intents[row["intent"]] = intents.get(row["intent"], 0) + 1
        return {"layout": self.layout_mode, "subjects": len(self.subjects),
                "linked_subjects": sum(1 for s in self.subjects if s.get("speaker_id")),
                "excluded_subjects": len(self.excluded_subjects),
                "intents": intents, "added": len(self.additions), "holds": len(self.holds)}


# ============================================================
# EVIDENCE
# ============================================================

def static_pattern(samples: Sequence[Any]) -> bool:
    """True when a face track is frozen in the frame AND shows no mouth motion (not a person)."""
    activity = [s.activity for s in samples if s.activity is not None]
    if len(samples) < 3 or not activity:
        return False            # no motion evidence recorded: never guess
    return (statistics.pstdev(s.cx for s in samples) < STATIC_PATTERN_MAX_STD
            and statistics.pstdev(s.cy for s in samples) < STATIC_PATTERN_MAX_STD
            and statistics.median(activity) < STATIC_PATTERN_MAX_ACTIVITY)


def camera_subjects(context: EditContext, style: StylePack,
                    excluded: list[dict[str, Any]] | None = None) -> list[CameraSubject]:
    """Face tracks that can carry a shot: persistent, confident, not background-sized,
    not a static pattern. ``excluded`` collects persistent tracks rejected with a reason."""
    visible_len = max(1e-6, context.clip.duration_s - context.clip.visible_start_s)
    rows: list[CameraSubject] = []
    for track in context.subject_tracks:
        if track.kind != "face":
            continue
        reliable = [s for s in track.samples if s.confidence >= style.camera.subject_min_confidence]
        if len(reliable) < 3:
            continue
        t0, t1 = reliable[0].t, reliable[-1].t
        span = t1 - t0
        if span < CAMERA_SUBJECT_MIN_SECONDS and span < CAMERA_SUBJECT_MIN_SHARE * visible_len:
            continue
        cx = statistics.median(s.cx for s in reliable)
        cy = statistics.median(s.cy for s in reliable)
        height = statistics.median(s.height for s in reliable)
        if height < CAMERA_SUBJECT_MIN_HEIGHT or statistics.median(s.confidence for s in reliable) < \
                CAMERA_SUBJECT_MIN_CONFIDENCE:
            continue
        if not CAMERA_SUBJECT_EDGE <= cx <= 1.0 - CAMERA_SUBJECT_EDGE:
            continue
        if not track.speaker_id and static_pattern(reliable):
            if excluded is not None:
                excluded.append({"track_id": track.subject_id, "t": [round(t0, 2), round(t1, 2)],
                                 "center": [round(cx, 3), round(cy, 3)],
                                 "reason": "static pattern: frozen in the frame with no mouth motion"})
            continue
        linked = track.speaker_id if track.speaker_confidence >= style.camera.speaker_link_min_confidence else None
        rows.append(CameraSubject(track.subject_id, linked, track.speaker_confidence if linked else 0.0, t0, t1, cx,
                                  cy, height))
    return sorted(rows, key=lambda s: (s.t0, s.track_id))


def screen_priority(context: EditContext) -> tuple[str, str]:
    """(layout_mode, reason): gameplay/screen content owns the frame."""
    content = str(context.layout_content_type or "").casefold()
    if content in SCREEN_CONTENT_TYPES:
        return ShotIntent.GAMEPLAY_PRIORITY.value, f"visual report layout content_type={content}"
    visible_len = max(1e-6, context.clip.duration_s - context.clip.visible_start_s)
    for track in context.subject_tracks:
        if track.kind != "face":
            continue
        reliable = [s for s in track.samples if s.confidence >= 0.5]
        if not reliable or (reliable[-1].t - reliable[0].t) < 0.5 * visible_len:
            continue
        height = statistics.median(s.height for s in reliable)
        cx = statistics.median(s.cx for s in reliable)
        cy = statistics.median(s.cy for s in reliable)
        corner = (cx < FACECAM_CORNER or cx > 1 - FACECAM_CORNER) and (cy < FACECAM_CORNER or cy > 1 - FACECAM_CORNER)
        if height < FACECAM_MAX_HEIGHT and corner and len(context.subject_tracks) == 1:
            return ShotIntent.SCREEN_PRIORITY.value, "persistent small corner face (face-cam overlay over screen content)"
    return "subjects", ""


def visible_subjects(subjects: Sequence[CameraSubject], tracks: Sequence[SubjectTrack], start: float, end: float,
                     style: StylePack) -> list[CameraSubject]:
    by_id = {t.subject_id: t for t in tracks}
    found = []
    for subject in subjects:
        track = by_id.get(subject.track_id)
        if track is None:
            continue
        conf, coverage = track_reliability(track, start, end, style.camera)
        if conf >= style.camera.subject_min_confidence and coverage >= WINDOW_COVERAGE:
            found.append(subject)
    return found


def dominant_speaker(context: EditContext, start: float, end: float) -> tuple[str | None, float]:
    """(speaker, share of spoken time) inside [start, end]; None when nobody speaks."""
    totals: dict[str, float] = {}
    for word in context.words:
        if not word.speaker_id:
            continue
        overlap = min(end, word.end) - max(start, word.start)
        if overlap > 0:
            totals[word.speaker_id] = totals.get(word.speaker_id, 0.0) + overlap
    if not totals:
        return None, 0.0
    speaker, value = sorted(totals.items(), key=lambda row: (-row[1], row[0]))[0]
    return speaker, value / max(1e-6, sum(totals.values()))


def speech_turns(context: EditContext, gap: float = 0.6) -> list[tuple[float, float, str]]:
    turns: list[list[Any]] = []
    for word in sorted(context.words, key=lambda w: w.start):
        if not word.speaker_id:
            continue
        if turns and turns[-1][2] == word.speaker_id and word.start - turns[-1][1] <= gap:
            turns[-1][1] = max(turns[-1][1], word.end)
        else:
            turns.append([word.start, word.end, word.speaker_id])
    return [(float(a), float(b), str(s)) for a, b, s in turns]


def _crosses_cut(context: EditContext, start: float, end: float) -> bool:
    return any(start + 0.05 < cut < end - 0.05 for cut in context.scene_changes)


# ============================================================
# PER-EVENT DIRECTION
# ============================================================

def _with(event: EditEvent, **changes: Any) -> EditEvent:
    return dataclasses.replace(event, **changes)


def _decision(event: EditEvent, intent: ShotIntent, reason: str, *, before: EditEvent,
              subjects: Sequence[CameraSubject] = (), speaker: str | None = None) -> dict[str, Any]:
    return {"event_id": event.event_id, "window": [round(event.start, 3), round(event.end, 3)],
            "intent": intent.value, "reason": reason,
            "planned": {"camera": before.camera.value, "motion": before.motion.value, "target": before.target.to_dict()},
            "directed": {"camera": event.camera.value, "motion": event.motion.value, "target": event.target.to_dict()},
            "subjects": [s.track_id for s in subjects], "speaker": speaker}


def direct_event(event: EditEvent, context: EditContext, subjects: Sequence[CameraSubject], style: StylePack,
                 layout_mode: str) -> tuple[EditEvent, dict[str, Any]]:
    before = event
    span = context.span(event.story_span_id) or context.dominant_span(event.start, event.end)
    role = span.role if span is not None else event.role
    wide = _with(event, camera=CameraMode.PRESERVE, target=EditTarget(TargetType.CENTER_SAFE, None))
    gentle = MotionPreset.SLOW_PUSH if event.motion is not MotionPreset.STATIC_CLEAN else MotionPreset.STATIC_CLEAN

    if layout_mode != "subjects":
        peak = role in SCREEN_PEAK_ROLES
        directed = _with(wide, motion=gentle if peak else MotionPreset.STATIC_CLEAN)
        return directed, _decision(directed, ShotIntent(layout_mode),
                                   "screen content owns the frame; faces are never framed over it"
                                   + ("; gentle whole-frame push on a story peak" if peak and gentle is not
                                      MotionPreset.STATIC_CLEAN else "; held"), before=before)
    visible = visible_subjects(subjects, context.subject_tracks, event.start, event.end, style)
    speaker, share = dominant_speaker(context, event.start, event.end)
    if span is not None and (span.required_regions or span.required_subject_ids):
        directed = _with(wide, motion=gentle)
        return directed, _decision(directed, ShotIntent.ACTION_REGION,
                                   "action span: required story regions/subjects stay inside the crop "
                                   "(engine widens on conflict)", before=before, subjects=visible, speaker=speaker)
    if span is not None and span.min_visible_fraction > 0:
        if len(visible) >= 2 and CameraMode.DUAL_SUBJECT in style.role(role).allowed_camera:
            directed = _with(event, camera=CameraMode.DUAL_SUBJECT, target=EditTarget(TargetType.CENTER_SAFE, None),
                             motion=gentle)
            return directed, _decision(directed, ShotIntent.TWO_SHOT,
                                       "action span with unknown location: every visible participant stays in "
                                       f"frame (>= {span.min_visible_fraction:.0%} visible)", before=before,
                                       subjects=visible, speaker=speaker)
        directed = _with(wide, motion=gentle)
        return directed, _decision(directed, ShotIntent.ACTION_REGION,
                                   f"action span with unknown location: >= {span.min_visible_fraction:.0%} of the "
                                   "frame stays visible", before=before, subjects=visible, speaker=speaker)
    if _crosses_cut(context, event.start, event.end) and event.target.type is not TargetType.CENTER_SAFE:
        directed = _with(wide, motion=gentle)
        return directed, _decision(directed, ShotIntent.WIDE_CONTEXT, "scene cut inside the window: no subject lock",
                                   before=before, subjects=visible, speaker=speaker)
    if len(visible) >= GROUP_MIN_SUBJECTS:
        directed = _with(wide, motion=MotionPreset.STATIC_CLEAN)
        return directed, _decision(directed, ShotIntent.WIDE_CONTEXT,
                                   f"group of {len(visible)} visible participants: wide context keeps everyone",
                                   before=before, subjects=visible, speaker=speaker)
    unconfirmed_close = bool(visible) and event.camera in (CameraMode.SPEAKER_CLOSE, CameraMode.REACTION_CLOSE) \
        and (visible[0].speaker_id is None or (event.camera is CameraMode.SPEAKER_CLOSE
                                               and visible[0].link_confidence < STRONG_LINK))
    if event.target.type is TargetType.SUBJECT and event.target.id in {s.track_id for s in visible} \
            and len(visible) < 2 and not unconfirmed_close:
        return event, _decision(event, ShotIntent.ACTIVE_SPEAKER_MEDIUM if event.camera is CameraMode.SPEAKER_MEDIUM
                                else ShotIntent.ACTIVE_SPEAKER_PUNCH, "planner subject is visible and unique",
                                before=before, subjects=visible, speaker=speaker)
    if len(visible) >= 2:
        linked = next((s for s in visible if s.speaker_id and s.speaker_id == speaker), None)
        exchange = share < DOMINANT_SHARE
        if role in TWO_SHOT_ROLES and (exchange or role is not StoryRole.ESCALATION
                                       or event.camera is CameraMode.DUAL_SUBJECT or linked is None):
            if CameraMode.DUAL_SUBJECT in style.role(role).allowed_camera:
                directed = _with(event, camera=CameraMode.DUAL_SUBJECT, target=EditTarget(TargetType.CENTER_SAFE,
                                                                                          None),
                                 motion=MotionPreset.SLOW_PUSH if event.motion is not MotionPreset.STATIC_CLEAN
                                 else MotionPreset.STATIC_CLEAN)
                return directed, _decision(directed, ShotIntent.TWO_SHOT,
                                           f"{role.value}: several participants carry the moment "
                                           f"({'exchange' if exchange else 'story before voice'})",
                                           before=before, subjects=visible, speaker=speaker)
        if linked is not None and not exchange and role not in (StoryRole.PAYOFF, StoryRole.REACTION):
            camera = event.camera if event.camera in (CameraMode.SPEAKER_MEDIUM, CameraMode.SPEAKER_CLOSE) \
                else CameraMode.SPEAKER_MEDIUM
            if linked.link_confidence < STRONG_LINK or camera not in style.role(role).allowed_camera:
                camera = CameraMode.SPEAKER_MEDIUM if CameraMode.SPEAKER_MEDIUM in style.role(role).allowed_camera \
                    else camera
            motion = event.motion if event.motion is not MotionPreset.SPEAKER_FOCUS else MotionPreset.SLOW_PUSH
            if camera is CameraMode.SPEAKER_MEDIUM and motion in (MotionPreset.PUNCH_IN, MotionPreset.PUNCH_IN_FAST):
                motion = MotionPreset.SLOW_PUSH
            directed = _with(event, camera=camera, motion=motion, target=EditTarget(TargetType.SUBJECT,
                                                                                     linked.track_id))
            intent = ShotIntent.ACTIVE_SPEAKER_PUNCH if camera is CameraMode.SPEAKER_CLOSE \
                else ShotIntent.ACTIVE_SPEAKER_MEDIUM
            return directed, _decision(directed, intent,
                                       f"speaker {speaker} owns {share:.0%} of the window; face link "
                                       f"{linked.link_confidence:.2f}", before=before, subjects=visible,
                                       speaker=speaker)
        directed = _with(wide, motion=MotionPreset.STATIC_CLEAN)
        reason = ("conversation exchange: both participants stay in frame" if exchange else
                  "active speaker has no confirmed visible face" if linked is None else
                  "story moment keeps every participant in frame")
        return directed, _decision(directed, ShotIntent.WIDE_CONTEXT, reason, before=before, subjects=visible,
                                   speaker=speaker)
    if len(visible) == 1:
        subject = visible[0]
        is_speaker = subject.speaker_id is not None and subject.speaker_id == speaker
        listener = subject.speaker_id is not None and speaker is not None and subject.speaker_id != speaker
        if listener and role is StoryRole.REACTION and CameraMode.REACTION_CLOSE in style.role(role).allowed_camera:
            directed = _with(event, camera=CameraMode.REACTION_CLOSE, target=EditTarget(TargetType.SUBJECT,
                                                                                         subject.track_id))
            return directed, _decision(directed, ShotIntent.REACTION, "reaction of the visible listener",
                                       before=before, subjects=visible, speaker=speaker)
        camera = event.camera
        close = (CameraMode.SPEAKER_CLOSE, CameraMode.REACTION_CLOSE)
        # A close-up needs a face confirmed as a participant (speaker link): the detector
        # may miss the real subject (turned / partly hidden face), so an unlinked face gets
        # medium framing at most, a speaker close-up additionally needs a strong link.
        unconfirmed = subject.speaker_id is None or (camera is CameraMode.SPEAKER_CLOSE and not (
            is_speaker and subject.link_confidence >= STRONG_LINK))
        reacting = camera is CameraMode.REACTION_CLOSE
        if unconfirmed and camera in close:
            camera = CameraMode.SPEAKER_MEDIUM if CameraMode.SPEAKER_MEDIUM in style.role(role).allowed_camera \
                else CameraMode.PRESERVE
        if camera in (CameraMode.PRESERVE, CameraMode.DUAL_SUBJECT, CameraMode.OBJECT_FOCUS):
            camera = CameraMode.SPEAKER_MEDIUM if CameraMode.SPEAKER_MEDIUM in style.role(role).allowed_camera \
                else CameraMode.PRESERVE
        motion = event.motion
        if camera is CameraMode.SPEAKER_MEDIUM and motion in (MotionPreset.PUNCH_IN_FAST,):
            motion = MotionPreset.PUNCH_IN
        target = EditTarget(TargetType.SUBJECT, subject.track_id) if camera is not CameraMode.PRESERVE \
            else EditTarget(TargetType.CENTER_SAFE, None)
        directed = _with(event, camera=camera, motion=motion, target=target)
        intent = (ShotIntent.REACTION if camera is CameraMode.REACTION_CLOSE or (reacting and camera is not
                                                                                  CameraMode.PRESERVE) else
                  ShotIntent.ACTIVE_SPEAKER_PUNCH if camera is CameraMode.SPEAKER_CLOSE else
                  ShotIntent.ACTIVE_SPEAKER_MEDIUM if camera is not CameraMode.PRESERVE else ShotIntent.WIDE_CONTEXT)
        reason = ("reaction of the single visible participant" if camera is CameraMode.REACTION_CLOSE else
                  "reaction span: single visible face, participant link unconfirmed: medium framing at most"
                  if reacting else
                  "single visible subject is the linked speaker" if is_speaker else
                  "single visible subject (speaker link unconfirmed: medium framing at most)")
        return directed, _decision(directed, intent, reason, before=before, subjects=visible, speaker=speaker)
    if event.motion in (MotionPreset.PUNCH_IN, MotionPreset.PUNCH_IN_FAST, MotionPreset.SNAP_REFRAME) \
            or event.camera is not CameraMode.PRESERVE:
        directed = _with(wide, motion=MotionPreset.SLOW_PUSH if event.motion is not MotionPreset.STATIC_CLEAN
                         else MotionPreset.STATIC_CLEAN)
        return directed, _decision(directed, ShotIntent.WIDE_CONTEXT,
                                   "no visible camera subject: whole-frame emphasis only", before=before,
                                   speaker=speaker)
    return event, _decision(event, ShotIntent.WIDE_CONTEXT, "whole-frame intent kept", before=before,
                            speaker=speaker)


# ============================================================
# COVERAGE (no unexplained long static shots)
# ============================================================

def _static_gaps(events: Sequence[EditEvent], start: float, end: float) -> list[tuple[float, float]]:
    busy = sorted((e.start, e.end) for e in events if e.is_camera_active)
    gaps: list[tuple[float, float]] = []
    cursor = start
    for a, b in busy:
        if a > cursor:
            gaps.append((cursor, min(a, end)))
        cursor = max(cursor, b)
    if cursor < end:
        gaps.append((cursor, end))
    return [(a, b) for a, b in gaps if b - a > 1e-6]


def _turn_rejection(context: EditContext, subjects: Sequence[CameraSubject], style: StylePack, start: float,
                    end: float, speaker: str) -> tuple[str, CameraSubject | None]:
    """('', subject) when a sustained turn may get restrained emphasis, else (reason, None)."""
    spans = context.spans_overlapping(start, end)
    if not spans or any(not s.camera_allowed for s in spans):
        return "story span locks the camera", None
    if any(s.min_visible_fraction > 0 or s.required_regions or s.required_subject_ids for s in spans):
        return "action span: full frame keeps the event visible", None
    if _crosses_cut(context, start, end):
        return "scene cut inside the turn: no subject lock", None
    visible = visible_subjects(subjects, context.subject_tracks, start, end, style)
    if not visible:
        return "no persistent visible subject", None
    if len(visible) >= GROUP_MIN_SUBJECTS:
        return f"group of {len(visible)} visible participants: wide context keeps everyone", None
    linked = next((s for s in visible if s.speaker_id == speaker), None)
    if linked is None:
        return "active speaker has no confirmed visible face", None
    if linked.link_confidence < STRONG_LINK:
        return f"speaker-face link {linked.link_confidence:.2f} below {STRONG_LINK:.2f}: conservative framing", None
    owner = context.dominant_span(start, end)
    policy = style.role(owner.role) if owner is not None else None
    if policy is None or CameraMode.SPEAKER_MEDIUM not in policy.allowed_camera             or MotionPreset.SLOW_PUSH not in policy.allowed_motion:
        return "story role does not allow speaker framing", None
    return "", linked


def _region_reason(context: EditContext, subjects: Sequence[CameraSubject], style: StylePack, start: float,
                   end: float, layout_mode: str) -> str:
    """Concrete reason why [start, end] stays on the original framing."""
    if layout_mode != "subjects":
        return "screen content owns the frame"
    spans = context.spans_overlapping(start, end)
    if spans and not any(s.camera_allowed for s in spans):
        return "story span locks the camera"
    if any(s.min_visible_fraction > 0 or s.required_regions or s.required_subject_ids for s in spans):
        return "action span: full frame keeps the event visible"
    visible = visible_subjects(subjects, context.subject_tracks, start, end, style)
    if not visible:
        return "no persistent visible subject"
    if len(visible) >= GROUP_MIN_SUBJECTS:
        return f"group of {len(visible)} visible participants: wide context keeps everyone"
    speaker, share = dominant_speaker(context, start, end)
    if speaker is None:
        return "no speech: nothing to single out"
    if len(visible) >= 2 and share < DOMINANT_SHARE:
        return "conversation exchange: wide context keeps both participants"
    linked = next((s for s in visible if s.speaker_id == speaker), None)
    if linked is None:
        return "active speaker has no confirmed visible face"
    if linked.link_confidence < STRONG_LINK:
        return f"speaker-face link {linked.link_confidence:.2f} below {STRONG_LINK:.2f}: conservative framing"
    return "restraint: no sustained single-speaker turn (>= 3 s) or an emphasis less than 8 s earlier"


def _two_shot_window(context: EditContext, subjects: Sequence[CameraSubject], style: StylePack, gap_start: float,
                     gap_end: float, not_before: float) -> tuple[float, float, StorySpan, list[CameraSubject]] | None:
    """A long action stretch (unknown action location, no required regions) where
    exactly two persistent participants are visible: a restrained two-shot keeps
    both of them and >= the span's minimum visible fraction of the frame."""
    a = max(gap_start + 0.2, not_before)
    b = min(gap_end - 0.2, a + ADDED_EVENT_MAX_S)
    if b - a < MIN_ADDED_TURN_S:
        return None
    spans = context.spans_overlapping(a, b)
    if not spans or any(not s.camera_allowed or s.required_regions or s.required_subject_ids for s in spans):
        return None
    if not any(s.min_visible_fraction > 0 for s in spans) or _crosses_cut(context, a, b):
        return None
    owner = context.dominant_span(a, b)
    if owner is None:
        return None
    policy = style.role(owner.role)
    if CameraMode.DUAL_SUBJECT not in policy.allowed_camera or MotionPreset.SLOW_PUSH not in policy.allowed_motion:
        return None
    visible = visible_subjects(subjects, context.subject_tracks, a, b, style)
    if len(visible) != 2 or not any(s.speaker_id for s in visible):
        return None
    speaker, _share = dominant_speaker(context, a, b)
    if speaker is None:
        return None
    return a, min(b, owner.end), owner, visible


def add_coverage(events: Sequence[EditEvent], context: EditContext, subjects: Sequence[CameraSubject],
                 style: StylePack, layout_mode: str) -> tuple[list[EditEvent], list[dict[str, Any]]]:
    """Restrained emphasis for long static regions with strong evidence.

    At most one added event per static gap: a sustained single-speaker turn
    with a confident face link gets a speaker medium; otherwise a long action
    stretch with exactly two visible participants gets a restrained two-shot.
    Never inside locked spans / required story geometry, never across a scene
    cut and never less than MIN_ADDED_SPACING_S after the previous added
    emphasis (no rhythmic zooms).
    """
    visible_start = context.clip.visible_start_s
    added: list[EditEvent] = []
    additions: list[dict[str, Any]] = []
    if layout_mode != "subjects":
        return added, additions
    turns = speech_turns(context)
    last_added = -1e9
    for gap_start, gap_end in _static_gaps(events, visible_start, context.clip.duration_s):
        if gap_end - gap_start < LONG_STATIC_S:
            continue
        for turn_start, turn_end, speaker in turns:
            a, b = max(turn_start, gap_start + 0.2), min(turn_end, gap_end - 0.2)
            if b - a < MIN_ADDED_TURN_S or a - last_added < MIN_ADDED_SPACING_S:
                continue
            _reason, linked = _turn_rejection(context, subjects, style, a, b, speaker)
            owner = context.dominant_span(a, b)
            if linked is None or owner is None:
                continue
            end = min(b, a + ADDED_EVENT_MAX_S, owner.end)
            if end - a < MIN_ADDED_TURN_S:
                continue
            chosen = EditEvent(
                event_id=f"v6_speaker_{len(added) + 1:02d}", start=round(a, 3), end=round(end, 3),
                role=owner.role, camera=CameraMode.SPEAKER_MEDIUM, motion=MotionPreset.SLOW_PUSH,
                target=EditTarget(TargetType.SUBJECT, linked.track_id), intensity=0.4, confidence=0.8,
                reason_code=ReasonCode.SPEAKER_SHIFT, story_span_id=owner.span_id,
                note="v6 direction: sustained single-speaker turn with a confirmed face link")
            additions.append({"event_id": chosen.event_id, "window": [chosen.start, chosen.end],
                              "intent": ShotIntent.ACTIVE_SPEAKER_MEDIUM.value, "speaker": speaker,
                              "subject": linked.track_id, "link_confidence": round(linked.link_confidence, 3),
                              "reason": "long static region with a confident single-speaker turn"})
            added.append(chosen)
            last_added = chosen.start
            break
        else:
            window = _two_shot_window(context, subjects, style, gap_start, gap_end, last_added + MIN_ADDED_SPACING_S)
            if window is None:
                continue
            a, end, owner, visible = window
            if end - a < MIN_ADDED_TURN_S:
                continue
            chosen = EditEvent(
                event_id=f"v6_two_shot_{len(added) + 1:02d}", start=round(a, 3), end=round(end, 3),
                role=owner.role, camera=CameraMode.DUAL_SUBJECT, motion=MotionPreset.SLOW_PUSH,
                target=EditTarget(TargetType.CENTER_SAFE, None), intensity=0.35, confidence=0.8,
                reason_code=ReasonCode.VISUAL_CLARITY, story_span_id=owner.span_id,
                note="v6 direction: restrained two-shot in a long two-participant action stretch")
            additions.append({"event_id": chosen.event_id, "window": [chosen.start, chosen.end],
                              "intent": ShotIntent.TWO_SHOT.value, "subjects": [s.track_id for s in visible],
                              "reason": "long action stretch with two visible participants: two-shot keeps both "
                                        "(>= 85% of the frame stays visible)"})
            added.append(chosen)
            last_added = chosen.start
    return added, additions


def _hold_segments(context: EditContext, subjects: Sequence[CameraSubject], style: StylePack, start: float,
                   end: float, layout_mode: str) -> list[tuple[float, float, str]]:
    """Per story segment reasons inside one static stretch (identical neighbours merged)."""
    cuts = sorted({start, end, *(t for span in context.spans for t in (span.start, span.end) if start < t < end)})
    rows: list[tuple[float, float, str]] = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a < 0.5:
            continue
        why = _region_reason(context, subjects, style, a, b, layout_mode)
        if rows and rows[-1][2] == why and abs(rows[-1][1] - a) <= 0.5:
            rows[-1] = (rows[-1][0], b, why)
        else:
            rows.append((a, b, why))
    return rows


def explain_holds(active_ranges_s: Sequence[tuple[float, float]], context: EditContext, style: StylePack,
                  report: "DirectionReport") -> list[dict[str, Any]]:
    """HOLD records for every long stretch the RENDERED camera path leaves untouched.

    Computed from the resolved camera path (what is actually rendered), so a
    long static wide shot always carries a concrete reason.
    """
    subjects = [CameraSubject(row["track_id"], row["speaker_id"], row["link_confidence"], row["t"][0], row["t"][1],
                              row["center"][0], row["center"][1], row["height"]) for row in report.subjects]
    holds: list[dict[str, Any]] = []
    cursor = context.clip.visible_start_s
    for a, b in sorted(active_ranges_s) + [(context.clip.duration_s, context.clip.duration_s)]:
        if a - cursor >= LONG_STATIC_S:
            restrained = [row for row in report.restraint if row["window"][0] < a and cursor < row["window"][1]]
            segments = _hold_segments(context, subjects, style, cursor, a, report.layout_mode)
            reason = "; ".join(f"{s0:.1f}-{s1:.1f}s {why}" for s0, s1, why in segments) if len(segments) > 1 \
                else (segments[0][2] if segments else _region_reason(context, subjects, style, cursor, a,
                                                                    report.layout_mode))
            if restrained:
                reason = "editing restraint: " + "; ".join(
                    f"{row['intent'] or 'planned'} {row['event_id']} switched off ({', '.join(row['codes'])})"
                    for row in restrained[:3]) + f" | {reason}"
            holds.append({"window": [round(cursor, 3), round(a, 3)], "intent": ShotIntent.HOLD.value,
                          "reason": reason,
                          "segments": [{"window": [round(s0, 3), round(s1, 3)], "reason": why}
                                       for s0, s1, why in segments]})
        cursor = max(cursor, b)
    report.holds = holds
    return holds


# ============================================================
# ENTRY
# ============================================================

REPETITION_CODES = frozenset({"repeated_preset_limit"})
# Validator categories whose notes can switch a camera event off. Schema / caption / time
# sanitation (e.g. an uncontrolled reason_code) never does, so it must not block a retry.
RESTRAINT_CATEGORIES = frozenset({"motion", "story", "camera", "density"})


def _alternative_motions(event: EditEvent, style: StylePack) -> list[MotionPreset]:
    """Restrained motions that change the preset but not the framing intent."""
    allowed = style.role(event.role).allowed_motion
    order = (MotionPreset.SPEAKER_FOCUS, MotionPreset.SLOW_PUSH) if event.camera is not CameraMode.PRESERVE \
        else (MotionPreset.SLOW_PUSH,)
    return [m for m in order if m in allowed and m is not event.motion]


def _active_ids(plan: EditPlan) -> set[str]:
    return {event.event_id for event in plan.events if event.is_camera_active}


def _restraint_rows(before: Sequence[EditEvent], checked: Any, intents: Mapping[str, str]) -> list[dict[str, Any]]:
    """Directed camera events the validator's restraint policy switched off (with its codes)."""
    after = {e.event_id: e for e in (checked.plan.events if checked.plan is not None else ())}
    codes: dict[str, list[str]] = {}
    for issue in checked.issues:
        if issue.event_id and (issue.category in RESTRAINT_CATEGORIES
                               or getattr(issue.severity, "value", "") == "rejected"):
            codes.setdefault(issue.event_id, []).append(issue.code)
    rows = []
    for event in before:
        kept = after.get(event.event_id)
        if event.is_camera_active and (kept is None or not kept.is_camera_active):
            rows.append({"event_id": event.event_id, "window": [round(event.start, 3), round(event.end, 3)],
                         "intent": intents.get(event.event_id, ""),
                         "codes": sorted(set(codes.get(event.event_id, ["rejected"])))})
    return rows


def direct_plan(plan: EditPlan, context: EditContext, style: StylePack, *,
                validate: Callable[[EditPlan], Any] | None = None) -> tuple[EditPlan, DirectionReport]:
    """Evidence-directed copy of a validated plan.

    ``validate`` (the Pro Edit validator) keeps coverage honest: an added
    emphasis is accepted only when the validated plan still keeps every
    story-driven camera event (it may never displace a planner event through
    the density / repetition budget); a second, different restrained motion is
    tried once, otherwise the addition is recorded as restraint. Call
    ``explain_holds`` with the resolved camera path afterwards.
    """
    report = DirectionReport()
    subjects = camera_subjects(context, style, report.excluded_subjects)
    report.subjects = [s.to_dict() for s in subjects]
    report.layout_mode, report.layout_reason = screen_priority(context)
    directed: list[EditEvent] = []
    for event in plan.events:
        if not event.is_camera_active:
            directed.append(event)
            continue
        new_event, decision = direct_event(event, context, subjects, style, report.layout_mode)
        report.decisions.append(decision)
        directed.append(new_event)
    core = dataclasses.replace(plan, events=tuple(sorted(directed, key=lambda e: (e.start, e.event_id))))
    if validate is None:
        added, additions = add_coverage(directed, context, subjects, style, report.layout_mode)
        report.additions = additions
        events = tuple(sorted([*directed, *added], key=lambda e: (e.start, e.event_id)))
        return dataclasses.replace(plan, events=events), report
    intents = {row["event_id"]: row.get("intent", "") for row in report.decisions}
    checked = validate(core)
    accepted = checked.raise_if_fatal()
    restrained = _restraint_rows(core.events, checked, intents)
    active = _active_ids(accepted)
    by_id = {event.event_id: event for event in core.events}
    for row in restrained:
        # Repetition only (not the density budget / story locks): vary the motion
        # once instead of losing a story-driven shot; never displace another event.
        event = by_id.get(row["event_id"])
        if event is None or set(row["codes"]) - REPETITION_CODES:
            report.restraint.append(row)
            continue
        replaced = False
        for motion in _alternative_motions(event, style):
            trial_event = dataclasses.replace(event, motion=motion)
            trial = dataclasses.replace(accepted, events=tuple(sorted(
                [e for e in accepted.events if e.event_id != event.event_id] + [trial_event],
                key=lambda e: (e.start, e.event_id))))
            result = validate(trial)
            if result.fatal or result.plan is None:
                continue
            now_active = _active_ids(result.plan)
            if active <= now_active and event.event_id in now_active:
                accepted, active, replaced = result.plan, now_active, True
                for decision in report.decisions:
                    if decision["event_id"] == event.event_id:
                        decision["reason"] += f"; motion varied to {motion.value} (repetition limit)"
                        decision["directed"]["motion"] = motion.value
                break
        if not replaced:
            report.restraint.append(row)
    candidates, rows = add_coverage(list(accepted.events), context, subjects, style, report.layout_mode)
    for candidate, row in zip(candidates, rows):
        allowed = style.role(candidate.role).allowed_motion
        motions = [candidate.motion] + [m for m in (MotionPreset.SPEAKER_FOCUS,) if m in allowed
                                        and m is not candidate.motion]
        placed = False
        for motion in motions:
            trial_event = dataclasses.replace(candidate, motion=motion)
            trial = dataclasses.replace(accepted, events=tuple(sorted([*accepted.events, trial_event],
                                                                      key=lambda e: (e.start, e.event_id))))
            result = validate(trial)
            if result.fatal or result.plan is None:
                continue
            now_active = _active_ids(result.plan)
            if active <= now_active and trial_event.event_id in now_active:
                accepted, active, placed = result.plan, now_active, True
                report.additions.append({**row, "motion": motion.value})
                break
        if not placed:
            report.restraint.append({"event_id": candidate.event_id, "window": row["window"],
                                     "intent": row["intent"], "codes": ["coverage_would_displace_story_events"]})
    return accepted, report
