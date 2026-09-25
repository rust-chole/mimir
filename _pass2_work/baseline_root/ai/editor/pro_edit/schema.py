"""Versioned EditPlan data contract (presentation intent only).

The plan contains ONLY bounded semantic choices: enums, a normalized
``intensity`` and ``confidence`` in [0, 1], time windows in PACED_CLIP seconds
and references to existing caption word IDs. It has no field that can carry
caption text, word timings, speaker labels, story boundaries, zoom scales,
crop geometry, output geometry or commands. Anything like that is rejected
by the validator.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ai.editor.pro_edit.timebase import TimelineDomain

EDIT_PLAN_SCHEMA_VERSION = 2
SUPPORTED_PLAN_SCHEMA_VERSIONS = frozenset({EDIT_PLAN_SCHEMA_VERSION})
PLAN_TIMELINE_DOMAIN = TimelineDomain.PACED_CLIP
MAX_PLAN_EVENTS = 48
MAX_EMPHASIS_WORDS_PER_EVENT = 6
MAX_EVENT_NOTE_CHARS = 160


class StoryRole(str, Enum):
    HOOK = "hook"
    SETUP = "setup"
    ESCALATION = "escalation"
    PAYOFF = "payoff"
    REACTION = "reaction"
    BRIDGE = "bridge"
    NEUTRAL = "neutral"


class CameraMode(str, Enum):
    PRESERVE = "preserve"
    SPEAKER_MEDIUM = "speaker_medium"
    SPEAKER_CLOSE = "speaker_close"
    REACTION_CLOSE = "reaction_close"
    DUAL_SUBJECT = "dual_subject"
    OBJECT_FOCUS = "object_focus"


class MotionPreset(str, Enum):
    STATIC_CLEAN = "static_clean"
    SLOW_PUSH = "slow_push"
    PUNCH_IN = "punch_in"
    PUNCH_IN_FAST = "punch_in_fast"
    SPEAKER_FOCUS = "speaker_focus"
    SNAP_REFRAME = "snap_reframe"
    SUBTLE_PULL_OUT = "subtle_pull_out"


class SupportVisual(str, Enum):
    NONE = "none"
    MEME = "meme"
    BROLL = "broll"


class CaptionStyle(str, Enum):
    DEFAULT = "default"
    EMPHASIS = "emphasis"
    IMPACT = "impact"


class EmphasisReason(str, Enum):
    """Why a word is emphasized (optional; absent = generic / locally derived).

    contrast / surprise are planner-only judgements; MIMIR never infers them.
    """

    GENERIC = "generic"
    NAME = "name"
    NUMBER = "number"
    PAYOFF = "payoff"
    REACTION = "reaction"
    CONTRAST = "contrast"
    SURPRISE = "surprise"


class SfxCue(str, Enum):
    NONE = "none"
    IMPACT_SOFT = "impact_soft"
    WHOOSH_SOFT = "whoosh_soft"
    POP_SOFT = "pop_soft"


class TargetType(str, Enum):
    CENTER_SAFE = "center_safe"
    SUBJECT = "subject"
    ACTIVE_SPEAKER = "active_speaker"
    DOMINANT_SUBJECT = "dominant_subject"


class ReasonCode(str, Enum):
    HOOK_EMPHASIS = "hook_emphasis"
    SETUP_CLARITY = "setup_clarity"
    SPEAKER_SHIFT = "speaker_shift"
    ESCALATION_RISE = "escalation_rise"
    PAYOFF_HIT = "payoff_hit"
    REACTION_HOLD = "reaction_hold"
    REACTION_AFTER_PAYOFF = "reaction_after_payoff"
    VISUAL_CLARITY = "visual_clarity"
    LOW_MOTION_COMPENSATION = "low_motion_compensation"
    BRIDGE_CONTINUITY = "bridge_continuity"
    UNCERTAIN_HOLD = "uncertain_hold"
    VALIDATOR_FALLBACK = "validator_fallback"


class Channel(str, Enum):
    CAMERA = "camera"
    CAPTION_STYLE = "caption_style"
    SUPPORT_VISUAL = "support_visual"
    SFX = "sfx"
    TRANSITION = "transition"


# The intro directive may only present the EXISTING cold-open.
INTRO_MOTIONS = frozenset({MotionPreset.STATIC_CLEAN, MotionPreset.SLOW_PUSH, MotionPreset.PUNCH_IN,
                           MotionPreset.PUNCH_IN_FAST})
INTRO_CAMERAS = frozenset({CameraMode.PRESERVE, CameraMode.SPEAKER_CLOSE, CameraMode.REACTION_CLOSE})

STRONG_MOTIONS = frozenset({MotionPreset.PUNCH_IN, MotionPreset.PUNCH_IN_FAST, MotionPreset.SNAP_REFRAME})
MILD_MOTIONS = frozenset({MotionPreset.STATIC_CLEAN, MotionPreset.SLOW_PUSH, MotionPreset.SUBTLE_PULL_OUT,
                          MotionPreset.SPEAKER_FOCUS})

# Deterministic conflict-resolution priority (higher wins). PAYOFF_PROTECTION >
# REACTION_PROTECTION > CAMERA_FOCUS > MOTION, per spec section 20.
ROLE_PRIORITY: dict[StoryRole, int] = {
    StoryRole.PAYOFF: 60,
    StoryRole.REACTION: 50,
    StoryRole.HOOK: 40,
    StoryRole.ESCALATION: 30,
    StoryRole.SETUP: 20,
    StoryRole.BRIDGE: 10,
    StoryRole.NEUTRAL: 0,
}


@dataclass(frozen=True)
class EditTarget:
    type: TargetType = TargetType.CENTER_SAFE
    id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type.value, "id": self.id}


@dataclass(frozen=True)
class EditEvent:
    event_id: str
    start: float
    end: float
    role: StoryRole
    camera: CameraMode
    motion: MotionPreset
    target: EditTarget
    intensity: float
    confidence: float
    reason_code: ReasonCode
    caption_style: CaptionStyle = CaptionStyle.DEFAULT
    emphasis_word_ids: tuple[int, ...] = ()
    sfx: SfxCue = SfxCue.NONE
    support_visual: SupportVisual = SupportVisual.NONE
    # Semantic link to the story span this presentation choice belongs to.
    story_span_id: str | None = None
    # Optional free-form debugging explanation. Never read by machine logic.
    note: str = ""
    # Optional (word_id, reason) for ids in emphasis_word_ids (compatible extension of plan v2).
    emphasis_reasons: tuple[tuple[int, EmphasisReason], ...] = ()

    @property
    def duration(self) -> float:
        return float(self.end) - float(self.start)

    @property
    def is_strong(self) -> bool:
        return self.motion in STRONG_MOTIONS

    @property
    def is_camera_active(self) -> bool:
        return not (self.motion is MotionPreset.STATIC_CLEAN and self.camera is CameraMode.PRESERVE)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "event_id": self.event_id,
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "role": self.role.value,
            "camera": self.camera.value,
            "motion": self.motion.value,
            "target": self.target.to_dict(),
            "intensity": round(float(self.intensity), 4),
            "caption_style": self.caption_style.value,
            "emphasis_word_ids": list(self.emphasis_word_ids),
            "sfx": self.sfx.value,
            "support_visual": self.support_visual.value,
            "confidence": round(float(self.confidence), 4),
            "reason_code": self.reason_code.value,
            "story_span_id": self.story_span_id,
        }
        if self.note:
            data["note"] = self.note
        if self.emphasis_reasons:
            data["emphasis_reasons"] = [{"word_id": wid, "reason": reason.value}
                                        for wid, reason in self.emphasis_reasons]
        return data


@dataclass(frozen=True)
class IntroDirective:
    """Presentation of the already-selected cold-open (whole intro, no times)."""

    camera: CameraMode
    motion: MotionPreset
    target: EditTarget
    intensity: float
    confidence: float
    reason_code: ReasonCode
    note: str = ""

    @property
    def is_camera_active(self) -> bool:
        return not (self.motion is MotionPreset.STATIC_CLEAN and self.camera is CameraMode.PRESERVE)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "camera": self.camera.value, "motion": self.motion.value, "target": self.target.to_dict(),
            "intensity": round(float(self.intensity), 4), "confidence": round(float(self.confidence), 4),
            "reason_code": self.reason_code.value,
        }
        if self.note:
            data["note"] = self.note
        return data


@dataclass(frozen=True)
class EditPlan:
    schema_version: int
    style_pack: str
    timeline_domain: TimelineDomain
    events: tuple[EditEvent, ...] = ()
    planner: str = "unknown"
    intro: IntroDirective | None = None
    extras: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "style_pack": self.style_pack,
            "timeline_domain": self.timeline_domain.value,
            "planner": self.planner,
            "intro": self.intro.to_dict() if self.intro is not None else None,
            "events": [event.to_dict() for event in self.events],
        }

    @property
    def camera_events(self) -> tuple[EditEvent, ...]:
        return tuple(event for event in self.events if event.is_camera_active)


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def plan_hash(
    plan: EditPlan,
    *,
    clip_identity: str,
    style_name: str,
    style_version: int,
    engine_version: int,
) -> str:
    """Traceable identifier: clip identity + plan + style version + engine."""
    payload = {
        "clip": clip_identity,
        "plan": plan.to_dict(),
        "style": [style_name, int(style_version)],
        "engine": int(engine_version),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def empty_plan(style_pack: str, planner: str) -> EditPlan:
    return EditPlan(
        schema_version=EDIT_PLAN_SCHEMA_VERSION,
        style_pack=style_pack,
        timeline_domain=PLAN_TIMELINE_DOMAIN,
        events=(),
        planner=planner,
    )


def _enum_values(enum_type: type[Enum]) -> list[str]:
    return [member.value for member in enum_type]  # type: ignore[attr-defined]


def planner_json_schema(style_pack: str, max_events: int = MAX_PLAN_EVENTS) -> dict[str, Any]:
    """Strict structured-output schema for model planners.

    Deliberately contains no text/time-rewrite/geometry fields.
    """
    reason_codes = [code for code in _enum_values(ReasonCode) if code != ReasonCode.VALIDATOR_FALLBACK.value]
    event = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "event_id": {"type": "string"},
            "start": {"type": "number"},
            "end": {"type": "number"},
            "role": {"type": "string", "enum": _enum_values(StoryRole)},
            "camera": {"type": "string", "enum": _enum_values(CameraMode)},
            "motion": {"type": "string", "enum": _enum_values(MotionPreset)},
            "target": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"type": "string", "enum": _enum_values(TargetType)},
                    "id": {"type": ["string", "null"]},
                },
                "required": ["type", "id"],
            },
            "intensity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "caption_style": {"type": "string", "enum": _enum_values(CaptionStyle)},
            "emphasis_word_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "maxItems": MAX_EMPHASIS_WORDS_PER_EVENT,
            },
            "emphasis_reasons": {
                "type": "array",
                "maxItems": MAX_EMPHASIS_WORDS_PER_EVENT,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"word_id": {"type": "integer"},
                                   "reason": {"type": "string", "enum": _enum_values(EmphasisReason)}},
                    "required": ["word_id", "reason"],
                },
            },
            "sfx": {"type": "string", "enum": _enum_values(SfxCue)},
            "support_visual": {"type": "string", "enum": _enum_values(SupportVisual)},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason_code": {"type": "string", "enum": reason_codes},
            "story_span_id": {"type": ["string", "null"]},
        },
        "required": [
            "event_id", "start", "end", "role", "camera", "motion", "target",
            "intensity", "caption_style", "emphasis_word_ids", "emphasis_reasons", "sfx",
            "support_visual", "confidence", "reason_code", "story_span_id",
        ],
    }
    intro = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "camera": {"type": "string", "enum": sorted(c.value for c in INTRO_CAMERAS)},
            "motion": {"type": "string", "enum": sorted(m.value for m in INTRO_MOTIONS)},
            "target": event["properties"]["target"],
            "intensity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason_code": {"type": "string", "enum": reason_codes},
        },
        "required": ["camera", "motion", "target", "intensity", "confidence", "reason_code"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "integer", "enum": sorted(SUPPORTED_PLAN_SCHEMA_VERSIONS)},
            "style_pack": {"type": "string", "enum": [style_pack]},
            "timeline_domain": {"type": "string", "enum": [PLAN_TIMELINE_DOMAIN.value]},
            "intro": {"anyOf": [intro, {"type": "null"}]},
            "events": {"type": "array", "maxItems": int(max_events), "items": event},
        },
        "required": ["schema_version", "style_pack", "timeline_domain", "intro", "events"],
    }
