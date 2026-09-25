"""Style packs: editorial policy + bounded physical limits (no FFmpeg here).

A StylePack says WHAT is allowed per story role and within WHICH physical
bounds presets may resolve. The planner never sees raw numbers such as zoom
scales; it only picks enums and a normalized intensity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from ai.editor.pro_edit.errors import ProEditError
from ai.editor.pro_edit.schema import (
    CameraMode,
    CaptionStyle,
    MotionPreset,
    SfxCue,
    StoryRole,
    SupportVisual,
)


@dataclass(frozen=True)
class RolePolicy:
    objective: str
    allowed_motion: frozenset[MotionPreset]
    allowed_camera: frozenset[CameraMode]
    preferred_motion: tuple[MotionPreset, ...]
    preferred_camera: tuple[CameraMode, ...]
    # Deterministic downgrade when the planner picks a disallowed motion.
    fallback_motion: MotionPreset
    max_intensity: float
    min_intensity: float = 0.0
    strong_motion_allowed: bool = True
    support_visual_allowed: frozenset[SupportVisual] = frozenset({SupportVisual.NONE})
    sfx_allowed: bool = False
    # Minimum on-screen hold for camera emphasis ("do not cut away too early").
    min_hold_s: float = 0.0


@dataclass(frozen=True)
class ScaleBand:
    low: float
    high: float

    def at(self, intensity: float) -> float:
        value = max(0.0, min(1.0, float(intensity)))
        return self.low + (self.high - self.low) * value


@dataclass(frozen=True)
class TimeBand:
    """Seconds as a function of intensity: ``at_low`` at 0, ``at_high`` at 1."""

    at_low: float
    at_high: float

    def at(self, intensity: float) -> float:
        value = max(0.0, min(1.0, float(intensity)))
        return self.at_low + (self.at_high - self.at_low) * value


@dataclass(frozen=True)
class PresetBounds:
    scale: ScaleBand
    attack_s: TimeBand
    release_s: TimeBand
    max_active_s: float
    easing_in: str
    easing_out: str


@dataclass(frozen=True)
class ZoomLimits:
    min_zoom: float = 1.0
    max_zoom: float = 1.30
    # Zoom ceiling inside payoff/must-keep ranges when the action location is
    # unknown (center-safe target): keeps ~85% of the frame visible.
    protected_center_safe_max: float = 1.18
    # Gameplay layouts often carry an edge facecam overlay: without subject
    # evidence a centered crop stays gentle so the overlay is not cut.
    overlay_layout_center_safe_max: float = 1.10
    # Floor for pan-only framings (a crop needs zoom > 1 to move at all).
    pan_min_zoom: float = 1.04


@dataclass(frozen=True)
class CameraMotionLimits:
    max_speed_norm_per_s: float = 0.35
    dead_zone_x: float = 0.04
    dead_zone_y: float = 0.05
    smoothing_tau_s: float = 0.16
    subject_min_confidence: float = 0.55
    subject_min_coverage: float = 0.60
    low_confidence_hold_timeout_s: float = 0.80
    speaker_switch_min_hold_s: float = 1.20
    speaker_switch_confidence: float = 0.60
    keyframe_interval_s: float = 0.20
    max_reframe_distance_norm: float = 0.35
    # Minimum audiovisual evidence before a speaker id may steer the camera.
    speaker_link_min_confidence: float = 0.5


@dataclass(frozen=True)
class DensityPolicy:
    strong_events_per_10s: float = 3.0
    min_gap_between_strong_s: float = 0.80
    consecutive_same_preset_limit: int = 2
    max_strong_motion_seconds_ratio: float = 0.35
    camera_changes_per_10s: float = 6.0


@dataclass(frozen=True)
class BudgetPolicy:
    """Effect budget per 10 s of clip, scaled by duration (min 1 each)."""

    strong_motion_per_10s: float = 1.6
    support_visual_per_10s: float = 0.7
    snap_reframe_per_10s: float = 0.7
    sfx_per_10s: float = 0.8
    camera_events_per_10s: float = 4.0
    max_strong_motion_events: int = 8
    max_support_visual_events: int = 3
    max_snap_reframes: int = 3
    max_sfx_events: int = 4


@dataclass(frozen=True)
class ConfidencePolicy:
    low: float = 0.35
    medium: float = 0.60
    high: float = 0.80
    mild_intensity_cap: float = 0.50
    medium_intensity_cap: float = 0.80


@dataclass(frozen=True)
class SfxPolicy:
    allowed: frozenset[SfxCue] = frozenset(SfxCue)
    min_gap_s: float = 1.5
    executed_in_v1: bool = False


@dataclass(frozen=True)
class SupportVisualPolicy:
    forbid_over_protected: bool = True
    executed_in_v1: bool = False


@dataclass(frozen=True)
class EventDurationLimits:
    min_event_s: float = 0.30
    max_event_s: float = 8.0
    merge_gap_s: float = 0.12


@dataclass(frozen=True)
class StylePack:
    name: str
    version: int
    roles: Mapping[StoryRole, RolePolicy]
    presets: Mapping[MotionPreset, PresetBounds]
    framing: Mapping[CameraMode, ScaleBand]
    zoom: ZoomLimits = field(default_factory=ZoomLimits)
    camera: CameraMotionLimits = field(default_factory=CameraMotionLimits)
    density: DensityPolicy = field(default_factory=DensityPolicy)
    budget: BudgetPolicy = field(default_factory=BudgetPolicy)
    confidence: ConfidencePolicy = field(default_factory=ConfidencePolicy)
    sfx: SfxPolicy = field(default_factory=SfxPolicy)
    support_visual: SupportVisualPolicy = field(default_factory=SupportVisualPolicy)
    durations: EventDurationLimits = field(default_factory=EventDurationLimits)
    caption_style_defaults: Mapping[StoryRole, CaptionStyle] = field(default_factory=dict)

    @property
    def allowed_motion(self) -> frozenset[MotionPreset]:
        return frozenset(self.presets)

    @property
    def allowed_camera(self) -> frozenset[CameraMode]:
        return frozenset(self.framing)

    def role(self, role: StoryRole) -> RolePolicy:
        return self.roles.get(role, self.roles[StoryRole.NEUTRAL])

    def describe_for_planner(self) -> dict[str, object]:
        """Compact, number-free policy summary for the planner prompt."""
        return {
            "name": self.name,
            "version": self.version,
            "roles": {
                role.value: {
                    "objective": policy.objective,
                    "allowed_motion": sorted(m.value for m in policy.allowed_motion),
                    "allowed_camera": sorted(c.value for c in policy.allowed_camera),
                    "preferred_motion": [m.value for m in policy.preferred_motion],
                    "preferred_camera": [c.value for c in policy.preferred_camera],
                    "max_intensity": policy.max_intensity,
                    "strong_motion_allowed": policy.strong_motion_allowed,
                    "support_visual_allowed": sorted(v.value for v in policy.support_visual_allowed),
                }
                for role, policy in self.roles.items()
            },
        }


def _roles_pro_stream_v1() -> dict[StoryRole, RolePolicy]:
    M, C, V = MotionPreset, CameraMode, SupportVisual
    return {
        StoryRole.HOOK: RolePolicy(
            objective="maximum immediate clarity; strong framing, limited but meaningful motion, no clutter",
            allowed_motion=frozenset({M.STATIC_CLEAN, M.SLOW_PUSH, M.PUNCH_IN, M.PUNCH_IN_FAST, M.SPEAKER_FOCUS}),
            allowed_camera=frozenset({C.PRESERVE, C.SPEAKER_MEDIUM, C.SPEAKER_CLOSE, C.REACTION_CLOSE}),
            preferred_motion=(M.PUNCH_IN, M.SLOW_PUSH),
            preferred_camera=(C.SPEAKER_CLOSE, C.REACTION_CLOSE),
            fallback_motion=M.SLOW_PUSH,
            max_intensity=0.85,
            sfx_allowed=True,
        ),
        StoryRole.SETUP: RolePolicy(
            objective="context comprehension; strong effects suppressed",
            allowed_motion=frozenset({M.STATIC_CLEAN, M.SLOW_PUSH, M.SPEAKER_FOCUS}),
            allowed_camera=frozenset({C.PRESERVE, C.SPEAKER_MEDIUM}),
            preferred_motion=(M.STATIC_CLEAN, M.SLOW_PUSH),
            preferred_camera=(C.PRESERVE, C.SPEAKER_MEDIUM),
            fallback_motion=M.SLOW_PUSH,
            max_intensity=0.60,
            strong_motion_allowed=False,
        ),
        StoryRole.ESCALATION: RolePolicy(
            objective="visual energy proportional to story intensity; avoid constant movement",
            allowed_motion=frozenset({M.STATIC_CLEAN, M.SLOW_PUSH, M.PUNCH_IN, M.SPEAKER_FOCUS, M.SNAP_REFRAME}),
            allowed_camera=frozenset({C.PRESERVE, C.SPEAKER_MEDIUM, C.SPEAKER_CLOSE, C.DUAL_SUBJECT}),
            preferred_motion=(M.SPEAKER_FOCUS, M.PUNCH_IN, M.SLOW_PUSH),
            preferred_camera=(C.SPEAKER_MEDIUM, C.SPEAKER_CLOSE),
            fallback_motion=M.SLOW_PUSH,
            max_intensity=0.85,
            support_visual_allowed=frozenset({V.NONE, V.MEME, V.BROLL}),
            sfx_allowed=True,
        ),
        StoryRole.PAYOFF: RolePolicy(
            objective="preserve and emphasize the actual selected event; payoff visibility beats decoration",
            allowed_motion=frozenset({M.STATIC_CLEAN, M.SLOW_PUSH, M.PUNCH_IN, M.PUNCH_IN_FAST, M.SPEAKER_FOCUS}),
            allowed_camera=frozenset({C.PRESERVE, C.SPEAKER_CLOSE, C.REACTION_CLOSE, C.OBJECT_FOCUS,
                                      C.DUAL_SUBJECT}),
            preferred_motion=(M.PUNCH_IN, M.PUNCH_IN_FAST),
            preferred_camera=(C.SPEAKER_CLOSE, C.OBJECT_FOCUS),
            fallback_motion=M.PUNCH_IN,
            max_intensity=1.0,
            sfx_allowed=True,
            min_hold_s=0.25,
        ),
        StoryRole.REACTION: RolePolicy(
            objective="preserve emotional/comedic consequence; brief visual hold; do not cut away early",
            allowed_motion=frozenset({M.STATIC_CLEAN, M.SLOW_PUSH, M.PUNCH_IN, M.SPEAKER_FOCUS}),
            allowed_camera=frozenset({C.PRESERVE, C.REACTION_CLOSE, C.SPEAKER_CLOSE, C.DUAL_SUBJECT}),
            preferred_motion=(M.PUNCH_IN, M.SLOW_PUSH),
            preferred_camera=(C.REACTION_CLOSE,),
            fallback_motion=M.SLOW_PUSH,
            max_intensity=0.85,
            sfx_allowed=True,
            min_hold_s=0.60,
        ),
        StoryRole.BRIDGE: RolePolicy(
            objective="maintain continuity; minimal motion",
            allowed_motion=frozenset({M.STATIC_CLEAN, M.SLOW_PUSH, M.SUBTLE_PULL_OUT}),
            allowed_camera=frozenset({C.PRESERVE, C.SPEAKER_MEDIUM}),
            preferred_motion=(M.STATIC_CLEAN, M.SLOW_PUSH),
            preferred_camera=(C.PRESERVE,),
            fallback_motion=M.STATIC_CLEAN,
            max_intensity=0.50,
            strong_motion_allowed=False,
            support_visual_allowed=frozenset({V.NONE, V.MEME, V.BROLL}),
        ),
        StoryRole.NEUTRAL: RolePolicy(
            objective="no story evidence; prefer the untouched source",
            allowed_motion=frozenset({M.STATIC_CLEAN, M.SLOW_PUSH, M.SUBTLE_PULL_OUT}),
            allowed_camera=frozenset({C.PRESERVE, C.SPEAKER_MEDIUM}),
            preferred_motion=(M.STATIC_CLEAN,),
            preferred_camera=(C.PRESERVE,),
            fallback_motion=M.STATIC_CLEAN,
            max_intensity=0.40,
            strong_motion_allowed=False,
        ),
    }


def _presets_pro_stream_v1() -> dict[MotionPreset, PresetBounds]:
    M = MotionPreset
    return {
        M.STATIC_CLEAN: PresetBounds(ScaleBand(1.0, 1.0), TimeBand(0.0, 0.0), TimeBand(0.0, 0.0), 8.0,
                                     "linear", "linear"),
        M.SLOW_PUSH: PresetBounds(ScaleBand(1.04, 1.10), TimeBand(3.0, 0.8), TimeBand(0.45, 0.35), 3.0,
                                  "smoothstep", "smoothstep"),
        M.PUNCH_IN: PresetBounds(ScaleBand(1.10, 1.20), TimeBand(0.35, 0.16), TimeBand(0.32, 0.22), 2.5,
                                 "ease_out_cubic", "smoothstep"),
        M.PUNCH_IN_FAST: PresetBounds(ScaleBand(1.14, 1.24), TimeBand(0.24, 0.10), TimeBand(0.28, 0.20), 2.0,
                                      "ease_out_cubic", "smoothstep"),
        # Speaker focus zoom comes from the camera framing band; attack/release
        # are the ease into/out of the framed follow.
        M.SPEAKER_FOCUS: PresetBounds(ScaleBand(1.04, 1.12), TimeBand(0.45, 0.30), TimeBand(0.45, 0.35), 8.0,
                                      "smoothstep", "smoothstep"),
        M.SNAP_REFRAME: PresetBounds(ScaleBand(1.08, 1.16), TimeBand(0.22, 0.08), TimeBand(0.35, 0.25), 4.0,
                                     "smoothstep", "smoothstep"),
        M.SUBTLE_PULL_OUT: PresetBounds(ScaleBand(1.0, 1.0), TimeBand(1.2, 0.6), TimeBand(0.0, 0.0), 3.0,
                                        "smoothstep", "smoothstep"),
    }


def _framing_pro_stream_v1() -> dict[CameraMode, ScaleBand]:
    C = CameraMode
    return {
        C.PRESERVE: ScaleBand(1.0, 1.0),
        C.SPEAKER_MEDIUM: ScaleBand(1.05, 1.10),
        C.SPEAKER_CLOSE: ScaleBand(1.12, 1.20),
        C.REACTION_CLOSE: ScaleBand(1.12, 1.30),
        C.DUAL_SUBJECT: ScaleBand(1.0, 1.10),
        C.OBJECT_FOCUS: ScaleBand(1.08, 1.18),
    }


PRO_STREAM_V1 = StylePack(
    name="pro_stream_v1",
    version=1,
    roles=MappingProxyType(_roles_pro_stream_v1()),
    presets=MappingProxyType(_presets_pro_stream_v1()),
    framing=MappingProxyType(_framing_pro_stream_v1()),
    caption_style_defaults=MappingProxyType({
        StoryRole.PAYOFF: CaptionStyle.IMPACT,
        StoryRole.REACTION: CaptionStyle.EMPHASIS,
    }),
)

_REGISTRY: dict[str, StylePack] = {PRO_STREAM_V1.name: PRO_STREAM_V1}


def get_style_pack(name: str) -> StylePack:
    key = str(name or "").strip()
    try:
        return _REGISTRY[key]
    except KeyError as error:
        raise ProEditError(
            f"unknown Pro Edit style pack {key!r}; available: {', '.join(sorted(_REGISTRY))}"
        ) from error


def available_style_packs() -> list[str]:
    return sorted(_REGISTRY)
