"""Typed settings. Each stage signs only the settings section it consumes.

Environment (``.env`` is read when python-dotenv is installed):

    OPENAI_API_KEY              production model provider
    MIMIR_LANGUAGE              transcription language (default ``en``)
    MIMIR_TERRA_MODEL / MIMIR_LUNA_MODEL   editorial model families
    MIMIR_ROUTE_<ROLE>=model[:effort]      per-role override, e.g. MIMIR_ROUTE_EDIT_DIRECTOR=gpt-5.6-terra:high
    MIMIR_CAPTION_ENTITIES      comma separated verified names/terms
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from mimir.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:  # optional convenience; production can set real environment variables instead
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:  # pragma: no cover - dotenv is optional
    pass


REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


@dataclass(frozen=True)
class ModelRoute:
    model: str
    effort: str | None = None      # reasoning effort for reasoning models; None for audio models

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ConfigError("empty model name in route")
        if self.effort is not None and self.effort not in REASONING_EFFORTS:
            raise ConfigError(f"invalid reasoning effort {self.effort!r}")


TERRA = os.getenv("MIMIR_TERRA_MODEL", "gpt-5.6-terra").strip() or "gpt-5.6-terra"
LUNA = os.getenv("MIMIR_LUNA_MODEL", "gpt-5.6-luna").strip() or "gpt-5.6-luna"

# High reasoning is reserved for decisions where a wrong answer changes the
# actual Short (story selection, edit direction); scouting and drafting use Luna.
DEFAULT_ROUTES: dict[str, ModelRoute] = {
    # audio
    "transcribe_fast": ModelRoute("gpt-4o-mini-transcribe"),      # whole-VOD lexical ear
    "transcribe_primary": ModelRoute("gpt-transcribe"),           # caption lexical authority
    "transcribe_crosscheck": ModelRoute("gpt-4o-transcribe"),     # model-diverse caption ear
    "timing": ModelRoute("whisper-1"),                            # word clock (timing authority)
    "diarize": ModelRoute("gpt-4o-transcribe-diarize"),           # speaker truth
    # editorial
    "story_scout": ModelRoute(LUNA, "medium"),
    "story_composer": ModelRoute(LUNA, "medium"),
    "story_judge": ModelRoute(TERRA, "medium"),
    "story_judge_escalation": ModelRoute(TERRA, "high"),
    "story_expansion": ModelRoute(TERRA, "medium"),
    "boundary_polish": ModelRoute(LUNA, "low"),
    "peak_probe": ModelRoute(LUNA, "medium"),                     # bounded multimodal look at VOD peaks
    "visual_observer": ModelRoute(LUNA, "medium"),                # bounded multimodal look at the selected Short
    "cold_open": ModelRoute(LUNA, "medium"),
    "hook_judge": ModelRoute(TERRA, "medium"),
    "edit_director": ModelRoute(TERRA, "medium"),
    "effects": ModelRoute(LUNA, "medium"),
    "final_reviewer": ModelRoute(TERRA, "medium"),
}


def _route_overrides() -> dict[str, ModelRoute]:
    routes = dict(DEFAULT_ROUTES)
    for role in DEFAULT_ROUTES:
        raw = os.getenv(f"MIMIR_ROUTE_{role.upper()}", "").strip()
        if not raw:
            continue
        model, _, effort = raw.partition(":")
        routes[role] = ModelRoute(model.strip(), effort.strip() or DEFAULT_ROUTES[role].effort)
    return routes


@dataclass(frozen=True)
class OutputSettings:
    width: int = 1080
    height: int = 1920
    fps: int = 0                    # 0 = auto (60 for >=50 fps sources, else 30; 24/25 kept)
    crf: int = 18
    preset: str = "medium"
    audio_bitrate: str = "192k"
    audio_rate: int = 48000
    loudness_lufs: float = -14.0
    true_peak_db: float = -1.0


@dataclass(frozen=True)
class TranscriptSettings:
    language: str = field(default_factory=lambda: os.getenv("MIMIR_LANGUAGE", "en").strip() or "en")
    chunk_seconds: float = 600.0
    domain_keywords: tuple[str, ...] = ("chat", "Twitch", "YouTube", "Discord", "stream", "streamer", "IRL",
                                        "dono", "no cap")


@dataclass(frozen=True)
class CaptionVerifySettings:
    """Caption-grade lexical verification of the selected story window."""

    word_accuracy_target: float = 0.97
    max_suspect_spans: int = 10
    suspect_max_words: int = 6
    micro_context_words: int = 8
    micro_pad_seconds: float = 1.25
    micro_min_seconds: float = 6.0
    micro_max_seconds: float = 10.0
    known_name_similarity: float = 0.64
    parallel_workers: int = 5
    clock_guard: bool = True
    # Evidence-backed orthographic aliases (e.g. {"shorty": "shawty"}); applied only
    # when an independent micro ear emitted the preferred spelling.
    orthographic_aliases: tuple[tuple[str, str], ...] = (("shorty", "shawty"),)


@dataclass(frozen=True)
class IdentitySettings:
    interactive: bool = False
    speaker_names: tuple[tuple[str, str], ...] = ()      # (speaker_id, name) confirmed by the user
    creator: str = ""
    entities: tuple[str, ...] = field(default_factory=lambda: tuple(
        " ".join(part.split()) for part in os.getenv("MIMIR_CAPTION_ENTITIES", "").split(",") if part.strip()))


@dataclass(frozen=True)
class StorySettings:
    min_duration: float = 12.0
    max_duration: float = 55.0
    target_min: float = 22.0
    target_ideal: float = 32.0
    target_max: float = 42.0
    short_exception_below: float = 18.0
    max_money_moments: int = 12
    max_candidates: int = 5
    judge_escalation_margin: float = 0.35
    boundary_max_extend: float = 3.0
    boundary_max_trim: float = 1.5
    story_index: int | None = None      # force a candidate (1-based rank) instead of the judge's choice
    peak_probe_max_regions: int = 6


@dataclass(frozen=True)
class PacingSettings:
    min_gap: float = 0.60
    auto_cut_gap: float = 1.05
    auto_cut_min_removable: float = 0.52
    keep_after_word: float = 0.18
    keep_before_word: float = 0.18
    min_removable: float = 0.22
    payoff_protect_before: float = 0.40
    payoff_protect_after: float = 0.50
    leading_trim_threshold: float = 0.70
    trailing_trim_threshold: float = 0.90
    leading_padding: float = 0.08
    trailing_padding: float = 0.18
    max_cut_ratio: float = 0.45
    max_cut_seconds: float = 16.0


@dataclass(frozen=True)
class ColdOpenSettings:
    min_duration: float = 2.2
    max_duration: float = 6.5
    weak_peak_min: float = 3.2
    moderate_peak_min: float = 2.85
    strong_peak_min: float = 2.35
    hook_min_words: int = 2
    hook_max_words: int = 7
    hook_max_chars: int = 46
    hook_accept_score: float = 8.0
    transition_frames: int = 5


@dataclass(frozen=True)
class VisionSettings:
    analysis_fps: float = 10.0
    analysis_width: int = 1280
    detect_every: int = 3
    observer: bool = True           # bounded multimodal observer on the selected Short
    observer_max_frames: int = 12


@dataclass(frozen=True)
class CameraSettings:
    max_zoom: float = 1.45          # relative to the tallest output-aspect crop
    max_upscale: float = 2.6        # output pixels per source pixel
    punch_zoom: float = 1.22
    medium_zoom: float = 1.08
    transition_seconds: float = 0.45
    punch_seconds: float = 0.18
    follow_deadzone: float = 0.035  # of source width
    follow_max_speed: float = 0.35  # source widths per second
    min_shot_seconds: float = 1.0
    max_punches_per_10s: int = 2
    jump_cut_mask_zoom: float = 1.06


@dataclass(frozen=True)
class CaptionStyle:
    font: str = "Arial"
    font_file: str = ""             # optional explicit font file (fontsdir derived from it)
    size: int = 74
    outline: int = 6
    shadow: int = 2
    words_per_group: int = 4
    max_group_chars: int = 30
    group_break_gap: float = 0.42
    max_hold_gap: float = 0.18
    min_event: float = 0.07
    terminal_hold: float = 0.32
    margin_v: int = 560
    margin_h: int = 90
    base_color: str = "&H00F2F2F2"
    active_color: str = "&H0000D7FF"
    emphasis_color: str = "&H000080FF"
    secondary_base_color: str = "&H00FFF7E8"
    secondary_active_color: str = "&H00FFE054"
    secondary_emphasis_color: str = "&H00FF6BA9"
    secondary_margin_v: int = 680
    hook_font: str = "Arial"
    hook_size: int = 96
    hook_color: str = "&H0000FFD7"
    hook_y_ratio: float = 0.24
    uppercase: bool = False


@dataclass(frozen=True)
class EffectsSettings:
    sfx_library: str = str(PROJECT_ROOT / "sfx_library")
    enable_accents: bool = True
    transition_whoosh: bool = True
    accent_volume: float = 0.5
    whoosh_volume: float = 0.35
    duck_threshold: float = 0.035
    unexpectedness_min: float = 0.70
    audio_min_fit: float = 7.35
    visual_min_fit: float = 7.9


@dataclass(frozen=True)
class QCSettings:
    reviewer: bool = False          # optional bounded multimodal final reviewer
    repair_passes: int = 1          # at most one controlled repair
    av_sync_tolerance: float = 0.045
    caption_timing_tolerance_frames: int = 1
    max_center_step: float = 0.012  # of source width per frame inside a shot
    pixel_samples: int = 10


@dataclass(frozen=True)
class RepairDirectives:
    """Deterministic repair actions from a failed QC (at most one round)."""

    round: int = 0
    widen_spans: tuple[str, ...] = ()
    conservative_camera: bool = False
    drop_visual_accents: bool = False
    rerender: bool = False


@dataclass(frozen=True)
class Settings:
    workspace: Path = PROJECT_ROOT / "workspace"
    output_dir: Path = PROJECT_ROOT / "output"
    output: OutputSettings = field(default_factory=OutputSettings)
    transcript: TranscriptSettings = field(default_factory=TranscriptSettings)
    caption_verify: CaptionVerifySettings = field(default_factory=CaptionVerifySettings)
    identity: IdentitySettings = field(default_factory=IdentitySettings)
    story: StorySettings = field(default_factory=StorySettings)
    pacing: PacingSettings = field(default_factory=PacingSettings)
    cold_open: ColdOpenSettings = field(default_factory=ColdOpenSettings)
    vision: VisionSettings = field(default_factory=VisionSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    captions: CaptionStyle = field(default_factory=CaptionStyle)
    effects: EffectsSettings = field(default_factory=EffectsSettings)
    qc: QCSettings = field(default_factory=QCSettings)
    repair: RepairDirectives = field(default_factory=RepairDirectives)
    routes: dict[str, ModelRoute] = field(default_factory=_route_overrides)

    def route(self, role: str) -> ModelRoute:
        try:
            return self.routes[role]
        except KeyError as error:
            raise ConfigError(f"no model route for role {role!r}") from error

    def with_(self, **changes: Any) -> "Settings":
        return replace(self, **changes)


def section(settings: Settings, name: str) -> dict[str, Any]:
    value = getattr(settings, name)
    return dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value


def routes_for(settings: Settings, *roles: str) -> dict[str, dict[str, str | None]]:
    return {role: {"model": settings.route(role).model, "effort": settings.route(role).effort} for role in roles}


def parse_speaker_names(raw: str) -> tuple[tuple[str, str], ...]:
    """``S1=Kai,S2=Tyla`` -> (("S1", "Kai"), ("S2", "Tyla"))."""
    rows: list[tuple[str, str]] = []
    for part in str(raw or "").split(","):
        if not part.strip():
            continue
        key, sep, value = part.partition("=")
        if not sep or not key.strip() or not value.strip():
            raise ConfigError(f"invalid speaker name mapping {part!r}; expected S1=Name")
        rows.append((key.strip().upper(), " ".join(value.split())))
    return tuple(rows)
