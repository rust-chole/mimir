"""Pro Edit caption presentation: a deterministic layer over caption truth.

TRUTH (owned by speaker_caption_support + captions; read-only here):
    word id, text, start/end on the exact final clip clock, speaker, label,
    uncertainty.
PRESENTATION (this module):
    pages, lines, position, typography, active highlight, emphasis.

    authoritative words -> CaptionTokenRef -> pages/lines (glyph-width layout)
    -> CaptionStyleDefinition --resolve_caption_style--> ResolvedCaptionStyle
    -> ASS (positioned lines) -> FFmpeg/libass

The planner contributes intent only: ``caption_style`` per event and
``emphasis_word_ids``. Font pixels, ASS tags, scale values, positions and
milliseconds live in the declarative tables of this module. Nothing here
retimes speech: every word becomes visible exactly at its measured onset (the
same tokens ``captions._profile_edited_words`` produces, verified at runtime),
and display holds only extend a page's end, capped at the next page in the
same lane, like the baseline terminal hold.

V4 adds, all deterministic and presentation-only: a validated BrandProfile
(``mimir_default`` == V3.1), platform safe zones, per-page placement (face /
story / activity / platform aware, one position per page cluster), a small
primitive library resolved from (style, emphasis reason, brand), emphasis
reasons (planner or safe local derivation) and a pluggable width shaper.

V5 adds deterministic LEGIBILITY (background luminance / complexity under
each placed page vs the page's actual colours -> minimum intervention:
outline boost, shadow, budgeted plate), TEXT_LIKE / UI occupancy and layout
evidence for placement, the reason -> primitive grammar and effect overrides
for the editorial energy coordinator. None of it touches text or timing.

Current-root binding (captions V24): WHO is displayed, and on which lane, is
caption truth. Token roles and labels come from the caption renderer's own
``_prepare_adaptive_render_words`` / ``_trusted_human_display_map`` (only
human-confirmed names are printed; a second lane exists only inside measured
speech overlap; raw diarization roles never create permanent A/B lanes or a
third lane). Hard breaks follow the renderer's visible-label rule, each lane on
its own clock, and in a diarization hard failure (``_speaker_render_hard_failure``)
pages in one lane never overlap. The raw ``speaker_raw`` id stays truth.

Speaker colours (captions V25): a page's COLOURS come from its voice's
``speaker_color`` (the profile's acoustic turns), its LANE only from measured
overlap. A colour change is a hard break, so a page never mixes two voices.

Geometry law: a page's geometry is fixed when it appears. Each line is one
explicitly positioned ASS event (``\\an2\\pos``); future words are transparent
but occupy their width; the active word changes colour (and, for emphasis
styles, outline thickness, which libass does not lay out); an emphasized
word's static scale is reserved from the page's first frame. No width-changing
animation exists.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ai.editor import captions as caption_truth
from ai.editor.pro_edit.caption_brand import MIMIR_DEFAULT, BrandProfile, speaker_colours
from ai.editor.pro_edit.caption_legibility import (
    LEGIBILITY_VERSION,
    Legibility,
    LegibilityDecision,
    TextAppearance,
    assess,
    decision_key,
)
from ai.editor.pro_edit.caption_placement import (
    PageGeometry,
    PlacementEvidence,
    PlacementReport,
    solve_placement,
)
from ai.editor.pro_edit.caption_platform import GENERIC, PlatformSafeZoneProfile
from ai.editor.pro_edit.caption_primitives import (
    LEGIBILITY_PLATE_MAX_RATIO,
    PageEffects,
    PageFacts,
    PrimitiveId,
    PrimitiveLog,
    V31_PRIMITIVES,
    library_manifest,
    resolve_page_effects,
)
from ai.editor.pro_edit.errors import ProEditError
from ai.editor.pro_edit.font_metrics import FontMetrics, load_metrics, select_shaper
from ai.editor.pro_edit.schema import CaptionStyle, EditPlan, StoryRole

CAPTION_PRESENTATION_VERSION = 5
CAPTION_STYLE_PACK = "mimir_caption_v1"
CAPTION_STYLE_PACK_VERSION = 2


class CaptionPresentationError(ProEditError):
    """Presentation could not be built safely; the baseline ASS is used."""


# ============================================================
# TRUTH: token references (never mutated)
# ============================================================

_MASK_FLAGS = ("masked_unknown", "uncertainty_mask", "caption_uncertain", "uncertain")
_MASK_SOURCES = ("uncertainty_mask",)


@dataclass(frozen=True)
class CaptionTokenRef:
    word_id: int              # index in the authoritative speaker profile
    text: str
    start: float              # exact final clip clock (as captions displays it)
    end: float
    speaker_raw: str          # raw diarization id (truth, unchanged)
    speaker_role: str         # DISPLAY lane role, as the caption renderer lays it out
    speaker_label: str        # DISPLAY label: printable (human-confirmed) name or ""
    normalized: str
    uncertain: bool = False   # truth-side uncertainty mask ("???"): shown, never emphasized
    speaker_color: str = ""   # DISPLAY colour of the voice: "" plain, "A"/"B"/"C", "neutral"

    @property
    def lane(self) -> str:
        return lane_for_role(self.speaker_role)


def lane_for_role(role: str) -> str:
    return role if role in ("secondary", "tertiary") else "main"


def _is_uncertainty_mask(raw: Mapping[str, Any], text: str) -> bool:
    if any(raw.get(flag) is True for flag in _MASK_FLAGS):
        return True
    if str(raw.get("selected_source") or raw.get("text_source") or "") in _MASK_SOURCES:
        return True
    stripped = text.strip()
    return "?" in stripped and all(ch in "?¿.…" for ch in stripped)


def edited_duration(profile: Mapping[str, Any] | None, clip_timeline: Mapping[str, Any]) -> float:
    """Caption clock length, same rule as captions.create_ass_for_clip."""
    profile_duration = 0.0
    if profile and str(profile.get("status", "")) == "ok":
        try:
            profile_duration = float(profile.get("clip_duration", 0.0) or 0.0)
        except (TypeError, ValueError):
            profile_duration = 0.0
    if profile_duration > 0:
        return profile_duration
    edited = clip_timeline.get("edited", {}) if isinstance(clip_timeline.get("edited"), Mapping) else {}
    try:
        return float(edited.get("estimated_duration", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def display_metadata(profile: Mapping[str, Any] | None, duration: float) -> list[tuple[str, str, str]] | None:
    """(speaker_role, speaker_label, speaker_color) of every displayed word under
    the CURRENT caption renderer's speaker policy, in ``captions._profile_edited_words`` order.

    Captions V24 owns WHO is displayable and on which lane: only human-confirmed
    names are printed (``_trusted_human_display_map``) and the secondary lane
    exists only inside measured speech overlap (``_prepare_adaptive_render_words``).
    The same functions are called here, so the presentation can never re-introduce
    permanent A/B lanes or unconfirmed labels. ``None``: the caption engine
    predates that policy and displays the raw profile metadata.
    """
    prepare = getattr(caption_truth, "_prepare_adaptive_render_words", None)
    trusted = getattr(caption_truth, "_trusted_human_display_map", None)
    if not callable(prepare) or not callable(trusted):
        return None
    source = dict(profile) if profile else None
    words = caption_truth._profile_edited_words(source, duration)
    prepared, _windows = prepare(words, speaker_profile=source, trusted_display_map=trusted(source))
    return [(str(word.get("speaker_role", "main") or "main"), str(word.get("speaker_label", "")),
             str(word.get("speaker_color", "") or "")) for word in prepared]


def strict_lane_timing(profile: Mapping[str, Any] | None) -> bool:
    """Captions V24 diarization hard-failure mode: pages in one lane never overlap."""
    hard_failure = getattr(caption_truth, "_speaker_render_hard_failure", None)
    return bool(hard_failure(dict(profile) if profile else None)) if callable(hard_failure) else False


def presentation_tokens(profile: Mapping[str, Any] | None, duration: float) -> tuple[CaptionTokenRef, ...]:
    """The exact word list the baseline caption renderer displays, with ids.

    Mirrors ``captions._profile_edited_words`` (same filters, clamp, minimum
    event floor, rounding and ordering) and keeps the profile index as id.
    Lane role and label are the renderer's DISPLAY metadata (``display_metadata``).
    ``verify_token_parity`` proves equality at runtime.
    """
    if not profile or str(profile.get("status", "")) != "ok":
        return ()
    if str(profile.get("timing_basis", "")).strip() not in {"edited_clip", "exact_final_48k_audio", "exact_final_short_audio"}:
        return ()
    rows: list[CaptionTokenRef] = []
    for index, raw in enumerate(profile.get("words", []) or []):
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("word", "")).strip()
        if not text:
            continue
        try:
            start = caption_truth.clamp(float(raw.get("edited_start", 0.0)), 0.0, duration)
            end = caption_truth.clamp(float(raw.get("edited_end", start)), 0.0, duration)
            float(raw.get("speaker_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if end - start < caption_truth.MIN_EVENT_DURATION:
            end = min(duration, start + caption_truth.MIN_EVENT_DURATION)
        if end <= start:
            continue
        rows.append(CaptionTokenRef(
            word_id=index,
            text=text,
            start=caption_truth.round_time(start),
            end=caption_truth.round_time(end),
            speaker_raw=str(raw.get("speaker_raw", "")),
            speaker_role=str(raw.get("speaker_role", "main")) or "main",
            speaker_label=str(raw.get("speaker_label", "")),
            normalized=caption_truth.normalize_word(text),
            uncertain=_is_uncertainty_mask(raw, text),
            speaker_color=_speaker_color(raw),
        ))
    rows.sort(key=lambda t: (t.start, t.end))
    display = display_metadata(profile, duration)
    if display is not None:
        if len(display) != len(rows):
            raise CaptionPresentationError(
                f"display metadata covers {len(display)} words, presentation has {len(rows)}")
        rows = [dataclasses.replace(row, speaker_role=role, speaker_label=label, speaker_color=color)
                for row, (role, label, color) in zip(rows, display)]
    return tuple(rows)


def _speaker_color(raw: Mapping[str, Any]) -> str:
    reader = getattr(caption_truth, "speaker_color", None)
    return reader(dict(raw)) if callable(reader) else ""


def verify_token_parity(tokens: Sequence[CaptionTokenRef], profile: Mapping[str, Any] | None,
                        duration: float) -> None:
    """Presentation words == the words the existing caption renderer shows:
    truth (text, start, end, raw speaker) from ``_profile_edited_words`` and the
    renderer's display lane/label policy (``display_metadata``)."""
    baseline = caption_truth._profile_edited_words(dict(profile) if profile else None, duration)
    ours = [(t.text, t.start, t.end, t.speaker_raw) for t in tokens]
    theirs = [(w["word"], w["edited_start"], w["edited_end"], w["speaker_raw"]) for w in baseline]
    if ours != theirs:
        raise CaptionPresentationError(
            f"presentation tokens differ from caption truth ({len(ours)} vs {len(theirs)} words)")
    display = display_metadata(profile, duration)
    expected = display if display is not None else [
        (w["speaker_role"], w["speaker_label"], str(w.get("speaker_color", "") or "")) for w in baseline]
    if [(t.speaker_role, t.speaker_label, t.speaker_color) for t in tokens] != list(expected):
        raise CaptionPresentationError(
            "presentation speaker lanes/labels/colours differ from the caption renderer policy")


# ============================================================
# LAYOUT PROFILES (aspect ratio -> geometry ratios)
# ============================================================

class LayoutProfileName(str, Enum):
    VERTICAL = "vertical"
    LANDSCAPE = "landscape"
    SQUARE = "square"
    GENERIC_PRESERVE = "generic_preserve"


@dataclass(frozen=True)
class CaptionLayoutProfile:
    name: LayoutProfileName
    font_ratio: float           # font px / reference dimension
    font_reference: str         # "width" | "height" | "min"
    max_width_ratio: float      # safe text width / frame width
    bottom_ratio: float         # main lane bottom / frame height
    max_lines: int = 2
    line_spacing: float = 1.04  # line advance / libass line height
    lane_gap_ratio: float = 0.5
    outline_ratio: float = 6.0 / 74.0
    shadow_ratio: float = 2.0 / 74.0


# Vertical reproduces the baseline MIMIR geometry at 1080x1920: 74 px Arial
# bold, main lane bottom at 1920-530 px, ~70 px side margins.
LAYOUT_PROFILES: Mapping[LayoutProfileName, CaptionLayoutProfile] = {
    LayoutProfileName.VERTICAL: CaptionLayoutProfile(
        LayoutProfileName.VERTICAL, 74.0 / 1080.0, "width", 0.86, 1390.0 / 1920.0),
    LayoutProfileName.LANDSCAPE: CaptionLayoutProfile(
        LayoutProfileName.LANDSCAPE, 0.058, "height", 0.72, 0.915),
    LayoutProfileName.SQUARE: CaptionLayoutProfile(
        LayoutProfileName.SQUARE, 0.056, "min", 0.86, 0.87),
    LayoutProfileName.GENERIC_PRESERVE: CaptionLayoutProfile(
        LayoutProfileName.GENERIC_PRESERVE, 0.060, "min", 0.84, 0.85),
}


def select_layout_profile(width: int, height: int) -> CaptionLayoutProfile:
    if width <= 0 or height <= 0:
        raise CaptionPresentationError(f"invalid output size {width}x{height}")
    aspect = width / height
    if aspect <= 0.62:
        return LAYOUT_PROFILES[LayoutProfileName.VERTICAL]
    if 0.9 <= aspect <= 1.1:
        return LAYOUT_PROFILES[LayoutProfileName.SQUARE]
    if aspect >= 1.55:
        return LAYOUT_PROFILES[LayoutProfileName.LANDSCAPE]
    return LAYOUT_PROFILES[LayoutProfileName.GENERIC_PRESERVE]


@dataclass(frozen=True)
class LayoutGeometry:
    profile: CaptionLayoutProfile
    width: int
    height: int
    font_px: float
    max_line_px: float
    main_bottom: float
    outline_px: float
    shadow_px: float

    @classmethod
    def for_output(cls, width: int, height: int) -> "LayoutGeometry":
        profile = select_layout_profile(width, height)
        reference = {"width": width, "height": height, "min": min(width, height)}[profile.font_reference]
        font = max(12.0, round(profile.font_ratio * reference, 1))
        return cls(profile=profile, width=int(width), height=int(height), font_px=font,
                   max_line_px=profile.max_width_ratio * width,
                   main_bottom=round(profile.bottom_ratio * height),
                   outline_px=round(profile.outline_ratio * font, 1),
                   shadow_px=round(profile.shadow_ratio * font, 1))


# ============================================================
# STYLE SYSTEM (declarative definitions -> resolver -> physical values)
# ============================================================

def _colour(value: str) -> str:
    """ASS &HAABBGGRR -> &HBBGGRR& for override tags."""
    digits = value.strip().upper().lstrip("&H").rstrip("&")
    return f"&H{digits[-6:]}&"


@dataclass(frozen=True)
class LanePalette:
    """Speaker-lane colours (speaker truth decides the lane, never the planner)."""

    style_name: str
    primary: str      # style PrimaryColour (&HAABBGGRR)
    inactive: str     # override colours (&HBBGGRR&)
    active: str
    highlight: str

    def role(self, name: str) -> str:
        return {"inactive": self.inactive, "active": self.active, "highlight": self.highlight}[name]


_MAIN_COLOURS = (caption_truth.BASE_TEXT_COLOR, _colour(caption_truth.INACTIVE_TEXT_COLOR),
                 _colour(caption_truth.ACTIVE_TEXT_COLOR), _colour(caption_truth.HIGHLIGHT_TEXT_COLOR))
PALETTES: Mapping[str, LanePalette] = {
    "main": LanePalette("MimirMain", *_MAIN_COLOURS),
    "secondary": LanePalette("MimirSecondary", caption_truth.SECONDARY_BASE_TEXT_COLOR,
                             _colour(caption_truth.SECONDARY_BASE_TEXT_COLOR),
                             _colour(caption_truth.SECONDARY_ACTIVE_TEXT_COLOR),
                             _colour(caption_truth.SECONDARY_HIGHLIGHT_TEXT_COLOR)),
    # Baseline renders a third speaker with the main palette; it gets its own
    # lane so simultaneous speech never overprints the main lane.
    "tertiary": LanePalette("MimirTertiary", *_MAIN_COLOURS),
}
LANE_LAYERS = {"main": 0, "secondary": 1, "tertiary": 2}
_LANE_STYLE_NAMES = {"main": "MimirMain", "secondary": "MimirSecondary", "tertiary": "MimirTertiary"}


def palettes_for(brand: BrandProfile) -> Mapping[str, LanePalette]:
    """Lane palettes of a brand profile (``mimir_default`` == PALETTES).

    A lane palette gives a page its ASS style line (where it sits); the words'
    colours come from the voice (``speaker_palettes_for``)."""
    return {lane: LanePalette(_LANE_STYLE_NAMES[lane], colours.style_primary, colours.text, colours.active,
                              colours.emphasis) for lane, colours in brand.lanes.items()}


def speaker_palettes_for(brand: BrandProfile) -> Mapping[str, LanePalette]:
    """Speaker colour key ("", "A", "B", "C", "neutral") -> colours of that voice."""
    return {key: LanePalette("", colours.style_primary, colours.text, colours.active, colours.emphasis)
            for key, colours in speaker_colours(brand).items()}


SPEAKER_PALETTES: Mapping[str, LanePalette] = speaker_palettes_for(MIMIR_DEFAULT)
PALETTE_ROLES = ("inactive", "active", "highlight")


@dataclass(frozen=True)
class CaptionStyleDefinition:
    """Engine parameters of one semantic caption style. Never planner output."""

    name: CaptionStyle
    base_font_scale: float = 1.0
    active_font_scale: float = 1.0          # geometry law: always 1.0 (see module docstring)
    emphasis_font_scale: float = 1.0        # static, reserved in the layout from the first frame
    outline_scale: float = 1.0
    active_outline_scale: float = 1.0       # accent on the active word (outline is not laid out)
    emphasis_outline_scale: float = 1.0     # accent on an emphasized active word
    animation_attack_ms: int = 50
    animation_release_ms: int = 0
    active_color: str = "active"            # palette role
    emphasis_color: str = "highlight"       # palette role
    max_lines: int = 2
    safe_width_ratio: float = 1.0           # of the layout profile's safe width
    max_pages_per_directive: int = 10_000

    def __post_init__(self) -> None:
        problems = []
        if self.active_font_scale != 1.0:
            problems.append("active_font_scale must be 1.0 (an active word never changes width)")
        if not 1.0 <= self.base_font_scale <= 1.15:
            problems.append("base_font_scale outside [1, 1.15]")
        if not 1.0 <= self.emphasis_font_scale <= 1.2:
            problems.append("emphasis_font_scale outside [1, 1.2]")
        for name in ("outline_scale", "active_outline_scale", "emphasis_outline_scale"):
            if not 1.0 <= getattr(self, name) <= 2.0:
                problems.append(f"{name} outside [1, 2]")
        if not 0 <= self.animation_attack_ms <= 150 or not 0 <= self.animation_release_ms <= 300:
            problems.append("animation must stay short (attack <= 150 ms, release <= 300 ms)")
        if self.active_color not in PALETTE_ROLES or self.emphasis_color not in PALETTE_ROLES:
            problems.append("colours must name a palette role")
        if self.max_lines not in (1, 2) or not 0.5 <= self.safe_width_ratio <= 1.0:
            problems.append("max_lines in {1,2} and safe_width_ratio in [0.5, 1]")
        if problems:
            raise ValueError(f"caption style {self.name.value}: " + "; ".join(problems))

    @property
    def rank(self) -> int:
        return STYLE_RANK[self.name]


STYLE_RANK = {CaptionStyle.DEFAULT: 0, CaptionStyle.EMPHASIS: 1, CaptionStyle.IMPACT: 2}
STYLE_DEFINITIONS: Mapping[CaptionStyle, CaptionStyleDefinition] = {
    # Stable, readable active-word highlight: colour only.
    CaptionStyle.DEFAULT: CaptionStyleDefinition(CaptionStyle.DEFAULT),
    # Slightly stronger active word; selected words larger (static) with a short outline accent.
    CaptionStyle.EMPHASIS: CaptionStyleDefinition(
        CaptionStyle.EMPHASIS, emphasis_font_scale=1.08, active_outline_scale=1.15, emphasis_outline_scale=1.35,
        animation_attack_ms=60, animation_release_ms=140, max_pages_per_directive=3),
    # Bounded payoff punctuation: larger page, stronger outline, one accent per emphasized word.
    CaptionStyle.IMPACT: CaptionStyleDefinition(
        CaptionStyle.IMPACT, base_font_scale=1.08, emphasis_font_scale=1.12, outline_scale=1.15,
        active_outline_scale=1.2, emphasis_outline_scale=1.6, animation_attack_ms=70, animation_release_ms=180,
        max_pages_per_directive=2),
}


@dataclass(frozen=True)
class ResolvedCaptionStyle:
    """Physical parameters for one page (lane + style + layout profile)."""

    name: CaptionStyle
    lane: str
    ass_style: str
    font_px: float
    outline_px: float
    active_outline_px: float
    emphasis_outline_px: float
    emphasis_scale: float
    attack_ms: int
    release_ms: int
    inactive_colour: str
    active_colour: str
    emphasis_colour: str
    label_colour: str
    max_lines: int
    max_line_px: float

    def to_dict(self) -> dict[str, Any]:
        return {"style": self.name.value, "lane": self.lane, "font_px": self.font_px, "outline_px": self.outline_px,
                "active_outline_px": self.active_outline_px, "emphasis_outline_px": self.emphasis_outline_px,
                "emphasis_scale": self.emphasis_scale, "attack_ms": self.attack_ms, "release_ms": self.release_ms,
                "max_lines": self.max_lines, "max_line_px": round(self.max_line_px, 1)}


def resolve_caption_style(style: CaptionStyle, lane: str, geometry: LayoutGeometry, *,
                          fit_scale: float = 1.0, palettes: Mapping[str, LanePalette] | None = None,
                          outline_boost: float = 1.0, colour: str = "",
                          speakers: Mapping[str, LanePalette] | None = None) -> ResolvedCaptionStyle:
    """``lane`` picks the ASS style (position), ``colour`` the voice's colours."""
    definition = STYLE_DEFINITIONS[style]
    lane_palette = (palettes or PALETTES)[lane]
    voices = speakers or SPEAKER_PALETTES
    palette = voices.get(colour) or voices[""]
    outline = round(geometry.outline_px * definition.outline_scale * fit_scale * outline_boost, 1)
    return ResolvedCaptionStyle(
        name=style, lane=lane, ass_style=lane_palette.style_name,
        font_px=round(geometry.font_px * definition.base_font_scale * fit_scale, 1),
        outline_px=outline,
        active_outline_px=round(outline * definition.active_outline_scale, 1),
        emphasis_outline_px=round(outline * definition.emphasis_outline_scale, 1),
        emphasis_scale=definition.emphasis_font_scale,
        attack_ms=definition.animation_attack_ms, release_ms=definition.animation_release_ms,
        inactive_colour=palette.inactive, active_colour=palette.role(definition.active_color),
        emphasis_colour=palette.role(definition.emphasis_color), label_colour=palette.active,
        max_lines=min(definition.max_lines, geometry.profile.max_lines),
        max_line_px=geometry.max_line_px * definition.safe_width_ratio,
    )


# Sparsity: emphasis must stay rare to keep meaning.
MAX_EMPHASIS_PER_SHORT_PAGE = 1       # pages of <= 4 words
MAX_EMPHASIS_PER_LONG_PAGE = 2
MAX_EMPHASIS_RATIO = 0.15             # of all words in the clip
MAX_IMPACT_PAGE_RATIO = 0.25          # of all pages
TERRA_MATCH_TOLERANCE_S = 0.30        # timeline marker (estimated paced time) vs final word clock
DIRECTIVE_WINDOW_TOLERANCE_S = 0.25   # same tolerance as the plan validator


# ============================================================
# PRESENTATION MODEL
# ============================================================

@dataclass(frozen=True)
class CaptionPresentationDirective:
    """Validated planner intent for a window (PACED_CLIP seconds)."""

    event_id: str
    start: float
    end: float
    style: CaptionStyle
    emphasis_word_ids: tuple[int, ...] = ()
    emphasis_reasons: tuple[tuple[int, str], ...] = ()     # planner-given reasons (optional)

    @classmethod
    def from_plan(cls, plan: EditPlan | None) -> tuple["CaptionPresentationDirective", ...]:
        if plan is None:
            return ()
        rows = [cls(e.event_id, e.start, e.end, e.caption_style, tuple(e.emphasis_word_ids),
                    tuple((wid, reason.value) for wid, reason in getattr(e, "emphasis_reasons", ())))
                for e in plan.events if e.caption_style is not CaptionStyle.DEFAULT or e.emphasis_word_ids]
        return tuple(sorted(rows, key=lambda d: (d.start, d.end, d.event_id)))


@dataclass(frozen=True)
class CaptionLine:
    word_ids: tuple[int, ...]
    width_px: float
    x: float
    y: float
    has_label: bool = False


@dataclass(frozen=True)
class CaptionPage:
    """A resolved caption page: fixed geometry for its whole lifetime."""

    page_id: int
    lane: str
    tokens: tuple[CaptionTokenRef, ...]
    lines: tuple[CaptionLine, ...]
    style: CaptionStyle
    emphasis_ids: frozenset[int]
    start: float
    end: float
    font_px: float
    fit_scale: float
    label: str
    directive_id: str = ""
    placement_zone: str = "bottom"
    outline_boost: float = 1.0                 # legibility (1.0 = brand look)
    shadow_px: float | None = None             # legibility drop shadow (None = style shadow)
    legibility: str = ""                       # verdict ("" = not analysed)
    plate_alpha: int | None = None             # legibility plate alpha (None = brand backplate alpha)

    @property
    def speech_end(self) -> float:
        return max(t.end for t in self.tokens)

    @property
    def speaker_role(self) -> str:
        return self.tokens[0].speaker_role

    @property
    def speaker_color(self) -> str:
        return self.tokens[0].speaker_color


ResolvedCaptionPage = CaptionPage


@dataclass
class EmphasisReport:
    requested: list[dict[str, Any]] = field(default_factory=list)
    applied: list[int] = field(default_factory=list)
    degraded: list[dict[str, Any]] = field(default_factory=list)
    terra_markers: int = 0
    terra_matched: int = 0

    @property
    def invalid_ignored(self) -> int:
        return sum(1 for d in self.degraded
                   if d["reason"] in ("unknown_word_id", "outside_event_window", "uncertainty_mask"))


@dataclass(frozen=True)
class CaptionPresentation:
    tokens: tuple[CaptionTokenRef, ...]
    pages: tuple[CaptionPage, ...]
    geometry: LayoutGeometry
    metrics_font: FontMetrics
    directives: tuple[CaptionPresentationDirective, ...]
    emphasis: EmphasisReport
    ass_text: str
    lane_bottoms: Mapping[str, float]
    settings: "RenderSettings | None" = None
    placement: PlacementReport | None = None
    primitive_log: PrimitiveLog | None = None
    reasons: Mapping[int, tuple[str, str]] = field(default_factory=dict)     # word_id -> (reason, source)
    brand: BrandProfile = MIMIR_DEFAULT
    platform: PlatformSafeZoneProfile = GENERIC
    shaper_note: str = "naive"
    legibility: tuple[LegibilityDecision, ...] = ()
    energy_overrides: Mapping[int, str] = field(default_factory=dict)     # page_id -> degradation

    @property
    def ass_signature(self) -> str:
        return hashlib.sha256(self.ass_text.encode("utf-8")).hexdigest()

    @property
    def first_start(self) -> float | None:
        return min((p.start for p in self.pages), default=None)

    def metrics(self) -> dict[str, Any]:
        return presentation_metrics(self)

    def manifest(self, *, caption_signature: str = "", page_limit: int = 600) -> dict[str, Any]:
        return build_manifest(self, caption_signature=caption_signature, page_limit=page_limit)


# ============================================================
# EMPHASIS (plan ids + Terra timeline markers, sparse, never drops a word)
# ============================================================

def terra_markers(clip_timeline: Mapping[str, Any]) -> list[tuple[str, float]]:
    """caption_emphasis markers in PACED_CLIP time (``edited_time``).

    The baseline compares the marker's RAW ``source_time`` to the final word
    clock, which only agrees before the first cut; the paced ``edited_time``
    is the matching domain.
    """
    rows: list[tuple[str, float]] = []
    for event in clip_timeline.get("events", []) or []:
        if not isinstance(event, Mapping) or event.get("type") != "caption_emphasis":
            continue
        try:
            when = float(event.get("edited_time"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(when):
            continue
        word = caption_truth.normalize_word(str(event.get("word", "")))
        if word:
            rows.append((word, when))
    return rows


def emphasis_candidates(
    tokens: Sequence[CaptionTokenRef],
    directives: Sequence[CaptionPresentationDirective],
    markers: Sequence[tuple[str, float]],
    report: EmphasisReport,
) -> dict[int, tuple[int, int]]:
    """word_id -> priority key (lower = stronger). Invalid requests degrade.

    A requested word must exist in the displayed truth, lie inside its event
    window and not be an uncertainty mask. Rejected ids keep normal styling.
    """
    by_id = {t.word_id: t for t in tokens}
    chosen: dict[int, tuple[int, int]] = {}
    order = 0
    for directive in directives:
        for word_id in directive.emphasis_word_ids:
            report.requested.append({"event_id": directive.event_id, "word_id": word_id})
            token = by_id.get(word_id)
            reason = None
            if token is None:
                reason = "unknown_word_id"
            elif (token.end < directive.start - DIRECTIVE_WINDOW_TOLERANCE_S
                  or token.start > directive.end + DIRECTIVE_WINDOW_TOLERANCE_S):
                reason = "outside_event_window"
            elif token.uncertain:
                reason = "uncertainty_mask"
            if reason is not None:
                report.degraded.append({"word_id": word_id, "event_id": directive.event_id, "reason": reason})
                continue
            if word_id not in chosen:
                chosen[word_id] = (0, order)
                order += 1
    report.terra_markers = len(markers)
    used: set[int] = set()
    for word, when in markers:
        best = None
        for token in tokens:
            if token.normalized != word or token.word_id in used or token.uncertain:
                continue
            delta = abs(token.start - when)
            if delta <= TERRA_MATCH_TOLERANCE_S and (best is None or delta < best[0]):
                best = (delta, token.word_id)
        if best is None:
            continue
        used.add(best[1])
        report.terra_matched += 1
        if best[1] not in chosen:
            chosen[best[1]] = (1, order)
            order += 1
    return chosen


# ============================================================
# MEASUREMENT + LINE BREAKING
# ============================================================

_FUNCTION_WORDS = frozenset({
    # English articles / prepositions / conjunctions / determiners
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "and", "or", "but", "with", "from", "by", "into",
    "my", "your", "his", "her", "our", "their", "its", "this", "that", "these", "those", "if", "than", "as",
    # Turkish modifiers / conjunctions that bind to the FOLLOWING word
    "ve", "ile", "veya", "ya", "ama", "fakat", "bir", "bu", "şu", "çok", "en", "daha", "her", "hiç",
})
_CLAUSE_END = (",", "—", "–")
_WORD_JOINER = "\u2060"


def _visible_text(text: str) -> str:
    cleaned = "".join(" " if unicodedata.category(ch) in ("Cc", "Zl", "Zp") else ch for ch in str(text))
    return " ".join(cleaned.split()) or str(text)


def display_text(text: str) -> str:
    """Presentation-safe ASS text; the truth text itself is never changed.

    * control / line-separator characters -> space (an ASS event is one line);
    * ``{`` / ``}`` -> ``\\{`` / ``\\}`` (libass literal braces);
    * a backslash is followed by U+2060 WORD JOINER (zero width) so it can
      never form ``\\N`` / ``\\n`` / ``\\h``: the baseline ``\\\\`` escape turns
      ``C:\\New`` into a line break (verified against libass).
    """
    visible = _visible_text(text)
    return visible.replace("\\", "\\" + _WORD_JOINER).replace("{", "\\{").replace("}", "\\}")


@dataclass(frozen=True)
class _Measure:
    metrics: FontMetrics
    font_px: float
    shaper: Any = None           # font_metrics shaper; None = naive advances (V3.1)

    def word(self, text: str, scale: float = 1.0) -> float:
        if self.shaper is not None:
            return self.shaper.width(_visible_text(text), self.font_px, scale_x=scale)
        return self.metrics.text_width(_visible_text(text), self.font_px, scale_x=scale)

    def space(self, scale: float = 1.0) -> float:
        if self.shaper is not None:
            return self.shaper.width(" ", self.font_px, scale_x=scale)
        return self.metrics.text_width(" ", self.font_px, scale_x=scale)


def _label_text(label: str) -> str:
    return f"{label}:" if label else ""


def line_width(texts: Sequence[str], scales: Sequence[float], measure: _Measure, label: str = "") -> float:
    """Width of one rendered line: optional ``LABEL:`` prefix + words.

    The space before word j is drawn with word j's override (static scale).
    """
    width = measure.word(_label_text(label)) if label else 0.0
    for index, (text, scale) in enumerate(zip(texts, scales)):
        if index > 0 or label:
            width += measure.space(scale)
        width += measure.word(text, scale)
    return width


def break_lines(texts: Sequence[str], scales: Sequence[float], measure: _Measure, limit: float, *,
                max_lines: int, label: str = "", emphasized: Sequence[bool] | None = None,
                bound: Sequence[bool] | None = None) -> tuple[tuple[int, ...], float] | None:
    """Best line split (end indices) for one page, or None if it cannot fit.

    Single line when it fits. Two lines: balanced widths, no single-word line
    when the page has 3+ words, never break after a word that binds to the
    next (function word, or inside a verified multi-word name: ``bound``),
    prefer breaking after a clause mark or after an emphasized word.
    """
    count = len(texts)
    whole = line_width(texts, scales, measure, label)
    if whole <= limit:
        return (count,), 0.0
    if max_lines < 2 or count < 2:
        return None
    best: tuple[float, int] | None = None
    for split in range(1, count):
        top = line_width(texts[:split], scales[:split], measure, label)
        bottom = line_width(texts[split:], scales[split:], measure)
        if top > limit or bottom > limit:
            continue
        cost = abs(top - bottom) / limit
        if count >= 3 and (split == 1 or split == count - 1):
            cost += 0.6
        if caption_truth.normalize_word(texts[split - 1]) in _FUNCTION_WORDS:
            cost += 0.3
        if bound is not None and bound[split - 1]:
            cost += NAME_SPLIT_COST
        if texts[split - 1].rstrip().endswith(_CLAUSE_END):
            cost -= 0.2
        if emphasized is not None and emphasized[split - 1]:
            cost -= 0.1
        if top > bottom:
            cost += 0.05
        if best is None or cost < best[0]:
            best = (cost, split)
    if best is None:
        return None
    return (best[1], count), best[0]


# ============================================================
# PAGINATION (hard breaks + dynamic programming over soft breaks)
# ============================================================

MAX_PAGE_WORDS = 6
MAX_LINE_WORDS = 4
PAGE_COST = 1.0
TWO_LINE_COST = 0.9
SHORT_PAGE_S = 0.75
SHORT_PAGE_WEIGHT = 3.0
LONG_PAGE_S = 2.6
LONG_PAGE_WEIGHT = 1.5
ORPHAN_COST = 1.6
DANGLING_FUNCTION_COST = 1.2
CLAUSE_BONUS = 0.7
PAUSE_BONUS_PER_S = 3.0
EXTRA_WORD_COST = 0.25
STYLE_BOUNDARY_BONUS = 0.8
EMPHASIS_BREAK_BONUS = 0.3
TERMINAL_FLASH_COST = 1.5
TERMINAL_FLASH_S = 0.2          # a last word shown for less than this is unreadable
MIN_FIT_SCALE = 0.3
NAME_SPLIT_COST = 2.0           # splitting a verified multi-word name across lines/pages


def _label_changed(previous: CaptionTokenRef, token: CaptionTokenRef) -> bool:
    """Captions V24 group rule: only a visible (human-confirmed) name transition
    is a speaker break; raw A/B diarization is metadata, not a layout command."""
    before, after = previous.speaker_label.strip(), token.speaker_label.strip()
    return bool(before or after) and before.casefold() != after.casefold()


def split_runs(tokens: Sequence[CaptionTokenRef]) -> list[list[CaptionTokenRef]]:
    """Hard breaks, same rules as the baseline groups (captions V25): a visible
    label change, a speaker-colour change, a gap of GROUP_BREAK_GAP or more,
    sentence punctuation. Each display lane is grouped on its own clock, like
    the baseline's lanes."""
    by_lane: dict[str, list[CaptionTokenRef]] = {}
    for token in tokens:
        by_lane.setdefault(token.lane, []).append(token)
    runs: list[list[CaptionTokenRef]] = []
    for lane_tokens in by_lane.values():
        current: list[CaptionTokenRef] = []
        for token in lane_tokens:
            if current:
                previous = current[-1]
                if (_label_changed(previous, token) or previous.speaker_color != token.speaker_color
                        or token.start - previous.end >= caption_truth.GROUP_BREAK_GAP):
                    runs.append(current)
                    current = []
            current.append(token)
            if caption_truth.should_break_after_word(token.text):
                runs.append(current)
                current = []
        if current:
            runs.append(current)
    runs.sort(key=lambda run: (run[0].start, run[0].end, LANE_LAYERS.get(run[0].lane, 0)))
    return runs


@dataclass(frozen=True)
class _Draft:
    tokens: tuple[CaptionTokenRef, ...]
    breaks: tuple[int, ...]
    fit_scale: float
    font_scale: float


def _style_at(directives: Sequence[CaptionPresentationDirective], start: float, end: float
              ) -> CaptionPresentationDirective | None:
    best = None
    best_key = (0.0, -1)
    for directive in directives:
        overlap = min(end, directive.end) - max(start, directive.start)
        if overlap <= 0 and not (directive.start <= start <= directive.end):
            continue
        key = (round(max(overlap, 0.0), 6), STYLE_RANK[directive.style])
        if key > best_key:
            best, best_key = directive, key
    return best


def paginate_run(
    run: Sequence[CaptionTokenRef],
    geometry: LayoutGeometry,
    metrics: FontMetrics,
    directives: Sequence[CaptionPresentationDirective],
    emphasized: Mapping[int, Any],
    shaper: Any = None,
    default_style: CaptionStyle = CaptionStyle.DEFAULT,
    allow_emphasis_scale: bool = True,
    bound_after: frozenset[int] = frozenset(),
) -> list[_Draft]:
    """Pages for one hard run. Costs come from rendered width, reading density
    (page duration), phrase cohesion, pauses, punctuation, emphasis and style
    boundaries; the terminal-readability rule is enforced across page breaks.
    ``bound_after``: word ids that bind to the next word (inside a verified
    multi-word name), so neither a line nor a page breaks there."""
    n = len(run)
    label = run[0].speaker_label.strip()
    token_style: list[CaptionStyleDefinition] = []
    for token in run:
        directive = _style_at(directives, token.start, token.end)
        token_style.append(STYLE_DEFINITIONS[directive.style if directive else default_style])

    def boundary_cost(i: int) -> float:
        """Cost of ending a page after run[i-1] (0 < i < n)."""
        previous, following = run[i - 1], run[i]
        cost = 0.0
        if caption_truth.normalize_word(previous.text) in _FUNCTION_WORDS:
            cost += DANGLING_FUNCTION_COST
        if previous.word_id in bound_after:
            cost += NAME_SPLIT_COST
        if previous.text.rstrip().endswith(_CLAUSE_END):
            cost -= CLAUSE_BONUS
        cost -= PAUSE_BONUS_PER_S * min(max(following.start - previous.end, 0.0), 0.4)
        if token_style[i - 1].name is not token_style[i].name:
            cost -= STYLE_BOUNDARY_BONUS
        if previous.word_id in emphasized:
            cost -= EMPHASIS_BREAK_BONUS
        if following.start - previous.start < TERMINAL_FLASH_S:
            # The page's last word would flash: the next page (same lane) caps its hold.
            cost += TERMINAL_FLASH_COST
        return cost

    def layout(j: int, i: int) -> tuple[tuple[int, ...], float, float, float] | None:
        page = run[j:i]
        styles = token_style[j:i]
        font_scale = max(s.base_font_scale for s in styles)
        measure = _Measure(metrics, geometry.font_px * font_scale, shaper)
        max_lines = min(min(s.max_lines for s in styles), geometry.profile.max_lines)
        widest_outline = geometry.outline_px * max(s.outline_scale * max(s.active_outline_scale,
                                                                          s.emphasis_outline_scale)
                                                   for s in styles)
        limit = geometry.max_line_px * min(s.safe_width_ratio for s in styles) - 2.0 * widest_outline
        texts = [t.text for t in page]
        flags = [run[k].word_id in emphasized for k in range(j, i)]
        scales = [token_style[k].emphasis_font_scale if flags[k - j] and allow_emphasis_scale else 1.0
                  for k in range(j, i)]
        # Every page of a same-speaker run carries the run's label (baseline rule).
        bound = [run[k].word_id in bound_after for k in range(j, i)] if bound_after else None
        found = break_lines(texts, scales, measure, limit, max_lines=max_lines, label=label, emphasized=flags,
                            bound=bound)
        if found is None:
            if len(page) != 1:
                return None
            width = line_width(texts, scales, measure, label)
            fit = max(MIN_FIT_SCALE, min(1.0, limit / max(width, 1e-6)))
            return (1,), 0.0, fit, font_scale
        breaks, split_cost = found
        widths = [b - a for a, b in zip((0,) + breaks[:-1], breaks)]
        if max(widths) > MAX_LINE_WORDS:
            return None
        return breaks, split_cost, 1.0, font_scale

    best = [math.inf] * (n + 1)
    back: list[tuple[int, tuple[int, ...], float, float]] = [(0, (), 1.0, 1.0)] * (n + 1)
    best[0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(0, i - MAX_PAGE_WORDS), i):
            if best[j] == math.inf:
                continue
            found = layout(j, i)
            if found is None:
                continue
            breaks, split_cost, fit, font_scale = found
            count = i - j
            duration = run[i - 1].end - run[j].start
            cost = PAGE_COST + split_cost
            if len(breaks) > 1:
                cost += TWO_LINE_COST
            if duration < SHORT_PAGE_S:
                cost += (SHORT_PAGE_S - duration) * SHORT_PAGE_WEIGHT
            if duration > LONG_PAGE_S:
                cost += (duration - LONG_PAGE_S) * LONG_PAGE_WEIGHT
            if count == 1 and n >= 2:
                cost += ORPHAN_COST
            if count > 4:
                cost += EXTRA_WORD_COST * (count - 4)
            if fit < 1.0:
                cost += 2.0 * (1.0 - fit)
            if i < n:
                cost += boundary_cost(i)
            total = best[j] + cost
            if total < best[i] - 1e-12:
                best[i] = total
                back[i] = (j, breaks, fit, font_scale)
    if best[n] == math.inf:
        raise CaptionPresentationError("pagination found no layout (should be impossible)")
    drafts: list[_Draft] = []
    i = n
    while i > 0:
        j, breaks, fit, font_scale = back[i]
        drafts.append(_Draft(tuple(run[j:i]), breaks, fit, font_scale))
        i = j
    drafts.reverse()
    return drafts


# ============================================================
# BUILD
# ============================================================

def _page_end(page_tokens: Sequence[CaptionTokenRef], next_start: float | None, duration: float,
              strict_no_overlap: bool = False) -> float:
    """Display end: terminal readability floor + small-gap hold, capped at the
    next page in the SAME lane (baseline get_event_end semantics per page).

    ``strict_no_overlap`` (captions V24 diarization hard failure): the minimum
    event floor may never make two pages of one lane visible at once."""
    last = page_tokens[-1]
    speech_end = max(t.end for t in page_tokens)
    end = max(speech_end, last.start + caption_truth.MIN_TERMINAL_DISPLAY_DURATION)
    if next_start is not None and next_start > last.start:
        end = min(end, next_start)
        if 0.0 <= next_start - speech_end <= caption_truth.MAX_HOLD_GAP:
            end = max(end, next_start)
        if strict_no_overlap:
            return min(max(end, last.start + caption_truth.MIN_EVENT_DURATION), next_start, duration)
    return min(max(end, last.start + caption_truth.MIN_EVENT_DURATION), duration)


EMPHASIS_REASONS = ("generic", "name", "number", "payoff", "reaction", "contrast", "surprise")
LOCAL_REASONS = ("name", "number", "payoff", "reaction", "generic")     # never contrast / surprise


def name_tokens(verified_names: Iterable[str]) -> frozenset[str]:
    """Normalized words of verified participant names (speaker identity truth)."""
    found: set[str] = set()
    for name in verified_names:
        for part in str(name).split():
            normalized = caption_truth.normalize_word(part)
            if len(normalized) >= 2:
                found.add(normalized)
    return frozenset(found)


def bound_name_words(tokens: Sequence[CaptionTokenRef], verified_names: Iterable[str]) -> frozenset[int]:
    """Word ids that must stay with the next word: every word of a verified
    multi-word name except its last (the name is spoken as consecutive words)."""
    names = [[caption_truth.normalize_word(part) for part in str(name).split()] for name in verified_names]
    names = [parts for parts in names if len(parts) >= 2 and all(parts)]
    if not names:
        return frozenset()
    bound: set[int] = set()
    for lane in {t.lane for t in tokens}:
        ordered = [t for t in tokens if t.lane == lane]
        for index in range(len(ordered)):
            for parts in names:
                window = ordered[index:index + len(parts)]
                if len(window) == len(parts) and [t.normalized for t in window] == parts:
                    bound.update(t.word_id for t in window[:-1])
    return frozenset(bound)


def derive_reason(token: CaptionTokenRef, names: frozenset[str], spans: Sequence[Any]) -> str:
    """Safe local reason: verified name, digit-bearing token, payoff/reaction span; else generic."""
    if token.normalized in names:
        return "name"
    if any(ch.isdigit() for ch in token.text):
        return "number"
    for span in spans:
        role = getattr(span, "role", None)
        if role in (StoryRole.PAYOFF, StoryRole.REACTION) and span.start <= token.start < span.end:
            return role.value
    return "generic"


def _line_boxes_px(page: CaptionPage, geometry: LayoutGeometry, font: FontMetrics, style: "ResolvedCaptionStyle",
                   scale: float) -> tuple[tuple[float, float, float, float], ...]:
    border = max(style.outline_px, style.active_outline_px, style.emphasis_outline_px)
    height = font.line_height(style.font_px) * scale
    return tuple((line.x - line.width_px / 2.0 - border, line.y - height - border,
                  line.x + line.width_px / 2.0 + border, line.y + border + geometry.shadow_px) for line in page.lines)


def build_presentation(
    *,
    profile: Mapping[str, Any] | None,
    clip_timeline: Mapping[str, Any],
    plan: EditPlan | None,
    width: int,
    height: int,
    metrics: FontMetrics | None = None,
    brand: BrandProfile = MIMIR_DEFAULT,
    platform: PlatformSafeZoneProfile = GENERIC,
    evidence: PlacementEvidence | None = None,
    spans: Sequence[Any] = (),
    verified_names: Sequence[str] = (),
    shaper_mode: str = "naive",
    font_family: str | None = None,
    bold: bool = True,
    background: Any | None = None,
) -> CaptionPresentation:
    """Caption truth -> pages -> placement -> primitives -> ASS (see module docstring).

    Defaults (mimir_default brand, generic platform, no evidence, naive shaper)
    reproduce V3.1 exactly.
    """
    duration = edited_duration(profile, clip_timeline)
    if duration <= 0:
        raise CaptionPresentationError("caption clock duration unavailable")
    tokens = presentation_tokens(profile, duration)
    if not tokens:
        raise CaptionPresentationError("no final-clip caption words (baseline caption path kept)")
    verify_token_parity(tokens, profile, duration)
    platform = platform.for_output(width, height)
    base_geometry = LayoutGeometry.for_output(width, height)
    geometry = dataclasses.replace(
        base_geometry,
        max_line_px=min(base_geometry.max_line_px * brand.safe_width_ratio,
                        (1.0 - 2.0 * platform.minimum_edge_padding) * width),
        outline_px=round(base_geometry.outline_px * brand.outline_scale, 1),
        shadow_px=round(base_geometry.shadow_px * brand.shadow_scale, 1))
    font = metrics or load_metrics(brand.font_family, True)
    shaper, shaper_note = select_shaper(font, shaper_mode)
    active_shaper = None if shaper_note == "naive" or shaper_note.startswith("naive ") else shaper
    allow_scale = PrimitiveId.STATIC_EMPHASIS_SCALE in brand.allowed_primitives
    directives = CaptionPresentationDirective.from_plan(plan)
    report = EmphasisReport()
    candidates = emphasis_candidates(tokens, directives, terra_markers(clip_timeline), report)

    drafts: list[_Draft] = []
    bound_after = bound_name_words(tokens, verified_names)
    for run in split_runs(tokens):
        drafts.extend(paginate_run(run, geometry, font, directives, candidates, active_shaper,
                                   brand.default_style, allow_scale, bound_after=bound_after))

    # Page style (dominant directive), sparse IMPACT, sparse emphasis.
    styled: list[tuple[_Draft, CaptionStyle, str]] = []
    per_directive: dict[str, int] = {}
    impact_budget = max(1, math.floor(MAX_IMPACT_PAGE_RATIO * len(drafts)))
    for draft in drafts:
        directive = _style_at(directives, draft.tokens[0].start, max(t.end for t in draft.tokens))
        style = directive.style if directive else brand.default_style
        directive_id = directive.event_id if directive else ""
        if directive is not None and style is not CaptionStyle.DEFAULT:
            used = per_directive.get(directive.event_id, 0)
            if used >= STYLE_DEFINITIONS[style].max_pages_per_directive:
                style = CaptionStyle.EMPHASIS if style is CaptionStyle.IMPACT else CaptionStyle.DEFAULT
            per_directive[directive.event_id] = used + 1
        if style is CaptionStyle.IMPACT:
            if impact_budget <= 0:
                style = CaptionStyle.EMPHASIS
            else:
                impact_budget -= 1
        styled.append((draft, style, directive_id))

    global_budget = max(1, math.floor(MAX_EMPHASIS_RATIO * len(tokens)))
    page_emphasis: list[frozenset[int]] = []
    ranked = sorted(((key, word_id) for word_id, key in candidates.items()))
    allowed_global = {word_id for _key, word_id in ranked[:global_budget]}
    for _key, word_id in ranked[global_budget:]:
        report.degraded.append({"word_id": word_id, "reason": "clip_emphasis_budget"})
    for draft, _style, _directive_id in styled:
        ids = [t.word_id for t in draft.tokens if t.word_id in allowed_global]
        ids.sort(key=lambda wid: candidates[wid])
        cap = MAX_EMPHASIS_PER_SHORT_PAGE if len(draft.tokens) <= 4 else MAX_EMPHASIS_PER_LONG_PAGE
        for word_id in ids[cap:]:
            report.degraded.append({"word_id": word_id, "reason": "page_emphasis_budget"})
        page_emphasis.append(frozenset(ids[:cap]))
    report.applied = sorted(set().union(*page_emphasis)) if page_emphasis else []

    # Emphasis reasons: planner reason if given (and valid for that event), else safe local derivation.
    planner_reasons: dict[int, str] = {}
    for directive in directives:
        for word_id, reason in directive.emphasis_reasons:
            if word_id in directive.emphasis_word_ids and reason in EMPHASIS_REASONS:
                planner_reasons.setdefault(word_id, reason)
    names = name_tokens(verified_names)
    by_id = {t.word_id: t for t in tokens}
    reasons: dict[int, tuple[str, str]] = {}
    for word_id in report.applied:
        if word_id in planner_reasons:
            reasons[word_id] = (planner_reasons[word_id], "planner")
        else:
            reasons[word_id] = (derive_reason(by_id[word_id], names, spans), "derived")

    # Lane geometry: fixed per clip (never jumps between pages). Lanes stack
    # upward: main, secondary above the tallest main page, tertiary above that.
    def advance(font_px: float) -> float:
        return font.line_height(font_px) * geometry.profile.line_spacing

    def block(lane: str) -> float:
        return max((len(d.breaks) * advance(geometry.font_px * d.font_scale)
                    for d, _s, _i in styled if d.tokens[0].lane == lane), default=advance(geometry.font_px))

    gap = geometry.profile.lane_gap_ratio * geometry.font_px
    lane_bottoms = {"main": float(geometry.main_bottom)}
    lane_bottoms["secondary"] = float(round(lane_bottoms["main"] - block("main") - gap))
    lane_bottoms["tertiary"] = float(round(lane_bottoms["secondary"] - block("secondary") - gap))

    strict_lanes = strict_lane_timing(profile)
    next_in_lane: dict[int, float | None] = {}
    last_index: dict[str, int] = {}
    for index, (draft, _s, _i) in enumerate(styled):
        lane = draft.tokens[0].lane
        if lane in last_index:
            next_in_lane[last_index[lane]] = draft.tokens[0].start
        last_index[lane] = index

    palettes = palettes_for(brand)
    speakers = speaker_palettes_for(brand)
    pages: list[CaptionPage] = []
    for index, ((draft, style, directive_id), emphasis) in enumerate(zip(styled, page_emphasis)):
        lane = draft.tokens[0].lane
        resolved = resolve_caption_style(style, lane, geometry, fit_scale=draft.fit_scale, palettes=palettes,
                                         colour=draft.tokens[0].speaker_color, speakers=speakers)
        measure = _Measure(font, resolved.font_px, active_shaper)
        label = draft.tokens[0].speaker_label.strip()
        lines: list[CaptionLine] = []
        starts = (0,) + draft.breaks[:-1]
        line_adv = advance(resolved.font_px)
        for line_index, (a, b) in enumerate(zip(starts, draft.breaks)):
            chunk = draft.tokens[a:b]
            scales = [resolved.emphasis_scale if t.word_id in emphasis and allow_scale else 1.0 for t in chunk]
            first = line_index == 0
            w = line_width([t.text for t in chunk], scales, measure, label if first else "")
            y = lane_bottoms[lane] - (len(draft.breaks) - 1 - line_index) * line_adv
            lines.append(CaptionLine(tuple(t.word_id for t in chunk), round(w, 1), round(geometry.width / 2.0),
                                     round(y), has_label=first and bool(label)))
        pages.append(CaptionPage(
            page_id=index, lane=lane, tokens=draft.tokens, lines=tuple(lines), style=style, emphasis_ids=emphasis,
            start=draft.tokens[0].start,
            end=_page_end(draft.tokens, next_in_lane.get(index), duration, strict_no_overlap=strict_lanes),
            font_px=resolved.font_px, fit_scale=draft.fit_scale, label=label, directive_id=directive_id,
        ))

    # Placement: one position per page cluster (V3.1 bottom without evidence).
    placement: PlacementReport | None = None
    busy: set[int] = set()
    if evidence is not None:
        geometries = []
        for page in pages:
            style = resolve_caption_style(page.style, page.lane, geometry, fit_scale=page.fit_scale,
                                          palettes=palettes, colour=page.speaker_color, speakers=speakers)
            scale = style.emphasis_scale if page.emphasis_ids and allow_scale else 1.0
            geometries.append(PageGeometry(page.page_id, page.start, page.end,
                                           _line_boxes_px(page, geometry, font, style, scale)))
        shifts, placement = solve_placement(geometries, evidence, width=width, height=height,
                                            profile_name=geometry.profile.name.value,
                                            policy=brand.caption_anchor_policy)
        zone_of = {pid: d.zone for d in placement.decisions for pid in d.page_ids}
        busy = {pid for d in placement.decisions if d.busy_background for pid in d.page_ids}
        pages = [dataclasses.replace(
            page, placement_zone=zone_of.get(page.page_id, "bottom"),
            lines=tuple(dataclasses.replace(line, y=round(line.y + shifts.get(page.page_id, 0.0)))
                        for line in page.lines)) for page in pages]
        lane_bottoms = dict(lane_bottoms)

    # Legibility (after placement: the background under the page's FINAL position).
    decisions: tuple[LegibilityDecision, ...] = ()
    if background is not None and brand.legibility != "off":
        decisions = assess_pages(pages, geometry, font, palettes, brand, background, allow_scale)
        by_page = {d.page_id: d for d in decisions}
        pages = [_apply_legibility(page, by_page.get(page.page_id)) for page in pages]
    legible = {d.page_id: d for d in decisions}

    facts = [PageFacts(page.page_id, page.style, page.start, page.end,
                       {wid: reasons[wid][0] for wid in page.emphasis_ids if wid in reasons},
                       page.page_id in busy,
                       legibility_plate=bool(legible.get(page.page_id) and legible[page.page_id].plate),
                       legibility_outline=page.outline_boost > 1.0,
                       legibility_shadow=page.shadow_px is not None) for page in pages]
    effects, primitive_log = resolve_page_effects(facts, brand.allowed_primitives, brand.budget)
    plates = PrimitiveId.STATIC_ACCENT_BACKPLATE in brand.allowed_primitives or any(e.draws for e in effects)
    settings = RenderSettings(
        palettes=palettes, speakers=speakers, font_family=font_family or brand.font_family, bold=bold,
        outline_colour=brand.outline_color, effects={e.page_id: e for e in effects}, plates=plates,
        plate_colour=brand.backplate_color, plate_alpha=brand.backplate_alpha, metrics=font, shaper=active_shaper,
        allow_emphasis_scale=allow_scale)
    ass_text = render_ass(pages, geometry, settings)
    presentation = CaptionPresentation(tokens=tokens, pages=tuple(pages), geometry=geometry, metrics_font=font,
                                       directives=directives, emphasis=report, ass_text=ass_text,
                                       lane_bottoms=lane_bottoms, settings=settings, placement=placement,
                                       primitive_log=primitive_log, reasons=reasons, brand=brand,
                                       platform=platform, shaper_note=shaper_note, legibility=decisions)
    verify_presentation(presentation)
    return presentation


def page_style(page: CaptionPage, geometry: LayoutGeometry, palettes: Mapping[str, LanePalette] | None = None,
               speakers: Mapping[str, LanePalette] | None = None) -> ResolvedCaptionStyle:
    """Resolved style of a page (its lane's style, its voice's colours) including
    its legibility outline boost."""
    return resolve_caption_style(page.style, page.lane, geometry, fit_scale=page.fit_scale, palettes=palettes,
                                 outline_boost=page.outline_boost, colour=page.speaker_color, speakers=speakers)


def _page_box_norm(page: CaptionPage, geometry: LayoutGeometry, font: FontMetrics, style: ResolvedCaptionStyle,
                   scale: float) -> tuple[float, float, float, float]:
    boxes = _line_boxes_px(page, geometry, font, style, scale)
    x0 = min(b[0] for b in boxes) / geometry.width
    y0 = min(b[1] for b in boxes) / geometry.height
    x1 = max(b[2] for b in boxes) / geometry.width
    y1 = max(b[3] for b in boxes) / geometry.height
    return (max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1))


def page_appearance(page: CaptionPage, style: ResolvedCaptionStyle, geometry: LayoutGeometry,
                    brand: BrandProfile) -> TextAppearance:
    fills = [style.inactive_colour, style.active_colour]
    if page.emphasis_ids:
        fills.append(style.emphasis_colour)
    if page.label:
        fills.append(style.label_colour)
    return TextAppearance(tuple(dict.fromkeys(fills)), brand.outline_color, style.outline_px, style.font_px,
                          geometry.shadow_px, brand.backplate_color, brand.backplate_alpha)


def assess_pages(pages: Sequence[CaptionPage], geometry: LayoutGeometry, font: FontMetrics,
                 palettes: Mapping[str, LanePalette], brand: BrandProfile, background: Any,
                 allow_scale: bool) -> tuple[LegibilityDecision, ...]:
    """Legibility verdict per page; plates are budgeted here (the most severe pages first)."""
    fingerprint = str(getattr(getattr(background, "inner", background), "fingerprint", ""))
    allow_plate = brand.legibility == "auto"
    rows: list[tuple[CaptionPage, Any, TextAppearance, str, LegibilityDecision]] = []
    speakers = speaker_palettes_for(brand)
    for page in pages:
        style = page_style(page, geometry, palettes, speakers)
        scale = style.emphasis_scale if page.emphasis_ids and allow_scale else 1.0
        box = _page_box_norm(page, geometry, font, style, scale)
        stats = background.region_stats(page.start, page.end, box)
        appearance = page_appearance(page, style, geometry, brand)
        key = decision_key(page.page_id, page.start, page.end, box, appearance, fingerprint)
        rows.append((page, stats, appearance, key, assess(page.page_id, stats, appearance, key=key,
                                                          allow_plate=allow_plate)))
    wanting = [r for r in rows if r[4].plate]
    limit = max(1, int(math.floor(LEGIBILITY_PLATE_MAX_RATIO * len(pages)))) if pages else 0
    wanting.sort(key=lambda r: (-float(r[4].metrics.get("complexity", 0.0)),
                                float(r[4].metrics.get("fill_contrast", 99.0)), r[0].page_id))
    over = {r[0].page_id for r in wanting[limit:]}
    decisions = []
    for page, stats, appearance, key, decision in rows:
        if page.page_id in over:
            decision = assess(page.page_id, stats, appearance, key=key, allow_plate=False)
            decision = dataclasses.replace(decision, degraded_from="LEGIBILITY_NEEDS_PLATE (plate budget)")
        decisions.append(decision)
    return tuple(decisions)


def _apply_legibility(page: CaptionPage, decision: LegibilityDecision | None) -> CaptionPage:
    if decision is None:
        return page
    return dataclasses.replace(page, legibility=decision.verdict.value, outline_boost=decision.outline_boost,
                               shadow_px=decision.shadow_px, plate_alpha=decision.plate_alpha)


def apply_effect_overrides(presentation: CaptionPresentation, overrides: Mapping[int, PageEffects],
                           notes: Mapping[int, str]) -> CaptionPresentation:
    """Re-render the SAME pages with calmer effects (editorial energy). Geometry is never
    increased: only decoration is removed, legibility treatments are kept."""
    if not overrides:
        return presentation
    settings = presentation.settings or DEFAULT_SETTINGS
    effects = {page.page_id: settings.effects_for(page) for page in presentation.pages}
    for page_id, override in overrides.items():
        if page_id in effects:
            effects[page_id] = override
    log = PrimitiveLog(degraded=list(presentation.primitive_log.degraded) if presentation.primitive_log else [])
    for page_id in sorted(effects):
        for primitive in effects[page_id].primitives:
            log.count(primitive)
    for page_id, note in sorted(notes.items()):
        log.degraded.append({"page_id": page_id, "primitive": "decoration", "reason": "editorial_energy",
                             "fallback": note})
    settings = dataclasses.replace(settings, effects=effects)
    updated = dataclasses.replace(presentation, settings=settings, primitive_log=log,
                                  ass_text=render_ass(presentation.pages, presentation.geometry, settings),
                                  energy_overrides={**dict(presentation.energy_overrides), **dict(notes)})
    verify_presentation(updated)
    return updated


# ============================================================
# ASS
# ============================================================

def _cs(seconds: float) -> int:
    return int(round(max(0.0, float(seconds)) * 100))


def _ass_time_cs(cs: int) -> str:
    hours, rest = divmod(cs, 360000)
    minutes, rest = divmod(rest, 6000)
    secs, centis = divmod(rest, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _num(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text or "0"


class _LineState:
    """Override state inside one Dialogue line; tags are emitted only on change
    (no ``\\r``: a reset would also drop page-level ``\\fs``/``\\bord``)."""

    def __init__(self, outline: float, outline_colour: str = "") -> None:
        self.visible = True
        self.colour: str | None = None
        self.scale = 100.0
        self.outline = outline
        self.default_outline_colour = outline_colour
        self.outline_colour = outline_colour

    def block(self, *, visible: bool, colour: str | None, scale: float, outline: float,
              outline_colour: str | None = None) -> str:
        tags: list[str] = []
        if visible != self.visible:
            tags.append("\\1a&H00&\\3a&H00&" if visible else "\\1a&HFF&\\3a&HFF&")
            self.visible = visible
        if visible and colour is not None and colour != self.colour:
            tags.append(f"\\c{colour}")
            self.colour = colour
        wanted = outline_colour or self.default_outline_colour
        if visible and wanted and wanted != self.outline_colour:
            tags.append(f"\\3c{wanted}")
            self.outline_colour = wanted
        size = round(scale * 100.0, 2)
        if abs(size - self.scale) > 1e-6:
            tags.append(f"\\fscx{_num(size)}\\fscy{_num(size)}")
            self.scale = size
        if abs(outline - self.outline) > 0.05:
            tags.append(f"\\bord{_num(outline)}")
            self.outline = outline
        return "{" + "".join(tags) + "}" if tags else ""

    def activate(self, *, start_colour: str, target_colour: str, scale: float, base_outline: float,
                 accent_outline: float, attack_ms: int, release_ms: int, interval_ms: int,
                 outline_colour: str | None = None) -> str:
        """Active word: colour ease + optional outline accent, all inside the
        word's own interval (never past the next measured onset)."""
        tags = self.block(visible=True, colour=start_colour, scale=scale, outline=base_outline,
                          outline_colour=outline_colour)
        attack = max(0, min(attack_ms, interval_ms))
        release = max(0, min(release_ms, interval_ms - attack))
        accent = accent_outline > base_outline + 0.05
        grow = f"\\c{target_colour}" + (f"\\bord{_num(accent_outline)}" if accent else "")
        extra = f"\\t(0,{attack},{grow})"
        self.colour = target_colour
        if accent:
            if release > 0:
                extra += f"\\t({attack},{attack + release},\\bord{_num(base_outline)})"
            else:
                self.outline = accent_outline
        body = tags[1:-1] if tags else ""
        return "{" + body + extra + "}"


@dataclass(frozen=True)
class RenderSettings:
    """Brand + primitive decisions for the ASS writer (defaults == V3.1)."""

    palettes: Mapping[str, LanePalette] = field(default_factory=lambda: PALETTES)
    speakers: Mapping[str, LanePalette] = field(default_factory=lambda: SPEAKER_PALETTES)
    font_family: str = caption_truth.FONT_NAME
    bold: bool = True
    outline_colour: str = caption_truth.OUTLINE_COLOR
    effects: Mapping[int, PageEffects] = field(default_factory=dict)
    plates: bool = False
    plate_colour: str = "&H000000&"
    plate_alpha: int = 0x8C
    metrics: FontMetrics | None = None
    shaper: Any = None
    allow_emphasis_scale: bool = True

    def effects_for(self, page: CaptionPage) -> PageEffects:
        found = self.effects.get(page.page_id)
        if found is not None:
            return found
        base = {PrimitiveId.STATIC, PrimitiveId.ACTIVE_COLOR_EASE}
        if page.style is not CaptionStyle.DEFAULT:
            base |= {PrimitiveId.ACTIVE_OUTLINE_EASE, PrimitiveId.STATIC_EMPHASIS_SCALE}
        if page.style is CaptionStyle.IMPACT:
            base.add(PrimitiveId.SOFT_IMPACT_OUTLINE)
        return PageEffects(page.page_id, frozenset(base & V31_PRIMITIVES))

    def text_layer(self, lane: str) -> int:
        return LANE_LAYERS[lane] * 2 + 1 if self.plates else LANE_LAYERS[lane]


DEFAULT_SETTINGS = RenderSettings()


def _accent_outline(page: CaptionPage, style: ResolvedCaptionStyle, effects: PageEffects, emphasized: bool) -> float:
    if emphasized:
        if page.style is CaptionStyle.IMPACT and effects.has(PrimitiveId.SOFT_IMPACT_OUTLINE):
            return style.emphasis_outline_px
        if effects.has(PrimitiveId.ACTIVE_OUTLINE_EASE):
            return style.emphasis_outline_px if page.style is not CaptionStyle.IMPACT else style.active_outline_px
        return style.outline_px
    return style.active_outline_px if effects.has(PrimitiveId.ACTIVE_OUTLINE_EASE) else style.outline_px


def render_line_text(page: CaptionPage, line: CaptionLine, active_index: int, geometry: LayoutGeometry,
                     interval_ms: int = 10_000, settings: RenderSettings = DEFAULT_SETTINGS,
                     fade_ms: int = 0) -> str:
    """One positioned line of a page with word ``active_index`` active."""
    style = page_style(page, geometry, settings.palettes, settings.speakers)
    effects = settings.effects_for(page)
    order = {t.word_id: i for i, t in enumerate(page.tokens)}
    by_id = {t.word_id: t for t in page.tokens}
    lead = [f"\\an2\\pos({_num(line.x)},{_num(line.y)})"]
    if abs(style.font_px - geometry.font_px) > 0.05:
        lead.append(f"\\fs{_num(style.font_px)}")
    if abs(style.outline_px - geometry.outline_px) > 0.05:
        lead.append(f"\\bord{_num(style.outline_px)}")
    if page.shadow_px is not None and abs(page.shadow_px - geometry.shadow_px) > 0.05:
        lead.append(f"\\shad{_num(page.shadow_px)}")
    if fade_ms > 0:
        lead.append(f"\\fad({fade_ms},0)")
    parts = ["{" + "".join(lead) + "}"]
    contrast_outline = _colour(style.emphasis_colour)
    state = _LineState(style.outline_px, _colour(settings.outline_colour) if effects.contrast_word_ids else "")
    if line.has_label:
        parts.append(state.block(visible=True, colour=style.label_colour, scale=1.0, outline=style.outline_px)
                     + display_text(_label_text(page.label)))
    for position, word_id in enumerate(line.word_ids):
        index = order[word_id]
        emphasized = word_id in page.emphasis_ids
        scaled = emphasized and settings.allow_emphasis_scale and effects.has(PrimitiveId.STATIC_EMPHASIS_SCALE)
        scale = style.emphasis_scale if scaled else 1.0
        contrast = emphasized and word_id in effects.contrast_word_ids
        ring = contrast_outline if contrast else None
        if index > active_index:
            block = state.block(visible=False, colour=None, scale=scale, outline=style.outline_px)
        elif index == active_index:
            block = state.activate(
                start_colour=style.inactive_colour,
                target_colour=(style.active_colour if contrast else
                               style.emphasis_colour if emphasized else style.active_colour),
                scale=scale, base_outline=style.outline_px,
                accent_outline=_accent_outline(page, style, effects, emphasized),
                attack_ms=style.attack_ms, release_ms=style.release_ms, interval_ms=interval_ms,
                outline_colour=ring)
        else:
            fill = style.inactive_colour if contrast or not emphasized else style.emphasis_colour
            block = state.block(visible=True, colour=fill, scale=scale, outline=style.outline_px,
                                outline_colour=ring)
        spacer = " " if (position > 0 or line.has_label) else ""
        parts.append(f"{block}{spacer}{display_text(by_id[word_id].text)}")
    return "".join(parts)


PLATE_PAD_RATIO = 0.18
BADGE_BORDER_RATIO = 0.035       # badge accent border / font px
UNDERLINE_RATIO = 0.07           # underline thickness / font px


@dataclass(frozen=True)
class _Drawing:
    first_index: int                      # first word whose onset shows the drawing
    box: tuple[int, int, int, int]        # x, y, w, h (px, clamped inside the frame)
    kind: str                             # plate | badge | underline
    border: float = 0.0


def _word_spans(page: CaptionPage, line: CaptionLine, style: ResolvedCaptionStyle, settings: RenderSettings,
                measure: "_Measure") -> list[tuple[int, float, float, float]]:
    """(word_id, left px, width px, scale) of every word of a line (same metrics as the layout)."""
    by_id = {t.word_id: t for t in page.tokens}
    scales = [style.emphasis_scale if wid in page.emphasis_ids and settings.allow_emphasis_scale else 1.0
              for wid in line.word_ids]
    left = line.x - line.width_px / 2.0
    cursor = measure.word(_label_text(page.label)) if line.has_label else 0.0
    spans = []
    for position, word_id in enumerate(line.word_ids):
        if position > 0 or line.has_label:
            cursor += measure.space(scales[position])
        width = measure.word(by_id[word_id].text, scales[position])
        spans.append((word_id, left + cursor, width, scales[position]))
        cursor += width
    return spans


def _drawings(page: CaptionPage, geometry: LayoutGeometry, settings: RenderSettings) -> list[_Drawing]:
    """Every static vector drawing of a page (plates, badges, underlines), clamped inside the frame."""
    effects = settings.effects_for(page)
    if not effects.draws or settings.metrics is None:
        return []
    style = page_style(page, geometry, settings.palettes, settings.speakers)
    measure = _Measure(settings.metrics, style.font_px, settings.shaper)
    pad = style.font_px * PLATE_PAD_RATIO
    order = {t.word_id: i for i, t in enumerate(page.tokens)}
    font = settings.metrics
    descent = font.descent_units * style.font_px / font.size_basis_units
    raw: list[tuple[int, tuple[float, float, float, float], str, float]] = []
    for line in page.lines:
        spans = _word_spans(page, line, style, settings, measure)
        height = font.line_height(style.font_px) * max([1.0] + [sc for _w, _l, _wd, sc in spans])
        left = line.x - line.width_px / 2.0
        if effects.plate_lines:
            raw.append((order[line.word_ids[0]],
                        (left - pad, line.y - height - pad * 0.5, line.width_px + 2 * pad, height + pad), "plate",
                        0.0))
        for word_id, word_left, width, scale in spans:
            if word_id in effects.plate_word_ids or word_id in effects.badge_word_ids:
                badge = word_id in effects.badge_word_ids
                raw.append((order[word_id], (word_left - pad * 0.6, line.y - height - pad * 0.4,
                                             width + 1.2 * pad, height + 0.8 * pad),
                            "badge" if badge else "plate",
                            round(max(1.0, BADGE_BORDER_RATIO * style.font_px), 1) if badge else 0.0))
            if word_id in effects.underline_word_ids:
                # Inside the line box: between the baseline and the box bottom (no new geometry).
                thickness = max(2.0, UNDERLINE_RATIO * style.font_px * scale)
                top = line.y - descent * scale * 0.75
                thickness = min(thickness, max(1.0, line.y - top))
                raw.append((order[word_id], (word_left, top, width, thickness), "underline", 0.0))
    drawings = []
    for index, (x, y, w, h), kind, border in raw:
        x0, y0 = max(border, x), max(border, y)
        x1, y1 = min(float(geometry.width) - border, x + w), min(float(geometry.height) - border, y + h)
        if x1 - x0 >= 2 and y1 - y0 >= 1:
            drawings.append(_Drawing(index, (int(round(x0)), int(round(y0)), max(1, int(round(x1 - x0))),
                                             max(1, int(round(y1 - y0)))), kind, border))
    return drawings


def _plate_boxes(page: CaptionPage, geometry: LayoutGeometry, settings: RenderSettings
                 ) -> list[tuple[int, tuple[int, int, int, int]]]:
    """(first word index, (x, y, w, h) px) of every drawing; badge borders are included."""
    rows = []
    for drawing in _drawings(page, geometry, settings):
        x, y, w, h = drawing.box
        b = int(math.ceil(drawing.border))
        rows.append((drawing.first_index, (x - b, y - b, w + 2 * b, h + 2 * b)))
    return rows


def _drawing_text(drawing: _Drawing, style: ResolvedCaptionStyle, settings: RenderSettings,
                  plate_alpha: int | None = None) -> str:
    x, y, w, h = drawing.box
    alpha = settings.plate_alpha if plate_alpha is None else plate_alpha
    if drawing.kind == "underline":
        head = (f"\\an7\\pos({x},{y})\\bord0\\shad0\\c{style.emphasis_colour}\\1a&H00&"
                f"\\3a&HFF&\\4a&HFF&\\p1")
    elif drawing.kind == "badge":
        head = (f"\\an7\\pos({x},{y})\\bord{_num(drawing.border)}\\shad0\\c{settings.plate_colour}"
                f"\\3c{style.emphasis_colour}\\1a&H{settings.plate_alpha:02X}&\\3a&H00&\\4a&HFF&\\p1")
    else:
        head = (f"\\an7\\pos({x},{y})\\bord0\\shad0\\c{settings.plate_colour}"
                f"\\1a&H{alpha:02X}&\\3a&HFF&\\4a&HFF&\\p1")
    return "{" + head + "}" + f"m 0 0 l {w} 0 {w} {h} 0 {h}" + "{\\p0}"


def page_events(page: CaptionPage, geometry: LayoutGeometry, settings: RenderSettings = DEFAULT_SETTINGS
                ) -> list[tuple[int, int, int, str, str]]:
    """(start_cs, end_cs, layer, style, text) for every visible state of a page."""
    boundaries = [_cs(t.start) for t in page.tokens] + [_cs(page.end)]
    for k in range(1, len(boundaries)):
        boundaries[k] = max(boundaries[k], boundaries[k - 1])
    order = {t.word_id: i for i, t in enumerate(page.tokens)}
    layer = settings.text_layer(page.lane)
    style_name = settings.palettes[page.lane].style_name
    effects = settings.effects_for(page)
    events: list[tuple[int, int, int, str, str]] = []
    drawings = _drawings(page, geometry, settings)
    if drawings:
        style = page_style(page, geometry, settings.palettes, settings.speakers)
        legibility_alpha = page.plate_alpha if effects.plate_reason == "legibility" else None
        for drawing in drawings:
            a = boundaries[drawing.first_index]
            if boundaries[-1] > a:
                alpha = legibility_alpha if drawing.kind == "plate" else None
                events.append((a, boundaries[-1], layer - 1, style_name,
                               _drawing_text(drawing, style, settings, alpha)))
    for active in range(len(page.tokens)):
        a, b = boundaries[active], boundaries[active + 1]
        if b <= a:
            continue
        fade = 0
        if effects.fade_in_ms and active == 0:
            fade = min(effects.fade_in_ms, (b - a) * 10 // 3)
            fade = fade if fade >= 10 else 0
        for line in page.lines:
            if min(order[w] for w in line.word_ids) > active:
                continue
            events.append((a, b, layer, style_name,
                           render_line_text(page, line, active, geometry, (b - a) * 10, settings, fade)))
    return events


def ass_header(geometry: LayoutGeometry, settings: RenderSettings = DEFAULT_SETTINGS) -> str:
    font = settings.font_family
    size = _num(geometry.font_px)
    outline = _num(geometry.outline_px)
    shadow = _num(geometry.shadow_px)
    bold = "-1" if settings.bold else "0"
    rows = []
    for palette in settings.palettes.values():
        rows.append(
            f"Style: {palette.style_name},{font},{size},{palette.primary},{palette.primary},"
            f"{settings.outline_colour},{caption_truth.SHADOW_COLOR},{bold},0,0,0,100,100,0,0,1,{outline},"
            f"{shadow},2,0,0,0,1")
    styles = "\n".join(rows)
    return (
        "[Script Info]\n"
        f"Title: MIMIR Pro Edit Caption Presentation v{CAPTION_PRESENTATION_VERSION}\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {geometry.width}\n"
        f"PlayResY: {geometry.height}\n"
        "ScaledBorderAndShadow: yes\n"
        "WrapStyle: 2\n"
        "YCbCr Matrix: TV.709\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"{styles}\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def render_ass(pages: Sequence[CaptionPage], geometry: LayoutGeometry,
               settings: RenderSettings = DEFAULT_SETTINGS) -> str:
    events: list[tuple[int, int, int, str, str]] = []
    for page in pages:
        events.extend(page_events(page, geometry, settings))
    events.sort(key=lambda e: (e[0], e[2], e[1]))
    lines = [f"Dialogue: {layer},{_ass_time_cs(a)},{_ass_time_cs(b)},{style},,0,0,0,,{text}"
             for a, b, layer, style, text in events]
    return ass_header(geometry, settings) + "\n".join(lines) + "\n"


# ============================================================
# VERIFICATION + METRICS + MANIFEST
# ============================================================

def verify_presentation(presentation: CaptionPresentation) -> None:
    """Every token shown exactly in one page, at its onset, in time order;
    nothing wider than the frame; every word visible at least once."""
    tokens = presentation.tokens
    seen: list[int] = [t.word_id for page in presentation.pages for t in page.tokens]
    if sorted(seen) != sorted(t.word_id for t in tokens) or len(seen) != len(set(seen)):
        raise CaptionPresentationError("page token coverage differs from caption truth")
    by_id = {t.word_id: t for t in tokens}
    geometry = presentation.geometry
    for page in presentation.pages:
        for token in page.tokens:
            original = by_id[token.word_id]
            if (token.text, token.start, token.end) != (original.text, original.start, original.end):
                raise CaptionPresentationError(f"word {token.word_id} truth mutated in presentation")
        if page.end <= page.start:
            raise CaptionPresentationError(f"page {page.page_id} has no display time")
        if len({t.speaker_color for t in page.tokens}) != 1:
            raise CaptionPresentationError(f"page {page.page_id} mixes speaker colours")
        line_ids = [w for line in page.lines for w in line.word_ids]
        if line_ids != [t.word_id for t in page.tokens]:
            raise CaptionPresentationError(f"page {page.page_id} lines reorder or drop words")
        for line in page.lines:
            if line.width_px + 2 * geometry.outline_px > geometry.width + 0.5:
                raise CaptionPresentationError(f"page {page.page_id} line wider than the frame")
    shown = _visible_word_ids(presentation)
    missing = sorted(set(by_id) - shown)
    if missing:
        raise CaptionPresentationError(f"words never visible in the presentation ASS: {missing[:8]}")


def _visible_word_ids(presentation: CaptionPresentation) -> set[int]:
    shown: set[int] = set()
    for page in presentation.pages:
        boundaries = [_cs(t.start) for t in page.tokens] + [_cs(page.end)]
        for k in range(1, len(boundaries)):
            boundaries[k] = max(boundaries[k], boundaries[k - 1])
        for active in range(len(page.tokens)):
            if boundaries[active + 1] > boundaries[active]:
                shown.update(t.word_id for t in page.tokens[:active + 1])
    return shown


def page_safe_band(page: CaptionPage, presentation: CaptionPresentation) -> tuple[float, float]:
    """Normalized vertical extent (y0, y1) this page occupies on the output frame
    (text, largest outline accent and any backplate)."""
    geometry = presentation.geometry
    settings = presentation.settings or DEFAULT_SETTINGS
    style = page_style(page, geometry, settings.palettes, settings.speakers)
    scale = style.emphasis_scale if page.emphasis_ids else 1.0
    border = max(style.outline_px, style.active_outline_px, style.emphasis_outline_px)
    shadow = geometry.shadow_px if page.shadow_px is None else max(page.shadow_px, geometry.shadow_px)
    top = min(line.y for line in page.lines) - presentation.metrics_font.line_height(style.font_px) * scale - border
    bottom = max(line.y for line in page.lines) + border + shadow
    for _index, (_x, y, _w, h) in _plate_boxes(page, geometry, settings):
        top, bottom = min(top, y), max(bottom, y + h)
    return max(0.0, top / geometry.height), min(1.0, bottom / geometry.height)


def kerning_report(presentation: CaptionPresentation) -> dict[str, Any]:
    """Diagnostic only: how much classic kerning would tighten lines (layout stays unkerned)."""
    font = presentation.metrics_font
    worst = 0.0
    for page in presentation.pages:
        by_id = {t.word_id: t.text for t in page.tokens}
        for line in page.lines:
            text = " ".join(by_id[w] for w in line.word_ids)
            plain = font.text_width(text, 100.0)
            if plain > 0:
                worst = max(worst, (plain - font.kerned_width(text, 100.0)) / plain)
    return {"kern_pairs": len(font.kern_pairs), "max_line_tightening_ratio": round(worst, 4),
            "layout": "unkerned advances (libass build measured without kerning)"}


def _max_simultaneous_words(pages: Sequence[CaptionPage]) -> int:
    """Largest number of visible words at one instant (all lanes)."""
    moments: list[tuple[int, int, int]] = []
    for page in pages:
        end = _cs(page.end)
        for token in page.tokens:
            moments.append((_cs(token.start), 1, 1))
            moments.append((end, 0, -1))
    count = best = 0
    for _t, _kind, delta in sorted(moments):
        count += delta
        best = max(best, count)
    return best


def presentation_metrics(presentation: CaptionPresentation) -> dict[str, Any]:
    pages = presentation.pages
    geometry = presentation.geometry
    durations = [p.end - p.start for p in pages]
    cps = sorted(sum(len(t.text) for t in p.tokens) / max(p.speech_end - p.start, 0.3) for p in pages)
    wps = sorted(len(p.tokens) / max(p.speech_end - p.start, 0.3) for p in pages)
    widths = [line.width_px for p in pages for line in p.lines]
    overlaps = 0
    last: dict[str, float] = {}
    for page in pages:
        if page.lane in last and page.start < last[page.lane] - 1e-6:
            overlaps += 1
        last[page.lane] = page.end
    runs = split_runs(presentation.tokens)
    in_long_run = {t.word_id for r in runs if len(r) >= 2 for t in r}
    orphan = sum(1 for p in pages if len(p.tokens) == 1 and p.tokens[0].word_id in in_long_run)
    flashes = sum(1 for p in pages if p.end - p.tokens[-1].start < TERMINAL_FLASH_S - 0.005)
    bands = [page_safe_band(p, presentation) for p in pages]
    return {
        "words": len(presentation.tokens),
        "pages": len(pages),
        "avg_words_per_page": round(len(presentation.tokens) / max(1, len(pages)), 2),
        "max_words_per_page": max((len(p.tokens) for p in pages), default=0),
        "two_line_pages": sum(1 for p in pages if len(p.lines) > 1),
        "single_word_pages": sum(1 for p in pages if len(p.tokens) == 1),
        "orphan_pages": orphan,
        "fit_scaled_pages": sum(1 for p in pages if p.fit_scale < 1.0),
        "max_line_width_px": round(max(widths, default=0.0), 1),
        "max_line_width_ratio": round(max(widths, default=0.0) / geometry.width, 4),
        "min_page_s": round(min(durations, default=0.0), 3),
        "mean_page_s": round(sum(durations) / len(durations), 3) if durations else 0.0,
        "cps_p50": round(cps[len(cps) // 2], 2) if cps else 0.0,
        "cps_max": round(cps[-1], 2) if cps else 0.0,
        "wps_p50": round(wps[len(wps) // 2], 2) if wps else 0.0,
        "wps_max": round(wps[-1], 2) if wps else 0.0,
        "max_simultaneous_words": _max_simultaneous_words(pages),
        "terminal_flash_pages": flashes,
        "emphasized_words": len(presentation.emphasis.applied),
        "emphasis_ratio": round(len(presentation.emphasis.applied) / max(1, len(presentation.tokens)), 4),
        "invalid_emphasis_ignored": presentation.emphasis.invalid_ignored,
        "impact_pages": sum(1 for p in pages if p.style is CaptionStyle.IMPACT),
        "emphasis_pages": sum(1 for p in pages if p.style is CaptionStyle.EMPHASIS),
        "speaker_pages": sum(1 for p in pages if p.lane != "main"),
        "secondary_pages": sum(1 for p in pages if p.lane == "secondary"),
        "tertiary_pages": sum(1 for p in pages if p.lane == "tertiary"),
        "labeled_pages": sum(1 for p in pages if p.label),
        "speaker_color_pages": {colour or "plain": sum(1 for p in pages if p.speaker_color == colour)
                                for colour in sorted({p.speaker_color for p in pages})},
        "uncertain_words": sum(1 for t in presentation.tokens if t.uncertain),
        "same_lane_overlaps": overlaps,
        "placement_zones": dict(presentation.placement.zones) if presentation.placement else {"bottom": len(pages)},
        "placement_switches": presentation.placement.switches if presentation.placement else 0,
        "primitives": dict(presentation.primitive_log.counts) if presentation.primitive_log else {},
        "primitive_degradations": len(presentation.primitive_log.degraded) if presentation.primitive_log else 0,
        "safe_band": [round(min((b[0] for b in bands), default=0.0), 4),
                      round(max((b[1] for b in bands), default=0.0), 4)],
        "legibility": legibility_counts(presentation),
        "energy_degraded_pages": len(presentation.energy_overrides),
    }


def legibility_counts(presentation: CaptionPresentation) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in presentation.legibility:
        counts[decision.verdict.value] = counts.get(decision.verdict.value, 0) + 1
    if not presentation.legibility:
        counts["not_analysed"] = len(presentation.pages)
    return counts


def build_manifest(presentation: CaptionPresentation, *, caption_signature: str = "",
                   page_limit: int = 600) -> dict[str, Any]:
    """Deterministic record of what was rendered (no model output, no secrets)."""
    geometry = presentation.geometry
    directives = []
    for directive in presentation.directives:
        pages = [p.page_id for p in presentation.pages if p.directive_id == directive.event_id]
        directives.append({
            "event_id": directive.event_id, "style": directive.style.value,
            "window": [round(directive.start, 3), round(directive.end, 3)],
            "emphasis_word_ids": list(directive.emphasis_word_ids),
            "pages": pages,
            "page_styles": sorted({presentation.pages[i].style.value for i in pages}),
            "executed": bool(pages) or bool(set(directive.emphasis_word_ids) & set(presentation.emphasis.applied)),
        })
    metrics = presentation.metrics()
    rows = []
    for page in presentation.pages[:page_limit]:
        y0, y1 = page_safe_band(page, presentation)
        rows.append({"page_id": page.page_id, "word_ids": [t.word_id for t in page.tokens],
                     "start": page.start, "end": round(page.end, 3),
                     "lines": [list(line.word_ids) for line in page.lines], "style": page.style.value,
                     "emphasized_word_ids": sorted(page.emphasis_ids), "speaker_role": page.speaker_role,
                     "lane": page.lane, "speaker_color": page.speaker_color, "label": page.label,
                     "font_px": page.font_px,
                     "fit_scale": round(page.fit_scale, 3), "safe_region": [round(y0, 4), round(y1, 4)],
                     "placement_zone": page.placement_zone,
                     "legibility": page.legibility or "not_analysed", "outline_boost": page.outline_boost,
                     "shadow_px": page.shadow_px,
                     "energy": presentation.energy_overrides.get(page.page_id, ""),
                     "primitives": sorted(p.value for p in (presentation.settings or DEFAULT_SETTINGS)
                                          .effects_for(page).primitives),
                     "emphasis_reasons": {str(wid): presentation.reasons[wid][0]
                                          for wid in sorted(page.emphasis_ids) if wid in presentation.reasons}})
    return {
        "version": CAPTION_PRESENTATION_VERSION,
        "style_pack": f"{CAPTION_STYLE_PACK}@{CAPTION_STYLE_PACK_VERSION}",
        "layout_profile": geometry.profile.name.value,
        "play_res": [geometry.width, geometry.height],
        "font": presentation.metrics_font.to_dict(),
        "font_px": geometry.font_px,
        "safe_width_px": round(geometry.max_line_px, 1),
        "lane_bottoms": dict(presentation.lane_bottoms),
        "safe_region": {"band_y": metrics["safe_band"], "source": "presentation geometry (== burned ASS)"},
        "styles": {style.value: resolve_caption_style(style, "main", geometry).to_dict() for style in STYLE_DEFINITIONS},
        "truth": {"word_count": len(presentation.tokens), "caption_signature": caption_signature,
                  "parity": "captions._profile_edited_words", "retimed_words": 0,
                  "speaker_display": ("captions._prepare_adaptive_render_words"
                                      if callable(getattr(caption_truth, "_prepare_adaptive_render_words", None))
                                      else "raw profile metadata")},
        "emphasis": {"requested": presentation.emphasis.requested, "applied": presentation.emphasis.applied,
                     "degraded": presentation.emphasis.degraded,
                     "terra_markers": presentation.emphasis.terra_markers,
                     "terra_matched": presentation.emphasis.terra_matched},
        "directives": directives,
        "metrics": metrics,
        "brand": presentation.brand.to_dict(),
        "platform": presentation.platform.to_dict(),
        "placement": presentation.placement.to_dict() if presentation.placement else {"policy": "disabled"},
        "primitives": {"library": library_manifest(),
                       "degraded": presentation.primitive_log.degraded if presentation.primitive_log else []},
        "emphasis_reasons": {str(wid): {"reason": reason, "source": source}
                             for wid, (reason, source) in sorted(presentation.reasons.items())},
        "shaping": {"shaper": presentation.shaper_note, **kerning_report(presentation)},
        "legibility": {"version": LEGIBILITY_VERSION, "counts": metrics["legibility"],
                       "decisions": [d.to_dict() for d in presentation.legibility[:page_limit]]},
        "energy": {str(k): v for k, v in sorted(presentation.energy_overrides.items())},
        "pages": rows,
        "pages_truncated": max(0, len(presentation.pages) - page_limit),
        "ass_signature": presentation.ass_signature,
    }


def write_presentation(presentation: CaptionPresentation, ass_path: Path) -> Path:
    ass_path.parent.mkdir(parents=True, exist_ok=True)
    temp = ass_path.with_name(ass_path.name + ".tmp")
    temp.write_text(presentation.ass_text, encoding="utf-8-sig")
    temp.replace(ass_path)
    return ass_path


def first_dialogue_start(path: str | Path | None) -> float | None:
    """Same parser the intro renderer uses for the post-intro restart."""
    from ai.editor import intro_renderer

    return intro_renderer.get_first_caption_start(path)


def check_intro_handoff(presentation_ass: Path, baseline_ass: Path) -> None:
    """The intro renderer derives the main restart from the BASELINE ASS; the
    presentation must start its first caption at the same instant."""
    ours, theirs = first_dialogue_start(presentation_ass), first_dialogue_start(baseline_ass)
    if theirs is None:
        return
    if ours is None or abs(ours - theirs) > 0.011:
        raise CaptionPresentationError(f"first caption {ours} != baseline {theirs}; restart math would differ")


def mark_caption_channel(recommendations: Iterable[Mapping[str, Any]],
                         presentation: CaptionPresentation | None) -> tuple[dict[str, Any], ...]:
    executed = set()
    if presentation is not None:
        executed = {d["event_id"] for d in build_manifest(presentation, page_limit=0)["directives"] if d["executed"]}
    rows = []
    for row in recommendations:
        row = dict(row)
        if row.get("channel") == "caption_style":
            row["executed"] = row.get("event_id") in executed
            row["executor"] = "caption_presentation" if row["executed"] else "not_executed"
        rows.append(row)
    return tuple(rows)
