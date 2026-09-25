"""MIMIR caption visual primitives: a small, original, deterministic library.

Five primitives formalise what V3.1 already renders (no duplicate effects);
two are new and opt-in per brand profile (``mimir_default`` keeps V3.1 exactly):

    STATIC                  spoken words, style colour                 (V3.1)
    ACTIVE_COLOR_EASE       active word colour ease inside its interval (V3.1)
    ACTIVE_OUTLINE_EASE     active word outline swell (EMPHASIS/IMPACT)  (V3.1)
    STATIC_EMPHASIS_SCALE   emphasized word scale reserved in layout    (V3.1)
    SOFT_IMPACT_OUTLINE     stronger outline accent, IMPACT emphasis    (V3.1)
    PAGE_ALPHA_IN           bounded page fade-in (opt-in)
    STATIC_ACCENT_BACKPLATE static plate behind a page/word (opt-in; counted
                            by the caption safe region)

V5 adds five (library total 12, the stated maximum):

    STATIC_UNDERLINE_ACCENT static bar inside the line box under a NAME (opt-in)
    STATIC_NUMBER_BADGE     static word plate with an accent border for a NUMBER (opt-in, strong)
    STATIC_CONTRAST_ACCENT  outline takes the emphasis colour, fill stays base: CONTRAST (opt-in)
    LEGIBILITY_OUTLINE_BOOST page outline x <= 1.5 when the background needs it (legibility)
    LEGIBILITY_SHADOW       page drop shadow when the outline alone is not enough (legibility)

Reason -> primitive table (deterministic; a primitive the brand does not allow
falls back to the V4 behaviour for that reason):

    name      STATIC_UNDERLINE_ACCENT   (V4: word plate on EMPHASIS/IMPACT pages)
    number    STATIC_NUMBER_BADGE       (V4: word plate on EMPHASIS/IMPACT pages)
    contrast  STATIC_CONTRAST_ACCENT
    payoff    SOFT_IMPACT_OUTLINE on IMPACT pages, ACTIVE_OUTLINE_EASE otherwise (V3.1)
    reaction  ACTIVE_OUTLINE_EASE (V3.1)
    surprise  SOFT_IMPACT_OUTLINE on IMPACT pages only (strong, sparse; V3.1)
    generic   style colour (V3.1)

Geometry law: no primitive animates position or size. Outline thickness and
colour/alpha are not laid out by libass (verified in tests), a static scale is
reserved from the page's first frame, a plate is static and visible to
caption_guard. Selection is a deterministic table over (caption style,
emphasis reason, brand); the planner never names a primitive.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from ai.editor.pro_edit.schema import CaptionStyle

PRIMITIVE_LIBRARY_VERSION = 2
MAX_LIBRARY_SIZE = 12


class PrimitiveId(str, Enum):
    STATIC = "static"
    ACTIVE_COLOR_EASE = "active_color_ease"
    ACTIVE_OUTLINE_EASE = "active_outline_ease"
    STATIC_EMPHASIS_SCALE = "static_emphasis_scale"
    SOFT_IMPACT_OUTLINE = "soft_impact_outline"
    PAGE_ALPHA_IN = "page_alpha_in"
    STATIC_ACCENT_BACKPLATE = "static_accent_backplate"
    STATIC_UNDERLINE_ACCENT = "static_underline_accent"
    STATIC_NUMBER_BADGE = "static_number_badge"
    STATIC_CONTRAST_ACCENT = "static_contrast_accent"
    LEGIBILITY_OUTLINE_BOOST = "legibility_outline_boost"
    LEGIBILITY_SHADOW = "legibility_shadow"


_ALL_STYLES = frozenset(CaptionStyle)
_ACCENT_STYLES = frozenset({CaptionStyle.EMPHASIS, CaptionStyle.IMPACT})


@dataclass(frozen=True)
class CaptionPrimitive:
    primitive_id: PrimitiveId
    version: int
    geometry_behavior: str
    timing_behavior: str
    safe_bounds: str
    supported_styles: frozenset[CaptionStyle]
    strength: str                         # base | soft | strong
    fallback: PrimitiveId | None
    validation: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.primitive_id.value, "version": self.version, "geometry": self.geometry_behavior,
                "timing": self.timing_behavior, "safe_bounds": self.safe_bounds,
                "styles": sorted(s.value for s in self.supported_styles), "strength": self.strength,
                "fallback": self.fallback.value if self.fallback else None, "validation": self.validation}


P = PrimitiveId
PRIMITIVE_LIBRARY: Mapping[PrimitiveId, CaptionPrimitive] = {
    P.STATIC: CaptionPrimitive(
        P.STATIC, 1, "none", "none", "glyph boxes + style outline", _ALL_STYLES, "base", None,
        "always valid"),
    P.ACTIVE_COLOR_EASE: CaptionPrimitive(
        P.ACTIVE_COLOR_EASE, 1, "none (colour only)", "attack <= 150 ms inside the active word's interval",
        "unchanged", _ALL_STYLES, "base", P.STATIC, "\\t end <= interval; no geometry tag inside \\t"),
    P.ACTIVE_OUTLINE_EASE: CaptionPrimitive(
        P.ACTIVE_OUTLINE_EASE, 1, "none (libass does not lay out borders)",
        "outline swell: attack + release inside the active word's interval",
        "largest \\bord is counted by caption_guard", _ACCENT_STYLES, "soft", P.ACTIVE_COLOR_EASE,
        "\\t end <= interval; only \\c/\\bord inside \\t"),
    P.STATIC_EMPHASIS_SCALE: CaptionPrimitive(
        P.STATIC_EMPHASIS_SCALE, 1, "static_reserved (scale fixed from the page's first frame)", "none",
        "line width measured with the scale; \\fscy counted by caption_guard", _ACCENT_STYLES, "soft", P.STATIC,
        "identical scale tags in every state of a page"),
    P.SOFT_IMPACT_OUTLINE: CaptionPrimitive(
        P.SOFT_IMPACT_OUTLINE, 1, "none", "stronger outline swell inside the emphasized word's interval",
        "largest \\bord is counted by caption_guard", frozenset({CaptionStyle.IMPACT}), "strong",
        P.ACTIVE_OUTLINE_EASE, "IMPACT pages only; page/clip emphasis budgets"),
    P.PAGE_ALPHA_IN: CaptionPrimitive(
        P.PAGE_ALPHA_IN, 1, "none (event alpha)",
        "\\fad attack <= 60 ms and <= 1/3 of the first word interval; off for pages shorter than 0.6 s",
        "unchanged", _ALL_STYLES, "soft", P.STATIC, "word onset unchanged; fade never on a later line"),
    P.STATIC_ACCENT_BACKPLATE: CaptionPrimitive(
        P.STATIC_ACCENT_BACKPLATE, 1, "static_box (drawn plate behind text)", "none (page lifetime)",
        "plate drawing is parsed by caption_guard (\\p drawing bounds)", _ALL_STYLES, "strong", P.STATIC,
        "strong-primitive budget; plate inside frame"),
    P.STATIC_UNDERLINE_ACCENT: CaptionPrimitive(
        P.STATIC_UNDERLINE_ACCENT, 1, "static_box inside the line box (below the baseline, above the box bottom)",
        "none (page lifetime, from the word's onset)", "inside the text line box: safe region unchanged",
        _ALL_STYLES, "soft", P.STATIC, "bar inside the measured line box; drawing parsed by caption_guard"),
    P.STATIC_NUMBER_BADGE: CaptionPrimitive(
        P.STATIC_NUMBER_BADGE, 1, "static_box (word plate + accent border)", "none (from the word's onset)",
        "plate + border parsed by caption_guard", _ALL_STYLES, "strong", P.STATIC,
        "strong-primitive budget; no badge on a page that already has a page plate"),
    P.STATIC_CONTRAST_ACCENT: CaptionPrimitive(
        P.STATIC_CONTRAST_ACCENT, 1, "none (outline colour only)", "none", "unchanged", _ALL_STYLES, "soft",
        P.STATIC, "only \\3c changes; fill keeps the lane colour"),
    P.LEGIBILITY_OUTLINE_BOOST: CaptionPrimitive(
        P.LEGIBILITY_OUTLINE_BOOST, 1, "none (libass does not lay out borders)", "none (page lifetime)",
        "boosted \\bord is counted by the band and by caption_guard", _ALL_STYLES, "base", P.STATIC,
        "boost <= 1.5; only when the legibility analysis asks for it"),
    P.LEGIBILITY_SHADOW: CaptionPrimitive(
        P.LEGIBILITY_SHADOW, 1, "none (shadow offset is not laid out)", "none (page lifetime)",
        "\\shad is counted by the band and by caption_guard", _ALL_STYLES, "base", P.LEGIBILITY_OUTLINE_BOOST,
        "only when the legibility analysis asks for it"),
}
REASON_PRIMITIVES = frozenset({P.STATIC_UNDERLINE_ACCENT, P.STATIC_NUMBER_BADGE, P.STATIC_CONTRAST_ACCENT})
LEGIBILITY_PRIMITIVES = frozenset({P.LEGIBILITY_OUTLINE_BOOST, P.LEGIBILITY_SHADOW})
LEGIBILITY_PLATE_MAX_RATIO = 0.35     # of all pages; the rest fall back to outline + shadow
V31_PRIMITIVES = frozenset({P.STATIC, P.ACTIVE_COLOR_EASE, P.ACTIVE_OUTLINE_EASE, P.STATIC_EMPHASIS_SCALE,
                            P.SOFT_IMPACT_OUTLINE})
DEFAULT_PRIMITIVES = V31_PRIMITIVES
REQUIRED_PRIMITIVES = frozenset({P.STATIC, P.ACTIVE_COLOR_EASE})
PAGE_FADE_MAX_MS = 60
PAGE_FADE_MIN_PAGE_S = 0.6


@dataclass(frozen=True)
class PrimitiveBudget:
    """Per-short limits for STRONG new primitives (V3.1 budgets stay in the style engine)."""

    strong_min_gap_s: float = 3.0
    strong_per_10s: int = 2
    strong_max_page_ratio: float = 0.25

    @classmethod
    def parse(cls, raw: Any) -> "PrimitiveBudget":
        if not isinstance(raw, Mapping) or set(raw) - {"strong_min_gap_s", "strong_per_10s", "strong_max_page_ratio"}:
            raise ValueError("primitive_budget must be an object with known keys only")
        values: dict[str, Any] = {}
        for key, low, high in (("strong_min_gap_s", 0.5, 30.0), ("strong_per_10s", 0, 10),
                               ("strong_max_page_ratio", 0.0, 0.5)):
            if key in raw:
                value = raw[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                        or not low <= value <= high:
                    raise ValueError(f"primitive_budget.{key} must be in [{low}, {high}]")
                values[key] = int(value) if key == "strong_per_10s" else float(value)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {"strong_min_gap_s": self.strong_min_gap_s, "strong_per_10s": self.strong_per_10s,
                "strong_max_page_ratio": self.strong_max_page_ratio}


@dataclass(frozen=True)
class PageEffects:
    """Resolved primitives for one page (what the ASS writer may emit)."""

    page_id: int
    primitives: frozenset[PrimitiveId]
    fade_in_ms: int = 0
    plate_lines: bool = False
    plate_word_ids: frozenset[int] = frozenset()
    underline_word_ids: frozenset[int] = frozenset()
    badge_word_ids: frozenset[int] = frozenset()
    contrast_word_ids: frozenset[int] = frozenset()
    plate_reason: str = ""                     # busy_background | legibility | reason

    @property
    def draws(self) -> bool:
        """True when the page emits vector drawings (plates, badges, underlines)."""
        return bool(self.plate_lines or self.plate_word_ids or self.underline_word_ids or self.badge_word_ids)

    def has(self, primitive: PrimitiveId) -> bool:
        return primitive in self.primitives


@dataclass(frozen=True)
class PageFacts:
    """What the resolver needs to know about a page (no pixels)."""

    page_id: int
    style: CaptionStyle
    start: float
    end: float
    emphasized: Mapping[int, str]              # word_id -> emphasis reason
    busy_background: bool = False              # placement could not avoid visual activity
    legibility_plate: bool = False             # legibility analysis: plate (already budgeted)
    legibility_outline: bool = False           # legibility analysis: outline boost
    legibility_shadow: bool = False            # legibility analysis: drop shadow


@dataclass
class PrimitiveLog:
    degraded: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def count(self, primitive: PrimitiveId) -> None:
        self.counts[primitive.value] = self.counts.get(primitive.value, 0) + 1


def _base_set(style: CaptionStyle, allowed: frozenset[PrimitiveId]) -> set[PrimitiveId]:
    chosen = {P.STATIC, P.ACTIVE_COLOR_EASE}
    if style in _ACCENT_STYLES:
        for primitive in (P.ACTIVE_OUTLINE_EASE, P.STATIC_EMPHASIS_SCALE):
            if primitive in allowed:
                chosen.add(primitive)
    if style is CaptionStyle.IMPACT:
        if P.SOFT_IMPACT_OUTLINE in allowed:
            chosen.add(P.SOFT_IMPACT_OUTLINE)
    return chosen


def resolve_page_effects(pages: Sequence[PageFacts], allowed: Iterable[PrimitiveId], budget: PrimitiveBudget,
                         ) -> tuple[list[PageEffects], PrimitiveLog]:
    """Deterministic table (style x reason x brand x legibility) + strong-primitive budget.

    Plates: a legibility plate when the legibility analysis asked for one (it is
    budgeted there); a page plate when placement reports a busy background it
    could not avoid (brand opt-in); reason treatments per the module table.
    Budget overflow degrades to the fallback and is logged.
    """
    allowed_set = frozenset(allowed) | REQUIRED_PRIMITIVES
    log = PrimitiveLog()
    strong_times: list[float] = []
    max_strong_pages = int(math.floor(budget.strong_max_page_ratio * len(pages)))
    strong_pages = 0
    effects: list[PageEffects] = []

    def strong_available(page: PageFacts) -> bool:
        recent = [t for t in strong_times if page.start - t < 10.0]
        too_close = bool(strong_times) and page.start - strong_times[-1] < budget.strong_min_gap_s
        return not (too_close or len(recent) >= budget.strong_per_10s or strong_pages >= max_strong_pages)

    for page in pages:
        chosen = _base_set(page.style, allowed_set)
        fade = 0
        if P.PAGE_ALPHA_IN in allowed_set and page.end - page.start >= PAGE_FADE_MIN_PAGE_S:
            chosen.add(P.PAGE_ALPHA_IN)
            fade = PAGE_FADE_MAX_MS
        if page.legibility_outline:
            chosen.add(P.LEGIBILITY_OUTLINE_BOOST)
        if page.legibility_shadow:
            chosen.add(P.LEGIBILITY_SHADOW)
        plate_lines = False
        plate_reason = ""
        plate_words: set[int] = set()
        underline: set[int] = set()
        badges: set[int] = set()
        contrast: set[int] = set()
        if page.legibility_plate:
            chosen.add(P.STATIC_ACCENT_BACKPLATE)
            plate_lines, plate_reason = True, "legibility"
        # Reason grammar (new primitives only when the brand allows them).
        for word_id, reason in sorted(page.emphasized.items()):
            if reason == "name" and P.STATIC_UNDERLINE_ACCENT in allowed_set:
                underline.add(word_id)
            elif reason == "number" and P.STATIC_NUMBER_BADGE in allowed_set and not plate_lines:
                badges.add(word_id)
            elif reason == "contrast" and P.STATIC_CONTRAST_ACCENT in allowed_set:
                contrast.add(word_id)
        if badges:
            if strong_available(page):
                chosen.add(P.STATIC_NUMBER_BADGE)
                strong_times.append(page.start)
                strong_pages += 1
            else:
                log.degraded.append({"page_id": page.page_id, "primitive": P.STATIC_NUMBER_BADGE.value,
                                     "reason": "strong_primitive_budget", "fallback": P.STATIC.value})
                badges = set()
        if underline:
            chosen.add(P.STATIC_UNDERLINE_ACCENT)
        if contrast:
            chosen.add(P.STATIC_CONTRAST_ACCENT)
        if not plate_lines and P.STATIC_ACCENT_BACKPLATE in allowed_set:
            wants_page = page.busy_background
            wants_words = {wid for wid, reason in page.emphasized.items()
                           if reason in ("name", "number") and page.style in _ACCENT_STYLES
                           and wid not in underline and wid not in badges}
            if wants_page or wants_words:
                if not strong_available(page):
                    log.degraded.append({"page_id": page.page_id, "primitive": P.STATIC_ACCENT_BACKPLATE.value,
                                         "reason": "strong_primitive_budget", "fallback": P.STATIC.value})
                else:
                    chosen.add(P.STATIC_ACCENT_BACKPLATE)
                    plate_lines = wants_page
                    plate_reason = "busy_background" if wants_page else "reason"
                    plate_words = set() if wants_page else wants_words
                    strong_times.append(page.start)
                    strong_pages += 1
        for primitive in chosen:
            log.count(primitive)
        effects.append(PageEffects(page.page_id, frozenset(chosen), fade, plate_lines, frozenset(plate_words),
                                   frozenset(underline), frozenset(badges), frozenset(contrast), plate_reason))
    return effects, log


def library_manifest() -> list[dict[str, Any]]:
    return [primitive.to_dict() for primitive in PRIMITIVE_LIBRARY.values()]
