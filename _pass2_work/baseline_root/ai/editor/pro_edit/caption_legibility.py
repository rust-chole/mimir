"""Caption LEGIBILITY: deterministic contrast analysis + minimum intervention.

Motion is not readability: a static bright or busy background can make white
captions unreadable. For every caption page, at its placed position, the
background statistics (``caption_background``) are compared with the page's
ACTUAL appearance (brand fill colours of every visible state, outline colour
and width, shadow, plate) using WCAG relative luminance / contrast ratios:

    fill_contrast   worst ratio between any visible fill colour and the
                    background luminance range (p10 / median / p90)
    halo_contrast   fill colour vs outline colour (can the outline separate
                    the glyph from any background?)
    complexity      edge density + text-like clutter under the text

Ladder (least intrusive first, never skipping a rung that suffices):

    LEGIBILITY_CLEAR          no change
    LEGIBILITY_NEEDS_OUTLINE  stronger outline (x <= 1.5)
    LEGIBILITY_NEEDS_SHADOW   stronger outline + drop shadow
    LEGIBILITY_NEEDS_PLATE    static plate behind the page (budgeted)
    LEGIBILITY_UNCERTAIN      too little evidence -> no change

Never changes text, timing, position or line breaks. No LLM. Thresholds are
engineering calibrations (monotone by construction and by test; absolute
values are MANUAL_VISUAL_CALIBRATION).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from ai.editor.pro_edit.caption_background import RegionStats, relative_luminance

LEGIBILITY_VERSION = 1

TARGET_CONTRAST = 4.5          # WCAG AA body text; captions move, so the AA target is kept
MIN_CONTRAST = 3.0             # WCAG AA large text
HALO_CONTRAST = 4.5            # an outline this far from the fill separates the glyph on any background
CLEAN_COMPLEXITY = 0.10        # edge density below which the fill contrast alone decides
SHADOW_COMPLEXITY = 0.20       # above: outline alone is not enough
PLATE_COMPLEXITY = 0.34        # above: busy detail competes with strokes -> plate
TEXT_CLUTTER = 0.25            # text-like coverage under the caption counts as complexity
MAX_OUTLINE_BOOST = 1.5
OUTLINE_NEED_LOW = 0.065       # outline / font px needed when fill contrast is < MIN_CONTRAST
OUTLINE_NEED_MID = 0.05        # ... between MIN_CONTRAST and TARGET_CONTRAST
SHADOW_OFFSET_RATIO = 0.045    # shadow offset / font px
MAX_PLATE_OPACITY = 0.90       # = the brand maximum; a legibility plate never hides more than this
MIN_CONFIDENCE = 0.5
MIN_SAMPLES = 1


class Legibility(str, Enum):
    CLEAR = "LEGIBILITY_CLEAR"
    NEEDS_OUTLINE = "LEGIBILITY_NEEDS_OUTLINE"
    NEEDS_SHADOW = "LEGIBILITY_NEEDS_SHADOW"
    NEEDS_PLATE = "LEGIBILITY_NEEDS_PLATE"
    UNCERTAIN = "LEGIBILITY_UNCERTAIN"


LADDER = (Legibility.CLEAR, Legibility.NEEDS_OUTLINE, Legibility.NEEDS_SHADOW, Legibility.NEEDS_PLATE)


def ass_colour_luminance(colour: str) -> float:
    """Relative luminance of an ASS colour (&HBBGGRR& or &HAABBGGRR)."""
    digits = colour.strip().upper().removeprefix("&H").rstrip("&")
    digits = digits[-6:].rjust(6, "0")
    b, g, r = int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)
    return relative_luminance(r, g, b)


def contrast_ratio(a: float, b: float) -> float:
    high, low = max(a, b), min(a, b)
    return (high + 0.05) / (low + 0.05)


@dataclass(frozen=True)
class TextAppearance:
    """What the viewer sees for one page (brand + style + primitives)."""

    fill_colours: tuple[str, ...]      # every fill a word can show on this page
    outline_colour: str
    outline_px: float
    font_px: float
    shadow_px: float
    plate_colour: str
    plate_alpha: int                   # ASS alpha (0 opaque .. 255 transparent)

    def identity(self) -> dict[str, Any]:
        return {"fills": list(self.fill_colours), "outline": [self.outline_colour, round(self.outline_px, 2)],
                "font_px": round(self.font_px, 2), "shadow_px": round(self.shadow_px, 2),
                "plate": [self.plate_colour, self.plate_alpha]}


@dataclass(frozen=True)
class LegibilityDecision:
    page_id: int
    verdict: Legibility
    outline_boost: float = 1.0
    shadow_px: float | None = None
    plate: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    key: str = ""
    degraded_from: str = ""
    plate_alpha: int | None = None      # ASS alpha of a legibility plate (opacity sized to the background)

    def to_dict(self) -> dict[str, Any]:
        row = {"page_id": self.page_id, "verdict": self.verdict.value, "outline_boost": self.outline_boost,
               "shadow_px": self.shadow_px, "plate": self.plate, "plate_alpha": self.plate_alpha,
               "metrics": self.metrics, "key": self.key}
        if self.degraded_from:
            row["degraded_from"] = self.degraded_from
        return row


def decision_key(page_id: int, start: float, end: float, box: Sequence[float], appearance: TextAppearance,
                 background_fingerprint: str) -> str:
    """Identity of one verdict: clip analysis + page timing + region + style + version."""
    blob = json.dumps({"v": LEGIBILITY_VERSION, "page": page_id, "t": [round(start, 3), round(end, 3)],
                       "box": [round(v, 4) for v in box], "style": appearance.identity(),
                       "background": background_fingerprint}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _worst_fill_contrast(appearance: TextAppearance, stats: RegionStats) -> float:
    backgrounds = (stats.lum_p10, stats.lum_p50, stats.lum_p90)
    return min(contrast_ratio(ass_colour_luminance(fill), bg) for fill in appearance.fill_colours
               for bg in backgrounds)


def _halo_contrast(appearance: TextAppearance) -> float:
    outline = ass_colour_luminance(appearance.outline_colour)
    return min(contrast_ratio(ass_colour_luminance(fill), outline) for fill in appearance.fill_colours)


def _plate_contrast(appearance: TextAppearance, stats: RegionStats, opacity: float) -> float:
    """Fill vs the plate composited over the brightest / darkest background."""
    plate = ass_colour_luminance(appearance.plate_colour)
    worst = []
    for bg in (stats.lum_p10, stats.lum_p90):
        composite = opacity * plate + (1.0 - opacity) * bg
        worst.extend(contrast_ratio(ass_colour_luminance(fill), composite) for fill in appearance.fill_colours)
    return min(worst)


def plate_opacity(appearance: TextAppearance, stats: RegionStats) -> float | None:
    """Least plate opacity (>= the brand's) that reaches TARGET_CONTRAST, or None if even
    MAX_PLATE_OPACITY does not (the plate would not help)."""
    brand = 1.0 - appearance.plate_alpha / 255.0
    steps = [brand] + [v / 100.0 for v in range(int(brand * 100) + 1, int(MAX_PLATE_OPACITY * 100) + 1)]
    for opacity in steps:
        if opacity <= MAX_PLATE_OPACITY + 1e-9 or opacity == brand:
            if _plate_contrast(appearance, stats, opacity) >= TARGET_CONTRAST:
                return round(opacity, 2)
    return None


def assess(page_id: int, stats: RegionStats | None, appearance: TextAppearance, *, key: str = "",
           allow_plate: bool = True) -> LegibilityDecision:
    """Least intrusive treatment that meets the deterministic readability constraints."""
    if stats is None or stats.samples < MIN_SAMPLES or stats.confidence < MIN_CONFIDENCE:
        return LegibilityDecision(page_id, Legibility.UNCERTAIN, metrics={"reason": "insufficient background samples"},
                                  key=key)
    fill = _worst_fill_contrast(appearance, stats)
    halo = _halo_contrast(appearance)
    complexity = stats.edge_density + (TEXT_CLUTTER * stats.text_coverage)
    outline_ratio = appearance.outline_px / max(appearance.font_px, 1.0)
    metrics = {"fill_contrast": round(fill, 3), "halo_contrast": round(halo, 3), "complexity": round(complexity, 4),
               "outline_ratio": round(outline_ratio, 4), "background": stats.to_dict()}

    if fill >= TARGET_CONTRAST:
        needed = 0.0
    elif fill >= MIN_CONTRAST:
        needed = OUTLINE_NEED_MID
    else:
        needed = OUTLINE_NEED_LOW
    outline_helps = halo >= HALO_CONTRAST
    if needed > 0 and not outline_helps:
        needed = float("inf")          # the outline cannot separate the glyph (fill ~ outline colour)
    metrics["outline_needed_ratio"] = None if needed == float("inf") else round(needed, 4)

    # Below SHADOW_COMPLEXITY the existing outline decides; on clean backgrounds (< CLEAN_COMPLEXITY) the
    # fill contrast alone would already do, but the verdict is the same: no change.
    if complexity < SHADOW_COMPLEXITY and outline_ratio >= needed:
        return LegibilityDecision(page_id, Legibility.CLEAR, metrics=metrics, key=key)
    shadow = round(max(2.0, SHADOW_OFFSET_RATIO * appearance.font_px), 1)
    if complexity < SHADOW_COMPLEXITY and outline_ratio * MAX_OUTLINE_BOOST >= needed:
        boost = min(MAX_OUTLINE_BOOST, max(1.0, needed / max(outline_ratio, 1e-6)))
        boost = round(min(MAX_OUTLINE_BOOST, (int(boost * 20 + 0.999)) / 20.0), 2)   # up to the next 0.05
        return LegibilityDecision(page_id, Legibility.NEEDS_OUTLINE, outline_boost=max(1.05, boost),
                                  metrics=metrics, key=key)
    if complexity < PLATE_COMPLEXITY and outline_helps:
        return LegibilityDecision(page_id, Legibility.NEEDS_SHADOW, outline_boost=MAX_OUTLINE_BOOST,
                                  shadow_px=max(shadow, appearance.shadow_px), metrics=metrics, key=key)
    opacity = plate_opacity(appearance, stats)
    metrics["plate_opacity"] = opacity
    if allow_plate and opacity is not None:
        metrics["plate_contrast"] = round(_plate_contrast(appearance, stats, opacity), 3)
        return LegibilityDecision(page_id, Legibility.NEEDS_PLATE, plate=True, metrics=metrics, key=key,
                                  plate_alpha=int(round((1.0 - opacity) * 255)))
    # No plate (budget / brand / plate would not help): strongest non-plate treatment.
    return LegibilityDecision(page_id, Legibility.NEEDS_SHADOW, outline_boost=MAX_OUTLINE_BOOST,
                              shadow_px=max(shadow, appearance.shadow_px), metrics=metrics, key=key,
                              degraded_from=Legibility.NEEDS_PLATE.value)


def rank(verdict: Legibility) -> int:
    return LADDER.index(verdict) if verdict in LADDER else -1
