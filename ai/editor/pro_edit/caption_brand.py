"""MIMIR caption BrandProfile: a validated configuration model (no GUI, no raw ASS).

``mimir_default`` is built from the caption module's own constants, so it
reproduces the V3.1 presentation exactly (verified against frozen V3.1 output).
A custom profile is a strict JSON file (``MIMIR_CAPTION_BRAND_PROFILE=<path>``);
any invalid value rejects the whole profile and ``mimir_default`` is used.
Colours are ``#RRGGBB``; no ASS code or FFmpeg expression is accepted anywhere.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from ai.editor import captions as caption_truth
from ai.editor.pro_edit.caption_primitives import (
    DEFAULT_PRIMITIVES,
    PRIMITIVE_LIBRARY,
    REQUIRED_PRIMITIVES,
    REASON_PRIMITIVES,
    PrimitiveBudget,
    PrimitiveId,
)
from ai.editor.pro_edit.schema import CaptionStyle

BRAND_SCHEMA_VERSION = 1
_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")
_LANES = ("main", "secondary", "tertiary")
_FLAT_COLOURS = {
    "primary_text_color": ("main", "text"), "speaker_primary_color": ("main", "text"),
    "active_text_color": ("main", "active"), "emphasis_text_color": ("main", "emphasis"),
    "speaker_secondary_color": ("secondary", "text"), "speaker_secondary_active_color": ("secondary", "active"),
    "speaker_secondary_emphasis_color": ("secondary", "emphasis"),
    "speaker_tertiary_color": ("tertiary", "text"), "speaker_tertiary_active_color": ("tertiary", "active"),
    "speaker_tertiary_emphasis_color": ("tertiary", "emphasis"),
}
# Colour of a word whose voice is uncertain inside a multi-voice clip.
_NEUTRAL_COLOURS = {
    "speaker_neutral_color": "text", "speaker_neutral_active_color": "active",
    "speaker_neutral_emphasis_color": "emphasis",
}
_KEYS = frozenset({
    "profile_id", "version", "font_family", "optional_font_path", "outline_color", "outline_scale", "shadow_scale",
    "caption_anchor_policy", "safe_width_ratio", "default_style", "allowed_primitives", "backplate_color",
    "backplate_opacity", "primitive_budget", "legibility", *_FLAT_COLOURS, *_NEUTRAL_COLOURS,
})
ANCHOR_POLICIES = ("auto", "bottom")
# auto: legibility analysis may boost outline / add shadow / add a (budgeted) plate;
# no_plate: outline + shadow only; off: never change the brand look for legibility.
LEGIBILITY_POLICIES = ("auto", "no_plate", "off")


class BrandProfileError(ValueError):
    """Invalid brand profile data."""


def ass_colour(hex_rgb: str) -> str:
    """#RRGGBB -> &HBBGGRR& (override tag form)."""
    value = hex_rgb.lstrip("#").upper()
    return f"&H{value[4:6]}{value[2:4]}{value[0:2]}&"


def ass_style_colour(hex_rgb: str, alpha: int = 0) -> str:
    """#RRGGBB -> &HAABBGGRR (style line form)."""
    value = hex_rgb.lstrip("#").upper()
    return f"&H{alpha:02X}{value[4:6]}{value[2:4]}{value[0:2]}"


def _hex_from_ass(value: str) -> str:
    digits = value.strip().upper().lstrip("&H").rstrip("&")[-6:]
    return f"#{digits[4:6]}{digits[2:4]}{digits[0:2]}"


@dataclass(frozen=True)
class LaneColours:
    """Override colours in ASS form (&HBBGGRR&) + the style line PrimaryColour."""

    style_primary: str
    text: str
    active: str
    emphasis: str


@dataclass(frozen=True)
class BrandProfile:
    profile_id: str
    version: int
    font_family: str
    optional_font_path: str | None
    lanes: Mapping[str, LaneColours]          # lane STYLE lines (where a page sits)
    outline_color: str                        # style form &HAABBGGRR
    outline_scale: float = 1.0
    shadow_scale: float = 1.0
    caption_anchor_policy: str = "auto"
    safe_width_ratio: float = 1.0
    default_style: CaptionStyle = CaptionStyle.DEFAULT
    allowed_primitives: frozenset[PrimitiveId] = DEFAULT_PRIMITIVES
    backplate_color: str = "&H000000&"
    backplate_alpha: int = 0x8C               # ASS alpha (0 = opaque, 255 = invisible)
    budget: PrimitiveBudget = field(default_factory=PrimitiveBudget)
    legibility: str = "auto"
    # Explicit speaker colours ("C", "neutral"); missing ones follow speaker_colours().
    speakers: Mapping[str, LaneColours] = field(default_factory=dict)

    @property
    def is_default(self) -> bool:
        return self.profile_id == "mimir_default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id, "version": self.version, "font_family": self.font_family,
            "optional_font_path": self.optional_font_path,
            "lanes": {lane: {"text": _hex_from_ass(c.text), "active": _hex_from_ass(c.active),
                             "emphasis": _hex_from_ass(c.emphasis)} for lane, c in self.lanes.items()},
            "outline_color": _hex_from_ass(self.outline_color), "outline_scale": self.outline_scale,
            "shadow_scale": self.shadow_scale, "caption_anchor_policy": self.caption_anchor_policy,
            "safe_width_ratio": self.safe_width_ratio, "default_style": self.default_style.value,
            "allowed_primitives": sorted(p.value for p in self.allowed_primitives),
            "budget": self.budget.to_dict(), "legibility": self.legibility,
            "speakers": {key or "plain": {"text": _hex_from_ass(c.text), "active": _hex_from_ass(c.active),
                                          "emphasis": _hex_from_ass(c.emphasis)}
                         for key, c in speaker_colours(self).items()},
        }


def _v31_colour(value: str) -> str:
    digits = value.strip().upper().lstrip("&H").rstrip("&")
    return f"&H{digits[-6:]}&"


_MAIN = LaneColours(caption_truth.BASE_TEXT_COLOR, _v31_colour(caption_truth.INACTIVE_TEXT_COLOR),
                    _v31_colour(caption_truth.ACTIVE_TEXT_COLOR), _v31_colour(caption_truth.HIGHLIGHT_TEXT_COLOR))
_THIRD_VOICE = LaneColours(caption_truth.TERTIARY_BASE_TEXT_COLOR, _v31_colour(caption_truth.TERTIARY_BASE_TEXT_COLOR),
                           _v31_colour(caption_truth.TERTIARY_ACTIVE_TEXT_COLOR),
                           _v31_colour(caption_truth.TERTIARY_HIGHLIGHT_TEXT_COLOR))
_NEUTRAL_VOICE = LaneColours(caption_truth.NEUTRAL_BASE_TEXT_COLOR, _v31_colour(caption_truth.NEUTRAL_BASE_TEXT_COLOR),
                             _v31_colour(caption_truth.NEUTRAL_ACTIVE_TEXT_COLOR),
                             _v31_colour(caption_truth.NEUTRAL_HIGHLIGHT_TEXT_COLOR))


def speaker_colours(brand: "BrandProfile") -> dict[str, LaneColours]:
    """Speaker colour key -> colours. A colour names a VOICE (captions V25), never
    a lane: "" (plain look) and "A" are the main colours, "B" the secondary
    colours, "C" a third palette, "neutral" an uncertain voice (no hue)."""
    return {
        "": brand.lanes["main"],
        "A": brand.lanes["main"],
        "B": brand.lanes["secondary"],
        "C": brand.speakers.get("C", _THIRD_VOICE),
        "neutral": brand.speakers.get("neutral", _NEUTRAL_VOICE),
    }
MIMIR_DEFAULT = BrandProfile(
    profile_id="mimir_default", version=1, font_family=caption_truth.FONT_NAME, optional_font_path=None,
    lanes={
        "main": _MAIN,
        "secondary": LaneColours(caption_truth.SECONDARY_BASE_TEXT_COLOR,
                                 _v31_colour(caption_truth.SECONDARY_BASE_TEXT_COLOR),
                                 _v31_colour(caption_truth.SECONDARY_ACTIVE_TEXT_COLOR),
                                 _v31_colour(caption_truth.SECONDARY_HIGHLIGHT_TEXT_COLOR)),
        "tertiary": _MAIN,       # baseline renders a third speaker with the main palette
    },
    outline_color=caption_truth.OUTLINE_COLOR,
)


# Built-in opt-in look: mimir_default + the V5 emphasis-reason grammar (underline for
# names, badge for numbers, contrast outline). Colours / font / geometry stay V3.1.
MIMIR_EXPRESSIVE = replace(MIMIR_DEFAULT, profile_id="mimir_expressive",
                           allowed_primitives=DEFAULT_PRIMITIVES | REASON_PRIMITIVES)
BUILTIN_BRANDS: Mapping[str, BrandProfile] = {"mimir_default": MIMIR_DEFAULT, "mimir_expressive": MIMIR_EXPRESSIVE}


def _number(data: Mapping[str, Any], key: str, default: float, low: float, high: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise BrandProfileError(f"{key} must be a finite number")
    if not low <= float(value) <= high:
        raise BrandProfileError(f"{key} must be in [{low}, {high}]")
    return float(value)


def _colour(data: Mapping[str, Any], key: str) -> str | None:
    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, str) or not _HEX.match(value):
        raise BrandProfileError(f"{key} must be #RRGGBB")
    return value


def parse_brand(data: Any, *, base: BrandProfile = MIMIR_DEFAULT) -> BrandProfile:
    """Strict validation; missing fields inherit ``base`` (mimir_default)."""
    if not isinstance(data, Mapping):
        raise BrandProfileError("brand profile must be a JSON object")
    unknown = sorted(set(data) - _KEYS)
    if unknown:
        raise BrandProfileError(f"unknown keys: {unknown}")
    profile_id = str(data.get("profile_id") or "").strip()
    if not profile_id or len(profile_id) > 64 or profile_id in BUILTIN_BRANDS:
        raise BrandProfileError(f"profile_id required (<= 64 chars, not one of {sorted(BUILTIN_BRANDS)})")
    version = data.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise BrandProfileError("version must be a positive integer")
    family = data.get("font_family", base.font_family)
    if not isinstance(family, str) or not family.strip() or len(family) > 64 or any(c in family for c in ",{}\\"):
        raise BrandProfileError("font_family must be a plain family name")
    font_path = data.get("optional_font_path")
    if font_path is not None:
        if not isinstance(font_path, str) or not Path(font_path).is_file():
            raise BrandProfileError("optional_font_path must be an existing font file")
        if Path(font_path).suffix.lower() not in (".ttf", ".otf", ".ttc"):
            raise BrandProfileError("optional_font_path must be .ttf/.otf/.ttc")
    if "primary_text_color" in data and "speaker_primary_color" in data \
            and str(data["primary_text_color"]).upper() != str(data["speaker_primary_color"]).upper():
        raise BrandProfileError("primary_text_color and speaker_primary_color disagree")
    lanes = {lane: dict(text=c.text, active=c.active, emphasis=c.emphasis, style_primary=c.style_primary)
             for lane, c in base.lanes.items()}
    explicit: set[tuple[str, str]] = set()
    for key, (lane, role) in _FLAT_COLOURS.items():
        value = _colour(data, key)
        if value is not None:
            lanes[lane][role] = ass_colour(value)
            explicit.add((lane, role))
            if role == "text":
                lanes[lane]["style_primary"] = ass_style_colour(value)
    # A third speaker follows the main lane (baseline parity) for every role the
    # profile does not set explicitly for "tertiary".
    if base.lanes["tertiary"] == base.lanes["main"]:
        for role in ("text", "active", "emphasis"):
            if ("tertiary", role) not in explicit:
                lanes["tertiary"][role] = lanes["main"][role]
        if ("tertiary", "text") not in explicit:
            lanes["tertiary"]["style_primary"] = lanes["main"]["style_primary"]
    # A third VOICE gets its own colours only when the profile sets them: the
    # tertiary lane's main-palette parity must never make voice C look like A.
    speakers = dict(base.speakers)
    if any(lane == "tertiary" for lane, _role in explicit):
        tertiary = lanes["tertiary"]
        speakers["C"] = LaneColours(tertiary["style_primary"], tertiary["text"], tertiary["active"],
                                    tertiary["emphasis"])
    neutral_explicit = {role: _colour(data, key) for key, role in _NEUTRAL_COLOURS.items() if key in data}
    if neutral_explicit:
        current = speakers.get("neutral", _NEUTRAL_VOICE)
        values = {"text": current.text, "active": current.active, "emphasis": current.emphasis,
                  "style_primary": current.style_primary}
        for role, value in neutral_explicit.items():
            values[role] = ass_colour(value)
            if role == "text":
                values["style_primary"] = ass_style_colour(value)
        speakers["neutral"] = LaneColours(values["style_primary"], values["text"], values["active"],
                                          values["emphasis"])
    outline = _colour(data, "outline_color")
    anchor = str(data.get("caption_anchor_policy", base.caption_anchor_policy))
    if anchor not in ANCHOR_POLICIES:
        raise BrandProfileError(f"caption_anchor_policy must be one of {ANCHOR_POLICIES}")
    default_style = data.get("default_style", base.default_style.value)
    if default_style not in (CaptionStyle.DEFAULT.value, CaptionStyle.EMPHASIS.value):
        raise BrandProfileError("default_style must be default|emphasis (impact is never a default)")
    primitives = data.get("allowed_primitives")
    allowed = base.allowed_primitives
    if primitives is not None:
        if not isinstance(primitives, list) or not all(isinstance(p, str) for p in primitives):
            raise BrandProfileError("allowed_primitives must be a list of primitive ids")
        try:
            allowed = frozenset(PrimitiveId(p) for p in primitives)
        except ValueError as error:
            raise BrandProfileError(f"unsupported primitive id: {error}") from error
        if not REQUIRED_PRIMITIVES <= allowed:
            raise BrandProfileError(f"allowed_primitives must include {sorted(p.value for p in REQUIRED_PRIMITIVES)}")
    backplate = _colour(data, "backplate_color")
    opacity = _number(data, "backplate_opacity", 1.0 - base.backplate_alpha / 255.0, 0.1, 0.9)
    budget = base.budget
    if "primitive_budget" in data:
        budget = PrimitiveBudget.parse(data["primitive_budget"])
    legibility = data.get("legibility", base.legibility)
    if legibility not in LEGIBILITY_POLICIES:
        raise BrandProfileError(f"legibility must be one of {LEGIBILITY_POLICIES}")
    return BrandProfile(
        profile_id=profile_id, version=int(version), font_family=family.strip(), optional_font_path=font_path,
        lanes={lane: LaneColours(v["style_primary"], v["text"], v["active"], v["emphasis"]) for lane, v in lanes.items()},
        outline_color=ass_style_colour(outline) if outline else base.outline_color,
        outline_scale=_number(data, "outline_scale", base.outline_scale, 0.0, 3.0),
        shadow_scale=_number(data, "shadow_scale", base.shadow_scale, 0.0, 3.0),
        caption_anchor_policy=anchor,
        safe_width_ratio=_number(data, "safe_width_ratio", base.safe_width_ratio, 0.5, 1.0),
        default_style=CaptionStyle(default_style), allowed_primitives=allowed,
        backplate_color=ass_colour(backplate) if backplate else base.backplate_color,
        backplate_alpha=int(round((1.0 - opacity) * 255)), budget=budget, legibility=str(legibility),
        speakers=speakers,
    )


def load_brand(value: str | None) -> tuple[BrandProfile, tuple[str, ...]]:
    """``mimir_default`` or a JSON file path. Never raises; invalid -> mimir_default."""
    key = str(value or "mimir_default").strip()
    if key in ("", "mimir_default", "default"):
        return MIMIR_DEFAULT, ()
    if key in BUILTIN_BRANDS:
        return BUILTIN_BRANDS[key], ()
    try:
        data = json.loads(Path(key).read_text(encoding="utf-8"))
        return parse_brand(data), ()
    except (OSError, ValueError, BrandProfileError) as error:
        return MIMIR_DEFAULT, (f"caption brand profile rejected ({type(error).__name__}: {error}); "
                               "using mimir_default",)


def with_font(brand: BrandProfile, family: str) -> BrandProfile:
    return replace(brand, font_family=family)


def primitive_ids() -> list[str]:
    return sorted(p.value for p in PRIMITIVE_LIBRARY)
