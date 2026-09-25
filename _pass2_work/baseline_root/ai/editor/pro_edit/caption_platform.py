"""Platform safe zones for caption placement (versioned, normalized, data-driven).

A ``PlatformSafeZoneProfile`` describes screen regions a publishing platform's
UI may cover (normalized [0, 1] output-frame boxes), how strongly captions
must avoid them (``hard`` = never, ``soft`` = weighted cost), preferred caption
regions and a minimum edge padding.

No platform geometry is invented here. MIMIR ships only:

* ``generic`` (default): edge padding only; no UI assumptions. Placement is
  identical to V3.1 unless faces / story regions / activity say otherwise.
* ``generic_conservative``: MIMIR heuristics (NOT platform measurements):
  soft bottom band and soft right column on vertical outputs.
* ``custom``: a user JSON file (``MIMIR_CAPTION_PLATFORM_PROFILE_FILE``) with
  measured geometry and its provenance.

``tiktok`` / ``reels`` / ``shorts`` are recognised names, but no verified
geometry ships with MIMIR; they resolve to ``generic_conservative`` and say so.

Schema v2 (V5) makes the model dynamic without inventing data: a profile FILE
may hold several variants, each scoped by platform, output aspect, UI variant
(e.g. organic feed vs ad placement), device class, optional known overlay
regions and description/caption footprints (none / short / long). The variant
is chosen per OUTPUT (aspect) and configuration (MIMIR_CAPTION_PLATFORM_VARIANT /
_DEVICE / MIMIR_CAPTION_DESCRIPTION); the most specific match wins, ties keep
file order. v1 single-profile files stay valid. MIMIR still ships no measured
platform geometry: official platform pages were not reachable from the build
environment, so exact TikTok / Reels / Shorts coordinates remain BLOCKED_DATA.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

PLATFORM_PROFILE_SCHEMA_VERSION = 2
DESCRIPTION_FOOTPRINTS = ("none", "short", "long")
_SET_KEYS = frozenset({"schema_version", "profile_set_id", "source", "variants"})
_VARIANT_KEYS = frozenset({"platform", "output_aspect", "ui_variant", "device_class", "profile",
                           "overlay_regions", "description_footprints"})
NAMED_PLATFORMS = ("tiktok", "reels", "shorts")
_ASPECT_TOLERANCE = 0.06
_PROFILE_KEYS = frozenset({
    "profile_id", "version", "output_aspect", "source", "minimum_edge_padding", "reserved_regions",
    "preferred_caption_regions", "right_ui_column", "bottom_ui_region", "top_ui_region", "shorthand_mode",
    "shorthand_weight",
})


class PlatformProfileError(ValueError):
    """Invalid safe-zone profile data."""


@dataclass(frozen=True)
class NormBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def parse(cls, value: Any, what: str) -> "NormBox":
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise PlatformProfileError(f"{what}: box must be [x0, y0, x1, y1]")
        numbers = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                raise PlatformProfileError(f"{what}: non-finite coordinate {item!r}")
            numbers.append(float(item))
        box = cls(*numbers)
        if not (0.0 <= box.x0 < box.x1 <= 1.0 and 0.0 <= box.y0 < box.y1 <= 1.0):
            raise PlatformProfileError(f"{what}: box {numbers} outside [0,1] or empty")
        return box

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def intersection(self, other: "NormBox") -> float:
        w = min(self.x1, other.x1) - max(self.x0, other.x0)
        h = min(self.y1, other.y1) - max(self.y0, other.y0)
        return w * h if w > 0 and h > 0 else 0.0

    def contains(self, other: "NormBox") -> bool:
        return (self.x0 <= other.x0 + 1e-9 and self.y0 <= other.y0 + 1e-9 and self.x1 >= other.x1 - 1e-9
                and self.y1 >= other.y1 - 1e-9)

    def to_list(self) -> list[float]:
        return [round(self.x0, 4), round(self.y0, 4), round(self.x1, 4), round(self.y1, 4)]


@dataclass(frozen=True)
class ReservedRegion:
    region_id: str
    box: NormBox
    mode: str = "soft"          # "hard" | "soft"
    weight: float = 3.0

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.region_id, "box": self.box.to_list(), "mode": self.mode, "weight": self.weight}


@dataclass(frozen=True)
class PlatformSafeZoneProfile:
    profile_id: str
    version: int
    output_aspect: str = "any"                  # "any" or "W:H"
    source: str = ""
    verified: bool = False                      # True only for measured / documented geometry
    minimum_edge_padding: float = 0.02
    reserved_regions: tuple[ReservedRegion, ...] = ()
    preferred_caption_regions: tuple[NormBox, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)
    variants: tuple["PlatformVariant", ...] = ()
    selectors: tuple[tuple[str, str], ...] = ()       # (platform, ui_variant, device_class, description)

    def applies_to(self, width: int, height: int) -> bool:
        if self.output_aspect == "any":
            return True
        w, h = (float(v) for v in self.output_aspect.split(":"))
        return abs((width / height) - (w / h)) <= _ASPECT_TOLERANCE * (w / h)

    def for_output(self, width: int, height: int) -> "PlatformSafeZoneProfile":
        """The profile to use for this output; UI regions only apply to their aspect.
        A v2 profile set resolves its most specific variant for this output first."""
        if self.variants:
            return select_variant(self, width, height)
        if self.applies_to(width, height):
            return self
        return PlatformSafeZoneProfile(
            self.profile_id, self.version, self.output_aspect, self.source, self.verified,
            self.minimum_edge_padding, (), (),
            self.notes + (f"reserved regions not applied: output {width}x{height} != {self.output_aspect}",))

    def to_dict(self) -> dict[str, Any]:
        return {"profile_id": self.profile_id, "version": self.version, "output_aspect": self.output_aspect,
                "source": self.source, "verified": self.verified,
                "minimum_edge_padding": self.minimum_edge_padding,
                "reserved_regions": [r.to_dict() for r in self.reserved_regions],
                "preferred_caption_regions": [b.to_list() for b in self.preferred_caption_regions],
                "notes": list(self.notes), "schema_version": PLATFORM_PROFILE_SCHEMA_VERSION,
                "variants": len(self.variants), "selectors": dict(self.selectors)}


@dataclass(frozen=True)
class PlatformVariant:
    platform: str
    output_aspect: str
    ui_variant: str
    device_class: str
    profile: PlatformSafeZoneProfile
    overlay_regions: tuple[ReservedRegion, ...] = ()
    description_footprints: tuple[tuple[str, ReservedRegion], ...] = ()


def _aspect_matches(aspect: str, width: int, height: int) -> bool:
    if aspect == "any":
        return True
    w, h = (float(v) for v in aspect.split(":"))
    return abs((width / height) - (w / h)) <= _ASPECT_TOLERANCE * (w / h)


def select_variant(profile_set: "PlatformSafeZoneProfile", width: int, height: int) -> "PlatformSafeZoneProfile":
    """Most specific variant for (output aspect, platform, ui variant, device, description)."""
    wanted = dict(profile_set.selectors)
    best: tuple[int, int, PlatformVariant] | None = None
    for order, variant in enumerate(profile_set.variants):
        if not _aspect_matches(variant.output_aspect, width, height):
            continue
        score = 1 if variant.output_aspect != "any" else 0
        rejected = False
        for key, value in (("platform", variant.platform), ("ui_variant", variant.ui_variant),
                           ("device_class", variant.device_class)):
            if value == "any":
                continue
            if wanted.get(key) and wanted[key] != value:
                rejected = True
                break
            score += 2 if wanted.get(key) == value else 0
        if rejected:
            continue
        if best is None or score > best[0]:
            best = (score, order, variant)
    if best is None:
        return PlatformSafeZoneProfile(
            profile_set.profile_id, profile_set.version, "any", profile_set.source, False,
            profile_set.minimum_edge_padding, (), (),
            profile_set.notes + (f"no variant matches output {width}x{height} / {wanted}; edge padding only",))
    variant = best[2]
    regions = list(variant.profile.reserved_regions) + list(variant.overlay_regions)
    description = wanted.get("description", "")
    footprints = dict(variant.description_footprints)
    if description and description != "none" and description in footprints:
        regions.append(footprints[description])
    chosen = variant.profile
    return PlatformSafeZoneProfile(
        f"{profile_set.profile_id}/{chosen.profile_id}", chosen.version, "any", chosen.source or profile_set.source,
        False, chosen.minimum_edge_padding, tuple(regions), chosen.preferred_caption_regions,
        profile_set.notes + (f"variant platform={variant.platform} aspect={variant.output_aspect} "
                             f"ui={variant.ui_variant} device={variant.device_class} description={description or '-'}",),
        (), profile_set.selectors)


GENERIC = PlatformSafeZoneProfile(
    "generic", 1, "any", "MIMIR generic: edge padding only; no platform UI assumptions", False, 0.02)
GENERIC_CONSERVATIVE = PlatformSafeZoneProfile(
    "generic_conservative", 1, "9:16",
    "MIMIR heuristic for vertical outputs (NOT platform measurements): soft bottom band and right column",
    False, 0.04,
    (ReservedRegion("bottom_ui_region", NormBox(0.0, 0.88, 1.0, 1.0), "soft", 3.0),
     ReservedRegion("right_ui_column", NormBox(0.88, 0.40, 1.0, 0.88), "soft", 2.0)),
)
BUILTIN_PROFILES: Mapping[str, PlatformSafeZoneProfile] = {
    GENERIC.profile_id: GENERIC, GENERIC_CONSERVATIVE.profile_id: GENERIC_CONSERVATIVE,
}


def _parse_region(raw: Any, index: int, default_mode: str, default_weight: float) -> ReservedRegion:
    if not isinstance(raw, Mapping):
        raise PlatformProfileError(f"reserved_regions[{index}] must be an object")
    region_id = str(raw.get("id") or f"region_{index}")
    mode = str(raw.get("mode", default_mode))
    if mode not in ("hard", "soft"):
        raise PlatformProfileError(f"{region_id}: mode must be hard|soft")
    weight = raw.get("weight", default_weight)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) \
            or not 0.0 < float(weight) <= 10.0:
        raise PlatformProfileError(f"{region_id}: weight must be in (0, 10]")
    return ReservedRegion(region_id, NormBox.parse(raw.get("box"), region_id), mode, float(weight))


def parse_profile(data: Any, *, source_hint: str = "") -> PlatformSafeZoneProfile:
    """Validate a JSON profile (strict: unknown keys, NaN, bad boxes are errors)."""
    if not isinstance(data, Mapping):
        raise PlatformProfileError("profile must be a JSON object")
    unknown = sorted(set(data) - _PROFILE_KEYS)
    if unknown:
        raise PlatformProfileError(f"unknown keys: {unknown}")
    profile_id = str(data.get("profile_id") or "").strip()
    if not profile_id or len(profile_id) > 64:
        raise PlatformProfileError("profile_id required (<= 64 chars)")
    version = data.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise PlatformProfileError("version must be a positive integer")
    aspect = str(data.get("output_aspect", "any")).strip()
    if aspect != "any":
        try:
            w, h = (float(v) for v in aspect.split(":"))
            if w <= 0 or h <= 0 or not math.isfinite(w / h):
                raise ValueError
        except ValueError as error:
            raise PlatformProfileError(f"output_aspect {aspect!r} must be 'any' or 'W:H'") from error
    padding = data.get("minimum_edge_padding", 0.02)
    if isinstance(padding, bool) or not isinstance(padding, (int, float)) or not math.isfinite(padding) \
            or not 0.0 <= float(padding) <= 0.2:
        raise PlatformProfileError("minimum_edge_padding must be in [0, 0.2]")
    mode = str(data.get("shorthand_mode", "soft"))
    if mode not in ("hard", "soft"):
        raise PlatformProfileError("shorthand_mode must be hard|soft")
    shorthand_weight = data.get("shorthand_weight", 3.0)
    regions: list[ReservedRegion] = []
    raw_regions = data.get("reserved_regions", [])
    if not isinstance(raw_regions, list):
        raise PlatformProfileError("reserved_regions must be a list")
    for index, raw in enumerate(raw_regions):
        regions.append(_parse_region(raw, index, "soft", 3.0))
    for key in ("right_ui_column", "bottom_ui_region", "top_ui_region"):
        if data.get(key) is not None:
            regions.append(_parse_region({"id": key, "box": data[key], "mode": mode, "weight": shorthand_weight},
                                         len(regions), mode, 3.0))
    preferred = data.get("preferred_caption_regions", [])
    if not isinstance(preferred, list):
        raise PlatformProfileError("preferred_caption_regions must be a list")
    boxes = tuple(NormBox.parse(value, f"preferred_caption_regions[{i}]") for i, value in enumerate(preferred))
    source = str(data.get("source") or source_hint or "custom profile (no provenance given)")
    return PlatformSafeZoneProfile(profile_id, int(version), aspect, source, False, float(padding), tuple(regions),
                                   boxes)


def _selector(value: Any, what: str) -> str:
    text = str(value if value is not None else "any").strip().casefold() or "any"
    if len(text) > 48 or not all(c.isalnum() or c in "_-." for c in text):
        raise PlatformProfileError(f"{what} must be a short identifier")
    return text


def parse_profile_set(data: Any, *, source_hint: str = "", selectors: Mapping[str, str] | None = None
                      ) -> PlatformSafeZoneProfile:
    """Schema v2: {"schema_version": 2, "profile_set_id", "source", "variants": [...]} (strict)."""
    if not isinstance(data, Mapping):
        raise PlatformProfileError("profile set must be a JSON object")
    unknown = sorted(set(data) - _SET_KEYS)
    if unknown:
        raise PlatformProfileError(f"unknown keys: {unknown}")
    if data.get("schema_version") != 2:
        raise PlatformProfileError("profile set schema_version must be 2")
    set_id = str(data.get("profile_set_id") or "").strip()
    if not set_id or len(set_id) > 64:
        raise PlatformProfileError("profile_set_id required (<= 64 chars)")
    source = str(data.get("source") or source_hint or "custom profile set (no provenance given)")
    raw_variants = data.get("variants")
    if not isinstance(raw_variants, list) or not raw_variants or len(raw_variants) > 64:
        raise PlatformProfileError("variants must be a non-empty list (<= 64)")
    variants = []
    for index, raw in enumerate(raw_variants):
        if not isinstance(raw, Mapping):
            raise PlatformProfileError(f"variants[{index}] must be an object")
        extra = sorted(set(raw) - _VARIANT_KEYS)
        if extra:
            raise PlatformProfileError(f"variants[{index}]: unknown keys {extra}")
        profile = parse_profile(raw.get("profile"), source_hint=source)
        aspect = str(raw.get("output_aspect", profile.output_aspect)).strip()
        if aspect != "any":
            parse_profile({"profile_id": "aspect_check", "version": 1, "output_aspect": aspect})
        overlays_raw = raw.get("overlay_regions", [])
        if not isinstance(overlays_raw, list):
            raise PlatformProfileError(f"variants[{index}].overlay_regions must be a list")
        overlays = tuple(_parse_region(r, i, "soft", 3.0) for i, r in enumerate(overlays_raw))
        footprints_raw = raw.get("description_footprints", {})
        if not isinstance(footprints_raw, Mapping) or set(footprints_raw) - {"short", "long"}:
            raise PlatformProfileError(f"variants[{index}].description_footprints: keys short|long only")
        footprints = tuple((key, _parse_region({**dict(value), "id": f"description_{key}"}
                                               if isinstance(value, Mapping) else value, 0, "soft", 3.0))
                           for key, value in sorted(footprints_raw.items()))
        variants.append(PlatformVariant(_selector(raw.get("platform"), "platform"), aspect,
                                        _selector(raw.get("ui_variant"), "ui_variant"),
                                        _selector(raw.get("device_class"), "device_class"), profile, overlays,
                                        footprints))
    chosen = {k: v for k, v in (selectors or {}).items() if v}
    return PlatformSafeZoneProfile(set_id, 2, "any", source, False, 0.02, (), (),
                                   (f"profile set with {len(variants)} variant(s)",), tuple(variants),
                                   tuple(sorted(chosen.items())))


def load_platform_profile(name: str | None, file_path: str | None = None, *, variant: str = "", device: str = "",
                          description: str = "") -> tuple[PlatformSafeZoneProfile, tuple[str, ...]]:
    """Resolve the configured profile. Never raises; problems fall back to generic."""
    key = str(name or "generic").strip().casefold() or "generic"
    problems: list[str] = []
    if file_path or key == "custom":
        if not file_path:
            return GENERIC, ("MIMIR_CAPTION_PLATFORM_PROFILE=custom without MIMIR_CAPTION_PLATFORM_PROFILE_FILE; "
                             "using generic",)
        try:
            data = json.loads(Path(file_path).read_text(encoding="utf-8"))
            if isinstance(data, Mapping) and "variants" in data:
                selectors = {"platform": key if key in NAMED_PLATFORMS else "", "ui_variant": variant,
                             "device_class": device, "description": description}
                return parse_profile_set(data, source_hint=f"custom file {Path(file_path).name}",
                                         selectors=selectors), ()
            return parse_profile(data, source_hint=f"custom file {Path(file_path).name}"), ()
        except (OSError, ValueError, PlatformProfileError) as error:
            return GENERIC, (f"custom platform profile rejected ({type(error).__name__}: {error}); using generic",)
    if key in BUILTIN_PROFILES:
        return BUILTIN_PROFILES[key], ()
    if key in NAMED_PLATFORMS:
        problems.append(f"no verified '{key}' UI geometry ships with MIMIR; using generic_conservative "
                        "(provide measured geometry via MIMIR_CAPTION_PLATFORM_PROFILE_FILE)")
        return GENERIC_CONSERVATIVE, tuple(problems)
    return GENERIC, (f"unknown caption platform profile {key!r}; using generic",)
