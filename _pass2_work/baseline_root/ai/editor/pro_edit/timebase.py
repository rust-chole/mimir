"""Explicit timeline domains, rational frame clocks and domain conversion.

MIMIR timeline naming (see ai/editor/timeline.py):

* ``absolute_*``  -> T_VOD        seconds in the original source VOD
* ``source_*``    -> T_RAW_CLIP   seconds from the selected clip's VOD start
* ``edited_*``    -> T_PACED_CLIP seconds in the rendered pacing output
* intro renderer  -> T_INTRO / T_FINAL (cold-open + restarted main)

Pro Edit plans live in T_PACED_CLIP only. Conversions go through this module;
nothing converts silently. Raw->paced reuses MIMIR's own cut mapping
(``timeline.map_source_to_edited_time`` / ``teaser_analyzer.edited_to_source_time``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Iterable, Sequence

from ai.editor import timeline as mimir_timeline
from ai.editor.pro_edit.errors import EditPlanTimelineError


class TimelineDomain(str, Enum):
    VOD = "vod"
    RAW_CLIP = "raw_clip"
    PACED_CLIP = "paced_clip"
    INTRO = "intro"
    FINAL = "final"


@dataclass(frozen=True)
class Timestamp:
    seconds: float
    domain: TimelineDomain

    def __post_init__(self) -> None:
        if not isinstance(self.domain, TimelineDomain):
            raise EditPlanTimelineError(f"Timestamp domain must be TimelineDomain, got {self.domain!r}")
        if not math.isfinite(float(self.seconds)):
            raise EditPlanTimelineError("Timestamp seconds must be finite")

    def require(self, domain: TimelineDomain) -> float:
        if self.domain is not domain:
            raise EditPlanTimelineError(
                f"timestamp is in {self.domain.value}, expected {domain.value}; no silent conversion"
            )
        return float(self.seconds)


@dataclass(frozen=True)
class TimeRange:
    start: float
    end: float
    domain: TimelineDomain
    label: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.domain, TimelineDomain):
            raise EditPlanTimelineError(f"TimeRange domain must be TimelineDomain, got {self.domain!r}")
        if not (math.isfinite(float(self.start)) and math.isfinite(float(self.end))):
            raise EditPlanTimelineError("TimeRange bounds must be finite")
        if float(self.end) < float(self.start):
            raise EditPlanTimelineError(f"TimeRange end < start ({self.start} > {self.end})")

    @property
    def duration(self) -> float:
        return float(self.end) - float(self.start)

    def overlaps(self, other: "TimeRange") -> bool:
        if other.domain is not self.domain:
            raise EditPlanTimelineError("cannot compare ranges from different timeline domains")
        return float(self.start) < float(other.end) and float(other.start) < float(self.end)

    def overlap_seconds(self, start: float, end: float) -> float:
        return max(0.0, min(float(self.end), float(end)) - max(float(self.start), float(start)))

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
        }
        if self.label:
            data["label"] = self.label
        return data


# ============================================================
# RATIONAL FRAME CLOCK
# ============================================================

@dataclass(frozen=True)
class FrameRate:
    """Exact rational frame rate (e.g. 30000/1001). Never assumes integer fps."""

    num: int
    den: int

    def __post_init__(self) -> None:
        if int(self.num) <= 0 or int(self.den) <= 0:
            raise EditPlanTimelineError(f"invalid frame rate {self.num}/{self.den}")

    @classmethod
    def parse(cls, value: str | float | int | Fraction) -> "FrameRate":
        if isinstance(value, Fraction):
            fraction = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise EditPlanTimelineError(f"invalid frame rate {value!r}")
            fraction = Fraction(float(value)).limit_denominator(1001 * 1000)
        else:
            text = str(value).strip()
            try:
                if "/" in text:
                    left, right = text.split("/", 1)
                    fraction = Fraction(int(left), int(right))
                else:
                    fraction = Fraction(text).limit_denominator(1001 * 1000)
            except (ValueError, ZeroDivisionError) as error:
                raise EditPlanTimelineError(f"invalid frame rate {value!r}") from error
        if fraction <= 0:
            raise EditPlanTimelineError(f"invalid frame rate {value!r}")
        return cls(fraction.numerator, fraction.denominator)

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.num, self.den)

    @property
    def fps(self) -> float:
        return self.num / self.den

    @property
    def frame_duration(self) -> float:
        return self.den / self.num

    def frame_index(self, seconds: float) -> int:
        """Nearest frame index for a PACED_CLIP time (round-half-up, exact rational)."""
        if not math.isfinite(float(seconds)):
            raise EditPlanTimelineError("non-finite time cannot map to a frame")
        exact = Fraction(float(seconds)) * self.fraction
        return int(math.floor(exact + Fraction(1, 2)))

    def frame_time(self, index: int) -> float:
        return float(Fraction(int(index)) / self.fraction)

    def __str__(self) -> str:
        return f"{self.num}/{self.den}"


# ============================================================
# CLIP TIMELINE MAP (VOD <-> RAW_CLIP <-> PACED_CLIP)
# ============================================================

def _normalize_cuts(cut_ranges: Iterable[dict[str, object]], raw_duration: float) -> list[dict[str, float]]:
    cuts: list[dict[str, float]] = []
    for item in cut_ranges:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start", 0.0))  # type: ignore[arg-type]
            end = float(item.get("end", start))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end)):
            continue
        start = max(0.0, min(raw_duration, start))
        end = max(0.0, min(raw_duration, end))
        if end - start <= 1e-6:
            continue
        cuts.append({"start": start, "end": end})
    cuts.sort(key=lambda row: (row["start"], row["end"]))
    merged: list[dict[str, float]] = []
    for cut in cuts:
        if merged and cut["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], cut["end"])
        else:
            merged.append(dict(cut))
    return merged


@dataclass(frozen=True)
class ClipTimelineMap:
    """Maps selected-clip evidence between VOD, RAW_CLIP and PACED_CLIP.

    ``cut_ranges`` are RAW_CLIP ranges removed by the pacing render. They must
    be the FINAL list persisted by pacing_cutter (it can rewrite the timeline).
    """

    clip_vod_start: float
    raw_duration: float
    cut_ranges: tuple[tuple[float, float], ...]

    @classmethod
    def from_timeline_clip(cls, clip_timeline: dict[str, object]) -> "ClipTimelineMap":
        source = clip_timeline.get("source", {})
        if not isinstance(source, dict):
            raise EditPlanTimelineError("timeline clip has no source block")
        try:
            vod_start = float(source.get("absolute_start", 0.0))
            vod_end = float(source.get("absolute_end", vod_start))
            raw_duration = float(source.get("duration", vod_end - vod_start))
        except (TypeError, ValueError) as error:
            raise EditPlanTimelineError("timeline clip source bounds are not numeric") from error
        if not (math.isfinite(vod_start) and math.isfinite(raw_duration)) or raw_duration <= 0:
            raise EditPlanTimelineError("timeline clip source duration is invalid")
        raw_cuts = clip_timeline.get("cut_ranges", [])
        cuts = _normalize_cuts(raw_cuts if isinstance(raw_cuts, list) else [], raw_duration)
        return cls(
            clip_vod_start=vod_start,
            raw_duration=raw_duration,
            cut_ranges=tuple((row["start"], row["end"]) for row in cuts),
        )

    @property
    def _cut_dicts(self) -> list[dict[str, float]]:
        return [{"start": start, "end": end} for start, end in self.cut_ranges]

    @property
    def paced_duration(self) -> float:
        removed = sum(end - start for start, end in self.cut_ranges)
        return max(0.0, self.raw_duration - removed)

    # --- scalar conversions -------------------------------------------------

    def vod_to_raw(self, ts: Timestamp) -> Timestamp:
        return Timestamp(ts.require(TimelineDomain.VOD) - self.clip_vod_start, TimelineDomain.RAW_CLIP)

    def raw_to_vod(self, ts: Timestamp) -> Timestamp:
        return Timestamp(ts.require(TimelineDomain.RAW_CLIP) + self.clip_vod_start, TimelineDomain.VOD)

    def is_removed(self, ts: Timestamp) -> bool:
        raw = ts.require(TimelineDomain.RAW_CLIP)
        return any(start < raw < end for start, end in self.cut_ranges)

    def raw_to_paced(self, ts: Timestamp) -> Timestamp:
        raw = ts.require(TimelineDomain.RAW_CLIP)
        raw = max(0.0, min(self.raw_duration, raw))
        # Reuse MIMIR's authoritative mapping (removed time snaps to splice point).
        paced = mimir_timeline.map_source_to_edited_time(raw, self._cut_dicts)
        return Timestamp(max(0.0, min(self.paced_duration, float(paced))), TimelineDomain.PACED_CLIP)

    def paced_to_raw(self, ts: Timestamp) -> Timestamp:
        paced = max(0.0, min(self.paced_duration, ts.require(TimelineDomain.PACED_CLIP)))
        source_cursor = 0.0
        edited_cursor = 0.0
        for start, end in self.cut_ranges:
            kept = max(0.0, start - source_cursor)
            if paced <= edited_cursor + kept:
                return Timestamp(source_cursor + (paced - edited_cursor), TimelineDomain.RAW_CLIP)
            edited_cursor += kept
            source_cursor = max(source_cursor, end)
        return Timestamp(source_cursor + max(0.0, paced - edited_cursor), TimelineDomain.RAW_CLIP)

    def vod_to_paced(self, ts: Timestamp) -> Timestamp:
        return self.raw_to_paced(self.vod_to_raw(ts))

    def convert(self, ts: Timestamp, target: TimelineDomain) -> Timestamp:
        if ts.domain is target:
            return ts
        chain = {
            (TimelineDomain.VOD, TimelineDomain.RAW_CLIP): self.vod_to_raw,
            (TimelineDomain.RAW_CLIP, TimelineDomain.VOD): self.raw_to_vod,
            (TimelineDomain.RAW_CLIP, TimelineDomain.PACED_CLIP): self.raw_to_paced,
            (TimelineDomain.PACED_CLIP, TimelineDomain.RAW_CLIP): self.paced_to_raw,
            (TimelineDomain.VOD, TimelineDomain.PACED_CLIP): self.vod_to_paced,
            (TimelineDomain.PACED_CLIP, TimelineDomain.VOD): lambda value: self.raw_to_vod(self.paced_to_raw(value)),
        }
        converter = chain.get((ts.domain, target))
        if converter is None:
            raise EditPlanTimelineError(
                f"no clip-map conversion {ts.domain.value} -> {target.value}; use IntroTimelineMap"
            )
        return converter(ts)

    # --- range conversions --------------------------------------------------

    def range_to_paced(self, value: TimeRange) -> TimeRange | None:
        """Map a VOD/RAW range into PACED. Returns None when fully removed."""
        if value.domain is TimelineDomain.PACED_CLIP:
            return value
        if value.domain is TimelineDomain.VOD:
            raw = TimeRange(value.start - self.clip_vod_start, value.end - self.clip_vod_start,
                            TimelineDomain.RAW_CLIP, value.label)
        elif value.domain is TimelineDomain.RAW_CLIP:
            raw = value
        else:
            raise EditPlanTimelineError(f"cannot map {value.domain.value} range into paced clip")
        start = max(0.0, min(self.raw_duration, raw.start))
        end = max(0.0, min(self.raw_duration, raw.end))
        if end <= start:
            return None
        kept = end - start - sum(
            max(0.0, min(end, cut_end) - max(start, cut_start)) for cut_start, cut_end in self.cut_ranges
        )
        if kept <= 1e-6:
            return None
        paced_start = self.raw_to_paced(Timestamp(start, TimelineDomain.RAW_CLIP)).seconds
        paced_end = self.raw_to_paced(Timestamp(end, TimelineDomain.RAW_CLIP)).seconds
        if paced_end <= paced_start:
            return None
        return TimeRange(paced_start, paced_end, TimelineDomain.PACED_CLIP, raw.label)


@dataclass(frozen=True)
class IntroTimelineMap:
    """PACED_CLIP -> FINAL mapping created by the mandatory intro renderer.

    final = teaser_duration - transition + (paced - main_restart)
    Paced times before ``main_restart`` are not visible in the final Short.
    """

    teaser_duration: float
    transition_duration: float
    main_restart: float

    def paced_to_final(self, ts: Timestamp) -> Timestamp | None:
        paced = ts.require(TimelineDomain.PACED_CLIP)
        if paced < self.main_restart:
            return None
        return Timestamp(
            self.teaser_duration - self.transition_duration + (paced - self.main_restart),
            TimelineDomain.FINAL,
        )


def clip_ranges(ranges: Sequence[TimeRange], start: float, end: float) -> list[TimeRange]:
    """Intersect PACED ranges with [start, end] and drop empties."""
    result: list[TimeRange] = []
    for item in ranges:
        lo = max(float(item.start), float(start))
        hi = min(float(item.end), float(end))
        if hi > lo:
            result.append(TimeRange(lo, hi, item.domain, item.label))
    return result
