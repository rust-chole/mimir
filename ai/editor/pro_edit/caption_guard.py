"""Caption ownership guard: read-only references, integrity signatures, safe region.

Ownership: the caption system (speaker_caption_support + captions) owns word
text, word timing, speaker assignment and the ASS file. Pro Edit only holds
``CaptionWordRef`` references (ids into the final speaker profile) and may
only ever *reference* word ids. This module lets the pipeline prove that:

* ``caption_signature``: semantic signature of (id, text, start, end, speaker,
  label, confidence) over the authoritative final profile;
* ``ass_signature``: signature of the burned subtitle script (styles + events);
* ``CaptionIntegrity``: captured before planning, re-checked after render;
* ``caption_safe_region``: where captions occupy the OUTPUT frame (normalized),
  parsed from the real ASS styles, so framing can keep faces out of it.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ai.editor.pro_edit.errors import ProEditError

ACCEPTED_TIMING_BASES = frozenset({"edited_clip", "exact_final_48k_audio", "exact_final_short_audio"})
LINE_HEIGHT_FACTOR = 1.22
BAND_PADDING = 0.012


class CaptionIntegrityError(ProEditError):
    """Caption truth changed across the Pro Edit stage (must never happen)."""


@dataclass(frozen=True)
class CaptionWordRef:
    """Read-only reference to one authoritative caption word."""

    id: int
    text: str
    start: float
    end: float
    speaker_id: str | None
    speaker_label: str
    confidence: float
    speaker_color: str = ""      # the voice's caption colour key (profile truth, unchanged)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_profile(path: str | Path | None) -> Mapping[str, Any] | None:
    if path is None or not Path(path).is_file():
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def caption_words(profile: Mapping[str, Any] | None) -> tuple[CaptionWordRef, ...]:
    """Word id = index in the authoritative profile list (stable identity)."""
    if not profile or str(profile.get("status", "")) != "ok":
        return ()
    if str(profile.get("timing_basis", "")).strip() not in ACCEPTED_TIMING_BASES:
        return ()
    words: list[CaptionWordRef] = []
    for index, raw in enumerate(profile.get("words", []) or []):
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("word", "")).strip()
        start, end = _finite(raw.get("edited_start")), _finite(raw.get("edited_end"))
        if not text or start is None or end is None:
            continue
        words.append(CaptionWordRef(
            id=index,
            text=text,
            start=start,
            end=max(start, end),
            speaker_id=str(raw.get("speaker_raw") or "").strip() or None,
            speaker_label=str(raw.get("speaker_label") or ""),
            confidence=_finite(raw.get("speaker_confidence")) or 0.0,
            speaker_color=str(raw.get("speaker_color") or ""),
        ))
    return tuple(words)


def trusted_display_names(profile: Mapping[str, Any] | None) -> dict[str, str]:
    """raw speaker id -> the name the CURRENT caption renderer may print.

    Caption truth owns which identities are displayable. Captions V24 prints
    only human-confirmed names (``captions._trusted_human_display_map``); an
    older caption engine without that policy falls back to ``display_labels``.
    Pro Edit uses this for everything identity-related (planner context,
    emphasis reasons, font coverage), so it can never promote an automatic
    label to a confirmed name.
    """
    if not isinstance(profile, Mapping):
        return {}
    from ai.editor import captions as caption_truth

    trusted = getattr(caption_truth, "_trusted_human_display_map", None)
    labels = trusted(dict(profile)) if callable(trusted) else profile.get("display_labels")
    if not isinstance(labels, Mapping):
        return {}
    return {str(raw).strip(): str(name).strip() for raw, name in labels.items()
            if str(raw).strip() and str(name).strip()}


def caption_signature(words: Iterable[CaptionWordRef]) -> str:
    rows = [[w.id, w.text, round(w.start, 4), round(w.end, 4), w.speaker_id or "", w.speaker_label,
             round(w.confidence, 4), w.speaker_color] for w in sorted(words, key=lambda w: w.id)]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def ass_signature(path: str | Path | None) -> str:
    if path is None or not Path(path).is_file():
        return "absent"
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
    keep = [line.rstrip() for line in text.split("\n")
            if line.startswith(("Dialogue:", "Style:", "PlayResX:", "PlayResY:", "WrapStyle:"))]
    return hashlib.sha256("\n".join(keep).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CaptionIntegrity:
    profile_signature: str
    ass_signature: str
    word_count: int

    @classmethod
    def capture(cls, profile_path: str | Path | None, ass_path: str | Path | None) -> "CaptionIntegrity":
        words = caption_words(load_profile(profile_path))
        return cls(caption_signature(words), ass_signature(ass_path), len(words))

    def verify(self, profile_path: str | Path | None, ass_path: str | Path | None, *, stage: str) -> None:
        after = CaptionIntegrity.capture(profile_path, ass_path)
        if after != self:
            raise CaptionIntegrityError(
                f"caption truth changed during {stage}: profile {self.profile_signature[:12]}->"
                f"{after.profile_signature[:12]}, ass {self.ass_signature[:12]}->{after.ass_signature[:12]}")

    def to_dict(self) -> dict[str, Any]:
        return {"profile_signature": self.profile_signature, "ass_signature": self.ass_signature,
                "word_count": self.word_count}


# ============================================================
# CAPTION SAFE REGION (output-frame, normalized)
# ============================================================

@dataclass(frozen=True)
class CaptionBand:
    style: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class CaptionSafeRegion:
    bands: tuple[CaptionBand, ...]
    intervals: tuple[tuple[float, float], ...]
    source: str
    events: tuple[tuple[float, float, int], ...] = ()   # (start, end, index into bands)

    @property
    def empty(self) -> bool:
        return not self.bands

    def active_bands(self, start: float, end: float) -> tuple[CaptionBand, ...]:
        """Bands of the caption events shown at any time in [start, end] (PACED_CLIP s)."""
        if self.events:
            used = sorted({index for a, b, index in self.events if a < end and start < b})
            return tuple(self.bands[index] for index in used)
        if any(a < end and start < b for a, b in self.intervals):
            return self.bands
        return ()

    def top(self, start: float, end: float) -> float | None:
        bands = self.active_bands(start, end)
        return min((b.y0 for b in bands), default=None)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source,
                "bands": [{"style": b.style, "box": [round(b.x0, 4), round(b.y0, 4), round(b.x1, 4), round(b.y1, 4)]}
                          for b in self.bands],
                "active_intervals": len(self.intervals), "events": len(self.events)}


EMPTY_CAPTION_REGION = CaptionSafeRegion((), (), "none")
_TIME = re.compile(r"(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)")
_NUM = r"(-?\d+(?:\.\d+)?)"
_SCALE_TAG = re.compile(r"\\fscy" + _NUM)
_FS_TAG = re.compile(r"\\fs" + _NUM)
_BORD_TAG = re.compile(r"\\bord" + _NUM)
_SHAD_TAG = re.compile(r"\\shad" + _NUM)
_AN_TAG = re.compile(r"\\an([1-9])")
_POS_TAG = re.compile(r"\\pos\(\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*\)")
_OVERRIDE = re.compile(r"\{[^}]*\}")
_DRAW_TAG = re.compile(r"\\p(\d+)")
_DRAW_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _drawing_band(text: str, tags: str, style: Mapping[str, str], play_x: float, play_y: float
                  ) -> "CaptionBand | None":
    """Bounds of an ASS vector drawing event (e.g. a caption backplate)."""
    levels = [int(v) for v in _DRAW_TAG.findall(tags)]
    level = next((v for v in levels if v > 0), 0)
    position = _POS_TAG.search(tags)
    if level <= 0 or position is None:
        return None
    body = _OVERRIDE.sub(" ", text)
    numbers = [float(v) for v in _DRAW_NUMBER.findall(body)]
    if len(numbers) < 4:
        return None
    scale = 1.0 / (2 ** (level - 1))
    xs = [v * scale for v in numbers[0::2]]
    ys = [v * scale for v in numbers[1::2]]
    width, height = max(xs), max(ys)
    alignment = int(_finite(style.get("Alignment")) or 2)
    aligned = _AN_TAG.search(tags)
    if aligned:
        alignment = int(aligned.group(1))
    col, row = (alignment - 1) % 3, (alignment - 1) // 3
    left = float(position.group(1)) - (0.0, width / 2.0, width)[col]
    top = float(position.group(2)) - (height, height / 2.0, 0.0)[row]
    borders = [float(v) for v in _BORD_TAG.findall(tags) if float(v) >= 0]
    border = max(borders) if borders else (_finite(style.get("Outline")) or 0.0)
    return CaptionBand(
        style=style.get("Name", ""),
        x0=max(0.0, (left + min(xs) - border) / play_x - BAND_PADDING),
        y0=max(0.0, (top + min(ys) - border) / play_y - BAND_PADDING),
        x1=min(1.0, (left + max(xs) + border) / play_x + BAND_PADDING),
        y1=min(1.0, (top + max(ys) + border) / play_y + BAND_PADDING),
    )


def _ass_time(value: str) -> float | None:
    match = _TIME.fullmatch(value.strip())
    if not match:
        return None
    return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))


def _fields(line: str, fmt: Sequence[str]) -> dict[str, str]:
    payload = line.split(":", 1)[1].strip()
    parts = payload.split(",", len(fmt) - 1)
    return {name.strip(): value.strip() for name, value in zip(fmt, parts)}


def _event_band(event: Mapping[str, str], style: Mapping[str, str], play_x: float, play_y: float) -> CaptionBand:
    """Output-frame box of one Dialogue: style + event margins + override tags
    (\\an, \\pos, \\fs, largest \\fscy, \\bord, hard line breaks)."""
    text = event.get("Text", "")
    tags = "".join(_OVERRIDE.findall(text))
    drawing = _drawing_band(text, tags, style, play_x, play_y)
    if drawing is not None:
        return drawing
    size = _finite(style.get("Fontsize")) or 20.0
    sizes = [float(v) for v in _FS_TAG.findall(tags) if float(v) > 0]
    if sizes:
        size = max(sizes)
    scale = max([1.0] + [float(v) / 100.0 for v in _SCALE_TAG.findall(tags)])
    outline = _finite(style.get("Outline")) or 0.0
    borders = [float(v) for v in _BORD_TAG.findall(tags) if float(v) >= 0]
    if borders:
        outline = max(borders)
    shadows = [float(v) for v in _SHAD_TAG.findall(tags) if float(v) >= 0]
    outline += max(shadows) if shadows else (_finite(style.get("Shadow")) or 0.0)
    alignment = int(_finite(style.get("Alignment")) or 2)
    aligned = _AN_TAG.search(tags)
    if aligned:
        alignment = int(aligned.group(1))
    lines = 1 + len(re.findall(r"\\N", _OVERRIDE.sub("", text)))
    height = size * scale * LINE_HEIGHT_FACTOR * lines + 2 * outline
    position = _POS_TAG.search(tags)

    def margin(key: str) -> float:
        override = _finite(event.get(key))
        return override if override else (_finite(style.get(key)) or 0.0)

    if position:
        anchor_y = float(position.group(2))
        if alignment in (1, 2, 3):
            top, bottom = anchor_y - height + outline, anchor_y + outline
        elif alignment in (7, 8, 9):
            top, bottom = anchor_y - outline, anchor_y + height - outline
        else:
            top, bottom = anchor_y - height / 2.0, anchor_y + height / 2.0
        x0, x1 = 0.0, 1.0   # width unknown to the guard: full width, conservative
    else:
        margin_v = margin("MarginV")
        if alignment in (1, 2, 3):
            bottom = play_y - margin_v
            top = bottom - height
        elif alignment in (7, 8, 9):
            top = margin_v
            bottom = top + height
        else:
            top = (play_y - height) / 2.0
            bottom = top + height
        x0 = max(0.0, margin("MarginL") / play_x - BAND_PADDING)
        x1 = min(1.0, 1.0 - margin("MarginR") / play_x + BAND_PADDING)
    return CaptionBand(
        style=event.get("Style", ""),
        x0=x0,
        y0=max(0.0, top / play_y - BAND_PADDING),
        x1=x1,
        y1=min(1.0, bottom / play_y + BAND_PADDING),
    )


def caption_safe_region(ass_path: str | Path | None) -> CaptionSafeRegion:
    """Parse PlayRes, styles and every Dialogue from the ASS file actually burned.

    Each event gets its own vertical band (alignment row + MarginV or \\pos,
    font size / \\fs x the largest \\fscy it animates to, outline, line
    count), active only during that event. Horizontal extent is the style
    margins, or the full width for positioned events.
    """
    if ass_path is None or not Path(ass_path).is_file():
        return EMPTY_CAPTION_REGION
    text = Path(ass_path).read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
    play_x, play_y = 384.0, 288.0  # ASS defaults
    style_fmt: list[str] = []
    event_fmt: list[str] = []
    styles: dict[str, dict[str, str]] = {}
    events: list[dict[str, str]] = []
    for line in text.split("\n"):
        if line.startswith("PlayResX:"):
            play_x = _finite(line.split(":", 1)[1]) or play_x
        elif line.startswith("PlayResY:"):
            play_y = _finite(line.split(":", 1)[1]) or play_y
        elif line.startswith("Format:"):
            names = [n.strip() for n in line.split(":", 1)[1].split(",")]
            if "Fontsize" in names:
                style_fmt = names
            elif "Text" in names:
                event_fmt = names
        elif line.startswith("Style:") and style_fmt:
            row = _fields(line, style_fmt)
            styles[row.get("Name", "")] = row
        elif line.startswith("Dialogue:") and event_fmt:
            events.append(_fields(line, event_fmt))
    if not events or not styles:
        return EMPTY_CAPTION_REGION
    bands: list[CaptionBand] = []
    band_index: dict[CaptionBand, int] = {}
    timed: list[tuple[float, float, int]] = []
    intervals: list[tuple[float, float]] = []
    for event in events:
        style = styles.get(event.get("Style", ""))
        start, end = _ass_time(event.get("Start", "")), _ass_time(event.get("End", ""))
        if style is None or start is None or end is None or end <= start:
            continue
        band = _event_band(event, style, play_x, play_y)
        if band not in band_index:
            band_index[band] = len(bands)
            bands.append(band)
        timed.append((start, end, band_index[band]))
        intervals.append((start, end))
    if not bands:
        return EMPTY_CAPTION_REGION
    intervals.sort()
    merged: list[tuple[float, float]] = []
    for a, b in intervals:
        if merged and a <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return CaptionSafeRegion(tuple(bands), tuple(merged), "ass_events", tuple(sorted(timed)))
