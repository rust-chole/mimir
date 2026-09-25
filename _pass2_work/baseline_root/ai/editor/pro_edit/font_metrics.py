"""Rendered glyph widths for caption layout (no dependency, no bundled fonts).

Caption pages are laid out by the width libass will actually draw, not by
character count. libass sizes a face so that the OS/2 ``usWinAscent +
usWinDescent`` box equals the ASS font size (VSFilter compatibility), so::

    advance_px = advance_units * font_size / (usWinAscent + usWinDescent)

Resolution order (the same font libass is expected to pick):

1. the real font file: fontconfig ``fc-match`` (Linux/macOS/most FFmpeg
   builds) or ``%WINDIR%\\Fonts`` (Windows), parsed with a small read-only
   sfnt reader (``head``, ``hhea``, ``hmtx``, ``OS/2``, ``cmap`` 4/12);
2. a built-in Arial-Bold-compatible advance table (public Helvetica-Bold /
   Liberation-Sans-Bold metrics; Arial Bold is metric-compatible) with
   conservative widths for anything unknown.

Measurements are deliberately conservative: no kerning is subtracted and an
unknown glyph is assumed wide, so layout errs toward fewer words per line.
Classic ``kern`` pairs are parsed and reported (``kerned_width``) but never
narrow the layout: the FFmpeg/libass build MIMIR renders with was measured not
to apply kerning (identical raster widths with ``Kerning: yes/no``).

Shapers (``select_shaper``): ``naive`` (default, advances above) or an optional
HarfBuzz envelope (``uharfbuzz`` if installed): width = max(naive, shaped), so
complex shaping can only make the layout more conservative, never less.
"""
from __future__ import annotations

import functools
import os
import shutil
import struct
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

FONT_METRICS_VERSION = 3

# Arial-Bold-compatible advances per 1000 em for printable ASCII (U+0020..U+007E).
_ARIAL_BOLD_ASCII = (
    278, 333, 474, 556, 556, 889, 722, 238, 333, 333, 389, 584, 278, 333, 278, 278,  # ' '..'/'
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 333, 333, 584, 584, 584, 611,  # '0'..'?'
    975, 722, 722, 722, 722, 667, 611, 778, 722, 278, 556, 722, 611, 833, 722, 778,  # '@'..'O'
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 333, 278, 333, 584, 556,  # 'P'..'_'
    333, 556, 611, 556, 611, 556, 333, 611, 611, 278, 278, 556, 278, 889, 611, 611,  # '`'..'o'
    611, 611, 389, 556, 333, 611, 556, 778, 556, 556, 500, 389, 280, 389, 584,       # 'p'..'~'
)
_ARIAL_BOLD_EXTRA = {
    0x00A0: 278, 0x00AB: 556, 0x00BB: 556, 0x00BF: 611, 0x00A1: 333, 0x00DF: 611, 0x00E6: 889,
    0x00C6: 1000, 0x00F8: 611, 0x00D8: 778, 0x0131: 278, 0x0152: 1000, 0x0153: 944, 0x2018: 278,
    0x2019: 278, 0x201C: 500, 0x201D: 500, 0x201E: 500, 0x2013: 556, 0x2014: 1000, 0x2026: 1000,
    0x20AC: 556, 0x00B0: 400, 0x00A3: 556, 0x2022: 350, 0x00B7: 278,
}
# Arial: unitsPerEm 2048, usWinAscent 1854, usWinDescent 434 -> size basis 2288/2048 em.
_ARIAL_SIZE_BASIS_PER_1000 = 1000.0 * 2288.0 / 2048.0
_UNKNOWN_ADVANCE = 667          # conservative Latin-ish default (per 1000 em)
_WIDE_ADVANCE = 1000            # East Asian wide / fullwidth
_SYMBOL_ADVANCE = 1150          # other pictographs / symbols (usually from a fallback font)
EMOJI_SIZE_RATIO = 1.15         # wide emoji: measured 1.00 x ASS size (Noto Color Emoji, libass);
                                # +15% because the Windows emoji face (Segoe UI Emoji) is unmeasured

_WINDOWS_FILES = {
    ("arial", False): "arial.ttf", ("arial", True): "arialbd.ttf",
    ("impact", False): "impact.ttf", ("impact", True): "impact.ttf",
    ("segoe ui", False): "segoeui.ttf", ("segoe ui", True): "segoeuib.ttf",
}


class FontParseError(ValueError):
    """The font file is not a readable sfnt (TrueType / OpenType)."""


@dataclass(frozen=True)
class FontMetrics:
    family: str
    bold: bool
    source: str
    units_per_em: float
    size_basis_units: float
    ascent_units: float
    descent_units: float
    advances: Mapping[int, float] = field(repr=False, default_factory=dict)
    cmap: "_Cmap | None" = field(repr=False, default=None, compare=False)
    hmtx: tuple[int, ...] = field(repr=False, default=(), compare=False)
    resolved_family: str = ""       # family name inside the face actually used (name table)
    weight_class: int = 700          # OS/2 usWeightClass (400 regular, 700 bold)
    kern_pairs: Mapping[tuple[int, int], int] = field(repr=False, default_factory=dict, compare=False)

    @property
    def is_builtin(self) -> bool:
        return self.source.startswith("builtin:")

    def advance_units(self, char: str) -> float:
        code = ord(char)
        if self.cmap is not None and self.hmtx:
            glyph = self.cmap.glyph(code)
            if glyph:
                return float(self.hmtx[min(glyph, len(self.hmtx) - 1)])
            return self._fallback_units(char) * self.units_per_em / 1000.0
        if code in self.advances:
            return self.advances[code]
        return self._fallback_units(char) * self.units_per_em / 1000.0

    def _fallback_units(self, char: str) -> float:
        """Per-1000-em estimate for a glyph the primary face does not contain."""
        category = unicodedata.category(char)
        if category in ("Mn", "Me", "Cf"):
            return 0.0
        decomposed = unicodedata.normalize("NFD", char)
        if decomposed and decomposed[0] != char and 0x20 <= ord(decomposed[0]) <= 0x7E:
            return float(_ARIAL_BOLD_ASCII[ord(decomposed[0]) - 0x20])
        if category in ("So", "Sk") and unicodedata.east_asian_width(char) == "W":
            # Emoji-presentation pictographs come from a fallback emoji font whose advance
            # scales with the ASS font SIZE (libass raster: Noto Color Emoji = 1.00 x size),
            # not with this face's em: estimate EMOJI_SIZE_RATIO x size (never less).
            return EMOJI_SIZE_RATIO * 1000.0 * self.size_basis_units / self.units_per_em
        if unicodedata.east_asian_width(char) in ("W", "F"):
            return float(_WIDE_ADVANCE)
        if category == "So":
            return float(_SYMBOL_ADVANCE)
        return float(_UNKNOWN_ADVANCE)

    def text_width(self, text: str, font_px: float, *, scale_x: float = 1.0) -> float:
        """Horizontal advance of ``text`` in pixels at ASS font size ``font_px``."""
        units = sum(self.advance_units(char) for char in text)
        return units * float(font_px) / self.size_basis_units * float(scale_x)

    def kerning_units(self, text: str) -> float:
        """Sum of classic ``kern`` pair adjustments (usually negative)."""
        if not self.kern_pairs or self.cmap is None:
            return 0.0
        glyphs = [self.cmap.glyph(ord(char)) for char in text]
        return float(sum(self.kern_pairs.get((a, b), 0) for a, b in zip(glyphs, glyphs[1:])))

    def kerned_width(self, text: str, font_px: float, *, scale_x: float = 1.0) -> float:
        """Width with classic kerning (diagnostic; layout uses ``text_width``)."""
        units = sum(self.advance_units(char) for char in text) + self.kerning_units(text)
        return units * float(font_px) / self.size_basis_units * float(scale_x)

    def covers(self, text: str) -> bool:
        """True when every visible character has a glyph in this face."""
        if self.cmap is None:
            return True
        for char in text:
            if unicodedata.category(char) in ("Zs", "Cc", "Cf", "Mn", "Me"):
                continue
            if not self.cmap.glyph(ord(char)):
                return False
        return True

    def line_height(self, font_px: float) -> float:
        """libass line box height (ascent + descent) at ASS size ``font_px``."""
        return (self.ascent_units + self.descent_units) * float(font_px) / self.size_basis_units

    @property
    def substituted(self) -> bool:
        """True when the system resolved another family (e.g. Liberation Sans for
        Arial). Layout stays exact because the substitute's own metrics are used."""
        resolved = (self.resolved_family or self.family).casefold()
        return not self.is_builtin and resolved != self.family.casefold()

    def to_dict(self) -> dict[str, object]:
        return {"family": self.family, "resolved_family": self.resolved_family or self.family,
                "substituted": self.substituted, "bold": self.bold, "source": self.source,
                "units_per_em": self.units_per_em, "size_basis_units": self.size_basis_units,
                "version": FONT_METRICS_VERSION}


# ============================================================
# SFNT READER (read-only, bounds-checked)
# ============================================================

class _Cmap:
    """Codepoint -> glyph id from a format 4 or format 12 subtable."""

    def __init__(self, data: bytes, offset: int):
        self._data = data
        self._offset = offset
        self._format = struct.unpack_from(">H", data, offset)[0]
        if self._format not in (4, 12):
            raise FontParseError(f"unsupported cmap format {self._format}")
        self._cache: dict[int, int] = {}
        if self._format == 12:
            count = struct.unpack_from(">I", data, offset + 12)[0]
            self._groups = [struct.unpack_from(">III", data, offset + 16 + 12 * i) for i in range(count)]
        else:
            seg_x2 = struct.unpack_from(">H", data, offset + 6)[0]
            seg = seg_x2 // 2
            base = offset + 14
            self._ends = struct.unpack_from(f">{seg}H", data, base)
            self._starts = struct.unpack_from(f">{seg}H", data, base + seg_x2 + 2)
            self._deltas = struct.unpack_from(f">{seg}h", data, base + 2 * seg_x2 + 2)
            self._range_base = base + 3 * seg_x2 + 2
            self._ranges = struct.unpack_from(f">{seg}H", data, self._range_base)

    def glyph(self, code: int) -> int:
        if code in self._cache:
            return self._cache[code]
        glyph = self._lookup(code)
        self._cache[code] = glyph
        return glyph

    def _lookup(self, code: int) -> int:
        if self._format == 12:
            for start, end, first in self._groups:
                if start <= code <= end:
                    return first + (code - start)
            return 0
        if code > 0xFFFF:
            return 0
        for index, end in enumerate(self._ends):
            if code > end:
                continue
            start = self._starts[index]
            if code < start:
                return 0
            range_offset = self._ranges[index]
            if range_offset == 0:
                return (code + self._deltas[index]) & 0xFFFF
            address = self._range_base + 2 * index + range_offset + 2 * (code - start)
            if address + 2 > len(self._data):
                return 0
            glyph = struct.unpack_from(">H", self._data, address)[0]
            return (glyph + self._deltas[index]) & 0xFFFF if glyph else 0
        return 0


def _tables(data: bytes, face_index: int) -> dict[bytes, tuple[int, int]]:
    if len(data) < 12:
        raise FontParseError("file too small")
    base = 0
    if data[:4] == b"ttcf":
        count = struct.unpack_from(">I", data, 8)[0]
        if not 0 <= face_index < count:
            face_index = 0
        base = struct.unpack_from(">I", data, 12 + 4 * face_index)[0]
    version = data[base:base + 4]
    if version not in (b"\x00\x01\x00\x00", b"OTTO", b"true"):
        raise FontParseError("not a TrueType/OpenType font")
    count = struct.unpack_from(">H", data, base + 4)[0]
    tables: dict[bytes, tuple[int, int]] = {}
    for index in range(count):
        tag, _checksum, offset, length = struct.unpack_from(">4sIII", data, base + 12 + 16 * index)
        if offset + length > len(data):
            raise FontParseError(f"table {tag!r} out of bounds")
        tables[tag] = (offset, length)
    for required in (b"head", b"hhea", b"hmtx", b"cmap"):
        if required not in tables:
            raise FontParseError(f"missing {required.decode()} table")
    return tables


def _best_cmap(data: bytes, offset: int) -> _Cmap:
    count = struct.unpack_from(">H", data, offset + 2)[0]
    candidates: dict[tuple[int, int], int] = {}
    for index in range(count):
        platform, encoding, sub = struct.unpack_from(">HHI", data, offset + 4 + 8 * index)
        candidates[(platform, encoding)] = offset + sub
    for key in ((3, 10), (0, 4), (0, 6), (3, 1), (0, 3), (0, 1), (0, 0)):
        if key in candidates:
            try:
                return _Cmap(data, candidates[key])
            except (FontParseError, struct.error):
                continue
    raise FontParseError("no usable Unicode cmap subtable")


def _family_name(data: bytes, tables: Mapping[bytes, tuple[int, int]]) -> str:
    """Family (nameID 16, else 1) from the ``name`` table; "" when unreadable."""
    if b"name" not in tables:
        return ""
    offset, length = tables[b"name"]
    try:
        count, strings = struct.unpack_from(">HH", data, offset + 2)
        found: dict[int, str] = {}
        for index in range(count):
            platform, encoding, language, name_id, size, start = struct.unpack_from(
                ">HHHHHH", data, offset + 6 + 12 * index)
            if name_id not in (1, 16) or name_id in found:
                continue
            raw = data[offset + strings + start: offset + strings + start + size]
            if platform in (0, 3):
                found[name_id] = raw.decode("utf-16-be", errors="replace")
            elif platform == 1 and encoding == 0:
                found[name_id] = raw.decode("mac_roman", errors="replace")
        return (found.get(16) or found.get(1) or "").strip()
    except struct.error:
        return ""


def _kern_pairs(data: bytes, tables: Mapping[bytes, tuple[int, int]]) -> dict[tuple[int, int], int]:
    """Classic Microsoft ``kern`` table, format 0 horizontal subtables."""
    if b"kern" not in tables:
        return {}
    offset, length = tables[b"kern"]
    pairs: dict[tuple[int, int], int] = {}
    try:
        version, count = struct.unpack_from(">HH", data, offset)
        if version != 0:
            return {}
        cursor = offset + 4
        for _ in range(count):
            _sub_version, sub_length, coverage = struct.unpack_from(">HHH", data, cursor)
            fmt = coverage >> 8
            # horizontal kerning values only: not "minimum" values, not cross-stream
            usable = bool(coverage & 0x1) and not coverage & 0x2 and not coverage & 0x4
            if fmt == 0:
                n_pairs = struct.unpack_from(">H", data, cursor + 6)[0]
                if usable:
                    base = cursor + 14
                    for index in range(n_pairs):
                        left, right, value = struct.unpack_from(">HHh", data, base + 6 * index)
                        pairs[(left, right)] = value
                # the 16-bit length wraps on large tables; format 0 size is exact from nPairs
                sub_length = 14 + 6 * n_pairs
            cursor += max(sub_length, 6)
            if cursor >= offset + length:
                break
    except struct.error:
        return {}
    return pairs


def parse_font(path: str | Path, *, family: str = "", bold: bool = False, face_index: int = 0) -> FontMetrics:
    data = Path(path).read_bytes()
    try:
        tables = _tables(data, face_index)
        head = tables[b"head"][0]
        units_per_em = struct.unpack_from(">H", data, head + 18)[0]
        hhea = tables[b"hhea"][0]
        ascender, descender = struct.unpack_from(">hh", data, hhea + 4)
        metrics_count = struct.unpack_from(">H", data, hhea + 34)[0]
        hmtx_offset, hmtx_length = tables[b"hmtx"]
        metrics_count = min(metrics_count, hmtx_length // 4)
        advances = tuple(struct.unpack_from(">H", data, hmtx_offset + 4 * i)[0] for i in range(metrics_count))
        win_ascent, win_descent = 0, 0
        if b"OS/2" in tables and tables[b"OS/2"][1] >= 78:
            win_ascent, win_descent = struct.unpack_from(">HH", data, tables[b"OS/2"][0] + 74)
        cmap = _best_cmap(data, tables[b"cmap"][0])
        resolved = _family_name(data, tables)
        weight = 400
        if b"OS/2" in tables and tables[b"OS/2"][1] >= 6:
            weight = struct.unpack_from(">H", data, tables[b"OS/2"][0] + 4)[0]
        kern = _kern_pairs(data, tables)
    except struct.error as error:
        raise FontParseError(f"truncated font: {error}") from error
    if units_per_em <= 0 or not advances:
        raise FontParseError("invalid head/hmtx")
    size_basis = float(win_ascent + win_descent) or float(ascender - descender) or float(units_per_em)
    ascent = float(win_ascent or ascender)
    descent = float(win_descent or -descender)
    return FontMetrics(family=family or Path(path).stem, bold=bold, source=str(Path(path).resolve()),
                       units_per_em=float(units_per_em), size_basis_units=size_basis, ascent_units=ascent,
                       descent_units=descent, cmap=cmap, hmtx=advances, resolved_family=resolved,
                       weight_class=int(weight), kern_pairs=kern)


def builtin_metrics(family: str = "Arial", bold: bool = True) -> FontMetrics:
    advances = {0x20 + index: float(width) for index, width in enumerate(_ARIAL_BOLD_ASCII)}
    advances.update({code: float(width) for code, width in _ARIAL_BOLD_EXTRA.items()})
    return FontMetrics(family=family, bold=bold, source="builtin:arial_bold_compatible", units_per_em=1000.0,
                       size_basis_units=_ARIAL_SIZE_BASIS_PER_1000, ascent_units=1854 / 2.048,
                       descent_units=434 / 2.048, advances=advances)


# ============================================================
# FONT RESOLUTION
# ============================================================

def _fc_match(family: str, bold: bool) -> tuple[Path, int] | None:
    binary = shutil.which("fc-match")
    if not binary:
        return None
    pattern = f"{family}:bold" if bold else family
    try:
        # Explicit UTF-8: on Windows text=True alone decodes with the ANSI code page (cp1252 / cp1254)
        # and fails on non-ASCII font paths.
        result = subprocess.run([binary, "-f", "%{file}\n%{index}", pattern], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, timeout=5,
                                check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = (result.stdout or "").strip().splitlines()
    if result.returncode != 0 or not lines or not lines[0].strip():
        return None
    try:
        index = int(lines[1]) if len(lines) > 1 and lines[1].strip() else 0
    except ValueError:
        index = 0
    path = Path(lines[0].strip())
    return (path, index) if path.is_file() else None


def _windows_font(family: str, bold: bool) -> tuple[Path, int] | None:
    if not sys.platform.startswith("win"):
        return None
    name = _WINDOWS_FILES.get((family.strip().casefold(), bold))
    if not name:
        return None
    for root in (os.environ.get("WINDIR", r"C:\Windows"),):
        candidate = Path(root) / "Fonts" / name
        if candidate.is_file():
            return candidate, 0
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidate = Path(local) / "Microsoft" / "Windows" / "Fonts" / name
        if candidate.is_file():
            return candidate, 0
    return None


@functools.lru_cache(maxsize=16)
def load_metrics(family: str, bold: bool = True) -> FontMetrics:
    """Metrics of the face libass is expected to use; built-in table otherwise."""
    for resolver in (_windows_font, _fc_match):
        found = resolver(family, bold)
        if found is None:
            continue
        path, index = found
        try:
            return parse_font(path, family=family, bold=bold, face_index=index)
        except (OSError, FontParseError):
            continue
    return builtin_metrics(family, bold)


# ============================================================
# SHAPERS (layout width providers)
# ============================================================

class NaiveShaper:
    """Sum of advances (V3.1 behaviour, dependency-free)."""

    name = "naive"

    def __init__(self, metrics: FontMetrics) -> None:
        self.metrics = metrics

    def width(self, text: str, font_px: float, *, scale_x: float = 1.0) -> float:
        return self.metrics.text_width(text, font_px, scale_x=scale_x)


class HarfBuzzEnvelopeShaper:
    """Optional: HarfBuzz-shaped width with kerning off (libass parity measured),
    combined as max(naive, shaped) so the estimate never shrinks."""

    name = "harfbuzz_envelope"

    def __init__(self, metrics: FontMetrics, hb_module: Any) -> None:
        self.metrics = metrics
        self._hb = hb_module
        face = hb_module.Face(hb_module.Blob.from_file_path(metrics.source))
        self._font = hb_module.Font(face)
        self._upem = float(face.upem)
        self._cache: dict[str, float] = {}

    def _shaped_units(self, text: str) -> float:
        if text not in self._cache:
            buffer = self._hb.Buffer()
            buffer.add_str(text)
            buffer.guess_segment_properties()
            self._hb.shape(self._font, buffer, {"kern": False})
            self._cache[text] = float(sum(p.x_advance for p in buffer.glyph_positions))
        return self._cache[text]

    def width(self, text: str, font_px: float, *, scale_x: float = 1.0) -> float:
        naive = self.metrics.text_width(text, font_px, scale_x=scale_x)
        shaped = (self._shaped_units(text) * self.metrics.units_per_em / self._upem
                  * float(font_px) / self.metrics.size_basis_units * float(scale_x))
        return max(naive, shaped)


SHAPER_MODES = ("naive", "auto", "harfbuzz")


def select_shaper(metrics: FontMetrics, mode: str = "naive") -> tuple[Any, str]:
    """(shaper, note). ``auto``/``harfbuzz`` need uharfbuzz and a real font file."""
    mode = mode if mode in SHAPER_MODES else "naive"
    if mode == "naive":
        return NaiveShaper(metrics), "naive"
    if metrics.is_builtin:
        return NaiveShaper(metrics), "naive (builtin metrics: no font file to shape)"
    try:
        import uharfbuzz  # optional dependency
        return HarfBuzzEnvelopeShaper(metrics, uharfbuzz), "harfbuzz_envelope"
    except Exception as error:  # optional backend: never required
        return NaiveShaper(metrics), f"naive (harfbuzz unavailable: {type(error).__name__})"
