"""ASS subtitle writer: karaoke active word, emphasis, stable geometry, hook title."""
from __future__ import annotations

from typing import Any, Sequence

from mimir.config import CaptionStyle
from mimir.captions.layout import Group, RenderWord

HOOK_START_SCALE, HOOK_LAND, HOOK_BOUNCE = 142, 94, 106
HOOK_LAND_MS, HOOK_BOUNCE_MS, HOOK_SETTLE_MS = 85, 145, 225
HOOK_FADE_OUT_MS = 90


def ass_time(seconds: float) -> str:
    centis = int(round(max(0.0, seconds) * 100))
    h, rest = divmod(centis, 360000)
    m, rest = divmod(rest, 6000)
    s, cs = divmod(rest, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def escape(text: str) -> str:
    return (str(text).replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " "))


def header(style: CaptionStyle, width: int, height: int) -> str:
    font = style.font
    return (
        "[Script Info]\nTitle: MIMIR captions\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nScaledBorderAndShadow: yes\nWrapStyle: 2\nYCbCr Matrix: TV.709\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding\n"
        f"Style: Main,{font},{style.size},{style.base_color},{style.base_color},&H00000000,&H80000000,-1,0,0,0,100,100,"
        f"0,0,1,{style.outline},{style.shadow},2,{style.margin_h},{style.margin_h},{style.margin_v},1\n"
        f"Style: Secondary,{font},{style.size},{style.secondary_base_color},{style.secondary_base_color},&H00000000,"
        f"&H80000000,-1,0,0,0,100,100,0,0,1,{style.outline},{style.shadow},2,{style.margin_h},{style.margin_h},"
        f"{style.secondary_margin_v},1\n"
        f"Style: Hook,{style.hook_font},{style.hook_size},{style.hook_color},{style.hook_color},&H00000000,&H00000000,"
        f"-1,0,0,0,100,100,-1,0,1,7,0,5,60,60,0,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def _palette(style: CaptionStyle, lane: str) -> tuple[str, str, str, str]:
    if lane == "secondary":
        return style.secondary_base_color, style.secondary_active_color, style.secondary_emphasis_color, "Secondary"
    return style.base_color, style.active_color, style.emphasis_color, "Main"


def render_word(word: RenderWord, *, active: bool, style: CaptionStyle, lane: str, upper: bool) -> str:
    base, active_color, emphasis_color, name = _palette(style, lane)
    text = escape(word.text.upper() if upper else word.text)
    if active and word.emphasis:
        return f"{{\\b1\\c{emphasis_color}\\fscx94\\fscy94\\t(0,70,\\fscx116\\fscy116)}}{text}{{\\r{name}}}"
    if active:
        return f"{{\\b1\\c{active_color}\\fscx96\\fscy96\\t(0,70,\\fscx108\\fscy108)}}{text}{{\\r{name}}}"
    if word.emphasis:
        return f"{{\\b1\\c{emphasis_color}}}{text}{{\\r{name}}}"
    return f"{{\\c{base}}}{text}{{\\r{name}}}"


def group_text(group: Group, active_index: int, style: CaptionStyle) -> str:
    lane = group.lane
    _, active_color, _, name = _palette(style, lane)
    parts = []
    for index, word in enumerate(group.words):
        if index > active_index:  # future words keep the layout stable but stay invisible
            text = escape(word.text.upper() if style.uppercase else word.text)
            parts.append(f"{{\\alpha&HFF&}}{text}{{\\r{name}}}")
        else:
            parts.append(render_word(word, active=index == active_index, style=style, lane=lane,
                                     upper=style.uppercase))
    body = " ".join(parts)
    label = group.words[0].label
    if label:
        body = f"{{\\b1\\c{active_color}}}{escape(label)}:{{\\r{name}}} " + body
    return body


def dialogue(start: float, end: float, style_name: str, text: str, *, layer: int = 0) -> str:
    return f"Dialogue: {layer},{ass_time(start)},{ass_time(end)},{style_name},,0,0,0,,{text}"


def split_two_lines(text: str) -> tuple[str, str]:
    words = text.split()
    if len(words) < 3:
        return text, ""
    best = min(range(1, len(words)), key=lambda k: abs(len(" ".join(words[:k])) - len(" ".join(words[k:]))))
    return " ".join(words[:best]), " ".join(words[best:])


def hook_events(text: str, start: float, end: float, style: CaptionStyle, width: int, height: int,
                y: int | None = None) -> list[str]:
    """Bold title on the cold open: lands with a small bounce, fades before the restart."""
    if not text:
        return []
    shown = text.upper()
    first, second = split_two_lines(shown) if len(shown) > 18 else (shown, "")
    body = escape(first) + (("\\N" + escape(second)) if second else "")
    duration_ms = int(round((end - start) * 1000))
    fade_start = max(0, duration_ms - HOOK_FADE_OUT_MS)
    y = int(height * style.hook_y_ratio) if y is None else y
    tags = (f"{{\\an5\\pos({width // 2},{y})\\fscx{HOOK_START_SCALE}"
            f"\\fscy{HOOK_START_SCALE}\\t(0,{HOOK_LAND_MS},\\fscx{HOOK_LAND}\\fscy{HOOK_LAND})"
            f"\\t({HOOK_LAND_MS},{HOOK_BOUNCE_MS},\\fscx{HOOK_BOUNCE}\\fscy{HOOK_BOUNCE})"
            f"\\t({HOOK_BOUNCE_MS},{HOOK_SETTLE_MS},\\fscx100\\fscy100)"
            f"\\t({fade_start},{duration_ms},\\alpha&HFF&)}}")
    return [dialogue(start, end, "Hook", tags + body, layer=5)]


def seam_position(width: int, height: int, split: float) -> str:
    return f"{{\\an5\\pos({width // 2},{int(height * split)})}}"


def build_document(groups: Sequence[Group], events: Sequence[dict[str, Any]], hook: list[str], style: CaptionStyle,
                   width: int, height: int) -> str:
    lines = [header(style, width, height).rstrip("\n")]
    lines.extend(hook)
    for event in events:
        lines.append(dialogue(event["start"], event["end"], "Secondary" if event["lane"] == "secondary" else "Main",
                              event["prefix"] + event["text"], layer=1 if event["lane"] == "secondary" else 0))
    return "\n".join(lines) + "\n"
