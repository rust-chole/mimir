"""FFmpeg filter construction for resolved camera paths (testable, structured).

Rendering primitive: the ``perspective`` filter with per-frame evaluation maps
an axis-aligned source crop rectangle onto the full frame with sub-pixel
interpolation (no integer-pixel jitter, no size renegotiation, timestamps and
frame count untouched). It is combined with the existing ``subtitles`` burn
in ONE filter graph -> one encode, captions rendered on the final geometry.

The per-frame counter of ``perspective`` and the timeline ``n`` variable are
calibrated once against the installed FFmpeg (bases differ between builds),
and the graph is passed through a filter-script file so long expressions can
never hit the Windows command-line limit.
"""
from __future__ import annotations

import functools
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from ai.editor.pro_edit.camera import IDENTITY, CameraPath, CameraSegment, OutputProfile
from ai.editor.pro_edit.errors import EditRenderError
from ai.editor.pro_edit.media import run_command
from ai.editor.pro_edit.presets import ResolvedPlan

FILTER_BUILDER_VERSION = 2
_ALLOWED_FILTERS = frozenset({"crop", "perspective", "subtitles", "null"})
_CALIBRATION_TIMEOUT_S = 60


def fmt(value: float) -> str:
    """Deterministic compact decimal (no exponent) for expressions."""
    text = f"{float(value):.7f}".rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


_EASE_EXPR = {
    "linear": "{u}",
    "smoothstep": "({u})*({u})*(3-2*({u}))",
    "ease_out_cubic": "(1-pow(1-({u}),3))",
}


def _segment_channel_expr(segment: CameraSegment, channel: str, frame_var: str) -> str:
    start = getattr(segment.start, channel)
    end = getattr(segment.end, channel)
    if abs(end - start) <= 1e-12:
        return fmt(start)
    span = max(1, segment.end_frame - 1 - segment.start_frame)
    u = f"clip(({frame_var}-{segment.start_frame})/{span},0,1)"
    eased = _EASE_EXPR[segment.easing].format(u=u)
    return f"({fmt(start)}+({fmt(end - start)})*{eased})"


def _partition(path: CameraPath) -> list[tuple[int, int, CameraSegment | None]]:
    """Complete [0, frame_count) partition; None = identity gap."""
    rows: list[tuple[int, int, CameraSegment | None]] = []
    cursor = 0
    for segment in path.segments:
        if segment.start_frame > cursor:
            rows.append((cursor, segment.start_frame, None))
        rows.append((segment.start_frame, segment.end_frame, segment))
        cursor = segment.end_frame
    if cursor < path.frame_count:
        rows.append((cursor, path.frame_count, None))
    return rows


def channel_expression(path: CameraPath, channel: str, frame_var: str) -> str:
    """Balanced if/lt tree over the partition: depth O(log segments)."""
    identity = fmt(getattr(IDENTITY, channel))
    rows = [(a, b, identity if seg is None else _segment_channel_expr(seg, channel, frame_var))
            for a, b, seg in _partition(path)]
    # Merge neighbouring constant pieces with identical expressions.
    merged: list[list[object]] = []
    for a, b, expr in rows:
        if merged and merged[-1][2] == expr:
            merged[-1][1] = b
        else:
            merged.append([a, b, expr])

    def build(lo: int, hi: int) -> str:
        if lo == hi:
            return str(merged[lo][2])
        mid = (lo + hi) // 2
        boundary = int(merged[mid + 1][0])  # type: ignore[arg-type]
        return f"if(lt({frame_var},{boundary}),{build(lo, mid)},{build(mid + 1, hi)})"

    return build(0, len(merged) - 1) if merged else identity


def _coordinate_expressions(zoom: str, cx: str, cy: str) -> dict[str, str]:
    # st/ld keep each coordinate expression evaluating every channel tree once.
    # max(1, zoom) guards float round-off (e.g. 1.16 - 0.16 = 0.9999999999999999):
    # a zoom a hair below 1 would make the crop wider than the frame and invert
    # the clip() range, which the perspective filter rejects (EINVAL).
    z = f"st(0,max(1,{zoom}))"
    x0 = f"{z};st(1,{cx});clip(ld(1)*W-W/(2*ld(0)),0,max(0,W-W/ld(0)))"
    x1 = f"{z};st(1,{cx});clip(ld(1)*W-W/(2*ld(0)),0,max(0,W-W/ld(0)))+W/ld(0)"
    y0 = f"{z};st(1,{cy});clip(ld(1)*H-H/(2*ld(0)),0,max(0,H-H/ld(0)))"
    y1 = f"{z};st(1,{cy});clip(ld(1)*H-H/(2*ld(0)),0,max(0,H-H/ld(0)))+H/ld(0)"
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y0, "x2": x0, "y2": y1, "x3": x1, "y3": y1}


@dataclass(frozen=True)
class FfmpegCapabilities:
    version: str
    has_perspective: bool
    has_subtitles: bool
    perspective_frame_base: int | None
    timeline_frame_base: int | None
    script_option: str | None

    @property
    def camera_ready(self) -> bool:
        return self.has_perspective and self.perspective_frame_base is not None and self.script_option is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "has_perspective": self.has_perspective,
            "has_subtitles": self.has_subtitles,
            "perspective_frame_base": self.perspective_frame_base,
            "timeline_frame_base": self.timeline_frame_base,
            "script_option": self.script_option,
        }


def build_perspective_filter(path: CameraPath, caps: FfmpegCapabilities, *, interpolation: str = "linear") -> str:
    if caps.perspective_frame_base is None:
        raise EditRenderError("perspective frame counter base is not calibrated")
    if interpolation not in {"linear", "cubic"}:
        raise EditRenderError(f"unsupported interpolation {interpolation!r}")
    frame_var = f"(in-{caps.perspective_frame_base})" if caps.perspective_frame_base else "in"
    zoom = channel_expression(path, "zoom", frame_var)
    cx = channel_expression(path, "cx", frame_var)
    cy = channel_expression(path, "cy", frame_var)
    coords = _coordinate_expressions(zoom, cx, cy)
    options = [f"{key}='{value}'" for key, value in coords.items()]
    options += [f"interpolation={interpolation}", "sense=source", "eval=frame"]
    if caps.timeline_frame_base is not None:
        n_var = f"(n-{caps.timeline_frame_base})" if caps.timeline_frame_base else "n"
        ranges = path.active_ranges()
        if ranges:
            enable = "+".join(f"between({n_var},{a},{b - 1})" for a, b in ranges)
            options.append(f"enable='{enable}'")
    return "perspective=" + ":".join(options)


def build_video_filter(
    resolved: ResolvedPlan,
    caps: FfmpegCapabilities,
    *,
    subtitle_filter: str | None,
    interpolation: str = "linear",
) -> str:
    """Simple filter chain: [base crop] -> [camera] -> [subtitles]."""
    parts: list[str] = []
    if resolved.output_profile is not OutputProfile.PRESERVE:
        bx, by, bw, bh = resolved.base
        parts.append(f"crop={bw}:{bh}:{bx}:{by}:exact=1")
    if not resolved.path.is_identity:
        parts.append(build_perspective_filter(resolved.path, caps, interpolation=interpolation))
    if subtitle_filter:
        parts.append(subtitle_filter)
    if not parts:
        parts.append("null")
    graph = ",".join(parts)
    validate_filter_graph(graph)
    return graph


def _split_top_level(graph: str) -> list[str]:
    parts: list[str] = []
    depth_quote = False
    current: list[str] = []
    index = 0
    while index < len(graph):
        char = graph[index]
        if char == "\\" and index + 1 < len(graph):
            current.append(graph[index:index + 2])
            index += 2
            continue
        if char == "'":
            depth_quote = not depth_quote
        if char == "," and not depth_quote:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
        index += 1
    if depth_quote:
        raise EditRenderError("filter graph has unbalanced quotes")
    parts.append("".join(current))
    return parts


def validate_filter_graph(graph: str) -> None:
    """Structural check before execution: non-empty, known filters, no labels,
    no stray separators/newlines, balanced quotes and parentheses."""
    if not graph or not graph.strip():
        raise EditRenderError("empty filter graph")
    if any(ch in graph for ch in ("\n", "\r", "\x00")):
        raise EditRenderError("filter graph contains control characters")
    for part in _split_top_level(graph):
        name = part.split("=", 1)[0].strip()
        if not re.fullmatch(r"[a-z_]+", name) or name not in _ALLOWED_FILTERS:
            raise EditRenderError(f"filter graph uses unexpected filter {name!r}")
        if name == "perspective":
            body = part.split("=", 1)[1]
            unquoted = re.sub(r"'[^']*'", "", body)
            if ";" in unquoted or "[" in unquoted:
                raise EditRenderError("perspective options contain graph separators")
            for expr in re.findall(r"'([^']*)'", body):
                depth = 0
                for ch in expr:
                    depth += ch == "("
                    depth -= ch == ")"
                    if depth < 0:
                        break
                if depth != 0:
                    raise EditRenderError("unbalanced parentheses in camera expression")
    if graph.lstrip().startswith("["):
        raise EditRenderError("simple filter chain must not use stream labels")


# ============================================================
# CAPABILITY CALIBRATION (once per FFmpeg install)
# ============================================================

def _framemd5(args: Sequence[str], frames: int) -> list[str]:
    completed = run_command(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=size=64x36:rate=10:duration=2", *args,
         "-an", "-frames:v", str(frames), "-f", "framemd5", "-"],
        timeout=_CALIBRATION_TIMEOUT_S,
        what="ffmpeg calibration",
    )
    return [line.rsplit(",", 1)[-1].strip() for line in completed.stdout.splitlines()
            if line and not line.startswith("#")]


def _first_changed(reference: list[str], candidate: list[str]) -> int | None:
    for index, (a, b) in enumerate(zip(reference, candidate)):
        if a != b:
            return index
    return None


def _calibrate_base(reference: list[str], build: Callable[[int], str], probes: Sequence[int] = (5, 8)) -> int | None:
    bases: set[int] = set()
    for k in probes:
        changed = _first_changed(reference, _framemd5(["-vf", build(k)], len(reference)))
        if changed is None:
            return None
        bases.add(k - changed)
    return bases.pop() if len(bases) == 1 else None


@functools.lru_cache(maxsize=1)
def probe_capabilities() -> FfmpegCapabilities:
    """Deterministic per install; raises EditRenderError if FFmpeg is absent."""
    version_out = run_command(["ffmpeg", "-hide_banner", "-version"], timeout=30, what="ffmpeg -version").stdout
    version = version_out.splitlines()[0].strip() if version_out else "unknown"
    filters = run_command(["ffmpeg", "-hide_banner", "-filters"], timeout=30, what="ffmpeg -filters").stdout
    names = {line.split()[1] for line in filters.splitlines() if len(line.split()) >= 3 and line.startswith(" ")}
    has_perspective = "perspective" in names
    has_subtitles = "subtitles" in names
    persp_base: int | None = None
    timeline_base: int | None = None
    script_option: str | None = None
    if has_perspective:
        reference = _framemd5(["-vf", "format=gray"], 12)
        shift = "x0='{e}':y0=0:x1=W:y1=0:x2='{e}':y2=H:x3=W:y3=H:eval=frame"
        persp_base = _calibrate_base(
            reference, lambda k: "perspective=" + shift.format(e=f"if(lt(in,{k}),0,W/2)") + ",format=gray")
        timeline_base = _calibrate_base(
            reference,
            lambda k: "perspective=x0=W/2:y0=0:x1=W:y1=0:x2=W/2:y2=H:x3=W:y3=H:eval=frame"
                      f":enable='gte(n,{k})',format=gray")
        if timeline_base is not None and persp_base is not None:
            # Frames skipped by ``enable`` must still advance the perspective
            # counter: frames 0-2 untouched, 3-5 = W/4 shift, 6+ = W/2 shift.
            combined = _framemd5(["-vf", "perspective=" + shift.format(
                e=f"if(lt(in-{persp_base},6),W/4,W/2)") + f":enable='gte(n-{timeline_base},3)',format=gray"], 12)
            quarter = _framemd5(["-vf", "perspective=" + shift.format(e="W/4") + ",format=gray"], 12)
            half = _framemd5(["-vf", "perspective=" + shift.format(e="W/2") + ",format=gray"], 12)
            expected = reference[:3] + quarter[3:6] + half[6:12]
            if combined != expected:
                timeline_base = None
    with tempfile.TemporaryDirectory(prefix="mimir_pro_edit_caps_") as tmp:
        script = Path(tmp) / "graph.txt"
        script.write_text("null", encoding="utf-8")
        for option in ("-/vf", "-filter_script:v"):
            try:
                _framemd5([option, str(script)], 2)
            except EditRenderError:
                continue
            script_option = option
            break
    return FfmpegCapabilities(version, has_perspective, has_subtitles, persp_base, timeline_base, script_option)


def script_arguments(caps: FfmpegCapabilities, script_path: Path) -> list[str]:
    if caps.script_option is None:
        raise EditRenderError("FFmpeg supports no filter-script option")
    return [caps.script_option, str(script_path)]
