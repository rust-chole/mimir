"""Frame-safe Pro Edit execution: one filter graph, one encode, atomic output.

The executor understands only a ResolvedPlan (frame-domain geometry). It runs
FFmpeg with argument arrays (no shell), streams-copies audio, writes to a
temporary file, validates the result against the input (duration, frame
count, fps, geometry, audio clock) and only then atomically replaces the
target. It never runs anything derived from model text: the filter graph is
built exclusively from numbers produced by the deterministic preset engine.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ai.editor import caption_renderer
from ai.editor.pro_edit.errors import EditRenderError
from ai.editor.pro_edit.ffmpeg_filters import (
    FfmpegCapabilities,
    build_video_filter,
    script_arguments,
    validate_filter_graph,
)
from ai.editor.pro_edit.media import MediaInfo, probe_media, run_command
from ai.editor.pro_edit.presets import ResolvedPlan

RENDERER_VERSION = 1


def subtitle_filter_for(caption_file: str | Path, fonts_dir: str | Path | None = None) -> str:
    """Exactly the baseline caption burn-in filter (reuses caption_renderer escaping).

    ``fonts_dir`` (brand profile font file directory) is added only when set.
    """
    base = f"subtitles=filename='{caption_renderer.escape_filter_path(caption_file)}'"
    if fonts_dir is None:
        return base
    return base + f":fontsdir='{caption_renderer.escape_filter_path(fonts_dir)}'"


@dataclass(frozen=True)
class ExpectedOutput:
    width: int
    height: int
    source: MediaInfo


@dataclass(frozen=True)
class RenderResult:
    output_path: Path
    media: MediaInfo
    command: tuple[str, ...]


def build_render_command(edited_clip: Path, script_path: Path, output_path: Path, caps: FfmpegCapabilities) -> list[str]:
    """Same encoder settings as caption_renderer.render_captioned_clip; audio is copied."""
    return [
        "ffmpeg", "-y", "-hide_banner", "-nostdin", "-loglevel", "error",
        "-i", str(edited_clip),
        *script_arguments(caps, script_path),
        "-c:v", caption_renderer.VIDEO_CODEC,
        "-preset", caption_renderer.VIDEO_PRESET,
        "-crf", caption_renderer.VIDEO_CRF,
        "-pix_fmt", caption_renderer.PIXEL_FORMAT,
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]


def validate_output(result: MediaInfo, expected: ExpectedOutput) -> None:
    source = expected.source
    frame = source.fps.frame_duration
    problems: list[str] = []
    if (result.width, result.height) != (expected.width, expected.height):
        problems.append(f"resolution {result.width}x{result.height} != {expected.width}x{expected.height}")
    if result.fps.fraction != source.fps.fraction:
        problems.append(f"fps {result.fps} != {source.fps}")
    if source.frame_count and result.frame_count and result.frame_count != source.frame_count:
        problems.append(f"frame count {result.frame_count} != {source.frame_count}")
    if abs(result.video_duration_s - source.video_duration_s) > frame + 1e-3:
        problems.append(f"video duration {result.video_duration_s:.4f}s != {source.video_duration_s:.4f}s")
    if abs(result.duration_s - source.duration_s) > frame + 1e-2:
        problems.append(f"container duration {result.duration_s:.4f}s != {source.duration_s:.4f}s")
    if result.has_audio != source.has_audio:
        problems.append(f"audio presence {result.has_audio} != {source.has_audio}")
    if source.has_audio and result.audio_start_s is not None and source.audio_start_s is not None:
        if abs(result.audio_start_s - source.audio_start_s) > frame / 2:
            problems.append(f"audio start {result.audio_start_s:.4f}s != {source.audio_start_s:.4f}s")
    if abs(result.video_start_s - source.video_start_s) > frame / 2:
        problems.append(f"video start {result.video_start_s:.4f}s != {source.video_start_s:.4f}s")
    if problems:
        raise EditRenderError("post-render validation failed: " + "; ".join(problems))


def render_camera_captions(
    *,
    edited_clip: str | Path,
    caption_file: str | Path | None,
    output_path: str | Path,
    resolved: ResolvedPlan,
    caps: FfmpegCapabilities,
    source_media: MediaInfo,
    script_path: str | Path,
    interpolation: str = "cubic",
    keep_failed: bool = False,
    fonts_dir: str | Path | None = None,
) -> RenderResult:
    edited = Path(edited_clip).resolve()
    output = Path(output_path).resolve()
    script = Path(script_path).resolve()
    if not caps.camera_ready:
        raise EditRenderError("FFmpeg capabilities insufficient for camera rendering")
    if resolved.frame_count != source_media.frame_count:
        raise EditRenderError("resolved plan frame count does not match the paced clip")
    subtitle = subtitle_filter_for(caption_file, fonts_dir) if caption_file is not None else None
    graph = build_video_filter(resolved, caps, subtitle_filter=subtitle, interpolation=interpolation)
    validate_filter_graph(graph)
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(graph, encoding="utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.stem + ".proedit-tmp" + output.suffix)
    temp.unlink(missing_ok=True)
    command = build_render_command(edited, script, temp, caps)
    timeout = max(900.0, source_media.duration_s * 40.0)
    try:
        run_command(command, timeout=timeout, what="pro edit render")
        if not temp.is_file() or temp.stat().st_size < 1024:
            raise EditRenderError("pro edit render produced no output", command=command)
        width, height = resolved.output_size
        result = probe_media(temp)
        validate_output(result, ExpectedOutput(width, height, source_media))
    except EditRenderError:
        if not keep_failed:
            temp.unlink(missing_ok=True)
        raise
    os.replace(temp, output)
    return RenderResult(output, probe_media(output), tuple(command))
