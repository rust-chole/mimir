"""Sparse frame sampling via an FFmpeg rawvideo pipe (argument array, no shell)."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ai.editor.pro_edit.errors import SubjectResolutionError

DEFAULT_SAMPLE_FPS = 10.0
DEFAULT_MAX_WIDTH = 640


@dataclass(frozen=True)
class SampledFrame:
    index: int
    t: float  # PACED_CLIP seconds
    image: Any  # numpy uint8 array (H, W, 3), BGR


def analysis_size(width: int, height: int, max_width: int = DEFAULT_MAX_WIDTH) -> tuple[int, int]:
    if width <= max_width:
        w, h = width, height
    else:
        w, h = max_width, int(round(height * max_width / width))
    return w - w % 2, h - h % 2


def iterate_frames(path: str | Path, width: int, height: int, *, sample_fps: float = DEFAULT_SAMPLE_FPS,
                   max_width: int = DEFAULT_MAX_WIDTH, start_offset: float = 0.0) -> Iterator[SampledFrame]:
    import numpy as np

    w, h = analysis_size(width, height, max_width)
    command = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(Path(path)), "-an", "-sn",
        "-vf", f"fps={sample_fps:.6f},scale={w}:{h}:flags=area", "-pix_fmt", "bgr24", "-f", "rawvideo", "-",
    ]
    frame_bytes = w * h * 3
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
    except FileNotFoundError as error:
        raise SubjectResolutionError("ffmpeg not found for subject tracking") from error
    assert process.stdout is not None
    index = 0
    try:
        while True:
            chunk = process.stdout.read(frame_bytes)
            if len(chunk) < frame_bytes:
                break
            image = np.frombuffer(chunk, dtype=np.uint8).reshape((h, w, 3))
            yield SampledFrame(index, start_offset + index / sample_fps, image)
            index += 1
    finally:
        process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
        if process.poll() is None:
            process.kill()
        code = process.wait()
        if process.stderr:
            process.stderr.close()
        if code not in (0, None) and index == 0:
            raise SubjectResolutionError(f"frame sampling failed ({code}): {stderr[-500:]}")
