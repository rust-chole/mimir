"""Frame access through FFmpeg raw pipes (BGR24, deterministic, no seeking drift)."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from mimir.errors import MediaError
from mimir.media.ffmpeg import ffmpeg_bin


@dataclass(frozen=True)
class SampledFrame:
    t: float                 # time (source seconds when sampled from the source)
    image: np.ndarray        # BGR uint8 (h, w, 3)


def _scaled_size(width: int, height: int, target_width: int) -> tuple[int, int]:
    if target_width <= 0 or target_width >= width:
        w, h = width, height
    else:
        w = target_width
        h = int(round(height * target_width / width))
    return w - w % 2, h - h % 2


def iter_frames(path: str | Path, *, start: float = 0.0, duration: float | None = None, fps: float | None = None,
                width: int = 0, src_width: int, src_height: int) -> Iterator[SampledFrame]:
    """Yield frames of a range. With ``fps`` frames are resampled on an exact grid ``start + k/fps``."""
    w, h = _scaled_size(src_width, src_height, width)
    filters = []
    if fps:
        filters.append(f"fps={fps}:round=near")
    if (w, h) != (src_width, src_height):
        filters.append(f"scale={w}:{h}:flags=area")
    args = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-v", "error"]
    if start > 0:
        args += ["-ss", f"{start:.6f}"]
    args += ["-i", str(path)]
    if duration is not None:
        args += ["-t", f"{duration:.6f}"]
    if filters:
        args += ["-vf", ",".join(filters)]
    args += ["-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    frame_bytes = w * h * 3
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    index = 0
    try:
        assert process.stdout is not None
        while True:
            buffer = process.stdout.read(frame_bytes)
            if len(buffer) < frame_bytes:
                break
            image = np.frombuffer(buffer, dtype=np.uint8).reshape(h, w, 3)
            t = start + (index / fps if fps else 0.0)
            yield SampledFrame(t, image)
            index += 1
    finally:
        if process.stdout:
            process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
        code = process.wait()
        if code not in (0, None) and index == 0:
            raise MediaError(f"frame decode failed for {path}: {stderr.strip()[-400:]}")


def sample_frames(path: str | Path, *, start: float, duration: float, fps: float, width: int,
                  src_width: int, src_height: int) -> list[SampledFrame]:
    return list(iter_frames(path, start=start, duration=duration, fps=fps, width=width,
                            src_width=src_width, src_height=src_height))


def frame_at(path: str | Path, t: float, *, src_width: int, src_height: int, width: int = 0) -> np.ndarray:
    """Single frame nearest to ``t`` (accurate seek)."""
    frames = list(iter_frames(path, start=max(0.0, t), duration=0.5, fps=None, width=width,
                              src_width=src_width, src_height=src_height))
    if not frames:
        raise MediaError(f"no frame at {t:.3f}s in {path}")
    return frames[0].image


def frames_at_indices(path: str | Path, indices: list[int], *, fps: int, src_width: int, src_height: int,
                      width: int = 0) -> dict[int, np.ndarray]:
    """Decode a constant-frame-rate file once and keep only the requested frame indices."""
    wanted = set(int(i) for i in indices)
    result: dict[int, np.ndarray] = {}
    if not wanted:
        return result
    last = max(wanted)
    for index, frame in enumerate(iter_frames(path, fps=None, width=width, src_width=src_width,
                                              src_height=src_height)):
        if index in wanted:
            result[index] = frame.image.copy()
        if index >= last:
            break
    return result
