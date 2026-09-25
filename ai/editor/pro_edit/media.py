"""Media probing with exact rational frame rates (UTF-8 safe, argument arrays).

Existing MIMIR probe helpers return float fps and use platform text decoding;
Pro Edit needs rational fps, frame counts, stream start times and a CFR check
to keep frame-domain math exact, so it has this one dedicated probe.
"""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai.editor.pro_edit.errors import EditPlanTimelineError, EditRenderError
from ai.editor.pro_edit.timebase import FrameRate

CFR_RELATIVE_TOLERANCE = 0.002
CFR_FRAME_COUNT_TOLERANCE = 2
PROBE_TIMEOUT_S = 60


@dataclass(frozen=True)
class MediaInfo:
    path: str
    width: int
    height: int
    fps: FrameRate
    avg_fps: FrameRate | None
    duration_s: float
    video_duration_s: float
    frame_count: int
    has_audio: bool
    video_start_s: float
    audio_start_s: float | None
    pix_fmt: str
    is_cfr: bool

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": str(self.fps),
            "avg_fps": str(self.avg_fps) if self.avg_fps else None,
            "duration_s": round(self.duration_s, 6),
            "video_duration_s": round(self.video_duration_s, 6),
            "frame_count": self.frame_count,
            "has_audio": self.has_audio,
            "video_start_s": round(self.video_start_s, 6),
            "audio_start_s": None if self.audio_start_s is None else round(self.audio_start_s, 6),
            "pix_fmt": self.pix_fmt,
            "is_cfr": self.is_cfr,
        }


def run_command(command: list[str], *, timeout: float, what: str) -> subprocess.CompletedProcess[str]:
    """Run an argument-array command (never a shell string) with UTF-8 output."""
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError as error:
        raise EditRenderError(f"{what}: executable not found ({command[0]})", command=command) from error
    except subprocess.TimeoutExpired as error:
        raise EditRenderError(f"{what}: timed out after {timeout:.0f}s", command=command) from error
    if completed.returncode != 0:
        raise EditRenderError(
            f"{what}: exit code {completed.returncode}",
            returncode=completed.returncode,
            stderr=completed.stderr,
            command=command,
        )
    return completed


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _rate(value: Any) -> FrameRate | None:
    text = str(value or "").strip()
    if not text or text in {"0/0", "0"}:
        return None
    try:
        return FrameRate.parse(text)
    except EditPlanTimelineError:
        return None


def probe_media(path: str | Path) -> MediaInfo:
    target = Path(path).resolve()
    if not target.is_file():
        raise EditRenderError(f"media not found: {target}")
    completed = run_command(
        [
            "ffprobe", "-v", "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height,r_frame_rate,avg_frame_rate,"
            "nb_frames,duration,start_time,pix_fmt",
            "-of", "json",
            str(target),
        ],
        timeout=PROBE_TIMEOUT_S,
        what="ffprobe",
    )
    try:
        data = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as error:
        raise EditRenderError("ffprobe returned invalid JSON") from error

    video: dict[str, Any] | None = None
    audio: dict[str, Any] | None = None
    for stream in data.get("streams", []) or []:
        if not isinstance(stream, dict):
            continue
        if stream.get("codec_type") == "video" and video is None:
            video = stream
        elif stream.get("codec_type") == "audio" and audio is None:
            audio = stream
    if video is None:
        raise EditRenderError(f"no video stream: {target}")

    width = int(_float(video.get("width")))
    height = int(_float(video.get("height")))
    if width <= 0 or height <= 0:
        raise EditRenderError("video stream has no valid resolution")
    r_rate = _rate(video.get("r_frame_rate"))
    avg_rate = _rate(video.get("avg_frame_rate"))
    fps = r_rate or avg_rate
    if fps is None:
        raise EditRenderError("video stream has no readable frame rate")

    duration = _float((data.get("format") or {}).get("duration"))
    video_duration = _float(video.get("duration"), duration)
    video_start = _float(video.get("start_time"))
    nb_frames_raw = video.get("nb_frames")
    try:
        nb_frames = int(nb_frames_raw) if nb_frames_raw not in (None, "", "N/A") else 0
    except (TypeError, ValueError):
        nb_frames = 0
    expected_frames = int(round(video_duration * fps.fps)) if video_duration > 0 else 0
    frame_count = nb_frames if nb_frames > 0 else expected_frames

    is_cfr = True
    if avg_rate is not None and r_rate is not None:
        if abs(avg_rate.fps - r_rate.fps) / r_rate.fps > CFR_RELATIVE_TOLERANCE:
            is_cfr = False
    if nb_frames > 0 and expected_frames > 0 and abs(nb_frames - expected_frames) > CFR_FRAME_COUNT_TOLERANCE:
        is_cfr = False

    return MediaInfo(
        path=str(target),
        width=width,
        height=height,
        fps=fps,
        avg_fps=avg_rate,
        duration_s=duration if duration > 0 else video_duration,
        video_duration_s=video_duration,
        frame_count=frame_count,
        has_audio=audio is not None,
        video_start_s=video_start,
        audio_start_s=_float(audio.get("start_time")) if audio is not None else None,
        pix_fmt=str(video.get("pix_fmt", "")),
        is_cfr=is_cfr,
    )
