"""ffprobe -> MediaInfo."""
from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from mimir.errors import MediaError
from mimir.media.ffmpeg import ffprobe_bin, run


@dataclass(frozen=True)
class MediaInfo:
    path: str
    duration: float
    width: int
    height: int
    fps: float
    fps_fraction: str
    video_codec: str
    pix_fmt: str
    has_audio: bool
    audio_codec: str
    audio_rate: int
    audio_channels: int
    frame_count: int | None
    rotation: int = 0

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MediaInfo":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


def _fraction(value: str | None) -> Fraction:
    if not value or value in {"0/0", "N/A"}:
        return Fraction(0)
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return Fraction(0)


def probe(path: str | Path, *, count_frames: bool = False) -> MediaInfo:
    path = Path(path).resolve()
    if not path.is_file():
        raise MediaError(f"media file not found: {path}")
    args = [ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams"]
    if count_frames:
        args += ["-count_frames"]
    result = run([*args, str(path)], capture_stdout=True, what="ffprobe", timeout=600)
    try:
        data = json.loads(result.stdout.decode("utf-8", "replace"))
    except ValueError as error:
        raise MediaError(f"ffprobe returned invalid JSON for {path}") from error
    streams = data.get("streams", []) or []
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not (s.get("disposition") or {}).get("attached_pic")), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise MediaError(f"no video stream in {path}")
    fps = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate"))
    duration = float((data.get("format") or {}).get("duration") or video.get("duration") or 0.0)
    if duration <= 0:
        raise MediaError(f"unknown duration for {path}")
    rotation = 0
    for side in video.get("side_data_list", []) or []:
        if "rotation" in side:
            try:
                rotation = int(side["rotation"]) % 360
            except (TypeError, ValueError):
                rotation = 0
    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    if rotation in (90, 270):
        width, height = height, width
    frames = video.get("nb_read_frames") if count_frames else video.get("nb_frames")
    return MediaInfo(
        path=str(path),
        duration=duration,
        width=width,
        height=height,
        fps=float(fps) if fps else 0.0,
        fps_fraction=f"{fps.numerator}/{fps.denominator}" if fps else "0/1",
        video_codec=str(video.get("codec_name", "")),
        pix_fmt=str(video.get("pix_fmt", "")),
        has_audio=audio is not None,
        audio_codec=str((audio or {}).get("codec_name", "")),
        audio_rate=int((audio or {}).get("sample_rate") or 0),
        audio_channels=int((audio or {}).get("channels") or 0),
        frame_count=int(frames) if frames not in (None, "N/A", "") else None,
        rotation=rotation,
    )


def audio_duration(path: str | Path) -> float:
    """Duration of an audio-only asset (SFX library files)."""
    result = run([ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_format", str(Path(path).resolve())],
                 capture_stdout=True, what="ffprobe", timeout=120)
    try:
        return float(json.loads(result.stdout.decode("utf-8", "replace"))["format"]["duration"])
    except (KeyError, ValueError, TypeError) as error:
        raise MediaError(f"cannot read duration of {path}") from error


def output_fps_for(info: MediaInfo, configured: int = 0) -> int:
    """Integer output rate: exact audio samples per frame at 48 kHz, stable A/V sync."""
    if configured:
        return int(configured)
    if info.fps >= 49.0:
        return 60
    for exact in (24, 25):
        if abs(info.fps - exact) < 0.01:
            return exact
    return 30
