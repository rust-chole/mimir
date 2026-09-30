"""The canonical analysis audio of the FINAL edited short.

One artifact per caption run: mono 16 kHz PCM16 WAV extracted from the exact
edited clip that is rendered and published (never the source VOD window, never
pre-edit timing, never a cached file from an earlier edit). Every lexical ear
and every word-alignment provider reads this file or exact sample-accurate cuts
of it, so all evidence shares one clock. The render audio is untouched.
"""
from __future__ import annotations

import array
import base64
import hashlib
import math
import subprocess
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai.caption_stack.config import ALIGNER_MAX_INPUT_SECONDS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEMP_DIR = PROJECT_ROOT / "vod_output" / "temp" / "caption_stack"
ANALYSIS_SAMPLE_RATE = 16000
ENERGY_WINDOW_S = 0.10          # low-energy boundary search resolution (same idea as the official splitter)
CHUNK_SEARCH_S = 10.0           # how far back from the limit a chunk boundary may move


@dataclass(frozen=True)
class AnalysisAudio:
    path: Path
    duration: float
    sample_rate: int
    frames: int
    sha256: str
    source: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {"format": "wav/pcm_s16le/mono", "sample_rate": self.sample_rate, "duration": round(self.duration, 4),
                "sha256": self.sha256[:16], "source": dict(self.source)}


def _fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"name": path.name, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _read_info(path: Path) -> tuple[int, int, int, int]:
    with wave.open(str(path), "rb") as wav:
        return wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()


def load_analysis_audio(path: str | Path, *, source: dict[str, Any] | None = None) -> AnalysisAudio:
    """Wrap an existing mono 16 kHz PCM16 WAV (validated) as the analysis artifact."""
    path = Path(path).resolve()
    rate, channels, width, frames = _read_info(path)
    if channels != 1 or width != 2 or rate != ANALYSIS_SAMPLE_RATE:
        raise RuntimeError(f"analysis audio must be mono 16 kHz PCM16, got {channels} ch / {rate} Hz / {8 * width} bit")
    if frames <= 0:
        raise RuntimeError("analysis audio is empty")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return AnalysisAudio(path, frames / float(rate), rate, frames, digest, dict(source or {}))


def extract_analysis_audio(edited_clip_path: str | Path, *, directory: str | Path | None = None) -> AnalysisAudio:
    """Extract the analysis WAV from the final edited clip (always fresh; the caller deletes it)."""
    clip = Path(edited_clip_path).resolve()
    if not clip.is_file():
        raise FileNotFoundError(f"final edited clip not found: {clip}")
    fingerprint = _fingerprint(clip)
    out_dir = Path(directory) if directory else TEMP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(repr((str(clip), fingerprint)).encode("utf-8")).hexdigest()[:16]
    output = out_dir / f"analysis_{key}.wav"
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(clip), "-vn", "-map", "0:a:0",
               "-ac", "1", "-ar", str(ANALYSIS_SAMPLE_RATE), "-c:a", "pcm_s16le", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               check=False)
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size < 1024:
        raise RuntimeError("final-short analysis audio extraction failed: "
                           + (completed.stderr.strip()[-400:] or "ffmpeg produced no audio"))
    return load_analysis_audio(output, source=fingerprint)


def read_samples(audio: AnalysisAudio, start: float = 0.0, end: float | None = None) -> array.array:
    """PCM16 samples of [start, end) (sample-accurate, clamped to the file)."""
    first = max(0, min(audio.frames, int(round(float(start) * audio.sample_rate))))
    last = audio.frames if end is None else max(first, min(audio.frames, int(round(float(end) * audio.sample_rate))))
    with wave.open(str(audio.path), "rb") as wav:
        wav.setpos(first)
        raw = wav.readframes(last - first)
    samples = array.array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def cut_window(audio: AnalysisAudio, start: float, end: float, output: str | Path) -> tuple[Path, float, float]:
    """Write the exact [start, end) cut; returns (path, actual_start, actual_end) on the analysis clock."""
    first = max(0, min(audio.frames, int(round(float(start) * audio.sample_rate))))
    last = max(first, min(audio.frames, int(round(float(end) * audio.sample_rate))))
    if last - first < int(0.05 * audio.sample_rate):
        raise ValueError(f"audio window too short: {start:.3f}-{end:.3f}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(audio.path), "rb") as source:
        source.setpos(first)
        raw = source.readframes(last - first)
    with wave.open(str(output), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(audio.sample_rate)
        target.writeframes(raw)
    return output, first / float(audio.sample_rate), last / float(audio.sample_rate)


def wav_base64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


# ============================================================
# DETERMINISTIC CHUNK PLAN (long shorts only)
# ============================================================

def window_energies(audio: AnalysisAudio, start: float, end: float, window_s: float = ENERGY_WINDOW_S
                    ) -> list[tuple[float, float]]:
    """(window start time, RMS) for consecutive windows of [start, end)."""
    samples = read_samples(audio, start, end)
    size = max(1, int(round(window_s * audio.sample_rate)))
    first = int(round(float(start) * audio.sample_rate))
    rows: list[tuple[float, float]] = []
    for offset in range(0, max(0, len(samples) - size + 1), size):
        frame = samples[offset:offset + size]
        rms = math.sqrt(sum(float(v) * float(v) for v in frame) / float(len(frame)))
        rows.append(((first + offset) / float(audio.sample_rate), rms))
    return rows


def plan_chunks(audio: AnalysisAudio, max_seconds: float = ALIGNER_MAX_INPUT_SECONDS,
                search_seconds: float = CHUNK_SEARCH_S) -> list[tuple[float, float]]:
    """Contiguous [start, end) windows covering the whole short, each <= ``max_seconds``.

    A short within the limit is ONE window. Otherwise each boundary is the
    centre of the quietest ``ENERGY_WINDOW_S`` window within ``search_seconds``
    before the limit (the audio decides where; no split time is fixed), so a
    boundary falls in a pause whenever the audio has one near the limit.
    """
    duration = audio.duration
    max_seconds = float(max_seconds)
    if max_seconds <= 1.0:
        raise ValueError("chunk limit must exceed one second")
    if duration <= max_seconds:
        return [(0.0, duration)]
    search = max(ENERGY_WINDOW_S * 2, min(float(search_seconds), max_seconds / 2.0))
    windows: list[tuple[float, float]] = []
    start = 0.0
    while duration - start > max_seconds:
        limit = start + max_seconds
        energies = window_energies(audio, limit - search, limit)
        if energies:
            quiet_start, _rms = min(energies, key=lambda row: (row[1], -row[0]))
            boundary = min(limit, quiet_start + ENERGY_WINDOW_S / 2.0)
        else:
            boundary = limit
        boundary = round(boundary * audio.sample_rate) / float(audio.sample_rate)
        windows.append((start, boundary))
        start = boundary
    windows.append((start, duration))
    return windows
