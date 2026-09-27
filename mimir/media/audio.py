"""Audio extraction and measured acoustic evidence (numpy).

All times are seconds relative to the start of the decoded range; callers add
their own offset (source time).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mimir.media.ffmpeg import ffmpeg


def extract_wav(source: str | Path, output: str | Path, *, start: float = 0.0, duration: float | None = None,
                rate: int = 16000, channels: int = 1, enhance: bool = False) -> Path:
    """Decode (a range of) the source audio into PCM16 WAV."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    args = ["-v", "error"]
    if start > 0:
        args += ["-ss", f"{start:.6f}"]
    args += ["-i", str(source)]
    if duration is not None:
        args += ["-t", f"{max(0.01, duration):.6f}"]
    filters = []
    if enhance:
        # speech-band emphasis + gentle dynamic normalization for a second, independent ear
        filters.append("highpass=f=90,lowpass=f=7600,dynaudnorm=f=150:g=15")
    if filters:
        args += ["-af", ",".join(filters)]
    args += ["-vn", "-ac", str(channels), "-ar", str(rate), "-c:a", "pcm_s16le", str(output)]
    ffmpeg(args, timeout=3600)
    return output


def extract_mp3(source: str | Path, output: str | Path, *, start: float = 0.0, duration: float | None = None,
                rate: int = 16000, bitrate: str = "64k") -> Path:
    """Compact mono MP3 for upload-size-limited transcription endpoints."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    args = ["-v", "error"]
    if start > 0:
        args += ["-ss", f"{start:.6f}"]
    args += ["-i", str(source)]
    if duration is not None:
        args += ["-t", f"{max(0.01, duration):.6f}"]
    args += ["-vn", "-ac", "1", "-ar", str(rate), "-b:a", bitrate, str(output)]
    ffmpeg(args, timeout=3600)
    return output


def decode_pcm(source: str | Path, *, start: float = 0.0, duration: float | None = None,
               rate: int = 16000) -> np.ndarray:
    """Mono float32 samples in [-1, 1]."""
    args = ["-v", "error"]
    if start > 0:
        args += ["-ss", f"{start:.6f}"]
    args += ["-i", str(source)]
    if duration is not None:
        args += ["-t", f"{max(0.01, duration):.6f}"]
    args += ["-vn", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"]
    result = ffmpeg(args, capture_stdout=True, timeout=3600)
    samples = np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0
    return samples


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    import wave

    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError("expected PCM16 WAV")
        rate = handle.getframerate()
        channels = handle.getnchannels()
        raw = handle.readframes(handle.getnframes())
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def write_wav_mono(path: str | Path, samples: np.ndarray, rate: int) -> Path:
    import wave

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((clipped * 32767.0).astype("<i2").tobytes())
    return path


@dataclass(frozen=True)
class Envelope:
    """Short-term level series: ``times`` are window centers."""

    times: np.ndarray
    dbfs: np.ndarray
    peak_dbfs: np.ndarray
    hop: float

    def level_at(self, t: float) -> float:
        if len(self.times) == 0:
            return -120.0
        index = int(np.clip(round((t - self.times[0]) / self.hop), 0, len(self.times) - 1))
        return float(self.dbfs[index])


def envelope(samples: np.ndarray, rate: int, *, window: float = 0.32, hop: float = 0.10) -> Envelope:
    win = max(1, int(window * rate))
    step = max(1, int(hop * rate))
    if len(samples) < win:
        samples = np.pad(samples, (0, win - len(samples)))
    count = 1 + (len(samples) - win) // step
    index = np.arange(win)[None, :] + step * np.arange(count)[:, None]
    frames = samples[index]
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    peak = np.max(np.abs(frames), axis=1)
    dbfs = 20.0 * np.log10(np.maximum(rms, 1.0 / 32768.0))
    peak_dbfs = 20.0 * np.log10(np.maximum(peak, 1.0 / 32768.0))
    times = (np.arange(count) * step + win / 2.0) / rate
    return Envelope(times, dbfs, peak_dbfs, hop)


def audio_peaks(env: Envelope, *, max_peaks: int = 10, min_separation: float = 0.75,
                min_score: float = 0.30) -> list[dict[str, float]]:
    """Loudness/suddenness peaks (ported scoring: level 62 %, jump 26 %, near-clip 12 %)."""
    if len(env.times) == 0:
        return []
    db = env.dbfs
    floor = float(np.median(db))
    p90 = float(np.percentile(db, 90))
    ceiling = max(p90, float(db.max()) - 3.0, floor + 4.0)
    dynamic = max(3.0, ceiling - floor)
    back = max(1, int(round(0.65 / env.hop)))
    scores = np.zeros_like(db)
    for i in range(len(db)):
        level = min(1.0, max(0.0, (db[i] - floor) / dynamic))
        previous = db[max(0, i - back):i]
        previous_db = float(np.median(previous)) if len(previous) else floor
        jump = min(1.0, max(0.0, (db[i] - previous_db) / 10.0))
        near_clip = min(1.0, max(0.0, (env.peak_dbfs[i] + 6.0) / 6.0))
        scores[i] = 0.62 * level + 0.26 * jump + 0.12 * near_clip
    radius = max(1, int(round(0.40 / env.hop)))
    local = [i for i in range(len(scores))
             if scores[i] >= scores[max(0, i - radius):i + radius + 1].max()]
    local.sort(key=lambda i: (-scores[i], -db[i], env.times[i]))
    chosen: list[dict[str, float]] = []
    for i in local:
        if scores[i] < min_score:
            continue
        t = float(env.times[i])
        if any(abs(t - row["center"]) < min_separation for row in chosen):
            continue
        chosen.append({"center": round(t, 3), "start": round(max(0.0, t - 0.38), 3), "end": round(t + 0.48, 3),
                       "score": round(float(scores[i]), 3), "dbfs": round(float(db[i]), 2),
                       "peak_dbfs": round(float(env.peak_dbfs[i]), 2)})
        if len(chosen) >= max_peaks:
            break
    return chosen


def speech_frames(samples: np.ndarray, rate: int, frame_seconds: float = 0.020) -> dict[str, np.ndarray]:
    """20 ms energy + zero-crossing features (used by the caption clock guard)."""
    size = max(1, int(round(rate * frame_seconds)))
    count = len(samples) // size
    if count == 0:
        return {"start": np.zeros(0), "dbfs": np.zeros(0), "zcr": np.zeros(0)}
    frames = samples[:count * size].reshape(count, size)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    dbfs = 20.0 * np.log10(np.maximum(rms, 1.0 / 32768.0))
    signs = np.signbit(frames)
    zcr = np.count_nonzero(signs[:, 1:] != signs[:, :-1], axis=1) / max(1, size - 1)
    return {"start": np.arange(count) * frame_seconds, "dbfs": dbfs, "zcr": zcr.astype(np.float64)}


def cross_correlation_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag (in samples of the given series) maximizing normalized correlation of b against a."""
    if len(a) == 0 or len(b) == 0:
        return 0, 0.0
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    best_lag, best = 0, -math.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[lag:], b[:len(b) - lag] if lag else b
        else:
            x, y = a[:lag], b[-lag:]
        n = min(len(x), len(y))
        if n < 8:
            continue
        value = float(np.dot(x[:n], y[:n]) / n)
        if value > best:
            best, best_lag = value, lag
    return best_lag, float(best if best != -math.inf else 0.0)
