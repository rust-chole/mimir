"""Real synthesized speech (espeak-ng) with exact word timing for synthetic VODs."""
from __future__ import annotations

import hashlib
import re
import subprocess
import wave
from pathlib import Path

import numpy as np

RATE = 48000
WORD_GAP = 0.07
CACHE = Path(__file__).resolve().parent / ".speech_cache"


def _read(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    return data, rate


def synth_word(word: str, voice: str, speed: int, pitch: int) -> np.ndarray:
    spoken = re.sub(r"[^\w'\- ]+", "", word) or word
    key = hashlib.sha1(f"{spoken}|{voice}|{speed}|{pitch}".encode()).hexdigest()[:16]
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / f"{key}.npy"
    if cached.exists():
        return np.load(cached)
    raw = CACHE / f"{key}.wav"
    subprocess.run(["espeak-ng", "-v", voice, "-s", str(speed), "-p", str(pitch), "-w", str(raw), spoken],
                   check=True, capture_output=True)
    data, rate = _read(raw)
    raw.unlink()
    target = np.interp(np.arange(0, len(data) * RATE / rate) * rate / RATE, np.arange(len(data)), data)
    loud = np.nonzero(np.abs(target) > 0.02)[0]
    if len(loud):
        a, b = max(0, loud[0] - int(0.01 * RATE)), min(len(target), loud[-1] + int(0.015 * RATE))
        target = target[a:b]
    target = (target / (np.abs(target).max() + 1e-6) * 0.55).astype(np.float32)
    np.save(cached, target)
    return target


def speak_line(text: str, voice: str, speed: int, pitch: int, start: float
               ) -> tuple[list[tuple[str, float, float, np.ndarray]], float]:
    """Words of a line placed sequentially: [(display word, start, end, samples)], line end."""
    cursor = start
    rows = []
    for token in text.split():
        samples = synth_word(token, voice, speed, pitch)
        duration = len(samples) / RATE
        rows.append((token, round(cursor, 3), round(cursor + duration, 3), samples))
        cursor += duration + WORD_GAP
    return rows, cursor - WORD_GAP


def mix_into(track: np.ndarray, samples: np.ndarray, start: float, gain: float = 1.0) -> None:
    index = int(round(start * RATE))
    end = min(len(track), index + len(samples))
    if end > index:
        track[index:end] += samples[:end - index] * gain


def noise_burst(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(seconds * RATE)).astype(np.float32)


def impact(seed: int = 1) -> np.ndarray:
    n = int(0.7 * RATE)
    t = np.arange(n) / RATE
    boom = np.sin(2 * np.pi * (70 - 25 * t) * t) * np.exp(-t * 6)
    crack = noise_burst(0.7, seed) * np.exp(-t * 30)
    return (0.9 * boom + 0.6 * crack).astype(np.float32)


def cheer(seconds: float = 1.6, seed: int = 2) -> np.ndarray:
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    envelope = np.minimum(1.0, t / 0.25) * np.minimum(1.0, (seconds - t) / 0.5)
    noise = noise_burst(seconds, seed)
    kernel = np.ones(24) / 24
    band = np.convolve(noise, kernel, mode="same") - np.convolve(noise, np.ones(200) / 200, mode="same")
    return (band * envelope * 2.2).astype(np.float32)


def scream(seconds: float = 0.9) -> np.ndarray:
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    freq = 520 + 60 * np.sin(2 * np.pi * 7 * t)
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    wave_ = np.sign(np.sin(phase)) * 0.4 + np.sin(2 * phase) * 0.3
    envelope = np.minimum(1.0, t / 0.05) * np.minimum(1.0, (seconds - t) / 0.2)
    return (wave_ * envelope * 0.9).astype(np.float32)


def whoosh(seconds: float = 0.45, seed: int = 3) -> np.ndarray:
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    envelope = np.sin(np.pi * t / seconds) ** 2
    noise = noise_burst(seconds, seed)
    return (np.convolve(noise, np.ones(8) / 8, mode="same") * envelope * 0.5).astype(np.float32)


def write_wav(path: Path, samples: np.ndarray, rate: int = RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((clipped * 32767).astype("<i2").tobytes())
    return path
