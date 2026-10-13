"""Acoustic coverage of the timed caption words: do they cover the speech, and only speech?

Forced alignment times whatever words it is given. It cannot tell that a phrase is
missing (a neighbouring word simply stretches over the speech nobody transcribed)
or that a word was never said (it lands in a pause), and the structural validation
in ``alignment`` passes both. This check compares the timed words with an
independent voice-activity map (Silero VAD) of the same analysis audio:

    uncovered_speech   speech for >= MIN_UNCOVERED_S where no caption word is
    stretched_words    a word whose interval holds far more speech than the word needs
                       (the aligner spread it over speech the ears did not transcribe)
    unsupported_words  a word whose interval holds (almost) no speech

REPORT ONLY: nothing here changes a word or a time. The flags go to the caption
quality report, the caption truth document and the human review sheet; the caption
evaluation (``ai.caption_eval``) measures how often they point at real errors
before any flag may ever trigger an automatic action.

VAD is loaded lazily (``silero-vad``, weights ship inside the pip package; it needs
torch, which the forced aligner already requires). Missing package -> the check
reports ``unavailable`` and the caption run continues.
"""
from __future__ import annotations

import threading
import warnings
from typing import Any, Callable, Mapping, Sequence

from ai.caption_stack import audio as analysis_audio

COVERAGE_VERSION = 1
VAD_THRESHOLD = 0.5
VAD_MIN_SPEECH_MS = 250
VAD_MIN_SILENCE_MS = 150
VAD_PAD_MS = 60
WORD_PAD_S = 0.12               # aligner boundaries are ~40-80 ms off; never flag an edge
MIN_UNCOVERED_S = 0.8           # speech this long with no caption word is worth a human look
SECONDS_PER_CHAR = 0.09         # generous spoken length of a word (fast speech is ~0.06 s/char)
MIN_WORD_NEED_S = 0.30
STRETCH_EXCESS_S = 1.0          # speech inside one word beyond what the word itself needs
MIN_UNSUPPORTED_WORD_S = 0.15
MAX_WORD_SPEECH_RATIO = 0.10    # a word with less speech than this under it sits in a pause
MAX_ROWS = 20

Regions = list[tuple[float, float]]


class CoverageUnavailable(RuntimeError):
    """No voice-activity detector could be loaded."""


# ============================================================
# VOICE ACTIVITY (Silero VAD, lazily loaded once per process)
# ============================================================

_vad_lock = threading.Lock()
_vad: Any = None


def _load_vad() -> tuple[Any, Any, Any]:
    global _vad
    with _vad_lock:
        if isinstance(_vad, BaseException):
            raise _vad
        if _vad is None:
            try:
                import torch

                threads = torch.get_num_threads()
                import silero_vad                          # its import sets torch to ONE thread process-wide
                torch.set_num_threads(threads)             # ... which would slow the CPU aligner down
                with warnings.catch_warnings():            # torch.jit.load deprecation notice (harmless)
                    warnings.simplefilter("ignore", FutureWarning)
                    _vad = (torch, silero_vad, silero_vad.load_silero_vad())
            except Exception as error:
                _vad = CoverageUnavailable(f"voice-activity detector unavailable (pip install silero-vad): "
                                           f"{type(error).__name__}: {str(error)[:200]}")
                raise _vad from None
        return _vad


def reset_vad() -> None:
    global _vad
    with _vad_lock:
        _vad = None


def speech_regions(audio: analysis_audio.AnalysisAudio) -> Regions:
    """[start, end) seconds of detected speech on the analysis clock."""
    torch, silero_vad, model = _load_vad()
    samples = analysis_audio.read_samples(audio)
    tensor = torch.frombuffer(bytearray(samples.tobytes()), dtype=torch.int16).to(torch.float32) / 32768.0
    with _vad_lock:                                         # the model keeps recurrent state
        stamps = silero_vad.get_speech_timestamps(
            tensor, model, threshold=VAD_THRESHOLD, sampling_rate=audio.sample_rate,
            min_speech_duration_ms=VAD_MIN_SPEECH_MS, min_silence_duration_ms=VAD_MIN_SILENCE_MS,
            speech_pad_ms=VAD_PAD_MS, return_seconds=True)
    return [(float(row["start"]), float(row["end"])) for row in stamps]


# ============================================================
# ASSESSMENT (pure)
# ============================================================

def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _speech_in(start: float, end: float, regions: Regions) -> float:
    return sum(_overlap(start, end, a, b) for a, b in regions)


def _subtract(region: tuple[float, float], covers: Sequence[tuple[float, float]]) -> Regions:
    pieces = [region]
    for c0, c1 in covers:
        next_pieces: Regions = []
        for p0, p1 in pieces:
            if c1 <= p0 or c0 >= p1:
                next_pieces.append((p0, p1))
                continue
            if c0 > p0:
                next_pieces.append((p0, c0))
            if c1 < p1:
                next_pieces.append((c1, p1))
        pieces = next_pieces
    return pieces


def _chars(text: str) -> int:
    return sum(1 for ch in str(text) if ch.isalnum())


def assess(words: Sequence[Mapping[str, Any]], regions: Regions, duration: float) -> dict[str, Any]:
    """Coverage flags for timed words (``word``, ``edited_start``, ``edited_end``, ``token_ids``)."""
    rows = []
    for index, word in enumerate(words):
        try:
            start, end = float(word["edited_start"]), float(word["edited_end"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append((index, start, end, str(word.get("word", "")), word.get("token_ids")))
    rows.sort(key=lambda row: row[1])
    covers = [(max(0.0, s - WORD_PAD_S), e + WORD_PAD_S) for _i, s, e, _w, _t in rows]

    uncovered: list[dict[str, Any]] = []
    for region in regions:
        for p0, p1 in _subtract(region, covers):
            if p1 - p0 < MIN_UNCOVERED_S:
                continue
            before = [r for r in rows if r[2] <= p0 + 1e-6]
            after = [r for r in rows if r[1] >= p1 - 1e-6]
            uncovered.append({"start": round(p0, 3), "end": round(p1, 3), "seconds": round(p1 - p0, 3),
                              "after_word": before[-1][3] if before else "", "before_word": after[0][3] if after else ""})

    stretched: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    for index, start, end, text, token_ids in rows:
        speech = _speech_in(start, end, regions)
        need = max(MIN_WORD_NEED_S, _chars(text) * SECONDS_PER_CHAR)
        row = {"index": index, "word": text, "start": round(start, 3), "end": round(end, 3),
               "speech_s": round(speech, 3), "token_ids": token_ids}
        if speech - need >= STRETCH_EXCESS_S:
            stretched.append({**row, "expected_s": round(need, 3)})
        span = end - start
        padded = _speech_in(start - WORD_PAD_S, end + WORD_PAD_S, regions)
        if span >= MIN_UNSUPPORTED_WORD_S and padded < MAX_WORD_SPEECH_RATIO * span:
            unsupported.append(row)

    speech_total = sum(b - a for a, b in regions)
    return {
        "status": "ok", "version": COVERAGE_VERSION, "detector": "silero_vad",
        "speech_s": round(speech_total, 3), "duration_s": round(float(duration), 3), "regions": len(regions),
        "uncovered_speech": uncovered[:MAX_ROWS], "stretched_words": stretched[:MAX_ROWS],
        "unsupported_words": unsupported[:MAX_ROWS],
        "flags": len(uncovered) + len(stretched) + len(unsupported),
        "policy": "report only: no word or time is changed by a coverage flag",
    }


def check(audio: analysis_audio.AnalysisAudio, words: Sequence[Mapping[str, Any]], *,
          detector: Callable[[analysis_audio.AnalysisAudio], Regions] | None = None) -> dict[str, Any]:
    """Coverage report for timed words on ``audio``; never raises (unavailable/failed is a status)."""
    try:
        regions = (detector or speech_regions)(audio)
    except CoverageUnavailable as error:
        return {"status": "unavailable", "version": COVERAGE_VERSION, "reason": str(error)[:300], "flags": 0}
    except Exception as error:  # a detector failure never blocks captions
        return {"status": "failed", "version": COVERAGE_VERSION, "reason": f"{type(error).__name__}: {str(error)[:240]}",
                "flags": 0}
    return assess(words, regions, audio.duration)
