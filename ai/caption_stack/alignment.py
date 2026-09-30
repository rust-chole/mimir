"""WHEN each frozen word was spoken: one word-alignment provider per successful run.

    FrozenTranscript + final-short analysis audio
        -> provider.align (Qwen3-ForcedAligner by default; lazily loaded, reused)
        -> validate (parity, order, positive duration, monotonic, range, coverage,
                     overlap, pathological collapse)
        -> LOCAL recovery of a small failed region (re-align it with its trusted
           neighbours on the matching local audio; accepted only when the
           neighbours re-align where they already were)
        -> still invalid -> the configured fallback provider re-times the WHOLE
           transcript (never a mix of two clocks) -> validated the same way
        -> nothing valid -> AlignmentError (the caption stage fails closed)

No global shift, no invented timing, no borrowing across words: a timing that
fails validation is replaced only by another validated alignment of the same
frozen words. Downstream code sees ``AlignedUnit`` rows and nothing else.
"""
from __future__ import annotations

import math
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from ai.caption_stack import audio as analysis_audio
from ai.caption_stack.config import (
    ALIGNER_MAX_INPUT_SECONDS,
    CaptionStackSettings,
    aligner_language,
    resolve_device,
    resolve_dtype,
)

ALIGNMENT_VERSION = 1
MIN_WORD_SECONDS = 0.001        # end > start at millisecond resolution
MAX_WORD_SECONDS = 5.0          # one spoken word held longer than this is not a plausible alignment
RANGE_TOLERANCE_S = 0.05
OVERLAP_TOLERANCE_S = 0.010
COLLAPSE_RUN = 3                # >= this many consecutive words squeezed into COLLAPSE_SECONDS_PER_WORD each
COLLAPSE_SECONDS_PER_WORD = 0.02
RECOVERY_NEIGHBORS = 4
RECOVERY_PAD_S = 0.25
ANCHOR_TOLERANCE_S = 0.20
MAX_RECOVERY_REGIONS = 6


class AlignmentError(RuntimeError):
    """No valid word clock could be produced by a provider."""


@dataclass(frozen=True)
class AlignedUnit:
    """Timed frozen tokens ``[token_start, token_end)`` on the analysis clock (seconds)."""
    token_start: int
    token_end: int
    start: float
    end: float
    source: str

    def shifted(self, tokens: int) -> "AlignedUnit":
        return AlignedUnit(self.token_start + tokens, self.token_end + tokens, self.start, self.end, self.source)


class WordAlignmentProvider(Protocol):
    name: str
    max_input_seconds: float

    def describe(self) -> dict[str, Any]: ...

    def align(self, audio: analysis_audio.AnalysisAudio, window: tuple[float, float], tokens: Sequence[str],
              language: str, workdir: Path) -> list[AlignedUnit]: ...


# ============================================================
# VALIDATION
# ============================================================

@dataclass
class Validation:
    ok: bool
    fatal: bool = False
    bad: set[int] = field(default_factory=set)          # unit indices
    issues: list[str] = field(default_factory=list)


def validate_units(units: Sequence[AlignedUnit], token_count: int, window: tuple[float, float],
                   audio_duration: float) -> Validation:
    """Every check that must hold before a timing becomes caption truth."""
    issues: list[str] = []
    bad: set[int] = set()
    cursor = 0
    for unit in units:                                   # coverage + order + lexical parity (token ids)
        if unit.token_start != cursor or unit.token_end <= unit.token_start:
            return Validation(False, True, set(range(len(units))),
                              [f"units do not cover the frozen words in order at token {cursor}"])
        cursor = unit.token_end
    if cursor != token_count:
        return Validation(False, True, set(range(len(units))),
                          [f"{token_count - cursor} frozen word(s) have no timing"])
    lo, hi = float(window[0]), float(window[1])
    limit = min(hi, float(audio_duration)) + RANGE_TOLERANCE_S
    for index, unit in enumerate(units):
        if not (math.isfinite(unit.start) and math.isfinite(unit.end)):
            bad.add(index); issues.append(f"unit {index}: non-finite time")
            continue
        if unit.end - unit.start < MIN_WORD_SECONDS:
            bad.add(index); issues.append(f"unit {index}: non-positive duration {unit.start:.3f}-{unit.end:.3f}")
        if unit.start < max(0.0, lo) - RANGE_TOLERANCE_S or unit.end > limit:
            bad.add(index); issues.append(f"unit {index}: outside the audio window ({unit.start:.3f}-{unit.end:.3f})")
        words = max(1, unit.token_end - unit.token_start)
        if unit.end - unit.start > MAX_WORD_SECONDS * words:
            bad.add(index); issues.append(f"unit {index}: implausible duration {unit.end - unit.start:.2f}s")
        if index:
            previous = units[index - 1]
            if unit.start <= previous.start:
                bad.update({index - 1, index}); issues.append(f"unit {index}: start does not advance")
            elif unit.start < previous.end - OVERLAP_TOLERANCE_S:
                bad.update({index - 1, index}); issues.append(f"unit {index}: overlaps the previous word")
    at_window_start = [i for i, u in enumerate(units) if abs(u.start - lo) < 1e-3]
    if len(at_window_start) > 1:
        bad.update(at_window_start); issues.append(f"{len(at_window_start)} words collapsed onto the window start")
    for index in range(0, max(0, len(units) - COLLAPSE_RUN + 1)):
        run = units[index:index + COLLAPSE_RUN]
        if run[-1].end - run[0].start < COLLAPSE_SECONDS_PER_WORD * COLLAPSE_RUN:
            bad.update(range(index, index + COLLAPSE_RUN)); issues.append(f"units {index}+: collapsed region")
    return Validation(not bad, False, bad, issues)


# ============================================================
# QWEN3 FORCED ALIGNER (official qwen_asr implementation)
# ============================================================

def _kept(ch: str) -> bool:
    """Characters the official aligner keeps (Qwen3ForceAlignProcessor.is_kept_char)."""
    return ch == "'" or unicodedata.category(ch)[:1] in ("L", "N")


def aligner_clean(token: str) -> str:
    return "".join(ch for ch in str(token) if _kept(ch))


def map_items_to_tokens(tokens: Sequence[str], items: Sequence[Any], offset: float, source: str
                        ) -> list[AlignedUnit]:
    """Official aligner items -> one unit per frozen token, by exact character parity.

    The aligner re-tokenizes (drops punctuation, splits CJK characters); every
    item's text must consume the next characters of the frozen tokens exactly,
    in order, and every frozen character must be consumed. Any other shape is
    a lexical parity failure, never a partial clock.
    """
    stream = [aligner_clean(t).casefold() for t in tokens]
    if any(not s for s in stream):
        raise AlignmentError("a frozen word has no characters the aligner can time")
    spans: list[list[float]] = [[] for _ in tokens]
    token, char = 0, 0
    for item in items:
        text = aligner_clean(str(getattr(item, "text", ""))).casefold()
        start = float(getattr(item, "start_time"))
        end = float(getattr(item, "end_time"))
        if not text:
            raise AlignmentError("aligner returned an empty item")
        for ch in text:
            while token < len(stream) and char >= len(stream[token]):
                token, char = token + 1, 0
            if token >= len(stream) or stream[token][char] != ch:
                raise AlignmentError(f"lexical parity broken at frozen word {token}: aligner item {text!r}")
            spans[token].extend((start, end))
            char += 1
    while token < len(stream) and char >= len(stream[token]):
        token, char = token + 1, 0
    if token < len(stream):
        raise AlignmentError(f"aligner output stops before frozen word {token}")
    return [AlignedUnit(i, i + 1, round(offset + min(s), 3), round(offset + max(s), 3), source)
            for i, s in enumerate(spans)]


@dataclass
class LoadedAligner:
    model: Any
    device: str
    dtype: str
    max_input_seconds: float
    notes: list[str]


Loader = Callable[[CaptionStackSettings], LoadedAligner]


def load_official_aligner(settings: CaptionStackSettings) -> LoadedAligner:
    """Import and initialize ``qwen_asr.Qwen3ForcedAligner`` (only when timing is needed)."""
    try:
        import torch
        from qwen_asr import Qwen3ForcedAligner
    except ImportError as error:
        raise AlignmentError("the official Qwen forced aligner is not installed (pip install -r requirements.txt): "
                             f"{error}") from None
    device, device_note = resolve_device(settings.aligner_device, torch)
    dtype, dtype_name = resolve_dtype(settings.aligner_dtype, device, torch)
    model = Qwen3ForcedAligner.from_pretrained(settings.aligner_model, dtype=dtype, device_map=device)
    limit = ALIGNER_MAX_INPUT_SECONDS
    try:
        from qwen_asr.inference.utils import MAX_FORCE_ALIGN_INPUT_SECONDS

        limit = min(limit, float(MAX_FORCE_ALIGN_INPUT_SECONDS))
    except (ImportError, TypeError, ValueError):
        pass
    return LoadedAligner(model, device, dtype_name, limit, [n for n in (device_note,) if n])


_aligner_lock = threading.Lock()
_aligners: dict[tuple[str, str, str], LoadedAligner | BaseException] = {}


def shared_aligner(settings: CaptionStackSettings, loader: Loader = load_official_aligner) -> LoadedAligner:
    """Load once per process and reuse; a load failure is remembered, not retried per call."""
    key = (settings.aligner_model, settings.aligner_device, settings.aligner_dtype)
    with _aligner_lock:
        cached = _aligners.get(key)
        if cached is None:
            try:
                cached = loader(settings)
            except Exception as error:
                cached = error if isinstance(error, AlignmentError) else AlignmentError(
                    f"Qwen forced aligner could not be loaded: {type(error).__name__}: {str(error)[:300]}")
            _aligners[key] = cached
    if isinstance(cached, BaseException):
        raise cached
    return cached


def reset_shared_aligners() -> None:
    with _aligner_lock:
        _aligners.clear()


class QwenForcedAlignmentProvider:
    """Forced alignment ONLY: the frozen words go in, their times come out."""

    name = "qwen3_forced_aligner"

    def __init__(self, settings: CaptionStackSettings, loader: Loader = load_official_aligner) -> None:
        self.settings = settings
        self._loader = loader
        self._loaded: LoadedAligner | None = None

    @property
    def max_input_seconds(self) -> float:
        return self._loaded.max_input_seconds if self._loaded else ALIGNER_MAX_INPUT_SECONDS

    def describe(self) -> dict[str, Any]:
        loaded = self._loaded
        return {"provider": self.name, "model": self.settings.aligner_model, "role": "primary",
                "device": loaded.device if loaded else "not_loaded", "dtype": loaded.dtype if loaded else "not_loaded",
                "notes": list(loaded.notes) if loaded else []}

    def align(self, audio: analysis_audio.AnalysisAudio, window: tuple[float, float], tokens: Sequence[str],
              language: str, workdir: Path) -> list[AlignedUnit]:
        start, end = float(window[0]), float(window[1])
        if not tokens:
            return []
        self._loaded = shared_aligner(self.settings, self._loader)
        if end - start > self._loaded.max_input_seconds + 1e-6:
            raise AlignmentError(f"window {end - start:.1f}s exceeds the aligner limit "
                                 f"{self._loaded.max_input_seconds:.0f}s")
        if start <= 1e-6 and end >= audio.duration - 1e-6:
            path, offset = audio.path, 0.0
        else:
            path, offset, _end = analysis_audio.cut_window(audio, start, end, Path(workdir) / (
                f"align_{int(start * 1000):08d}_{int(end * 1000):08d}.wav"))
        results = self._loaded.model.align(audio=str(path), text=" ".join(tokens), language=aligner_language(language))
        if not results:
            raise AlignmentError("aligner returned no result")
        items = getattr(results[0], "items", results[0])
        return map_items_to_tokens(tokens, list(items), offset, f"{self.name}:{self.settings.aligner_model}")


def build_providers(settings: CaptionStackSettings, *, qwen_loader: Loader = load_official_aligner,
                    whisper_transcribe: Any = None) -> tuple[WordAlignmentProvider, WordAlignmentProvider | None]:
    from ai.caption_stack.legacy_whisper import LegacyWhisperAlignmentFallback

    def make(name: str) -> WordAlignmentProvider:
        if name == "qwen3_forced_aligner":
            return QwenForcedAlignmentProvider(settings, qwen_loader)
        return LegacyWhisperAlignmentFallback(whisper_transcribe)

    primary = make(settings.alignment_provider)
    fallback = None if settings.alignment_fallback_provider == "none" else make(settings.alignment_fallback_provider)
    return primary, fallback


# ============================================================
# LOCAL RECOVERY
# ============================================================

def _runs(indices: set[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for index in sorted(indices):
        if runs and index == runs[-1][1]:
            runs[-1] = (runs[-1][0], index + 1)
        else:
            runs.append((index, index + 1))
    return runs


def _token_times(units: Sequence[AlignedUnit]) -> dict[int, tuple[float, float]]:
    return {t: (u.start, u.end) for u in units for t in range(u.token_start, u.token_end)}


def recover_locally(provider: WordAlignmentProvider, audio: analysis_audio.AnalysisAudio,
                    tokens: Sequence[str], units: list[AlignedUnit], check: Validation,
                    chunk_window: tuple[float, float], language: str, workdir: Path
                    ) -> tuple[list[AlignedUnit], list[dict[str, Any]]]:
    """Re-align only the failed regions (+ trusted neighbours) on their local audio."""
    runs = _runs(check.bad)
    if not runs or len(runs) > MAX_RECOVERY_REGIONS or len(check.bad) == len(units):
        return units, [{"status": "skipped", "reason": "no localized failure" if runs else "nothing to recover",
                        "regions": len(runs)}]
    regions: list[list[int]] = []
    for r0, r1 in runs:
        lo, hi = max(0, r0 - RECOVERY_NEIGHBORS), min(len(units), r1 + RECOVERY_NEIGHBORS)
        if regions and lo <= regions[-1][1]:
            regions[-1][1] = max(regions[-1][1], hi)
            regions[-1][2].append((r0, r1))
        else:
            regions.append([lo, hi, [(r0, r1)]])
    output = list(units)
    report: list[dict[str, Any]] = []
    for lo, hi, bad_runs in reversed(regions):
        bad_units = {i for a, b in bad_runs for i in range(a, b)}
        anchors = [i for i in range(lo, hi) if i not in bad_units]
        record: dict[str, Any] = {"units": [lo, hi], "failed_units": sorted(bad_units), "anchors": len(anchors)}
        if not anchors:
            report.append({**record, "status": "failed", "reason": "no trusted neighbour to anchor the region"})
            continue
        start = units[lo].start - RECOVERY_PAD_S if lo not in bad_units else (
            units[lo - 1].end if lo > 0 else chunk_window[0])
        end = units[hi - 1].end + RECOVERY_PAD_S if (hi - 1) not in bad_units else (
            units[hi].start if hi < len(units) else chunk_window[1])
        if lo > 0:
            start = max(start, units[lo - 1].end)
        if hi < len(units):
            end = min(end, units[hi].start)
        start, end = max(chunk_window[0], start), min(chunk_window[1], end)
        t0, t1 = units[lo].token_start, units[hi - 1].token_end
        record["window"] = [round(start, 3), round(end, 3)]
        try:
            local = [u.shifted(t0) for u in provider.align(audio, (start, end), tokens[t0:t1], language, workdir)]
        except Exception as error:
            report.append({**record, "status": "failed", "reason": f"{type(error).__name__}: {str(error)[:200]}"})
            continue
        local_check = validate_units([u.shifted(-t0) for u in local], t1 - t0, (start, end), audio.duration)
        if not local_check.ok:
            report.append({**record, "status": "failed", "reason": "; ".join(local_check.issues[:3])})
            continue
        global_times, local_times = _token_times(units), _token_times(local)
        drift = max(abs(global_times[t][0] - local_times[t][0])
                    for i in anchors for t in range(units[i].token_start, units[i].token_end))
        if drift > ANCHOR_TOLERANCE_S:
            report.append({**record, "status": "failed", "reason": f"trusted neighbours moved {drift:.3f}s"})
            continue
        # Replace ONLY the failed runs; trusted words keep their original timing.
        swaps: list[tuple[int, int, list[AlignedUnit]]] = []
        for a, b in bad_runs:
            r_t0, r_t1 = units[a].token_start, units[b - 1].token_end
            replace = [u for u in local if r_t0 <= u.token_start and u.token_end <= r_t1]
            if not replace or replace[0].token_start != r_t0 or replace[-1].token_end != r_t1:
                swaps = []
                break
            swaps.append((r_t0, r_t1, replace))
        if not swaps:
            report.append({**record, "status": "failed", "reason": "local units do not tile the failed words"})
            continue
        for r_t0, r_t1, replace in reversed(swaps):
            first = next(i for i, u in enumerate(output) if u.token_start == r_t0)
            last = next(i for i, u in enumerate(output) if u.token_end == r_t1)
            output[first:last + 1] = replace
        report.append({**record, "status": "recovered", "anchor_drift_s": round(drift, 4)})
    return output, report


# ============================================================
# ONE TIMING AUTHORITY PER RUN
# ============================================================

class AlignmentCache:
    """Raw provider output per (provider, chunk window, exact tokens); the locator fills it."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, tuple[float, float], tuple[str, ...]], list[AlignedUnit]] = {}

    def get(self, name: str, window: tuple[float, float], tokens: Sequence[str]) -> list[AlignedUnit] | None:
        found = self._rows.get((name, tuple(window), tuple(tokens)))
        return list(found) if found is not None else None

    def put(self, name: str, window: tuple[float, float], tokens: Sequence[str], units: Sequence[AlignedUnit]) -> None:
        self._rows[(name, tuple(window), tuple(tokens))] = list(units)


@dataclass
class AlignmentOutcome:
    units: list[AlignedUnit]
    provider: str
    describe: dict[str, Any]
    attempts: list[dict[str, Any]]
    recoveries: list[dict[str, Any]]
    reused_locator: bool

    @property
    def degraded(self) -> bool:
        return any(a.get("status") == "failed" for a in self.attempts)


def _align_with(provider: WordAlignmentProvider, frozen: Any, audio: analysis_audio.AnalysisAudio, language: str,
                workdir: Path, cache: AlignmentCache) -> tuple[list[AlignedUnit], list[dict[str, Any]], bool]:
    everything: list[AlignedUnit] = []
    recoveries: list[dict[str, Any]] = []
    reused = False
    for chunk, window in enumerate(frozen.chunks):
        offset, tokens = frozen.chunk_tokens(chunk)
        if not tokens:
            continue
        units = cache.get(provider.name, window, tokens)
        if units is None:
            units = provider.align(audio, window, tokens, language, workdir)
            cache.put(provider.name, window, tokens, units)
        else:
            reused = True
        check = validate_units(units, len(tokens), window, audio.duration)
        if not check.ok and not check.fatal:
            units, report = recover_locally(provider, audio, tokens, list(units), check, window, language, workdir)
            recoveries.extend({"chunk": chunk, **row} for row in report)
            check = validate_units(units, len(tokens), window, audio.duration)
        if not check.ok:
            raise AlignmentError(f"chunk {chunk}: " + "; ".join(check.issues[:4]))
        everything.extend(u.shifted(offset) for u in units)
    whole = validate_units(everything, len(frozen.tokens), (0.0, audio.duration), audio.duration)
    if not whole.ok:
        raise AlignmentError("merged clock: " + "; ".join(whole.issues[:4]))
    return everything, recoveries, reused


def align_frozen(frozen: Any, audio: analysis_audio.AnalysisAudio, *, primary: WordAlignmentProvider,
                 fallback: WordAlignmentProvider | None, language: str, workdir: Path,
                 cache: AlignmentCache | None = None) -> AlignmentOutcome:
    """Time the frozen words with the primary provider, else (whole transcript) the fallback."""
    cache = cache or AlignmentCache()
    attempts: list[dict[str, Any]] = []
    for provider in [p for p in (primary, fallback) if p is not None]:
        try:
            units, recoveries, reused = _align_with(provider, frozen, audio, language, workdir, cache)
        except Exception as error:
            attempts.append({"provider": provider.name, "status": "failed",
                             "reason": f"{type(error).__name__}: {str(error)[:300]}"})
            continue
        attempts.append({"provider": provider.name, "status": "ok"})
        return AlignmentOutcome(units, provider.name, provider.describe(), attempts, recoveries, reused)
    raise AlignmentError("no word-alignment provider produced a valid clock for the frozen transcript: "
                         + " | ".join(f"{a['provider']}: {a['reason']}" for a in attempts))


def locator(provider: WordAlignmentProvider, audio: analysis_audio.AnalysisAudio,
            chunks: Sequence[tuple[float, float]], language: str, workdir: Path, cache: AlignmentCache
            ) -> Callable[[int, Sequence[str]], list[tuple[float, float] | None]]:
    """Evidence windows for lexical disputes: the provider aligns the PRIMARY ear's words.

    Evidence only. When the frozen words equal them, the final alignment reuses
    this exact provider output (same provider, audio and words) and validates it.
    """
    def locate(chunk: int, tokens: Sequence[str]) -> list[tuple[float, float] | None]:
        window = tuple(chunks[chunk])
        units = cache.get(provider.name, window, tokens)
        if units is None:
            units = provider.align(audio, window, tokens, language, workdir)
            cache.put(provider.name, window, tokens, units)
        check = validate_units(units, len(tokens), window, audio.duration)
        times: list[tuple[float, float] | None] = [None] * len(tokens)
        if check.fatal:
            return times
        for index, unit in enumerate(units):
            if index not in check.bad:
                for token in range(unit.token_start, unit.token_end):
                    times[token] = (unit.start, unit.end)
        return times

    return locate
