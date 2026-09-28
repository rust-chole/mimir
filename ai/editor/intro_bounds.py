"""Evidence-derived cold-open bounds.

The cold open IS the peak: real moving footage with its real audio. How long
it runs must come from the event itself, never from a score bucket, a creator,
a phrase or a fixed "reaction = N seconds" rule. This module turns evidence on
the PACED clip clock into one [start, end] window:

    event core        the selected peak (audio/visual candidate) and/or the
                      spoken phrase the editor chose (AI chooses WHAT)
    acoustic onset    where the sound of the event rises out of the local floor
    acoustic decay    where the reaction/impact falls back to the floor
    phrases           speech is never cut mid-word or mid-phrase: a phrase that
                      carries the event is kept whole, a phrase that merely
                      touches the edge is either kept whole or excluded at the
                      pause before it
    shot boundaries   a window never starts or ends with a sliver of a shot
    handles           a little of the silence that really exists around the
                      event (never borrowed from a neighbouring word)

Only generic safety bounds exist (a perceptual floor and a hard ceiling).
Everything here is deterministic and local (FFmpeg decode, no model call).
"""
from __future__ import annotations

import math
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

INTRO_BOUNDS_VERSION = 1

ENVELOPE_SAMPLE_RATE = 16000
ENVELOPE_FRAME_S = 0.02

# Generic safety bounds (not editorial targets).
MIN_INTRO_S = 1.0            # below this a cold open is not perceivable as a shot
MAX_INTRO_S = 6.5            # hard ceiling; a longer "intro" is a second story

# Acoustic event search.
NOISE_CONTEXT_S = 8.0        # local floor is measured around the event, not over the whole clip
MAX_ONSET_SEARCH_S = 2.0     # how far before the core a sound may have started
MAX_DECAY_SEARCH_S = 3.0     # how far after the core a reaction may keep sounding
ACTIVITY_BRIDGE_S = 0.14     # dips shorter than this belong to the same sound
MIN_EVENT_RISE_DB = 4.0      # the event must stand this far above the local floor to be measurable

# Speech.
PHRASE_PAUSE_S = 0.30        # a pause this long separates phrases
SENTENCE_PAUSE_S = 0.12      # after . ! ? a shorter pause already ends the phrase
EDGE_PHRASE_KEEP_SHARE = 0.5 # an edge phrase is kept whole when this share of it lies in the window

# Handles / shots.
MAX_LEAD_HANDLE_S = 0.15
MAX_TAIL_HANDLE_S = 0.30
SHOT_SLIVER_S = 0.45         # a shot shorter than this at a window edge is a flash frame
SCENE_THRESHOLD = 0.30


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Phrase:
    start: float
    end: float
    first: int
    last: int
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Envelope:
    """Frame levels (dBFS) of the paced clip, ``frame_s`` apart, starting at 0."""

    db: tuple[float, ...]
    frame_s: float = ENVELOPE_FRAME_S

    def index(self, t: float) -> int:
        return max(0, min(len(self.db) - 1, int(math.floor(t / self.frame_s)))) if self.db else 0

    def time(self, index: int) -> float:
        return index * self.frame_s

    def window(self, start: float, end: float) -> list[float]:
        if not self.db:
            return []
        a, b = self.index(max(0.0, start)), self.index(max(0.0, end))
        return list(self.db[a:b + 1])


@dataclass(frozen=True)
class EventEvidence:
    core_start: float
    core_end: float
    lead_in_start: float | None = None   # the editor's chosen spoken lead-in, when it starts before the core
    source: str = "peak"


@dataclass
class IntroBounds:
    start: float
    end: float
    core_start: float
    core_end: float
    onset: float | None = None
    completion: float | None = None
    phrases: list[dict[str, Any]] = field(default_factory=list)
    shot_snaps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": INTRO_BOUNDS_VERSION,
            "policy": "event_evidence",
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "core": [round(self.core_start, 3), round(self.core_end, 3)],
            "acoustic_onset": None if self.onset is None else round(self.onset, 3),
            "acoustic_completion": None if self.completion is None else round(self.completion, 3),
            "phrases": self.phrases,
            "shot_snaps": self.shot_snaps,
            "notes": self.notes,
        }


# ============================================================
# EVIDENCE EXTRACTION
# ============================================================

def audio_envelope(media_path: str | Path | None, *, frame_s: float = ENVELOPE_FRAME_S) -> Envelope:
    """Frame RMS levels of the media's audio (empty when there is no audio / no FFmpeg)."""
    if not media_path or shutil.which("ffmpeg") is None:
        return Envelope(())
    path = Path(media_path)
    if not path.is_file():
        return Envelope(())
    command = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn", "-ac", "1",
               "-ar", str(ENVELOPE_SAMPLE_RATE), "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"]
    try:
        completed = subprocess.run(command, capture_output=True, check=False, timeout=600)
    except (OSError, subprocess.TimeoutExpired):
        return Envelope(())
    if completed.returncode != 0 or not completed.stdout:
        return Envelope(())
    return envelope_from_pcm(completed.stdout, ENVELOPE_SAMPLE_RATE, frame_s=frame_s)


def envelope_from_pcm(raw: bytes, sample_rate: int, *, frame_s: float = ENVELOPE_FRAME_S) -> Envelope:
    frame = max(1, int(round(sample_rate * frame_s)))
    try:
        import numpy as np

        samples = np.frombuffer(raw[: len(raw) - len(raw) % 2], dtype="<i2").astype(np.float64)
        count = len(samples) // frame
        if count <= 0:
            return Envelope((), frame_s)
        blocks = samples[: count * frame].reshape(count, frame)
        rms = np.sqrt((blocks * blocks).mean(axis=1))
        db = 20.0 * np.log10(np.maximum(rms, 1.0) / 32768.0)
        return Envelope(tuple(float(x) for x in db), frame_s)
    except ImportError:  # pure-Python fallback (slower, same numbers)
        import array
        import sys

        values = array.array("h")
        values.frombytes(raw[: len(raw) - len(raw) % 2])
        if sys.byteorder != "little":
            values.byteswap()
        levels = []
        for offset in range(0, len(values) - frame + 1, frame):
            chunk = values[offset:offset + frame]
            rms = math.sqrt(sum(float(v) * float(v) for v in chunk) / len(chunk))
            levels.append(20.0 * math.log10(max(rms, 1.0) / 32768.0))
        return Envelope(tuple(levels), frame_s)


_PTS = re.compile(r"pts_time:([0-9.]+)")


def scene_cuts(media_path: str | Path | None, start: float, end: float,
               threshold: float = SCENE_THRESHOLD) -> list[float]:
    """Shot changes (paced-clip seconds) inside [start, end]; empty when unavailable."""
    if not media_path or shutil.which("ffmpeg") is None or end <= start:
        return []
    path = Path(media_path)
    if not path.is_file():
        return []
    start = max(0.0, float(start))
    command = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "info", "-ss", f"{start:.3f}", "-i", str(path),
               "-t", f"{max(0.05, end - start):.3f}", "-an", "-sn",
               "-vf", f"scale=160:-2,select='gt(scene,{threshold})',showinfo", "-f", "null", "-"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                   check=False, timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        return []
    cuts = []
    for line in completed.stderr.splitlines():
        if "showinfo" not in line:
            continue
        match = _PTS.search(line)
        if match:
            cuts.append(round(start + float(match.group(1)), 3))
    return sorted(set(cuts))


def words_from_profile(profile: Mapping[str, Any] | None) -> list[Word]:
    """Final caption words on the exact paced-clip clock (the most accurate word timing MIMIR has)."""
    if not isinstance(profile, Mapping) or str(profile.get("status", "")) != "ok":
        return []
    result = []
    for row in profile.get("words", []) or []:
        if not isinstance(row, Mapping):
            continue
        text = str(row.get("word", "")).strip()
        try:
            start, end = float(row.get("edited_start")), float(row.get("edited_end"))
        except (TypeError, ValueError):
            continue
        if text and end > start >= 0.0:
            result.append(Word(text, start, end))
    result.sort(key=lambda w: (w.start, w.end))
    return result


def words_from_rows(rows: Iterable[Mapping[str, Any]], start_key: str = "edited_start",
                    end_key: str = "edited_end", text_key: str = "word") -> list[Word]:
    result = []
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        try:
            start, end = float(row.get(start_key)), float(row.get(end_key))
        except (TypeError, ValueError):
            continue
        text = str(row.get(text_key, "")).strip()
        if text and end > start >= 0.0:
            result.append(Word(text, start, end))
    result.sort(key=lambda w: (w.start, w.end))
    return result


def speech_phrases(words: Sequence[Word], *, pause: float = PHRASE_PAUSE_S,
                   sentence_pause: float = SENTENCE_PAUSE_S) -> list[Phrase]:
    phrases: list[Phrase] = []
    first = 0
    for index in range(1, len(words) + 1):
        boundary = index == len(words)
        if not boundary:
            gap = words[index].start - words[index - 1].end
            sentence_end = words[index - 1].text.rstrip("\"')]").endswith((".", "!", "?"))
            boundary = gap >= pause or (sentence_end and gap >= sentence_pause)
        if boundary:
            chunk = words[first:index]
            if chunk:
                phrases.append(Phrase(chunk[0].start, max(w.end for w in chunk), first, index - 1,
                                      " ".join(w.text for w in chunk)))
            first = index
    return phrases


# ============================================================
# ACOUSTIC EVENT SHAPE
# ============================================================

def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return -90.0
    return ordered[max(0, min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1)))))]


def acoustic_extent(envelope: Envelope, core_start: float, core_end: float, clip_duration: float
                    ) -> tuple[float | None, float | None, str]:
    """(onset, completion, note) of the sound that carries the event.

    The local floor is the quiet level around the event; the activity level is
    set relative to how loud the event itself is, so a scream over loud game
    audio and a quiet gasp in a silent room are measured the same way. When the
    sound never falls back to the floor inside the search window (continuous
    music / chatter) the acoustic evidence is reported as uninformative instead
    of stretching the intro to the search limit.
    """
    if not envelope.db:
        return None, None, "no audio envelope"
    context = envelope.window(core_start - NOISE_CONTEXT_S, core_end + NOISE_CONTEXT_S)
    event = envelope.window(core_start, core_end)
    if not context or not event:
        return None, None, "event outside the audio"
    floor = _percentile(context, 0.20)
    peak = max(event)
    rise = peak - floor
    if rise < MIN_EVENT_RISE_DB:
        return None, None, f"event only {rise:.1f} dB above the local floor; no acoustic boundary"
    threshold = floor + max(3.0, 0.30 * rise)
    bridge = max(1, int(round(ACTIVITY_BRIDGE_S / envelope.frame_s)))

    def walk(start_index: int, step: int, limit_index: int) -> tuple[int, bool]:
        """Last active frame reached walking from start_index; True when a quiet stretch was found."""
        index = start_index
        last_active = start_index
        quiet = 0
        while (step < 0 and index > limit_index) or (step > 0 and index < limit_index):
            index += step
            if envelope.db[index] >= threshold:
                last_active = index
                quiet = 0
            else:
                quiet += 1
                if quiet > bridge:
                    return last_active, True
        return last_active, False

    core_first = envelope.index(core_start)
    core_last = envelope.index(core_end)
    onset_limit = envelope.index(max(0.0, core_start - MAX_ONSET_SEARCH_S))
    decay_limit = envelope.index(min(clip_duration, core_end + MAX_DECAY_SEARCH_S))
    onset_index, onset_found = walk(core_first, -1, onset_limit)
    decay_index, decay_found = walk(core_last, 1, decay_limit)
    onset = envelope.time(onset_index) if onset_found else None
    completion = min(clip_duration, envelope.time(decay_index) + envelope.frame_s) if decay_found else None
    notes = []
    if onset is None:
        notes.append("sound already active before the onset search window")
    if completion is None:
        notes.append("sound still active after the decay search window")
    return onset, completion, "; ".join(notes) or f"event {rise:.1f} dB above floor"


# ============================================================
# BOUNDS
# ============================================================

def _inside_word(t: float, words: Sequence[Word]) -> Word | None:
    for word in words:
        if word.start < t < word.end:
            return word
    return None


def _silence_before(t: float, words: Sequence[Word]) -> float:
    previous_end = max((w.end for w in words if w.end <= t + 1e-6), default=0.0)
    return max(0.0, t - previous_end)


def _silence_after(t: float, words: Sequence[Word], clip_duration: float) -> float:
    next_start = min((w.start for w in words if w.start >= t - 1e-6), default=clip_duration)
    return max(0.0, next_start - t)


def compute_intro_bounds(
    event: EventEvidence,
    *,
    clip_duration: float,
    words: Sequence[Word] = (),
    envelope: Envelope | None = None,
    cuts: Sequence[float] = (),
    min_duration: float = MIN_INTRO_S,
    max_duration: float = MAX_INTRO_S,
) -> IntroBounds:
    duration = max(0.0, float(clip_duration))
    core_start = max(0.0, min(duration, float(event.core_start)))
    core_end = max(core_start, min(duration, float(event.core_end)))
    bounds = IntroBounds(start=core_start, end=core_end, core_start=core_start, core_end=core_end)
    words = sorted(words, key=lambda w: (w.start, w.end))

    # 1. Acoustic shape of the event itself.
    onset = completion = None
    if envelope is not None:
        onset, completion, note = acoustic_extent(envelope, core_start, core_end, duration)
        bounds.notes.append(f"audio: {note}")
    bounds.onset, bounds.completion = onset, completion
    start = min(core_start, onset if onset is not None else core_start)
    end = max(core_end, completion if completion is not None else core_end)
    if event.lead_in_start is not None and event.lead_in_start < start:
        start = max(0.0, float(event.lead_in_start))
        bounds.notes.append("editor-selected spoken lead-in kept")

    # 2. Speech: never cut a phrase that carries the event; edge phrases whole or out.
    phrases = speech_phrases(words)
    kept: list[Phrase] = []
    for phrase in phrases:
        if phrase.end <= start or phrase.start >= end:
            continue
        carries_event = phrase.start < core_end and phrase.end > core_start
        if event.lead_in_start is not None and phrase.start <= event.lead_in_start < phrase.end:
            carries_event = True
        overlap = min(end, phrase.end) - max(start, phrase.start)
        share = overlap / max(1e-6, phrase.duration)
        if carries_event or share >= EDGE_PHRASE_KEEP_SHARE:
            kept.append(phrase)
            start = min(start, phrase.start)
            end = max(end, phrase.end)
        elif phrase.start < start:
            start = phrase.end + min(0.05, max(0.0, (start - phrase.end) / 2))
        else:
            end = max(core_end, phrase.start - 0.02)
    bounds.phrases = [{"text": p.text[:160], "span": [round(p.start, 3), round(p.end, 3)]} for p in kept]

    # 3. Never begin or end inside a word (a word the phrase pass did not cover).
    word = _inside_word(start, words)
    if word is not None:
        start = word.start
    word = _inside_word(end, words)
    if word is not None:
        end = word.end

    # 4. Handles from silence that really exists.
    start = max(0.0, start - min(MAX_LEAD_HANDLE_S, _silence_before(start, words)))
    end = min(duration, end + min(MAX_TAIL_HANDLE_S, _silence_after(end, words, duration)))

    # 5. Shot boundaries: no flash-frame slivers at either edge.
    for cut in sorted(float(c) for c in cuts):
        if start < cut <= min(start + SHOT_SLIVER_S, core_start) and _inside_word(cut, words) is None \
                and not any(p.start < cut < p.end for p in kept):
            bounds.shot_snaps.append(f"start {start:.3f}->{cut:.3f} (shot change)")
            start = cut
        elif max(end - SHOT_SLIVER_S, core_end) <= cut < end and _inside_word(cut, words) is None \
                and not any(p.start < cut < p.end for p in kept):
            bounds.shot_snaps.append(f"end {end:.3f}->{cut:.3f} (shot change)")
            end = cut

    # 6. Generic safety bounds.
    if end - start > max_duration:
        start, end = _fit_ceiling(start, end, core_start, core_end, kept, words, max_duration, bounds)
    if end - start < min_duration:
        start, end = _reach_floor(start, end, words, duration, min_duration, bounds)

    bounds.start = round(max(0.0, start), 3)
    bounds.end = round(min(duration, max(end, start + 0.05)), 3)
    return bounds


def _fit_ceiling(start: float, end: float, core_start: float, core_end: float, kept: list[Phrase],
                 words: Sequence[Word], max_duration: float, bounds: IntroBounds) -> tuple[float, float]:
    """Drop whole phrases farthest from the core; only a single over-long phrase is cut at a word gap."""
    phrases = sorted(kept, key=lambda p: -max(core_start - p.end, p.start - core_end, 0.0))
    while end - start > max_duration and phrases:
        far = phrases.pop(0)
        if far.start < core_end and far.end > core_start:
            break  # never drop the phrase that carries the event
        if far.end <= core_start:
            start = max(start, far.end)
        else:
            end = min(end, far.start)
        bounds.notes.append(f"ceiling: dropped edge phrase {far.text[:40]!r}")
    if end - start > max_duration:
        centre = (core_start + core_end) / 2.0
        start = max(start, min(core_start, centre - max_duration / 2.0))
        end = start + max_duration
        word = _inside_word(start, words)
        if word is not None:
            start = word.end
        word = _inside_word(end, words)
        if word is not None:
            end = word.start
        bounds.notes.append("ceiling: event phrase longer than the safety ceiling; cut at word gaps")
    return start, end


def _reach_floor(start: float, end: float, words: Sequence[Word], duration: float, min_duration: float,
                 bounds: IntroBounds) -> tuple[float, float]:
    """Extend a too-short window into real silence/sound around it, never into a partial word."""
    missing = min_duration - (end - start)
    tail_room = min(_silence_after(end, words, duration), missing)
    end += tail_room
    missing = min_duration - (end - start)
    if missing > 0:
        lead_room = min(_silence_before(start, words), missing)
        start -= lead_room
        missing -= lead_room
    if missing > 0:
        # Silence is exhausted: include the neighbouring phrase whole rather than a fragment
        # (the following one first: it is the reaction/consequence of the event).
        phrases = speech_phrases(words)
        after = [p for p in phrases if p.start >= end - 1e-6]
        before = [p for p in phrases if p.end <= start + 1e-6]
        for phrase in ([after[0]] if after else []) + ([before[-1]] if before else []):
            if end - start >= min_duration:
                break
            if phrase.start >= end - 1e-6 and phrase.end - start <= MAX_INTRO_S:
                end = phrase.end
                bounds.notes.append(f"floor: kept following phrase {phrase.text[:40]!r}")
            elif phrase.end <= start + 1e-6 and end - phrase.start <= MAX_INTRO_S:
                start = phrase.start
                bounds.notes.append(f"floor: kept preceding phrase {phrase.text[:40]!r}")
    if end - start < min_duration:
        centre = (start + end) / 2.0
        start = max(0.0, min(centre - min_duration / 2.0, duration - min_duration))
        end = min(duration, start + min_duration)
    bounds.notes.append(f"floor: extended to the {min_duration:.1f}s perceptual minimum")
    return start, end
