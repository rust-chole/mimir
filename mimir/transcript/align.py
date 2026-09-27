"""Map lexical-authority text onto the immutable timing-authority word clock.

Contract (ported from the proven CLEAN V3 caption clock):

* the lexical ear owns visible wording, the timing ear owns measured onsets;
* exact anchors are never shifted to make the text fit;
* text-only words are placed only in the local gap before the next anchor;
  if that gap is too small, time is borrowed from the TAIL of the previous
  token - a later measured onset is never pushed;
* no global offset, no silence snapping, no cumulative drift;
* one lexical word is one token: simultaneous speech keeps both words, each with its
  own measured interval (coincident onsets and measured overlaps are preserved, never
  merged into one token and never trimmed away);
* the result is audited for monotonicity (non-decreasing onsets; overlap only where
  the timing ear measured it) and fails closed.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

from mimir.models.provider import TimedWord
from mimir.transcript.tokens import canonical_word, tokenize

MIN_WORD = 0.035
MIN_GAP = 0.005
OVERLAP_TOLERANCE = 0.010   # boundary jitter below this is trimmed; above it is measured overlap
MEASURED = ("anchor", "replace")


class AlignmentError(RuntimeError):
    pass


@dataclass
class AlignedWord:
    text: str
    start: float
    end: float
    source: str        # anchor | replace | insert_gap | insert_tail | insert_leading | insert_trailing
    overlaps_previous: bool = False   # measured simultaneous speech with the previous token


def _r(value: float) -> float:
    return round(float(value), 3)


def clock_tokens(words: Sequence[TimedWord]) -> list[TimedWord]:
    rows = []
    for word in words:
        text = str(word.text).strip()
        if not text or not canonical_word(text):
            continue
        start = float(word.start)
        rows.append(TimedWord(text, start, max(start, float(word.end))))
    return rows


def align_text_to_clock(text: str, clock_words: Sequence[TimedWord], duration: float
                        ) -> tuple[list[AlignedWord], float]:
    """Return (aligned words, sequence alignment ratio)."""
    target = tokenize(text)
    clock = clock_tokens(clock_words)
    duration = max(0.0, float(duration))
    if not target or not clock:
        raise AlignmentError("alignment needs both lexical text and a word clock")
    matcher = SequenceMatcher(None, [canonical_word(w.text) for w in clock],
                              [canonical_word(t) for t in target], autojunk=False)
    ratio = float(matcher.ratio())
    rows: list[AlignedWord] = []

    def append(word: str, start: float, end: float, source: str, subdivided: bool = False) -> None:
        """``subdivided``: a later share of one span split between several words (sequential by
        construction, never simultaneous speech; no minimum display duration is imposed)."""
        word = word.strip()
        if not word:
            return
        start = max(0.0, min(duration, float(start)))
        end = max(start + (0.001 if subdivided else 0.025), min(duration, float(end)))
        overlap = False
        if rows and rows[-1].end > start:
            previous = rows[-1]
            measured = source in MEASURED and previous.source in MEASURED and not subdivided
            if measured and (start <= previous.start + 0.001 or previous.end - start > OVERLAP_TOLERANCE):
                # Simultaneous speech measured by the timing ear (two voices, one clock):
                # both words keep their own measured interval and stay separate tokens.
                start = max(start, previous.start)
                overlap = True
            else:
                safe_end = min(previous.end, start - MIN_GAP)
                if safe_end <= previous.start:
                    safe_end = start
                previous.end = _r(min(start, max(previous.start + 0.001, safe_end)))
        rows.append(AlignedWord(word, _r(start), _r(end), source, overlap))

    def distribute(words: list[str], start: float, end: float, source: str) -> None:
        words = [w for w in words if w.strip()]
        if not words:
            return
        start = max(0.0, min(duration, float(start)))
        end = max(start + 0.025, min(duration, float(end)))
        span = end - start
        if len(words) == 1:
            append(words[0], start, end, source)
            return
        if span < MIN_WORD * len(words):
            # no room for comfortable durations: equal shares, still one token per word
            share = span / len(words)
            for index, word in enumerate(words):
                append(word, start + index * share, start + (index + 1) * share, source, subdivided=index > 0)
            return
        weights = [max(1, len(canonical_word(w))) for w in words]
        remaining_weight = float(sum(weights))
        cursor = start
        for index, (word, weight) in enumerate(zip(words, weights)):
            if index == len(words) - 1:
                token_end = end
            else:
                rest = MIN_WORD * (len(words) - index - 1)
                remaining = end - cursor
                share = remaining * (weight / max(1.0, remaining_weight))
                token_end = min(end, cursor + max(MIN_WORD, min(share, remaining - rest)))
            append(word, cursor, token_end, source, subdivided=index > 0)
            remaining_weight -= weight
            cursor = token_end

    def place_insertion(words: list[str], position: int) -> None:
        words = [w for w in words if w.strip()]
        if not words:
            return
        prev_clock = clock[position - 1] if position > 0 else None
        next_clock = clock[position] if position < len(clock) else None
        next_start = next_clock.start if next_clock is not None else duration
        previous_end = prev_clock.end if prev_clock is not None else 0.0
        required = max(0.055, MIN_WORD * len(words))
        gap_start, gap_end = max(0.0, previous_end), min(duration, next_start)
        if gap_end - gap_start >= required:
            distribute(words, gap_start + MIN_GAP, gap_end - MIN_GAP, "insert_gap")
            return
        if rows and prev_clock is not None:
            previous = rows[-1]
            hard_right = gap_end if next_clock is not None else min(duration, max(gap_end, previous_end + required))
            insertion_start = max(previous.start + MIN_WORD, hard_right - required)
            if hard_right - insertion_start >= 0.025:
                previous.end = _r(max(previous.start + 0.025, insertion_start - MIN_GAP))
                distribute(words, insertion_start, hard_right - (MIN_GAP if next_clock is not None else 0.0),
                           "insert_tail")
                return
        if next_clock is not None:
            end = max(0.025, next_start - MIN_GAP)
            start = max(0.0, end - required)
            if end > start + 0.02:
                distribute(words, start, end, "insert_leading")
                return
        if rows:
            start = rows[-1].end
            end = min(duration, max(start + 0.04, start + required))
            if end > start:
                distribute(words, start, end, "insert_trailing")
                return
        raise AlignmentError("no local clock room for lexical-only words")

    for tag, s0, s1, t0, t1 in matcher.get_opcodes():
        source_slice = clock[s0:s1]
        target_slice = target[t0:t1]
        if tag == "equal":
            for timed, token in zip(source_slice, target_slice):
                append(token, timed.start, timed.end, "anchor")
        elif tag == "replace":
            if source_slice:
                distribute(target_slice, source_slice[0].start,
                           min(duration, max(source_slice[0].start + 0.025, source_slice[-1].end)), "replace")
            else:
                place_insertion(target_slice, s0)
        elif tag == "insert":
            place_insertion(target_slice, s0)
        # "delete": the timing ear heard an extra token; lexical authority omits it.

    if not rows:
        raise AlignmentError("alignment produced no words")
    previous_start = previous_end = -1.0
    for row in rows:
        if row.start + 1e-9 < previous_start or (
                row.start + 1e-9 < previous_end - OVERLAP_TOLERANCE and not row.overlaps_previous):
            raise AlignmentError("clock monotonicity violated")
        if len(tokenize(row.text)) != 1:
            raise AlignmentError(f"token {row.text!r} is not exactly one lexical word")
        if row.end <= row.start:
            raise AlignmentError("zero/negative word duration")
        previous_start, previous_end = row.start, row.end
    return rows, ratio


def text_agreement(left: str, right: str) -> float:
    a = [canonical_word(t) for t in tokenize(left)]
    b = [canonical_word(t) for t in tokenize(right)]
    if not a and not b:
        return 1.0
    return float(SequenceMatcher(None, a, b, autojunk=False).ratio())
