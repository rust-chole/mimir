"""Caption words in OUTPUT time, grouping and display timing (ported CLEAN V3 caption policy).

* acoustic word times are never changed; only DISPLAY ends get readability holds;
* groups: <= N words / <= M characters, break on real pauses, sentence ends,
  a visible (human-confirmed) name change and every timeline segment change;
* raw diarization ids are metadata, never printed; only confirmed names are;
* a second visual lane exists only inside measured speech overlap;
* uncertain words are shown but never emphasized.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from mimir.config import CaptionStyle
from mimir.timeline.schema import COLD_OPEN, STORY, Timeline

MIN_VISIBLE_FRACTION = 0.6
MIN_TRUE_OVERLAP = 0.045
OVERLAP_PAD = 0.12
EPISODE_GAP = 0.35     # overlapping pairs closer than this belong to one overlap episode
MAX_TWO_LANE_EVENT = 1.15


@dataclass
class RenderWord:
    key: str                 # "<word id>@<segment kind>"
    word_id: str
    segment: str
    segment_index: int
    text: str
    start: float             # output seconds (acoustic, unchanged)
    end: float
    speaker: str
    label: str               # confirmed name or ""
    uncertain: bool
    emphasis: bool = False
    lane: str = "main"


@dataclass
class Group:
    words: list[RenderWord] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def lane(self) -> str:
        return self.words[0].lane


def _canon(text: str) -> str:
    return re.sub(r"[^\w']+", "", text.casefold())


def map_words(words: Sequence[dict[str, Any]], timeline: Timeline, names: dict[str, str]) -> list[RenderWord]:
    rows: list[RenderWord] = []
    for word in words:
        start, end = float(word["start"]), float(word["end"])
        for out_start, out_end, segment_index in timeline.map_interval(start, end, kinds=(COLD_OPEN, STORY)):
            if (out_end - out_start) < MIN_VISIBLE_FRACTION * (end - start) - 1e-6:
                continue  # a word mostly outside the kept media is not shown
            segment = timeline.segments[segment_index]
            speaker = word.get("speaker") or ""
            rows.append(RenderWord(f"{word['id']}@{segment.kind}", word["id"], segment.kind, segment_index,
                                   word["text"], round(out_start, 4), round(out_end, 4), speaker,
                                   names.get(speaker, ""), bool(word.get("uncertain"))))
    rows.sort(key=lambda w: (w.start, w.end))
    return rows


def mark_emphasis(rows: Sequence[RenderWord], highlights: Sequence[str], payoff_out: Sequence[tuple[float, float]],
                  names: dict[str, str]) -> None:
    phrases = [[_canon(t) for t in phrase.split() if _canon(t)] for phrase in highlights]
    name_tokens = {_canon(part) for name in names.values() for part in name.split()}
    canon = [_canon(w.text) for w in rows]
    for phrase in phrases:
        if not phrase:
            continue
        for i in range(len(rows) - len(phrase) + 1):
            if canon[i:i + len(phrase)] == phrase:
                for k in range(len(phrase)):
                    rows[i + k].emphasis = True
    for row, value in zip(rows, canon):
        if value in name_tokens or any(ch.isdigit() for ch in value):
            row.emphasis = True
        if any(a <= row.start < b for a, b in payoff_out) and len(value) >= 4:
            row.emphasis = True
    for row in rows:
        if row.uncertain:
            row.emphasis = False


def assign_lanes(rows: Sequence[RenderWord], measured: Sequence[tuple[float, float, str]] = ()
                 ) -> list[tuple[float, float]]:
    """A second lane only for the INTERRUPTING speaker inside measured simultaneous speech.

    Sequential turn-taking stays on one lane. Overlap is decided per EPISODE, never per word
    pair (in interleaved speech both voices alternately "start later"):

    * ``measured``: diarization overlaps mapped to output time as (start, end, interrupter),
      covering the interrupter's whole turn - authoritative;
    * elsewhere, overlapping words of two speakers form an episode whose interrupter is the
      speaker who cut in at its first overlapping pair.

    A word moves to the secondary lane only when its onset lies inside a window of its own
    speaker; the held speaker stays on the main lane throughout.
    """
    windows: list[tuple[float, float, str]] = [(a - OVERLAP_PAD, b + OVERLAP_PAD, speaker)
                                               for a, b, speaker in measured]
    measured_windows = list(windows)
    episodes: list[dict[str, Any]] = []
    for i, first in enumerate(rows):
        for second in rows[i + 1:]:
            if second.start >= first.end:
                break
            if not first.speaker or not second.speaker or first.speaker == second.speaker or \
                    first.segment_index != second.segment_index:
                continue
            if min(first.end, second.end) - max(first.start, second.start) < MIN_TRUE_OVERLAP:
                continue
            early, late = (first, second) if first.start <= second.start else (second, first)
            if any(a <= late.start <= b for a, b, _ in measured_windows):
                continue  # diarization already measured this overlap
            speakers = {early.speaker, late.speaker}
            episode = episodes[-1] if episodes else None
            if episode and episode["speakers"] == speakers and late.start <= episode["end"] + EPISODE_GAP:
                episode["end"] = max(episode["end"], *(w.end for w in (early, late)
                                                       if w.speaker == episode["interrupter"]))
            else:
                episodes.append({"speakers": speakers, "interrupter": late.speaker, "start": late.start,
                                 "end": late.end})
    windows += [(e["start"] - OVERLAP_PAD, e["end"] + OVERLAP_PAD, e["interrupter"]) for e in episodes]
    merged: list[tuple[float, float, str]] = []
    for a, b, speaker in sorted(windows):
        if merged and merged[-1][2] == speaker and a <= merged[-1][1] + 0.12:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b), speaker)
        else:
            merged.append((a, b, speaker))
    for row in rows:
        if any(speaker == row.speaker and a <= row.start <= b for a, b, speaker in merged):
            row.lane = "secondary"
    return [(a, b) for a, b, _ in merged]


def group_words(rows: Sequence[RenderWord], style: CaptionStyle) -> list[Group]:
    groups: list[Group] = []
    for lane in ("main", "secondary"):
        current = Group()
        chars = 0
        for row in (r for r in rows if r.lane == lane):
            if current.words:
                previous = current.words[-1]
                projected = chars + 1 + len(row.text)
                if (row.segment_index != previous.segment_index and row.segment == COLD_OPEN) or \
                        row.segment != previous.segment or \
                        (previous.label or row.label) and previous.label.casefold() != row.label.casefold() or \
                        row.start - previous.end >= style.group_break_gap or \
                        len(current.words) >= style.words_per_group or projected > style.max_group_chars:
                    groups.append(current)
                    current, chars = Group(), 0
            current.words.append(row)
            chars += len(row.text) + (1 if len(current.words) > 1 else 0)
            if row.text.rstrip().endswith((".", "!", "?", ";", ":")):
                groups.append(current)
                current, chars = Group(), 0
        if current.words:
            groups.append(current)
    groups.sort(key=lambda g: (g.start, g.lane))
    return groups


def event_end(group: Group, index: int, style: CaptionStyle, next_group_start: float | None, limit: float,
              two_lane: bool) -> float:
    """DISPLAY end of the event showing word ``index`` active (acoustic times stay untouched)."""
    word = group.words[index]
    start, end = word.start, word.end
    if index + 1 < len(group.words):
        next_start = group.words[index + 1].start
        if 0.0 <= next_start - end <= style.max_hold_gap:
            end = next_start
        end = min(end, next_start)
        end = max(end, min(next_start, start + style.min_event))
    else:
        desired = max(end, start + style.terminal_hold)
        if next_group_start is not None and next_group_start > start:
            desired = min(desired, next_group_start)
            if 0.0 <= next_group_start - end <= style.max_hold_gap:
                desired = max(desired, next_group_start)
        end = desired
    if two_lane:
        end = min(end, start + MAX_TWO_LANE_EVENT)
    return min(max(end, start + style.min_event), limit)
