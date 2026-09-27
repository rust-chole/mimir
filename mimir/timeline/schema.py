"""Timeline data model."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Iterable

COLD_OPEN = "cold_open"
STORY = "story"


@dataclass(frozen=True)
class Segment:
    index: int
    kind: str                # cold_open | story
    source_start: float
    source_end: float        # == source_start + frames / fps (exact)
    start_frame: int         # output frames [start_frame, end_frame)
    end_frame: int

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "kind": self.kind, "source_start": self.source_start,
                "source_end": self.source_end, "start_frame": self.start_frame, "end_frame": self.end_frame}


@dataclass(frozen=True)
class Timeline:
    fps: int
    segments: tuple[Segment, ...]
    story_start: float
    story_end: float
    peak_start: float
    peak_end: float

    # ------------------------------------------------------------ basics

    @property
    def frame_count(self) -> int:
        return self.segments[-1].end_frame if self.segments else 0

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps

    def seconds(self, frame: int) -> float:
        return frame / self.fps

    def kind_segments(self, kind: str) -> list[Segment]:
        return [s for s in self.segments if s.kind == kind]

    @property
    def main_start_frame(self) -> int:
        story = self.kind_segments(STORY)
        return story[0].start_frame if story else 0

    def segment_at(self, frame: int) -> Segment:
        starts = [s.start_frame for s in self.segments]
        index = max(0, bisect_right(starts, frame) - 1)
        return self.segments[min(index, len(self.segments) - 1)]

    def source_time(self, frame: int) -> float:
        """Source time shown by output frame ``frame`` (frame start)."""
        segment = self.segment_at(frame)
        return segment.source_start + (frame - segment.start_frame) / self.fps

    # ----------------------------------------------------------- mapping

    def map_time(self, t: float, kinds: Iterable[str] = (STORY,)) -> list[float]:
        """Output times at which source time ``t`` is shown (the cold open may repeat it)."""
        kinds = set(kinds)
        result = []
        for segment in self.segments:
            if segment.kind in kinds and segment.source_start <= t < segment.source_end:
                result.append(segment.start_frame / self.fps + (t - segment.source_start))
        return result

    def map_interval(self, start: float, end: float, kinds: Iterable[str] = (STORY,)) -> list[tuple[float, float, int]]:
        """Visible output pieces (out_start, out_end, segment index) of a source interval."""
        kinds = set(kinds)
        pieces = []
        for segment in self.segments:
            if segment.kind not in kinds:
                continue
            a, b = max(start, segment.source_start), min(end, segment.source_end)
            if b > a:
                base = segment.start_frame / self.fps - segment.source_start
                pieces.append((a + base, b + base, segment.index))
        return pieces

    def removed_ranges(self) -> list[tuple[float, float]]:
        """Source ranges inside the story that pacing removed."""
        story = self.kind_segments(STORY)
        gaps = []
        for left, right in zip(story, story[1:]):
            if right.source_start > left.source_end + 1e-6:
                gaps.append((left.source_end, right.source_start))
        return gaps

    # ---------------------------------------------------------- serialize

    def to_dict(self) -> dict[str, Any]:
        return {"fps": self.fps, "frame_count": self.frame_count, "duration": round(self.duration, 6),
                "story_start": self.story_start, "story_end": self.story_end,
                "peak_start": self.peak_start, "peak_end": self.peak_end,
                "segments": [s.to_dict() for s in self.segments]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Timeline":
        return cls(int(data["fps"]), tuple(Segment(**{k: s[k] for k in ("index", "kind", "source_start", "source_end",
                                                                         "start_frame", "end_frame")})
                                           for s in data["segments"]),
                   float(data["story_start"]), float(data["story_end"]), float(data["peak_start"]),
                   float(data["peak_end"]))


def quantize(ranges: list[tuple[str, float, float]], fps: int) -> tuple[Segment, ...]:
    """Frame-quantize (kind, source_start, source_end) ranges into consecutive output segments."""
    segments = []
    frame = 0
    for kind, start, end in ranges:
        frames = int(round((end - start) * fps))
        if frames < 2:
            continue
        segments.append(Segment(len(segments), kind, round(start, 6), round(start + frames / fps, 6), frame,
                                frame + frames))
        frame += frames
    return tuple(segments)
