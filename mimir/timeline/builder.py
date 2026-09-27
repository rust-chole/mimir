"""Stage ``timeline``: cold open + paced story -> frame-quantized canonical timeline."""
from __future__ import annotations

from typing import Any, Sequence

from mimir.config import Settings, section
from mimir.core.stage import StageContext, StageOutput
from mimir.errors import StageError
from mimir.timeline.pacing import apply_budget, apply_protection, apply_story_integrity, suggest_cuts
from mimir.timeline.schema import COLD_OPEN, STORY, Timeline, quantize

EDGE_HANDLE_BEFORE = 0.06
EDGE_HANDLE_AFTER = 0.14
MIN_KEEP_SECONDS = 0.12


def snap_edges(start: float, end: float, words: Sequence[dict[str, Any]]) -> tuple[float, float]:
    """Never cut through a word at the story edges."""
    for word in words:
        ws, we = float(word["start"]), float(word["end"])
        if ws < start < we:
            start = ws - EDGE_HANDLE_BEFORE
        if ws < end < we:
            end = we + EDGE_HANDLE_AFTER
    return max(0.0, start), end


def keep_ranges(start: float, end: float, cuts: Sequence[dict[str, Any]]) -> list[tuple[float, float]]:
    ranges = []
    cursor = start
    for cut in sorted((c for c in cuts if c["action"] == "cut"), key=lambda c: c["start"]):
        a, b = max(cursor, cut["start"]), min(end, cut["end"])
        if b <= a:
            continue
        if a - cursor >= MIN_KEEP_SECONDS:
            ranges.append((cursor, a))
        cursor = b
    if end - cursor >= MIN_KEEP_SECONDS:
        ranges.append((cursor, end))
    return ranges


class TimelineStage:
    name = "timeline"
    version = 1
    deps = ("probe", "story", "caption_verify", "cold_open")

    def params(self, settings: Settings) -> Any:
        return {"pacing": section(settings, "pacing")}

    def run(self, ctx: StageContext) -> StageOutput:
        media = ctx.dep("probe").json("media")
        story = ctx.dep("story").json("story")
        words = ctx.dep("caption_verify").json("caption_words")["words"]
        cold = ctx.dep("cold_open").json("cold_open")
        fps = int(media["output_fps"])
        start, end = snap_edges(float(story["start"]), float(story["end"]), words)
        end = min(end, float(media["duration"]))
        payoff = (float(story["payoff"]["start"]), float(story["payoff"]["end"]))
        cuts = suggest_cuts(words, start, end, payoff, ctx.settings.pacing)
        protected = story["protected_ranges"]
        apply_protection(cuts, protected)
        budget = apply_budget(cuts, end - start, ctx.settings.pacing)
        bridges = apply_story_integrity(cuts, protected)
        kept = keep_ranges(start, end, cuts)
        if not kept:
            raise StageError(self.name, "pacing removed the whole story")
        ranges = [(COLD_OPEN, float(cold["window"]["start"]), float(cold["window"]["end"]))]
        ranges += [(STORY, a, b) for a, b in kept]
        segments = quantize(ranges, fps)
        if not segments or segments[0].kind != COLD_OPEN:
            raise StageError(self.name, "cold open segment vanished during quantization")
        timeline = Timeline(fps, segments, round(start, 6), round(end, 6), float(cold["peak"]["start"]),
                            float(cold["peak"]["end"]))
        removed = sum(b - a for a, b in timeline.removed_ranges())
        return StageOutput(data={"timeline": timeline.to_dict(), "pacing": {
            "story_range": [round(start, 3), round(end, 3)], "suggestions": cuts, "cut_budget": budget,
            "bridges": bridges, "removed_seconds": round(removed, 3),
            "main_duration": round(sum(s.frames for s in timeline.kind_segments(STORY)) / fps, 3)}})
