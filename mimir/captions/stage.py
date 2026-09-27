"""Stage ``captions``: frozen caption truth + canonical timeline -> ASS + event manifest."""
from __future__ import annotations

from typing import Any

from mimir.config import CaptionStyle, Settings, section
from mimir.core.stage import StageContext, StageOutput
from mimir.captions.ass import build_document, group_text, hook_events, seam_position
from mimir.captions.layout import assign_lanes, event_end, group_words, map_words, mark_emphasis
from mimir.errors import StageError
from mimir.timeline.schema import COLD_OPEN, STORY, Timeline
from mimir.transcript.truth import truth_signature

HOOK_MIN_SECONDS = 1.7
HOOK_END_CLEARANCE = 0.12
HOOK_BAND_MARGIN = 40


def hook_y(plan: dict[str, Any], end_frame: int, style: CaptionStyle, height: int) -> int:
    """Hook placement: inside the empty band above fitted content when that band can hold it.

    A fitted (letterboxed) window leaves blurred fill above the content; the hook then sits in
    that band instead of covering the top of a screen/UI. Cropped and stacked frames keep the
    style position.
    """
    default = int(height * style.hook_y_ratio)
    frames = range(min(end_frame, plan["frame_count"]))
    if not frames or any(plan["layout"][i] != 0 for i in frames):
        return default
    band = min(height * (0.5 - plan["windows"][i][1] / plan["windows"][i][2]) for i in frames)
    block = style.hook_size * 2.4  # two lines incl. bounce overshoot and outline
    if band < block + 2 * HOOK_BAND_MARGIN:
        return default
    return int(round(max(band / 2, block / 2 + HOOK_BAND_MARGIN)))


class CaptionStage:
    name = "captions"
    version = 1
    deps = ("caption_truth", "story", "cold_open", "timeline", "edit_compile")

    def params(self, settings: Settings) -> Any:
        return {"style": section(settings, "captions"), "size": [settings.output.width, settings.output.height]}

    def run(self, ctx: StageContext) -> StageOutput:
        truth = ctx.dep("caption_truth").json("caption_truth")
        if truth_signature(truth["words"]) != truth["signature"]:
            raise StageError(self.name, "caption truth was modified after it was frozen")
        story = ctx.dep("story").json("story")
        cold = ctx.dep("cold_open").json("cold_open")
        timeline = Timeline.from_dict(ctx.dep("timeline").json("timeline"))
        plan = ctx.dep("edit_compile").json("render_plan")
        style = ctx.settings.captions
        width, height = ctx.settings.output.width, ctx.settings.output.height
        names = truth.get("confirmed_names", {})
        rows = map_words(truth["words"], timeline, names)
        payoff_out = [(a, b) for a, b, _ in timeline.map_interval(story["payoff"]["start"], story["payoff"]["end"],
                                                                  kinds=(STORY, COLD_OPEN))]
        mark_emphasis(rows, story.get("caption_highlights", []), payoff_out, names)
        measured = [(a, b, overlap["interrupter"]) for overlap in truth.get("overlaps", [])
                    for a, b, _ in timeline.map_interval(*overlap["turn"], kinds=(STORY, COLD_OPEN))]
        overlap_windows = assign_lanes(rows, measured)
        groups = group_words(rows, style)
        events = []
        two_lane = bool(overlap_windows)
        split = plan["geometry"]["stack_split"]
        for index, group in enumerate(groups):
            segment = timeline.segments[group.words[0].segment_index]
            later = [g.start for g in groups[index + 1:] if g.lane == group.lane]
            next_start = later[0] if later else None
            last_segment = timeline.segments[group.words[-1].segment_index]
            limit = (last_segment.end_frame if group.words[0].segment == STORY else segment.end_frame) / timeline.fps
            frame = min(plan["frame_count"] - 1, int(group.start * timeline.fps))
            prefix = seam_position(width, height, split) if plan["caption_band"][frame] == "seam" else ""
            for k, word in enumerate(group.words):
                end = event_end(group, k, style, next_start, limit, two_lane)
                if end <= word.start:
                    continue
                events.append({"group": index, "word_key": word.key, "word_id": word.word_id, "text":
                               group_text(group, k, style), "start": round(word.start, 3), "end": round(end, 3),
                               "lane": group.lane, "prefix": prefix, "segment": word.segment})
        by_lane: dict[str, list[dict[str, Any]]] = {}
        for event in sorted(events, key=lambda e: e["start"]):
            lane = by_lane.setdefault(event["lane"], [])
            if lane and event["start"] < lane[-1]["end"]:
                # simultaneous words on one lane: the earlier event yields; a coincident onset is
                # displayed 20 ms later (display only - acoustic word times are untouched)
                event["start"] = round(max(event["start"], lane[-1]["start"] + 0.02), 3)
                event["end"] = round(max(event["end"], event["start"] + 0.02), 3)
                lane[-1]["end"] = event["start"]
            lane.append(event)
        hook = cold.get("hook", {})
        hook_lines: list[str] = []
        hook_window = None
        if hook.get("accepted") and hook.get("text"):
            co = timeline.kind_segments(COLD_OPEN)[0]
            co_end = co.end_frame / timeline.fps
            end = max(min(HOOK_MIN_SECONDS, co_end), co_end - HOOK_END_CLEARANCE)
            y = hook_y(plan, int(round(end * timeline.fps)), style, height)
            hook_lines = hook_events(hook["text"], 0.0, end, style, width, height, y)
            hook_window = [0.0, round(end, 3)]
        document = build_document(groups, events, hook_lines, style, width, height)
        path = ctx.out_dir / "captions.ass"
        path.write_text(document, encoding="utf-8")
        manifest = {
            "truth_signature": truth["signature"],
            "words": [{"key": w.key, "word_id": w.word_id, "text": w.text, "start": w.start, "end": w.end,
                       "speaker": w.speaker, "label": w.label, "uncertain": w.uncertain, "emphasis": w.emphasis,
                       "lane": w.lane, "segment": w.segment} for w in rows],
            "groups": [{"index": i, "lane": g.lane, "words": [w.key for w in g.words], "start": g.start,
                        "label": g.words[0].label} for i, g in enumerate(groups)],
            "events": [{k: e[k] for k in ("group", "word_key", "word_id", "start", "end", "lane", "segment")}
                       for e in events],
            "overlap_windows": [[round(a, 3), round(b, 3)] for a, b in overlap_windows],
            "hook": {"text": hook.get("text", ""), "window": hook_window},
        }
        return StageOutput(data={"captions": manifest}, files={"ass": path})
