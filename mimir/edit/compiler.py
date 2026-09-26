"""Stage ``edit_compile``: Deterministic Edit Compiler (validated intents -> per-frame render plan).

Computes every crop, zoom, position, movement, smoothing, timing and bound.
Required story content is contained by construction (the solver only widens),
camera changes inside a shot are eased, shot cuts and timeline segment
boundaries may cut, follow motion is dead-zoned/smoothed/speed-capped, and
the resulting path is verified before it is written.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Sequence

from mimir.config import Settings, section
from mimir.core.stage import StageContext, StageOutput
from mimir.edit.camera import SpanPlan, StackState, State, build_path, center_steps, follow
from mimir.edit.context import box_at, samples_of
from mimir.edit.framing import (
    MARGINS,
    STACK_SPLIT,
    clamp_window,
    Geometry,
    Window,
    clip_box,
    full_frame,
    panel_box,
    solve,
    stack_bottom_geometry,
    union,
)
from mimir.edit.intents import Intent
from mimir.errors import StageError
from mimir.timeline.schema import COLD_OPEN, STORY, Timeline

REACTION_ZOOM = 1.15
ACTION_ZOOM = 1.10


def _required_boxes(span: dict[str, Any]) -> list[list[float]]:
    return [r["box"] for r in span["required"]]


class Compiler:
    def __init__(self, settings: Settings, geo: Geometry, timeline: Timeline, context: dict[str, Any],
                 vision: dict[str, Any], plan: dict[str, Any]) -> None:
        self.settings = settings
        self.camera = settings.camera
        self.geo = geo
        self.timeline = timeline
        self.context = context
        self.vision = vision
        self.decisions = {row["id"]: row for row in plan["spans"]}
        self.faces = {f["id"]: samples_of(f) for f in vision["faces"] if not f["static_pattern"]}
        self.layout = vision["layout"]
        self.fps = timeline.fps
        self.notes: list[dict[str, Any]] = []
        # on-screen HUD/UI must be fully in or fully out (never half a score counter) outside talking layouts
        self.hud = [r["box"] for r in vision.get("ui_regions", [])] \
            if self.layout["class"] in ("gameplay", "facecam_gameplay", "screen_content", "scene") else []

    # --------------------------------------------------------------- goals

    def _subject_goals(self, span: dict[str, Any], target: str, margins: tuple[float, float, float],
                       desired_h: float, frames: range) -> tuple[list[State], bool]:
        samples = self.faces.get(target, [])
        visible = next((v for v in span["visible"] if v["id"] == target), None)
        if visible is None or not samples:
            window, notes = solve(self.geo, desired_h=self.geo.h_inside, required=_required_boxes(span))
            return [window] * len(frames), False
        others = [v["box"] for v in span["visible"] if v["id"] != target]
        base, notes = solve(self.geo, desired_h=desired_h, required=_required_boxes(span), subject=visible["box"],
                            subject_margins=margins, avoid=others)
        if notes:
            self.notes.append({"span": span["id"], "notes": notes})
        goals: list[State] = []
        median = visible["box"]
        for frame in frames:
            box = box_at(samples, self.timeline.source_time(frame))
            if box is None:
                goals.append(base)
                continue
            shifted = [box[0], box[1], box[2], box[3]]
            window, _ = solve(self.geo, desired_h=base.h, required=_required_boxes(span), subject=shifted,
                              subject_margins=margins, avoid=others)
            # keep the span's zoom (no pumping); only the position follows
            goals.append(Window(window.cx, window.cy, base.h) if abs(window.h - base.h) < 1e-6 else base)
        return goals, True

    def _stack_state(self, span: dict[str, Any]) -> StackState:
        top_geo_aspect = self.geo.out_w / (self.geo.out_h * STACK_SPLIT)
        top = panel_box(self.geo, self.layout["facecam_box"], top_geo_aspect)
        bottom_geo = stack_bottom_geometry(self.geo)
        facecam = self.layout["facecam_box"]
        required = [r["box"] for r in span["required"]
                    if not (r["box"][0] >= facecam[0] - 0.02 and r["box"][2] <= facecam[2] + 0.02
                            and r["box"][1] >= facecam[1] - 0.02 and r["box"][3] <= facecam[3] + 0.02)]
        actions = sorted(span["actions"], key=lambda a: -a["intensity"])
        focus = [actions[0]["box"]] if actions else []
        window, notes = solve(bottom_geo, desired_h=bottom_geo.h_inside, required=required + focus)
        if not focus and not required:
            window = Window(0.5, window.cy, window.h)
        return StackState(top, window)

    def span_plan(self, span: dict[str, Any], previous: State | None, hard_start: bool, mask_zoom: float) -> SpanPlan:
        decision = self.decisions[span["id"]]
        intent = Intent(decision["intent"])
        f0, f1 = span["frames"]
        frames = range(f0, f1)
        h_inside = self.geo.h_inside
        easing, transition = "smooth", int(round(self.camera.transition_seconds * self.fps))
        follow_mode = False
        required = _required_boxes(span)
        if intent is Intent.HOLD and previous is not None and not hard_start:
            return SpanPlan(span["id"], (f0, f1), [previous] * len(frames), False, easing, transition, False)
        if intent is Intent.HOLD:
            intent = Intent.WIDE_CONTEXT
        if intent is Intent.GAMEPLAY_PRIORITY and self.layout["class"] == "facecam_gameplay":
            goals: list[State] = [self._stack_state(span)] * len(frames)
        elif intent in (Intent.GAMEPLAY_PRIORITY, Intent.SCREEN_PRIORITY):
            goals = [full_frame(self.geo)] * len(frames)
        elif intent is Intent.WIDE_CONTEXT:
            boxes = list(required) + [v["box"] for v in span["visible"]] + [a["box"] for a in span["actions"]] + \
                [e["box"] for e in span["elements"] if e["kind"] in ("object", "action", "person", "face")]
            if boxes:
                window, _ = solve(self.geo, desired_h=h_inside / mask_zoom, required=boxes,
                                  avoid=[v["box"] for v in span["visible"]] + self.hud)
            else:
                window = full_frame(self.geo)
            goals = [window] * len(frames)
        elif intent is Intent.TWO_SHOT:
            faces = sorted(span["visible"], key=lambda v: -v["size"])[:3]
            window, _ = solve(self.geo, desired_h=h_inside / mask_zoom, required=required,
                              pair=[v["box"] for v in faces],
                              avoid=[v["box"] for v in span["visible"] if v not in faces])
            goals = [window] * len(frames)
        elif intent is Intent.ACTION_REGION:
            action = next((a for a in span["actions"] if a["id"] == decision["target_id"]), None)
            boxes = list(required) + ([action["box"]] if action else [])
            window, _ = solve(self.geo, desired_h=h_inside / ACTION_ZOOM, required=boxes,
                              avoid=[v["box"] for v in span["visible"]] + self.hud)
            goals = [window] * len(frames)
        else:
            if intent is Intent.SPEAKER_PUNCH:
                zoom = self.camera.punch_zoom + (0.08 if decision["intensity"] == "strong" else 0.0)
                margins, easing = MARGINS["punch"], "punch"
                transition = int(round(self.camera.punch_seconds * self.fps))
            elif intent is Intent.REACTION:
                zoom, margins = REACTION_ZOOM, MARGINS["reaction"]
                transition = int(round(0.30 * self.fps))
            else:
                zoom, margins = self.camera.medium_zoom * mask_zoom, MARGINS["medium"]
            goals, follow_mode = self._subject_goals(span, decision["target_id"], margins, h_inside / zoom, frames)
            if follow_mode:
                goals = follow(goals, self.fps, deadzone=self.camera.follow_deadzone,  # type: ignore[arg-type]
                               max_speed=self.camera.follow_max_speed, geo=self.geo)
        return SpanPlan(span["id"], (f0, f1), goals, hard_start, easing, max(1, transition), follow_mode)

    # ------------------------------------------------------------- compile

    def _target_box(self, span: dict[str, Any]) -> list[float] | None:
        decision = self.decisions[span["id"]]
        if decision["intent"] not in (Intent.SPEAKER_MEDIUM.value, Intent.SPEAKER_PUNCH.value, Intent.REACTION.value):
            return None
        samples = self.faces.get(decision["target_id"])
        if not samples:
            return None
        return box_at(samples, self.timeline.source_time(span["frames"][0]), tolerance=1.0)

    def hard_frames(self) -> set[int]:
        frames = {s.start_frame for s in self.timeline.segments}
        for cut in self.vision.get("shot_cuts", []):
            for out_t in self.timeline.map_time(cut, kinds=(STORY, COLD_OPEN)):
                frames.add(int(round(out_t * self.fps)))
        return frames

    def compile(self) -> dict[str, Any]:
        hard = self.hard_frames()
        spans = self.context["spans"]
        plans: list[SpanPlan] = []
        previous: State | None = None
        story_segments = self.timeline.kind_segments(STORY)
        jump_cuts = {s.start_frame for s in story_segments[1:]} - {
            int(round(t * self.fps)) for c in self.vision.get("shot_cuts", [])
            for t in self.timeline.map_time(c, kinds=(STORY,))}
        story_order = {s.index: k for k, s in enumerate(story_segments)}
        for span in spans:
            f0 = span["frames"][0]
            segment = self.timeline.segment_at(f0)
            mask = 1.0
            if segment.kind == STORY and story_order[segment.index] % 2 == 1 and segment.start_frame in jump_cuts:
                mask = self.camera.jump_cut_mask_zoom  # alternating framing across a jump cut hides the jump
            hard_start = f0 in hard
            if previous is not None and not hard_start and span["required"]:
                # never ease INTO required content: widen with a cut, narrow smoothly
                window = previous.bottom if isinstance(previous, StackState) else previous
                geo = stack_bottom_geometry(self.geo) if isinstance(previous, StackState) else self.geo
                if not all(window.contains(geo, r["box"], tol=0.02) for r in span["required"]):
                    hard_start = True
            target_box = self._target_box(span)
            if previous is not None and not hard_start and target_box is not None:
                # a different subject that is not in frame: cut to it (never whip-pan across the room)
                window = previous.bottom if isinstance(previous, StackState) else previous
                if isinstance(previous, StackState) or not window.contains(self.geo, target_box, tol=0.01):
                    hard_start = True
            plan = self.span_plan(span, previous, hard_start, mask)
            plans.append(plan)
            previous = plan.goals[-1]
        states, phases = build_path(plans)
        bottom_geo = stack_bottom_geometry(self.geo)
        states = [StackState(s.top, clamp_window(bottom_geo, s.bottom)) if isinstance(s, StackState)
                  else clamp_window(self.geo, s) for s in states]
        if len(states) != self.timeline.frame_count:
            raise StageError("edit_compile", f"path has {len(states)} frames, timeline {self.timeline.frame_count}")
        self._verify(states, phases)
        return self._serialize(states, phases, plans)

    def _verify(self, states: Sequence[State], phases: Sequence[str]) -> None:
        geo = self.geo
        for index, state in enumerate(states):
            window = state.bottom if isinstance(state, StackState) else state
            if not all(math.isfinite(v) for v in (window.cx, window.cy, window.h)) or window.h <= 0:
                raise StageError("edit_compile", f"frame {index}: invalid window {window}")
            if isinstance(state, Window):
                if state.h < geo.h_min - 1e-6 or state.h > geo.h_full + 1e-6:
                    raise StageError("edit_compile", f"frame {index}: window height {state.h:.4f} out of bounds")
                if state.h <= geo.h_inside + 1e-9 and not state.inside(geo, tol=1e-3):
                    raise StageError("edit_compile", f"frame {index}: crop leaves the source")
        steps = center_steps(states, phases)
        limit = self.settings.qc.max_center_step
        if steps and max(steps) > limit + 1e-6:
            raise StageError("edit_compile", f"camera step {max(steps):.4f} exceeds {limit}")

    def _serialize(self, states: Sequence[State], phases: Sequence[str], plans: Sequence[SpanPlan]) -> dict[str, Any]:
        layout, windows, tops = [], [], {}
        for index, state in enumerate(states):
            if isinstance(state, StackState):
                layout.append(1)
                windows.append([round(state.bottom.cx, 5), round(state.bottom.cy, 5), round(state.bottom.h, 5)])
                tops[index] = [round(v, 5) for v in state.top]
            else:
                layout.append(0)
                windows.append([round(state.cx, 5), round(state.cy, 5), round(state.h, 5)])
        top_runs = []
        for index in sorted(tops):
            if top_runs and top_runs[-1]["end"] == index and top_runs[-1]["box"] == tops[index]:
                top_runs[-1]["end"] = index + 1
            else:
                top_runs.append({"start": index, "end": index + 1, "box": tops[index]})
        decisions = self.decisions
        return {
            "fps": self.fps, "frame_count": len(states),
            "output": [self.geo.out_w, self.geo.out_h], "source": [self.geo.src_w, self.geo.src_h],
            "geometry": {"h_inside": self.geo.h_inside, "h_full": self.geo.h_full, "h_min": self.geo.h_min,
                         "stack_split": STACK_SPLIT},
            "layout": layout, "windows": windows, "stack_tops": top_runs, "phases": phases,
            "spans": [{"id": p.span_id, "frames": list(p.frames), "intent": decisions[p.span_id]["intent"],
                       "target_id": decisions[p.span_id]["target_id"], "hard_start": p.hard_start,
                       "follow": p.follow, "transition_frames": p.transition_frames} for p in plans],
            "hard_frames": sorted(self.hard_frames()),
            "caption_band": ["seam" if code == 1 else "low" for code in layout],
            "notes": self.notes,
        }


class EditCompileStage:
    name = "edit_compile"
    version = 1
    deps = ("probe", "vision", "timeline", "edit_context", "edit_validation")

    def params(self, settings: Settings) -> Any:
        return {"camera": section(settings, "camera"), "size": [settings.output.width, settings.output.height],
                "max_center_step": settings.qc.max_center_step}

    def run(self, ctx: StageContext) -> StageOutput:
        media = ctx.dep("probe").json("media")
        geo = Geometry(int(media["width"]), int(media["height"]), ctx.settings.output.width,
                       ctx.settings.output.height, ctx.settings.camera.max_zoom, ctx.settings.camera.max_upscale)
        timeline = Timeline.from_dict(ctx.dep("timeline").json("timeline"))
        compiler = Compiler(ctx.settings, geo, timeline, ctx.dep("edit_context").json("edit_context"),
                            ctx.dep("vision").json("vision"), ctx.dep("edit_validation").json("validated_plan"))
        plan = compiler.compile()
        for note in compiler.notes:
            ctx.ledger.info("framing", f"{note['span']}: {', '.join(note['notes'])}")
        return StageOutput(data={"render_plan": plan})
