import pytest

from mimir.config import Settings, RepairDirectives
from mimir.edit.camera import SpanPlan, build_path, center_steps, follow
from mimir.edit.context import build_spans
from mimir.timeline.schema import COLD_OPEN, STORY, Timeline, quantize
from mimir.edit.framing import MARGINS, Geometry, Window, clamp_window, full_frame, solve
from mimir.edit.intents import Intent
from mimir.edit.validator import validate_plan

GEO = Geometry(1920, 1080, 1080, 1920, 1.45, 2.6)


class TestFraming:
    def test_geometry_of_a_landscape_source(self):
        assert GEO.h_inside == 1.0
        assert GEO.width(1.0) == pytest.approx(0.316, abs=1e-3)
        assert GEO.h_full == pytest.approx(3.16, abs=0.01)
        assert GEO.h_min == pytest.approx(1 / 1.45, abs=1e-3)

    def test_required_content_always_ends_up_in_frame(self):
        people = [[0.2, 0.3, 0.3, 0.5], [0.7, 0.3, 0.8, 0.5]]
        window, notes = solve(GEO, desired_h=0.8, required=people)
        assert all(window.contains(GEO, box) for box in people)
        assert window.h > 1.0 and any("widened" in n for n in notes)

    def test_speaker_crop_stays_inside_and_keeps_the_face_above_captions(self):
        face = [0.45, 0.25, 0.55, 0.45]
        window, _ = solve(GEO, desired_h=0.9, subject=face, subject_margins=MARGINS["medium"])
        assert window.inside(GEO)
        _, y0, _, y1 = window.to_output(GEO, face)
        assert y1 <= 0.61

    def test_neighbour_faces_are_never_half_cut(self):
        target, neighbour = [0.40, 0.3, 0.46, 0.42], [0.53, 0.3, 0.59, 0.42]
        window, _ = solve(GEO, desired_h=0.8, subject=target, subject_margins=MARGINS["punch"], avoid=[neighbour])
        x0, _, x1, _ = window.box(GEO)
        assert not (neighbour[0] < x1 < neighbour[2]) and not (neighbour[0] < x0 < neighbour[2])

    def test_fit_window_and_clamping(self):
        full = full_frame(GEO)
        for corner in ([0, 0, 0.01, 0.01], [0.99, 0.99, 1, 1]):
            assert full.contains(GEO, corner)
        clamped = clamp_window(GEO, Window(0.0, 0.0, 0.9))
        assert clamped.inside(GEO)


class TestCameraPath:
    def plan(self, span, frames, goals, hard=False, easing="smooth", transition=10, follow_=False):
        return SpanPlan(span, frames, goals, hard, easing, transition, follow_)

    def test_cuts_are_instant_and_transitions_are_eased(self):
        a, b = Window(0.3, 0.5, 1.0), Window(0.7, 0.5, 0.8)
        states, phases = build_path([self.plan("s0", (0, 20), [a] * 20),
                                     self.plan("s1", (20, 40), [b] * 20),
                                     self.plan("s2", (40, 60), [a] * 20, hard=True)])
        assert states[19] == a and phases[20] == "transition" and states[29] == b
        assert 0.3 < states[24].cx < 0.7
        assert states[40] == a and phases[40] == "cut"

    def test_follow_ignores_detector_noise_inside_the_dead_zone(self):
        import random

        random.seed(1)
        goals = [Window(0.5 + random.uniform(-0.01, 0.01), 0.5, 0.9) for _ in range(90)]
        path = follow(goals, 30, deadzone=0.035, max_speed=0.35, geo=GEO)
        assert max(abs(w.cx - 0.5) for w in path) < 0.011
        assert len({round(w.cx, 6) for w in path}) <= 2

    def test_follow_speed_is_capped(self):
        goals = [Window(0.3, 0.5, 0.9)] + [Window(0.7, 0.5, 0.9)] * 60
        path = follow(goals, 30, deadzone=0.0, max_speed=0.3, geo=GEO)
        steps = center_steps(path, ["follow"] * len(path))
        assert max(steps) <= 0.3 / 30 + 1e-9


def context(spans, layout="talking_head"):
    return {"fps": 30, "spans": spans, "layout": {"class": layout}}


def span(sid, role, allowed, visible=(), required=(), frames=(0, 60), dominant="S1", actions=()):
    return {"id": sid, "role": role, "allowed": list(allowed), "visible": list(visible), "required": list(required),
            "frames": list(frames), "dominant_speaker": dominant, "actions": list(actions)}


FACE_A = {"id": "face_00", "speaker": "S1", "link": 0.9, "box": [0.2, 0.3, 0.3, 0.5], "size": 0.2}
FACE_B = {"id": "face_01", "speaker": "S2", "link": 0.9, "box": [0.7, 0.3, 0.8, 0.5], "size": 0.2}


class TestValidator:
    def test_intent_without_evidence_walks_down_the_ladder(self):
        ctx = context([span("s0", "setup", ["HOLD", "WIDE_CONTEXT", "SPEAKER_MEDIUM"], [FACE_A])])
        plan = {"decisions": {"s0": {"intent": "SPEAKER_PUNCH", "target_id": "face_00", "intensity": "normal"}}}
        result = validate_plan(ctx, plan, Settings())
        assert result["spans"][0]["intent"] == "SPEAKER_MEDIUM"
        assert result["corrections"][0]["code"] == "evidence_does_not_allow"

    def test_empty_target_is_resolved_from_evidence_not_counted_as_disagreement(self):
        ctx = context([span("s0", "setup", ["HOLD", "WIDE_CONTEXT", "SPEAKER_MEDIUM"], [FACE_A])])
        plan = {"decisions": {"s0": {"intent": "SPEAKER_MEDIUM", "target_id": "", "intensity": "normal"}}}
        result = validate_plan(ctx, plan, Settings())
        assert result["spans"][0]["target_id"] == "face_00"
        assert result["corrections"] == [] and result["correction_ratio"] == 0
        assert result["resolved_targets"] == [{"span": "s0", "intent": "SPEAKER_MEDIUM", "target": "face_00"}]
        wrong = {"decisions": {"s0": {"intent": "SPEAKER_MEDIUM", "target_id": "face_09", "intensity": "normal"}}}
        assert validate_plan(ctx, wrong, Settings())["corrections"][0]["code"] == "target_not_visible"

    def test_unlinked_face_is_never_guessed_as_the_speaker(self):
        # one visible face, but mouth activity never linked it to the voice: no speaker target
        unlinked = {**FACE_A, "speaker": None, "link": 0.0}
        ctx = context([span("s0", "setup", ["HOLD", "WIDE_CONTEXT", "SPEAKER_MEDIUM"], [unlinked])])
        plan = {"decisions": {"s0": {"intent": "SPEAKER_MEDIUM", "target_id": "", "intensity": "normal"}}}
        result = validate_plan(ctx, plan, Settings())
        assert result["spans"][0]["intent"] == "WIDE_CONTEXT" and result["spans"][0]["target_id"] == ""
        assert result["corrections"][0]["code"] == "no_target_evidence"

    def test_downgrade_in_gameplay_layout_protects_the_gameplay(self):
        facecam = {**FACE_A, "box": [0.86, 0.72, 0.92, 0.82], "size": 0.1}
        ctx = context([span("s0", "reaction", ["HOLD", "WIDE_CONTEXT", "GAMEPLAY_PRIORITY"], [facecam])],
                      layout="facecam_gameplay")
        plan = {"decisions": {"s0": {"intent": "SPEAKER_PUNCH", "target_id": "face_00", "intensity": "strong"}}}
        result = validate_plan(ctx, plan, Settings())
        assert result["spans"][0]["intent"] == "GAMEPLAY_PRIORITY"
        assert result["corrections"][-1]["code"] == "layout_protects_content"
        # an explicit WIDE_CONTEXT choice is respected (the compiler shows the whole frame there)
        plan = {"decisions": {"s0": {"intent": "WIDE_CONTEXT", "target_id": "", "intensity": "normal"}}}
        assert validate_plan(ctx, plan, Settings())["spans"][0]["intent"] == "WIDE_CONTEXT"

    def test_story_outranks_the_voice(self):
        required = [{"kind": "subject", "id": f["id"], "box": f["box"], "reason": "payoff"} for f in (FACE_A, FACE_B)]
        ctx = context([span("s0", "payoff", ["HOLD", "WIDE_CONTEXT", "TWO_SHOT", "SPEAKER_MEDIUM", "SPEAKER_PUNCH",
                                              "REACTION"], [FACE_A, FACE_B], required)])
        plan = {"decisions": {"s0": {"intent": "SPEAKER_PUNCH", "target_id": "face_00", "intensity": "strong"}}}
        result = validate_plan(ctx, plan, Settings())
        assert result["spans"][0]["intent"] == Intent.TWO_SHOT.value
        assert any(c["code"] == "story_outranks_voice" for c in result["corrections"])

    def test_punch_restraint_and_missing_decisions(self):
        allowed = ["HOLD", "WIDE_CONTEXT", "SPEAKER_MEDIUM", "SPEAKER_PUNCH"]
        ctx = context([span(f"s{i}", "escalation", allowed, [FACE_A], frames=(i * 30, i * 30 + 30)) for i in range(4)])
        plan = {"decisions": {f"s{i}": {"intent": "SPEAKER_PUNCH", "target_id": "face_00", "intensity": "normal"}
                              for i in range(3)}}
        result = validate_plan(ctx, plan, Settings())
        intents = [row["intent"] for row in result["spans"]]
        assert intents[:3].count("SPEAKER_PUNCH") == 2 and intents[3] == "HOLD"

    def test_qc_repair_widens_named_spans(self):
        ctx = context([span("s0", "setup", ["HOLD", "WIDE_CONTEXT", "SPEAKER_MEDIUM"], [FACE_A])])
        plan = {"decisions": {"s0": {"intent": "SPEAKER_MEDIUM", "target_id": "face_00", "intensity": "normal"}}}
        settings = Settings().with_(repair=RepairDirectives(round=1, widen_spans=("s0",)))
        assert validate_plan(ctx, plan, settings)["spans"][0]["intent"] == "WIDE_CONTEXT"


def test_a_source_shot_cut_is_a_hard_boundary_and_never_eased_across():
    timeline = Timeline(30, quantize([(COLD_OPEN, 20.0, 22.0), (STORY, 10.0, 24.0)], 30), 10.0, 24.0, 20.0, 22.0)
    story = {"beats": [{"role": "setup", "start": 10.0, "end": 24.0}]}
    spans = build_spans(timeline, story, {"shot_cuts": [15.0]}, [], 1.0)
    cut_frame = int(round(timeline.map_time(15.0, kinds=(STORY,))[0] * 30))
    assert any(s["frames"][0] == cut_frame for s in spans)
    geo = Geometry(1920, 1080, 1080, 1920, 1.45, 2.6)
    left, right = Window(0.3, 0.5, geo.h_inside), Window(0.7, 0.5, geo.h_inside)
    plans = [SpanPlan(f"s{i}", tuple(sp["frames"]), [left if sp["frames"][0] < cut_frame else right]
                      * (sp["frames"][1] - sp["frames"][0]), sp["frames"][0] == cut_frame, "smooth", 15, False)
             for i, sp in enumerate(spans)]
    states, phases = build_path(plans)
    assert phases[cut_frame] == "cut" and states[cut_frame] == right and states[cut_frame - 1] == left
    assert "transition" not in phases[cut_frame:cut_frame + 15]
