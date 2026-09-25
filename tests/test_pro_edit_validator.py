from __future__ import annotations

import json
import unittest

import pro_edit_fixtures as fx
from ai.editor.pro_edit.schema import (
    CameraMode,
    EditEvent,
    MotionPreset,
    StoryRole,
    SupportVisual,
    TargetType,
    planner_json_schema,
)
from ai.editor.pro_edit.style import PRO_STREAM_V1
from ai.editor.pro_edit.validator import PlanStatus, Severity, validate_plan, validate_plan_payload


class ValidatorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = fx.Workspace()
        self.ctx = fx.make_context(self.ws)

    def tearDown(self) -> None:
        self.ws.cleanup()

    def run_plan(self, *events: dict, **top: object):
        return validate_plan_payload(fx.plan(*events, **top), self.ctx, PRO_STREAM_V1, planner="test")

    def codes(self, report) -> set[str]:
        return {issue.code for issue in report.issues}

    def only(self, report) -> EditEvent:
        self.assertIsNotNone(report.plan)
        self.assertEqual(len(report.plan.events), 1)
        return report.plan.events[0]


class SchemaTests(ValidatorTestCase):
    def test_valid_plan_accepted(self) -> None:
        report = self.run_plan(fx.event("e1", 13.0, 14.2))
        self.assertEqual(report.status, PlanStatus.VALID)
        self.assertEqual(report.events_valid, 1)
        self.assertEqual(self.only(report).motion, MotionPreset.PUNCH_IN)

    def test_missing_field_rejected(self) -> None:
        bad = fx.event("e1", 13.0, 14.2)
        del bad["motion"]
        report = self.run_plan(bad)
        self.assertEqual(report.events_rejected, 1)
        self.assertIn("missing_required_field", self.codes(report))
        self.assertEqual(report.plan.events, ())

    def test_unknown_enum_rejected_as_static(self) -> None:
        report = self.run_plan(fx.event("e1", 13.0, 14.2, motion="spin_360"))
        self.assertIn("unknown_motion_static_clean", self.codes(report))
        self.assertEqual(report.events_rejected, 1)
        self.assertEqual(report.plan.camera_events, ())
        self.assertIn("unknown_role", self.codes(self.run_plan(fx.event("e1", 13.0, 14.2, role="climax"))))

    def test_wrong_schema_version_fatal(self) -> None:
        for version in (1, 3, "2", True, None):
            report = self.run_plan(fx.event("e1", 13.0, 14.2), schema_version=version)
            self.assertEqual(report.status, PlanStatus.FATAL, msg=repr(version))
            self.assertIsNone(report.plan)

    def test_wrong_timeline_domain_fatal(self) -> None:
        report = self.run_plan(fx.event("e1", 13.0, 14.2), timeline_domain="vod")
        self.assertEqual(report.status, PlanStatus.FATAL)
        self.assertIn("timeline_domain_mismatch", self.codes(report))

    def test_style_mismatch_fatal(self) -> None:
        self.assertEqual(self.run_plan(style_pack="other").status, PlanStatus.FATAL)

    def test_numeric_types(self) -> None:
        self.assertIn("numeric_type_invalid", self.codes(self.run_plan(fx.event("e1", "13", 14.2))))
        self.assertIn("numeric_type_invalid", self.codes(self.run_plan(fx.event("e1", 13.0, 14.2, intensity=True))))
        report = self.run_plan(fx.event("e1", 13.0, 14.2, intensity=float("nan")))
        self.assertIn("non_finite_number", self.codes(report))
        self.assertIn("non_finite_number", self.codes(self.run_plan(fx.event("e1", 13.0, float("inf")))))

    def test_duplicate_event_ids(self) -> None:
        report = self.run_plan(fx.event("e1", 13.0, 14.2), fx.event("e1", 16.3, 17.3, role="reaction"))
        self.assertIn("duplicate_event_id", self.codes(report))
        self.assertEqual(len(report.plan.events), 1)

    def test_plan_json_schema_has_no_truth_fields(self) -> None:
        schema = planner_json_schema("pro_stream_v1")
        props = schema["properties"]["events"]["items"]["properties"]
        for forbidden in ("text", "caption", "words", "scale", "zoom", "width", "speaker_label", "payoff_start"):
            self.assertNotIn(forbidden, props)
        self.assertFalse(schema["properties"]["events"]["items"]["additionalProperties"])
        json.dumps(schema)  # serializable


class TimeTests(ValidatorTestCase):
    def test_negative_start_rejected(self) -> None:
        self.assertIn("time_negative_start", self.codes(self.run_plan(fx.event("e1", -1.0, 2.0, role="hook"))))

    def test_end_not_after_start_rejected(self) -> None:
        self.assertIn("time_end_not_after_start", self.codes(self.run_plan(fx.event("e1", 14.0, 14.0))))
        self.assertIn("time_end_not_after_start", self.codes(self.run_plan(fx.event("e1", 14.0, 13.0))))

    def test_out_of_range_end_rejected(self) -> None:
        self.assertIn("time_end_out_of_range", self.codes(self.run_plan(fx.event("e1", 27.0, 28.5, role="bridge"))))

    def test_floating_point_boundary_tolerated(self) -> None:
        duration = self.ctx.clip.duration_s
        report = self.run_plan(fx.event("e1", 26.0, duration + 1e-7, role="bridge", motion="slow_push",
                                        intensity=0.3, reason_code="bridge_continuity"),
                               fx.event("e2", -1e-7, 1.5, role="hook", motion="slow_push", intensity=0.4,
                                        reason_code="hook_emphasis"))
        self.assertEqual(report.events_rejected, 0, report.error_summary())
        ends = {e.event_id: (e.start, e.end) for e in report.plan.events}
        self.assertLessEqual(ends["e1"][1], duration)
        self.assertEqual(ends["e2"][0], 0.0)

    def test_event_before_visible_main_rejected(self) -> None:
        ctx = fx.make_context(self.ws, visible_start=5.0)
        report = validate_plan_payload(fx.plan(fx.event("e1", 1.0, 4.0, role="setup", motion="slow_push")), ctx,
                                       PRO_STREAM_V1)
        self.assertIn("time_not_visible", self.codes(report))


class TargetAndMotionTests(ValidatorTestCase):
    def test_known_subject_kept_unknown_falls_back(self) -> None:
        from ai.editor.pro_edit.subjects import SubjectSample, SubjectTrack

        class Provider:
            name = "test"

            def load(self, duration_s):
                samples = tuple(SubjectSample(t / 10, 0.6, 0.4, 0.2, 0.3, 0.9) for t in range(0, 280))
                return (SubjectTrack("face_0", "face", samples, speaker_id="A"),)

        ctx = fx.make_context(self.ws, provider=Provider())
        good = validate_plan_payload(fx.plan(fx.event("e1", 13.0, 14.2, camera="speaker_close",
                                                      target={"type": "subject", "id": "face_0"})), ctx, PRO_STREAM_V1)
        self.assertEqual(good.plan.events[0].target.type, TargetType.SUBJECT)
        bad = validate_plan_payload(fx.plan(fx.event("e1", 13.0, 14.2, target={"type": "subject", "id": "face_9"})),
                                    ctx, PRO_STREAM_V1)
        self.assertEqual(bad.plan.events[0].target.type, TargetType.CENTER_SAFE)
        self.assertIn("unknown_subject_center_safe", self.codes(bad))

    def test_no_subject_evidence_center_safe(self) -> None:
        report = self.run_plan(fx.event("e1", 13.0, 14.2, target={"type": "subject", "id": "speaker_B"}))
        self.assertEqual(self.only(report).target.type, TargetType.CENTER_SAFE)

    def test_intensity_bounds(self) -> None:
        for value, expected in ((0.0, 0.0), (1.0, 1.0), (-0.2, 0.0), (1.05, 1.0)):
            report = self.run_plan(fx.event("e1", 13.0, 14.2, intensity=value))
            self.assertAlmostEqual(self.only(report).intensity, expected, msg=repr(value))
        self.assertIn("intensity_clamped", self.codes(self.run_plan(fx.event("e1", 13.0, 14.2, intensity=1.05))))

    def test_render_parameters_ignored(self) -> None:
        report = self.run_plan(fx.event("e1", 13.0, 14.2, scale=1.387))
        self.assertIn("render_parameter_ignored", self.codes(report))
        self.assertNotIn("scale", self.only(report).to_dict())

    def test_confidence_gates(self) -> None:
        low = self.only(self.run_plan(fx.event("e1", 13.0, 14.2, confidence=0.2)))
        self.assertEqual(low.motion, MotionPreset.STATIC_CLEAN)
        mid = self.only(self.run_plan(fx.event("e1", 13.0, 14.2, confidence=0.5, motion="punch_in_fast",
                                               intensity=0.9)))
        self.assertEqual(mid.motion, MotionPreset.SLOW_PUSH)
        self.assertLessEqual(mid.intensity, PRO_STREAM_V1.confidence.mild_intensity_cap)

    def test_role_owned_by_story_and_policy_downgrade(self) -> None:
        # Planner claims payoff during setup and asks for a strong punch.
        event = self.only(self.run_plan(fx.event("e1", 3.0, 4.5, role="payoff", motion="punch_in_fast")))
        self.assertEqual(event.role, StoryRole.SETUP)
        self.assertNotIn(event.motion, {MotionPreset.PUNCH_IN, MotionPreset.PUNCH_IN_FAST})

    def test_reaction_close_requires_evidence(self) -> None:
        event = self.only(self.run_plan(fx.event("e1", 7.0, 8.5, role="escalation", camera="reaction_close")))
        self.assertNotEqual(event.camera, CameraMode.REACTION_CLOSE)


class OverlapAndDensityTests(ValidatorTestCase):
    def test_compatible_channel_overlap_accepted(self) -> None:
        report = self.run_plan(
            fx.event("cam", 16.3, 17.6, role="reaction", camera="reaction_close", sfx="impact_soft",
                     reason_code="reaction_hold"))
        event = self.only(report)
        self.assertEqual(event.camera, CameraMode.REACTION_CLOSE)
        self.assertEqual(event.sfx.value, "impact_soft")

    def test_conflicting_camera_overlap_resolved_by_priority(self) -> None:
        report = self.run_plan(
            fx.event("payoff", 13.0, 15.0, motion="punch_in", confidence=0.9),
            fx.event("focus", 14.0, 16.8, role="payoff", motion="speaker_focus", camera="speaker_close",
                     confidence=0.7),
        )
        camera = [e for e in report.plan.events if e.is_camera_active]
        for a in camera:
            for b in camera:
                if a is not b:
                    self.assertFalse(a.start < b.end and b.start < a.end, f"{a.event_id} overlaps {b.event_id}")
        by_id = {e.event_id: e for e in report.plan.events}
        self.assertEqual((by_id["payoff"].start, by_id["payoff"].end), (13.0, 15.0))
        self.assertTrue({"camera_overlap_trimmed", "camera_overlap_rejected"} & self.codes(report))

    def test_strong_effect_density_limited(self) -> None:
        events = [fx.event(f"p{i}", 13.0 + i * 0.5, 13.4 + i * 0.5, motion="punch_in_fast") for i in range(6)]
        report = self.run_plan(*events)
        strong = [e for e in report.plan.events if e.is_strong]
        self.assertLessEqual(len(strong), report.budget["max_strong_motion_events"])
        for a in strong:
            for b in strong:
                if a is not b:
                    self.assertGreaterEqual(max(a.start - b.end, b.start - a.end),
                                            PRO_STREAM_V1.density.min_gap_between_strong_s - 1e-9)

    def test_validator_is_idempotent(self) -> None:
        first = self.run_plan(
            fx.event("a", 1.0, 2.0, role="hook", motion="punch_in", reason_code="hook_emphasis"),
            fx.event("b", 7.0, 9.0, role="escalation", motion="slow_push", intensity=0.5),
            fx.event("c", 13.0, 14.5), fx.event("d", 14.0, 17.0, role="reaction", motion="speaker_focus"),
            fx.event("e", 16.3, 17.6, role="reaction", camera="reaction_close", motion="punch_in"),
        )
        second = validate_plan(first.plan, self.ctx, PRO_STREAM_V1)
        self.assertEqual(second.status, PlanStatus.VALID, second.error_summary() + [i.code for i in second.issues])
        self.assertEqual(second.plan.to_dict()["events"], first.plan.to_dict()["events"])


class ProtectedRangeAndCaptionTests(ValidatorTestCase):
    def test_payoff_replacement_blocked(self) -> None:
        report = self.run_plan(fx.event("e1", 13.0, 14.2, support_visual="broll"))
        self.assertEqual(self.only(report).support_visual, SupportVisual.NONE)
        codes = self.codes(report)
        self.assertTrue({"payoff_source_visibility_protected", "support_visual_not_allowed_for_role",
                         "story_span_source_required"} & codes)

    def test_support_visual_outside_protected_allowed(self) -> None:
        report = self.run_plan(fx.event("e1", 20.0, 22.0, role="bridge", motion="static_clean", support_visual="meme",
                                        reason_code="bridge_continuity"))
        self.assertEqual(self.only(report).support_visual, SupportVisual.MEME)

    def test_valid_emphasis_word_ids_accepted(self) -> None:
        report = self.run_plan(fx.event("e1", 13.0, 14.2, emphasis_word_ids=[7, 8]))
        self.assertEqual(self.only(report).emphasis_word_ids, (7, 8))

    def test_unknown_word_ids_rejected(self) -> None:
        report = self.run_plan(fx.event("e1", 13.0, 14.2, emphasis_word_ids=[8, 999, 0]))
        self.assertEqual(self.only(report).emphasis_word_ids, (8,))
        self.assertIn("caption_unknown_word_id", self.codes(report))
        self.assertIn("caption_word_outside_event", self.codes(report))

    def test_caption_text_mutation_is_fatal(self) -> None:
        for key, value in (("caption", "THIS IS CRAZY"), ("text", "x"), ("word_timings", []),
                           ("speaker_label", "KAI")):
            report = self.run_plan(fx.event("e1", 13.0, 14.2, **{key: value}))
            self.assertEqual(report.status, PlanStatus.FATAL, key)
            self.assertIn("truth_mutation_attempt", self.codes(report))
        self.assertEqual(self.run_plan(transcript=[]).status, PlanStatus.FATAL)

    def test_command_and_geometry_injection_fatal(self) -> None:
        self.assertEqual(self.run_plan(fx.event("e1", 13.0, 14.2, ffmpeg="-vf crop")).status, PlanStatus.FATAL)
        self.assertEqual(self.run_plan(fx.event("e1", 13.0, 14.2), output_profile="9:16").status, PlanStatus.FATAL)
        issues = self.run_plan(fx.event("e1", 13.0, 14.2, aspect_ratio=0.5625)).issues
        self.assertTrue(any(i.severity is Severity.FATAL and i.code == "output_geometry_mutation" for i in issues))


if __name__ == "__main__":
    unittest.main()
