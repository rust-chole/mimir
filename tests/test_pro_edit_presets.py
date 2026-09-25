from __future__ import annotations

import itertools
import math
import unittest

import pro_edit_fixtures as fx
from ai.editor.pro_edit.camera import (
    HARD_MAX_ZOOM,
    IDENTITY,
    CameraPath,
    CameraSegment,
    CameraState,
    OutputProfile,
    base_window,
    crop_window,
    effective_center,
    verify_path,
)
from ai.editor.pro_edit.errors import PresetResolutionError
from ai.editor.pro_edit.presets import allocate_ramps, resolve_plan
from ai.editor.pro_edit.schema import MotionPreset
from ai.editor.pro_edit.style import PRO_STREAM_V1, CameraMotionLimits
from ai.editor.pro_edit.subjects import (
    SubjectSample,
    SubjectTrack,
    smooth_center_path,
    stable_target_segments,
)
from ai.editor.pro_edit.validator import validate_plan_payload

MOTIONS = ["slow_push", "punch_in", "punch_in_fast", "speaker_focus"]


class TrackProvider:
    name = "test"

    def __init__(self, tracks):
        self.tracks = tuple(tracks)

    def load(self, duration_s):
        return self.tracks


def face_track(cx=0.7, cy=0.35, conf=0.9, jitter=0.0, subject_id="face_0", speaker="A", w=0.12, h=0.2):
    samples = []
    for i in range(0, 281):
        t = i / 10
        dx = jitter * math.sin(i * 1.7)
        samples.append(SubjectSample(t, min(1, max(0, cx + dx)), cy, w, h, conf))
    return SubjectTrack(subject_id, "face", tuple(samples), speaker_id=speaker, speaker_confidence=0.9)


class PresetMathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = fx.Workspace()

    def tearDown(self) -> None:
        self.ws.cleanup()

    def resolve(self, events, provider=None, ctx=None):
        ctx = ctx or fx.make_context(self.ws, provider=provider)
        report = validate_plan_payload(fx.plan(*events), ctx, PRO_STREAM_V1)
        self.assertFalse(report.fatal, report.error_summary())
        return ctx, report, resolve_plan(report.plan, ctx, PRO_STREAM_V1)

    def assert_invariants(self, ctx, resolved):
        W, H = ctx.clip.width, ctx.clip.height
        for frame in range(resolved.frame_count):
            state = resolved.path.state_at(frame)
            self.assertGreaterEqual(state.zoom, 1.0 - 1e-9)
            self.assertLessEqual(state.zoom, PRO_STREAM_V1.zoom.max_zoom + 1e-9)
            self.assertTrue(0.0 <= state.cx <= 1.0 and 0.0 <= state.cy <= 1.0)
            x, y, w, h = crop_window(state, W, H)
            self.assertGreaterEqual(x, -1e-6)
            self.assertGreaterEqual(y, -1e-6)
            self.assertLessEqual(x + w, W + 1e-6)
            self.assertLessEqual(y + h, H + 1e-6)
        for op in resolved.ops:
            self.assertGreater(op.attack_frames, 0)
            self.assertGreaterEqual(op.end_frame - op.start_frame, op.attack_frames + op.hold_frames)
            self.assertEqual(op.attack_frames + op.hold_frames + op.release_frames, op.end_frame - op.start_frame)
        # Identity at the very first and last frame: no residual transform.
        self.assertTrue(resolved.path.state_at(0).close_to(IDENTITY) or resolved.ops[0].start_frame == 0)
        self.assertTrue(resolved.path.state_at(resolved.frame_count - 1).close_to(IDENTITY))

    def test_every_preset_bounded_at_intensity_extremes(self) -> None:
        for motion, intensity in itertools.product(MOTIONS, (0.0, 0.5, 1.0)):
            with self.subTest(motion=motion, intensity=intensity):
                ctx, _, resolved = self.resolve([fx.event("e", 13.0, 15.0, motion=motion, intensity=intensity,
                                                          camera="speaker_close" if motion == "speaker_focus"
                                                          else "preserve")])
                self.assertEqual(len(resolved.ops), 1, resolved.dropped)
                self.assert_invariants(ctx, resolved)
                bounds = PRO_STREAM_V1.presets[MotionPreset(motion)]
                peak = resolved.ops[0].params.scale_peak
                self.assertLessEqual(peak, min(bounds.scale.high if motion != "speaker_focus" else 1.30,
                                               PRO_STREAM_V1.zoom.protected_center_safe_max) + 1e-9)

    def test_punch_in_mapping_matches_spec(self) -> None:
        # Direct resolver check (bypasses role intensity caps): bridge region, center-safe cap 1.30.
        from ai.editor.pro_edit.schema import (CameraMode, EditEvent, EditPlan, EditTarget, ReasonCode,
                                               StoryRole, TargetType)
        from ai.editor.pro_edit.timebase import TimelineDomain
        ctx = fx.make_context(self.ws)
        for motion, intensity, peak, attack in (("punch_in", 0.0, 1.10, 0.35), ("punch_in", 1.0, 1.20, 0.16),
                                                ("punch_in_fast", 0.0, 1.14, 0.24), ("punch_in_fast", 1.0, 1.24, 0.10),
                                                ("slow_push", 0.0, 1.04, None), ("slow_push", 1.0, 1.10, None)):
            event = EditEvent("e", 20.0, 22.0, StoryRole.BRIDGE, CameraMode.PRESERVE, MotionPreset(motion),
                              EditTarget(TargetType.CENTER_SAFE), intensity, 0.9, ReasonCode.PAYOFF_HIT)
            plan = EditPlan(1, "pro_stream_v1", TimelineDomain.PACED_CLIP, (event,), "test")
            resolved = resolve_plan(plan, ctx, PRO_STREAM_V1)
            params = resolved.ops[0].params
            with self.subTest(motion=motion, intensity=intensity):
                self.assertAlmostEqual(params.scale_peak, peak, places=6)
                if attack is not None:
                    self.assertAlmostEqual(params.attack_s, attack, delta=ctx.clip.fps.frame_duration)
                self.assert_invariants(ctx, resolved)

    def test_protected_payoff_center_safe_cap(self) -> None:
        _, _, resolved = self.resolve([fx.event("e", 13.0, 14.5, motion="punch_in_fast", intensity=1.0)])
        self.assertLessEqual(resolved.ops[0].params.scale_peak, PRO_STREAM_V1.zoom.protected_center_safe_max)

    def test_static_clean_is_identity(self) -> None:
        _, _, resolved = self.resolve([fx.event("e", 3.0, 5.0, role="setup", motion="static_clean")])
        self.assertTrue(resolved.path.is_identity)
        self.assertEqual(resolved.ops, ())

    def test_no_transform_discontinuity_between_adjacent_events(self) -> None:
        ctx, _, resolved = self.resolve([
            fx.event("a", 13.0, 14.0, motion="punch_in", intensity=0.6),
            fx.event("b", 14.05, 16.0, motion="slow_push", intensity=0.6),
        ])
        self.assertTrue(resolved.ops[0].params.merged_with_next)
        verify_path(resolved.path, max_zoom=PRO_STREAM_V1.zoom.max_zoom, fps=ctx.clip.fps.fps)
        zooms = [resolved.path.state_at(f).zoom for f in range(resolved.frame_count)]
        steps = [abs(b - a) for a, b in zip(zooms, zooms[1:])]
        self.assertLess(max(steps), 0.2)

    def test_determinism(self) -> None:
        events = [fx.event("a", 13.0, 14.0), fx.event("b", 16.3, 17.6, role="reaction", camera="reaction_close")]
        _, _, first = self.resolve(events)
        _, _, second = self.resolve(events)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.path.sample(), second.path.sample())

    def test_subject_targeted_punch_respects_safety_box(self) -> None:
        provider = TrackProvider([face_track(cx=0.7, cy=0.5, w=0.3, h=0.5)])
        ctx, _, resolved = self.resolve([fx.event("e", 16.3, 17.6, role="reaction", camera="reaction_close",
                                                  intensity=1.0, target={"type": "subject", "id": "face_0"})],
                                        provider=provider)
        params = resolved.ops[0].params
        self.assertTrue(params.target_reliable)
        self.assertGreater(params.anchor_x, 0.5)
        # Face h=0.5 with 0.45/0.35 forehead/chin margins needs 0.9 of the height -> zoom <= 1/0.9.
        self.assertLessEqual(params.scale_peak, 1 / 0.9 + 1e-6)
        op = resolved.ops[0]
        for frame in range(op.start_frame, op.end_frame):
            x, y, w, h = crop_window(resolved.path.state_at(frame), 1.0, 1.0)
            self.assertLessEqual(y, 0.5 - 0.25 - 0.45 * 0.5 + 1e-6, "forehead margin cut")
            self.assertGreaterEqual(y + h, 0.5 + 0.25 + 0.35 * 0.5 - 1e-6, "chin margin cut")
            self.assertLessEqual(x, 0.7 - 0.15 - 0.3 * 0.3 + 1e-6)
            self.assertGreaterEqual(x + w, 0.7 + 0.15 + 0.3 * 0.3 - 1e-6)
        self.assert_invariants(ctx, resolved)

    def test_face_at_frame_edge_uses_available_frame(self) -> None:
        provider = TrackProvider([face_track(cx=0.75, cy=0.3, w=0.3, h=0.5)])
        _, _, resolved = self.resolve([fx.event("e", 16.3, 17.6, role="reaction", camera="reaction_close",
                                                intensity=1.0, target={"type": "subject", "id": "face_0"})],
                                      provider=provider)
        op = resolved.ops[0]
        x, y, w, h = crop_window(resolved.path.state_at(op.start_frame + op.attack_frames), 1.0, 1.0)
        self.assertAlmostEqual(y, 0.0, places=6)  # clamped to the top edge, forehead kept
        self.assertGreaterEqual(y + h, 0.3 + 0.25 + 0.35 * 0.5 - 1e-6)

    def test_low_confidence_subject_falls_back_to_center(self) -> None:
        provider = TrackProvider([face_track(cx=0.9, conf=0.2)])
        ctx, _, resolved = self.resolve([fx.event("e", 16.3, 17.6, role="reaction", camera="reaction_close",
                                                  target={"type": "subject", "id": "face_0"})], provider=provider)
        params = resolved.ops[0].params
        self.assertFalse(params.target_reliable)
        self.assertEqual((params.anchor_x, params.anchor_y), (0.5, 0.5))

    def test_speaker_focus_velocity_and_dead_zone(self) -> None:
        def run(jitter: float):
            provider = TrackProvider([face_track(cx=0.7, jitter=jitter)])
            return self.resolve([fx.event("e", 7.0, 9.0, role="escalation", motion="speaker_focus",
                                          camera="speaker_medium", target={"type": "subject", "id": "face_0"})],
                                provider=provider)
        ctx, _, steady = run(0.0)
        _, _, jittery = run(0.02)
        for resolved in (steady, jittery):
            self.assertLessEqual(resolved.metrics["max_follow_speed"],
                                 PRO_STREAM_V1.camera.max_speed_norm_per_s * 1.05 + 1e-9)
            self.assert_invariants(ctx, resolved)
        op = steady.ops[0]
        late = steady.path.state_at(op.start_frame + op.attack_frames + op.hold_frames - 1)
        self.assertGreater(effective_center(late)[0], 0.53)  # crop pans toward the subject (reachable range)
        diff = max(abs(steady.path.state_at(f).cx - jittery.path.state_at(f).cx)
                   for f in range(op.start_frame, op.end_frame))
        self.assertLess(diff, 0.02)  # +-0.02 subject noise does not become camera shake

    def test_dead_zone_holds_camera_on_small_motion(self) -> None:
        limits = PRO_STREAM_V1.camera
        samples = [SubjectSample(i / 10, 0.7 + 0.02 * math.sin(i * 1.7), 0.4, 0.1, 0.1, 0.9) for i in range(40)]
        path = smooth_center_path(samples, [i / 30 for i in range(120)], limits, initial=(0.7, 0.4))
        self.assertTrue(all(abs(cx - 0.7) < 1e-9 and abs(cy - 0.4) < 1e-9 for cx, cy in path))

    def test_multi_speaker_focus_uses_hysteresis(self) -> None:
        words = [("a", 6.0 + i * 0.3, 6.0 + i * 0.3 + 0.25, "A") for i in range(5)]      # A 6.0-7.5
        words += [("uh", 7.5, 7.75, "B")]                                                  # short B interjection
        words += [("a2", 7.8, 8.0, "A")]
        words += [("b", 8.1 + i * 0.3, 8.1 + i * 0.3 + 0.25, "B") for i in range(7)]       # B takes over 8.1-10.1
        provider = TrackProvider([face_track(cx=0.3, subject_id="face_A", speaker="A"),
                                  face_track(cx=0.7, subject_id="face_B", speaker="B")])
        ctx = fx.make_context(self.ws, provider=provider, words=words)
        from ai.editor.pro_edit.schema import (CameraMode, EditEvent, EditPlan, EditTarget, ReasonCode,
                                               StoryRole, TargetType)
        from ai.editor.pro_edit.timebase import TimelineDomain
        event = EditEvent("e", 6.0, 10.4, StoryRole.ESCALATION, CameraMode.SPEAKER_MEDIUM, MotionPreset.SPEAKER_FOCUS,
                          EditTarget(TargetType.ACTIVE_SPEAKER), 0.6, 0.9, ReasonCode.SPEAKER_SHIFT)
        resolved = resolve_plan(EditPlan(2, "pro_stream_v1", TimelineDomain.PACED_CLIP, (event,), "test"), ctx,
                                PRO_STREAM_V1)
        op = resolved.ops[0]
        self.assertTrue(any(n.startswith("active_speaker_hysteresis:A>B") for n in op.params.notes), op.params.notes)
        fps = ctx.clip.fps
        during_interjection = resolved.path.state_at(fps.frame_index(7.7)).cx
        late = resolved.path.state_at(fps.frame_index(10.0)).cx
        self.assertLess(during_interjection, 0.48)   # stays on A (left) during a 0.25 s interjection
        self.assertGreater(late, 0.52)  # sustained B speech moves the camera to B (right)
        self.assertLessEqual(resolved.metrics["max_follow_speed"],
                             PRO_STREAM_V1.camera.max_speed_norm_per_s * 1.05 + 1e-9)
        self.assert_invariants(ctx, resolved)

    def test_output_profiles_are_explicit_and_even(self) -> None:
        self.assertEqual(base_window(1920, 1080, OutputProfile.PRESERVE), (0, 0, 1920, 1080))
        for profile, aspect in ((OutputProfile.PORTRAIT_9_16, 9 / 16), (OutputProfile.SQUARE_1_1, 1.0),
                                (OutputProfile.LANDSCAPE_16_9, 16 / 9)):
            bx, by, bw, bh = base_window(1920, 1080, profile)
            self.assertEqual((bw % 2, bh % 2, bx % 2, by % 2), (0, 0, 0, 0))
            self.assertAlmostEqual(bw / bh, aspect, delta=0.01)
            self.assertLessEqual(bx + bw, 1920)
            self.assertLessEqual(by + bh, 1080)


class CameraMathTests(unittest.TestCase):
    def test_crop_window_clamped_for_all_anchors(self) -> None:
        for z, cx, cy in itertools.product((1.0, 1.1, 1.35), (0.0, 0.3, 1.0), (0.0, 0.7, 1.0)):
            x, y, w, h = crop_window(CameraState(z, cx, cy), 1920, 1080)
            self.assertAlmostEqual(w, 1920 / z)
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x + w, 1920 + 1e-9)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(y + h, 1080 + 1e-9)
        # Zoom above the hard limit is clamped.
        _, _, w, _ = crop_window(CameraState(2.0, 0.5, 0.5), 100, 100)
        self.assertAlmostEqual(w, 100 / HARD_MAX_ZOOM)
        self.assertEqual(effective_center(CameraState(1.2, 1.0, 0.0)), (1 - 0.5 / 1.2, 0.5 / 1.2))

    def test_verify_path_detects_jump(self) -> None:
        peak = CameraState(1.2, 0.5, 0.5)
        bad = CameraPath(30, (CameraSegment(10, 20, peak, peak, "linear", "x", "hold"),))
        with self.assertRaises(PresetResolutionError):
            verify_path(bad, max_zoom=1.3, fps=30.0)
        with self.assertRaises(PresetResolutionError):
            CameraPath(30, (CameraSegment(0, 10, IDENTITY, peak), CameraSegment(5, 12, peak, IDENTITY)))

    def test_allocate_ramps(self) -> None:
        self.assertEqual(allocate_ramps(30, 6, 6, 4), (6, 18, 6))
        attack, hold, release = allocate_ramps(8, 10, 8, 3)
        self.assertEqual(attack + hold + release, 8)
        self.assertGreaterEqual(attack, 2)
        with self.assertRaises(PresetResolutionError):
            allocate_ramps(3, 2, 2, 0)


class SmoothingTests(unittest.TestCase):
    limits = CameraMotionLimits()

    def test_low_confidence_holds_previous_target(self) -> None:
        samples = [SubjectSample(t / 10, 0.8, 0.5, 0.1, 0.1, 0.9 if t < 10 else 0.1) for t in range(30)]
        times = [t / 30 for t in range(0, 90)]
        path = smooth_center_path(samples, times, self.limits)
        at_one = path[30][0]
        just_after = path[45][0]
        self.assertGreaterEqual(just_after, at_one - 1e-9)  # holds / keeps approaching, no jump to center
        self.assertGreater(just_after, 0.55)
        # Sustained low confidence beyond timeout drifts back toward center, speed-limited.
        self.assertLess(path[-1][0], just_after)
        for a, b in zip(path, path[1:]):
            self.assertLessEqual(math.hypot(b[0] - a[0], b[1] - a[1]), self.limits.max_speed_norm_per_s / 30 + 1e-9)

    def test_hysteresis_prevents_ping_pong(self) -> None:
        obs = []
        for i in range(60):  # A dominant with 0.5 s B interjections
            t = i * 0.1
            candidate = "B" if 20 <= i < 25 else "A"
            obs.append((t, candidate, 0.9))
        for i in range(60, 90):
            obs.append((i * 0.1, "B", 0.9))
        changes = stable_target_segments(obs, self.limits)
        self.assertEqual([c[1] for c in changes], ["A", "B"])
        self.assertAlmostEqual(changes[1][0], 6.0, places=6)


if __name__ == "__main__":
    unittest.main()
