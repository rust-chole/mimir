"""Keyless tests for caption / intro / story protection, framing and tracking logic."""
from __future__ import annotations

import copy
import json
import unittest

import pro_edit_fixtures as fx
from ai.editor import captions
from ai.editor.pro_edit.caption_guard import (
    CaptionIntegrity,
    CaptionIntegrityError,
    CaptionWordRef,
    caption_safe_region,
    caption_signature,
    caption_words,
)
from ai.editor.pro_edit.camera import crop_window
from ai.editor.pro_edit.errors import EditPlanTimelineError
from ai.editor.pro_edit.framing import Box, FramingRequest, SubjectBox, crop_contains, solve_framing
from ai.editor.pro_edit.intro_timeline import (
    IntroIntegrityError,
    build_intro_timeline,
    verify_final_duration,
    verify_handoff,
)
from ai.editor.pro_edit.presets import resolve_intro_plan, resolve_plan
from ai.editor.pro_edit.schema import CameraMode, StoryRole
from ai.editor.pro_edit.speaker_link import associate_speakers
from ai.editor.pro_edit.story import ACTION_MIN_VISIBLE_FRACTION, Protection, story_signature
from ai.editor.pro_edit.style import PRO_STREAM_V1
from ai.editor.pro_edit.subjects import SubjectSample, SubjectTrack
from ai.editor.pro_edit.timebase import TimelineDomain as D, Timestamp
from ai.editor.pro_edit.validator import PlanStatus, validate_plan_payload
from ai.editor.pro_edit.vision.detectors import Detection
from ai.editor.pro_edit.vision.frames import SampledFrame
from ai.editor.pro_edit.vision.tracker import MultiSubjectTracker, TrackerConfig

TEASER = {"clip_index": 1, "recommended": True, "edited": {"teaser_start": 13.2, "teaser_end": 14.8, "duration": 1.6}}


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = fx.Workspace()
        self.profile_path = self.ws.write_json("speakers.json", fx.speaker_profile())
        timeline = fx.BASE_TIMELINE["timelines"][0]
        profile = fx.speaker_profile()
        profile["clip_duration"] = 28.0
        self.ass = captions.create_ass_for_clip({}, copy.deepcopy(timeline), self.ws.root / "caps.ass", profile)

    def tearDown(self) -> None:
        self.ws.cleanup()

    def context(self, **kw):
        from ai.editor.pro_edit.context import ContextInputs, build_edit_context
        from ai.editor.pro_edit.timebase import ClipTimelineMap

        timeline = copy.deepcopy(fx.BASE_TIMELINE)
        intro = None
        if kw.pop("with_intro", False):
            intro = build_intro_timeline(TEASER, caption_path=self.ass, clean_duration=28.0, main_duration=28.0,
                                         clip_map=ClipTimelineMap.from_timeline_clip(timeline["timelines"][0]))
        return build_edit_context(ContextInputs(
            timeline_data=timeline, clip_index=1, media=fx.media(),
            speaker_profile_path=self.profile_path,
            video_report_path=self.ws.write_json("report.json", fx.VIDEO_REPORT),
            caption_path=self.ass, intro=intro, **kw))


class CaptionProtectionTests(Base):
    def test_signature_detects_any_truth_change(self) -> None:
        before = CaptionIntegrity.capture(self.profile_path, self.ass)
        before.verify(self.profile_path, self.ass, stage="noop")
        for mutate in (lambda w: w.update(word="CHANGED"), lambda w: w.update(edited_start=w["edited_start"] + 0.01),
                       lambda w: w.update(speaker_raw="B" if w["speaker_raw"] == "A" else "A")):
            profile = fx.speaker_profile()
            mutate(profile["words"][3])
            path = self.ws.write_json("mut.json", profile)
            with self.assertRaises(CaptionIntegrityError):
                before.verify(path, self.ass, stage="mutation")
        self.ass.write_text(self.ass.read_text(encoding="utf-8-sig") + "Dialogue: 0,0:00:01.00,0:00:02.00,ViralMain,,0,0,0,,X\n",
                            encoding="utf-8")
        with self.assertRaises(CaptionIntegrityError):
            before.verify(self.profile_path, self.ass, stage="ass edit")

    def test_context_holds_references_matching_authoritative_profile(self) -> None:
        ctx = self.context()
        words = caption_words(fx.speaker_profile())
        self.assertEqual(ctx.caption_signature, caption_signature(words))
        self.assertEqual({w.id for w in ctx.words}, {w.id for w in words})

    def test_safe_region_parsed_from_real_mimir_ass(self) -> None:
        region = caption_safe_region(self.ass)
        self.assertFalse(region.empty)
        top = region.top(13.0, 14.0)
        # ViralMain MarginV 530 / ViralSecondary 650 at PlayResY 1920, bottom aligned.
        self.assertTrue(0.55 < top < 0.70, top)
        self.assertTrue(all(b.y1 <= 0.76 for b in region.bands))
        self.assertIsNone(region.top(26.0, 27.0))  # no caption on screen there

    def test_replacement_fields_are_fatal(self) -> None:
        ctx = self.context()
        for key in ("replacement_text", "replacement_start", "replacement_end", "replacement_speaker"):
            report = validate_plan_payload(fx.plan(fx.event("e", 13.0, 14.0, **{key: "x"})), ctx, PRO_STREAM_V1)
            self.assertEqual(report.status, PlanStatus.FATAL, key)


class StoryProtectionTests(Base):
    def test_spans_protection_and_signature(self) -> None:
        ctx = self.context()
        payoff = next(s for s in ctx.spans if s.role is StoryRole.PAYOFF)
        self.assertTrue(payoff.source_required and payoff.camera_allowed and not payoff.support_visual_allowed)
        self.assertEqual(payoff.visual_importance, "action")  # impact + peak region in the visual report
        self.assertEqual(payoff.min_visible_fraction, ACTION_MIN_VISIBLE_FRACTION)
        bridge = next(s for s in ctx.spans if s.role is StoryRole.BRIDGE and not s.must_keep)
        self.assertTrue(bridge.protection & Protection.SUPPORT_VISUAL_ALLOWED)
        self.assertEqual(ctx.story_signature, story_signature(ctx.spans, ctx.clip_identity))
        self.assertEqual(self.context().story_signature, ctx.story_signature)  # deterministic

    def test_payoff_action_geometry_limits_crop(self) -> None:
        ctx = self.context()
        report = validate_plan_payload(fx.plan(fx.event("e", 13.0, 14.5, motion="punch_in_fast", intensity=1.0,
                                                        story_span_id="payoff_01")), ctx, PRO_STREAM_V1)
        resolved = resolve_plan(report.plan, ctx, PRO_STREAM_V1)
        for frame in range(resolved.ops[0].start_frame, resolved.ops[0].end_frame):
            self.assertGreaterEqual(1 / resolved.path.state_at(frame).zoom, ACTION_MIN_VISIBLE_FRACTION - 1e-6)

    def test_span_reference_semantics(self) -> None:
        ctx = self.context()
        report = validate_plan_payload(fx.plan(fx.event("e", 12.0, 15.0, story_span_id="payoff_01")), ctx,
                                       PRO_STREAM_V1)
        event = report.plan.events[0]
        self.assertEqual((event.start, event.end, event.story_span_id), (13.0, 15.0, "payoff_01"))
        outside = validate_plan_payload(fx.plan(fx.event("e", 20.0, 22.0, story_span_id="payoff_01")), ctx,
                                        PRO_STREAM_V1)
        self.assertIn("event_outside_story_span", {i.code for i in outside.issues})
        unknown = validate_plan_payload(fx.plan(fx.event("e", 13.0, 14.0, story_span_id="payoff_99")), ctx,
                                        PRO_STREAM_V1)
        self.assertEqual(unknown.plan.events[0].story_span_id, "payoff_01")


class FramingTests(unittest.TestCase):
    def test_face_kept_above_caption_band_with_headroom(self) -> None:
        face = Box.from_center(0.5, 0.5, 0.12, 0.2)
        framing = solve_framing(FramingRequest(1.2, CameraMode.SPEAKER_CLOSE, 1.3, (SubjectBox(face, "face"),),
                                               caption_top=0.62))
        self.assertTrue(framing.caption_clear)
        size = 1 / framing.zoom
        x0, y0, _, _ = crop_window_state(framing)
        self.assertLessEqual((face.y1 - y0) / size, 0.62 + 1e-9)  # chin above the burned captions
        self.assertLess((face.cy - y0) / size, 0.5)              # face sits above center (headroom)
        low = solve_framing(FramingRequest(1.2, CameraMode.SPEAKER_CLOSE, 1.3,
                                           (SubjectBox(Box.from_center(0.5, 0.66, 0.12, 0.2), "face"),),
                                           caption_top=0.62))
        self.assertFalse(low.caption_clear)  # already inside the band at 1x: reported, never hidden
        self.assertIn("caption_overlap_unavoidable", low.notes)

    def test_story_region_beats_face_close_up(self) -> None:
        action = Box(0.62, 0.55, 0.95, 0.95)  # object on the table, away from the face
        framing = solve_framing(FramingRequest(1.3, CameraMode.SPEAKER_CLOSE, 1.3,
                                               (SubjectBox(Box.from_center(0.3, 0.35, 0.1, 0.18), "face"),),
                                               required=(action,)))
        self.assertTrue(crop_contains(framing.zoom, framing.anchor, action))

    def test_dual_subject_keeps_both(self) -> None:
        a, b = Box.from_center(0.3, 0.4, 0.1, 0.18), Box.from_center(0.7, 0.4, 0.1, 0.18)
        framing = solve_framing(FramingRequest(1.2, CameraMode.DUAL_SUBJECT, 1.3,
                                               (SubjectBox(a, "face"), SubjectBox(b, "face"))))
        self.assertTrue(crop_contains(framing.zoom, framing.anchor, a) and crop_contains(framing.zoom, framing.anchor, b))


def crop_window_state(framing):
    from ai.editor.pro_edit.camera import CameraState
    return crop_window(CameraState(framing.zoom, *framing.anchor), 1.0, 1.0)


class IntroProtectionTests(Base):
    def test_intro_clocks_and_handoff(self) -> None:
        intro = build_intro_timeline(TEASER, caption_path=self.ass, clean_duration=28.0, main_duration=28.0)
        self.assertEqual(intro.intro_to_source(Timestamp(0.5, D.INTRO)).seconds, 13.7)
        self.assertEqual(intro.intro_to_source(Timestamp(0.5, D.INTRO)).domain, D.PACED_CLIP)
        with self.assertRaises(EditPlanTimelineError):
            intro.intro_to_source(Timestamp(0.5, D.PACED_CLIP))
        final = intro.main_to_final(Timestamp(intro.main_restart + 1.0, D.PACED_CLIP))
        self.assertAlmostEqual(final.seconds, intro.teaser_duration - intro.transition + 1.0)
        verify_handoff(intro, intro, frame_duration=1 / 30)
        moved = build_intro_timeline({**TEASER, "edited": {**TEASER["edited"], "teaser_start": 12.0}},
                                     caption_path=self.ass, clean_duration=28.0, main_duration=28.0)
        with self.assertRaises(IntroIntegrityError):
            verify_handoff(intro, moved, frame_duration=1 / 30)
        with self.assertRaises(IntroIntegrityError):
            verify_final_duration(intro, intro.expected_final_duration + 0.5, frame_duration=1 / 30)

    def test_intro_directive_rules(self) -> None:
        ctx = self.context(with_intro=True)
        intro = {"camera": "preserve", "motion": "punch_in", "target": {"type": "center_safe", "id": None},
                 "intensity": 0.7, "confidence": 0.9, "reason_code": "hook_emphasis"}
        report = validate_plan_payload(fx.plan(intro=intro), ctx, PRO_STREAM_V1)
        self.assertEqual(report.status, PlanStatus.VALID)
        resolved = resolve_intro_plan(report.plan.intro, ctx, PRO_STREAM_V1)
        a, b = resolved.path.active_ranges()[0]
        fps = ctx.clip.fps
        self.assertGreaterEqual(a, fps.frame_index(13.2))
        self.assertLessEqual(b, fps.frame_index(14.8) + 1)  # camera never leaves the selected teaser
        for key in ("start", "teaser_start"):
            fatal = validate_plan_payload(fx.plan(intro={**intro, key: 1.0}), ctx, PRO_STREAM_V1)
            self.assertEqual(fatal.status, PlanStatus.FATAL, key)
        no_intro = validate_plan_payload(fx.plan(intro=intro), self.context(), PRO_STREAM_V1)
        self.assertIsNone(no_intro.plan.intro)


    def test_intro_camera_keeps_face_out_of_hook_text(self) -> None:
        from ai.editor.pro_edit.intro_timeline import hook_text_band

        class Provider:
            name = "static_test"

            def load(self, duration_s):
                # Face just above the hook text rows at 1x (the camera would push it under them).
                samples = tuple(SubjectSample(round(13.0 + i * 0.1, 3), 0.5, 0.27, 0.07, 0.12, 0.95)
                                for i in range(25))
                return (SubjectTrack("face_1", "face", samples),)

        plain = self.context(with_intro=True, subject_provider=Provider())
        band = hook_text_band({"intro_text": "wait for it", "intro": {"duration": 1.0}}, plain.intro,
                              width=plain.clip.width, height=plain.clip.height)
        with_hook = self.context(with_intro=True, subject_provider=Provider(), hook_band=band)
        intro = {"camera": "speaker_close", "motion": "punch_in", "target": {"type": "dominant_subject", "id": None},
                 "intensity": 0.8, "confidence": 0.9, "reason_code": "hook_emphasis"}
        directive = validate_plan_payload(fx.plan(intro=intro), with_hook, PRO_STREAM_V1).plan.intro

        def face_rows(ctx):
            resolved = resolve_intro_plan(directive, ctx, PRO_STREAM_V1)
            op = resolved.ops[0]
            state = resolved.path.state_at(op.start_frame + op.attack_frames + 1)
            size = 1.0 / state.zoom
            y0 = min(max(state.cy - size / 2, 0.0), 1.0 - size)
            return (0.27 - 0.06 - y0) / size, (0.27 + 0.06 - y0) / size, resolved

        top, bottom, resolved = face_rows(with_hook)
        self.assertTrue(bottom <= band.y0 + 1e-6 or top >= band.y1 - 1e-6, (top, bottom, band))
        free_top, free_bottom, _ = face_rows(plain)
        self.assertTrue(free_top < band.y1 and free_bottom > band.y0)   # without the band it would overlap
        self.assertIsNotNone(with_hook.to_dict()["intro_hook_text_band"])


class SpeakerLinkTests(unittest.TestCase):
    def test_links_only_with_audiovisual_evidence(self) -> None:
        words = []
        for i in range(10):
            spk = "A" if i % 2 == 0 else "B"
            words.append(CaptionWordRef(i, "w", i * 1.0, i * 1.0 + 0.8, spk, "", 0.9))

        def track(tid, speaking_speaker):
            samples = []
            for k in range(100):
                t = k * 0.1
                talking = any(w.speaker_id == speaking_speaker and w.start <= t <= w.end for w in words)
                samples.append(SubjectSample(t, 0.5, 0.4, 0.1, 0.2, 0.9, 0.08 if talking else 0.01, "detected"))
            return SubjectTrack(tid, "face", tuple(samples))

        linked = {t.subject_id: t for t in associate_speakers([track("face_00", "A"), track("face_01", "B")], words)}
        self.assertEqual((linked["face_00"].speaker_id, linked["face_01"].speaker_id), ("A", "B"))
        self.assertGreaterEqual(linked["face_00"].speaker_confidence, 0.5)
        flat = SubjectTrack("face_02", "face", tuple(SubjectSample(k * 0.1, 0.5, 0.4, 0.1, 0.2, 0.9, 0.02, "detected")
                                                      for k in range(100)))
        self.assertIsNone(associate_speakers([flat], words)[0].speaker_id)  # no evidence -> no pretended link


class ScriptedDetector:
    name = "scripted"

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return self.script(self._t)


class TrackerLogicTests(unittest.TestCase):
    def run_tracker(self, boxes_at, n=60, cut_at=None, truth=None):
        import numpy as np

        detector = ScriptedDetector(boxes_at)
        tracker = MultiSubjectTracker(detector, TrackerConfig(detect_every=3))
        rng = np.random.default_rng(7)
        patch = rng.integers(0, 255, (24, 20, 3), dtype=np.uint8)  # textured "face"

        def frames():
            for i in range(n):
                t = i / 10
                detector._t = t
                image = np.full((90, 160, 3), 40 if cut_at is None or t < cut_at else 200, np.uint8)
                for det in (truth or boxes_at)(t):
                    x, y = int(det.x0 * 160), int(det.y0 * 90)
                    image[y:y + 24, x:x + 20] = patch
                yield SampledFrame(i, t, image)
        return tracker.run(frames()), tracker

    @fx.needs_opencv
    def test_identity_survives_missed_detections_and_is_sparse(self) -> None:
        def boxes(t):
            if 2.0 <= t < 3.0:
                return []  # detector misses the face for 1 s
            return truth(t)

        def truth(t):
            cx = 0.3 + 0.02 * t
            return [Detection(cx - 0.0625, 0.3, cx + 0.0625, 0.5667, 0.9)]
        tracks, tracker = self.run_tracker(boxes, truth=truth)
        gap = [s for s in tracks[0].samples if 2.0 <= s.t < 3.0]
        self.assertTrue(gap and all(s.source == "tracked" for s in gap))  # optical flow bridged the miss
        for sample in gap:
            self.assertAlmostEqual(sample.cx, 0.3 + 0.02 * sample.t, delta=0.02)
        self.assertEqual(len(tracks), 1)
        self.assertLess(tracker.stats["detector_runs"], 40)  # sparse: not every sampled frame
        xs = [s.cx for s in tracks[0].samples]
        self.assertAlmostEqual(xs[-1], 0.3 + 0.02 * 5.9, delta=0.02)

    @fx.needs_opencv
    def test_two_subjects_do_not_swap(self) -> None:
        def boxes(t):
            return [Detection(0.10, 0.3, 0.20, 0.5, 0.9), Detection(0.70, 0.3, 0.80, 0.5, 0.9)]
        tracks, _ = self.run_tracker(boxes)
        self.assertEqual(len(tracks), 2)
        left = [t for t in tracks if t.samples[0].cx < 0.5][0]
        self.assertTrue(all(s.cx < 0.5 for s in left.samples))

    @fx.needs_opencv
    def test_scene_cut_forces_redetection(self) -> None:
        tracks, tracker = self.run_tracker(lambda t: [Detection(0.4, 0.3, 0.5, 0.5, 0.9)], cut_at=3.05)
        self.assertGreaterEqual(tracker.stats["scene_cuts"], 1)
        self.assertEqual(len(tracks), 1)  # reacquired the same subject after the cut


if __name__ == "__main__":
    unittest.main()
