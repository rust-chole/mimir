"""MIMIR V6 camera: evidence-directed shot intent, story-safe framing, stability,
explained HOLDs and pixel proof that the plan reached the rendered frames.

Synthetic evidence only (no names, no fixture coordinates in production code):
every scenario is built from generic face tracks, speaker words and story spans.
"""
from __future__ import annotations

import copy
import math
import shutil
import statistics
import subprocess
import tempfile
import unittest
from pathlib import Path

import pro_edit_fixtures as fx
from ai.editor.pro_edit import render_proof
from ai.editor.pro_edit.context import visual_report_body
from ai.editor.pro_edit.direction import ShotIntent, direct_plan, explain_holds
from ai.editor.pro_edit.framing import Box, crop_contains
from ai.editor.pro_edit.presets import resolve_plan
from ai.editor.pro_edit.schema import CameraMode, MotionPreset, TargetType
from ai.editor.pro_edit.style import PRO_STREAM_V1
from ai.editor.pro_edit.subjects import SubjectSample, SubjectTrack
from ai.editor.pro_edit.validator import validate_plan, validate_plan_payload
from ai.editor.pro_edit.vision.tracker import consolidate_tracks

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
STYLE = PRO_STREAM_V1


class Provider:
    name = "test"

    def __init__(self, tracks, regions=()):
        self.tracks = tuple(tracks)
        self.regions = tuple(regions)

    def load(self, duration_s):
        return self.tracks

    def load_regions(self, duration_s):
        return self.regions


def face(subject_id: str, cx: float, *, speaker: str | None, link: float = 0.9, cy: float = 0.4, w: float = 0.1,
         h: float = 0.16, conf: float = 0.9, jitter: float = 0.0, t0: float = 0.0, t1: float = 28.0) -> SubjectTrack:
    samples = []
    for i in range(int(round(t0 * 10)), int(round(t1 * 10)) + 1):
        dx = jitter * math.sin(i * 1.7)
        samples.append(SubjectSample(i / 10, min(1.0, max(0.0, cx + dx)), cy, w, h, conf, None, "detected"))
    return SubjectTrack(subject_id, "face", tuple(samples), speaker_id=speaker,
                        speaker_confidence=link if speaker else 0.0, speaker_evidence="test" if speaker else "")


def turn(speaker: str, start: float, end: float, step: float = 0.35) -> list[tuple[str, float, float, str]]:
    rows, t, i = [], start, 0
    while t + 0.3 <= end:
        rows.append((f"w{speaker}{i}", round(t, 3), round(t + 0.3, 3), speaker))
        t += step
        i += 1
    return rows


def validator(ctx):
    return lambda candidate: validate_plan(candidate, ctx, STYLE)


class CameraDirectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = fx.Workspace()

    def tearDown(self) -> None:
        self.ws.cleanup()

    def context(self, tracks=(), words=None, regions=(), report=fx.VIDEO_REPORT):
        return fx.make_context(self.ws, provider=Provider(tracks, regions), words=words, report=report)

    def direct(self, ctx, *events):
        report = validate_plan_payload(fx.plan(*events), ctx, STYLE)
        self.assertFalse(report.fatal, report.error_summary())
        directed, direction = direct_plan(report.plan, ctx, STYLE, validate=validator(ctx))
        resolved = resolve_plan(directed, ctx, STYLE, strict_center=True)
        rate = ctx.clip.fps.fps
        explain_holds([(a / rate, b / rate) for a, b in resolved.path.active_ranges()], ctx, STYLE, direction)
        return directed, direction, resolved

    @staticmethod
    def decision(direction, event_id):
        return next(d for d in direction.decisions if d["event_id"] == event_id)

    def test_confident_active_speaker_gets_speaker_framing(self) -> None:
        ctx = self.context([face("face_0", 0.68, speaker="A")], words=turn("A", 1.0, 6.8))
        _plan, direction, resolved = self.direct(ctx, fx.event("e1", 2.0, 5.0, role="setup", motion="slow_push",
                                                               reason_code="setup_clarity"))
        self.assertEqual(self.decision(direction, "e1")["intent"], ShotIntent.ACTIVE_SPEAKER_MEDIUM.value)
        op = resolved.ops[0]
        self.assertEqual(op.params.target_subject, "face_0")
        self.assertGreater(op.params.scale_peak, 1.03)
        self.assertGreater(op.params.anchor_x, 0.5)          # framed toward the speaker, not the centre

    def test_long_static_region_with_confident_turn_gets_one_restrained_emphasis(self) -> None:
        ctx = self.context([face("face_0", 0.68, speaker="A")], words=turn("A", 1.0, 6.8))
        _plan, direction, resolved = self.direct(ctx)
        self.assertEqual(len(direction.additions), 1)
        self.assertEqual(direction.additions[0]["intent"], ShotIntent.ACTIVE_SPEAKER_MEDIUM.value)
        self.assertEqual(resolved.ops[0].params.target_subject, "face_0")
        self.assertTrue(all(h["reason"] for h in direction.holds))

    def test_two_participants_carrying_the_moment_get_a_two_shot(self) -> None:
        ctx = self.context([face("face_0", 0.3, speaker="A"), face("face_1", 0.72, speaker="B")],
                           words=turn("B", 16.3, 17.8))
        _plan, direction, resolved = self.direct(ctx, fx.event("r1", 16.3, 17.9, role="reaction",
                                                               camera="reaction_close", reason_code="reaction_hold"))
        self.assertEqual(self.decision(direction, "r1")["intent"], ShotIntent.TWO_SHOT.value)
        op = next(o for o in resolved.ops if o.source_event_id == "r1")
        self.assertEqual(op.params.camera, CameraMode.DUAL_SUBJECT.value)
        for frame in range(op.start_frame, op.end_frame, 3):
            state = resolved.path.state_at(frame)
            for cx in (0.3, 0.72):
                self.assertTrue(crop_contains(state.zoom, (state.cx, state.cy), Box.from_center(cx, 0.4, 0.1, 0.16)))

    def test_story_critical_region_stays_visible_while_the_speaker_talks(self) -> None:
        region = (12.5, 17.5, (0.04, 0.55, 0.24, 0.9), "object")
        ctx = self.context([face("face_0", 0.7, speaker="A")], words=turn("A", 12.9, 15.5), regions=[region])
        _plan, direction, resolved = self.direct(ctx, fx.event("p1", 13.0, 15.0, camera="speaker_close",
                                                               target={"type": "subject", "id": "face_0"}))
        self.assertEqual(self.decision(direction, "p1")["intent"], ShotIntent.ACTION_REGION.value)
        geometry = render_proof.story_geometry_check(resolved, ctx, STYLE, step=1)
        self.assertEqual(geometry["status"], "passed", geometry)
        self.assertGreater(geometry["frames_checked"], 0)

    def test_gameplay_payload_is_never_cropped_away_for_a_face(self) -> None:
        report = copy.deepcopy(fx.VIDEO_REPORT)
        report["layout"] = {"orientation": "landscape", "content_type": "gameplay"}
        facecam = face("face_0", 0.88, speaker="A", cy=0.85, w=0.06, h=0.1)
        ctx = self.context([facecam], words=turn("A", 12.9, 15.5), report=report)
        _plan, direction, resolved = self.direct(
            ctx, fx.event("p1", 13.0, 15.0, camera="speaker_close", target={"type": "subject", "id": "face_0"}),
            fx.event("s1", 2.0, 5.0, role="setup", motion="slow_push", reason_code="setup_clarity"))
        self.assertEqual(direction.layout_mode, ShotIntent.GAMEPLAY_PRIORITY.value)
        for op in resolved.ops:
            self.assertIsNone(op.params.target_subject)
            self.assertAlmostEqual(op.params.anchor_x, 0.5, places=3)
            self.assertLessEqual(op.params.scale_peak, STYLE.zoom.overlay_layout_center_safe_max + 1e-6)
        self.assertNotIn("s1", {op.source_event_id for op in resolved.ops})   # non-peak roles hold

    def test_weak_speaker_face_link_stays_conservative(self) -> None:
        weak = self.context([face("face_0", 0.68, speaker="A", link=0.6)], words=turn("A", 6.8, 9.0))
        directed, direction, _resolved = self.direct(weak, fx.event("x1", 7.0, 9.0, role="escalation",
                                                                    camera="speaker_close",
                                                                    reason_code="escalation_rise"))
        event = next(e for e in directed.events if e.event_id == "x1")
        self.assertIsNot(event.camera, CameraMode.SPEAKER_CLOSE)
        self.assertIsNot(event.motion, MotionPreset.PUNCH_IN_FAST)
        # No face evidence at all: whole-frame intent only, never a subject lock.
        blind = self.context([], words=turn("A", 6.8, 9.0))
        directed, direction, resolved = self.direct(blind, fx.event("x1", 7.0, 9.0, role="escalation",
                                                                     camera="speaker_close",
                                                                     reason_code="escalation_rise"))
        self.assertEqual(self.decision(direction, "x1")["intent"], ShotIntent.WIDE_CONTEXT.value)
        self.assertTrue(all(op.params.target_subject is None for op in resolved.ops))
        # Weak links never earn an added emphasis; the HOLD says why.
        _plan, direction, _resolved = self.direct(self.context([face("face_0", 0.68, speaker="A", link=0.6)],
                                                               words=turn("A", 1.0, 6.8)))
        self.assertEqual(direction.additions, [])
        self.assertTrue(any("below" in h["reason"] for h in direction.holds), direction.holds)

    def test_noisy_tracking_does_not_make_the_camera_jitter(self) -> None:
        noisy = face("face_0", 0.66, speaker="A", jitter=0.03)
        ctx = self.context([noisy], words=turn("A", 1.0, 6.8))
        _plan, _direction, resolved = self.direct(ctx, fx.event("e1", 2.0, 6.0, role="setup",
                                                                motion="speaker_focus", reason_code="setup_clarity",
                                                                target={"type": "subject", "id": "face_0"}))
        op = resolved.ops[0]
        hold = [resolved.path.state_at(f) for f in range(op.start_frame + op.attack_frames,
                                                          op.start_frame + op.attack_frames + op.hold_frames)]
        self.assertGreater(len(hold), 10)
        centers = [s.cx for s in hold]
        raw = [s.cx for s in noisy.samples if 2.0 <= s.t <= 6.0]
        self.assertLess(statistics.pstdev(centers), 0.2 * statistics.pstdev(raw))
        steps = [abs(b - a) for a, b in zip(centers, centers[1:])]
        self.assertLessEqual(max(steps), STYLE.camera.max_speed_norm_per_s / ctx.clip.fps.fps + 1e-6)

    def test_every_long_hold_has_a_concrete_reason(self) -> None:
        ctx = self.context([face("face_0", 0.3, speaker="A"), face("face_1", 0.72, speaker="B")],
                           words=turn("A", 1.0, 3.0) + turn("B", 3.2, 5.0) + turn("A", 5.2, 7.0))
        _plan, direction, resolved = self.direct(ctx)
        self.assertTrue(direction.holds)
        for hold in direction.holds:
            self.assertTrue(hold["reason"].strip(), hold)
            self.assertGreaterEqual(hold["window"][1] - hold["window"][0], 6.0)

    def test_added_emphasis_never_displaces_a_story_event(self) -> None:
        ctx = self.context([face("face_0", 0.3, speaker="A"), face("face_1", 0.72, speaker="B")],
                           words=turn("A", 0.4, 1.7) + turn("A", 2.2, 6.9) + turn("B", 16.3, 17.8))
        plain = validate_plan_payload(fx.plan(
            fx.event("h1", 0.4, 1.8, role="hook", motion="slow_push", reason_code="hook_emphasis"),
            fx.event("r1", 16.3, 17.9, role="reaction", motion="slow_push", reason_code="reaction_hold")),
            ctx, STYLE)
        directed, direction = direct_plan(plain.plan, ctx, STYLE, validate=validator(ctx))
        active = {e.event_id for e in directed.events if e.is_camera_active}
        self.assertTrue({"h1", "r1"} <= active)
        for row in direction.additions:
            self.assertIn(row["event_id"], active)

    def test_a_static_pattern_never_carries_a_shot(self) -> None:
        from ai.editor.pro_edit.direction import camera_subjects

        def still(sid: str, cx: float, activity: float) -> SubjectTrack:
            return SubjectTrack(sid, "face", tuple(SubjectSample(i / 10, cx, 0.42, 0.1, 0.16, 0.9, activity,
                                                                 "detected") for i in range(0, 200)))

        # A poster/logo the detector calls a face (frozen, no mouth motion) vs a still but talking face-cam.
        ctx = self.context([still("pattern", 0.8, 0.0004), still("facecam", 0.3, 0.02)], words=turn("A", 16.3, 17.8))
        excluded: list = []
        self.assertEqual({s.track_id for s in camera_subjects(ctx, STYLE, excluded)}, {"facecam"})
        self.assertEqual([row["track_id"] for row in excluded], ["pattern"])
        _plan, direction, resolved = self.direct(
            self.context([still("pattern", 0.8, 0.0004)], words=turn("A", 16.3, 17.8)),
            fx.event("r1", 16.3, 17.9, role="reaction", camera="reaction_close", reason_code="reaction_hold"))
        self.assertEqual(direction.excluded_subjects[0]["track_id"], "pattern")
        self.assertTrue(all(op.params.target_subject != "pattern" for op in resolved.ops))

    def test_an_unconfirmed_face_never_gets_a_reaction_close_up(self) -> None:
        reaction = fx.event("r1", 16.3, 17.9, role="reaction", camera="reaction_close", reason_code="reaction_hold")
        directed, direction, _resolved = self.direct(self.context([face("face_0", 0.62, speaker=None)],
                                                                   words=turn("B", 16.3, 17.8)), reaction)
        event = next(e for e in directed.events if e.event_id == "r1")
        self.assertNotIn(event.camera, (CameraMode.REACTION_CLOSE, CameraMode.SPEAKER_CLOSE))
        self.assertIn("medium framing at most", self.decision(direction, "r1")["reason"])
        # The same reaction on a face confirmed as a participant (the listener) keeps the close-up.
        directed, direction, _resolved = self.direct(self.context([face("face_0", 0.62, speaker="A")],
                                                                   words=turn("B", 16.3, 17.8)), reaction)
        self.assertIs(next(e for e in directed.events if e.event_id == "r1").camera, CameraMode.REACTION_CLOSE)

    def test_v6_wide_intent_is_frame_centred_while_v5_keeps_its_resolution(self) -> None:
        ctx = self.context([face("face_0", 0.75, speaker="A")], words=turn("A", 1.0, 6.8))
        plan = validate_plan_payload(fx.plan(fx.event("e1", 2.0, 5.0, role="setup", motion="slow_push",
                                                      reason_code="setup_clarity")), ctx, STYLE).plan
        v6 = resolve_plan(plan, ctx, STYLE, strict_center=True).ops[0]
        v5 = resolve_plan(plan, ctx, STYLE).ops[0]
        self.assertIsNone(v6.params.target_subject)
        self.assertAlmostEqual(v6.params.anchor_x, 0.5, places=4)
        self.assertEqual(v5.params.target_subject, "face_0")


class EvidenceTests(unittest.TestCase):
    def test_video_brain_package_is_unwrapped(self) -> None:
        package = {"analyzer_version": 6, "report": copy.deepcopy(fx.VIDEO_REPORT)}
        self.assertEqual(visual_report_body(package)["layout"]["content_type"], "streamer")
        self.assertEqual(visual_report_body(fx.VIDEO_REPORT)["layout"]["content_type"], "streamer")
        ws = fx.Workspace()
        try:
            ctx = fx.make_context(ws, report=package)
            self.assertEqual(ctx.layout_content_type, "streamer")
            self.assertTrue(ctx.visual_events)
        finally:
            ws.cleanup()

    def test_speaker_face_link_needs_consistent_audiovisual_evidence(self) -> None:
        from ai.editor.pro_edit.caption_guard import CaptionWordRef
        from ai.editor.pro_edit.speaker_link import associate_speakers

        turns = [("A", 0.0, 4.0), ("B", 4.5, 8.5), ("A", 9.0, 13.0), ("B", 13.5, 17.5), ("A", 18.0, 22.0)]
        words, wid = [], 0
        for speaker, a, b in turns:
            t = a
            while t + 0.3 <= b:
                words.append(CaptionWordRef(wid, f"w{wid}", t, t + 0.3, speaker, "", 0.9))
                wid += 1
                t += 0.4

        def talking(speaker, t):
            return any(s == speaker and a <= t <= b for s, a, b in turns)

        def track(sid, cx, activity):
            samples = tuple(SubjectSample(i / 10, cx, 0.4, 0.1, 0.16, 0.9, activity(i / 10), "detected")
                            for i in range(0, 221))
            return SubjectTrack(sid, "face", samples)

        wobble = lambda t: 0.004 * (1 + math.sin(t * 7.3))  # noqa: E731  (mouth micro-motion noise)
        face_a = track("face_a", 0.3, lambda t: (0.03 if talking("A", t) else 0.005) + wobble(t))
        face_b = track("face_b", 0.7, lambda t: (0.03 if talking("B", t) else 0.005) + wobble(t + 1.0))
        linked = {t.subject_id: (t.speaker_id, t.speaker_confidence) for t in associate_speakers([face_a, face_b],
                                                                                                  words)}
        self.assertEqual(linked["face_a"][0], "A")
        self.assertEqual(linked["face_b"][0], "B")
        self.assertGreaterEqual(min(linked["face_a"][1], linked["face_b"][1]), 0.5)
        # No mouth evidence (a flat track) and identical evidence for two faces: no pretended certainty.
        flat = track("face_flat", 0.5, lambda t: 0.01 + wobble(t))
        self.assertIsNone(associate_speakers([flat], words)[0].speaker_id)
        twin = track("face_twin", 0.7, lambda t: (0.03 if talking("A", t) else 0.005) + wobble(t))
        result = {t.subject_id: t.speaker_id for t in associate_speakers([face_a, twin], words)}
        self.assertNotEqual(list(result.values()).count("A"), 2)

    def test_fragments_stitch_into_one_subject_but_never_across_a_scene_cut(self) -> None:
        def fragment(sid, t0, t1, cx):
            return SubjectTrack(sid, "face", tuple(SubjectSample(round(t, 2), cx, 0.4, 0.1, 0.16, 0.9, None,
                                                                 "detected") for t in
                                                   [t0 + i * 0.1 for i in range(int((t1 - t0) * 10))]))
        pieces = [fragment("face_00", 0.0, 4.0, 0.30), fragment("face_01", 5.0, 9.0, 0.31)]
        self.assertEqual(len(consolidate_tracks(pieces)), 1)
        self.assertEqual(len(consolidate_tracks(pieces, cuts=[4.5])), 2)


class PlanIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = fx.Workspace()
        self.ctx = fx.make_context(self.ws)

    def tearDown(self) -> None:
        self.ws.cleanup()

    def test_emphasis_reasons_never_replace_the_event_reason_code(self) -> None:
        from ai.editor.pro_edit.schema import ReasonCode

        word = next(w.id for w in self.ctx.words if w.text == "THIS")
        for reasons in ([{"word_id": word, "reason": "payoff"}], [{"word_id": word, "reason": "not-a-reason"}]):
            with self.subTest(reasons=reasons):
                report = validate_plan_payload(fx.plan(fx.event("p1", 13.0, 15.0, emphasis_word_ids=[word],
                                                                emphasis_reasons=reasons)), self.ctx, STYLE)
                event = report.plan.events[0]
                self.assertIs(event.reason_code, ReasonCode.PAYOFF_HIT)
                self.assertEqual(event.to_dict()["reason_code"], "payoff_hit")
                again = validate_plan(report.plan, self.ctx, STYLE)          # re-validation stays clean
                self.assertNotIn("reason_code_invalid", {i.code for i in again.issues})

    def test_only_policy_notes_count_as_the_reason_an_event_was_switched_off(self) -> None:
        from types import SimpleNamespace

        from ai.editor.pro_edit.direction import _restraint_rows
        from ai.editor.pro_edit.validator import Severity, ValidationIssue

        event = validate_plan_payload(fx.plan(fx.event("p1", 13.0, 15.0)), self.ctx, STYLE).plan.events[0]
        checked = SimpleNamespace(plan=SimpleNamespace(events=()), issues=(
            ValidationIssue("reason_code_invalid", Severity.SANITIZED, "", "p1", "schema"),
            ValidationIssue("caption_unknown_word_id", Severity.SANITIZED, "", "p1", "caption"),
            ValidationIssue("repeated_preset_limit", Severity.SANITIZED, "", "p1", "density")))
        self.assertEqual(_restraint_rows([event], checked, {})[0]["codes"], ["repeated_preset_limit"])

    def test_a_restrained_hold_still_names_its_story_reasons(self) -> None:
        from ai.editor.pro_edit.direction import DirectionReport

        report = DirectionReport(restraint=[{"event_id": "x1", "window": [2.0, 4.0], "intent": "WIDE_CONTEXT",
                                             "codes": ["repeated_preset_limit"]}])
        holds = explain_holds([], self.ctx, STYLE, report)
        self.assertTrue(holds)
        self.assertTrue(holds[0]["reason"].startswith("editing restraint: WIDE_CONTEXT x1 switched off"))
        self.assertIn(" | ", holds[0]["reason"])
        self.assertTrue(all(segment["reason"] for segment in holds[0]["segments"]))


class VisionRuntimeTests(unittest.TestCase):
    """OpenCV failures are classified (not installed / refused by Windows / broken) with the remedy."""

    def test_import_failures_are_classified_with_the_remedy(self) -> None:
        from ai.editor.pro_edit.vision import cv_runtime as cv

        refused = ImportError("DLL load failed while importing cv2: An Application Control policy has blocked "
                              "this file.")
        kind, reason = cv.classify_import_error(refused, installed="opencv-python-headless 9.9", sac_state="on")
        self.assertEqual(kind, "blocked_by_windows_app_control")
        self.assertIn("Smart App Control", reason)
        self.assertIn(cv.INSTALL_HINT, reason)
        for state in ("off", "evaluation", None):           # evaluation mode audits only, it never blocks
            self.assertEqual(cv.classify_import_error(refused, installed="x 1", sac_state=state)[0], "dll_load_failed")
        self.assertEqual(cv.classify_import_error(ModuleNotFoundError("No module named 'cv2'"))[0], "not_installed")
        self.assertEqual(cv.classify_import_error(ImportError("numpy.core.multiarray failed to import"),
                                                  installed="x 1")[0], "import_failed")

    def test_consumers_report_the_classified_cause_instead_of_not_installed(self) -> None:
        import sys
        from unittest import mock

        from ai.editor.pro_edit.vision import cv_runtime as cv
        from ai.editor.pro_edit.vision.detectors import load_detector

        with mock.patch.dict(sys.modules, {"cv2": None}):
            with self.assertRaises(ImportError):                    # existing ImportError fallbacks still apply
                cv.load_opencv()
            detector, reason = load_detector()
            self.assertIsNone(detector)
            self.assertEqual(reason, cv.opencv_status()["reason"])
            proof = render_proof.prove_camera(source_path="missing.mp4", rendered_path="missing.mp4",
                                              resolved=_one_op_plan(), source_size=(320, 180))
            self.assertEqual(proof["status"], "unavailable")
            self.assertIn("OpenCV unavailable", proof["reason"])


def _one_op_plan():
    ws = fx.Workspace()
    try:
        ctx = fx.make_context(ws, media_info=fx.media(width=320, height=180))
        return resolve_plan(validate_plan_payload(fx.plan(fx.event("p1", 13.0, 15.0)), ctx, STYLE).plan, ctx, STYLE,
                            strict_center=True)
    finally:
        ws.cleanup()


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available on PATH")
@fx.needs_opencv
class PixelProofTests(unittest.TestCase):
    """The planned camera must be visible in the rendered frames (and absent from a baseline render)."""

    @classmethod
    def setUpClass(cls) -> None:
        from ai.editor.pro_edit.ffmpeg_filters import probe_capabilities
        from ai.editor.pro_edit.media import probe_media

        cls._tmp = tempfile.TemporaryDirectory(prefix="mimir v6 proof ş'")
        cls.dir = Path(cls._tmp.name)
        cls.clip = cls.dir / "paced clip ✓.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=320x180:rate=30000/1001:duration=28", "-f", "lavfi", "-i",
                        "sine=frequency=300:sample_rate=48000:duration=28", "-shortest", "-c:v", "libx264",
                        "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        str(cls.clip)], check=True, capture_output=True)
        cls.media = probe_media(cls.clip)
        cls.caps = probe_capabilities()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_rendered_frames_reflect_the_camera_plan(self) -> None:
        from ai.editor.pro_edit.executor import render_camera_captions

        ws = fx.Workspace()
        try:
            ctx = fx.make_context(ws, media_info=self.media)
            plan = validate_plan_payload(fx.plan(fx.event("p1", 13.0, 15.0)), ctx, STYLE).plan
            resolved = resolve_plan(plan, ctx, STYLE, strict_center=True)
            self.assertTrue(resolved.ops)
            output = self.dir / "camera render.mp4"
            render_camera_captions(edited_clip=self.clip, caption_file=None, output_path=output, resolved=resolved,
                                   caps=self.caps, source_media=self.media, script_path=self.dir / "graph.txt")
            proof = render_proof.prove_camera(source_path=self.clip, rendered_path=output, resolved=resolved,
                                              source_size=(self.media.width, self.media.height))
            self.assertEqual(proof["status"], "passed", proof)
            self.assertTrue(any(s["verdict"] == "identity_ok" for s in proof["samples"]))
            # A render WITHOUT the camera (the baseline) is caught: the plan did not reach its pixels.
            baseline = render_proof.prove_camera(source_path=self.clip, rendered_path=self.clip, resolved=resolved,
                                                 source_size=(self.media.width, self.media.height))
            self.assertEqual(baseline["status"], "failed", baseline)
        finally:
            ws.cleanup()


if __name__ == "__main__":
    unittest.main()
