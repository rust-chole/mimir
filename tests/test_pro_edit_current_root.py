"""Current-root binding of Pro Edit V5 (captions V24 speaker policy, pipeline, renderers).

These invariants are specific to THIS MIMIR root and are not covered by the
V5 reference suite, which was written against the older V16 caption engine.
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path, PureWindowsPath
from unittest import mock

import pro_edit_fixtures as fx
import test_pro_edit_captions as base
from ai.editor import caption_renderer, captions, intro_renderer
from ai.editor.pro_edit import caption_presentation as cp
from ai.editor.pro_edit.caption_guard import trusted_display_names

ROOT = Path(__file__).resolve().parent.parent


def overlapping_dialogue() -> list[dict]:
    """A speaks while B talks OVER A ("wait no": measured overlap); much later B answers
    sequentially ("he just", outside every padded overlap window). Every B word carries the
    raw profile role 'secondary', like real dual profiles."""
    spec = [("so", 0.30, 0.52, "A"), ("what", 0.58, 0.84, "A"), ("happened", 0.90, 1.26, "A"),
            ("then", 1.32, 1.56, "A"), ("wait", 0.62, 0.84, "B"), ("no", 0.95, 1.15, "B"),
            ("he", 3.00, 3.20, "B"), ("just", 3.26, 3.50, "B"), ("left", 5.00, 5.20, "A"),
            ("the", 5.26, 5.40, "A"), ("room", 5.46, 5.80, "A")]
    return [{"word": word, "edited_start": start, "edited_end": end, "speaker_raw": raw,
             "speaker_role": "secondary" if raw == "B" else "main", "speaker_label": "", "speaker_confidence": 0.9}
            for word, start, end, raw in spec]


class DisplayPolicyBindingTests(unittest.TestCase):
    def test_display_metadata_is_the_v24_renderers_own_policy(self) -> None:
        rows = overlapping_dialogue()
        data = base.human_profile(rows, {"A": "KAI", "B": "TYLA"}, primary_speaker="A", secondary_speaker="B")
        duration = cp.edited_duration(data, base.timeline())
        words = captions._profile_edited_words(data, duration)
        prepared, windows = captions._prepare_adaptive_render_words(
            words, speaker_profile=data, trusted_display_map=captions._trusted_human_display_map(data))
        self.assertTrue(windows)                                            # measured overlap exists
        tokens = cp.presentation_tokens(data, duration)
        self.assertEqual([(t.speaker_role, t.speaker_label) for t in tokens],
                         [(w["speaker_role"], w["speaker_label"]) for w in prepared])
        cp.verify_token_parity(tokens, data, duration)
        # Sequential B words ("he just") stay on the main lane; only overlapping B words go up.
        roles = {t.text: t.speaker_role for t in tokens}
        self.assertEqual((roles["wait"], roles["no"]), ("secondary", "secondary"))
        self.assertEqual((roles["he"], roles["just"]), ("main", "main"))
        self.assertEqual(roles["what"], "main")
        labels = {t.text: t.speaker_label for t in tokens}
        self.assertEqual((labels["wait"], labels["he"], labels["so"]), ("TYLA", "TYLA", "KAI"))
        # Raw truth is untouched.
        self.assertEqual([t.speaker_raw for t in tokens], [w["speaker_raw"] for w in words])

    def test_parity_rejects_a_token_with_a_foreign_lane(self) -> None:
        rows = overlapping_dialogue()
        data = base.human_profile(rows, {"A": "KAI"}, primary_speaker="A", secondary_speaker="B")
        tokens = list(cp.presentation_tokens(data, 99.0))
        forged = tokens[0]
        tokens[0] = cp.CaptionTokenRef(forged.word_id, forged.text, forged.start, forged.end, forged.speaker_raw,
                                       "secondary", forged.speaker_label, forged.normalized)
        with self.assertRaises(cp.CaptionPresentationError):
            cp.verify_token_parity(tokens, data, 99.0)

    def test_sequential_raw_secondary_never_creates_a_permanent_lane(self) -> None:
        # Real-profile shape: many words carry speaker_role='secondary' with no overlap.
        rows = base.words("first speaker line here then second speaker answers now", secondary=range(5, 10),
                          label="TYLA")
        for data in (base.profile(rows), base.human_profile(rows, {"B": "TYLA"})):
            built = cp.build_presentation(profile=data, clip_timeline=base.timeline(), plan=None, width=1080,
                                          height=1920, metrics=base.BUILTIN)
            self.assertEqual({p.lane for p in built.pages}, {"main"})
        named = cp.build_presentation(profile=base.human_profile(rows, {"B": "TYLA"}), clip_timeline=base.timeline(),
                                      plan=None, width=1080, height=1920, metrics=base.BUILTIN)
        self.assertIn("TYLA", {p.label for p in named.pages})             # confirmed name is still shown
        unnamed = cp.build_presentation(profile=base.profile(rows), clip_timeline=base.timeline(), plan=None,
                                        width=1080, height=1920, metrics=base.BUILTIN)
        self.assertFalse(any(p.label for p in unnamed.pages))              # word-level label alone is not trust


class TrustedNamesTests(unittest.TestCase):
    def test_only_human_confirmed_names_are_trusted(self) -> None:
        auto = {"status": "ok", "display_labels": {"A": "KAI", "B": "TYLA"},
                "speaker_names": {"source": "single_auto_plain_no_prompt"}}
        self.assertEqual(trusted_display_names(auto), {})
        manual = copy.deepcopy(auto)
        manual["speaker_names"]["source"] = "manual_voice_calibrated_v28"
        self.assertEqual(trusted_display_names(manual), {"A": "KAI", "B": "TYLA"})
        partial = copy.deepcopy(auto)
        partial["identity_calibration"] = {"B": {"raw_speaker": "B", "human_verified": True}}
        self.assertEqual(trusted_display_names(partial), {"B": "TYLA"})
        self.assertEqual(trusted_display_names(None), {})

    def test_planner_context_sees_only_confirmed_identities(self) -> None:
        workspace = fx.Workspace()
        try:
            context = fx.make_context(workspace)            # fixture: display_labels without human source
            self.assertEqual(dict(context.speaker_identities), {})
        finally:
            workspace.cleanup()
        workspace = fx.Workspace()
        try:
            original = fx.speaker_profile

            def confirmed(words=None):
                data = original(words)
                data["speaker_names"] = {"A": "KAI", "B": "TYLA", "source": "manual_voice_calibrated_v28"}
                return data

            fx.speaker_profile = confirmed
            try:
                context = fx.make_context(workspace)
            finally:
                fx.speaker_profile = original
            self.assertEqual(dict(context.speaker_identities), {"A": "KAI", "B": "TYLA"})
        finally:
            workspace.cleanup()

    def test_name_reason_needs_a_confirmed_name(self) -> None:
        token = cp.CaptionTokenRef(0, "Kai", 0.0, 0.3, "A", "main", "", "kai")
        self.assertEqual(cp.derive_reason(token, cp.name_tokens(trusted_display_names(
            {"display_labels": {"A": "KAI"}, "speaker_names": {"source": "auto"}}).values()), ()), "generic")
        self.assertEqual(cp.derive_reason(token, cp.name_tokens(trusted_display_names(
            {"display_labels": {"A": "KAI"}, "speaker_names": {"source": "manual"}}).values()), ()), "name")


class HardFailureLaneTimingTests(unittest.TestCase):
    def rows(self) -> list[dict]:
        rows = base.words("done. next part", gap=0.02, sentence_gap=0.01)
        rows[0].update(edited_start=1.0, edited_end=1.02)
        rows[1].update(edited_start=1.03, edited_end=1.3)
        rows[2].update(edited_start=1.35, edited_end=1.6)
        return rows

    def build(self, data: dict) -> cp.CaptionPresentation:
        return cp.build_presentation(profile=data, clip_timeline=base.timeline(), plan=None, width=1080,
                                     height=1920, metrics=base.BUILTIN)

    def test_hard_failure_pages_in_one_lane_never_overlap(self) -> None:
        normal = base.profile(self.rows())
        failed = dict(normal, speaker_boundary_audit={"hard_failure": True, "suspicious": True})
        self.assertFalse(cp.strict_lane_timing(normal))
        self.assertTrue(cp.strict_lane_timing(failed))
        first_normal, second_normal = self.build(normal).pages[:2]
        first_failed, second_failed = self.build(failed).pages[:2]
        # Normal mode keeps the baseline minimum-event floor (a <= 70 ms same-lane overlap) ...
        self.assertGreater(first_normal.end, second_normal.start)
        # ... the V24 hard-failure lane never shows two pages at once, and every word stays visible.
        self.assertLessEqual(first_failed.end, second_failed.start)
        self.assertGreater(first_failed.end, first_failed.start)
        self.assertEqual(self.build(failed).metrics()["same_lane_overlaps"], 0)


class RendererAndPipelineTests(unittest.TestCase):
    WINDOWS_PATHS = (r"C:\Users\yusuf\Desktop\mimir_unified_clean_v3\vod_output\captions\kai kapi\clip_01.ass",
                     r"D:\MIMIR Türkçe çğıİöşü\altyazı [1], a;b.ass")

    def test_both_renderers_escape_filter_paths_identically(self) -> None:
        class WindowsPathLike:
            def __init__(self, value: str) -> None:
                self.value = PureWindowsPath(value)

            def resolve(self):
                return self

            def as_posix(self) -> str:
                return self.value.as_posix()

            def __str__(self) -> str:
                return str(self.value)

        for raw in self.WINDOWS_PATHS + (r"C:\Users\O'Brien\it's.ass",):
            with mock.patch.object(caption_renderer, "Path", lambda value: WindowsPathLike(str(value))), \
                    mock.patch.object(intro_renderer, "Path", lambda value: WindowsPathLike(str(value))):
                a = caption_renderer.escape_filter_path(raw)
                b = intro_renderer.escape_filter_path(raw)
            self.assertEqual(a, b)
            self.assertTrue(a.startswith(raw[0] + "\\:/"))
            self.assertNotIn("\\'", a.replace("'\\\\\\''", ""))              # no broken \' left behind

    def test_current_root_intro_behaviour_is_preserved(self) -> None:
        source = (ROOT / "ai" / "editor" / "intro_renderer.py").read_text(encoding="utf-8-sig")
        self.assertIn("LOCKED PEAK VALIDATION", source)                     # locked peak must stay inside the intro
        self.assertIn("Intro composition clock mismatch", source)           # renderer proves intro + main duration
        pipeline = (ROOT / "ai" / "shorts_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("main-only fallback YASAK", pipeline)
        guard = pipeline[pipeline.index("final_timeline_doc = intro_renderer.load_final_timeline(final_preview_path)"):]
        guard = guard[:guard.index("_stage_done(final_preview_path)")]
        self.assertIn("Mandatory intro structural guard", guard)            # no cold open -> no publish
        self.assertLess(guard.index("raise ShortsPipelineError"), guard.index("_verify_pro_edit_intro"))

    def test_single_production_path_always_runs_the_verified_presentation(self) -> None:
        code = (
            "import os, sys, json\n"
            "for key in ('MIMIR_PRO_EDIT', 'MIMIR_V6'): os.environ[key] = '0'\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "import ai.shorts_pipeline as sp\n"
            "pkg, cfg = sp._load_pro_edit()\n"
            "src = {'path': 'x.mp4', 'size': 1, 'mtime_ns': 2}\n"
            "kw = dict(creator_name='C', clip_index=1, enable_memes=True, enable_video_brain=False,"
            " video_brain_model='m')\n"
            "print(json.dumps({'enabled': cfg.enabled, 'v6': cfg.v6,"
            " 'presentation_in_signature': sp._request_signature(src, presentation_signature='a', **kw)"
            " != sp._request_signature(src, presentation_signature='b', **kw),"
            " 'flags_removed': not hasattr(sp, '_pro_edit_requested') and not hasattr(sp, '_v6_requested')}))\n")
        result = subprocess.run([sys.executable, "-X", "utf8", "-c", code], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(json.loads(result.stdout.strip().splitlines()[-1]),
                         {"enabled": True, "v6": True, "presentation_in_signature": True, "flags_removed": True})

    def test_baseline_captions_are_untouched_by_the_port(self) -> None:
        self.assertEqual(captions.CAPTION_VERSION, 24)
        with tempfile.TemporaryDirectory() as tmp:
            data = base.human_profile(overlapping_dialogue(), {"A": "KAI"}, primary_speaker="A",
                                      secondary_speaker="B")
            ass = captions.create_ass_for_clip({}, base.timeline(), Path(tmp) / "b.ass", data)
            first = intro_renderer.get_first_caption_start(ass)
            built = cp.build_presentation(profile=data, clip_timeline=base.timeline(), plan=None, width=1080,
                                          height=1920, metrics=base.BUILTIN)
            presentation = cp.write_presentation(built, Path(tmp) / "p.ass")
            cp.check_intro_handoff(presentation, ass)                       # same first caption instant
            self.assertAlmostEqual(intro_renderer.get_first_caption_start(presentation), first, places=2)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe not available on PATH")
class NoOpenCvDegradationTests(unittest.TestCase):
    """The production .venv may not have the optional OpenCV/numpy: every pixel-evidence
    layer must degrade, and the caption presentation must still be built and rendered."""

    @classmethod
    def setUpClass(cls) -> None:
        import test_pro_edit_render as render

        cls._tmp = tempfile.TemporaryDirectory(prefix="mimir no opencv ş ")
        cls.root = Path(cls._tmp.name)
        cls.clip = cls.root / "clip_01_demo_edited.mp4"
        render.make_clip(cls.clip, "testsrc2")
        timeline_data = render.timeline_for("demo no opencv")
        cls.timeline = cls.root / "timeline.json"
        cls.timeline.write_text(json.dumps(timeline_data), encoding="utf-8")
        cls.profile = cls.root / "speakers.json"
        cls.profile.write_text(json.dumps(render.profile()), encoding="utf-8")
        cls.ass = Path(captions.create_ass_for_clip({}, timeline_data["timelines"][0], cls.root / "caps.ass",
                                                    render.profile()))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_without_opencv_evidence_degrades_and_the_presentation_still_renders(self) -> None:
        from ai.editor.pro_edit.config import ProEditConfig
        from ai.editor.pro_edit.media import probe_media
        from ai.editor.pro_edit.planner import StaticEditPlanner
        from ai.editor.pro_edit.stage import ProEditRequest, prepare_pro_edit, render_with_fallback

        outcome: dict = {}
        with mock.patch.dict(sys.modules, {"cv2": None, "numpy": None}):     # import cv2 -> ImportError
            prep = prepare_pro_edit(ProEditRequest(
                config=ProEditConfig(enabled=True, planner="static"), timeline_path=self.timeline, clip_index=1,
                edited_clip_path=self.clip, caption_path=self.ass, output_path=self.root / "nocv.mp4",
                artifact_dir=self.root / "nocv", input_signature="sig-nocv", force=True,
                speaker_profile_path=self.profile, planner=StaticEditPlanner()))
            self.assertTrue(prep.ready, prep.reason)
            self.assertEqual(prep.tracking.get("status"), "unavailable")         # default local tracker
            self.assertEqual(prep.captions["status"], "presentation")
            self.assertEqual((prep.captions["activity"], prep.captions["background"]), ("failed", "failed"))
            self.assertEqual(prep.captions["level"], "v5_placement")
            output = render_with_fallback(prep, caption_file=self.ass,
                                          baseline=lambda: (_ for _ in ()).throw(AssertionError("baseline used")),
                                          outcome=outcome)
        self.assertEqual((outcome["status"], outcome["captions"]), ("pro_edit", "presentation"))
        rendered, source = probe_media(output), probe_media(self.clip)
        self.assertEqual((rendered.width, rendered.height, rendered.frame_count),
                         (source.width, source.height, source.frame_count))
        self.assertEqual(rendered.fps.fraction, source.fps.fraction)
        self.assertTrue(rendered.has_audio)


if __name__ == "__main__":
    unittest.main()
