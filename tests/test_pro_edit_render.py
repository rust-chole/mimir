"""Internal render smoke/integration tests (real FFmpeg, synthetic media).

Not a deliverable video: everything is rendered into a temp directory and
deleted. Skipped (reported as SKIPPED) when ffmpeg/ffprobe are unavailable.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pro_edit_fixtures as fx  # noqa: F401  (sys.path)
from ai.editor import caption_renderer, captions, intro_renderer
from ai.editor.pro_edit.config import ProEditConfig
from ai.editor.pro_edit.ffmpeg_filters import probe_capabilities
from ai.editor.pro_edit.media import probe_media
from ai.editor.pro_edit.planner import RuleBasedEditPlanner, StaticEditPlanner
from ai.editor.pro_edit.stage import ProEditRequest, prepare_pro_edit, render_with_fallback

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
FPS = "30000/1001"
DURATION = 8.0


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)


def make_clip(path: Path, source: str) -> None:
    joiner = ":" if "=" in source else "="
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         f"{source}{joiner}size=640x360:rate={FPS}:duration={DURATION}",
         "-f", "lavfi", "-i", f"sine=frequency=330:sample_rate=48000:duration={DURATION}",
         "-shortest", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k", "-ac", "2", str(path)])


def audio_md5(path: Path) -> str:
    out = run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-map", "0:a:0", "-c", "copy", "-f", "md5", "-"]).stdout
    return out.strip()


def psnr(a: Path, b: Path, first: int, last: int) -> float:
    graph = (f"[0:v]select='between(n,{first},{last})',setpts=N/TB[x];"
             f"[1:v]select='between(n,{first},{last})',setpts=N/TB[y];[x][y]psnr")
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(a), "-i", str(b), "-filter_complex", graph,
                           "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    match = re.search(r"average:(inf|[0-9.]+)", proc.stderr)
    if not match:
        raise AssertionError(proc.stderr[-800:])
    return float("inf") if match.group(1) == "inf" else float(match.group(1))


def timeline_for(stem: str) -> dict:
    return {
        "version": 3,
        "source": {"video_stem": stem},
        "timelines": [{
            "clip_index": 1, "title": "demo",
            "source": {"absolute_start": 100.0, "absolute_end": 108.0, "duration": DURATION},
            "edited": {"estimated_duration": DURATION},
            "cut_ranges": [],
            "payoff": {"source_start": 3.0, "source_end": 4.6},
            "protected_ranges": [{"start": 2.9, "end": 5.0}],
            "hook": {"text": "watch this"},
            "editorial": {"anchor_moments": [{"start": 104.8, "end": 106.5, "type": "reaction", "strength": 8}]},
            "events": [],
        }],
    }


def profile() -> dict:
    words = [(f"word{i}", 0.4 + i * 0.35, 0.4 + i * 0.35 + 0.28, "A") for i in range(20)]
    data = fx.speaker_profile(words)
    data["clip_duration"] = DURATION
    return data


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available on PATH")
class RenderIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="mimir pro edit ✓ çıktı ")
        cls.root = Path(cls._tmp.name)
        cls._saved = {
            "cr_edited": caption_renderer.EDITED_CLIPS_DIR, "cr_preview": caption_renderer.PREVIEW_DIR,
            "ir_final": intro_renderer.FINAL_PREVIEWS_DIR, "ir_assets": intro_renderer.INTRO_ASSETS_DIR,
        }
        caption_renderer.EDITED_CLIPS_DIR = cls.root / "edited_clips"
        caption_renderer.PREVIEW_DIR = cls.root / "previews"
        intro_renderer.FINAL_PREVIEWS_DIR = cls.root / "final_previews"
        intro_renderer.INTRO_ASSETS_DIR = cls.root / "intro_assets"
        cls.caps = probe_capabilities()
        cls.sets = {}
        for stem, source in (("demo textured", "testsrc2"), ("demo flat", "color=c=0x404040")):
            folder = caption_renderer.EDITED_CLIPS_DIR / stem
            folder.mkdir(parents=True)
            clip = folder / "clip_01_demo_edited.mp4"
            make_clip(clip, source)
            timeline_path = cls.root / f"{stem}_timeline_v3.json"
            timeline_path.write_text(json.dumps(timeline_for(stem)), encoding="utf-8")
            profile_path = cls.root / f"{stem}_speakers.json"
            profile_path.write_text(json.dumps(profile()), encoding="utf-8")
            ass = captions.create_ass_for_clip({}, timeline_for(stem)["timelines"][0],
                                               cls.root / f"{stem}_captions.ass", profile())
            baseline = Path(caption_renderer.render_captioned_clip(timeline_path, 1, caption_path=ass))
            cls.sets[stem] = {"clip": clip, "timeline": timeline_path, "profile": profile_path, "ass": Path(ass),
                              "baseline": baseline}

    def test_caption_render_consumes_the_declared_paced_clip(self) -> None:
        # The pipeline's baseline render (also the Pro Edit fallback) must render the paced
        # clip every other stage used, never re-discover one: here the timeline's own
        # edited_clips folder is empty, the declared clip is elsewhere.
        stem = "demo declared"
        timeline_path = self.root / f"{stem}_timeline_v3.json"
        timeline_path.write_text(json.dumps(timeline_for(stem)), encoding="utf-8")
        ass = captions.create_ass_for_clip({}, timeline_for(stem)["timelines"][0], self.root / f"{stem}.ass",
                                           profile())
        declared = self.root / "declared paced ✓.mp4"
        shutil.copy2(self.sets["demo flat"]["clip"], declared)
        with self.assertRaises(FileNotFoundError):                 # directory discovery finds nothing
            caption_renderer.render_captioned_clip(timeline_path, 1, caption_path=ass)
        output = Path(caption_renderer.render_captioned_clip(timeline_path, 1, caption_path=ass,
                                                             edited_clip_path=declared))
        self.assertGreater(psnr(output, self.sets["demo flat"]["baseline"], 0, 60), 40.0)
        with self.assertRaises(FileNotFoundError):                 # a missing declared clip is never replaced
            caption_renderer.render_captioned_clip(timeline_path, 1, caption_path=ass,
                                                   edited_clip_path=self.root / "missing.mp4")

    @classmethod
    def tearDownClass(cls) -> None:
        caption_renderer.EDITED_CLIPS_DIR = cls._saved["cr_edited"]
        caption_renderer.PREVIEW_DIR = cls._saved["cr_preview"]
        intro_renderer.FINAL_PREVIEWS_DIR = cls._saved["ir_final"]
        intro_renderer.INTRO_ASSETS_DIR = cls._saved["ir_assets"]
        cls._tmp.cleanup()

    def prepare(self, stem: str, *, planner=None, caps=None, name: str = "pro.mp4", captions: bool = True):
        data = self.sets[stem]
        request = ProEditRequest(
            config=ProEditConfig(enabled=True, planner="rules", captions=captions),
            timeline_path=data["timeline"], clip_index=1, edited_clip_path=data["clip"],
            caption_path=data["ass"], output_path=self.root / stem / name,
            artifact_dir=self.root / "artifacts" / stem, input_signature=f"sig-{name}", force=True,
            speaker_profile_path=data["profile"], planner=planner or RuleBasedEditPlanner(),
            capabilities=caps or self.caps,
        )
        return prepare_pro_edit(request)

    def test_pro_render_contract(self) -> None:
        data = self.sets["demo textured"]
        prep = self.prepare("demo textured")
        self.assertTrue(prep.ready, prep.reason)
        outcome: dict = {}
        out = render_with_fallback(prep, caption_file=data["ass"],
                                   baseline=lambda: (_ for _ in ()).throw(AssertionError("baseline used")),
                                   outcome=outcome)
        self.assertEqual(outcome["status"], "pro_edit", outcome)
        src, res = probe_media(data["clip"]), probe_media(out)
        self.assertEqual((res.width, res.height), (src.width, src.height))       # geometry preserved
        self.assertEqual(res.fps.fraction, src.fps.fraction)                    # 30000/1001 kept
        self.assertEqual(res.frame_count, src.frame_count)                      # duration invariant
        self.assertLessEqual(abs(res.video_duration_s - src.video_duration_s), src.fps.frame_duration)
        self.assertTrue(res.has_audio)
        self.assertEqual(audio_md5(out), audio_md5(data["clip"]))               # audio stream-copied bit-exact
        self.assertAlmostEqual(res.audio_start_s or 0.0, src.audio_start_s or 0.0, places=3)
        active = prep.resolved.path.active_ranges()
        self.assertTrue(active)
        inside = active[0][0] + (active[0][1] - active[0][0]) // 2
        before = max(0, active[0][0] - 12)
        # Outside camera ops the pixels match the existing caption render; inside they change.
        self.assertGreater(psnr(out, data["baseline"], before, before + 3), 38.0)
        self.assertLess(psnr(out, data["baseline"], inside, inside + 2), 30.0)
        artifacts = prep.artifacts
        for path in (artifacts.context, artifacts.plan, artifacts.resolved, artifacts.filter_script):
            self.assertTrue(path.is_file(), path)
        self.assertFalse(list(out.parent.glob("*.proedit-tmp*")))

    def test_burned_captions_are_not_zoomed(self) -> None:
        data = self.sets["demo flat"]
        prep = self.prepare("demo flat", captions=False, name="baseline_ass.mp4")
        self.assertTrue(prep.ready, prep.reason)
        outcome: dict = {}
        out = render_with_fallback(prep, caption_file=data["ass"], baseline=lambda: data["baseline"], outcome=outcome)
        self.assertEqual(outcome["status"], "pro_edit", outcome)
        self.assertEqual(outcome["captions"], "baseline_ass")
        a, b = prep.resolved.path.active_ranges()[0]
        # Flat source: a camera transform of the background is invisible, so any
        # difference would come from transformed captions. Captions are burned
        # AFTER the camera -> frames match the baseline caption render.
        self.assertGreater(psnr(out, data["baseline"], a + 3, b - 3), 45.0)

    def test_presentation_captions_are_not_zoomed(self) -> None:
        data = self.sets["demo flat"]
        prep = self.prepare("demo flat", name="presentation.mp4")
        self.assertTrue(prep.ready, prep.reason)
        self.assertIsNotNone(prep.presentation_ass)
        reference = self.root / "demo flat" / "presentation_reference.mp4"
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(data["clip"]), "-vf",
             f"subtitles=filename='{caption_renderer.escape_filter_path(prep.presentation_ass)}'",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy",
             str(reference)])
        outcome: dict = {}
        out = render_with_fallback(prep, caption_file=data["ass"], baseline=lambda: data["baseline"], outcome=outcome)
        self.assertEqual((outcome["status"], outcome["captions"]), ("pro_edit", "presentation"), outcome)
        a, b = prep.resolved.path.active_ranges()[0]
        self.assertGreater(psnr(out, reference, a + 3, b - 3), 45.0)
        # The presentation ASS differs from the baseline ASS only in presentation.
        self.assertNotEqual(prep.presentation_ass.read_bytes(), Path(data["ass"]).read_bytes())
        self.assertLess(psnr(out, data["baseline"], a + 3, b - 3), 45.0)

    def test_render_failure_falls_back_to_existing_output(self) -> None:
        data = self.sets["demo textured"]
        broken = replace(self.caps, script_option="-definitely_not_an_option")
        prep = self.prepare("demo textured", caps=broken, name="broken.mp4")
        self.assertTrue(prep.ready, prep.reason)
        outcome: dict = {}
        out = render_with_fallback(prep, caption_file=data["ass"], baseline=lambda: data["baseline"], outcome=outcome)
        self.assertEqual(outcome["status"], "fallback")
        self.assertEqual(out, data["baseline"])
        self.assertFalse(prep.output_path.exists())
        self.assertFalse(list(prep.output_path.parent.glob("*.proedit-tmp*")))

    def test_static_plan_uses_existing_render(self) -> None:
        data = self.sets["demo textured"]
        prep = self.prepare("demo textured", planner=StaticEditPlanner(), name="static.mp4", captions=False)
        self.assertEqual(prep.status, "static")
        outcome: dict = {}
        out = render_with_fallback(prep, caption_file=data["ass"], baseline=lambda: data["baseline"], outcome=outcome)
        self.assertEqual(out, data["baseline"])
        self.assertEqual(outcome["status"], "baseline")

    def test_static_plan_with_caption_presentation_renders_captions_only(self) -> None:
        data = self.sets["demo textured"]
        prep = self.prepare("demo textured", planner=StaticEditPlanner(), name="static_captions.mp4")
        self.assertTrue(prep.ready, prep.reason)
        self.assertTrue(prep.resolved.is_identity)
        self.assertIn("no camera change", prep.reason)
        outcome: dict = {}
        out = render_with_fallback(prep, caption_file=data["ass"], baseline=lambda: data["baseline"], outcome=outcome)
        self.assertEqual((outcome["status"], outcome["captions"]), ("pro_edit", "presentation"), outcome)
        src, res = probe_media(data["clip"]), probe_media(out)
        self.assertEqual((res.width, res.height, res.frame_count), (src.width, src.height, src.frame_count))
        self.assertEqual(audio_md5(out), audio_md5(data["clip"]))

    def test_presentation_render_failure_uses_camera_with_baseline_ass(self) -> None:
        data = self.sets["demo textured"]
        prep = self.prepare("demo textured", name="level2.mp4")
        self.assertTrue(prep.ready, prep.reason)
        prep.presentation_ass.write_text(prep.presentation_ass.read_text(encoding="utf-8-sig") + "\n",
                                         encoding="utf-8-sig")   # tampered after preparation
        outcome: dict = {}
        render_with_fallback(prep, caption_file=data["ass"], baseline=lambda: data["baseline"], outcome=outcome)
        self.assertEqual((outcome["status"], outcome["captions"]), ("pro_edit", "baseline_ass"), outcome)
        self.assertIn("presentation ASS changed", outcome["failures"][0])

    def test_variable_frame_rate_input_falls_back(self) -> None:
        vfr = self.root / "vfr.mp4"
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=30:duration=4",
             "-vf", "select='if(lt(n,60),not(mod(n,2)),1)'", "-fps_mode", "vfr", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", str(vfr)])
        data = self.sets["demo textured"]
        request = ProEditRequest(
            config=ProEditConfig(enabled=True, planner="rules"), timeline_path=data["timeline"], clip_index=1,
            edited_clip_path=vfr, caption_path=data["ass"], output_path=self.root / "vfr_out.mp4",
            artifact_dir=self.root / "artifacts" / "vfr", input_signature="vfr", force=True,
            planner=RuleBasedEditPlanner(), capabilities=self.caps)
        prep = prepare_pro_edit(request)
        self.assertEqual(prep.status, "fallback")
        self.assertIn("constant frame rate", prep.reason)

    def test_full_composition_with_mandatory_intro(self) -> None:
        data = self.sets["demo textured"]
        prep = self.prepare("demo textured", name="for_intro.mp4")
        outcome: dict = {}
        main = render_with_fallback(prep, caption_file=data["ass"], baseline=lambda: data["baseline"], outcome=outcome)
        self.assertEqual(outcome["status"], "pro_edit", outcome)
        teaser_path = self.root / "demo_teasers.json"
        teaser_path.write_text(json.dumps({"version": 1, "teasers": [{
            "clip_index": 1, "recommended": True,
            "edited": {"teaser_start": 3.2, "teaser_end": 4.6, "duration": 1.4}}]}), encoding="utf-8")
        intro_path = self.root / "demo_intros.json"
        intro_path.write_text(json.dumps({"version": 1,
                                          "inputs": {"timeline": str(data["timeline"]), "teaser": str(teaser_path)},
                                          "intros": [{"clip_index": 1, "recommended": True, "score": 9.0,
                                                      "title": "demo", "intro_text": "WAIT FOR IT",
                                                      "quality_gate": {"accepted": True}}]}), encoding="utf-8")
        final = intro_renderer.run_renderer(intro_json_path=intro_path, clip_index=1, edited_clip_path=data["clip"],
                                            captioned_preview_path=main, caption_path=data["ass"])[0]
        info, main_info = probe_media(final), probe_media(main)
        self.assertEqual((info.width, info.height), (main_info.width, main_info.height))
        self.assertTrue(info.has_audio)
        self.assertGreater(info.duration_s, main_info.duration_s)  # mandatory cold-open was prepended

    def test_plan_cache_avoids_second_model_call(self) -> None:
        from test_pro_edit_planner import FakeClient, ModelEditPlanner, good_plan_text

        client = FakeClient([good_plan_text(), good_plan_text()])
        planner = ModelEditPlanner("gpt-test", "medium", client=client)
        data = self.sets["demo textured"]

        def request(force: bool) -> ProEditRequest:
            return ProEditRequest(
                config=ProEditConfig(enabled=True, planner="model"), timeline_path=data["timeline"], clip_index=1,
                edited_clip_path=data["clip"], caption_path=data["ass"], output_path=self.root / "cache" / "p.mp4",
                artifact_dir=self.root / "artifacts" / "cache", input_signature="cache-sig", force=force,
                speaker_profile_path=data["profile"], planner=planner, capabilities=self.caps)

        first = prepare_pro_edit(request(force=True))
        second = prepare_pro_edit(request(force=False))
        self.assertEqual(len(client.responses.calls), 1)          # <= 1 planning call per short
        self.assertEqual(first.model_calls, 1)
        self.assertEqual(second.model_calls, 0)
        self.assertEqual(first.plan_id, second.plan_id)          # deterministic re-resolution
        self.assertEqual(first.resolved.to_dict(), second.resolved.to_dict())

    def test_prepare_failure_is_contained(self) -> None:
        data = self.sets["demo textured"]
        request = ProEditRequest(
            config=ProEditConfig(enabled=True, planner="rules"), timeline_path=self.root / "missing.json",
            clip_index=1, edited_clip_path=data["clip"], caption_path=data["ass"], output_path=self.root / "x.mp4",
            artifact_dir=self.root / "artifacts" / "missing", input_signature="m", force=True,
            capabilities=self.caps)
        prep = prepare_pro_edit(request)
        self.assertEqual(prep.status, "fallback")
        self.assertIn("timeline unreadable", prep.reason)
        self.assertTrue(prep.warnings)
        self.assertFalse(prep.ready)

    def test_ffmpeg_camera_matches_python_crop_math(self) -> None:
        from ai.editor.pro_edit.camera import IDENTITY, CameraPath, CameraSegment, CameraState, crop_window
        from ai.editor.pro_edit.ffmpeg_filters import build_perspective_filter

        src = self.sets["demo textured"]["clip"]
        info = probe_media(src)
        peak = CameraState(1.22, 0.62, 0.40)
        path = CameraPath(info.frame_count, (
            CameraSegment(10, 20, IDENTITY, peak, "ease_out_cubic", "e", "attack"),
            CameraSegment(20, 40, peak, peak, "linear", "e", "hold"),
            CameraSegment(40, 50, peak, IDENTITY, "smoothstep", "e", "release")))
        script = self.root / "math_graph.txt"
        script.write_text(build_perspective_filter(path, self.caps, interpolation="cubic"), encoding="utf-8")
        out = self.root / "math_check.mp4"
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), self.caps.script_option, str(script),
             "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(out)])

        def score(frame: int, reference_frame: int) -> float:
            x, y, w, h = crop_window(path.state_at(reference_frame), info.width, info.height)
            graph = (f"[0:v]select='eq(n,{frame})',crop={w:.4f}:{h:.4f}:{x:.4f}:{y:.4f}:exact=1,"
                     f"scale={info.width}:{info.height}:flags=bicubic,setpts=PTS-STARTPTS[r];"
                     f"[1:v]select='eq(n,{frame})',setpts=PTS-STARTPTS[t];[t][r]psnr")
            proc = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(src), "-i", str(out), "-filter_complex",
                                   graph, "-frames:v", "1", "-f", "null", "-"], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace")
            return float(re.search(r"average:([0-9.]+|inf)", proc.stderr).group(1).replace("inf", "99"))

        for frame in (12, 15, 25, 44):
            self.assertGreater(score(frame, frame), 30.0, f"frame {frame} geometry mismatch")
        self.assertLess(score(12, 16), 25.0)  # sensitivity: a 4-frame offset is detected

    def test_release_round_off_below_one_renders(self) -> None:
        """Regression: 1.16 - 0.16 == 0.9999999999999999 used to crash perspective."""
        from ai.editor.pro_edit.camera import IDENTITY, CameraPath, CameraSegment, CameraState
        from ai.editor.pro_edit.ffmpeg_filters import build_perspective_filter

        src = self.sets["demo textured"]["clip"]
        info = probe_media(src)
        peak = CameraState(1.16, 0.5, 0.5)
        path = CameraPath(info.frame_count, (CameraSegment(10, 16, IDENTITY, peak, "ease_out_cubic"),
                                             CameraSegment(16, 30, peak, peak), CameraSegment(30, 38, peak, IDENTITY,
                                                                                               "smoothstep")))
        script = self.root / "roundoff.txt"
        script.write_text(build_perspective_filter(path, self.caps), encoding="utf-8")
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), self.caps.script_option, str(script),
             "-an", "-f", "null", "-"])


if __name__ == "__main__":
    unittest.main()
