"""End-to-end run_pipeline regression (offline, real FFmpeg, fake model stages).

Runs tests/pipeline_harness.py against a disposable COPY of this repository so
no artifact is written into the working tree. ~1 minute; skipped without FFmpeg.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pro_edit_fixtures as fx  # noqa: F401
from ai.editor.pro_edit.media import probe_media

ROOT = Path(__file__).resolve().parent.parent
HARNESS = Path(__file__).resolve().parent / "pipeline_harness.py"
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available on PATH")
class PipelineEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="mimir e2e ")
        base = Path(cls._tmp.name)
        cls.copy = base / "repo copy"
        shutil.copytree(ROOT, cls.copy, ignore=shutil.ignore_patterns(".git", "vod_output", "__pycache__", ".venv*",
                                                                     "venv*", "_v5_*", ".env"))
        cls.video = base / "media ✓" / "harness vod.mp4"
        cls.video.parent.mkdir()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=640x360:rate=30000/1001:duration=30", "-f", "lavfi", "-i",
                        "sine=frequency=300:sample_rate=48000:duration=30", "-shortest", "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                        "-ac", "2", str(cls.video)], check=True, capture_output=True)
        cls.results = {}
        for mode in ("off", "on", "on_broken", "on_planner_down", "on_stage_crash", "on_captions_off",
                     "on_caption_crash"):
            out = base / f"{mode}.json"
            # UTF-8 mode: on Windows a piped stdout defaults to the ANSI code page (cp1254 here),
            # which cannot encode the pipeline's console banner.
            proc = subprocess.run([sys.executable, str(HARNESS), "--root", str(cls.copy), "--video", str(cls.video),
                                   "--mode", mode, "--out", str(out)], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=900,
                                  env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
            if proc.returncode != 0:
                raise AssertionError(f"harness mode {mode} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
            cls.results[mode] = json.loads(out.read_text(encoding="utf-8"))
            # Keep each mode's final: later runs overwrite the published path.
            final = Path(cls.results[mode]["final_output"])
            kept = base / f"{mode}_final.mp4"
            shutil.copy2(final, kept)
            cls.results[mode]["kept_final"] = str(kept)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_feature_off_never_imports_or_writes_pro_edit(self) -> None:
        off = self.results["off"]
        self.assertIsNone(off["pro_edit"])
        self.assertFalse(off["pro_edit_imported"])
        self.assertIsNone(off.get("v6"))                 # MIMIR V6 is default OFF: not imported, no artifacts
        self.assertFalse(off.get("v6_imported"))
        self.assertEqual(off["pro_edit_artifacts"], [])
        self.assertNotIn("pro_edit", off["stages"])
        self.assertEqual(off["warnings"], [])

    def assert_same_structure(self, result: dict) -> None:
        a, b = probe_media(result["kept_final"]), probe_media(self.results["off"]["kept_final"])
        self.assertEqual((a.width, a.height), (b.width, b.height))
        self.assertEqual(a.fps.fraction, b.fps.fraction)
        self.assertEqual(a.frame_count, b.frame_count)  # intro + main timing unchanged
        self.assertLessEqual(abs(a.duration_s - b.duration_s), b.fps.frame_duration + 0.01)
        self.assertTrue(a.has_audio)

    def test_feature_on_changes_presentation_not_structure(self) -> None:
        on, off = self.results["on"], self.results["off"]
        self.assertEqual(on["pro_edit"]["status"], "ready")
        self.assertEqual(on["pro_edit"]["render"]["status"], "pro_edit")
        self.assertEqual(on["pro_edit"]["render"]["captions"], "presentation")
        self.assertEqual(on["pro_edit"]["render"]["failures"], [])
        self.assertEqual(on["pro_edit"]["captions"]["status"], "presentation")
        self.assertEqual(on["pro_edit"]["intro_render"]["status"], "pro_edit")      # existing intro, new camera
        self.assertEqual(on["pro_edit"]["intro_verification"]["status"], "verified")  # teaser + handoff unchanged
        self.assertEqual(on["warnings"], [])
        self.assertEqual(on["stages"]["caption_render"], "done")
        self.assertNotEqual(on["final_md5"], off["final_md5"])
        self.assertTrue(any(name.endswith("_captions_v1.ass") for name in on["pro_edit_artifacts"]))
        self.assertTrue(any("caption_presentation" in name for name in on["pro_edit_artifacts"]))
        self.assert_same_structure(on)

    def test_pro_edit_failures_degrade_to_existing_output(self) -> None:
        off = self.results["off"]["final_md5"]
        broken = self.results["on_broken"]
        self.assertEqual(broken["final_md5"], off)
        self.assertEqual(broken["stages"]["caption_render"], "fallback")
        self.assertTrue(any("Pro Edit render" in w for w in broken["warnings"]))
        crash = self.results["on_stage_crash"]
        self.assertEqual(crash["final_md5"], off)
        self.assertEqual(crash["pro_edit"]["status"], "fallback")
        self.assertTrue(any("beklenmedik" in w for w in crash["warnings"]))
        # Planner unavailable + caption presentation off: exactly the existing output.
        quiet = self.results["on_captions_off"]
        self.assertEqual(quiet["final_md5"], off)
        self.assertEqual(quiet["pro_edit"]["status"], "static")
        self.assertEqual(quiet["pro_edit"]["captions"]["status"], "disabled")
        self.assertTrue(any("static fallback" in w for w in quiet["warnings"]))

    def test_planner_down_keeps_caption_presentation(self) -> None:
        down = self.results["on_planner_down"]
        self.assertEqual(down["pro_edit"]["status"], "ready")        # captions only, no camera change
        self.assertIn("no camera change", down["pro_edit"]["reason"])
        self.assertEqual(down["pro_edit"]["render"]["captions"], "presentation")
        self.assertEqual(down["pro_edit"]["intro_camera"], False)
        self.assertTrue(any("static fallback" in w for w in down["warnings"]))
        self.assertNotEqual(down["final_md5"], self.results["off"]["final_md5"])
        self.assert_same_structure(down)

    def test_caption_presentation_failure_keeps_camera_with_baseline_ass(self) -> None:
        crash = self.results["on_caption_crash"]
        self.assertEqual(crash["pro_edit"]["status"], "ready")
        self.assertEqual(crash["pro_edit"]["captions"]["status"], "baseline_ass")
        self.assertEqual(crash["pro_edit"]["render"]["captions"], "baseline_ass")
        self.assertTrue(any("caption presentation" in w for w in crash["warnings"]))
        self.assertNotEqual(crash["final_md5"], self.results["on"]["final_md5"])
        self.assert_same_structure(crash)


if __name__ == "__main__":
    unittest.main()
