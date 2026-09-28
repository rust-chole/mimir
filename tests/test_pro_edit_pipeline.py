"""End-to-end run_pipeline regression (offline, real FFmpeg, fake model stages).

Runs tests/pipeline_harness.py against a disposable COPY of this repository so
no artifact is written into the working tree. Skipped without FFmpeg.

There is ONE production path. These tests prove, on real renders:
* a normal run publishes only after the rendered MP4 passed final QC;
* the cold open is moving peak footage with real audio, no speech captions,
  and a hard restart; the headline is optional;
* a Pro Edit render failure or crash is REJECTED (nothing reaches final/),
  instead of silently publishing the legacy render.
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


def make_source(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=640x360:rate=30000/1001:duration=30", "-f", "lavfi", "-i",
                    "anoisesrc=color=pink:sample_rate=48000:duration=30:amplitude=0.1", "-shortest", "-c:v",
                    "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-b:a", "128k", "-ac", "2", str(path)], check=True, capture_output=True)
    return path


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available on PATH")
@fx.needs_opencv
class PipelineEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="mimir e2e ")
        base = Path(cls._tmp.name)
        cls.copy = base / "repo copy"
        shutil.copytree(ROOT, cls.copy, ignore=shutil.ignore_patterns(".git", "vod_output", "__pycache__", ".venv*",
                                                                     "venv*", ".env"))
        cls.results = {}
        for mode in ("run", "no_headline", "broken_render", "stage_crash", "planner_down"):
            # One source per mode: a rejected run must be judged on its own final/ state.
            video = make_source(base / "media ✓" / f"harness {mode}.mp4")
            out = base / f"{mode}.json"
            # UTF-8 mode: on Windows a piped stdout defaults to the ANSI code page.
            proc = subprocess.run([sys.executable, str(HARNESS), "--root", str(cls.copy), "--video", str(video),
                                   "--mode", mode, "--out", str(out)], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=900,
                                  env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
            if proc.returncode != 0:
                raise AssertionError(f"harness mode {mode} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
            cls.results[mode] = json.loads(out.read_text(encoding="utf-8"))
            cls.results[mode]["stdout"] = proc.stdout

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def qc(self, mode: str) -> dict[str, dict]:
        return {row["check"]: row for row in self.results[mode]["final_qc"]["checks"]}

    def test_normal_run_publishes_only_a_qc_verified_short(self) -> None:
        run = self.results["run"]
        self.assertEqual(run["status"], "published", run.get("v6"))
        self.assertEqual(run["pro_edit"]["render"]["status"], "pro_edit")
        self.assertEqual(run["pro_edit"]["render"]["captions"], "presentation")
        self.assertEqual(run["pro_edit"]["intro_verification"]["status"], "verified")
        for name, row in self.qc("run").items():
            self.assertEqual(row["status"], "pass", (name, row["detail"]))
        for name in ("no_captions_in_intro", "intro_is_source_footage", "intro_moving", "intro_audio",
                     "main_matches_render", "main_av_sync", "caption_words_timing", "composition_clock"):
            self.assertIn(name, self.qc("run"))
        self.assertTrue(Path(run["final_output"]).with_suffix(".qc.json").is_file())
        self.assertEqual(run["stages"]["publish"], "done")

    def test_cold_open_is_a_hard_cut_from_clean_moving_footage(self) -> None:
        doc = self.results["run"]["final_timeline"]
        self.assertEqual(doc["restart"]["transition"], "hard_cut")
        self.assertIs(doc["intro"]["normal_captions"], False)
        self.assertTrue(doc["intro"]["headline"])
        final = probe_media(self.results["run"]["final_output"])
        frame = final.fps.frame_duration
        self.assertLessEqual(abs(final.duration_s - doc["expected_final_duration"]), 2 * frame + 0.05)
        paced = doc["intro"]["paced"]
        self.assertAlmostEqual(doc["intro"]["duration"], paced[1] - paced[0], places=3)

    def test_no_headline_is_a_legitimate_cold_open(self) -> None:
        result = self.results["no_headline"]
        self.assertEqual(result["status"], "published")
        self.assertEqual(result["final_timeline"]["intro"]["headline"], "")
        self.assertEqual(self.qc("no_headline")["intro_headline"]["status"], "pass")
        self.assertEqual(self.qc("no_headline")["no_captions_in_intro"]["status"], "pass")

    def assert_rejected(self, mode: str) -> dict:
        result = self.results[mode]
        self.assertIn("YAYINLANMADI", result["error"])
        self.assertEqual(result["run_status"], "rejected")
        self.assertEqual(result["publish_status"], "rejected")
        self.assertFalse(result["published_exists"])
        self.assertTrue(any(name.endswith("_REJECTED.mp4") for name in result["rejected"]))
        self.assertTrue(any(name.endswith("_REJECTED.qc.json") for name in result["rejected"]))
        return result

    def test_pro_edit_render_failure_is_rejected_not_silently_published(self) -> None:
        result = self.assert_rejected("broken_render")
        failed = result["v6_state"]["gate"]["failed"]
        self.assertIn("no_blocking_fallback", failed)
        self.assertIn("v6_render_path", failed)

    def test_pro_edit_stage_crash_is_rejected_not_silently_published(self) -> None:
        result = self.assert_rejected("stage_crash")
        self.assertIn("no_blocking_fallback", result["v6_state"]["gate"]["failed"])

    def test_planner_down_is_still_verified_before_publishing(self) -> None:
        down = self.results["planner_down"]
        self.assertIn(down["status"], ("published", "published_degraded"))
        self.assertEqual(down["pro_edit"]["render"]["captions"], "presentation")
        for name, row in self.qc("planner_down").items():
            self.assertEqual(row["status"], "pass", (name, row["detail"]))


if __name__ == "__main__":
    unittest.main()
