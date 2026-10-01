"""Render backend through the REAL run_pipeline (offline, real FFmpeg, simulated GPU encoder).

Runs tests/pipeline_harness.py against a disposable COPY of this repository. No GPU
is needed: the harness makes detection report a working 'NVENC' whose encoder is a
libx264 stand-in (other settings than the CPU profile) or an encoder FFmpeg lacks.
Skipped without FFmpeg / OpenCV. Multi-minute: verify_unified_clean.py runs it only
with MIMIR_VERIFY_E2E=1.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import pro_edit_fixtures as fx  # noqa: F401  (sys.path)
from ai.editor import render_backend as rb

ROOT = Path(__file__).resolve().parent.parent
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available on PATH")
@fx.needs_opencv
class HardwareEndToEndTests(unittest.TestCase):
    """The REAL run_pipeline on real FFmpeg with a simulated GPU encoder (tests/pipeline_harness.py).

    hw_ok        -> every encode goes through the 'hardware' backend, QC passes;
    cpu_after_hw -> the same source rerun on the CPU: only render stages re-render, the
                    paced clip is pinned (same edit, other encoder), upstream caches are reused;
    hw_broken    -> detection passed but the first real encode fails: one CPU retry, the
                    broken encoder is never tried again, the short still passes the same QC."""

    UPSTREAM = ("transcription", "clip_analysis", "pacing", "timeline", "speaker_preflight", "captions",
                "caption_truth_v6", "teaser_analysis", "intro_analysis")

    @classmethod
    def setUpClass(cls) -> None:
        from concurrent.futures import ThreadPoolExecutor
        import sys

        harness = Path(__file__).resolve().parent / "pipeline_harness.py"
        cls._tmp = tempfile.TemporaryDirectory(prefix="mimir hw e2e ")
        base = Path(cls._tmp.name)
        cls.copy = base / "repo copy"
        shutil.copytree(ROOT, cls.copy, ignore=shutil.ignore_patterns(".git", "vod_output", "__pycache__", ".venv*",
                                                                     "venv*", ".env"))

        def source(name: str) -> Path:
            path = base / "media ✓" / f"{name}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                            "testsrc2=size=640x360:rate=30000/1001:duration=30", "-f", "lavfi", "-i",
                            "anoisesrc=color=pink:sample_rate=48000:duration=30:amplitude=0.1", "-shortest",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a",
                            "aac", "-b:a", "128k", "-ac", "2", str(path)], check=True, capture_output=True)
            return path

        def run(mode: str, video: Path) -> tuple[str, dict]:
            out = base / f"{mode}.json"
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
            env.pop("HARNESS_RENDER_BACKEND", None)
            proc = subprocess.run([sys.executable, str(harness), "--root", str(cls.copy), "--video", str(video),
                                   "--mode", mode, "--out", str(out)], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=1500, env=env)
            if proc.returncode != 0:
                raise AssertionError(f"harness mode {mode} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
            return mode, {**json.loads(out.read_text(encoding="utf-8")), "stdout": proc.stdout}

        def chain() -> list[tuple[str, dict]]:
            video = source("hw switch vod")
            return [run("hw_ok", video), run("cpu_after_hw", video)]

        with ThreadPoolExecutor(max_workers=2) as pool:
            switched = pool.submit(chain)
            broken = pool.submit(lambda: run("hw_broken", source("hw broken vod")))
            cls.results = dict([*switched.result(), broken.result()])

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def assert_published_and_qc_clean(self, mode: str) -> None:
        result = self.results[mode]
        self.assertEqual(result["status"], "published", (mode, result.get("v6")))
        self.assertEqual(result["v6"]["status"], "passed", result["v6"])
        for row in result["final_qc"]["checks"]:
            self.assertEqual(row["status"], "pass", (mode, row["check"], row["detail"]))

    def test_simulated_hardware_encodes_every_render_and_passes_the_same_qc(self) -> None:
        self.assert_published_and_qc_clean("hw_ok")
        backend = self.results["hw_ok"]["render_backend"]
        self.assertEqual(backend["selected"]["backend"], rb.NVENC)
        self.assertFalse(backend["fallback_occurred"])
        for stage in ("pacing_cut", "pro_edit_main", "intro_render"):
            self.assertEqual(backend["stages"][stage]["backend"], rb.NVENC, (stage, backend["stages"]))
        self.assertTrue(all(row["backend"] == rb.NVENC for row in backend["encodes"]))
        rows = {row["label"]: row for row in self.results["hw_ok"]["render_profile"]["rows"]}
        for label in ("Pacing encode", "Caption / Pro Edit render", "Intro render", "Final QC"):
            self.assertIn(label, rows)
        self.assertEqual(rows["Pacing encode"]["backend"], rb.NVENC)
        self.assertGreater(self.results["hw_ok"]["render_profile"]["total_seconds"], 0)

    def test_switching_back_to_cpu_rerenders_only_render_stages(self) -> None:
        self.assert_published_and_qc_clean("cpu_after_hw")
        result = self.results["cpu_after_hw"]
        stages = result["stages"]
        for name in self.UPSTREAM:
            self.assertEqual(stages[name], "skipped", (name, stages))      # no paid / semantic stage re-ran
        for name in ("pacing_cut", "caption_render", "intro_final_base", "publish"):
            self.assertEqual(stages[name], "done", (name, stages))         # render identity changed
        self.assertEqual(result["pacing_clock"], "pinned_equivalent_render")
        self.assertIn("only the video encoder differs", result["stdout"])
        self.assertEqual(result["render_backend"]["selected"]["backend"], rb.CPU)
        self.assertNotEqual(result["final_md5"], self.results["hw_ok"]["final_md5"])

    def test_runtime_hardware_failure_retries_once_and_disables_the_encoder(self) -> None:
        self.assert_published_and_qc_clean("hw_broken")
        backend = self.results["hw_broken"]["render_backend"]
        self.assertTrue(backend["fallback_occurred"])
        self.assertEqual(backend["unhealthy"][rb.NVENC]["stage"], "pacing_cut")
        hardware = [row for row in backend["encodes"] if row["backend"] == rb.NVENC]
        self.assertEqual(len(hardware), 1)                                  # never tried again
        self.assertFalse(hardware[0]["ok"])
        cpu = [row for row in backend["encodes"] if row["backend"] == rb.CPU]
        self.assertTrue(cpu and all(row["ok"] for row in cpu))
        self.assertEqual(cpu[0].get("fallback_from"), rb.NVENC)
        self.assertTrue(backend["stages"]["pacing_cut"]["fallback"])
        self.assertIn("h264_mimir_harness_missing", backend["fallback_reason"])
        self.assertIn("retried once with libx264", self.results["hw_broken"]["stdout"])


if __name__ == "__main__":
    unittest.main()
