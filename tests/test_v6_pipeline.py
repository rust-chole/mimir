"""MIMIR production path end-to-end through the REAL run_pipeline orchestration (offline).

Real FFmpeg stages (pacing cut, caption ASS + burn-in, Pro Edit camera render,
intro render, publish) on a synthetic source; only paid/interactive model
stages are replaced by deterministic fakes (tests/pipeline_harness.py).
Runs against a disposable repository copy under a path with spaces, an
apostrophe and non-ASCII characters. Proves:

* the production path runs caption truth -> evidence director -> render ->
  pixel proof -> rendered-MP4 QC -> final gate, with no fallback;
* ``--rerender`` (alias ``--force-v6``) recomputes only the presentation /
  downstream stages while every upstream cache (transcription ... intro
  analysis) is reused.
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

import pro_edit_fixtures as fx

ROOT = Path(__file__).resolve().parent.parent
HARNESS = Path(__file__).resolve().parent / "pipeline_harness.py"
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
UPSTREAM = ("transcription", "clip_analysis", "pacing", "timeline", "speaker_preflight", "pacing_cut", "captions",
            "teaser_analysis", "intro_analysis")
V6_STAGES = ("caption_truth_v6", "caption_render", "intro_final_base", "publish")


def make_video(path: Path, seconds: float, source: str = "testsrc2") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joiner = ":" if "=" in source else "="
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"{source}{joiner}size=320x180:rate=30:duration={seconds}", "-f", "lavfi", "-i",
                    f"sine=frequency=300:sample_rate=48000:duration={seconds}", "-shortest", "-c:v", "libx264",
                    "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
                   check=True, capture_output=True)
    return path


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available on PATH")
class PacedClipIdentityTests(unittest.TestCase):
    """The paced clip handed downstream is always THIS timeline clip's own render.

    Regression: after the temp cleanup removed a clip's paced video, a glob over
    edited_clips/ adopted another source's newer clip_01 video (timeline 39.54 s
    vs media 25.23 s) and every downstream stage consumed foreign media."""

    @classmethod
    def setUpClass(cls) -> None:
        from ai import shorts_pipeline

        cls.sp = shorts_pipeline
        cls._tmp = tempfile.TemporaryDirectory(prefix="mimir paced ş'")
        cls.root = Path(cls._tmp.name)
        cls.five = make_video(cls.root / "five.mp4", 5)
        cls.three = make_video(cls.root / "three.mp4", 3, "color=c=0x303030")
        # Raw clip 6 s, final persisted cut 2-3 s -> the timeline declares a 5 s paced clip.
        cls.timeline_clip = {"clip_index": 1, "cut_ranges": [{"start": 2.0, "end": 3.0}],
                             "source": {"absolute_start": 10.0, "absolute_end": 16.0, "duration": 6.0}}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_only_the_exact_expected_clip_is_ever_resolved(self) -> None:
        foreign = self.root / "edited_clips" / "other source" / "clip_01_Other_edited.mp4"
        foreign.parent.mkdir(parents=True)
        shutil.copy2(self.three, foreign)                          # another source's (newer) clip_01 video
        expected = self.root / "edited_clips" / "this source" / "clip_01_This Story_edited.mp4"
        self.assertIsNone(self.sp._resolve_edited_clip(expected))       # cleaned up -> re-render, never adopt
        expected.parent.mkdir(parents=True)
        shutil.copy2(self.five, expected)
        self.assertEqual(self.sp._resolve_edited_clip(expected), expected.resolve())

    def test_reuse_requires_the_paced_duration_the_timeline_declares(self) -> None:
        self.assertAlmostEqual(self.sp._declared_paced_duration(self.timeline_clip), 5.0, places=3)
        self.assertTrue(self.sp._paced_clip_matches(self.five, self.timeline_clip))
        self.assertFalse(self.sp._paced_clip_matches(self.three, self.timeline_clip))
        self.assertFalse(self.sp._paced_clip_matches(None, self.timeline_clip))
        self.assertFalse(self.sp._paced_clip_matches(self.root / "missing.mp4", self.timeline_clip))

    def test_identical_re_render_keeps_the_clock_downstream_caches_recorded(self) -> None:
        path = self.root / "render ✓.mp4"
        shutil.copy2(self.five, path)
        first = self.sp._media_identity(path)
        record = {"signature": "sig", "path": str(path), "media": first}
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, first["mtime_ns"] + 5_000_000_000))   # a re-render's new clock
        kept = self.sp._keep_render_clock(path, record, "sig")
        self.assertEqual(kept.get("clock"), "restored_identical_render")
        self.assertEqual(path.stat().st_mtime_ns, first["mtime_ns"])
        # Another pacing signature or different bytes keep the new clock (dependents re-run).
        os.utime(path, ns=(stat.st_atime_ns, first["mtime_ns"] + 5_000_000_000))
        self.assertNotIn("clock", self.sp._keep_render_clock(path, record, "other sig"))
        shutil.copy2(self.three, path)
        self.assertNotIn("clock", self.sp._keep_render_clock(path, record, "sig"))
        self.assertNotEqual(path.stat().st_mtime_ns, first["mtime_ns"])


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available on PATH")
@fx.needs_opencv
class RunStatusEndToEndTests(unittest.TestCase):
    """The REAL pipeline fails right after the short reached final/, then reruns cleanly."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="mimir run status e2e ş'")
        base = Path(cls._tmp.name)
        cls.copy = base / "repo copy"
        shutil.copytree(ROOT, cls.copy, ignore=shutil.ignore_patterns(".git", "vod_output", "__pycache__", ".venv*",
                                                                     "venv*", "_v5_*", "_pass2_*", ".env", "*.zip"))
        cls.video = make_video(base / "media ✓" / "status vod.mp4", 30)
        cls.results = {}
        for mode in ("fail_after_publish", "rerender"):
            out = base / f"{mode}.json"
            proc = subprocess.run([sys.executable, str(HARNESS), "--root", str(cls.copy), "--video", str(cls.video),
                                   "--mode", mode, "--out", str(out)], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=900,
                                  env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
            if proc.returncode != 0:
                raise AssertionError(f"harness mode {mode} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
            cls.results[mode] = json.loads(out.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_a_failure_after_publication_names_the_published_short(self) -> None:
        result = self.results["fail_after_publish"]
        self.assertEqual(result["error_type"], "OSError")
        self.assertEqual(result["run_status"], "failed_after_publish")
        self.assertIn("review sheet write failure", result["run_error"])
        self.assertTrue(result["published_exists"])
        self.assertEqual(result["published_output"], result["final_output_state"])
        self.assertTrue(result["published_output"].endswith("status vod_short.mp4"))

    def test_a_successful_rerun_clears_the_stale_failure(self) -> None:
        result = self.results["rerender"]
        self.assertIn(result["status"], ("published", "published_degraded"))
        self.assertEqual(result["run_status"], "success")
        self.assertIsNone(result["run_error"])


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available on PATH")
class PublishFailureEndToEndTests(unittest.TestCase):
    """A rerun whose publish copy comes out incomplete leaves the earlier short in
    final/ byte for byte, and its state says it failed without publishing."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="mimir publish e2e ş'")
        base = Path(cls._tmp.name)
        cls.copy = base / "repo copy"
        shutil.copytree(ROOT, cls.copy, ignore=shutil.ignore_patterns(".git", "vod_output", "__pycache__", ".venv*",
                                                                     "venv*", "_v5_*", "_pass2_*", ".env", "*.zip"))
        cls.video = make_video(base / "media ✓" / "publish vod.mp4", 30)
        cls.results = {}
        for mode in ("run", "publish_copy_broken"):
            out = base / f"{mode}.json"
            proc = subprocess.run([sys.executable, str(HARNESS), "--root", str(cls.copy), "--video", str(cls.video),
                                   "--mode", mode, "--out", str(out)], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=900,
                                  env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
            if proc.returncode != 0:
                raise AssertionError(f"harness mode {mode} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
            cls.results[mode] = json.loads(out.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_earlier_short_survives_and_the_run_reports_no_publication(self) -> None:
        good, broken = self.results["run"], self.results["publish_copy_broken"]
        self.assertEqual(good["status"], "published")
        self.assertEqual(broken["error_type"], "ShortsPipelineError")
        self.assertIn("Publish kopyası eksik", broken["error"])
        self.assertEqual((broken["run_status"], broken["publish_status"]), ("failed", None))
        self.assertIsNone(broken["published_output"])              # the CLI says nothing was published
        self.assertTrue(broken["published_exists"])
        self.assertEqual(broken["published_md5"], good["final_md5"])  # the earlier short, byte for byte
        self.assertEqual(broken["staged_leftovers"], [])


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available on PATH")
class SpeakerColorEndToEndTests(unittest.TestCase):
    """The REAL pipeline with two voices taking turns (A B A B ...): no naming prompt,
    the burned captions colour each voice on the one main lane, and the frozen truth,
    QC and the final gate accept the colours."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="mimir colours e2e ş'")
        base = Path(cls._tmp.name)
        cls.copy = base / "repo copy"
        shutil.copytree(ROOT, cls.copy, ignore=shutil.ignore_patterns(".git", "vod_output", "__pycache__", ".venv*",
                                                                     "venv*", "_v5_*", "_pass2_*", ".env", "*.zip"))
        video = make_video(base / "media ✓" / "colour vod.mp4", 30)
        out = base / "run.json"
        proc = subprocess.run([sys.executable, str(HARNESS), "--root", str(cls.copy), "--video", str(video),
                               "--mode", "run", "--out", str(out)], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=900, stdin=subprocess.DEVNULL,
                              env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
                                   "HARNESS_SPEAKERS": "dual"})
        if proc.returncode != 0:
            raise AssertionError(f"harness failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
        cls.result = json.loads(out.read_text(encoding="utf-8"))
        output = cls.copy / "vod_output"
        cls.presentation = [p for p in output.rglob("*.ass")
                            if "MIMIR Pro Edit Caption Presentation" in p.read_text(encoding="utf-8-sig")]
        cls.truths = sorted(output.rglob("*_caption_truth_v*.json"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_short_is_published_with_the_speaker_colours_accepted(self) -> None:
        self.assertEqual(self.result["status"], "published", self.result.get("v6_state"))
        self.assertEqual(self.result["run_status"], "success")
        gate = (self.result.get("v6_state") or {}).get("gate") or {}
        self.assertNotIn("speaker_ownership", " ".join(map(str, gate.get("failed") or [])))

    def test_the_burned_captions_colour_each_voice_on_the_main_lane(self) -> None:
        from ai.editor.pro_edit import caption_presentation as cp

        self.assertTrue(self.presentation)
        text = self.presentation[-1].read_text(encoding="utf-8-sig")
        events = [line for line in text.splitlines() if line.startswith("Dialogue:")]
        self.assertTrue(events)
        self.assertEqual({line.split(",")[3] for line in events}, {"MimirMain"})       # sequential turns: one lane
        self.assertIn(cp.PALETTES["main"].active, text)                              # voice A
        self.assertIn(cp.PALETTES["secondary"].active, text)                         # voice B

    def test_the_frozen_truth_carries_the_colours(self) -> None:
        self.assertTrue(self.truths)
        truth = json.loads(self.truths[-1].read_text(encoding="utf-8"))
        self.assertEqual(truth.get("speaker_issues"), [])
        self.assertEqual({row[6] for row in truth["words"]}, {"A", "B"})             # frozen with the words


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available on PATH")
@fx.needs_opencv
class V6PipelineEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="mimir v6 e2e ş'")
        base = Path(cls._tmp.name)
        cls.copy = base / "repo copy"
        shutil.copytree(ROOT, cls.copy, ignore=shutil.ignore_patterns(".git", "vod_output", "__pycache__", ".venv*",
                                                                     "venv*", "_v5_*", "_pass2_*", ".env", "*.zip"))
        cls.video = base / "media ✓" / "harness vod.mp4"
        cls.video.parent.mkdir()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=640x360:rate=30000/1001:duration=30", "-f", "lavfi", "-i",
                        "anoisesrc=color=pink:sample_rate=48000:duration=30:amplitude=0.1", "-shortest", "-c:v",
                        "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        "-b:a", "128k", "-ac", "2", str(cls.video)], check=True, capture_output=True)
        cls.results = {}

        def run(mode: str, key: str) -> None:
            out = base / f"{key}.json"
            proc = subprocess.run([sys.executable, str(HARNESS), "--root", str(cls.copy), "--video", str(cls.video),
                                   "--mode", mode, "--out", str(out)], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=900,
                                  env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
            if proc.returncode != 0:
                raise AssertionError(f"harness mode {mode} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
            cls.results[key] = json.loads(out.read_text(encoding="utf-8"))
            cls.results[key]["stdout"] = proc.stdout

        for mode, key in (("run", "v6"), ("rerender", "v6_force")):
            run(mode, key)
        # The incident: the temp cleanup removed this clip's paced video while another
        # source's newer clip_01 video is on disk. The rerun must re-render THIS clip.
        edited_root = cls.copy / "vod_output" / "edited_clips"
        next(edited_root.glob("*/clip_01_*_edited.mp4")).unlink()
        make_video(edited_root / "zz other source" / "clip_01_Other Story_edited.mp4", 12)
        run("rerender", "v6_rerun_after_cleanup")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def manifest(self, mode: str) -> dict:
        return json.loads(Path(self.results[mode]["v6"]["manifest"]).read_text(encoding="utf-8"))

    def test_v6_runs_the_real_path_and_passes_the_final_gate(self) -> None:
        result = self.results["v6"]
        self.assertEqual(result["v6"]["status"], "passed", result["v6"])
        self.assertEqual(result["status"], "published")
        self.assertEqual(result["v6"]["fallbacks"], 0)
        self.assertEqual(result["stages"]["caption_truth_v6"], "done")
        self.assertEqual(result["pro_edit"]["render"]["status"], "pro_edit")
        manifest = self.manifest("v6")
        checks = {c["check"]: c["status"] for c in manifest["gate"]["checks"]}
        for name in ("output_file", "caption_truth_frozen", "word_timing", "speaker_ownership", "verified_names",
                     "caption_presentation", "story_preserved", "camera_plan", "hold_reasons",
                     "story_regions_visible", "camera_pixels_main", "camera_pixels_final", "v6_render_path",
                     "no_blocking_fallback", "no_captions_in_intro", "caption_words_timing", "main_av_sync"):
            self.assertEqual(checks.get(name), "pass", (name, checks))
        for key in ("caption_truth", "camera_plan", "render_proof", "manifest"):
            self.assertTrue(Path(manifest["artifacts"][key]).is_file(), key)
        self.assertIn("[MIMIR_V6] status=passed", result["stdout"])

    def test_force_v6_reruns_only_v6_dependent_stages(self) -> None:
        stages = self.results["v6_force"]["stages"]
        for name in UPSTREAM:
            self.assertEqual(stages[name], "skipped", (name, stages))
        for name in V6_STAGES:
            self.assertEqual(stages[name], "done", (name, stages))
        self.assertEqual(stages["pro_edit"], "ready")
        self.assertEqual(self.results["v6_force"]["v6"]["status"], "passed")
        self.assertTrue(self.manifest("v6_force")["force"])

    def test_rerun_after_cleanup_rerenders_this_clip_never_a_foreign_one(self) -> None:
        result = self.results["v6_rerun_after_cleanup"]
        stages = result["stages"]
        self.assertEqual(stages["pacing_cut"], "done", stages)            # re-rendered, foreign clip_01 ignored
        self.assertEqual(stages["pro_edit"], "ready", result["pro_edit"])  # EditContext accepted the paced media
        self.assertEqual(result["v6"]["status"], "passed", result["v6"])
        # Byte-identical re-render keeps the recorded clock: no downstream (paid) stage re-runs.
        self.assertIn("byte-identical", result["stdout"])
        for name in ("captions", "teaser_analysis", "intro_analysis"):
            self.assertEqual(stages[name], "skipped", (name, stages))


if __name__ == "__main__":
    unittest.main()
