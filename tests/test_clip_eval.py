"""Clip evaluation: Tier 1 metrics, blind labeling, cached runs of Stage 2 in isolation, compare + gate,
the LLM judge and its calibration, the holdout guard.

Hermetic: no model is ever called. Stage 2 is replaced by a fake (in-process, or in the real worker
subprocess through MIMIR_CLIP_EVAL_TEST_ANALYZER), MIMIR_CLIP_EVAL_OFFLINE=1 makes the budget meter refuse
every OpenAI SDK call, and the judge gets a fake client. Paths contain Turkish letters, spaces and an
apostrophe on purpose.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from ai.clip_eval import calibrate, cli, guard, judge, metrics, pricing, report, reviews, runner, sheet, variants, worker
from ai.clip_eval import cases as case_store
from ai.clip_eval.cases import Labels, Moment, NeverPick

import clip_eval_fakes as fakes

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
PREFIX = "mimir clip ş ğ İ 'x' "


def setUpModule() -> None:
    os.environ[pricing.OFFLINE_ENV] = "1"           # any OpenAI SDK call in this process is refused


def labels(*moments: tuple[float, float, str], never: tuple[tuple[float, float], ...] = (), note: str = "") -> Labels:
    return Labels("primary", True, [Moment(s, e, g, note) for s, e, g in moments],
                  [NeverPick(s, e, "never") for s, e in never])


def clip(start: float, end: float) -> dict:
    return {"start": start, "end": end, "duration": end - start}


def write_transcript(path: Path, video: Path, *, duration: float = 900.0, prefix: str = "w", step: float = 0.5) -> Path:
    words, segments, t, i = [], [], 0.0, 0
    while t < duration:
        words.append({"id": i, "start": round(t, 3), "end": round(t + 0.4, 3), "duration": 0.4, "word": f"{prefix}{i}"})
        t += step
        i += 1
    for k in range(0, len(words), 10):
        chunk = words[k:k + 10]
        segments.append({"id": k // 10, "start": chunk[0]["start"], "end": chunk[-1]["end"],
                         "duration": round(chunk[-1]["end"] - chunk[0]["start"], 3),
                         "text": " ".join(w["word"] for w in chunk)})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 2, "source": {"video_path": str(video), "duration": duration},
                                "transcription": {"model": "gpt-4o-mini-transcribe"}, "segments": segments,
                                "words": words}), encoding="utf-8")
    return path


class Workspace:
    """Three cases (two dev, one holdout) with frozen transcripts and finished labels, under a non-ASCII path."""

    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix=PREFIX)
        self.dir = Path(self._tmp.name)
        self.root = self.dir / "eval root"
        self.videos: dict[str, Path] = {}
        for case_id, split, prefix in (("dev_one", "dev", "a"), ("dev_two", "dev", "b"), ("hold_one", "holdout", "h")):
            video = self.dir / "vods" / f"Kai'nin şaka VOD {case_id}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(os.urandom(2048))                       # not media: duration comes from the transcript
            transcript = write_transcript(self.dir / "transcripts" / f"{video.stem}.json", video, prefix=prefix)
            case_store.prepare(video, root=self.root, case_id=case_id, split=split, transcript=transcript,
                               use_visual_report=False, log=lambda *_: None)
            self.videos[case_id] = video
        self.label("dev_one", {"moments": [{"start": 98, "end": 132, "grade": "great", "note": "chair breaks"},
                                           {"start": 300, "end": 330, "grade": "ok", "note": ""},
                                           {"start": 500, "end": 520, "grade": "bad", "note": "near miss"}],
                               "negatives": [{"start": 700, "end": 760, "note": "Raid Shadow segment"}], "finished": True})
        self.label("dev_two", {"moments": [{"start": 40, "end": 70, "grade": "great", "note": "clutch"},
                                           {"start": 400, "end": 430, "grade": "great", "note": "rage quit"}],
                               "finished": True})
        self.label("hold_one", {"moments": [{"start": 600, "end": 640, "grade": "great",
                                             "note": "the secret holdout punchline about the cat"}], "finished": True})

    def case(self, case_id: str) -> case_store.Case:
        return case_store.list_cases(self.root, [case_id])[0]

    def label(self, case_id: str, payload: dict, annotator: str = "primary") -> tuple:
        return case_store.save_labels(self.case(case_id), payload, annotator=annotator)

    def variant(self, name: str, text: str) -> None:
        folder = variants.variants_dir(self.root)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{name}.env").write_text(text, encoding="utf-8")

    def cleanup(self) -> None:
        self._tmp.cleanup()


FAKE_SET = "MIMIR_CLIP_EVAL_FAKE_SET"             # which picks the fake Stage 2 returns (set by a variant file)


def inprocess_executor(picks_by_set: dict | None = None, *, touch: Path | None = None):
    """Runs the REAL worker functions in this process; Stage 2 itself is the fake."""
    calls: list[tuple[str, str]] = []

    def executor(job, env, cwd, log_path):
        chosen = env.get(FAKE_SET, "")
        calls.append((job["mode"], chosen))
        if job["mode"] == "describe":
            return worker.describe(job)
        picks = (picks_by_set or {}).get(chosen, fakes.DEFAULT_PICKS)

        def analyzer(transcript_path, video_report_path=None):
            from ai.editor import clip_analyzer as ca

            analysis = fakes.analysis_for(Path(transcript_path).stem, picks, version=ca.ANALYZER_VERSION, model="fake")
            ca.save_analysis(transcript_path, analysis, video_report_path)
            if touch is not None:
                touch.write_text("changed by Stage 2", encoding="utf-8")
            return analysis

        meter = pricing.Meter(pricing.load_prices(), budget_usd=job.get("budget_usd"), offline=True)
        return worker.run_clip(job, meter=meter, analyzer=analyzer)

    return executor, calls


def quiet(*_args, **_kwargs) -> None:
    return None


# ============================================================
# TIER 1 METRICS
# ============================================================

class IntervalMetricTests(unittest.TestCase):
    def test_tiou_edge_cases(self) -> None:
        self.assertEqual(metrics.tiou((0, 10), (0, 10)), 1.0)
        self.assertAlmostEqual(metrics.tiou((0, 10), (5, 15)), 5 / 15)
        self.assertEqual(metrics.tiou((0, 10), (10, 20)), 0.0)                 # touching is not overlapping
        self.assertEqual(metrics.tiou((0, 10), (20, 30)), 0.0)
        self.assertAlmostEqual(metrics.tiou((0, 30), (10, 20)), 10 / 30)      # contained
        self.assertEqual(metrics.tiou((5, 5), (0, 10)), 0.0)                  # zero length
        self.assertEqual(metrics.tiou((5, 5), (5, 5)), 0.0)
        self.assertEqual(metrics.tiou((10, 0), (0, 10)), 0.0)                 # inverted
        self.assertEqual(metrics.tiou((0, 10), (2, 8)), metrics.tiou((2, 8), (0, 10)))

    def test_average_precision(self) -> None:
        self.assertEqual(metrics.average_precision([(0, 10), (20, 30)], [(0, 10), (20, 30)], 0.5), 1.0)
        self.assertEqual(metrics.average_precision([(50, 60), (0, 10)], [(0, 10)], 0.5), 0.5)   # hit at rank 2
        self.assertEqual(metrics.average_precision([], [(0, 10)], 0.5), 0.0)                     # empty predictions
        self.assertIsNone(metrics.average_precision([(0, 10)], [], 0.5))                        # no positive
        self.assertEqual(metrics.average_precision([(0, 10)], [(0, 10)], 0.7), 1.0)
        self.assertEqual(metrics.average_precision([(0, 14)], [(0, 10)], 0.75), 0.0)            # tIoU 0.71 < 0.75

    def test_overlapping_labels_match_one_to_one_for_ap_but_not_for_recall(self) -> None:
        truths = [(0, 10), (2, 12)]                                        # two overlapping great moments
        preds = [(1, 11)]                                                  # tIoU 0.82 with both
        self.assertEqual(metrics.average_precision(preds, truths, 0.5), 0.5)   # one prediction = one match
        self.assertEqual(metrics.recall_at(preds, truths, 1), 1.0)            # but it captures both moments
        self.assertEqual(metrics.recall_at([(1, 11), (2, 12)], truths, 3), 1.0)
        self.assertEqual(metrics.average_precision([(1, 11), (2, 12)], truths, 0.5), 1.0)

    def test_hit_recall_and_boundaries_of_one_output(self) -> None:
        gold = labels((98, 132, "great"), (400, 430, "great"), (500, 520, "bad"), never=((700, 760),))
        words = metrics.WordIndex([(99.0, 99.4, "hey"), (129.8, 130.6, "unbelievable")])
        row = metrics.score_repeat([clip(100, 130), clip(500, 520), clip(705, 740)], gold, words, duration=1000)
        self.assertEqual(row["hit1"], {"0.3": 1.0, "0.5": 1.0, "0.7": 1.0})
        self.assertEqual((row["recall1"], row["recall3"]), (0.5, 0.5))
        self.assertAlmostEqual(row["best_tiou"], 30 / 34)
        self.assertEqual((row["start_error_s"], row["end_error_s"]), (2.0, -2.0))
        self.assertEqual(row["covers_great"], 0.0)                         # holds 30 of 34 s (< 90 %)
        self.assertEqual((row["negative_top"], row["negative_any"]), (0.0, 1.0))
        self.assertEqual((row["mid_word_start"], row["mid_word_end"]), (0.0, 1.0))
        self.assertEqual(row["mid_word_words"], [None, "unbelievable"])
        self.assertEqual((row["duration_ok"], row["in_target"], row["schema_ok"]), (1.0, 1.0, True))
        self.assertIn("ends mid-word ('unbelievable')", row["feedback"])
        self.assertIn("1 of 2 great moments are not among the top 3", row["feedback"])
        self.assertIn("starts 2.0 s after the labeled start", row["feedback"])

    def test_empty_predictions_and_labels_without_a_great_moment(self) -> None:
        gold = labels((98, 132, "great"))
        row = metrics.score_repeat([], gold, None, duration=1000, schema=["no clips"])
        self.assertEqual((row["hit1"]["0.5"], row["recall3"], row["map"], row["best_tiou"]), (0.0, 0.0, 0.0, 0.0))
        self.assertFalse(row["schema_ok"])
        self.assertEqual(row["duration_ok"], 0.0)
        self.assertIn("no usable clip", row["feedback"])
        row = metrics.score_repeat([clip(10, 40)], labels((10, 40, "ok")), None, duration=1000)
        self.assertIsNone(row["hit1"])                                      # undefined, not zero
        self.assertIsNone(row["map"])
        self.assertIn("no great moment labeled", row["feedback"])

    def test_never_pick_and_mid_word_tolerances(self) -> None:
        gold = labels((0, 30, "great"), never=((40, 60),))
        self.assertEqual(metrics.score_repeat([clip(15, 40.2)], gold, None, duration=100)["negative_top"], 0.0)
        self.assertEqual(metrics.score_repeat([clip(15, 41)], gold, None, duration=100)["negative_top"], 1.0)
        words = metrics.WordIndex([(10.0, 10.6, "hello"), (11.0, 11.3, "there")])
        self.assertIsNotNone(words.word_at(10.3))
        self.assertIsNone(words.word_at(10.02))                             # within 50 ms of the edge: clean
        self.assertIsNone(words.word_at(10.8))                              # between words

    def test_duration_constraints_come_from_the_run(self) -> None:
        gold = labels((0, 60, "great"))
        row = metrics.score_repeat([clip(0, 60)], gold, None, duration=100)
        self.assertEqual(row["duration_ok"], 0.0)                           # default Stage 2 max is 55 s
        self.assertIn("outside 12-55 s", row["feedback"])
        row = metrics.score_repeat([clip(0, 60)], gold, None, duration=100, constraints={"max_duration": 65.0})
        self.assertEqual(row["duration_ok"], 1.0)
        self.assertEqual(metrics.score_repeat([clip(90, 120)], gold, None, duration=100)["duration_ok"], 0.0)

    def test_ranked_clips_publish_what_the_pipeline_publishes(self) -> None:
        from ai import shorts_pipeline as sp

        analyses = [
            {"clips": [clip(0, 20), {**clip(30, 60), "terra_selected": True}, clip(70, 90)]},
            {"clips": [{**clip(0, 20), "score": 7}, {**clip(30, 62), "score": 9}, {**clip(70, 100), "score": 9,
                                                                                   "covered_moment_ids": [1, 2]}]},
            {"clips": [{**clip(0, 50), "score": 8}, {**clip(60, 92), "score": 8}]},              # duration tie-break
            {"clips": [{**clip(0, 30), "score": 5}, {**clip(40, 70), "score": 5}]},              # position tie-break
        ]
        for analysis in analyses:
            with self.subTest(analysis=analysis):
                ranked = metrics.ranked_clips(analysis)
                self.assertEqual(ranked[0]["position"], sp._select_clip(analysis, None)[0])
                self.assertEqual(sorted(r["position"] for r in ranked), [1, 2, 3][:len(analysis["clips"])])
        self.assertEqual(metrics.ranked_clips({"clips": [{"start": "x"}, clip(5, 9)]})[0]["start"], 5.0)
        self.assertEqual(metrics.schema_problems({"version": 2, "clips": [{"start": "x"}]}),
                         ["clip 1: start/end missing or end <= start"])
        self.assertIn("2 clips are marked terra_selected", metrics.schema_problems(
            {"version": 2, "clips": [{**clip(0, 20), "terra_selected": True}, {**clip(30, 50), "terra_selected": True}]}))

    def test_stability_spread_and_bootstrap(self) -> None:
        stab = metrics.stability([[0, 30], [0, 30], [100, 130]])
        self.assertAlmostEqual(stab["top_change_rate"], 2 / 3)
        self.assertEqual(stab["distinct_tops"], 2)
        self.assertIsNone(metrics.stability([[0, 30]])["top_change_rate"])
        gold = labels((0, 30, "great"))
        rows = {"a": [metrics.score_repeat([clip(0, 30)], gold, None, duration=100),
                      metrics.score_repeat([clip(50, 80)], gold, None, duration=100)],
                "b": [metrics.score_repeat([clip(0, 30)], gold, None, duration=100),
                      metrics.score_repeat([clip(0, 30)], gold, None, duration=100)]}
        cases = [metrics.case_summary(k, "dev", v) for k, v in rows.items()]
        aggregate = metrics.aggregate(cases, rows)
        self.assertEqual(aggregate["metrics"]["hit1"]["mean"], 0.75)
        spread = aggregate["metrics"]["hit1"]["spread"]                    # sample std across repeat means
        self.assertEqual((spread["min"], spread["max"], spread["repeats"]), (0.5, 1.0, 2))
        self.assertAlmostEqual(spread["std"], 0.5 ** 0.5 / 2)
        self.assertIn("changed across repeats", cases[0]["feedback"])
        self.assertEqual(metrics.bootstrap_ci([0.0, 1.0, 1.0, 0.0]), metrics.bootstrap_ci([0.0, 1.0, 1.0, 0.0]))
        self.assertIsNone(metrics.bootstrap_ci([1.0]))
        noisy = metrics.paired_delta({"a": 0.0, "b": 1.0}, {"a": 1.0, "b": 0.0})
        self.assertTrue(noisy["maybe_noise"])
        clear = metrics.paired_delta({k: 0.0 for k in "abcdef"}, {k: 1.0 for k in "abcdef"})
        self.assertEqual((clear["delta"], clear["maybe_noise"]), (1.0, False))

    def test_human_ceiling_and_cohens_kappa(self) -> None:
        gold = labels((0, 30, "great"), (100, 130, "great"))
        other = Labels("second", True, [Moment(5, 30, "great"), Moment(300, 330, "great")])
        agreement = metrics.human_agreement(gold, other)
        self.assertEqual((agreement["matched"], agreement["hit_05"]), (1, 0.5))
        self.assertAlmostEqual(agreement["tiou_mean"], 25 / 30)
        kappa = metrics.cohen_kappa([("A", "A"), ("B", "B"), ("tie", "A"), ("A", "A")])
        self.assertAlmostEqual(kappa["kappa"], 0.5555555, places=5)
        self.assertEqual(metrics.cohen_kappa([("A", "A"), ("A", "A")])["kappa"], 1.0)
        self.assertIsNone(metrics.cohen_kappa([])["kappa"])


# ============================================================
# CASES AND LABELS
# ============================================================

class CaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix=PREFIX)
        self.dir = Path(self._tmp.name)
        self.root = self.dir / "eval"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_names_times_and_powershell_quoting(self) -> None:
        self.assertEqual(case_store.case_slug("Kai'nin şaka İzmir ğ"), "kai_nin_saka_izmir_g")
        self.assertEqual(case_store.case_slug("Çılgın Öğle"), "cilgin_ogle")
        self.assertEqual(case_store.parse_clock("1:02:03.5"), 3723.5)
        self.assertEqual(case_store.parse_clock("2:03,5"), 123.5)
        self.assertEqual(case_store.parse_clock(75), 75.0)
        for bad in ("1:75", "a", "", "1:2:3:4", "-5"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                case_store.parse_clock(bad)
        self.assertEqual(case_store.clock(3723.5), "1:02:03.5")
        self.assertEqual(case_store.clock(59.97), "1:00.0")
        self.assertEqual(case_store.parse_clock(case_store.clock(1234.5)), 1234.5)
        self.assertEqual(case_store.ps_quote(r"C:\VODs\Kai's VOD.mp4"), "\"C:\\VODs\\Kai's VOD.mp4\"")
        self.assertEqual(case_store.ps_quote(r"C:\a$b\Kai's.mp4"), "'C:\\a$b\\Kai''s.mp4'")

    def test_default_input_paths_are_the_pipelines(self) -> None:
        from ai import vod_processor
        from ai.video_brain import config as vb_config
        from ai.video_brain.video_analyzer import _safe_filename

        video = self.dir / "Kai's: şaka?.mp4"
        self.assertEqual(case_store.default_transcript_path(video),
                         (Path(vod_processor.TRANSCRIPT_DIR) / f"{video.stem}.json").resolve())
        self.assertEqual(case_store.default_visual_report_path(video),
                         (Path(vb_config.VIDEO_BRAIN_OUTPUT_DIR) / f"{_safe_filename(video.stem)}_video_report_v6.json"
                          ).resolve())

    def test_prepare_freezes_inputs_and_never_touches_labels(self) -> None:
        video = self.dir / "VODs İ" / "Kai'nin şaka VOD.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"\0" * 4096)
        transcript = write_transcript(self.dir / "t" / "kai.json", video, duration=120.0)
        report = self.dir / "t" / "report.json"
        report.write_text(json.dumps({"report": {"visual_events": []}}), encoding="utf-8")
        messages: list[str] = []
        case = case_store.prepare(video, root=self.root, transcript=transcript, visual_report=report,
                                  log=messages.append)
        self.assertEqual(case.case_id, "kai_nin_saka_vod")
        self.assertEqual((case.split, case.duration), ("dev", 120.0))                  # duration from the transcript
        self.assertEqual(case.transcript_path.read_bytes(), transcript.read_bytes())
        self.assertEqual(case.meta["transcript"]["words"], 240)
        self.assertIsNotNone(case.visual_report_path)
        self.assertEqual(len(case.video_sha256), 64)
        self.assertEqual(case.label_status(), "missing")
        status, problems, _w = case_store.save_labels(case, {"moments": [{"start": "0:10", "end": "0:40"}],
                                                             "finished": True}, annotator="primary")
        self.assertEqual((status, problems), ("ready", []))
        with self.assertRaises(FileExistsError):
            case_store.prepare(video, root=self.root, transcript=transcript, log=messages.append)
        transcript.write_text(transcript.read_text(encoding="utf-8").replace("w1 ", "w1x "), encoding="utf-8")
        case = case_store.prepare(video, root=self.root, transcript=transcript, use_visual_report=False, refresh=True,
                                  log=messages.append)
        self.assertEqual(case.transcript_path.read_bytes(), transcript.read_bytes())    # re-frozen
        self.assertIsNone(case.visual_report_path)
        self.assertEqual(case.label_status(), "ready")                                  # labels kept
        other = self.dir / "other" / "Kai'nin şaka VOD.mp4"
        other.parent.mkdir()
        other.write_bytes(b"\1" * 4096)
        with self.assertRaises(FileExistsError):                                        # same id, other video
            case_store.prepare(other, root=self.root, log=messages.append)
        missing = case_store.prepare(other, root=self.root, case_id="other_vod", log=messages.append)
        self.assertFalse(missing.has_transcript())
        self.assertTrue(any("ai.clip_eval transcribe" in m for m in messages))

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
    def test_prepare_probes_a_real_video_and_makes_a_proxy(self) -> None:
        video = self.dir / "gerçek çıktı ş" / "Kai's VOD.mkv"
        video.parent.mkdir(parents=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=64x64:d=3",
                        "-f", "lavfi", "-i", "sine=frequency=220:duration=3", "-shortest", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", str(video)], check=True)
        case = case_store.prepare(video, root=self.root, proxy=True, log=quiet)
        self.assertAlmostEqual(case.duration, 3.0, delta=0.15)
        self.assertIsNotNone(case.proxy_path)

    def test_label_validation_and_annotators(self) -> None:
        video = self.dir / "v.mp4"
        video.write_bytes(b"x" * 100)
        case = case_store.prepare(video, root=self.root, transcript=write_transcript(self.dir / "v.json", video,
                                                                                    duration=300.0), log=quiet)
        bad = {"moments": [{"start": 50, "end": 40}], "finished": True}
        self.assertEqual(case_store.save_labels(case, bad, annotator="primary")[0], "invalid")
        self.assertEqual(case.label_status(), "missing")                    # invalid labels are never written
        late = {"moments": [{"start": 250, "end": 400}]}
        self.assertIn("after the video", case_store.save_labels(case, late, annotator="primary")[1][0])
        self.assertIn("at least one moment", case_store.save_labels(case, {"finished": True}, annotator="p")[1][0])
        self.assertIn("grade", case_store.save_labels(case, {"moments": [{"start": 1, "end": 20, "grade": "wow"}]},
                                                      annotator="p")[1][0])
        status, _p, warnings = case_store.save_labels(case, {"moments": [{"start": 10, "end": 40, "grade": "ok"}]},
                                                       annotator="Ayşe Kaya")
        self.assertEqual(status, "todo")                                    # not finished: never scored
        self.assertTrue(any("no 'great' moment" in w for w in warnings))
        self.assertTrue(any("guideline is 2-6" in w for w in warnings))
        self.assertEqual(case.gold_annotator, "ayse_kaya")                  # the first person to label owns the gold
        self.assertIsNone(case.gold())
        case_store.save_labels(case, {"moments": [{"start": 12, "end": 40}], "finished": True}, annotator="second")
        case_store.save_labels(case, {"moments": [{"start": 10, "end": 40}, {"start": 100, "end": 130}],
                                      "finished": True}, annotator="Ayşe Kaya")
        case = case_store.list_cases(self.root)[0]
        self.assertEqual(case.annotators(), ["ayse_kaya", "second"])
        self.assertEqual(len(case.gold().moments), 2)
        self.assertEqual([o.annotator for o in case.other_finished_labels()], ["second"])


# ============================================================
# LABELING TOOLS (blind)
# ============================================================

class LabelingToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = Workspace()
        # MIMIR's past pick for dev_one, as a harvested review would store it: must never reach the annotator.
        case_store._write_json(case_store.eval_root(self.ws.root) / reviews.WEAK_DIR / "dev_one.json",
                               {"rows": [{"decision": "accept", "range": {"start": 100, "end": 130,
                                                                          "title": "SECRET MIMIR PICK"}}]})

    def tearDown(self) -> None:
        self.ws.cleanup()

    def test_sheet_round_trip_is_blind(self) -> None:
        path, count = sheet.export_sheet(root=self.ws.root, annotator="second")
        self.assertEqual(count, 3)
        text = path.read_text(encoding="utf-8")
        self.assertNotIn("SECRET MIMIR PICK", text)
        self.assertNotIn("chair breaks", text)                             # the gold annotator's labels stay hidden
        self.assertIn("=== dev_one [ ]", text)
        self.assertIn("# [0:30.0] a60 a61", text)                          # 30-second transcript windows
        self.assertIn("annotator: second", text)
        edited = text.replace("=== dev_two [ ]", "=== dev_two [x]", 1)
        edited = edited.replace("=== dev_two [x]", "=== dev_two [x]\nmoment: 0:41 - 1:10.5 | great | clutch play\n"
                                "never: 7:00 - 7:30 | AFK\nmoment: 6:40 - 7:10 | rage", 1)
        edited += "\n=== not_here [x]\nmoment: 1 - 2 | great\n=== hold_one [x]\nmoment: 10:00 - x | great\n"
        sent = self.ws.dir / "filled ş.txt"
        sent.write_text(edited, encoding="utf-8")
        rows = {r["case_id"]: r["status"] for r in sheet.import_sheet(sent, root=self.ws.root)}
        self.assertTrue(rows["dev_two"].startswith("saved for second (labels ready)"), rows["dev_two"])
        self.assertTrue(rows["not_here"].startswith("unknown case"))
        self.assertTrue(rows["hold_one"].startswith("not saved"))
        self.assertEqual(rows["dev_one"], "skipped: empty")
        saved = self.ws.case("dev_two").labels("second")
        self.assertEqual([(m.start, m.end, m.grade, m.note) for m in saved.moments],
                         [(41.0, 70.5, "great", "clutch play"), (400.0, 430.0, "great", "rage")])
        self.assertEqual([(n.start, n.note) for n in saved.negatives], [(420.0, "AFK")])
        self.assertEqual(self.ws.case("dev_two").labels().annotator, "primary")         # the gold is untouched
        again = {r["case_id"]: r["status"] for r in sheet.import_sheet(sent, root=self.ws.root)}
        self.assertEqual(again["dev_two"], "unchanged")

    def test_the_editor_serves_video_ranges_and_saves_labels_blind(self) -> None:
        from ai.clip_eval import editor

        handler = type("H", (editor._Handler,), {"root": case_store.eval_root(self.ws.root), "annotator": "second"})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            index = urllib.request.urlopen(base + "/").read().decode("utf-8")
            self.assertIn("dev_one", index)
            page = urllib.request.urlopen(base + "/case/dev_one").read().decode("utf-8")
            self.assertIn("a60 a61", page)
            for secret in ("SECRET MIMIR PICK", "chair breaks", "Raid Shadow segment"):
                self.assertNotIn(secret, page)
            video = self.ws.videos["dev_one"].read_bytes()
            request = urllib.request.Request(base + "/video/dev_one", headers={"Range": "bytes=10-19"})
            with urllib.request.urlopen(request) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.read(), video[10:20])
            request = urllib.request.Request(base + "/video/dev_one", headers={"Range": "bytes=-5"})
            self.assertEqual(urllib.request.urlopen(request).read(), video[-5:])
            body = json.dumps({"moments": [{"start": 98.5, "end": 131, "grade": "great", "note": "the chair"}],
                               "negatives": [], "finished": True}).encode("utf-8")
            request = urllib.request.Request(base + "/save/dev_one", data=body, method="POST",
                                             headers={"Content-Type": "application/json"})
            answer = json.loads(urllib.request.urlopen(request).read())
            self.assertEqual(answer["status"], "ready")
            self.assertEqual(self.ws.case("dev_one").labels("second").moments[0].note, "the chair")
            bad = json.dumps({"moments": [{"start": 50, "end": 10}], "finished": True}).encode("utf-8")
            request = urllib.request.Request(base + "/save/dev_one", data=bad, method="POST")
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request)
            self.assertEqual(caught.exception.code, 422)
            with self.assertRaises(urllib.error.HTTPError):
                urllib.request.urlopen(base + "/case/..%2F..%2Fsecret")
        finally:
            server.shutdown()
            server.server_close()


class ReviewHarvestTests(unittest.TestCase):
    def test_decisions_and_published_ranges_become_weak_labels(self) -> None:
        self.assertEqual(reviews.parse_decision("## Decision\n- [x] Accept\n- [ ] Reject. Reason:\n"), ("accept", ""))
        self.assertEqual(reviews.parse_decision("## Decision\n- [ ] Accept\n- [X] Reject. Reason: boring start\n"),
                         ("reject", "boring start"))
        self.assertEqual(reviews.parse_decision("## Decision\n- [x] Accept\n- [x] Reject.\n")[0], "conflict")
        self.assertEqual(reviews.parse_decision("## Decision\n- [ ] Accept\n")[0], "undecided")
        ws = Workspace()
        try:
            vod = ws.dir / "vod_output"
            stem = ws.videos["dev_one"].stem
            (vod / "final").mkdir(parents=True)
            published = vod / "final" / f"{stem}_short.mp4"
            published.write_bytes(b"short")
            (vod / "final" / f"{stem}_short.qc.json").write_text(json.dumps(
                {"status": "passed", "source": str(ws.videos["dev_one"]), "published": str(published)}),
                encoding="utf-8")
            (vod / "final" / f"{stem}_short.review.md").write_text(
                "# Human review\n## Decision\n- [x] Accept\n- [ ] Reject. Reason:\n", encoding="utf-8")
            case_store._write_json(vod / "pipeline" / f"{stem}_pipeline.json", {"selected_clip": {"clip_index": 2}})
            case_store._write_json(vod / "analysis" / f"{stem}_clips.json",
                                   {"clips": [clip(10, 40), {**clip(100, 130), "title": "the chair"}]})
            os.utime(published, (published.stat().st_atime, published.stat().st_mtime + 100))
            (vod / "rejected").mkdir()
            (vod / "rejected" / "x_short_REJECTED.qc.json").write_text(json.dumps(
                {"status": "rejected", "source": str(ws.dir / "unknown.mp4"), "failed": ["captions"]}),
                encoding="utf-8")
            rows = reviews.harvest(root=ws.root, vod_output=vod, log=quiet)
            self.assertEqual([(r["case_id"], r["decision"]) for r in rows],
                             [("dev_one", "accept"), (None, "qc_rejected")])
            self.assertEqual(rows[0]["range"], {"start": 100.0, "end": 130.0, "clip_index": 2, "title": "the chair",
                                                "uncertain": False})
            weak = reviews.check_against(ws.case("dev_one"), [[101, 131], [0, 20]], root=ws.root)
            self.assertEqual((weak["accepted"], weak["accepted_repicked"]), (1, 1))
            self.assertEqual(reviews.count(ws.root), 1)
        finally:
            ws.cleanup()


# ============================================================
# PRICING AND THE BUDGET METER
# ============================================================

class PricingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prices = pricing.load_prices()

    def test_costs_and_estimates(self) -> None:
        cost = pricing.token_cost(self.prices, "gpt-5.6-terra", input_tokens=1_000_000, cached_tokens=400_000,
                                  output_tokens=100_000)
        self.assertAlmostEqual(cost, 0.6 * 2.5 + 0.4 * 0.25 + 0.1 * 15.0)
        self.assertIsNotNone(pricing.token_cost(self.prices, "gpt-5.6-terra-2026-07-30", input_tokens=1))
        self.assertIsNone(pricing.token_cost(self.prices, "mystery-model", input_tokens=1))
        configuration = {"routing": {"scout": {"model": "gpt-5.6-luna", "effort": "medium"},
                                     "judge": {"model": "mystery-model", "effort": "medium"},
                                     "escalation": {"model": "mystery-model", "effort": "high"}}}
        estimate = pricing.estimate_clip_job(self.prices, configuration, {"transcript_chars": 100_000})
        self.assertEqual(estimate["unpriced"], ["mystery-model"])
        self.assertGreater(estimate["expected"], 0)
        priced = {"routing": {**configuration["routing"], "judge": {"model": "gpt-5.6-terra", "effort": "medium"},
                              "escalation": {"model": "gpt-5.6-terra", "effort": "high"}}}
        estimate = pricing.estimate_clip_job(self.prices, priced, {"transcript_chars": 100_000})
        self.assertEqual(estimate["unpriced"], [])
        self.assertGreater(estimate["worst"], estimate["expected"])        # conditional passes count fully in worst
        audio = pricing.estimate_transcribe(self.prices, 3600)
        self.assertAlmostEqual(audio["expected"], 60 * (0.003 + 0.006))

    def test_the_meter_refuses_before_the_call_and_cannot_be_swallowed(self) -> None:
        sent: list[dict] = []

        class Usage:
            input_tokens, output_tokens = 10_000, 2_000
            input_tokens_details = type("D", (), {"cached_tokens": 4_000})()
            output_tokens_details = type("D", (), {"reasoning_tokens": 1_500})()

        def original(resource, **kwargs):
            sent.append(kwargs)
            return type("R", (), {"usage": Usage()})()

        meter = pricing.Meter(self.prices, budget_usd=1.0, offline=False)
        create = meter._wrap_responses(original)
        create(None, model="gpt-5.6-luna", input="hello", reasoning={"effort": "low"})
        self.assertEqual(len(sent), 1)
        self.assertAlmostEqual(meter.spent, (6_000 * 1.0 + 4_000 * 0.1 + 2_000 * 6.0) / 1e6)
        with self.assertRaises(pricing.BudgetExceeded):
            try:                                                            # how Stage 2 wraps optional passes
                create(None, model="gpt-5.6-terra", input="x" * 3_000_000, reasoning={"effort": "high"})
            except Exception:  # noqa: BLE001 - the point: a refusal is not an Exception
                self.fail("BudgetExceeded was caught by `except Exception`")
        self.assertEqual(len(sent), 1)                                      # refused BEFORE the call
        with self.assertRaises(pricing.BudgetExceeded):                     # and the refusal is sticky
            create(None, model="gpt-5.6-luna", input="tiny")
        strict = pricing.Meter(self.prices, budget_usd=10.0, offline=False)
        with self.assertRaises(pricing.BudgetExceeded):
            strict._wrap_responses(original)(None, model="mystery-model", input="x")
        offline = pricing.Meter(self.prices, budget_usd=10.0, offline=True)
        with self.assertRaises(pricing.OfflineCallBlocked):
            offline._wrap_transcription(original)(None, model="whisper-1")
        summary = meter.summary()
        self.assertEqual((summary["calls"], summary["cached_tokens"], summary["reasoning_tokens"]), (1, 4_000, 1_500))

    def test_streamed_chat_usage_is_booked_when_the_stream_ends(self) -> None:
        chunks = [type("C", (), {"usage": None})(), type("C", (), {"usage": type("U", (), {
            "prompt_tokens": 1_000_000, "completion_tokens": 0})()})()]
        meter = pricing.Meter(self.prices, budget_usd=None, strict_unpriced=False, offline=False)
        stream = meter._wrap_chat(lambda resource, **kwargs: iter(chunks))(None, model="gpt-5.6-luna", messages=[],
                                                                           stream=True)
        self.assertEqual(meter.calls, [])
        self.assertEqual(len(list(stream)), 2)
        self.assertAlmostEqual(meter.spent, 1.0)

    def test_the_installed_patch_reaches_the_sdk_class(self) -> None:
        from openai.resources.responses import Responses

        original = Responses.create
        meter = pricing.Meter(self.prices, budget_usd=1.0, offline=True).install()
        try:
            self.assertIsNot(Responses.create, original)
            with self.assertRaises(pricing.OfflineCallBlocked):
                Responses.create(object(), model="gpt-5.6-luna", input="x")
        finally:
            meter.uninstall()
        self.assertIs(Responses.create, original)


# ============================================================
# RUNNER (Stage 2 in isolation, mocked) / COMPARE / GATE / GUARD
# ============================================================

class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = Workspace()
        self.root = self.ws.root

    def tearDown(self) -> None:
        self.ws.cleanup()

    def run_variant(self, executor, **kwargs):
        kwargs.setdefault("max_usd", 50.0)
        return runner.run_variant(kwargs.pop("variant", "baseline"), root=self.root, executor=executor, log=quiet,
                                  **kwargs)

    def test_no_paid_job_without_a_cap_and_dry_run_only_estimates(self) -> None:
        executor, calls = inprocess_executor()
        lines: list[str] = []
        plan, records, code = runner.run_variant("baseline", root=self.root, executor=executor, dry_run=True,
                                                 log=lines.append)
        self.assertEqual((code, records), (0, []))
        self.assertEqual([c[0] for c in calls], ["describe"])                 # describing Stage 2 is free
        self.assertIn("expected $", "\n".join(lines))
        self.assertIn("UNVERIFIED", "\n".join(lines))
        self.assertEqual(len(plan.to_run), 2)                                 # dev only by default
        self.assertFalse(plan.directory.exists())
        _plan, _records, code = runner.run_variant("baseline", root=self.root, executor=executor, log=lines.append)
        self.assertEqual(code, 2)
        self.assertIn("No model was called", lines[-1])
        _plan, _records, code = runner.run_variant("baseline", root=self.root, executor=executor, max_usd=0.0001,
                                                   log=lines.append)
        self.assertEqual(code, 2)
        self.assertIn("above --max-usd", lines[-1])
        self.assertEqual([c[0] for c in calls], ["describe"] * 3)
        self.assertFalse(plan.directory.exists())

    def test_runs_are_isolated_cached_scored_and_compared_with_the_gate(self) -> None:
        executor, calls = inprocess_executor()
        plan, records, code = self.run_variant(executor, split="all", repeats=2)
        self.assertEqual(code, 0, records)
        self.assertEqual(len(records), 6)
        self.assertTrue(plan.name.startswith("baseline__"))
        self.assertTrue(all(r["status"] == "ok" and r["production_cache_untouched"] for r in records))
        from ai.editor import clip_analyzer as ca

        self.assertEqual(Path(ca.ANALYSIS_DIR), case_store.PROJECT_ROOT / "vod_output" / "analysis")   # restored
        for case_id in ("dev_one", "dev_two", "hold_one"):
            self.assertFalse((Path(ca.ANALYSIS_DIR) / f"{case_id}_clips.json").exists())  # never the production cache
        analysis = json.loads((plan.directory / "hyp" / "dev_one" / "r1.analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(analysis["inputs"]["clip_selection_owner"], "Terra")            # what save_analysis wrote
        info = json.loads((plan.directory / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(info["configuration"]["routing"]["scout"]["model"], ca.CLIP_SCOUT_MODEL)
        self.assertEqual({m["module"] for m in info["configuration"]["code"]},
                         {"ai.editor.clip_analyzer", "ai.model_config", "ai.openai_client"})
        from ai import shorts_pipeline as sp

        self.assertIn(sp._module_fingerprint(ca), info["configuration"]["code"])        # the pipeline's own record

        # Cached: nothing runs again; more repeats run only the new ones.
        _plan, records, _code = self.run_variant(executor, split="all", repeats=2)
        self.assertEqual(records, [])
        _plan, records, _code = self.run_variant(executor, split="dev", repeats=3)
        self.assertEqual([(r["case_id"], r["repeat"]) for r in records], [("dev_one", 3), ("dev_two", 3)])

        result = runner.score_run(plan.name, root=self.root)
        dev, hold = result["splits"]["dev"], result["splits"]["holdout"]
        self.assertEqual((dev["aggregate"]["videos"], hold["aggregate"]["videos"]), (2, 1))
        self.assertEqual(dev["aggregate"]["metrics"]["hit1"]["mean"], 1.0)
        self.assertEqual(dev["aggregate"]["metrics"]["recall3"]["mean"], 0.75)            # dev_two: 1 of 2 great
        self.assertEqual(dev["aggregate"]["metrics"]["negative_any"]["mean"], 0.0)
        self.assertEqual(hold["cases"][0]["case_id"], "hold_one")
        text = (plan.directory / "report.md").read_text(encoding="utf-8")
        self.assertIn(report.HOLDOUT_BANNER, text)
        self.assertIn("## dev (2 videos)", text)
        self.assertIn("## holdout (1 video)", text)
        self.assertIn("one video moves a rate by 0.50", text)

        # A candidate that loses the holdout video: the gate fails, the report shows why.
        self.ws.variant("judge-high", f"# Terra at high effort\nMIMIR_CLIP_JUDGE_REASONING=high\n{FAKE_SET}=worse\n")
        worse = {**fakes.DEFAULT_PICKS, "hold_one": [[10.0, 40.0]], "dev_two": [[200.0, 228.0], [40.0, 70.0]]}
        executor_b, _calls = inprocess_executor({"worse": worse})
        plan_b, records_b, code_b = self.run_variant(executor_b, variant="judge-high", split="all", repeats=2)
        self.assertEqual(code_b, 0)
        self.assertNotEqual(plan_b.name, plan.name)
        text, decision = runner.compare(plan.name, plan_b.name, root=self.root, gate=True)
        self.assertFalse(decision["passed"])
        self.assertEqual([c["metric"] for c in decision["checks"] if c["failed"]], ["hit1", "recall3"])
        self.assertIn("variant MIMIR_CLIP_JUDGE_REASONING: `(unset)` -> `high`", text)
        self.assertIn("### Most regressed", text)
        self.assertIn("**dev_two**: HIT@1 -1.00", text)
        self.assertIn("**hold_one**", text)
        self.assertIn("**FAIL**", text)
        self.assertIn(report.HOLDOUT_BANNER, text)
        text, decision = runner.compare(plan.name, plan.name, root=self.root, gate=True)
        self.assertTrue(decision["passed"])
        with self.assertRaises(ValueError):                                   # a name holds one configuration
            self.run_variant(executor_b, variant="judge-high", name=plan.name)
        self.assertEqual(runner.resolve_run("judge-high", self.root), plan_b.name)

        # Phase 2 groundwork: the AI-readable export holds dev videos only, and the guard checks it.
        path = guard.write_feedback(plan_b.name, root=self.root)
        exported = path.read_text(encoding="utf-8")
        self.assertIn("dev_two", exported)
        self.assertNotIn("hold_one", exported)
        self.assertNotIn("secret holdout punchline", exported)
        payload = json.loads(exported)
        self.assertTrue(all(c["feedback"] for c in payload["cases"]))
        self.assertEqual(payload["split"], "dev")

    def test_stage2_may_not_touch_the_production_cache(self) -> None:
        sentinel = self.ws.dir / "production_cache.json"
        sentinel.write_text("{}", encoding="utf-8")
        executor, _calls = inprocess_executor(touch=sentinel)

        def guarded(job, env, cwd, log_path):
            if job["mode"] == "clip":
                job = {**job, "guard_paths": [*job["guard_paths"], str(sentinel)]}
            return executor(job, env, cwd, log_path)

        _plan, records, code = self.run_variant(guarded, only=["dev_one"])
        self.assertEqual((code, records[0]["status"]), (1, "error"))
        self.assertIn("production Stage 2 cache changed", records[0]["error"])
        self.assertFalse(records[0]["production_cache_untouched"])

    def test_cases_without_labels_or_transcript_are_not_run(self) -> None:
        video = self.ws.dir / "vods" / "new one.mp4"
        video.write_bytes(b"new")
        case_store.prepare(video, root=self.root, case_id="new_one", log=quiet)
        self.ws.label("dev_two", {"moments": [{"start": 40, "end": 70}], "finished": False})
        executor, _calls = inprocess_executor()
        plan = runner.plan_run("baseline", root=self.root, executor=executor)
        reasons = {row["case_id"]: row["reason"] for row in plan.skipped}
        self.assertIn("labels todo", reasons["dev_two"])
        self.assertIn("labels missing", reasons["new_one"])
        plan = runner.plan_run("baseline", root=self.root, executor=executor, include_unlabeled=True)
        reasons = {row["case_id"]: row["reason"] for row in plan.skipped}
        self.assertIn("python -m ai.clip_eval transcribe", reasons["new_one"])
        self.assertEqual([j.case.case_id for j in plan.jobs], ["dev_one", "dev_two"])

    def test_a_moved_video_is_pointed_at_for_stage2(self) -> None:
        moved = self.ws.videos["dev_one"].with_name("moved ğ.mp4")
        self.ws.videos["dev_one"].rename(moved)
        case = self.ws.case("dev_one")
        case.meta["source"]["video"] = str(moved)
        case_store._write_json(case.directory / "meta.json", case.meta)
        transcript, notes = runner._job_transcript(self.ws.case("dev_one"), self.ws.dir / "job")
        self.assertEqual(json.loads(transcript.read_text(encoding="utf-8"))["source"]["video_path"], str(moved))
        self.assertIn("rewritten", notes[0])

    def test_variant_files_are_validated(self) -> None:
        self.assertEqual(variants.parse_env_text("# c\nexport MIMIR_CLIP_SCOUT_MODEL='gpt-5.6-terra'  # x\n\n"),
                         {"MIMIR_CLIP_SCOUT_MODEL": "gpt-5.6-terra"})
        for bad in ("OPENAI_API_KEY=sk-1", "NOT A LINE", "lower=x"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                variants.parse_env_text(bad)
        self.ws.variant("captions", "MIMIR_CAPTION_PRIMARY_PROVIDER=qwen3_asr\n")
        self.assertEqual(variants.load("captions", self.root).ignored_keys("clip"), ["MIMIR_CAPTION_PRIMARY_PROVIDER"])
        self.assertEqual(variants.load("captions", self.root).ignored_keys("full"), [])
        with self.assertRaises(FileNotFoundError):
            variants.load("nope", self.root)
        with self.assertRaises(ValueError):
            variants.load("../escape", self.root)

    def test_full_stage_runs_a_byte_identical_copy_of_the_code(self) -> None:
        workspace = runner.make_workspace(self.ws.dir / "work space ş")
        self.assertEqual(worker.pipeline_code_fingerprint(workspace),
                         worker.pipeline_code_fingerprint(case_store.PROJECT_ROOT))
        self.assertTrue((workspace / "main.py").is_file())
        self.assertFalse((workspace / ".env").exists())
        self.assertFalse((workspace / "vod_output").exists())
        executor, _calls = inprocess_executor()
        plan = runner.plan_run("baseline", stage="full", only=["dev_one"], root=self.root, executor=executor)
        estimate = plan.jobs[0].estimate
        self.assertGreater(estimate["unmetered_usd"], 0)                  # Stage 1 + Gemini are reserved up front
        self.assertGreater(estimate["expected"], pricing.load_prices()["estimate"]["full_stage"]["after_selection_usd"])
        self.assertIn("pipeline", runner.format_plan(plan))


class GuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = Workspace()

    def tearDown(self) -> None:
        self.ws.cleanup()

    def test_holdout_identifiers_notes_and_transcript_text_are_refused(self) -> None:
        guard.assert_ai_safe({"case": "dev_one", "note": "chair breaks", "text": "a10 a11 a12 a13 a14 a15 a16 a17"},
                             self.ws.root)
        guard.assert_ai_safe("hold_one_b is another video", self.ws.root)     # whole-token match only
        leaks = ("hold_one", self.ws.case("hold_one").video_sha256,
                 "the secret holdout punchline about the cat",
                 "... h40 h41 h42 h43 h44 h45 h46 h47 ...",                    # a stretch of the holdout transcript
                 self.ws.videos["hold_one"].name)
        for leak in leaks:
            with self.subTest(leak=leak[:30]), self.assertRaises(guard.HoldoutLeak):
                guard.assert_ai_safe({"x": leak}, self.ws.root)
        with self.assertRaises(guard.HoldoutLeak):
            guard.require_dev(case_store.list_cases(self.ws.root))
        self.assertEqual([c.case_id for c in guard.optimization_cases(self.ws.root)], ["dev_one", "dev_two"])

    def test_ai_view_drops_holdout_even_if_a_dev_section_was_tampered(self) -> None:
        score = {"splits": {"dev": {"cases": [{"case_id": "hold_one", "split": "holdout"}]}}}
        with self.assertRaises(guard.HoldoutLeak):
            guard.ai_view(score)


# ============================================================
# TIER 2: JUDGE AND CALIBRATION
# ============================================================

def scored(tops: dict[str, list[list[float]]], run: str = "r") -> dict:
    cases = [{"case_id": k, "repeat_rows": [{"top": t} for t in v]} for k, v in tops.items()]
    return {"run": run, "splits": {"dev": {"cases": cases}, "holdout": {"cases": []}}}


class FakeJudge:
    """prefer(first_view, second_view) -> 'first' | 'second' | 'tie'."""

    def __init__(self, prefer) -> None:
        self.prefer = prefer
        self.calls = 0

    def __call__(self, model, system, prompt, schema):
        self.calls += 1
        self.assertions(system, schema)
        first, second = prompt.split("=== second clip")
        verdict = self.prefer(first, second)
        crit = {"hook_3s": "yes", "standalone": "yes", "payoff_inside": "no", "clean_cut": "yes"}
        return {"first": crit, "second": crit, "better": verdict, "reason": f"{verdict} is better"}, \
            {"input_tokens": 2000, "output_tokens": 300}

    @staticmethod
    def assertions(system: str, schema: dict) -> None:
        assert "hook_3s" in system and "good:" in system and "bad:" in system and "1-10" not in system
        assert schema["properties"]["better"]["enum"] == ["first", "second", "tie"]


class JudgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = Workspace()
        self.root = self.ws.root
        self.env = mock.patch.dict(os.environ, {judge.MODEL_ENV: "gemini-3.7-flash"})
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        self.ws.cleanup()

    def test_both_orders_position_bias_cache_and_budget(self) -> None:
        a = scored({"dev_one": [[100, 130]], "dev_two": [[40, 70]], "hold_one": [[600, 640]]}, "A")
        b = scored({"dev_one": [[300, 330]], "dev_two": [[40, 70]], "hold_one": [[10, 40]]}, "B")
        honest = FakeJudge(lambda first, second: "first" if "a200" in first else "second")   # likes clip A
        result = judge.compare_runs(a, b, root=self.root, max_usd=1.0, client=honest)
        rows = {r["case_id"]: r for r in result["rows"]}
        self.assertEqual(set(rows), {"dev_one", "dev_two"})                  # the holdout video is never judged
        self.assertEqual((rows["dev_one"]["verdict"], rows["dev_one"]["ab"], rows["dev_one"]["ba"]), ("A", "A", "A"))
        self.assertTrue(rows["dev_two"]["identical"])                        # same clip: tie, no call
        self.assertEqual((honest.calls, result["losses"], result["wins"], result["identical"]), (2, 1, 0, 1))
        self.assertFalse(result["trusted"])                                  # never calibrated
        self.assertGreater(result["spent_usd"], 0)
        again = judge.compare_runs(a, b, root=self.root, max_usd=1.0, client=honest)
        self.assertEqual((honest.calls, again["spent_usd"]), (2, 0.0))       # cached
        biased = FakeJudge(lambda first, second: "first")                   # always the first one shown
        result = judge.compare_runs(a, scored({"dev_one": [[500, 520]]}, "C"), root=self.root, max_usd=1.0,
                                    client=biased)
        self.assertEqual((result["rows"][0]["verdict"], result["order_disagreements"]), ("tie", 1))
        broke = judge.compare_runs(a, scored({"dev_one": [[700, 730]]}, "D"), root=self.root, max_usd=0.000001,
                                   client=biased)
        self.assertIn("budget", broke["refused"])
        self.assertIsNone(broke["rows"][0]["verdict"])
        dry = judge.compare_runs(a, scored({"dev_one": [[700, 730]]}, "D"), root=self.root, dry_run=True,
                                 client=biased)
        self.assertEqual((dry["planned_calls"], biased.calls), (2, 2))        # estimated, not called
        nocap = judge.compare_runs(a, scored({"dev_one": [[700, 730]]}, "D"), root=self.root, client=biased)
        self.assertIn("--max-usd", nocap["refused"])
        with self.assertRaises(guard.HoldoutLeak):
            judge.judge_pair(judge.Session(root=self.root, max_usd=1, dry_run=False, client=honest),
                             self.ws.case("hold_one"), [600, 640], [10, 40])
        with mock.patch.dict(os.environ, {judge.MODEL_ENV: "gpt-5.6-terra"}), self.assertRaises(ValueError):
            judge.judge_model()

    def test_calibration_is_blind_and_trust_needs_kappa(self) -> None:
        tops_a = {"dev_one": [[100 + 40 * i, 130 + 40 * i] for i in range(6)],
                  "dev_two": [[10 + 40 * i, 40 + 40 * i] for i in range(6)]}
        tops_b = {"dev_one": [[400 + 40 * i, 430 + 40 * i] for i in range(6)],
                  "dev_two": [[410 + 40 * i, 440 + 40 * i] for i in range(5)] + [[210, 240]],
                  "hold_one": [[1, 30]]}
        added = calibrate.add_pairs(scored(tops_a, "A"), scored(tops_b, "B"), root=self.root, seed=3)
        self.assertEqual(added, 11)                                          # dev_two r6 is the same clip
        self.assertEqual(calibrate.add_pairs(scored(tops_a, "A"), scored(tops_b, "B"), root=self.root), 0)
        pairs = calibrate.load_pairs(self.root)
        self.assertEqual({p["first"] for p in pairs}, {"A", "B"})            # sides drawn at random
        for pair in pairs:                                                  # the human always prefers clip A
            one, _two = calibrate.shown(pair)
            calibrate.save_human(pair["id"], "1" if one == pair["clips"]["A"] else "2", root=self.root)
        self.assertTrue(all(calibrate.human_ab(p, calibrate.human_labels(self.root)[p["id"]]) == "A" for p in pairs))

        def likes_a(first: str, second: str) -> str:              # A's clips start early in the video (low word ids)
            ids = lambda text: [int(w[1:]) for w in text.split() if w[:1] in "ab" and w[1:].isdigit()]
            return "first" if min(ids(first) or [10 ** 9]) < min(ids(second) or [10 ** 9]) else "second"

        result = calibrate.run_judge(root=self.root, max_usd=1.0, client=FakeJudge(likes_a))
        self.assertEqual((result["n"], result["kappa"], result["trusted"]), (11, 1.0, True))
        self.assertTrue(calibrate.trust(judge.judge_model(), root=self.root)["trusted"])
        contrary = calibrate.compute(judge.judge_model(), {p["id"]: "B" for p in pairs}, root=self.root)
        self.assertFalse(contrary["trusted"])
        self.assertFalse(calibrate.trust(judge.judge_model(), root=self.root)["trusted"])

    def test_the_terminal_session_hides_the_sides(self) -> None:
        calibrate.add_pairs(scored({"dev_one": [[100, 130]]}, "A"), scored({"dev_one": [[300, 330]]}, "B"),
                            root=self.root, seed=1)
        shown_text: list[str] = []
        labeled = calibrate.text_session(root=self.root, ask=lambda prompt: "t", say=shown_text.append)
        self.assertEqual(labeled, 1)
        output = "\n".join(shown_text)
        self.assertIn("Clip 1", output)
        self.assertNotIn("A", [w.strip(":") for w in output.split()])          # no run side, no run name
        self.assertEqual(list(calibrate.human_labels(self.root).values()), ["tie"])


# ============================================================
# THE REAL WORKER SUBPROCESS (Stage 2 is the fake; every OpenAI call is refused)
# ============================================================

class WorkerProcessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = Workspace()
        self.env = mock.patch.dict(os.environ, {worker.TEST_ANALYZER_ENV: "clip_eval_fakes:fake_create_clip_analysis",
                                                pricing.OFFLINE_ENV: "1", "OPENAI_API_KEY": "sk-test-not-a-key",
                                                "PYTHONPATH": str(case_store.PROJECT_ROOT / "tests")})
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        self.ws.cleanup()

    def test_variants_reach_stage2_in_their_own_process_and_paid_calls_are_refused(self) -> None:
        self.ws.variant("scout-terra", "MIMIR_CLIP_SCOUT_MODEL=gpt-5.6-terra\n")
        plan, records, code = runner.run_variant("scout-terra", only=["dev_one"], max_usd=5.0, root=self.ws.root,
                                                 log=quiet)
        self.assertEqual(code, 0, records)
        self.assertEqual(plan.configuration["routing"]["scout"]["model"], "gpt-5.6-terra")
        analysis = json.loads((plan.directory / "hyp" / "dev_one" / "r1.analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(analysis["seen"]["scout"], "gpt-5.6-terra")         # the variant reached Stage 2
        self.assertEqual(records[0]["clips"][0]["start"], 100.0)
        self.assertTrue(records[0]["production_cache_untouched"])
        self.assertFalse((case_store.PROJECT_ROOT / "vod_output" / "analysis" / "dev_one_clips.json").exists())
        with mock.patch.dict(os.environ, {"MIMIR_CLIP_EVAL_FAKE_MODE": "paid"}):
            _plan, records, code = runner.run_variant("baseline", only=["dev_one"], max_usd=5.0, root=self.ws.root,
                                                      log=quiet)
        self.assertEqual((code, records[0]["status"]), (1, "budget_stopped"))
        self.assertIn(pricing.OFFLINE_ENV, records[0]["error"])
        self.assertNotIn("clips", records[0])                                # a refused run is never scored


class IsolationTests(unittest.TestCase):
    def test_the_pipeline_never_imports_the_clip_eval(self) -> None:
        import ast

        offenders = []
        for path in sorted((case_store.PROJECT_ROOT / "ai").rglob("*.py")):
            if "clip_eval" in path.relative_to(case_store.PROJECT_ROOT).parts:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                         else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                if any("clip_eval" in name for name in names):
                    offenders.append(str(path))
        for path in (case_store.PROJECT_ROOT / "main.py", case_store.PROJECT_ROOT / "verify_unified_clean.py"):
            self.assertNotIn("ai.clip_eval", path.read_text(encoding="utf-8"))
        self.assertEqual(offenders, [])


class CommandLineTests(unittest.TestCase):
    def test_status_and_a_run_without_a_cap(self) -> None:
        ws = Workspace()
        try:
            with mock.patch("builtins.print") as printed:
                self.assertEqual(cli.main(["--root", str(ws.root), "status"]), 0)
            output = "\n".join(str(c.args[0]) for c in printed.call_args_list if c.args)
            self.assertIn("3/3 cases have finished labels", output)
            self.assertIn("1 holdout", output)
            self.assertIn("reserve at least 3 videos for holdout", output)
            executor, calls = inprocess_executor()
            with mock.patch.object(runner, "spawn", executor), mock.patch("builtins.print"):
                self.assertEqual(cli.main(["--root", str(ws.root), "run", "--variant", "baseline"]), 2)
                self.assertEqual(cli.main(["--root", str(ws.root), "run", "--dry-run", "--repeats", "3"]), 0)
            self.assertEqual([c[0] for c in calls], ["describe", "describe"])
            with mock.patch("builtins.print"):
                self.assertEqual(cli.main(["--root", str(ws.root), "split", "holdout", "dev_two"]), 0)
            self.assertEqual(ws.case("dev_two").split, "holdout")
        finally:
            ws.cleanup()


if __name__ == "__main__":
    unittest.main()
