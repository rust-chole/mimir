"""Measured caption clock: health, overlapping-chunk recovery, splicing, no synthetic timing."""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai import vod_processor
from ai.editor import caption_clock as cc
from ai.editor import caption_judge
from ai.editor import speaker_caption_support as scs

HAVE_FFMPEG = bool(shutil.which("ffmpeg"))


def rows(spec):
    return [{"word": w, "start": a, "end": b} for w, a, b in spec]


def evenly(n, start=0.2, step=0.4, dur=0.3, word="w"):
    return [{"word": f"{word}{i}", "start": round(start + i * step, 3), "end": round(start + i * step + dur, 3)}
            for i in range(n)]


def true_clock(duration=30.0):
    return evenly(int((duration - 0.4) / 0.4), word="t")


class ChunkFakes:
    """cutter encodes the chunk start in the path; timer answers from a TRUE clock (chunk-relative)."""

    def __init__(self, truth, fail=False):
        self.truth, self.fail, self.calls = truth, fail, []

    def cutter(self, audio, *, start, end, label, enhanced):
        return Path(f"/nonexistent/{start:.3f}_{end:.3f}.wav")

    def timer(self, path):
        self.calls.append(path.name)
        if self.fail:
            raise TimeoutError("timing ear down")
        start, end = (float(x) for x in path.stem.split("_"))
        return {"words": [{"word": r["word"], "start": round(r["start"] - start, 3), "end": round(r["end"] - start, 3)}
                          for r in self.truth if start <= r["start"] and r["end"] <= end]}


class HealthTests(unittest.TestCase):
    def test_healthy_clock_with_normal_imperfections(self) -> None:
        clock = evenly(40)
        clock[5]["end"] = clock[5]["start"]                          # a few zero-length words
        clock[9]["start"] = clock[8]["start"] - 0.1                   # small jitter
        self.assertTrue(cc.assess(clock, duration=17.0, expected_words=42).healthy)

    def test_repeated_words_are_not_a_collapse(self) -> None:
        clock = rows([("no", 1.0, 1.2), ("no", 1.3, 1.5), ("no", 1.6, 1.8), ("no", 1.9, 2.1), ("way", 2.2, 2.6)])
        self.assertTrue(cc.assess(clock, duration=4.0, expected_words=5).healthy)

    def test_legitimate_simultaneous_speech(self) -> None:
        clock = rows([("wait", 1.0, 1.4), ("no", 1.0, 1.3), ("stop", 1.2, 1.6), ("okay", 2.0, 2.4)])
        health = cc.assess(clock, duration=4.0, expected_words=4)
        self.assertTrue(health.healthy, health.to_dict())
        self.assertEqual(health.metrics["largest_same_instant"], 2)

    def test_missing_timing_word_is_healthy(self) -> None:
        self.assertTrue(cc.assess(evenly(19), duration=9.0, expected_words=20).healthy)

    def test_empty_and_corrupt_clocks_are_fatal(self) -> None:
        self.assertEqual(cc.assess([], duration=5.0).fatal, ["no_words"])
        collapsed = [{"word": f"w{i}", "start": 3.0, "end": 3.0} for i in range(20)]
        health = cc.assess(collapsed, duration=10.0, expected_words=20)
        self.assertIn("collapsed_to_one_instant", health.fatal)
        self.assertIn("mostly_nonpositive_durations", health.fatal)
        negative = [{"word": f"w{i}", "start": 1 + i, "end": 0.5 + i} for i in range(6)]
        self.assertIn("mostly_nonpositive_durations", cc.assess(negative, duration=9.0).fatal)

    def test_one_ordinary_soft_symptom_is_not_enough_to_distrust_the_clock(self) -> None:
        clock = evenly(10)
        clock[-1]["end"] = 9.0                                                             # one word past the end
        health = cc.assess(clock, duration=4.2, expected_words=10)
        self.assertEqual(health.soft, ["words_outside_audio"])
        self.assertTrue(health.healthy and health.usable_for_publish)
        clock[3]["start"], clock[7]["start"] = 0.0, 0.0                                    # + regressions
        self.assertFalse(cc.assess(clock, duration=4.2, expected_words=10).healthy)

    def test_low_coverage_alone_asks_for_one_recovery_attempt_but_never_blocks(self) -> None:
        health = cc.assess(evenly(10), duration=5.0, expected_words=25)                    # 40 % of the words
        self.assertEqual((health.fatal, health.soft), ([], ["low_coverage"]))
        self.assertTrue(health.needs_recovery_attempt)
        self.assertTrue(health.usable_for_publish)
        nearly = cc.assess(evenly(99), duration=45.0, expected_words=100)                 # 0.99 is not "low"
        self.assertFalse(nearly.needs_recovery_attempt)


class ChunkMergeTests(unittest.TestCase):
    def test_windows_overlap_and_ownership_meets_in_the_middle(self) -> None:
        windows = cc.chunk_windows(25.0, chunk=10.0, overlap=1.5)
        self.assertEqual(windows, [(0.0, 10.0), (8.5, 18.5), (17.0, 25.0)])
        self.assertEqual(cc.ownership(windows), [(0.0, 9.25), (9.25, 17.75), (17.75, 25.0)])

    def test_overlapping_chunk_boundaries_dedupe_and_report_seams(self) -> None:
        a = rows([("one", 1.0, 1.3), ("two", 8.8, 9.1), ("three", 9.4, 9.8)])
        b = rows([("two", 8.82, 9.1), ("three", 9.41, 9.8), ("four", 12.0, 12.3)])
        merged, report = cc.merge_chunks([((0.0, 10.0), a), ((8.5, 18.5), b)])
        self.assertEqual([r["word"] for r in merged], ["one", "two", "three", "four"])
        self.assertAlmostEqual(merged[1]["start"], 8.8)            # the owner's measurement, never averaged
        self.assertAlmostEqual(merged[2]["start"], 9.41)
        self.assertEqual(report["seams"][0]["matched_anchors"], 2)
        self.assertFalse(report["seams"][0]["disagrees"])


class RecoveryTests(unittest.TestCase):
    def recover(self, primary, fakes, duration=30.0, expected=None, reference=None):
        return cc.recover_clock({"words": primary}, audio_path=Path("final.wav"), duration=duration,
                                expected_words=len(true_clock(duration)) if expected is None else expected,
                                timer=fakes.timer, cutter=fakes.cutter, reference=reference)

    def test_healthy_primary_never_triggers_recovery(self) -> None:
        fakes = ChunkFakes(true_clock())
        data, report = self.recover(true_clock(), fakes)
        self.assertEqual(fakes.calls, [])
        self.assertEqual((report["status"], report["recovery_attempted"]), ("healthy", False))
        self.assertEqual(data["words"], true_clock())

    def test_recovery_splices_only_the_damaged_region(self) -> None:
        truth = true_clock()
        primary = [dict(r) for r in truth]
        damaged = [r for r in primary if 18.0 <= r["start"] < 26.0]
        for r in damaged:                                             # transcript-sized collapse in one region
            r["start"] = r["end"] = 20.0
        data, report = self.recover(primary, ChunkFakes(truth))
        self.assertEqual(report["status"], "recovered")
        self.assertEqual(report["source"], "whisper_chunk_recovery")
        regions = report["attempts"][0]["replaced_regions"]
        self.assertTrue(regions and all(r["reason"] == "collapsed" for r in regions))
        kept = [r for r in data["words"] if r["start"] < 17.0]
        self.assertEqual([(r["start"], r["end"]) for r in kept],
                         [(r["start"], r["end"]) for r in primary if r["start"] < 17.0])   # untouched anchors
        self.assertTrue(cc.assess(data["words"], duration=30.0).healthy)

    def test_empty_primary_is_recovered_from_measured_chunks(self) -> None:
        data, report = self.recover([], ChunkFakes(true_clock()))
        self.assertEqual(report["selected"], "whisper_chunk_recovery")
        self.assertEqual(len(data["words"]), len(true_clock()))

    def test_no_measured_clock_blocks_with_no_synthetic_fallback(self) -> None:
        with self.assertRaises(cc.ClockUnavailable):
            self.recover([], ChunkFakes(true_clock(), fail=True))
        zero = [{"word": f"w{i}", "start": 1.0 + i * 0.3, "end": 1.0 + i * 0.3} for i in range(30)]
        with self.assertRaises(cc.ClockUnavailable):                   # zero durations after all recovery
            self.recover(zero, ChunkFakes(zero))
        self.assertFalse([name for name in dir(cc) if re.search("interpol|synthetic|distribute", name, re.I)])

    def test_independent_reference_clock_only_when_structurally_healthier(self) -> None:
        truth = true_clock()
        collapsed = [{**r, "start": 5.0, "end": 5.0} for r in truth]
        data, report = self.recover(collapsed, ChunkFakes(truth, fail=True), reference=lambda: truth)
        self.assertEqual(report["selected"], "vod_reference_clock")
        self.assertTrue(report["attempts"][-1]["healthier_than_primary"])
        with self.assertRaises(cc.ClockUnavailable):
            self.recover(collapsed, ChunkFakes(truth, fail=True), reference=lambda: collapsed)

    def low_coverage_primary(self):
        truth = true_clock()
        return truth, [r for r in truth if r["start"] < 11.0]                              # ~36 % coverage, no fatal

    def test_low_coverage_attempts_recovery_and_a_healthier_recovery_wins(self) -> None:
        truth, primary = self.low_coverage_primary()
        fakes = ChunkFakes(truth)
        data, report = self.recover(primary, fakes)
        self.assertTrue(fakes.calls)                                                        # the attempt was made
        self.assertEqual((report["status"], report["selected"]), ("recovered", "whisper_chunk_recovery"))
        self.assertEqual(report["primary_health"]["soft"], ["low_coverage"])
        early = [(r["start"], r["end"]) for r in data["words"] if r["start"] < 9.0]
        self.assertEqual(early, [(r["start"], r["end"]) for r in primary if r["start"] < 9.0])   # anchors untouched
        self.assertEqual(len(data["words"]), len(truth))

    def test_low_coverage_keeps_the_original_measured_clock_when_recovery_fails_or_is_no_better(self) -> None:
        truth, primary = self.low_coverage_primary()
        for fakes in (ChunkFakes(truth, fail=True), ChunkFakes(primary)):                  # fails / hears no more
            data, report = self.recover(primary, fakes)
            self.assertTrue(fakes.calls)
            self.assertEqual((report["status"], report["selected"]), ("degraded", "whisper_primary"))
            self.assertIn("original measured clock is kept", report["reason"])
            self.assertEqual(data["words"], primary)                                         # never blocked

    def test_a_fatally_broken_clock_attempts_recovery_and_blocks_only_without_a_survivor(self) -> None:
        truth = true_clock()
        collapsed = [{**r, "start": 7.0, "end": 7.0} for r in truth]
        data, report = self.recover(collapsed, ChunkFakes(truth))
        self.assertEqual(report["status"], "recovered")
        fakes = ChunkFakes(truth, fail=True)
        with self.assertRaises(cc.ClockUnavailable):
            self.recover(collapsed, fakes)
        self.assertTrue(fakes.calls)                                                         # tried before blocking

    def test_soft_symptoms_after_recovery_publish_degraded(self) -> None:
        sparse = true_clock()[::3]                                     # measured but covers a third of the words
        sparse[4]["start"], sparse[9]["start"] = 0.1, 0.1
        data, report = self.recover(sparse, ChunkFakes(sparse, fail=True))
        self.assertEqual(report["status"], "degraded")
        self.assertTrue(data["words"])


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
class CaptionStageClockTests(unittest.TestCase):
    """The real caption stage: a collapsed primary clock is recovered from real chunk cuts of the audio."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="mimir clock ş ")
        cls.audio = Path(cls._tmp.name) / "final.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "anoisesrc=color=pink:sample_rate=48000:duration=24:amplitude=0.05", "-ac", "1",
                        "-c:a", "pcm_s16le", str(cls.audio)], check=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_collapsed_primary_clock_is_recovered_before_alignment(self) -> None:
        words = [f"w{i}" for i in range(55)]
        truth = [{"word": w, "start": round(0.3 + i * 0.42, 3), "end": round(0.6 + i * 0.42, 3)}
                 for i, w in enumerate(words)]
        primary = [dict(r) for r in truth]
        for r in primary[20:40]:
            r["start"] = r["end"] = 10.0
        text = " ".join(words)

        def timer(path, **_):
            match = re.search(r"_(\d{7})_(\d{7})_clock_chunk", Path(path).name)
            if not match:
                return {"words": primary}
            start, end = int(match.group(1)) / 1000.0, int(match.group(2)) / 1000.0
            return {"words": [{**r, "start": round(r["start"] - start, 3), "end": round(r["end"] - start, 3)}
                              for r in truth if start <= r["start"] and r["end"] <= end]}

        with mock.patch.object(scs, "_crosscheck_with_confidence", return_value=(text, [], "logprobs")), \
                mock.patch.object(vod_processor, "transcribe_word_timing", side_effect=timer), \
                mock.patch.object(caption_judge, "default_judge", return_value=None):
            out, _ratio, _source, quality = scs._transcribe_edited_words(self.audio, 24.0, accurate_text=text)
        self.assertEqual(quality["clock_health"]["status"], "recovered")
        by_word = {w["word"]: w for w in out}
        self.assertAlmostEqual(by_word["w30"]["edited_start"], truth[30]["start"], places=2)
        self.assertAlmostEqual(by_word["w5"]["edited_start"], truth[5]["start"], places=3)


if __name__ == "__main__":
    unittest.main()
