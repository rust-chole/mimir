"""Caption accuracy: suspicion routing, strict acoustic resolution, clock integrity.

No network: every ear is a deterministic fake. Real FFmpeg cuts the micro windows.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai import vod_processor
from ai.editor import speaker_caption_support as scs

HAVE_FFMPEG = bool(shutil.which("ffmpeg"))


def whisper(words: list[tuple[str, float, float]]) -> dict:
    return {"words": [{"word": w, "start": s, "end": e} for w, s, e in words], "text": " ".join(w for w, _, _ in words)}


class TokenConfidenceTests(unittest.TestCase):
    def test_token_stream_maps_to_word_confidence(self) -> None:
        words = scs._token_word_probabilities([(" Hello", 0.9), (",", 0.8), (" wor", 0.3), ("ld", 0.95)])
        self.assertEqual(words, [("Hello,", 0.8), ("world", 0.3)])

    def test_agreeing_but_unsure_content_words_become_suspects(self) -> None:
        reference = [("so", 0.2), ("Marra", 0.3), ("said", 0.97), ("hi", 0.1), ("again", 0.9)]
        rows = scs._low_confidence_spans("so Marra said hi again", reference)
        self.assertEqual([(r["start"], r["source"]) for r in rows], [(1, "low_confidence")])   # "so"/"hi" too short
        # A word the ears DISAGREE on is the disagreement locator's job, not this one.
        self.assertEqual(scs._low_confidence_spans("so Mara said hi again", reference), [])

    def test_confidence_suspects_rank_after_acoustic_disagreements(self) -> None:
        rows = [{"start": 5, "end": 6, "severity": 0.8, "source": "low_confidence", "reason": "x"},
                {"start": 1, "end": 2, "severity": 1.0, "source": "asr_disagreement", "reason": "y"}]
        with mock.patch.object(scs, "CAPTION_MICRO_MAX_SPANS", 1):
            merged = scs._merge_suspect_spans(rows, 10)
        self.assertEqual(merged[0]["sources"], ["asr_disagreement"])

    def test_crosscheck_falls_back_to_plain_text_without_logprobs(self) -> None:
        client = mock.Mock()
        client.audio.transcriptions.create.side_effect = TypeError("include not supported")
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "a.wav"
            audio.write_bytes(b"RIFF")
            with mock.patch.object(scs, "client", client), \
                    mock.patch.object(vod_processor, "transcribe_caption_crosscheck_text", return_value="plain text"):
                text, words, status = scs._crosscheck_with_confidence(audio)
        self.assertEqual((text, words), ("plain text", []))
        self.assertTrue(status.startswith("logprobs_unavailable"))

    def test_crosscheck_reads_logprobs(self) -> None:
        client = mock.Mock()
        client.audio.transcriptions.create.return_value = {
            "text": "hi Marra", "logprobs": [{"token": "hi", "logprob": -0.01}, {"token": " Mar", "logprob": -2.0},
                                             {"token": "ra", "logprob": -0.1}]}
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "a.wav"
            audio.write_bytes(b"RIFF")
            with mock.patch.object(scs, "client", client):
                text, words, status = scs._crosscheck_with_confidence(audio)
        self.assertEqual((text, status), ("hi Marra", "logprobs"))
        self.assertLess(dict(words)["Marra"], 0.2)


class ClockIntegrityTests(unittest.TestCase):
    def assert_clock(self, rows, anchors: dict[str, float]) -> None:
        for previous, current in zip(rows, rows[1:]):
            self.assertLessEqual(float(previous["end"]), float(current["start"]) + 0.011)
        for row in rows:
            if row["word"] in anchors:
                self.assertAlmostEqual(float(row["start"]), anchors[row["word"]], places=6)

    def test_repeated_words_never_drift_later_anchors(self) -> None:
        data = whisper([("no", 1.0, 1.2), ("no", 1.3, 1.5), ("way", 1.6, 1.9), ("seriously", 4.0, 4.6)])
        rows, _ = vod_processor.align_text_to_fixed_whisper_clock("no no no no way seriously", data, 6.0)
        self.assertEqual(" ".join(r["word"] for r in rows).split(), "no no no no way seriously".split())
        self.assert_clock(rows, {"way": 1.6, "seriously": 4.0})

    def test_a_missing_or_hallucinated_whisper_span_stays_local(self) -> None:
        data = whisper([("go", 0.5, 0.7), ("thanks", 1.0, 1.3), ("for", 1.35, 1.5), ("watching", 1.55, 2.0),
                        ("then", 5.0, 5.2), ("he", 5.3, 5.4), ("fell", 5.5, 5.9)])
        rows, _ = vod_processor.align_text_to_fixed_whisper_clock("go then he fell", data, 7.0)
        self.assertEqual([r["word"] for r in rows], ["go", "then", "he", "fell"])
        self.assert_clock(rows, {"go": 0.5, "then": 5.0, "he": 5.3, "fell": 5.5})

    def test_repeated_phrase_far_apart_anchors_each_occurrence_locally(self) -> None:
        data = whisper([("let's", 1.0, 1.2), ("go", 1.25, 1.5), ("wait", 3.0, 3.3), ("let's", 8.0, 8.2),
                        ("go", 8.25, 8.6)])
        rows, _ = vod_processor.align_text_to_fixed_whisper_clock("let's go wait what let's go", data, 10.0)
        self.assertEqual([r["word"] for r in rows], "let's go wait what let's go".split())
        self.assert_clock(rows, {"wait": 3.0})
        self.assertAlmostEqual(float(rows[-2]["start"]), 8.0, places=6)
        what = next(r for r in rows if r["word"] == "what")
        self.assertTrue(3.3 <= float(what["start"]) < 8.0)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
class SuspicionToResolutionTests(unittest.TestCase):
    """A low-confidence word the ears agree on is re-heard locally; it changes only by strict majority."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="mimir caption ş ")
        cls.audio = Path(cls._tmp.name) / "final.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "anoisesrc=color=pink:sample_rate=48000:duration=8:amplitude=0.05", "-ac", "1",
                        "-c:a", "pcm_s16le", str(cls.audio)], check=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def run_stage(self, micro_answer):
        primary = "then Marra told me to wait here"
        timing = whisper([(w, 0.5 + i * 0.5, 0.8 + i * 0.5) for i, w in enumerate(primary.split())])
        crosscheck = (primary, [(w, 0.25 if w == "Marra" else 0.97) for w in primary.split()], "logprobs")
        micro_calls = []

        def micro(view, *, model, label, **kwargs):
            micro_calls.append(label)
            return micro_answer(label)

        with mock.patch.object(scs, "_crosscheck_with_confidence", return_value=crosscheck), \
                mock.patch.object(vod_processor, "transcribe_word_timing", return_value=timing), \
                mock.patch.object(vod_processor, "transcribe_caption_micro_pass", side_effect=micro):
            words, _ratio, _source, quality = scs._transcribe_edited_words(
                self.audio, 8.0, accurate_text=primary, known_names=None)
        return words, quality, micro_calls

    def test_unanimous_acoustic_ears_correct_the_agreeing_but_unsure_word(self) -> None:
        words, quality, calls = self.run_stage(lambda label: "then Mara told me to wait here")
        self.assertEqual(quality["low_confidence_flagged_spans"], 1)
        self.assertEqual(len(calls), 3)                                # one strict round was enough
        self.assertEqual([w["word"] for w in words][1], "Mara")
        self.assertEqual(quality["micro_accuracy"]["corrected_spans"], 1)

    def test_split_ears_keep_the_primary_word_and_report_it_unresolved(self) -> None:
        answers = iter(["then Mara told me", "then Marra told me", "then Mora told me", "then Mara told me",
                        "then Marra told me"])
        words, quality, calls = self.run_stage(lambda label: next(answers))
        self.assertEqual(len(calls), 5)                                # 3 + 2 precision ears, no more
        self.assertEqual([w["word"] for w in words][1], "Marra")       # never a fluent guess
        self.assertEqual(quality["micro_accuracy"]["unresolved_spans"], 1)
        detail = quality["micro_accuracy"]["details"][0]
        self.assertFalse(detail["resolved"])                           # caption truth marks it uncertain


if __name__ == "__main__":
    unittest.main()
