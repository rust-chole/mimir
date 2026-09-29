"""Caption accuracy: suspicion routing, strict acoustic resolution, clock integrity.

No network: every ear is a deterministic fake. Real FFmpeg cuts the micro windows.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai import vod_processor
from ai.editor import caption_judge
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


class JudgeGuardTests(unittest.TestCase):
    def span(self, **overrides):
        base = dict(span_id="s", core=(1, 2), current="Marra", window_text="then Marra told me",
                    context_before="", context_after="", audio_window=(0.0, 3.0), suspicion=["asr_disagreement"],
                    ears=[{"ear": "asr_1", "heard_window": "then Mara told me", "heard_core": "Mara"},
                          {"ear": "asr_2", "heard_window": "then told me", "heard_core": ""},
                          {"ear": "asr_3", "heard_window": "then told me", "heard_core": ""}],
                    strict=None, clock=[("then", 0.5, 0.8), ("Mara", 1.0, 1.3)], core_clock=["Mara"])
        base.update(overrides)
        return caption_judge.SpanEvidence(**base)

    def decide(self, span, verdict, text, ears=("asr_1",), confidence=0.9):
        return caption_judge.check_span_decision(span, {"verdict": verdict, "text": text, "confidence": confidence,
                                                        "supporting_ears": list(ears), "reason": ""})

    def test_the_judge_can_never_emit_a_time(self) -> None:
        for schema in (caption_judge.SPAN_JUDGE_SCHEMA, caption_judge.ENTITY_JUDGE_SCHEMA):
            fields = schema["properties"]["decisions"]["items"]["properties"]
            self.assertFalse([f for f in fields if re.search(r"start|end|time|clock|second", f)], fields)

    def test_grounding_guard(self) -> None:
        span = self.span()
        self.assertEqual(self.decide(span, "use_heard", "Mara")[0].phrase, "Mara")
        self.assertIsNone(self.decide(span, "use_heard", "Maria")[0])                     # never heard
        self.assertIsNone(self.decide(span, "use_heard", "Mara", ears=())[0])              # no supporting ear
        self.assertIsNone(self.decide(span, "use_heard", "Mara told me then Mara now")[0])  # not local
        self.assertIsNone(self.decide(span, "use_heard", "")[0])                          # the clock heard a word
        deletion = self.decide(self.span(core_clock=[]), "use_heard", "")[0]
        self.assertEqual((deletion.phrase, deletion.resolved), ("", True))                # 2 of 3 ears + clock silent
        unsure = self.decide(span, "use_heard", "Mara", confidence=0.3)[0]
        self.assertEqual((unsure.phrase, unsure.resolved), (None, False))                  # low confidence = uncertain

    def test_a_word_only_in_the_primary_context_is_not_evidence(self) -> None:
        span = self.span(window_text="then Marra told me to wait here", context_after="to wait here")
        accepted, problem = self.decide(span, "use_heard", "wait")
        self.assertIsNone(accepted)
        self.assertIn("no ear heard at the disputed core", problem)

    def test_one_context_prompted_ear_is_not_enough_two_are(self) -> None:
        ears = [{"ear": "asr_1", "heard_core": "Marra", "prompted": False},
                {"ear": "asr_3", "heard_core": "Mara", "prompted": True},
                {"ear": "asr_4", "heard_core": "Marra", "prompted": True}]
        span = self.span(ears=ears, core_clock=[])
        self.assertIsNone(self.decide(span, "use_heard", "Mara", ears=("asr_3",))[0])     # a context echo
        ears[2]["heard_core"] = "Mara"
        accepted, _ = self.decide(self.span(ears=ears, core_clock=[]), "use_heard", "Mara", ears=("asr_3", "asr_4"))
        self.assertEqual(accepted.phrase, "Mara")
        self.assertIn("prompted only", accepted.grounding[0][1])

    def test_the_cited_ears_must_carry_the_change(self) -> None:
        span = self.span(ears=[{"ear": "asr_1", "heard_core": "Mara", "prompted": False},
                               {"ear": "asr_2", "heard_core": "Marra", "prompted": False}], core_clock=[])
        accepted, problem = self.decide(span, "use_heard", "Mara", ears=("asr_2",))
        self.assertIsNone(accepted)
        self.assertIn("do not carry", problem)
        accepted, problem = self.decide(span, "use_heard", "Mara", ears=("asr_1", "asr_2"))
        self.assertIsNone(accepted)
        self.assertIn("asr_2", problem)
        accepted, problem = self.decide(span, "use_heard", "Mara", ears=("asr_1", "made_up_ear"))
        self.assertIsNone(accepted)
        self.assertIn("unknown supporting ear", problem)

    def test_unchanged_words_need_no_support_and_evidence_may_combine(self) -> None:
        span = self.span(current="go to the store now", window_text="so go to the store now ok",
                         ears=[{"ear": "asr_1", "heard_core": "go to a store now", "prompted": False},
                               {"ear": "asr_2", "heard_core": "go to the stall now", "prompted": False}],
                         core_clock=["go", "to", "a", "stall", "now"])
        accepted, _ = self.decide(span, "use_heard", "go to a stall now", ears=("asr_1", "asr_2"))
        self.assertEqual(accepted.phrase, "go to a stall now")
        self.assertEqual([word for word, _ in accepted.grounding], ["a", "stall"])      # only the changes

    def test_a_rejected_answer_falls_back_per_span(self) -> None:
        strict = self.span(strict={"phrase": "Mara", "votes": 4, "of": 5})
        decisions, meta = caption_judge.resolve_spans(
            [strict, self.span(span_id="t")], primary_text="x", crosscheck_text="",
            judge=lambda **_: {"decisions": [
                {"span_id": sid, "verdict": "use_heard", "text": "Maria", "supporting_ears": ["asr_1"],
                 "confidence": 0.9, "reason": ""} for sid in ("s", "t")]})
        self.assertEqual((decisions["s"].phrase, decisions["s"].source), ("Mara", "strict_vote_4_of_5"))
        self.assertEqual((decisions["t"].phrase, decisions["t"].resolved), (None, False))   # primary + uncertain
        self.assertTrue(decisions["t"].guard)
        self.assertEqual(meta["status"], "judged")                                         # the run goes on
        self.assertEqual(len(meta["guard_rejected"]), 2)

    def test_verified_name_may_be_used_when_an_ear_heard_a_near_spelling(self) -> None:
        span = self.span()
        accepted, _ = caption_judge.check_span_decision(
            span, {"verdict": "use_heard", "text": "Mira", "confidence": 0.9, "supporting_ears": ["asr_1"],
                   "reason": ""}, verified_names=["Mira"])
        self.assertEqual(accepted.phrase, "Mira")


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

    PRIMARY = "then Marra told me to wait here"

    def run_stage(self, micro_answer, judge=None):
        primary = self.PRIMARY
        timing = whisper([(w, 0.5 + i * 0.5, 0.8 + i * 0.5) for i, w in enumerate(primary.split())])
        crosscheck = (primary, [(w, 0.25 if w == "Marra" else 0.97) for w in primary.split()], "logprobs")
        micro_calls = []

        def micro(view, *, model, label, **kwargs):
            micro_calls.append(label)
            return micro_answer(label)

        with mock.patch.object(scs, "_crosscheck_with_confidence", return_value=crosscheck), \
                mock.patch.object(vod_processor, "transcribe_word_timing", return_value=timing), \
                mock.patch.object(vod_processor, "transcribe_caption_micro_pass", side_effect=micro), \
                mock.patch.object(caption_judge, "default_judge", return_value=judge):
            words, _ratio, _source, quality = scs._transcribe_edited_words(
                self.audio, 8.0, accurate_text=primary, known_names=None)
        return words, quality, micro_calls

    SPLIT = ["then Mara told me", "then Marra told me", "then Mora told me", "then Mara told me",
             "then Marra told me"]

    def split_answers(self):
        answers = iter(self.SPLIT)
        return lambda label: next(answers)

    def fake_judge(self, verdict, text="", ears=("asr_1", "asr_4"), confidence=0.86):
        calls = []

        def judge(*, name, instructions, payload, schema):
            calls.append(payload)
            return {"decisions": [{"span_id": s["span_id"], "verdict": verdict, "text": text,
                                   "supporting_ears": list(ears), "confidence": confidence, "reason": "test"}
                                  for s in payload["spans"]]}
        return judge, calls

    def test_unanimous_ears_settle_the_span_without_the_strong_model(self) -> None:
        judge, calls = self.fake_judge("keep_primary")
        words, quality, _ = self.run_stage(lambda label: "then Mara told me to wait here", judge)
        self.assertEqual(calls, [])                                    # no strong-model budget spent
        self.assertEqual([w["word"] for w in words][1], "Mara")
        self.assertTrue(quality["micro_accuracy"]["details"][0]["selected_source"].startswith("settled_unanimous"))

    def by_ear(self, heard: dict[str, str]):
        labels = {"raw acoustic": "asr_1", "enhanced acoustic": "asr_2", "context diverse": "asr_3",
                  "enhanced diverse": "asr_4", "context precision": "asr_5"}
        return lambda label: f"then {heard[labels[label]]} told me"

    def test_a_strong_acoustic_majority_is_settled_without_the_strong_model(self) -> None:
        judge, calls = self.fake_judge("unresolved")
        heard = {"asr_1": "Mara", "asr_2": "Mara", "asr_3": "Marra", "asr_4": "Mara", "asr_5": "Mara"}
        words, quality, micro_calls = self.run_stage(self.by_ear(heard), judge)
        self.assertEqual(len(micro_calls), 5)
        self.assertEqual(calls, [])                                   # only a prompted ear dissented
        self.assertEqual([w["word"] for w in words][1], "Mara")
        self.assertEqual(quality["micro_accuracy"]["details"][0]["selected_source"], "settled_acoustic_4_of_5")

    def test_an_unprompted_dissent_still_reaches_the_judge(self) -> None:
        judge, calls = self.fake_judge("unresolved")
        heard = {"asr_1": "Mara", "asr_2": "Marra", "asr_3": "Mara", "asr_4": "Mara", "asr_5": "Mara"}
        self.run_stage(self.by_ear(heard), judge)
        self.assertEqual(len(calls), 1)                                # one batched call, never per word

    def test_judge_decides_a_split_span_from_the_evidence_and_the_clock_keeps_timing(self) -> None:
        judge, calls = self.fake_judge("use_heard", "Mara")
        words, quality, micro_calls = self.run_stage(self.split_answers(), judge)
        self.assertEqual(len(micro_calls), 5)
        span = calls[0]["spans"][0]
        self.assertEqual([e["ear"] for e in span["ears"]], ["asr_1", "asr_2", "asr_3", "asr_4", "asr_5", "crosscheck"])
        self.assertEqual([e["prompted_with_context"] for e in span["ears"]][:3], [False, False, True])
        self.assertIn(["Marra", 1.0, 1.3], span["measured_clock_words"])          # measured timing candidates
        self.assertEqual(span["crosscheck_low_confidence"], [["Marra", 0.25]])
        self.assertIsNone(span["strict_acoustic_vote"])                            # 2/2/1: no strict majority
        self.assertEqual(calls[0]["primary_transcript"]["text"], self.PRIMARY)
        self.assertEqual([w["word"] for w in words][1], "Mara")
        self.assertAlmostEqual(words[1]["edited_start"], 1.0, places=3)            # the measured clock, not the judge
        self.assertEqual(quality["lexical_judge"]["status"], "judged")
        detail = quality["micro_accuracy"]["details"][0]
        self.assertEqual((detail["selected_source"], detail["resolved"]), ("caption_judge", True))

    def test_a_word_no_ear_heard_is_rejected_and_the_strict_vote_decides(self) -> None:
        judge, _ = self.fake_judge("use_heard", "Maria")
        words, quality, _ = self.run_stage(self.split_answers(), judge)
        self.assertEqual([w["word"] for w in words][1], "Marra")                   # primary kept
        detail = quality["micro_accuracy"]["details"][0]
        self.assertFalse(detail["resolved"])                                        # and marked uncertain
        self.assertIn("no ear heard", detail["lexical_decision"]["guard"])
        self.assertEqual(quality["lexical_judge"]["guard_rejected"][0]["span_id"], detail["lexical_decision"]["span_id"])

    def test_judge_unresolved_keeps_the_primary_word_marked_uncertain(self) -> None:
        judge, _ = self.fake_judge("unresolved")
        words, quality, _ = self.run_stage(self.split_answers(), judge)
        self.assertEqual([w["word"] for w in words][1], "Marra")
        self.assertFalse(quality["micro_accuracy"]["details"][0]["resolved"])

    def test_judge_failure_falls_back_to_the_strict_vote_and_is_recorded(self) -> None:
        def broken(**kwargs):
            raise TimeoutError("api down")
        words, quality, _ = self.run_stage(self.split_answers(), broken)
        self.assertEqual(quality["lexical_judge"]["status"], "failed")
        self.assertEqual([w["word"] for w in words][1], "Marra")

    def test_unanimous_acoustic_ears_correct_the_agreeing_but_unsure_word(self) -> None:
        words, quality, calls = self.run_stage(lambda label: "then Mara told me to wait here")
        self.assertEqual(quality["low_confidence_flagged_spans"], 1)
        self.assertEqual(len(calls), 3)                                # one strict round was enough
        self.assertEqual([w["word"] for w in words][1], "Mara")
        self.assertEqual(quality["micro_accuracy"]["corrected_spans"], 1)

    def test_split_ears_keep_the_primary_word_and_report_it_unresolved(self) -> None:
        words, quality, calls = self.run_stage(self.split_answers())            # no judge: strict vote
        self.assertEqual(len(calls), 5)                                # 3 + 2 precision ears, no more
        self.assertEqual([w["word"] for w in words][1], "Marra")       # never a fluent guess
        self.assertEqual(quality["micro_accuracy"]["unresolved_spans"], 1)
        detail = quality["micro_accuracy"]["details"][0]
        self.assertFalse(detail["resolved"])                           # caption truth marks it uncertain


if __name__ == "__main__":
    unittest.main()
