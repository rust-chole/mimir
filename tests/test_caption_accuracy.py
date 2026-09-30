"""Caption accuracy: A/B dispute detection, bounded escalation, judge grounding, freeze, legacy clock.

No network: every ear and the judge are deterministic fakes. Real FFmpeg cuts the evidence windows.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from ai.caption_stack import audio as analysis_audio
from ai.caption_stack import config, legacy_whisper, lexical, qwen_omni
from ai.editor import caption_judge

HAVE_FFMPEG = bool(shutil.which("ffmpeg"))


def whisper(words: list[tuple[str, float, float]]) -> dict:
    return {"words": [{"word": w, "start": s, "end": e} for w, s, e in words], "text": " ".join(w for w, _, _ in words)}


def ear(text: str, role: str = lexical.PRIMARY, uncertain: tuple = ()) -> lexical.EarTranscript:
    return lexical.EarTranscript(role, "qwen_omni", "fake-qwen", role == lexical.PRECISION,
                                 tuple(lexical.display_tokens(text)), tuple(uncertain))


class DisputeDetectionTests(unittest.TestCase):
    def disputes(self, a: str, b: str, **kwargs) -> list[lexical.Dispute]:
        return lexical.find_disputes(ear(a), ear(b, lexical.PRECISION), chunk=0, **kwargs)

    def test_harmless_differences_are_not_disputes(self) -> None:
        for a, b in (("I said: no WAY!", "i said no way"),               # case + punctuation
                     ("don’t do that", "don't do that"),                 # apostrophe glyph
                     ("a well-known streamer", "a well known streamer"),  # hyphenation
                     ("he got twelve wins", "he got 12 wins")):           # same number, other spelling
            with self.subTest(a=a):
                self.assertEqual(self.disputes(a, b), [])

    def test_meaningful_differences_are_local_disputes(self) -> None:
        cases = {("I can do it", "I can't do it"): "negation",
                 ("he got 12 wins", "he got 20 wins"): "number",
                 ("we were late", "we're late"): "contraction"}
        for (a, b), flag in cases.items():
            with self.subTest(a=a):
                found = self.disputes(a, b)
                self.assertEqual(len(found), 1)
                self.assertIn(flag, found[0].flags)
        missing = self.disputes("told you not to go", "told you to go")
        self.assertEqual(missing[0].kinds, ["primary_only_words"])
        self.assertEqual(missing[0].candidates[lexical.PRECISION]["tokens"], ())
        inserted = self.disputes("told you to go", "told you not to go")
        self.assertEqual(inserted[0].kinds, ["precision_only_words"])
        self.assertIn("not", inserted[0].candidates[lexical.PRECISION]["tokens"])

    def test_disputes_stay_local_and_never_overlap(self) -> None:
        a = " ".join(f"w{i}" for i in range(40))
        b = " ".join(f"w{i}" if i % 3 else f"x{i}" for i in range(40))   # a difference every third word
        found = self.disputes(a, b)
        self.assertTrue(all(d.core[1] - d.core[0] <= lexical.MAX_DISPUTE_TOKENS + 1 for d in found))
        for left, right in zip(found, found[1:]):
            self.assertLessEqual(left.core[1], right.core[0])

    def test_an_ear_doubting_a_word_both_heard_opens_a_dispute_with_its_alternatives(self) -> None:
        doubt = qwen_omni.UncertainSpan("Marra", ("Mara",))
        found = lexical.find_disputes(ear("then Marra told me", uncertain=(doubt,)),
                                      ear("then Marra told me", lexical.PRECISION), chunk=0)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].kinds, ["self_uncertain:qwen_primary"])
        self.assertEqual(found[0].candidates["qwen_primary_alt1_1"]["tokens"], ("Mara",))


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
class ResolutionTests(unittest.TestCase):
    """A dispute changes words only by agreement of an independent ear or a grounded judge verdict."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="mimir caption ş ")
        wav = Path(cls._tmp.name) / "final.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "anoisesrc=color=pink:sample_rate=16000:duration=8:amplitude=0.05", "-ac", "1",
                        "-c:a", "pcm_s16le", str(wav)], check=True)
        cls.audio = analysis_audio.load_analysis_audio(wav)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    PRIMARY = "then Marra told me to wait here"
    PRECISION = "then Mara told me to wait here"

    def run_stage(self, fallback_heard: str | None, judge=None, *, fallback: str = "openai_transcribe"):
        settings = config.CaptionStackSettings(
            primary_provider="qwen_omni", qwen_model="fake-qwen", qwen_reasoning_effort="none", qwen_timeout_s=30,
            qwen_max_retries=0, dashscope_base_url="https://example.invalid/v1", transcribe_fallback_provider=fallback,
            transcribe_fallback_model="fake-diverse", alignment_provider="qwen3_forced_aligner",
            alignment_fallback_provider="none", aligner_model="fake", aligner_device="cpu", aligner_dtype="auto",
            language="en", dashscope_api_key="sk-test")
        calls: list[tuple[str, bool]] = []

        def openai(path, *, prompted, model, language, verified_terms):
            calls.append((Path(path).name, prompted))
            return fallback_heard or ""

        def locate(chunk, tokens):
            return [(0.5 + i * 0.5, 0.8 + i * 0.5) for i in range(len(tokens))]

        passes = [(ear(self.PRIMARY), ear(self.PRECISION, lexical.PRECISION))]
        with tempfile.TemporaryDirectory() as tmp:
            frozen, report = lexical.resolve_and_freeze(
                self.audio, [(0.0, self.audio.duration)], passes, settings=settings, locator=locate, judge=judge,
                ears=lexical.Ears(openai=openai), workdir=Path(tmp))
        return frozen, report, calls

    def fake_judge(self, verdict, text="", ears=("fallback_local",), confidence=0.86):
        calls = []

        def judge(*, name, instructions, payload, schema):
            calls.append(payload)
            return {"decisions": [{"span_id": s["span_id"], "verdict": verdict, "text": text,
                                   "supporting_ears": list(ears), "confidence": confidence, "reason": "test"}
                                  for s in payload["spans"]]}
        return judge, calls

    def test_an_independent_ear_agreeing_with_one_reading_settles_it_without_the_judge(self) -> None:
        judge, judge_calls = self.fake_judge("keep_primary")
        frozen, report, calls = self.run_stage("then Mara told me", judge)
        self.assertEqual(judge_calls, [])                                   # no strong-model budget spent
        self.assertEqual(frozen.tokens[1], "Mara")
        self.assertEqual(frozen.status[1], "resolved")
        self.assertEqual(calls, [("c0_d0.wav", False)])                     # one small, unprompted local window
        self.assertTrue(report["spans"][0]["decided_by"].startswith("agreement_qwen_precision"))
        frozen, _report, _calls = self.run_stage("then Marra told me", judge)
        self.assertEqual((frozen.tokens[1], frozen.status[1]), ("Marra", "resolved"))

    def test_the_judge_decides_a_split_from_the_smallest_evidence_package(self) -> None:
        judge, judge_calls = self.fake_judge("use_heard", "Mara", ears=("qwen_precision",))
        frozen, report, _calls = self.run_stage("then Mora told me", judge)
        span = judge_calls[0]["spans"][0]
        self.assertEqual([e["ear"] for e in span["ears"]], ["qwen_primary", "qwen_precision", "fallback_local"])
        self.assertEqual([e["prompted_with_context"] for e in span["ears"]], [False, True, False])
        self.assertNotIn("measured_clock_words", span)                     # no timing evidence in a WHAT decision
        self.assertLessEqual(span["audio_window_s"][1] - span["audio_window_s"][0], lexical.LOCAL_MAX_WINDOW_S)
        self.assertEqual(judge_calls[0]["primary_transcript"]["text"], self.PRIMARY)
        self.assertEqual((frozen.tokens[1], frozen.status[1]), ("Mara", "resolved"))
        self.assertEqual(report["judge"]["status"], "judged")

    def test_a_word_no_ear_heard_is_rejected_and_the_primary_stays_uncertain(self) -> None:
        judge, _ = self.fake_judge("use_heard", "Maria", ears=("qwen_precision",))
        frozen, report, _ = self.run_stage("then Mora told me", judge)
        self.assertEqual((frozen.tokens[1], frozen.status[1]), ("Marra", "uncertain"))
        self.assertIn("no ear heard", report["spans"][0]["guard"])

    def test_no_judge_or_a_failing_judge_keeps_primary_words_marked_uncertain(self) -> None:
        def broken(**kwargs):
            raise TimeoutError("api down")
        for judge, status in ((None, "unavailable"), (broken, "failed")):
            with self.subTest(status=status):
                frozen, report, _ = self.run_stage("then Mora told me", judge)
                self.assertEqual((frozen.tokens[1], frozen.status[1]), ("Marra", "uncertain"))
                self.assertEqual(report["judge"]["status"], status)

    def test_a_deletion_is_never_settled_by_agreement_alone(self) -> None:
        settings_frozen, report, _ = self.run_stage("then told me", None)
        self.assertEqual(settings_frozen.tokens[1], "Marra")               # kept, not silently deleted
        self.assertEqual(report["settled_by_agreement"], 0)

    def test_without_a_fallback_ear_no_extra_listening_happens(self) -> None:
        judge, judge_calls = self.fake_judge("unresolved")
        frozen, report, calls = self.run_stage("then Mara told me", judge, fallback="none")
        self.assertEqual(calls, [])
        self.assertEqual(report["fallback_calls"], {"full": 0, "local": 0})
        self.assertEqual(len(judge_calls[0]["spans"][0]["ears"]), 2)

    def test_the_frozen_transcript_is_immutable_and_self_verifying(self) -> None:
        frozen, _report, _ = self.run_stage("then Mara told me", None)
        self.assertTrue(frozen.intact())
        with self.assertRaises(Exception):
            frozen.tokens = ("x",)                                         # type: ignore[misc]
        with self.assertRaises(TypeError):
            frozen.spans[0]["to"] = "rewritten"                            # type: ignore[index]
        forged = lexical.FrozenTranscript(("then", "Maria"), frozen.status[:2], frozen.chunk_of[:2], frozen.chunks,
                                          (), frozen.language, frozen.signature)
        self.assertFalse(forged.intact())


class JudgeGuardTests(unittest.TestCase):
    def span(self, **overrides):
        base = dict(span_id="s", core=(1, 2), current="Marra", window_text="then Marra told me",
                    context_before="", context_after="", audio_window=(0.0, 3.0), suspicion=["different_word"],
                    ears=[{"ear": "qwen_primary", "heard_window": "then Marra told me", "heard_core": "Marra"},
                          {"ear": "qwen_precision", "heard_window": "then Mara told me", "heard_core": "Mara"},
                          {"ear": "fallback_local", "heard_window": "then told me", "heard_core": ""}],
                    strict=None, clock=[], core_clock=[])
        base.update(overrides)
        return caption_judge.SpanEvidence(**base)

    def decide(self, span, verdict, text, ears=("qwen_precision",), confidence=0.9, names=()):
        return caption_judge.check_span_decision(span, {"verdict": verdict, "text": text, "confidence": confidence,
                                                        "supporting_ears": list(ears), "reason": ""}, names)

    def test_the_judge_can_never_emit_a_time(self) -> None:
        for schema in (caption_judge.SPAN_JUDGE_SCHEMA, caption_judge.ENTITY_JUDGE_SCHEMA):
            fields = schema["properties"]["decisions"]["items"]["properties"]
            self.assertFalse([f for f in fields if re.search(r"start|end|time|clock|second", f)], fields)

    def test_grounding_guard(self) -> None:
        span = self.span()
        self.assertEqual(self.decide(span, "use_heard", "Mara")[0].phrase, "Mara")
        self.assertIsNone(self.decide(span, "use_heard", "Maria")[0])                      # never heard
        self.assertIsNone(self.decide(span, "use_heard", "Mara", ears=())[0])               # no supporting ear
        self.assertIsNone(self.decide(span, "use_heard", "Mara told me then Mara now")[0])  # not local
        self.assertIsNone(self.decide(span, "use_heard", "")[0])                           # 1 of 3 heard nothing
        silent = self.span(ears=[{"ear": "qwen_primary", "heard_core": "Marra"},
                                 {"ear": "qwen_precision", "heard_core": ""},
                                 {"ear": "fallback_local", "heard_core": ""}])
        deletion = self.decide(silent, "use_heard", "", ears=("qwen_precision", "fallback_local"))[0]
        self.assertEqual((deletion.phrase, deletion.resolved), ("", True))                 # 2 of 3 heard nothing
        unsure = self.decide(span, "use_heard", "Mara", confidence=0.3)[0]
        self.assertEqual((unsure.phrase, unsure.resolved), (None, False))                   # low confidence = uncertain

    def test_verified_name_may_be_used_when_an_ear_heard_a_near_spelling(self) -> None:
        accepted, _ = self.decide(self.span(), "use_heard", "Mira", names=["Mira"])
        self.assertEqual(accepted.phrase, "Mira")

    def test_the_payload_carries_only_evidence_that_exists(self) -> None:
        payload = self.span().to_payload()
        self.assertNotIn("measured_clock_words", payload)
        self.assertNotIn("deterministic_vote", payload)
        self.assertEqual(json.loads(json.dumps(payload))["span_id"], "s")


class LegacyClockIntegrityTests(unittest.TestCase):
    """The migration-fallback Whisper clock keeps its invariants (frozen words, fixed anchors)."""

    def assert_clock(self, rows, anchors: dict[str, float]) -> None:
        for previous, current in zip(rows, rows[1:]):
            self.assertLessEqual(float(previous["end"]), float(current["start"]) + 0.011)
        for row in rows:
            if row["word"] in anchors:
                self.assertAlmostEqual(float(row["start"]), anchors[row["word"]], places=6)

    def test_repeated_words_never_drift_later_anchors(self) -> None:
        data = whisper([("no", 1.0, 1.2), ("no", 1.3, 1.5), ("way", 1.6, 1.9), ("seriously", 4.0, 4.6)])
        rows, _ = legacy_whisper.align_text_to_fixed_whisper_clock("no no no no way seriously", data, 6.0)
        self.assertEqual(" ".join(r["word"] for r in rows).split(), "no no no no way seriously".split())
        self.assert_clock(rows, {"way": 1.6, "seriously": 4.0})

    def test_a_missing_or_hallucinated_whisper_span_stays_local(self) -> None:
        data = whisper([("go", 0.5, 0.7), ("thanks", 1.0, 1.3), ("for", 1.35, 1.5), ("watching", 1.55, 2.0),
                        ("then", 5.0, 5.2), ("he", 5.3, 5.4), ("fell", 5.5, 5.9)])
        rows, _ = legacy_whisper.align_text_to_fixed_whisper_clock("go then he fell", data, 7.0)
        self.assertEqual([r["word"] for r in rows], ["go", "then", "he", "fell"])
        self.assert_clock(rows, {"go": 0.5, "then": 5.0, "he": 5.3, "fell": 5.5})

    def test_repeated_phrase_far_apart_anchors_each_occurrence_locally(self) -> None:
        data = whisper([("let's", 1.0, 1.2), ("go", 1.25, 1.5), ("wait", 3.0, 3.3), ("let's", 8.0, 8.2),
                        ("go", 8.25, 8.6)])
        rows, _ = legacy_whisper.align_text_to_fixed_whisper_clock("let's go wait what let's go", data, 10.0)
        self.assertEqual([r["word"] for r in rows], "let's go wait what let's go".split())
        self.assert_clock(rows, {"wait": 3.0})
        self.assertAlmostEqual(float(rows[-2]["start"]), 8.0, places=6)
        what = next(r for r in rows if r["word"] == "what")
        self.assertTrue(3.3 <= float(what["start"]) < 8.0)


if __name__ == "__main__":
    unittest.main()
