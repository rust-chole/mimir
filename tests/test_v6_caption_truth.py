"""MIMIR V6 caption truth: generic entity canonicalization, fail-closed ambiguity,
targeted escalation, timing/speaker invariants and the frozen truth.

Every mechanism is exercised with names that appear nowhere in production code;
the real Tyla/Tyler failure is ONE regression fixture of the same general path.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pro_edit_fixtures as fx  # noqa: F401  (repo root on sys.path)
from ai.editor import caption_truth
from ai.editor import participant_name_lock as lock
from ai.editor import v6_runtime


def profile(sentence: str, speakers: str | list[str], names: dict[str, str], *, flags: list[str] = (),
            start: float = 0.5, step: float = 0.4) -> dict:
    tokens = sentence.split()
    speaker_list = speakers if isinstance(speakers, list) else [speakers] * len(tokens)
    words = [{"word": t, "edited_start": round(start + i * step, 3), "edited_end": round(start + i * step + 0.3, 3),
              "speaker_raw": speaker_list[i], "speaker_label": "", "speaker_confidence": 0.9}
             for i, t in enumerate(tokens)]
    data = {"version": 25, "status": "ok", "timing_basis": "exact_final_48k_audio",
            "clip_duration": round(start + len(tokens) * step + 1.0, 3), "validated_identity_map": names,
            "participant_speakers": sorted(set(speaker_list)), "words": words}
    if flags:
        data["caption_quality"] = {"known_name_flagged_spans": [
            {"reason": f"possible known-name spelling mismatch: {token} vs {name}"} for token, name in flags]}
    return data


def texts(words: list[dict]) -> str:
    return " ".join(w["word"] for w in words)


class EntityCanonicalizationTests(unittest.TestCase):
    def test_verified_identity_with_plausible_confusion_and_context_is_canonical(self) -> None:
        # Unseen names: a direct address to a verified participant.
        data = profile("Brenner, would you pass me the charger?", "A", {"A": "JONAH", "B": "BRENNA"})
        result = lock.lock_participant_names(data)
        self.assertEqual(texts(result.words), "Brenna, would you pass me the charger?")
        row = result.audit["corrections"][0]
        self.assertEqual((row["from"], row["to"], row["participant"]), ("Brenner,", "Brenna,", "B"))
        self.assertIn("direct_address", row["evidence"])
        # Third-person mention by the co-participant, supported by MIMIR's own known-name flag.
        data = profile("I was talking to Kayler yesterday.", "A", {"A": "JONAH", "B": "KAYLA"},
                       flags=[("Kayler", "kayla")])
        self.assertEqual(texts(lock.lock_participant_names(data).words), "I was talking to Kayla yesterday.")

    def test_real_regression_tyla_heard_as_tyler(self) -> None:
        # Recorded failure: every ASR ear (3/3 micro votes) wrote the verified participant
        # TYLA as "Tyler." while KAI addressed her; MIMIR's detector flagged it.
        data = profile("My only option is, um, Tyler. Would you like to go on a date with me?", "A",
                       {"A": "KAI", "B": "TYLA"}, flags=[("Tyler.", "tyla")])
        index = 5
        start = data["words"][index]["edited_start"]
        data["caption_quality"]["micro_accuracy"] = {"details": [{
            "span": [index, index + 1], "audio_window": [0.0, start + 3.0], "selected_phrase": "Tyler.",
            "candidate_votes": [{"phrase": "Tyler.", "votes": 3, "sources": ["asr_1", "asr_2", "asr_3"]}],
            "resolved": True}]}
        result = lock.lock_participant_names(data)
        self.assertEqual(result.words[index]["word"], "Tyla.")
        self.assertEqual(len(result.audit["corrections"]), 1)

    def test_competing_similar_verified_identities_fail_closed(self) -> None:
        data = profile("Tyler, come over here.", "A", {"A": "KAI", "B": "TYLA", "C": "TYLOR"})
        result = lock.lock_participant_names(data)
        self.assertEqual(texts(result.words), "Tyler, come over here.")
        self.assertEqual(result.audit["rejected"][0]["reason"], "ambiguous_between_entities")

    def test_unknown_name_is_never_invented(self) -> None:
        data = profile("Marcus is coming over tonight, you know him?", "A", {"A": "KAI", "B": "TYLA"})
        result = lock.lock_participant_names(data)
        self.assertFalse(result.changed)
        self.assertEqual(result.audit["corrections"], [])
        self.assertEqual(result.audit["rejected"], [])

    def test_similar_ordinary_words_and_other_people_stay_unchanged(self) -> None:
        roster = {"A": "WILL", "B": "TYLA"}
        for sentence in ("Well, I think you are right.",          # ordinary word, same phonetic key as WILL
                         "The tiler fixed the floor for you.",    # different onset, not a candidate
                         "I love Taylor Swift so much.",          # different leading vowel sound
                         "Have you seen Tyler Perry movies?"):    # another capitalized full name
            with self.subTest(sentence=sentence):
                result = lock.lock_participant_names(profile(sentence, "A", roster))
                self.assertEqual(texts(result.words), sentence)
        # The participant herself saying a close name is another person (no self third-person reference).
        result = lock.lock_participant_names(profile("I talked to Tyler yesterday.", "B", roster,
                                                     flags=[("Tyler", "tyla")]))
        self.assertEqual(texts(result.words), "I talked to Tyler yesterday.")

    def test_correction_keeps_ids_timing_and_speaker(self) -> None:
        data = profile("Hey Tylor, did you see that?", ["A", "A", "A", "A", "B", "B"], {"A": "KAI", "B": "TYLA"})
        data["words"][1]["timing_source"] = "whisper-1"
        before = json.loads(json.dumps(data["words"]))
        result = lock.lock_participant_names(data)
        self.assertEqual(result.words[1]["word"], "Tyla,")
        lock.verify_lock_invariants(before, result.words)
        for old, new in zip(before, result.words):
            for key in ("edited_start", "edited_end", "speaker_raw", "speaker_label", "timing_source"):
                self.assertEqual(old.get(key), new.get(key))
        self.assertEqual(len(before), len(result.words))

    def test_lexical_confidence_is_combined_with_agreement(self) -> None:
        data = profile("I sat with Tyler all day.", "A", {"A": "KAI", "B": "TYLA"})
        weak = {3: [lock.AltObservation("Tyla", "diverse_ear", prompted=False, probability=0.2)]}
        result = lock.lock_participant_names(data, alternatives=weak)
        row = next(r for r in result.audit["corrections"] + result.audit["rejected"] if r["index"] == 3)
        self.assertIn("asr_alternative_prompted", row["evidence"])     # low probability -> weak evidence only
        self.assertNotIn("asr_alternative_exact", row["evidence"])
        strong_other = {3: [lock.AltObservation("Tyson", "diverse_ear", prompted=False, probability=0.95),
                            lock.AltObservation("Tyler", "diverse_ear_2", prompted=False, probability=0.5),
                            lock.AltObservation("Tyler", "diverse_ear_3", prompted=False, probability=0.5)]}
        result = lock.lock_participant_names(data, alternatives=strong_other)
        self.assertEqual(result.words[3]["word"], "Tyler")
        self.assertEqual(result.audit["rejected"][0]["reason"], "independent_ear_heard_a_different_word")


class CaptionTruthStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="mimir v6 truth ş ")
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write(self, data: dict) -> Path:
        path = self.dir / "clip_01_speakers_v25.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_uncertain_high_impact_term_escalates_locally(self) -> None:
        # Third-person mention by the co-participant, no detector flag: text evidence is
        # not enough -> two small-window ears (unprompted diverse + verified-vocabulary).
        sentence = ("So yesterday after the long stream I finally talked to Tyler about the whole thing and "
                    "honestly it went better than I expected.")
        data = profile(sentence, "A", {"A": "KAI", "B": "TYLA"})
        path = self.write(data)
        index = sentence.split().index("Tyler")
        calls = []

        def ear(audio, start, end, *, prompted, vocabulary, context_before="", context_after=""):
            calls.append((start, end, prompted, tuple(vocabulary)))
            heard = "finally talked to Tyla about the" if not prompted else "finally talked to Tyla about the"
            return caption_truth.EarResult(heard, "fake-diverse" if not prompted else "fake-primary", prompted,
                                           (("ĠTy", 0.93), ("la", 0.91)) if not prompted else ())

        result = caption_truth.run_caption_truth(path, edited_clip_path=None, ear=ear,
                                                 audio_source=lambda: self.dir / "audio.wav")
        self.assertEqual(len(calls), 2)
        self.assertEqual({c[2] for c in calls}, {False, True})
        word = data["words"][index]
        for start, end, _prompted, vocabulary in calls:
            self.assertLessEqual(end - start, 3.5)                       # a local window, never the whole clip
            self.assertLessEqual(start, word["edited_start"])
            self.assertGreaterEqual(end, word["edited_end"])
            self.assertEqual(vocabulary, ("Kai", "Tyla"))                # small verified vocabulary only
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["words"][index]["word"], "Tyla")
        self.assertEqual(result.audit["escalations"], 1)
        # Cached: a re-run reuses the recorded ears (no new inference).
        calls.clear()
        caption_truth.run_caption_truth(path, edited_clip_path=None, ear=ear,
                                        audio_source=lambda: self.dir / "audio.wav")
        self.assertEqual(calls, [])

    def test_escalation_without_acoustic_support_fails_closed_and_marks_uncertain(self) -> None:
        sentence = "Yesterday I finally talked to Tyler about the whole stream thing."
        data = profile(sentence, "A", {"A": "KAI", "B": "TYLA"})
        path = self.write(data)
        index = sentence.split().index("Tyler")

        def ear(audio, start, end, *, prompted, vocabulary, context_before="", context_after=""):
            return caption_truth.EarResult("finally talked to Tyler about the", "fake", prompted)

        result = caption_truth.run_caption_truth(path, edited_clip_path=None, ear=ear,
                                                 audio_source=lambda: self.dir / "audio.wav")
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["words"][index]["word"], "Tyler")
        self.assertTrue(saved["words"][index][caption_truth.UNCERTAIN_FLAG])
        self.assertTrue(any(u["reason"] == "entity_spelling_unverified" for u in
                            json.loads(result.truth_path.read_text(encoding="utf-8"))["uncertain"]))

    def test_missing_ear_is_an_explicit_fallback(self) -> None:
        data = profile("Yesterday I finally talked to Tyler about the whole stream thing.", "A",
                       {"A": "KAI", "B": "TYLA"})
        result = caption_truth.run_caption_truth(self.write(data), edited_clip_path=None)
        self.assertTrue(result.fallbacks)
        self.assertEqual(result.fallbacks[0]["level"], "text_evidence_only")

    def test_frozen_truth_detects_any_later_mutation(self) -> None:
        data = profile("Brenner, would you pass me the charger?", "A", {"A": "JONAH", "B": "BRENNA"})
        path = self.write(data)
        result = caption_truth.run_caption_truth(path, edited_clip_path=None)
        self.assertTrue(caption_truth.verify_frozen_truth(path, result.truth_path)[0])
        for field, value in (("word", "Brenner,"), ("edited_start", 0.51), ("speaker_raw", "B")):
            mutated = json.loads(path.read_text(encoding="utf-8"))
            mutated["words"][0][field] = value
            other = self.dir / f"mutated_{field}.json"
            other.write_text(json.dumps(mutated), encoding="utf-8")
            with self.subTest(field=field):
                self.assertFalse(caption_truth.verify_frozen_truth(other, result.truth_path)[0])

    def test_timing_and_speaker_validation(self) -> None:
        data = profile("one two three four", "A", {"A": "KAI"})
        self.assertEqual(caption_truth.timing_issues(data), [])
        data["words"][2]["edited_start"] = 0.1
        self.assertTrue(caption_truth.timing_issues(data))
        data = profile("one two three four", "A", {"A": "KAI"})
        data["words"][1]["speaker_label"] = "SOMEONE"
        self.assertTrue(caption_truth.speaker_issues(data))

    def test_feature_off_name_lock_removes_v6_flags(self) -> None:
        data = profile("Yesterday I finally talked to Tyler about the whole stream thing.", "A",
                       {"A": "KAI", "B": "TYLA"})
        path = self.write(data)
        caption_truth.run_caption_truth(path, edited_clip_path=None)
        flagged = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(any(caption_truth.MARKER in w for w in flagged["words"]))
        lock.apply_to_profile_file(path)
        cleared = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(any(caption_truth.MARKER in w or caption_truth.UNCERTAIN_FLAG in w
                             for w in cleared["words"]))
        self.assertNotIn("caption_truth", cleared)


def unresolved(data: dict, votes: list[tuple[str, list[str]]], window: tuple[float, float] | None = None) -> dict:
    """Attach one caption-stage micro span the ears could not resolve (phrase -> sources)."""
    words = data["words"]
    w0, w1 = window if window is not None else (0.0, words[-1]["edited_end"] + 0.2)
    data.setdefault("caption_quality", {})["micro_accuracy"] = {"details": [{
        "span": [0, 1], "audio_window": [w0, w1], "resolved": False,
        "candidate_votes": [{"phrase": phrase, "votes": len(sources), "sources": sources} for phrase, sources in votes],
    }]}
    return data


def flagged(data: dict) -> list[str]:
    return [texts([data["words"][i]]) for ids, _reason in caption_truth.unresolved_span_word_ids(data, data["words"])
            for i in ids]


class UncertaintyGranularityTests(unittest.TestCase):
    """Unresolved micro spans: only words the independent ear majority did not hear are uncertain."""

    def test_majority_confirmed_words_stay_certain_including_numbers(self) -> None:
        sentence = "He got 12 wins in arena then he said please stop it now"
        data = unresolved(profile(sentence, "A", {}), [
            (sentence, ["asr_1", "asr_5"]),
            ("He got 12 wins in arena then he said please stop", ["asr_2"]),
            ("He got wins in arena", ["asr_4"]),
            ("", ["asr_3"])])
        self.assertEqual(flagged(data), ["it", "now"])

    def test_disputed_number_is_uncertain_and_never_rewritten(self) -> None:
        sentence = "He got 12 wins in arena"
        data = unresolved(profile(sentence, "A", {}), [
            ("He got 20 wins in arena", ["asr_1", "asr_2", "asr_4", "asr_5"]), (sentence, ["asr_3"])])
        self.assertEqual(flagged(data), ["12"])
        with tempfile.TemporaryDirectory(prefix="mimir v6 unc ş ") as tmp:
            path = Path(tmp) / "clip_01_speakers_v25.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            caption_truth.run_caption_truth(path, edited_clip_path=None)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(texts(saved["words"]), sentence)                       # fail closed: text kept
        self.assertEqual([w["word"] for w in saved["words"] if w.get(caption_truth.UNCERTAIN_FLAG)], ["12"])

    def test_an_ear_mapped_elsewhere_does_not_widen_the_dispute(self) -> None:
        sentence = "wait I flip the phone come on now all right do you want your phone back"
        data = unresolved(profile(sentence, "A", {}), [
            ("flip", ["asr_1", "asr_5"]), ("flipped", ["asr_2", "asr_3"]),
            ("all right do you want your", ["asr_4"])])
        self.assertEqual(flagged(data), ["flip"])

    def test_a_word_holding_several_spoken_tokens_is_judged_by_its_tokens(self) -> None:
        data = profile("not allowed back to the mall ever okay per management", "A", {})
        words = data["words"]
        merged = dict(words[5], word="mall ever", edited_end=words[6]["edited_end"])
        data["words"] = words[:5] + [merged] + words[7:]
        unresolved(data, [("not allowed back to the mall ever okay for management", ["asr_1", "asr_2", "asr_5"]),
                          ("yes sir ok", ["asr_4"]), ("", ["asr_3"])])
        self.assertEqual(flagged(data), ["per"])

    def test_a_prompted_only_majority_is_not_confirmation(self) -> None:
        sentence = "that was the big 5 moment"
        data = unresolved(profile(sentence, "A", {}), [
            (sentence, ["asr_3", "asr_4", "asr_5"]), ("that was the big file moment", ["asr_1", "asr_2"])])
        self.assertEqual(flagged(data), ["5"])

    def test_few_recorded_ears_keep_whole_region_flagging(self) -> None:
        sentence = "honestly that was crazy man"
        data = unresolved(profile(sentence, "A", {}), [("that was crazy", ["asr_1"]), ("that is lazy", ["asr_2"])])
        self.assertEqual(flagged(data), ["that", "was", "crazy"])


class PresentationAndGateTests(unittest.TestCase):
    def test_presentation_never_emphasizes_uncertain_truth(self) -> None:
        from ai.editor.pro_edit import caption_presentation as cp
        from ai.editor.pro_edit.schema import CaptionStyle

        data = profile("this is the part nobody expected", "A", {})
        data["words"][4][caption_truth.UNCERTAIN_FLAG] = True
        tokens = cp.presentation_tokens(data, 10.0)
        self.assertTrue(tokens[4].uncertain)
        report = cp.EmphasisReport()
        directive = cp.CaptionPresentationDirective("evt", 0.0, 10.0, CaptionStyle.IMPACT, (3, 4))
        chosen = cp.emphasis_candidates(tokens, [directive], [], report)
        self.assertIn(3, chosen)
        self.assertNotIn(4, chosen)
        self.assertEqual(report.degraded[0]["reason"], "uncertainty_mask")

    def test_verified_multi_word_name_stays_on_one_line_and_page(self) -> None:
        from ai.editor.pro_edit import caption_presentation as cp

        data = profile("and then right after that Brenna Holloway walked straight in with the whole crew", "A", {},
                       step=0.3)
        tokens = cp.presentation_tokens(data, 20.0)
        first = next(t.word_id for t in tokens if t.normalized == "brenna")
        second = next(t.word_id for t in tokens if t.normalized == "holloway")
        self.assertEqual(cp.bound_name_words(tokens, ["Brenna Holloway"]), frozenset({first}))
        for width, height in ((1920, 1080), (720, 1280), (480, 480)):
            with self.subTest(size=(width, height)):
                built = cp.build_presentation(profile=data, clip_timeline={}, plan=None, width=width,
                                              height=height, verified_names=["Brenna Holloway"])
                page = next(p for p in built.pages if first in {t.word_id for t in p.tokens})
                self.assertIn(second, {t.word_id for t in page.tokens})
                line = next(line for line in page.lines if first in line.word_ids)
                self.assertIn(second, line.word_ids)
                self.assertLessEqual(len(page.lines), 2)

    def test_gate_never_passes_a_run_with_a_fallback(self) -> None:
        run = v6_runtime.V6Run(enabled=True)
        run.fallback("face_tracking", "opencv not installed", "center_safe_framing")
        with tempfile.TemporaryDirectory() as tmp:
            gate = v6_runtime.final_quality_gate(
                run=run, final_output=Path(tmp) / "missing.mp4", profile_path=None, truth_path=None,
                burned_ass=None, prep=None, main_proof=None, final_proof=None, intro_proof=None,
                story_intact=(False, "not prepared"), intro_handoff=None, render_status="baseline")
        self.assertEqual(gate["status"], v6_runtime.GATE_FAILED)
        checks = {c["check"]: c for c in gate["checks"]}
        self.assertEqual(checks["no_silent_fallback"]["status"], "fail")
        self.assertIn("face_tracking->center_safe_framing", checks["no_silent_fallback"]["detail"])
        for name in ("output_file", "caption_truth_frozen", "camera_pixels_main", "v6_render_path"):
            self.assertEqual(checks[name]["status"], "fail", name)
        run.gate = gate
        self.assertTrue(any(line.startswith("[MIMIR_V6_FALLBACK] face_tracking") for line in run.console_lines()))

    def test_final_proof_is_never_a_pass_when_the_main_render_verified_nothing(self) -> None:
        self.assertIsNone(v6_runtime.untraceable_final_proof({"status": "passed"}))
        for main, expected in (({"status": "unavailable"}, "fail"), ({"status": "failed"}, "fail"), (None, "fail"),
                               ({"status": "inconclusive"}, "warn"), ({"status": "no_camera_ops"}, "pass")):
            with self.subTest(main=main):
                final = v6_runtime.untraceable_final_proof(main)
                self.assertEqual(v6_runtime.check_proof("camera_pixels_final", final, required=True)["status"],
                                 expected)

    def test_blocked_opencv_is_recorded_first_with_its_remedy(self) -> None:
        from types import SimpleNamespace
        from unittest import mock

        from ai import shorts_pipeline

        reason = ("opencv-python-headless 9.9 is installed but Windows Smart App Control (on) refused to load its "
                  "native module; python -m pip install -r requirements.txt")
        prep = SimpleNamespace(status="ready", reason="ok", direction={"status": "directed"},
                               tracking={"status": "unavailable", "reason": reason}, captions={"status": "presentation",
                               "level": "v5_placement", "activity": "failed", "background": "failed"}, energy={})
        run = v6_runtime.V6Run(enabled=True)
        with mock.patch("ai.editor.pro_edit.vision.cv_runtime.opencv_status",
                        return_value={"status": "blocked_by_windows_app_control", "reason": reason}):
            shorts_pipeline._record_pro_edit_v6(run, prep)
        self.assertEqual(run.fallbacks[0]["subsystem"], "vision_runtime")
        self.assertIn("requirements.txt", run.fallbacks[0]["reason"])
        self.assertTrue(all("see vision_runtime" in r["reason"] for r in run.fallbacks
                            if r["subsystem"].startswith("caption_evidence")))
        import contextlib
        import io

        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            shorts_pipeline._print_friendly_result({"final_output": "x.mp4", "v6": {
                "status": "failed", "failed_checks": ["no_silent_fallback"],
                "fallback_rows": [{"subsystem": r["subsystem"], "level": r["level"], "reason": r["reason"]}
                                  for r in run.fallbacks]}})
        self.assertIn("neden: opencv-python-headless 9.9 is installed but Windows Smart App Control", printed.getvalue())

    def test_gate_detects_replaced_spelling_in_burned_captions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = profile("Brenner, would you pass me the charger?", "A", {"A": "JONAH", "B": "BRENNA"})
            path = root / "clip_01_speakers_v25.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            truth = caption_truth.run_caption_truth(path, edited_clip_path=None).truth_path
            header = "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            good = root / "good.ass"
            good.write_text(header + "Dialogue: 0,0:00:00.00,0:00:01.00,Main,,0,0,0,,{\\an2}JONAH: {\\c&H00FFFF&}"
                            "Brenna,{\\r} would you{\\p1}m 0 0 l 10 0 10 10{\\p0} pass me the charger?\n",
                            encoding="utf-8")
            bad = root / "bad.ass"
            bad.write_text(header + "Dialogue: 0,0:00:00.00,0:00:01.00,Main,,0,0,0,,Brenner, would you\n",
                           encoding="utf-8")
            self.assertEqual(v6_runtime.check_burned_names(truth, good, ["JONAH"])["status"], "pass")
            failed = v6_runtime.check_burned_names(truth, bad, ["JONAH"])
            self.assertEqual(failed["status"], "fail")
            self.assertIn("Brenner,", failed["contradictions"])


if __name__ == "__main__":
    unittest.main()
