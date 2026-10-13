"""Caption evaluation: normalization-aware scoring, case preparation, cached runs, comparison.

Hermetic: no model is called (systems are injected); FFmpeg makes real media.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai.caption_eval import cases as case_store
from ai.caption_eval import cli, metrics, runner, systems

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def hyp(text: str, *, start: float = 0.0, step: float = 0.4, flagged: tuple[str, ...] = ()) -> list[dict]:
    return [{"word": w, "start": round(start + i * step, 3), "end": round(start + i * step + 0.3, 3),
             "status": "uncertain" if w in flagged else "agreed"} for i, w in enumerate(text.split())]


def score(gold: str, words: list[dict], **kwargs) -> metrics.CaseScore:
    return metrics.score_case("c", metrics.gold_tokens(gold, names=kwargs.pop("names", ()),
                                                       times=kwargs.pop("times", None)),
                              metrics.hypothesis_tokens(words), **kwargs)


class ScoringTests(unittest.TestCase):
    def test_harmless_differences_are_not_errors(self) -> None:
        result = score("Uh, I'm gonna win twenty-one games, OK?", hyp("i am going to win 21 games okay"))
        self.assertEqual((result.sub, result.dele, result.ins), (0, 0, 0))
        self.assertEqual(result.wer, 0.0)

    def test_the_edit_alignment_is_minimal(self) -> None:
        ops = metrics.align(list("abcdef"), list("abxdf"))
        self.assertEqual([op for op, _i, _j in ops], ["equal", "equal", "sub", "equal", "del", "equal"])

    def test_inaudible_gold_matches_anything_at_no_cost(self) -> None:
        for heard in ("we won the [x]", "we won bro the", "we won the"):
            with self.subTest(heard=heard):
                result = score("we won [inaudible] the", hyp(heard.replace("[x]", "")))
                self.assertEqual(result.errors_total, 0)
        self.assertEqual(metrics.gold_tokens("a [two words] b")[1].display, "[...]")

    def test_critical_words_are_names_numbers_and_negations(self) -> None:
        gold = "I told you, Tyla, we are not doing 12 runs."
        result = score(gold, hyp("I told you Tyler we are doing 20 runs"))
        self.assertEqual(result.critical["name"], [1, 1])
        self.assertEqual(result.critical["negation"], [1, 1])
        self.assertEqual(result.critical["number"], [1, 1])
        # Sentence-initial capitals, "I", and short all-caps interjections are not names.
        kinds = {t.display: t.kind for t in metrics.gold_tokens("Wait. I said OK, Chat is here and NASA called.")}
        self.assertEqual((kinds["Wait."], kinds["I"], kinds["OK,"], kinds["Chat"], kinds["NASA"]),
                         ("", "", "", "name", "name"))
        self.assertEqual(metrics.gold_tokens("they don't care")[2].kind, "negation")   # do, NOT
        inserted = score("we did it", hyp("we did not do it"))
        self.assertEqual(inserted.critical_insertions, 1)

    def test_dropped_and_invented_phrases(self) -> None:
        result = score("one two three four five six seven", hyp("one two six seven"))
        self.assertEqual([p["text"] for p in result.dropped_phrases], ["three four five"])
        result = score("we won", hyp("thanks for watching we won"))
        self.assertEqual([p["text"] for p in result.invented_phrases], ["thanks for watching"])

    def test_onset_timing_uses_correct_words_only(self) -> None:
        times = [(0.0, 0.3), (0.5, 0.8), (1.0, 1.3)]
        result = score("a b c", hyp("a x c", start=0.05, step=0.5), times=times, timing_reference="human")
        self.assertEqual([round(x, 3) for x in result.onset_errors], [0.05, 0.05])
        aggregate = metrics.aggregate([result])
        self.assertEqual(aggregate["timing"]["within_100ms"], 1.0)
        self.assertEqual(aggregate["timing"]["references"], ["human"])

    def test_uncertain_and_coverage_flags_are_measured_against_real_errors(self) -> None:
        result = score("we were never there", hyp("we were ever there", flagged=("ever", "there")),
                       coverage={"uncovered_speech": [{"start": 0.7, "end": 1.0}, {"start": 5.0, "end": 6.0}]})
        self.assertEqual((result.flagged, result.flagged_wrong), (2, 1))
        self.assertEqual((result.wrong_hyp, result.wrong_hyp_flagged), (1, 1))
        self.assertEqual((result.coverage_flags, result.coverage_flags_on_errors), (2, 1))
        aggregate = metrics.aggregate([result])
        self.assertEqual(aggregate["uncertain_flag_precision"], 0.5)
        self.assertEqual(aggregate["coverage_flag_precision"], 0.5)

    def test_aggregate_is_micro_summed(self) -> None:
        a = score("one two three four", hyp("one two three four"))
        b = score("five six", hyp("five"))
        self.assertAlmostEqual(metrics.aggregate([a, b])["wer"], 1 / 6, places=4)
        self.assertAlmostEqual(metrics.aggregate([a, b])["case_wer_mean"], 0.25, places=4)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
class CasePreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="mimir eval ş ")
        self.dir = Path(self._tmp.name)
        self.root = self.dir / "eval"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def media(self, path: Path, seconds: float) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=black:s=64x64:d={seconds}",
                        "-f", "lavfi", "-i", f"sine=frequency=220:sample_rate=48000:duration={seconds}", "-shortest",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)], check=True)
        return path

    def profile(self, words: list[tuple[str, float, float]]) -> Path:
        path = self.dir / "profile.json"
        path.write_text(json.dumps({
            "words": [{"word": w, "edited_start": s, "edited_end": e, "lexical_status": "agreed"} for w, s, e in words],
            "caption_quality": {"settings": {"language": "en"}, "lexical": {"verified_terms": ["Tyla"]}}}),
            encoding="utf-8")
        return path

    def test_a_clip_case_has_audio_a_draft_and_a_todo_gold_that_is_never_overwritten(self) -> None:
        clip = self.media(self.dir / "edited" / "My VOD" / "clip_01_win_edited.mp4", 3.0)
        case = case_store.prepare_from_clip(clip, root=self.root, profile=self.profile(
            [("hey", 0.2, 0.4), ("Tyla", 0.5, 0.9)]))
        self.assertAlmostEqual(case.duration, 3.0, delta=0.1)
        self.assertEqual((case.language, case.names), ("en", ["Tyla"]))
        self.assertEqual(case.gold_status(), "todo")
        self.assertIn("hey Tyla", case.gold_text())
        self.assertEqual([w["word"] for w in case.published_words()], ["hey", "Tyla"])
        case.gold_path.write_text("hey Tyla!\n", encoding="utf-8")
        self.assertEqual(case.gold_status(), "ready")
        with self.assertRaises(FileExistsError):
            case_store.prepare_from_clip(clip, root=self.root)
        case_store.prepare_from_clip(clip, root=self.root, profile=self.dir / "profile.json", force=True)
        self.assertEqual(case.gold_text(), "hey Tyla!\n")                   # a human's gold survives --force

    def test_a_published_short_maps_its_story_through_the_final_timeline(self) -> None:
        published = self.media(self.dir / "final" / "My VOD_short.mp4", 5.0)
        edited = self.dir / "edited" / "My VOD" / "clip_02_win_edited.mp4"
        timeline = self.dir / "tmp" / "intro.timeline.json"
        timeline.parent.mkdir(parents=True)
        timeline.write_text(json.dumps({"kind": "mimir_final_timeline", "intro": {"duration": 1.5,
                                        "source_path": str(edited)}, "restart": {"main_restart_paced": 2.0}}),
                            encoding="utf-8")
        profile = self.profile([("before", 1.0, 1.4), ("go", 2.5, 2.8), ("now", 3.0, 3.3)])
        case = case_store.prepare_from_final(published, root=self.root, timeline=timeline, profile=profile)
        self.assertAlmostEqual(case.duration, 3.5, delta=0.1)                 # published minus the cold open
        rows = case.published_words()
        self.assertEqual([(w["word"], w["start"]) for w in rows], [("go", 0.5), ("now", 1.0)])
        self.assertEqual(case.meta["source"]["restart_paced"], 2.0)
        with mock.patch.object(case_store, "PROJECT_ROOT", self.dir):
            self.assertEqual(case_store.discover_timeline(published, self.dir), timeline)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="mimir eval runs ")
        self.root = Path(self._tmp.name)
        clip = self.root / "src" / "clip_01_a_edited.mp4"
        clip.parent.mkdir(parents=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "sine=frequency=220:sample_rate=48000:duration=2", "-c:a", "pcm_s16le", str(clip)], check=True)
        for case_id, gold in (("one", "we are not done"), ("two", "twelve kills Tyla")):
            case = case_store.prepare_from_clip(clip, root=self.root, case_id=case_id)
            case.gold_path.write_text(gold + "\n", encoding="utf-8")
        case_store.prepare_from_clip(clip, root=self.root, case_id="draft_only")      # gold still TODO

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_runs_are_cached_scored_and_compared(self) -> None:
        calls: list[tuple[str, str, str]] = []
        heard = {"one": "we are done", "two": "12 kills Tyler", "draft_only": "x"}

        def fake(case, system):
            calls.append((case.case_id, system, os.environ.get("MIMIR_TEST_SWITCH", "")))
            if case.case_id == "draft_only" and system == "stack":
                raise RuntimeError("provider down")
            return {"words": [{"word": w} for w in heard[case.case_id].split()], "meta": {}, "elapsed_s": 0.0}

        name, results = runner.run_system("stack", overrides={"MIMIR_TEST_SWITCH": "on"}, root=self.root,
                                          runner=fake, log=lambda m: None)
        self.assertEqual(sorted(r["status"] for r in results), ["error", "ok", "ok"])   # one failure never stops a run
        self.assertEqual({c[2] for c in calls}, {"on"})
        self.assertNotIn("MIMIR_TEST_SWITCH", os.environ)                                # restored afterwards
        runner.run_system("stack", overrides={"MIMIR_TEST_SWITCH": "on"}, root=self.root, runner=fake, log=lambda m: None)
        self.assertEqual(len(calls), 4)                                  # only the failed case ran again
        result = runner.score_run(name, root=self.root)
        aggregate = result["aggregate"]
        self.assertEqual(aggregate["cases"], 2)
        self.assertEqual((aggregate["del"], aggregate["sub"]), (1, 1))  # "not" lost; "Tyla" -> "Tyler"
        self.assertEqual(aggregate["critical"]["negation"]["errors"], 1)
        self.assertEqual({r["case_id"] for r in result["skipped"]}, {"draft_only"})
        report = (runner.run_dir(name, self.root) / "report.md").read_text(encoding="utf-8")
        self.assertIn("names wrong or lost | 100.0% (1/1)", report)
        self.assertIn('"not" missing [negation]', report)

        runner.run_system("ear:openai:fake", root=self.root, runner=lambda case, system: {
            "words": [{"word": w} for w in heard[case.case_id].replace("done", "not done").split()]},
            log=lambda m: None)
        text = runner.compare([name, "ear-openai-fake"], root=self.root)
        self.assertIn("| WER (lower is better) |", text)
        self.assertIn("(-25.0)", text)
        with self.assertRaises(ValueError):
            runner.run_system("stack", name=name, root=self.root, runner=fake)     # a name holds one configuration

    def test_a_changed_setting_reruns_instead_of_reusing_a_stale_hypothesis(self) -> None:
        calls: list[str] = []

        def fake(case, system):
            calls.append(case.case_id)
            return {"words": [{"word": "x"}], "meta": {}}

        runner.run_system("stack", name="s", root=self.root, runner=fake, log=lambda m: None)
        runner.run_system("stack", name="s", root=self.root, runner=fake, log=lambda m: None)
        self.assertEqual(len(calls), 3)                                   # cached the second time
        with mock.patch.dict(os.environ, {"MIMIR_CAPTION_COVERAGE_CHECK": "off"}):
            runner.run_system("stack", name="s", root=self.root, runner=fake, log=lambda m: None)
        self.assertEqual(len(calls), 6)                                   # .env changed -> every case re-runs
        self.assertEqual(len(runner.score_run("s", root=self.root)["configurations"]), 1)

    def test_systems_and_overrides_are_validated(self) -> None:
        for system in ("published", "stack", "ear:qwen_omni", "ear:qwen_omni:names", "ear:qwen3_asr",
                       "ear:qwen3_asr_flash:names", "ear:openai:gpt-4o-transcribe"):
            self.assertTrue(systems.known(system), system)
        for system in ("ear", "ear:openai", "ear:qwen3_asr:other", "whisper"):
            self.assertFalse(systems.known(system), system)
        self.assertEqual(runner.env_from_pairs(["MIMIR_CAPTION_KEYWORDS=chat,dono"]),
                         {"MIMIR_CAPTION_KEYWORDS": "chat,dono"})
        for bad in (["DASHSCOPE_API_KEY=x"], ["OPENAI_API_KEY=x"], ["lower=x"], ["NOEQUALS"]):
            with self.assertRaises(ValueError):
                runner.env_from_pairs(bad)

    def test_the_gold_editor_saves_text_names_and_the_finished_mark(self) -> None:
        from ai.caption_eval import editor

        case = case_store.list_cases(self.root, ["draft_only"])[0]
        status = editor.save_gold(case, "# a note\r\nhey  Tyla\r\n", names="Tyla, KAI ,", done=False)
        self.assertEqual(status, "todo")                                     # not finished: still not scored
        self.assertEqual(metrics.gold_body(case.gold_text()).strip(), "hey  Tyla")
        self.assertEqual(editor.save_gold(case, "hey Tyla", names="Tyla, KAI", done=True), "ready")
        self.assertNotIn(case_store.GOLD_TODO, case.gold_text())
        self.assertEqual(case_store.list_cases(self.root, ["draft_only"])[0].names, ["Tyla", "KAI"])

    def test_the_command_line_reports_status(self) -> None:
        with mock.patch("builtins.print") as printed:
            self.assertEqual(cli.main(["--root", str(self.root), "status"]), 0)
        output = "\n".join(str(c.args[0]) for c in printed.call_args_list)
        self.assertIn("2/3 cases have a finished gold transcript", output)


if __name__ == "__main__":
    unittest.main()
