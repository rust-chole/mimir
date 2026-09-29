"""Cross-cutting production contracts of the single MIMIR path (fast, no media)."""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import pkgutil
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import ai
from ai.editor import human_review, intro_analyzer, intro_bounds, intro_renderer, meme_renderer, teaser_analyzer

ROOT = Path(__file__).resolve().parent.parent


def _schemas():
    for info in pkgutil.walk_packages(ai.__path__, "ai."):
        module = importlib.import_module(info.name)
        for name in dir(module):
            value = getattr(module, name)
            if name.endswith("SCHEMA") and isinstance(value, dict):
                yield f"{info.name}.{name}", value


def _objects(node, path=""):
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            yield path, node
        for key, value in node.items():
            yield from _objects(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _objects(value, f"{path}[{index}]")


class StrictSchemaTests(unittest.TestCase):
    def test_every_strict_schema_requires_every_property(self) -> None:
        """OpenAI strict structured outputs reject a schema whose object does not require
        all its properties; such a stage then silently lives on its fallback."""
        seen = 0
        for name, schema in _schemas():
            for path, obj in _objects(schema, name):
                seen += 1
                self.assertEqual(sorted(obj["properties"]), sorted(obj.get("required", [])), path)
                self.assertIs(obj.get("additionalProperties"), False, path)
        self.assertGreater(seen, 10)


class HardcodeAuditTests(unittest.TestCase):
    # Identifiers of development videos, people and phrases. Tests may use them as
    # fixtures; production code may never special-case them.
    FORBIDDEN = ("kai16", "kaityla", "kai tyla", "tyla", "speedates", "speedcuce", "speedtakla", "shawty", "shorty",
                 "yusuf")

    def test_production_code_has_no_test_video_specifics(self) -> None:
        offenders = []
        for path in (ROOT / "ai").rglob("*.py"):
            text = path.read_text(encoding="utf-8-sig").casefold()
            for word in self.FORBIDDEN:
                for match in re.finditer(rf"\b{re.escape(word)}\b", text):
                    line = text[:match.start()].count("\n") + 1
                    offenders.append(f"{path.relative_to(ROOT)}:{line}: {word}")
        self.assertEqual(offenders, [])


class IntroContractTests(unittest.TestCase):
    def test_restart_never_trims_protected_story(self) -> None:
        with mock.patch.object(intro_renderer, "get_first_caption_start", return_value=6.0):
            free, _ = intro_renderer.calculate_main_restart_seconds(caption_path="x.ass", main_duration=30.0)
            kept, _ = intro_renderer.calculate_main_restart_seconds(caption_path="x.ass", main_duration=30.0,
                                                                    protected_ranges=[(2.0, 4.0)])
        self.assertAlmostEqual(free, 6.0 - intro_renderer.POST_INTRO_SPEECH_PREROLL_SECONDS)
        self.assertLessEqual(kept, 2.0 - intro_renderer.POST_INTRO_PROTECTED_PREROLL_SECONDS + 1e-9)

    def test_restart_is_a_hard_cut_on_the_frame_grid(self) -> None:
        self.assertEqual(intro_renderer.calculate_transition_duration(2.0, 20.0), 0.0)
        self.assertAlmostEqual(intro_renderer.snap_to_frame(1.2345, 30.0), 37 / 30.0, places=6)

    def test_protected_ranges_map_through_the_rendered_cuts(self) -> None:
        clip = {"source": {"duration": 30.0}, "cut_ranges": [{"start": 1.0, "end": 3.0}],
                "protected_ranges": [{"start": 5.0, "end": 7.0}]}
        (start, end), = intro_renderer.protected_edited_ranges(clip)
        self.assertAlmostEqual(start, 3.0 - 0.015, places=2)
        self.assertAlmostEqual(end, 5.0 + 0.015, places=2)

    def test_final_clock_maps_story_time_and_rejects_trimmed_time(self) -> None:
        doc = {"intro": {"duration": 2.0}, "restart": {"main_restart_paced": 1.5}}
        self.assertAlmostEqual(intro_renderer.paced_to_final(doc, 4.0), 4.5)
        self.assertIsNone(intro_renderer.paced_to_final(doc, 1.0))

    def test_no_headline_intro_is_accepted_by_the_renderer(self) -> None:
        record = intro_analyzer._build_no_intro_result(
            clip={"title": "t"}, clip_index=1, creator_name=None, score=6.1, reason="judge rejected",
            gemini_support_used=False, candidates=[], candidate_scores=[], repair_rounds_used=0)
        self.assertTrue(record["recommended"])
        self.assertEqual(record["intro_text"], "")
        self.assertEqual(intro_renderer.intro_quality_status(record)[2], "no_headline")
        rejected_text = {**record, "intro_text": "SOMETHING", "quality_gate": {"accepted": True, "no_headline": True}}
        self.assertFalse(intro_renderer.intro_quality_status(rejected_text)[0])   # a 6.1/10 headline never renders


class HeadlineGroundingTests(unittest.TestCase):
    EVIDENCE = "[0.0-2.0] so Mira bet 500 subs on this. [2.1-3.0] Oh my God he actually fell. Chat, look."

    def test_numbers_and_names_must_be_evidenced_and_verified(self) -> None:
        check = intro_analyzer.headline_grounding_problems
        self.assertEqual(check("HE BET 500 SUBS ON THIS", evidence_text=self.EVIDENCE), [])
        self.assertTrue(check("HE BET 5000 SUBS ON THIS", evidence_text=self.EVIDENCE))
        self.assertTrue(check("MIRA SET HIM UP", evidence_text=self.EVIDENCE))
        self.assertEqual(check("MIRA SET HIM UP", evidence_text=self.EVIDENCE, verified_names=["Mira"]), [])
        self.assertEqual(check("OH MY GOD HE FELL", evidence_text=self.EVIDENCE), [])

    def test_validator_discards_an_ungrounded_candidate(self) -> None:
        ok, problems, _ = intro_analyzer.validate_intro_choice(
            {"intro_text": "HE LOST 2000 SUBS", "curiosity_target": "why the bet went wrong so fast",
             "uses_specific_name": False, "specific_name": ""},
            teaser_text="", creator_name=None, evidence_text=self.EVIDENCE)
        self.assertFalse(ok)
        self.assertTrue(any("number" in p for p in problems))


class TeaserEvidenceTests(unittest.TestCase):
    def test_spoken_choice_is_bounded_by_the_event_not_a_bucket(self) -> None:
        clip = {"clip_index": 1, "title": "t", "edited": {"estimated_duration": 20.0},
                "source": {"absolute_start": 100.0, "absolute_end": 120.0, "duration": 20.0}, "cut_ranges": []}
        texts = "so then he said what the hell is that and left".split()
        words = [{"id": i, "word": w, "edited_start": 8.0 + i * 0.3, "edited_end": 8.0 + i * 0.3 + 0.22,
                  "relative_start": 8.0 + i * 0.3, "relative_end": 8.22 + i * 0.3} for i, w in enumerate(texts)]
        choice = {"recommended": True, "score": 8.0, "teaser_type": "reaction", "selection_mode": "spoken_phrase",
                  "peak_id": -1, "peak_alignment_score": 5.0, "start_word_id": 4, "end_word_id": 8,
                  "reason": "", "viewer_question": "", "spoiler_risk": "low"}
        result = teaser_analyzer.build_teaser_result(clip, words, choice, None)
        edited = result["edited"]
        self.assertEqual(edited["duration_policy"], "event_evidence")
        self.assertNotIn("adaptive_minimum_duration", edited)
        for word in words:
            self.assertFalse(word["edited_start"] < edited["teaser_start"] < word["edited_end"])
            self.assertFalse(word["edited_start"] < edited["teaser_end"] < word["edited_end"])
        self.assertGreaterEqual(edited["duration"], intro_bounds.MIN_INTRO_S - 1e-6)


class MemeFinalClockTests(unittest.TestCase):
    def event(self, main_start: float, doc: dict):
        clip = {"clip_index": 1}
        discovery = {"slot": {"timing": {"final_start": 999.0, "main_edited_start": main_start, "max_duration": 0.8}},
                     "candidate": {"media_type": "audio", "name": "boom", "provider": "web"}}
        with mock.patch.object(meme_renderer, "choose_single_best_discovery", return_value=discovery), \
                mock.patch.object(meme_renderer, "resolve_local_asset", return_value=Path("boom.wav")), \
                mock.patch.object(meme_renderer, "get_media_info",
                                  return_value={"has_audio": True, "has_video": False, "duration": 0.8}):
            return meme_renderer.build_single_event(clip, 40.0, doc)

    def test_meme_lands_on_the_final_clock_not_teaser_plus_main(self) -> None:
        doc = {"intro": {"duration": 2.0}, "restart": {"main_restart_paced": 3.0}}
        event, _ = self.event(10.0, doc)
        self.assertAlmostEqual(event["start"], 2.0 + (10.0 - 3.0))

    def test_meme_is_never_placed_in_trimmed_story_or_the_cold_open(self) -> None:
        doc = {"intro": {"duration": 2.0}, "restart": {"main_restart_paced": 3.0}}
        self.assertIsNone(self.event(2.0, doc)[0])           # cut away by the restart
        self.assertIsNone(self.event(3.2, doc)[0])           # right on the restart (opening protection)


class EffectsChainVersionTests(unittest.TestCase):
    def test_every_meme_stage_accepts_what_the_previous_stage_writes(self) -> None:
        """The analyzer -> discovery -> renderer chain once rejected its own outputs (hard-coded
        versions), so no meme/SFX ever reached a short."""
        from ai.editor import meme_analyzer, meme_discovery

        meme_discovery._validate_slot_package({"version": meme_analyzer.MEME_ANALYZER_VERSION, "clips": []})
        meme_renderer.validate_slot_package({"version": meme_analyzer.MEME_ANALYZER_VERSION,
                                             "inputs": {"timeline": "t.json"}})
        meme_renderer.validate_discovery_package({"version": meme_discovery.DISCOVERY_VERSION, "clips": [],
                                                  "inputs": {"meme_slots": "s.json"}})
        source = (ROOT / "ai" / "shorts_pipeline.py").read_text(encoding="utf-8")
        self.assertNotRegex(source, r'_contains_clip\(meme_slot_path, "clips", selected_clip_index, \d')


class ModelRoutingTests(unittest.TestCase):
    STRONG = {"caption_judge", "edit_director"}

    def test_astra_is_spent_on_captions_and_editing_only(self) -> None:
        """Defaults (no .env, no MIMIR_* overrides): the strongest model decides caption words and
        editorial intent; scouting, drafting, classification and support stay on cheaper tiers."""
        code = ("import json, dotenv; dotenv.load_dotenv = lambda *a, **k: False\n"
                "from ai import model_config as m\n"
                "print(json.dumps({'plan': m.model_plan(), 'astra': m.DEFAULT_ASTRA_MODEL}))")
        env = {k: v for k, v in os.environ.items() if not k.startswith("MIMIR_")}
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True,
                             check=True).stdout
        data = json.loads(out)
        plan, astra = data["plan"], data["astra"]
        self.assertEqual(astra, "gpt-6-astra")
        for role in self.STRONG:
            self.assertEqual(plan[role]["model"], astra, role)
        for role, row in plan.items():
            if role not in self.STRONG:
                self.assertNotEqual(row["model"], astra, f"{role} must not spend Astra budget")

    def test_the_edit_director_runs_on_the_director_model(self) -> None:
        from ai import model_config
        from ai.editor.pro_edit.config import load_config

        config = load_config(environ={})
        self.assertEqual((config.model, config.reasoning_effort),
                         (model_config.EDIT_DIRECTOR_MODEL, model_config.EDIT_DIRECTOR_REASONING_EFFORT))


class EditDirectorIntentTests(unittest.TestCase):
    GEOMETRY = re.compile(r"crop|zoom|scale|coord|pixel|^x\d?$|^y\d?$|^x0|^y0|^x1|^y1|width|height|position|box|"
                          r"rect|offset|center_x|center_y|^cx$|^cy$", re.I)

    def test_the_director_schema_can_only_express_intent(self) -> None:
        from ai.editor.pro_edit.schema import planner_json_schema

        names = []
        for _path, obj in _objects(planner_json_schema("pro_stream_v1")):
            names.extend(obj["properties"])
        self.assertTrue(names)
        self.assertEqual([n for n in names if self.GEOMETRY.search(n)], [])

    def test_scene_evidence_reaches_the_director_as_words(self) -> None:
        import dataclasses
        from types import SimpleNamespace

        import pro_edit_fixtures as fx
        from ai.editor.pro_edit import request as planner_request
        from ai.editor.pro_edit import stage
        from ai.editor.pro_edit.style import get_style_pack
        from ai.editor.pro_edit.subjects import SubjectSample, SubjectTrack

        workspace = fx.Workspace()
        self.addCleanup(workspace.cleanup)
        context = fx.make_context(workspace)
        face = SubjectTrack("face_1", "face", (SubjectSample(1.0, 0.8, 0.2, 0.1, 0.15, 0.9),), speaker_id="A")
        context = dataclasses.replace(context, subject_tracks=(face,), scene_changes=(3.0, 9.5))
        request = SimpleNamespace(config=SimpleNamespace(caption_activity=False, caption_ui=False,
                                                         caption_layout=False), force=False)
        evidence = stage.director_evidence(request, SimpleNamespace(artifacts=object()), context, fx.media())
        self.assertEqual(evidence["faces"], [{"subject_id": "face_1", "kind": "face", "where": "upper-right",
                                              "size": "small", "speaker_id": "A"}])
        self.assertIn("layout", evidence)
        payload = planner_request.build_payload(dataclasses.replace(context, director_evidence=evidence),
                                                get_style_pack("pro_stream_v1"))
        self.assertEqual(payload["shot_cuts_s"], [3.0, 9.5])
        self.assertEqual(payload["scene"]["faces"][0]["where"], "upper-right")
        for key in ("speaker_names", "motion_events", "story_spans", "caption_words", "subjects"):
            self.assertIn(key, payload)
        # Evidence carries region words, never boxes the director could echo back.
        def keys(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    yield k
                    yield from keys(v)
            elif isinstance(node, list):
                for v in node:
                    yield from keys(v)
        self.assertEqual([k for k in keys(payload["scene"]) if self.GEOMETRY.search(k)], [])


class HumanAcceptanceTests(unittest.TestCase):
    def test_no_model_reviews_the_final_short(self) -> None:
        """The acceptance layer is a human; no AI reviewer or reviewer-triggered repair exists."""
        self.assertIsNone(importlib.util.find_spec("ai.editor.final_review"))
        offenders = [str(path.relative_to(ROOT)) for path in (ROOT / "ai").rglob("*.py")
                     if re.search(r"final_review|MIMIR_FINAL_REVIEW|render_static_camera",
                                  path.read_text(encoding="utf-8-sig"))]
        self.assertEqual(offenders, [])

    def test_review_packet_points_at_uncertain_and_changed_words_on_the_final_clock(self) -> None:
        doc = {"intro": {"duration": 2.0, "paced": [10.0, 12.0]}, "restart": {"main_restart_paced": 1.0}}
        truth = {
            "words": [[0, "gone", 0.4, 0.6, "A", "S1"], [1, "Marra", 5.0, 5.3, "A", "S1"],
                      [2, "said", 5.4, 5.6, "A", "S1"], [3, "stop", 8.0, 8.3, "B", "Mira"]],
            "uncertain": [{"word_ids": [1], "reason": "asr_ears_disagree"},
                          {"word_ids": [0], "reason": "asr_ears_disagree"}],     # trimmed by the restart
            "lexical_decisions": [{"changed": True, "paced_start": 8.0, "from": "top", "to": "stop",
                                   "decided_by": "caption judge", "confidence": 0.9},
                                  {"changed": False, "paced_start": 5.4, "from": "said", "to": "said"},
                                  {"changed": False, "paced_start": 8.0, "from": "stop", "to": "stop",
                                   "decided_by": "strict_vote", "guard": "changed word(s) no ear heard: top"}],
            "clock_health": {"status": "recovered", "attempts": [
                {"method": "overlapping_chunks_exact_final_audio", "selected": True,
                 "replaced_regions": [{"region": [11.0, 20.0], "reason": "collapsed"}]}]},
        }
        packet = human_review.build_review_packet(
            published=Path("x_short.mp4"), source=Path("x.mp4"), status="published",
            qc_rows=[{"check": "a", "status": "pass"}], timeline_doc=doc, truth=truth, headline="",
            degradations=["face_tracking -> stable_wide: no face"])
        focus = [(item["kind"], item["final_s"]) for item in packet["focus"]]
        self.assertEqual(focus, [("caption_uncertain", 6.0), ("caption_changed_by_judge", 9.0),
                                 ("judge_answer_set_aside", 9.0), ("clock_recovered", 12.0)])
        self.assertEqual(packet["speakers"], {"Mira": 1, "S1": 3})
        text = human_review.render_markdown(packet)
        self.assertIn('0:06.0 caption "Marra"', text)
        self.assertIn('"top" -> "stop"', text)
        self.assertIn("lacked acoustic support", text)
        self.assertIn("caption timing re-measured from here", text)
        self.assertIn("none (no grounded headline was approved)", text)
        self.assertIn("face_tracking -> stable_wide", text)

if __name__ == "__main__":
    unittest.main()
