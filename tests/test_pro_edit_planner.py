from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import pro_edit_fixtures as fx
from ai.editor.pro_edit.errors import PlannerOutputError, PlannerUnavailableError
from ai.editor.pro_edit.planner import (
    ProviderEditPlanner,
    RuleBasedEditPlanner,
    StaticEditPlanner,
    parse_planner_json,
)
from ai.editor.pro_edit.providers import OpenAIResponsesProvider, ReplayProvider


def ModelEditPlanner(model, effort, client):  # noqa: N802 - live-provider path with an injected client
    return ProviderEditPlanner(OpenAIResponsesProvider(model, effort, client=client))
from ai.editor.pro_edit.schema import MotionPreset, StoryRole
from ai.editor.pro_edit.style import PRO_STREAM_V1
from ai.editor.pro_edit.validator import PlanStatus


class FakeResponses:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(output_text=answer, status="completed", output=[], id="resp_test", usage=None)


class FakeClient:
    def __init__(self, answers):
        self.responses = FakeResponses(answers)


def good_plan_text() -> str:
    return json.dumps(fx.plan(fx.event("e1", 13.0, 14.2), fx.event("e2", 16.3, 17.6, role="reaction",
                                                                      camera="reaction_close")))


class ParseTests(unittest.TestCase):
    def test_strict_json(self) -> None:
        self.assertEqual(parse_planner_json('{"a": 1}'), {"a": 1})
        self.assertEqual(parse_planner_json('```json\n{"a": 1}\n```'), {"a": 1})
        for bad in ("", "   ", "not json", '{"a": 1', '{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}',
                    '{"a": 1} {"b": 2}', "[1, 2]", 'Sure! {"a": 1}'):
            with self.subTest(bad=bad):
                with self.assertRaises(PlannerOutputError):
                    parse_planner_json(bad)


class PlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = fx.Workspace()
        self.ctx = fx.make_context(self.ws)

    def tearDown(self) -> None:
        self.ws.cleanup()

    def test_static_planner_is_empty_and_valid(self) -> None:
        outcome = StaticEditPlanner().plan(self.ctx, PRO_STREAM_V1)
        self.assertEqual(outcome.report.status, PlanStatus.VALID)
        self.assertEqual(outcome.plan.events, ())
        self.assertEqual(outcome.model_calls, 0)

    def test_rules_planner_is_deterministic_and_role_aware(self) -> None:
        first = RuleBasedEditPlanner().plan(self.ctx, PRO_STREAM_V1)
        second = RuleBasedEditPlanner().plan(self.ctx, PRO_STREAM_V1)
        self.assertEqual(first.plan.to_dict(), second.plan.to_dict())
        self.assertFalse(first.report.fatal)
        payoff = [e for e in first.plan.events if e.role is StoryRole.PAYOFF]
        self.assertEqual(len(payoff), 1)
        self.assertIn(payoff[0].motion, {MotionPreset.PUNCH_IN, MotionPreset.PUNCH_IN_FAST})
        self.assertFalse(any(e.role is StoryRole.SETUP and e.is_camera_active for e in first.plan.events))

    def test_model_planner_single_call_structured_output(self) -> None:
        client = FakeClient([good_plan_text()])
        outcome = ModelEditPlanner("gpt-test", "medium", client=client).plan(self.ctx, PRO_STREAM_V1)
        self.assertEqual(outcome.model_calls, 1)
        self.assertEqual(len(client.responses.calls), 1)
        call = client.responses.calls[0]
        self.assertEqual(call["text"]["format"]["type"], "json_schema")
        self.assertTrue(call["text"]["format"]["strict"])
        self.assertIn("UNCERTAIN => LESS EDITING", call["instructions"])
        payload = json.loads(call["input"])
        self.assertIn("story_spans", payload)
        self.assertIn("effect_budget", payload)
        self.assertIn("caption_words", payload)
        self.assertEqual(outcome.exchanges[0][1].response_id, "resp_test")
        self.assertEqual(len(outcome.plan.events), 2)

    def test_invalid_json_triggers_exactly_one_repair(self) -> None:
        client = FakeClient(['{"schema_version": 1, "events": [', good_plan_text()])
        outcome = ModelEditPlanner("gpt-test", "medium", client=client).plan(self.ctx, PRO_STREAM_V1)
        self.assertTrue(outcome.repair_attempted)
        self.assertEqual(outcome.model_calls, 2)
        repair_input = json.loads(client.responses.calls[1]["input"])
        self.assertIn("errors", repair_input)
        self.assertNotIn("story_spans", repair_input)  # validator errors only, no full re-analysis

    def test_fatal_twice_raises_no_unbounded_retry(self) -> None:
        fatal = json.dumps(fx.plan(fx.event("e1", 13.0, 14.2, caption="THIS IS CRAZY")))
        client = FakeClient([fatal, fatal, good_plan_text()])
        with self.assertRaises(PlannerOutputError):
            ModelEditPlanner("gpt-test", "medium", client=client).plan(self.ctx, PRO_STREAM_V1)
        self.assertEqual(len(client.responses.calls), 2)

    def test_planner_unavailable(self) -> None:
        client = FakeClient([ConnectionError("offline")])
        with self.assertRaises(PlannerUnavailableError):
            ModelEditPlanner("gpt-test", "medium", client=client).plan(self.ctx, PRO_STREAM_V1)

    def test_replay_enters_the_same_boundary(self) -> None:
        live = ModelEditPlanner("gpt-test", "medium", client=FakeClient([good_plan_text()])).plan(self.ctx, PRO_STREAM_V1)
        recording = self.ws.write_json("raw_response.json", live.exchanges[0][1].to_artifact())
        replay = ProviderEditPlanner(ReplayProvider(recording)).plan(self.ctx, PRO_STREAM_V1)
        self.assertEqual(replay.plan.to_dict()["events"], live.plan.to_dict()["events"])
        self.assertNotIn("api_key", json.dumps(live.exchanges[0][0].to_artifact()).lower())

    def test_hallucinated_ids_are_sanitized_not_trusted(self) -> None:
        text = json.dumps(fx.plan(fx.event("e1", 13.0, 14.2, target={"type": "subject", "id": "face_9"},
                                           emphasis_word_ids=[4242, 8])))
        outcome = ModelEditPlanner("gpt-test", "low", client=FakeClient([text])).plan(self.ctx, PRO_STREAM_V1)
        event = outcome.plan.events[0]
        self.assertEqual(event.target.type.value, "center_safe")
        self.assertEqual(event.emphasis_word_ids, (8,))
        self.assertEqual(outcome.report.status, PlanStatus.SANITIZED)


if __name__ == "__main__":
    unittest.main()
