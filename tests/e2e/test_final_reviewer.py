"""The optional multimodal final reviewer on real rendered Shorts.

The reviewer role is played by the scripted stand-in (no live model in CI); what is exercised is
everything MIMIR owns: choosing the story moments, pairing SOURCE and RENDERED frames with their
evidence, the strict schema, mapping findings to deterministic repairs, and the one-repair bound.
"""
from __future__ import annotations

import dataclasses
import shutil

import pytest

from mimir.errors import QualityGateError
from tests.e2e.harness import artifact, run_scenario, settings_for
from tests.synth import scenarios
from tests.synth.provider import ScriptedProvider

pytestmark = pytest.mark.render

MOMENTS = {"cold_open_peak", "main_restart", "setup", "escalation", "payoff", "reaction"}


def reviewer_settings(root):
    scenario = scenarios.two_speakers()
    settings = settings_for(scenario, root)
    return scenario, settings.with_(qc=dataclasses.replace(settings.qc, reviewer=True))


@pytest.fixture(scope="module")
def reviewed(tmp_path_factory):
    root = tmp_path_factory.mktemp("reviewer")
    scenario, settings = reviewer_settings(root)
    result, provider, truth = run_scenario(scenario, root, settings)
    return root, result, provider


def rerun_with(base_root, tmp_path, script):
    """Same job from a copy of the reviewed workspace; only QC (and a repair round) run again."""
    shutil.copytree(base_root / "workspace", tmp_path / "workspace")
    scenario, settings = reviewer_settings(tmp_path)
    provider_box = {}

    def make(*args):
        provider = ScriptedProvider(*args)
        provider.review_script = script
        provider_box["p"] = provider
        return provider

    import tests.e2e.harness as harness
    original = harness.ScriptedProvider
    harness.ScriptedProvider = make
    try:
        result, provider, _ = run_scenario(scenario, tmp_path, settings, rerun=("qc",))
        return result, provider, None
    except QualityGateError as error:
        return None, provider_box["p"], error
    finally:
        harness.ScriptedProvider = original


def payoff_issue(kind):
    def script(frame_ids, evidence):
        fid = next(f for f, line in evidence.items() if "moment=payoff" in line)
        return {"verdict": "fail", "issues": [{"type": kind, "frame_id": fid, "severity": "high",
                                               "expected": "both participants of the payoff visible",
                                               "observed": "one participant cut out"}]}
    return script


def cold_open_issue(frame_ids, evidence):
    fid = next(f for f, line in evidence.items() if "moment=cold_open_peak" in line)
    return {"verdict": "fail", "issues": [{"type": "weak_or_wrong_cold_open", "frame_id": fid, "severity": "high",
                                           "expected": "the payoff peak", "observed": "an unrelated moment"}]}


def test_reviewer_compares_story_moments_with_evidence(reviewed):
    root, result, provider = reviewed
    assert result.qc["passed"]
    check = next(c for c in result.qc["checks"] if c["name"] == "final_review")
    assert check["passed"] and check["details"]["verdict"] == "pass"
    request = provider.review_requests[0]
    moments = {line.split("moment=")[1].split(" | ")[0] for line in request["evidence"].values()}
    assert MOMENTS <= moments
    assert request["images"] == 2 * len(request["frame_ids"]) <= 16
    for line in request["evidence"].values():
        for field in ("segment=", "beat=", "intent=", "required=", "burned_captions=", "caption_truth="):
            assert field in line
    payoff = next(line for line in request["evidence"].values() if "moment=payoff" in line)
    assert "R1 subject" in payoff and "R2 subject" in payoff      # both participants are REQUIRED there
    assert provider.calls["final_reviewer"] == 1


def test_visual_failure_gets_exactly_one_repair_round(reviewed, tmp_path):
    root, _, _ = reviewed
    result, provider, error = rerun_with(root, tmp_path, [payoff_issue("important_visual_cropped"), None])
    assert error is None and result.qc["passed"] and result.qc["repair_round"] == 1
    assert provider.calls["final_reviewer"] == 2
    corrections = artifact(result, "edit_validation", "validated_plan")["corrections"]
    assert any(c["code"] == "qc_repair_widen" for c in corrections)
    assert result.published_video.is_file()


def test_persistent_failure_stops_after_one_repair_round(reviewed, tmp_path):
    root, _, _ = reviewed
    issue = payoff_issue("important_visual_cropped")
    result, provider, error = rerun_with(root, tmp_path, [issue, issue, issue])
    assert result is None and isinstance(error, QualityGateError)
    assert "after 1 repair round" in str(error)
    assert provider.calls["final_reviewer"] == 2                   # no judge -> re-render loop


def test_story_level_failure_is_not_auto_repaired(reviewed, tmp_path):
    root, _, _ = reviewed
    result, provider, error = rerun_with(root, tmp_path, [cold_open_issue])
    assert result is None and isinstance(error, QualityGateError)
    assert "after 0 repair round" in str(error)
    assert provider.calls["final_reviewer"] == 1
