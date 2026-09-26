"""Stage-specific caching on a real job: changes re-run only the stages that depend on them."""
from __future__ import annotations

import dataclasses

import pytest

from tests.e2e.harness import run_scenario, settings_for
from tests.synth import scenarios

pytestmark = pytest.mark.render

UPSTREAM = {"probe", "transcript", "vod_evidence", "story", "caption_verify", "speakers", "vision"}


def test_changes_invalidate_only_their_dependents(tmp_path):
    scenario = scenarios.single_speaker()
    settings = settings_for(scenario, tmp_path)
    first, provider, _ = run_scenario(scenario, tmp_path, settings)
    assert first.qc["passed"]

    caption_style = settings.with_(captions=dataclasses.replace(settings.captions, size=80))
    result, provider, _ = run_scenario(scenario, tmp_path, caption_style)
    assert set(result.report.executed) <= {"captions", "render", "qc", "publish"}
    assert "captions" in result.report.executed and "render" in result.report.executed
    assert provider.calls == {}                      # no model call for a style change

    camera = settings.with_(camera=dataclasses.replace(settings.camera, medium_zoom=1.12))
    result, provider, _ = run_scenario(scenario, tmp_path, camera)
    assert "edit_compile" in result.report.executed
    assert not (UPSTREAM | {"edit_context", "edit_direction", "cold_open", "timeline"}) & set(result.report.executed)

    names = settings.with_(identity=dataclasses.replace(settings.identity, speaker_names=(("S1", "Sam"),)))
    result, provider, _ = run_scenario(scenario, tmp_path, names)
    assert {"identity", "caption_truth"} <= set(result.report.executed)
    assert not UPSTREAM & set(result.report.executed)
    assert provider.calls["transcribe_primary"] == 0 and provider.calls["story_scout"] == 0
