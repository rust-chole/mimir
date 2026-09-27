"""A diarizer that returns nothing must not become a confident single speaker in the rendered Short."""
from __future__ import annotations

import json

import pytest

from tests.e2e.harness import artifact, run_scenario
from tests.synth import scenarios
from tests.synth.provider import ScriptedProvider

pytestmark = pytest.mark.render


class SilentDiarizer(ScriptedProvider):
    def diarize(self, role, route, audio, *, language, known_speakers=(), meta=None):
        self.calls[role] += 1
        return []


def test_empty_diarization_publishes_conservatively_and_says_so(tmp_path):
    scenario = scenarios.two_speakers()
    result, provider, truth = run_scenario(scenario, tmp_path, make_provider=SilentDiarizer)
    assert provider.calls["diarize"] == 1
    speakers = artifact(result, "speakers", "speakers")
    assert speakers["mode"] == "unresolved" and speakers["resolution"]["status"] == "unresolved"
    words = artifact(result, "caption_truth", "caption_truth")["words"]
    assert words and all(not w["speaker"] for w in words)                    # no fabricated S1
    identity = artifact(result, "identity", "identity")
    assert identity["speakers"] == {} and not identity["ambiguous"]        # no identity from the fallback
    captions = artifact(result, "captions", "captions")
    assert all(not r["label"] and r["lane"] == "main" for r in captions["words"])
    plan = artifact(result, "edit_compile", "render_plan")
    assert not {s["intent"] for s in plan["spans"]} & {"SPEAKER_MEDIUM", "SPEAKER_PUNCH", "REACTION"}
    # QC tells the unresolved result apart from a confirmed one, and it is not a silent pass
    check = next(c for c in result.qc["checks"] if c["name"] == "speaker_resolution")
    assert not check["passed"] and check["severity"] == "warn" and check["details"]["status"] == "unresolved"
    assert check["details"]["violations"] == []
    assert result.qc["passed"]                                               # safe to publish, but flagged
    manifest = json.loads(result.manifest.read_text())
    assert manifest["speakers"]["status"] == "unresolved" and "speaker_resolution" in manifest["qc"]["warnings"]
    summary = json.loads((result.job_dir / "run_summary.json").read_text())
    assert "speaker_resolution" in summary["qc_warnings"]
