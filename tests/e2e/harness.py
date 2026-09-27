"""Shared helpers for real-render end-to-end tests."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from mimir.config import Settings
from mimir.pipeline import JobResult, run_job
from tests.synth import build, speech
from tests.synth.provider import ScriptedProvider
from tests.synth.scenarios import Scenario

MEDIA_CACHE = Path(__file__).resolve().parents[1] / "synth" / ".media_cache"


def sfx_library(root: Path) -> Path:
    library = root / "sfx"
    for category, make in (("transition", speech.whoosh), ("disbelief", speech.scream), ("impact", speech.impact)):
        path = library / category / f"{category}.wav"
        if not path.exists():
            speech.write_wav(path, make())
    return library


def settings_for(scenario: Scenario, root: Path, **changes: Any) -> Settings:
    base = Settings(workspace=root / "workspace", output_dir=root / "output")
    return base.with_(
        identity=dataclasses.replace(base.identity, entities=scenario.entities, speaker_names=scenario.speaker_names),
        effects=dataclasses.replace(base.effects, sfx_library=str(sfx_library(root))),
        output=dataclasses.replace(base.output, preset="veryfast"),
        **changes,
    )


def run_scenario(scenario: Scenario, root: Path, settings: Settings | None = None, make_provider: Any = None,
                 **kwargs: Any) -> tuple[JobResult, ScriptedProvider, dict[str, Any]]:
    video, truth = build.build(scenario, MEDIA_CACHE)
    provider = (make_provider or ScriptedProvider)(scenario, truth)
    result = run_job(video, settings or settings_for(scenario, root), provider, **kwargs)
    return result, provider, truth


def artifact(result: JobResult, stage: str, name: str) -> Any:
    return result.report.artifacts[stage].json(name)


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
