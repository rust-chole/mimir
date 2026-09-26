import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from mimir.config import ModelRoute, Settings, parse_speaker_names
from mimir.core.artifacts import ArtifactStore
from mimir.core.runner import PipelineRunner
from mimir.core.stage import SourceInfo, StageContext, StageOutput
from mimir.errors import ConfigError, ModelError
from mimir.media.ffmpeg import escape_filter_path
from mimir.models.provider import AudioMeta, ModelProvider, Transcription
from mimir.models.replay import RecordingProvider, ReplayProvider


class Echo:
    def __init__(self, name, deps, key, version=1):
        self.name, self.deps, self.key, self.version = name, deps, key, version
        self.runs = 0

    def params(self, settings):
        return {"key": getattr(settings.captions, self.key) if self.key else None}

    def run(self, ctx: StageContext) -> StageOutput:
        self.runs += 1
        upstream = {d: ctx.dep(d).json("value") for d in self.deps if d != "source"}
        path = ctx.out_dir / "blob.bin"
        path.write_bytes(b"x" * 10)
        return StageOutput(data={"value": {"stage": self.name, "upstream": upstream,
                                           "param": self.params(ctx.settings)}}, files={"blob": path})


def make(tmp_path, stages, settings=None):
    source = SourceInfo(tmp_path / "v.mp4", "abc", "job")
    return PipelineRunner(stages, settings=settings or Settings(), provider=None, store=ArtifactStore(tmp_path / "job"),
                          source=source)


def test_stage_specific_invalidation(tmp_path):
    a, b, c = Echo("a", ("source",), None), Echo("b", ("a",), "size"), Echo("c", ("a",), None)
    make(tmp_path, [a, b, c]).run()
    assert (a.runs, b.runs, c.runs) == (1, 1, 1)
    make(tmp_path, [a, b, c]).run()
    assert (a.runs, b.runs, c.runs) == (1, 1, 1)
    settings = Settings().with_(captions=Settings().captions.__class__(size=80))
    report = make(tmp_path, [a, b, c], settings).run()
    assert (a.runs, b.runs, c.runs) == (1, 2, 1) and report.executed == ["b"]
    report = make(tmp_path, [a, b, c]).run()      # switching back reuses the older signature
    assert b.runs == 2 and report.executed == []


def test_resumed_report_runs_only_the_remaining_stages(tmp_path):
    a, b, c = Echo("a", ("source",), None), Echo("b", ("a",), "size"), Echo("c", ("b",), None)
    runner = make(tmp_path, [a, b, c])
    report = runner.run(until="b")
    assert report.executed == ["a", "b"] and "c" not in report.artifacts
    report = runner.run(report=report)
    assert (a.runs, b.runs, c.runs) == (1, 1, 1) and report.executed == ["a", "b", "c"]


def test_tampered_artifacts_are_not_reused(tmp_path):
    a = Echo("a", ("source",), None)
    report = make(tmp_path, [a]).run()
    blob = report.artifacts["a"].path("blob")
    blob.write_bytes(b"y" * 10)
    make(tmp_path, [a]).run()
    assert a.runs == 2


def test_version_bump_and_rerun_invalidate(tmp_path):
    a = Echo("a", ("source",), None)
    make(tmp_path, [a]).run()
    a.version = 2
    make(tmp_path, [a]).run()
    assert a.runs == 2
    runner = PipelineRunner([a], settings=Settings(), provider=None, store=ArtifactStore(tmp_path / "job"),
                            source=SourceInfo(tmp_path / "v.mp4", "abc", "job"), rerun=["a"])
    runner.run()
    assert a.runs == 3


def test_unknown_dependency_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        make(tmp_path, [Echo("b", ("a",), None)])


def test_filter_path_escaping_handles_drive_colons_and_quotes(tmp_path):
    escaped = escape_filter_path(tmp_path / "it's" / "a:b.ass")
    assert "\\:" in escaped and "'\\''" in escaped and "\\\\" not in escaped


def test_config_helpers():
    assert parse_speaker_names("s1=Kai, S2 = Tyla") == (("S1", "Kai"), ("S2", "Tyla"))
    with pytest.raises(ConfigError):
        parse_speaker_names("Kai")
    with pytest.raises(ConfigError):
        ModelRoute("x", "turbo")


class Fixed(ModelProvider):
    name = "fixed"

    def json_task(self, role, route, *, instructions, input_text, schema, schema_name, images=()):
        return {"answer": input_text.upper()}

    def transcribe(self, role, route, audio, *, language, prompt=None, keywords=(), word_timestamps=False,
                   logprobs=False, meta=AudioMeta()):
        return Transcription("hello world")

    def diarize(self, role, route, audio, *, language, known_speakers=(), meta=AudioMeta()):
        return []


def test_record_then_replay_is_exact_and_unknown_requests_fail(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF....")
    route = ModelRoute("m", "low")
    recorder = RecordingProvider(Fixed(), tmp_path / "rec")
    assert recorder.json_task("r", route, instructions="i", input_text="abc", schema={}, schema_name="s") == \
        {"answer": "ABC"}
    recorder.transcribe("t", route, audio, language="en")
    replay = ReplayProvider(tmp_path / "rec")
    assert replay.json_task("r", route, instructions="i", input_text="abc", schema={}, schema_name="s") == \
        {"answer": "ABC"}
    assert replay.transcribe("t", route, audio, language="en").text == "hello world"
    with pytest.raises(ModelError):
        replay.json_task("r", route, instructions="i", input_text="other", schema={}, schema_name="s")


def test_openai_provider_request_shapes(monkeypatch):
    from mimir.models import openai_provider

    calls = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(("responses", kwargs))
            return type("R", (), {"output_text": json.dumps({"ok": True})})()

    class FakeTranscriptions:
        def create(self, **kwargs):
            calls.append(("audio", {k: v for k, v in kwargs.items() if k != "file"}))
            return {"text": "hi", "words": [{"word": "hi", "start": 0.1, "end": 0.3}], "segments": []}

    class FakeClient:
        responses = FakeResponses()
        audio = type("A", (), {"transcriptions": FakeTranscriptions()})()

    provider = openai_provider.OpenAIProvider.__new__(openai_provider.OpenAIProvider)
    provider._client = FakeClient()
    provider._call = lambda what, fn, **kw: fn(**kw)
    assert provider.json_task("story_judge", ModelRoute("gpt-5.6-terra", "high"), instructions="x", input_text="y",
                              schema={"type": "object"}, schema_name="n") == {"ok": True}
    kwargs = calls[-1][1]
    assert kwargs["reasoning"] == {"effort": "high"} and kwargs["text"]["format"]["strict"] is True
    Path("/tmp/_mimir_fake.wav").write_bytes(b"x")
    result = provider.transcribe("transcribe_primary", ModelRoute("gpt-transcribe"), Path("/tmp/_mimir_fake.wav"),
                                 language="en", keywords=["Tyla"])
    assert calls[-1][1]["keywords"] == ["Tyla"] and calls[-1][1]["languages"] == ["en"]
    provider.transcribe("timing", ModelRoute("whisper-1"), Path("/tmp/_mimir_fake.wav"), language="en",
                        word_timestamps=True)
    assert calls[-1][1]["timestamp_granularities"] == ["word", "segment"]
    assert result.text == "hi"
