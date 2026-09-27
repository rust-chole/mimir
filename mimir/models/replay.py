"""Record / replay providers for regression runs.

``RecordingProvider`` wraps a real provider and stores every response keyed by
a content hash of the request (role, model, prompt text, schema, image and
audio bytes). ``ReplayProvider`` serves exactly those responses and fails
loudly on any request it has not seen, so a regression run can never
silently call a live model or improvise an answer.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Sequence

from mimir.config import ModelRoute
from mimir.core.jsonio import content_hash, read_json, sha256_file, write_json
from mimir.errors import ModelError
from mimir.models.provider import (
    AudioMeta,
    DiarizedSegment,
    ImageInput,
    ModelProvider,
    TimedWord,
    Transcription,
)


def json_key(role: str, route: ModelRoute, instructions: str, input_text: str, schema_name: str,
             images: Sequence[ImageInput]) -> str:
    import hashlib

    return content_hash({
        "kind": "json_task", "role": role, "model": route.model, "effort": route.effort,
        "instructions": instructions, "input": input_text, "schema": schema_name,
        "images": [hashlib.sha256(image.jpeg).hexdigest() for image in images],
    })


def audio_key(kind: str, role: str, route: ModelRoute, audio: Path, **options: Any) -> str:
    return content_hash({"kind": kind, "role": role, "model": route.model, "audio": sha256_file(audio),
                         "options": options})


def _transcription_to_dict(value: Transcription) -> dict[str, Any]:
    return {"text": value.text, "words": [dataclasses.asdict(w) for w in value.words],
            "token_probs": [list(p) for p in value.token_probs]}


def _transcription_from_dict(data: dict[str, Any]) -> Transcription:
    return Transcription(
        text=str(data.get("text", "")),
        words=tuple(TimedWord(str(w["text"]), float(w["start"]), float(w["end"])) for w in data.get("words", [])),
        token_probs=tuple((str(t), float(p)) for t, p in data.get("token_probs", [])),
    )


class RecordingProvider(ModelProvider):
    name = "recording"

    def __init__(self, inner: ModelProvider, directory: str | Path) -> None:
        self.inner = inner
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _store(self, key: str, role: str, payload: Any) -> None:
        write_json(self.directory / f"{role}__{key[:24]}.json", {"key": key, "role": role, "response": payload})

    def json_task(self, role, route, *, instructions, input_text, schema, schema_name, images=()):
        response = self.inner.json_task(role, route, instructions=instructions, input_text=input_text,
                                        schema=schema, schema_name=schema_name, images=images)
        self._store(json_key(role, route, instructions, input_text, schema_name, images), role, response)
        return response

    def transcribe(self, role, route, audio, *, language, prompt=None, keywords=(), word_timestamps=False,
                   logprobs=False, meta=AudioMeta()):
        response = self.inner.transcribe(role, route, audio, language=language, prompt=prompt, keywords=keywords,
                                         word_timestamps=word_timestamps, logprobs=logprobs, meta=meta)
        key = audio_key("transcribe", role, route, audio, prompt=prompt, keywords=list(keywords),
                        words=word_timestamps, logprobs=logprobs, language=language)
        self._store(key, role, _transcription_to_dict(response))
        return response

    def diarize(self, role, route, audio, *, language, known_speakers=(), meta=AudioMeta()):
        response = self.inner.diarize(role, route, audio, language=language, known_speakers=known_speakers,
                                      meta=meta)
        key = audio_key("diarize", role, route, audio, language=language,
                        known=[(name, sha256_file(path)) for name, path in known_speakers])
        self._store(key, role, [dataclasses.asdict(s) for s in response])
        return response


class ReplayProvider(ModelProvider):
    name = "replay"

    def __init__(self, directory: str | Path) -> None:
        self.responses: dict[str, Any] = {}
        for path in sorted(Path(directory).glob("*.json")):
            row = read_json(path)
            self.responses[str(row["key"])] = row["response"]

    def _get(self, key: str, role: str) -> Any:
        if key not in self.responses:
            raise ModelError(f"replay: no recorded response for {role} ({key[:12]})")
        return self.responses[key]

    def json_task(self, role, route, *, instructions, input_text, schema, schema_name, images=()):
        return dict(self._get(json_key(role, route, instructions, input_text, schema_name, images), role))

    def transcribe(self, role, route, audio, *, language, prompt=None, keywords=(), word_timestamps=False,
                   logprobs=False, meta=AudioMeta()):
        key = audio_key("transcribe", role, route, audio, prompt=prompt, keywords=list(keywords),
                        words=word_timestamps, logprobs=logprobs, language=language)
        return _transcription_from_dict(self._get(key, role))

    def diarize(self, role, route, audio, *, language, known_speakers=(), meta=AudioMeta()):
        key = audio_key("diarize", role, route, audio, language=language,
                        known=[(name, sha256_file(path)) for name, path in known_speakers])
        return [DiarizedSegment(**row) for row in self._get(key, role)]
