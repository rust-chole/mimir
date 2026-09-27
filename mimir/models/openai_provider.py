"""Production provider: OpenAI Responses API (structured outputs) + audio transcription."""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

from mimir.config import ModelRoute
from mimir.core import log
from mimir.errors import ConfigError, ModelError
from mimir.models.provider import (
    AudioMeta,
    DiarizedSegment,
    ImageInput,
    ModelProvider,
    TimedWord,
    Transcription,
)

MAX_ATTEMPTS = 3
TIMEOUT_SECONDS = 900.0


def _as_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    for attribute in ("model_dump", "dict"):
        method = getattr(response, attribute, None)
        if callable(method):
            try:
                data = method()
                if isinstance(data, dict):
                    return data
            except Exception:  # pragma: no cover - SDK shape differences
                continue
    return {}


class OpenAIProvider(ModelProvider):
    name = "openai"

    def __init__(self, api_key: str | None = None) -> None:
        key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
        if not key:
            raise ConfigError("OPENAI_API_KEY is not set (production provider)")
        try:
            from openai import OpenAI
        except ImportError as error:  # pragma: no cover - dependency declared in requirements
            raise ConfigError("the 'openai' package is required for the production provider") from error
        self._client = OpenAI(api_key=key, timeout=TIMEOUT_SECONDS, max_retries=0)

    # ------------------------------------------------------------------ retry

    def _call(self, what: str, function, **kwargs) -> Any:
        from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

        delay = 2.0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return function(**kwargs)
            except (APIConnectionError, APITimeoutError, RateLimitError) as error:
                if attempt == MAX_ATTEMPTS:
                    raise ModelError(f"{what}: {type(error).__name__}: {error}") from error
            except APIStatusError as error:
                if error.status_code < 500 or attempt == MAX_ATTEMPTS:
                    raise ModelError(f"{what}: HTTP {error.status_code}: {error}") from error
            log.warn(f"{what}: transient failure, retry {attempt}/{MAX_ATTEMPTS - 1} in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2.0
        raise ModelError(f"{what}: exhausted retries")  # pragma: no cover

    # ------------------------------------------------------------- json task

    def json_task(self, role: str, route: ModelRoute, *, instructions: str, input_text: str,
                  schema: dict[str, Any], schema_name: str, images: Sequence[ImageInput] = ()) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": input_text}]
        for image in images:
            content.append({"type": "input_text", "text": image.label})
            content.append({"type": "input_image", "image_url": image.data_url(), "detail": "low"})
        kwargs: dict[str, Any] = {
            "model": route.model,
            "instructions": instructions,
            "input": [{"role": "user", "content": content}],
            "text": {"format": {"type": "json_schema", "name": schema_name, "strict": True, "schema": schema}},
        }
        if route.effort and route.effort != "none":
            kwargs["reasoning"] = {"effort": route.effort}
        response = self._call(f"{role} ({route.model})", self._client.responses.create, **kwargs)
        raw = str(getattr(response, "output_text", "") or "").strip()
        if not raw:
            raise ModelError(f"{role}: empty structured output")
        try:
            data = json.loads(raw)
        except ValueError as error:
            raise ModelError(f"{role}: invalid JSON output: {raw[:300]}") from error
        if not isinstance(data, dict):
            raise ModelError(f"{role}: structured output is not an object")
        return data

    # ------------------------------------------------------------ transcribe

    def transcribe(self, role: str, route: ModelRoute, audio: Path, *, language: str, prompt: str | None = None,
                   keywords: Sequence[str] = (), word_timestamps: bool = False, logprobs: bool = False,
                   meta: AudioMeta = AudioMeta()) -> Transcription:
        kwargs: dict[str, Any] = {"model": route.model}
        if prompt:
            kwargs["prompt"] = prompt
        if route.model == "gpt-transcribe":
            if keywords:
                kwargs["keywords"] = [k for k in keywords if k.strip()][:32]
            if language:
                kwargs["languages"] = [language]
        elif language:
            kwargs["language"] = language
        if word_timestamps:
            kwargs["response_format"] = "verbose_json"
            kwargs["timestamp_granularities"] = ["word", "segment"]
        elif logprobs and route.model != "whisper-1":
            kwargs["response_format"] = "json"
            kwargs["include"] = ["logprobs"]
        with Path(audio).open("rb") as handle:
            response = self._call(f"{role} ({route.model})", self._client.audio.transcriptions.create,
                                  file=handle, **kwargs)
        data = _as_dict(response)
        text = str(getattr(response, "text", "") or data.get("text", "")).strip()
        words = tuple(
            TimedWord(str(w.get("word", "")).strip(), float(w.get("start", 0.0)), float(w.get("end", 0.0)))
            for w in (data.get("words") or []) if str(w.get("word", "")).strip()
        )
        probs = []
        for row in data.get("logprobs") or []:
            try:
                import math

                probs.append((str(row.get("token", "")), float(math.exp(float(row.get("logprob", -99.0))))))
            except (TypeError, ValueError):
                continue
        return Transcription(text=text, words=words, token_probs=tuple(probs))

    # --------------------------------------------------------------- diarize

    def diarize(self, role: str, route: ModelRoute, audio: Path, *, language: str,
                known_speakers: Sequence[tuple[str, Path]] = (), meta: AudioMeta = AudioMeta()
                ) -> list[DiarizedSegment]:
        kwargs: dict[str, Any] = {"model": route.model, "response_format": "diarized_json",
                                  "chunking_strategy": "auto"}
        if language:
            kwargs["language"] = language
        if known_speakers:
            kwargs["known_speaker_names"] = [name for name, _ in known_speakers][:4]
            kwargs["known_speaker_references"] = [
                "data:audio/wav;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")
                for _, path in known_speakers
            ][:4]
        with Path(audio).open("rb") as handle:
            response = self._call(f"{role} ({route.model})", self._client.audio.transcriptions.create,
                                  file=handle, **kwargs)
        data = _as_dict(response)
        segments = []
        for raw in data.get("segments") or []:
            try:
                start, end = float(raw.get("start", 0.0)), float(raw.get("end", 0.0))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            segments.append(DiarizedSegment(str(raw.get("speaker", "")).strip() or "unknown", start, end,
                                            str(raw.get("text", "")).strip()))
        return segments
