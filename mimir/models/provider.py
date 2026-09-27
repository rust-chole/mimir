"""Provider contract used by every stage.

Three capabilities, each called with a *role* (what the call is for) and the
routed model:

* ``json_task``   structured editorial reasoning (optionally with images); the
                  output must satisfy a strict JSON schema;
* ``transcribe``  one ASR "ear" (text, optional word clock, optional token probabilities);
* ``diarize``     speaker-attributed segments.

Model output is advisory data: every consumer validates it deterministically.
"""
from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from mimir.config import ModelRoute


@dataclass(frozen=True)
class TimedWord:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Transcription:
    text: str
    words: tuple[TimedWord, ...] = ()
    token_probs: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class DiarizedSegment:
    speaker: str
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class ImageInput:
    """A JPEG image shown to a multimodal model, with a factual label (e.g. time)."""

    jpeg: bytes
    label: str

    def data_url(self) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(self.jpeg).decode("ascii")


@dataclass(frozen=True)
class AudioMeta:
    """Where an audio file came from. Metadata only (never sent to a model)."""

    source_start: float = 0.0
    source_end: float = 0.0
    view: str = "raw"            # raw | enhanced
    extra: Mapping[str, Any] = field(default_factory=dict)


class ModelProvider(ABC):
    name = "abstract"

    @abstractmethod
    def json_task(self, role: str, route: ModelRoute, *, instructions: str, input_text: str,
                  schema: dict[str, Any], schema_name: str, images: Sequence[ImageInput] = ()) -> dict[str, Any]:
        ...

    @abstractmethod
    def transcribe(self, role: str, route: ModelRoute, audio: Path, *, language: str, prompt: str | None = None,
                   keywords: Sequence[str] = (), word_timestamps: bool = False, logprobs: bool = False,
                   meta: AudioMeta = AudioMeta()) -> Transcription:
        ...

    @abstractmethod
    def diarize(self, role: str, route: ModelRoute, audio: Path, *, language: str,
                known_speakers: Sequence[tuple[str, Path]] = (), meta: AudioMeta = AudioMeta()
                ) -> list[DiarizedSegment]:
        ...
