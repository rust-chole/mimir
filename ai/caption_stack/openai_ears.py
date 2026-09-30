"""OpenAI transcription as a BOUNDED caption ear (never the primary authority by default).

Used only when genuinely needed: the Qwen provider is unavailable or returned no
valid evidence, the two Qwen ears materially disagree (one full-short
model-diverse listen), or a local disputed phrase needs an independent ear
(a few seconds of audio around it). ``gpt-transcribe`` receives its keyword and
language hints through ``extra_body``; older models use the singular
``language`` field.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

DEFAULT_DOMAIN_KEYWORDS = "chat,Twitch,YouTube,Discord,stream,streamer,IRL,dono,no cap"
_PLACEHOLDER_NAMES = {"a", "b", "c", "speaker a", "speaker b", "speaker c", "main", "secondary", "unknown"}


def domain_keywords() -> list[str]:
    """Generic livestream vocabulary (``MIMIR_CAPTION_KEYWORDS``); never participant names."""
    raw = os.getenv("MIMIR_CAPTION_KEYWORDS", DEFAULT_DOMAIN_KEYWORDS)
    return clean_terms(str(raw).split(","), limit=16)


def clean_terms(values: Sequence[str] | None, *, limit: int = 12) -> list[str]:
    """De-duplicated verified names/terms; speaker placeholders are never names."""
    result: list[str] = []
    seen: set[str] = set()
    for raw in values or []:
        value = " ".join(str(raw).strip().split())
        key = value.casefold()
        if not value or key in seen or key in _PLACEHOLDER_NAMES:
            continue
        seen.add(key)
        result.append(value)
    return result[:limit]


def transcribe(audio_path: str | Path, *, model: str, language: str, prompt: str | None = None,
               keywords: Sequence[str] | None = None, client: Any = None) -> str:
    """One OpenAI file transcription with model-correct hint fields."""
    if client is None:
        from ai.openai_client import client as shared_client

        client = shared_client
    audio_path = Path(audio_path).resolve()
    with audio_path.open("rb") as audio_file:
        kwargs: dict[str, Any] = {"model": str(model), "file": audio_file}
        if prompt:
            kwargs["prompt"] = prompt
        if str(model) == "gpt-transcribe":
            extra: dict[str, Any] = {}
            clean = clean_terms(keywords, limit=32)
            if clean:
                extra["keywords"] = clean
            if language:
                extra["languages"] = [language]
            if extra:
                kwargs["extra_body"] = extra
        elif language:
            kwargs["language"] = language
        response = client.audio.transcriptions.create(**kwargs)
    text = getattr(response, "text", None)
    if text is None and isinstance(response, dict):
        text = response.get("text")
    return " ".join(str(text or "").split())


def acoustic_listen(audio_path: str | Path, *, model: str, language: str, client: Any = None) -> str:
    """Unprompted ear: generic domain keywords only, no names, no context."""
    return transcribe(audio_path, model=model, language=language, keywords=domain_keywords(), client=client)


def precision_listen(audio_path: str | Path, *, model: str, language: str, verified_terms: Sequence[str],
                     client: Any = None) -> str:
    """Verified-spelling ear: the names are spelling references, never proof they were said."""
    from ai import vod_processor

    names = clean_terms(verified_terms)
    prompt = vod_processor.TRANSCRIPTION_PROMPT + (
        "\n\nVERIFIED SPELLINGS (reference only): " + ", ".join(names)
        + ". If the audio clearly says one of these, use this spelling exactly. They do NOT prove the word was "
          "spoken; never force one into ambiguous audio." if names else "") + (
        "\n\nPRECISION LISTEN: listen from scratch. Preserve contractions, negations, names, numbers, slang, "
        "profanity, clipped words and repetitions exactly as audible. Never make the sentence more grammatical.")
    return transcribe(audio_path, model=model, language=language, prompt=prompt,
                      keywords=[*domain_keywords(), *names], client=client)
