"""Qwen Omni (Alibaba Cloud Model Studio / DashScope) as the primary caption ear.

Provider boundary: nothing outside this module knows about DashScope. The
OpenAI Python SDK talks to the OpenAI-compatible endpoint; the endpoint and
the key come from configuration only (``DASHSCOPE_BASE_URL``,
``DASHSCOPE_API_KEY``), no region is assumed and neither is ever printed.

Request: text-only output, ``reasoning_effort`` from config (``none`` for
listening), JSON Object mode, streamed (Omni models stream their output; the
text deltas are joined here). Qwen JSON Object mode is not a strict schema, so
every response goes through:

    provider text -> JSON parse -> type validation -> required fields
                  -> semantic validation (lexical consistency, no annotations)

The evidence contract is deliberately small: language, verbatim text, its
utterances, and the spans the ear itself doubts. It carries no timing: WHEN a
word was said is decided by the word-alignment provider only.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ai.caption_stack.config import CaptionStackSettings

EVIDENCE_VERSION = 1
MAX_UNCERTAIN_SPANS = 12
MAX_ALTERNATIVES = 4
_ANNOTATION = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]+\*")
_TOKEN = re.compile(r"\S+")


class QwenUnavailable(RuntimeError):
    """Qwen cannot be used for this run (configuration, auth, transport)."""


class QwenInvalidResponse(RuntimeError):
    """Qwen answered, but not with valid caption evidence (after bounded retries)."""


@dataclass(frozen=True)
class UncertainSpan:
    heard: str
    alternatives: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"heard": self.heard, "alternatives": list(self.alternatives)}


@dataclass(frozen=True)
class QwenCaptionEvidence:
    language: str
    text: str
    utterances: tuple[str, ...]
    uncertain_spans: tuple[UncertainSpan, ...]
    notes: tuple[str, ...] = ()


# ============================================================
# PROMPTS
# ============================================================

_CONTRACT = """
Return ONLY one JSON object, no markdown:
{"language": "<ISO 639-1 code>", "text": "<the complete verbatim transcript>",
 "utterances": [{"text": "<one utterance>"}],
 "uncertain_spans": [{"heard": "<words exactly as written in text>", "alternatives": ["<another hearing>"]}]}
- "utterances" split "text" in order: together they contain exactly the words of "text", nothing added or removed.
- "uncertain_spans" lists only words you are genuinely unsure about ([] when none).
- No timestamps, no speaker names, no sound descriptions, no commentary.
""".strip()

_LISTENING_RULES = """
You are a verbatim speech-recognition ear for the captions of one short video (gaming, livestream or creator
content). Transcribe exactly what is audibly spoken in the attached audio.

- Keep profanity, slang, contractions, negations, repetitions, fillers, stutters, clipped words and unfinished
  sentences exactly as spoken.
- Never improve grammar, paraphrase, summarize, censor, complete unfinished speech, or replace strange speech
  with more natural speech. Never add a word that is not audible.
- Audible evidence always beats expected wording. When a word is unclear, write what you actually hear and list
  it in uncertain_spans with the alternatives you considered.
- Transcribe speech only: no descriptions of music, noise, laughter or actions.
""".strip()


def primary_instructions(language: str, domain_terms: Sequence[str]) -> str:
    """Pass A: the least-biased ear. Minimal domain context, no participant names."""
    parts = [_LISTENING_RULES, f"Expected language: {language}."]
    if domain_terms:
        parts.append("Livestream vocabulary that often occurs (spelling reference only): "
                     + ", ".join(domain_terms) + ". \"chat\" usually means the stream audience, not a person.")
    parts.append(_CONTRACT)
    return "\n\n".join(parts)


def precision_instructions(language: str, domain_terms: Sequence[str], verified_terms: Sequence[str]) -> str:
    """Pass B: listen again from scratch, with verified spellings as reference only."""
    parts = [_LISTENING_RULES, f"Expected language: {language}.",
             "PRECISION LISTEN: listen from scratch, word by word. Negations, contractions, numbers, names, "
             "repetitions and short function words are critical."]
    if domain_terms:
        parts.append("Livestream vocabulary that often occurs (spelling reference only): " + ", ".join(domain_terms)
                     + ".")
    if verified_terms:
        parts.append("Verified names / terms: " + ", ".join(verified_terms) + ". These are SPELLING REFERENCES ONLY. "
                     "They do NOT prove that a name or word was spoken. Use one only when the audio clearly says it; "
                     "never force a reference into ambiguous audio.")
    parts.append(_CONTRACT)
    return "\n\n".join(parts)


# ============================================================
# PARSING + VALIDATION (never trust json.loads alone)
# ============================================================

def _extract_json(raw: str) -> Any:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except ValueError:
        first, last = text.find("{"), text.rfind("}")
        if first < 0 or last <= first:
            raise QwenInvalidResponse("response is not JSON") from None
        try:
            return json.loads(text[first:last + 1])
        except ValueError as error:
            raise QwenInvalidResponse(f"response is not valid JSON: {error}") from None


def _lexical_keys(text: str) -> list[str]:
    from ai.caption_stack import lexical

    return lexical.keys_of(lexical.display_tokens(text))


def _clean_text(value: str, notes: list[str], where: str) -> str:
    cleaned = " ".join(_ANNOTATION.sub(" ", value).split())
    if cleaned != " ".join(value.split()):
        notes.append(f"{where}: removed bracketed non-speech annotation(s)")
    return cleaned


def parse_caption_evidence(raw: str) -> QwenCaptionEvidence:
    """Provider text -> validated evidence; raises ``QwenInvalidResponse``."""
    data = _extract_json(raw)
    if not isinstance(data, Mapping):
        raise QwenInvalidResponse("response JSON is not an object")
    notes: list[str] = []
    unknown = sorted(set(data) - {"language", "text", "utterances", "uncertain_spans"})
    if unknown:
        notes.append("ignored fields: " + ", ".join(unknown[:6]))

    text = data.get("text")
    if not isinstance(text, str):
        raise QwenInvalidResponse("'text' is missing or not a string")
    language = data.get("language", "")
    if not isinstance(language, str):
        raise QwenInvalidResponse("'language' is not a string")

    raw_utterances = data.get("utterances", [])
    if not isinstance(raw_utterances, list):
        raise QwenInvalidResponse("'utterances' is not a list")
    utterances: list[str] = []
    for item in raw_utterances:
        value = item.get("text") if isinstance(item, Mapping) else None
        if not isinstance(value, str):
            raise QwenInvalidResponse("an utterance is not an object with a string 'text'")
        value = _clean_text(value, notes, "utterance")
        if value:
            utterances.append(value)

    raw_spans = data.get("uncertain_spans", [])
    if not isinstance(raw_spans, list):
        raise QwenInvalidResponse("'uncertain_spans' is not a list")

    text = _clean_text(text, notes, "text")
    if not text and utterances:
        text = " ".join(utterances)
        notes.append("text was empty; joined utterances")
    if not _lexical_keys(text):
        raise QwenInvalidResponse("no lexical content in 'text'")
    if any(marker in text for marker in ("{", "}", "```", "\"text\"")):
        raise QwenInvalidResponse("'text' contains JSON/markdown instead of speech")
    if utterances and _lexical_keys(" ".join(utterances)) != _lexical_keys(text):
        raise QwenInvalidResponse("'utterances' do not contain exactly the words of 'text'")

    text_keys = _lexical_keys(text)
    spans: list[UncertainSpan] = []
    for item in raw_spans:
        if not isinstance(item, Mapping) or not isinstance(item.get("heard"), str):
            raise QwenInvalidResponse("an uncertain span is not an object with a string 'heard'")
        alternatives = item.get("alternatives", [])
        if not isinstance(alternatives, list) or not all(isinstance(a, str) for a in alternatives):
            raise QwenInvalidResponse("uncertain span 'alternatives' is not a list of strings")
        heard = " ".join(item["heard"].split())
        heard_keys = _lexical_keys(heard)
        if not heard_keys or not _contains(text_keys, heard_keys):
            notes.append(f"dropped uncertain span not found in text: {heard[:60]!r}")
            continue
        clean_alternatives: list[str] = []
        for alternative in alternatives:
            value = " ".join(alternative.split())
            if value and _lexical_keys(value) != heard_keys and value not in clean_alternatives:
                clean_alternatives.append(value)
        spans.append(UncertainSpan(heard, tuple(clean_alternatives[:MAX_ALTERNATIVES])))
    if len(spans) > MAX_UNCERTAIN_SPANS:
        notes.append(f"kept the first {MAX_UNCERTAIN_SPANS} of {len(spans)} uncertain spans")
        spans = spans[:MAX_UNCERTAIN_SPANS]
    return QwenCaptionEvidence(language.strip().lower(), text, tuple(utterances), tuple(spans), tuple(notes))


def _contains(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    size = len(needle)
    return any(list(haystack[i:i + size]) == list(needle) for i in range(0, len(haystack) - size + 1))


# ============================================================
# CLIENT
# ============================================================

_client_lock = threading.Lock()
_clients: dict[tuple[str, int, float, int], Any] = {}


def _client(settings: CaptionStackSettings) -> Any:
    if not settings.dashscope_api_key:
        raise QwenUnavailable("DASHSCOPE_API_KEY is not set")
    if not settings.dashscope_base_url:
        raise QwenUnavailable("DASHSCOPE_BASE_URL is not set")
    key = (settings.dashscope_base_url, hash(settings.dashscope_api_key), float(settings.qwen_timeout_s),
           int(settings.qwen_max_retries))
    with _client_lock:
        client = _clients.get(key)
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=settings.dashscope_api_key, base_url=settings.dashscope_base_url,
                            timeout=float(settings.qwen_timeout_s), max_retries=int(settings.qwen_max_retries))
            _clients[key] = client
    return client


def _redact(message: str, settings: CaptionStackSettings) -> str:
    text = str(message)
    if settings.dashscope_api_key:
        text = text.replace(settings.dashscope_api_key, "***")
    return text[:300]


def _stream_text(stream: Any) -> tuple[str, dict[str, Any]]:
    """Join the text deltas of a streamed chat completion (reasoning deltas are ignored)."""
    choices = getattr(stream, "choices", None)
    if choices is not None:                     # a non-streamed completion object
        message = getattr(choices[0], "message", None) if choices else None
        return str(getattr(message, "content", "") or ""), {}
    parts: list[str] = []
    usage: dict[str, Any] = {}
    for chunk in stream:
        for choice in getattr(chunk, "choices", None) or []:
            delta = getattr(choice, "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if isinstance(content, str):
                parts.append(content)
        raw_usage = getattr(chunk, "usage", None)
        if raw_usage is not None:
            dump = getattr(raw_usage, "model_dump", None)
            usage = dict(dump()) if callable(dump) else dict(raw_usage) if isinstance(raw_usage, Mapping) else {}
    return "".join(parts), usage


def _is_transient(error: Exception) -> bool:
    import openai

    return isinstance(error, (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError,
                              openai.InternalServerError))


def listen(audio_base64: str, *, instructions: str, settings: CaptionStackSettings,
           audio_format: str = "wav", create: Callable[..., Any] | None = None
           ) -> tuple[QwenCaptionEvidence, dict[str, Any]]:
    """One Qwen listen with bounded retries on invalid evidence -> (evidence, meta)."""
    if create is None:
        create = _client(settings).chat.completions.create
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": f"data:;base64,{audio_base64}", "format": audio_format}},
            {"type": "text", "text": "Transcribe this audio. Return the JSON object only."},
        ]},
    ]
    attempts = 1 + max(0, int(settings.qwen_max_retries))
    problems: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            stream = create(model=settings.qwen_model, messages=messages, modalities=["text"],
                            reasoning_effort=settings.qwen_reasoning_effort,
                            response_format={"type": "json_object"},
                            stream=True, stream_options={"include_usage": True})
            raw, usage = _stream_text(stream)
        except Exception as error:  # the SDK already retried transient transport errors
            kind = "transient" if _is_transient(error) else "request"
            raise QwenUnavailable(f"{settings.qwen_model} {kind} error: {type(error).__name__}: "
                                  f"{_redact(str(error), settings)}") from None
        try:
            evidence = parse_caption_evidence(raw)
        except QwenInvalidResponse as error:
            problems.append(f"attempt {attempt}: {error}")
            continue
        return evidence, {"provider": "qwen_omni", "model": settings.qwen_model,
                          "reasoning_effort": settings.qwen_reasoning_effort, "attempts": attempt,
                          "invalid_attempts": problems, "usage": usage, "evidence_version": EVIDENCE_VERSION}
    raise QwenInvalidResponse("; ".join(problems) or "no valid caption evidence")
