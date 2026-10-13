"""Qwen3-ASR as a caption ear: a dedicated speech recognizer, local or through DashScope.

Qwen3-ASR is Qwen's ASR model post-trained from Qwen3-Omni for one job: transcription.
The Omni ear (``qwen_omni``) asks a general chat model to transcribe AND produce JSON,
utterances and its own doubts in one call; Qwen3-ASR returns plain text and takes free
text as recognition context (names, jargon) in its system message.

    qwen3_asr         local, official ``qwen_asr`` package (already installed for the forced
                      aligner); model ``MIMIR_QWEN_ASR_MODEL`` (Qwen/Qwen3-ASR-1.7B), loaded
                      lazily once per process; device / dtype like the aligner
    qwen3_asr_flash   DashScope OpenAI-compatible endpoint (same key and URL as the Omni ear),
                      model ``MIMIR_QWEN_ASR_API_MODEL`` (qwen3-asr-flash); context as the
                      system message, ``asr_options`` (language, ITN off) in the request body

Selected with ``MIMIR_CAPTION_PRIMARY_PROVIDER``; NOT the default. The caption evaluation
(``python -m ai.caption_eval``) compares it with the Omni ear on real shorts first.

Pass A (primary) gets only the generic livestream vocabulary as context; pass B
(precision) also gets the verified names, as the Omni precision ear does. The
returned evidence has the Omni ear's shape (text, no self-reported doubts).
"""
from __future__ import annotations

import os
import threading
from typing import Any, Callable, Sequence

from ai.caption_stack.config import CaptionStackSettings, asr_language, resolve_device, resolve_dtype

ASR_PROVIDERS = ("qwen3_asr", "qwen3_asr_flash")
DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_API_MODEL = "qwen3-asr-flash"
MAX_NEW_TOKENS = 2048            # qwen_asr's from_pretrained default (512) would truncate a 3-minute short


class AsrUnavailable(RuntimeError):
    """Qwen3-ASR cannot be used for this run (package, model, configuration, transport)."""


def context_text(domain_terms: Sequence[str], verified_terms: Sequence[str] = ()) -> str:
    """Recognition context: what kind of audio this is, its vocabulary, and (pass B) verified names.

    ``MIMIR_CAPTION_CONTEXT`` adds free text (game title, channel topic, recurring jargon)."""
    parts = ["Captions of a short gaming / livestream / creator video. Transcribe exactly what is said."]
    if domain_terms:
        parts.append("Vocabulary: " + ", ".join(domain_terms) + ". \"chat\" usually means the stream audience.")
    if verified_terms:
        parts.append("Names (spelling reference; use only when actually said): " + ", ".join(verified_terms) + ".")
    extra = " ".join(str(os.getenv("MIMIR_CAPTION_CONTEXT", "") or "").split())
    if extra:
        parts.append(extra)
    return " ".join(parts)


# ============================================================
# LOCAL (official qwen_asr)
# ============================================================

_lock = threading.Lock()
_models: dict[tuple[str, str, str], Any] = {}


def _load_local(settings: CaptionStackSettings) -> Any:
    key = (settings.qwen_asr_model, settings.qwen_asr_device, settings.qwen_asr_dtype)
    with _lock:
        cached = _models.get(key)
        if cached is None:
            try:
                import torch
                from qwen_asr import Qwen3ASRModel

                device, _note = resolve_device(settings.qwen_asr_device, torch)
                dtype, _name = resolve_dtype(settings.qwen_asr_dtype, device, torch)
                cached = Qwen3ASRModel.from_pretrained(settings.qwen_asr_model, dtype=dtype, device_map=device,
                                                       max_new_tokens=MAX_NEW_TOKENS, max_inference_batch_size=1)
            except Exception as error:
                cached = AsrUnavailable(f"Qwen3-ASR {settings.qwen_asr_model} could not be loaded: "
                                        f"{type(error).__name__}: {str(error)[:300]}")
            _models[key] = cached
    if isinstance(cached, BaseException):
        raise cached
    return cached


def reset_local_models() -> None:
    with _lock:
        _models.clear()


def transcribe_local(path: str, *, context: str, settings: CaptionStackSettings,
                     loader: Callable[[CaptionStackSettings], Any] = _load_local) -> tuple[str, dict[str, Any]]:
    model = loader(settings)
    try:
        language = asr_language(settings.language)
    except ValueError as error:
        raise AsrUnavailable(str(error)) from None
    try:
        results = model.transcribe(audio=str(path), context=context, language=language)
    except Exception as error:
        raise AsrUnavailable(f"Qwen3-ASR transcription failed: {type(error).__name__}: {str(error)[:300]}") from None
    if not results:
        raise AsrUnavailable("Qwen3-ASR returned no result")
    text = " ".join(str(getattr(results[0], "text", "") or "").split())
    return text, {"provider": "qwen3_asr", "model": settings.qwen_asr_model,
                  "language": str(getattr(results[0], "language", "") or "")}


# ============================================================
# DASHSCOPE (qwen3-asr-flash, OpenAI-compatible)
# ============================================================

def transcribe_api(audio_base64: str, *, context: str, settings: CaptionStackSettings,
                   create: Callable[..., Any] | None = None) -> tuple[str, dict[str, Any]]:
    from ai.caption_stack import qwen_omni

    if create is None:
        try:
            create = qwen_omni._client(settings).chat.completions.create
        except qwen_omni.QwenUnavailable as error:
            raise AsrUnavailable(str(error)) from None
    code = str(settings.language or "").replace("_", "-").split("-", 1)[0].casefold()
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": f"data:;base64,{audio_base64}", "format": "wav"}},
        ]},
    ]
    try:
        response = create(model=settings.qwen_asr_api_model, messages=messages, stream=False,
                          extra_body={"asr_options": {"enable_itn": False, **({"language": code} if code else {})}})
    except Exception as error:
        raise AsrUnavailable(f"{settings.qwen_asr_api_model} request error: {type(error).__name__}: "
                             f"{qwen_omni._redact(str(error), settings)}") from None
    choices = getattr(response, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None
    content = getattr(message, "content", "") if message is not None else ""
    if isinstance(content, list):                    # some compatible responses return content parts
        content = " ".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
    text = " ".join(str(content or "").split())
    if not text:
        raise AsrUnavailable(f"{settings.qwen_asr_api_model} returned no text")
    return text, {"provider": "qwen3_asr_flash", "model": settings.qwen_asr_api_model}


def listen(path: str, audio_base64: Callable[[], str], *, context: str, settings: CaptionStackSettings,
           provider: str = "") -> tuple[str, dict[str, Any]]:
    """One Qwen3-ASR listen -> (text, meta); ``provider`` defaults to the configured primary provider."""
    if (provider or settings.primary_provider) == "qwen3_asr_flash":
        return transcribe_api(audio_base64(), context=context, settings=settings)
    return transcribe_local(path, context=context, settings=settings)
