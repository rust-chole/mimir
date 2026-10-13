"""Centralized settings for the final-caption stack.

Every operational choice is read from the environment (``.env`` via
``ai.model_config``) at call time, so a test or a long-running process sees
the current value. Defaults are model IDs only; endpoints, keys, devices and
paths are never assumed. ``public()`` is the only view that may be logged or
hashed into a cache signature: it never contains the API key.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from ai import model_config  # noqa: F401  (loads .env once, like every other MIMIR module)

DEFAULT_QWEN_OMNI_MODEL = "qwen3.8-omni-flash"
DEFAULT_QWEN_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
DEFAULT_TRANSCRIBE_FALLBACK_MODEL = "gpt-transcribe"
DEFAULT_QWEN_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_QWEN_ASR_API_MODEL = "qwen3-asr-flash"

# qwen3_asr / qwen3_asr_flash: Qwen3-ASR ears (local / DashScope); measured with ai.caption_eval first.
PRIMARY_PROVIDERS = ("qwen_omni", "openai_transcribe", "qwen3_asr", "qwen3_asr_flash")
TRANSCRIBE_FALLBACK_PROVIDERS = ("openai_transcribe", "none")
ALIGNMENT_PROVIDERS = ("qwen3_forced_aligner", "legacy_whisper")
ALIGNMENT_FALLBACK_PROVIDERS = ("legacy_whisper", "none")
DEVICES = ("auto", "cpu", "cuda", "mps")
DTYPES = ("auto", "float32", "float16", "bfloat16")
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
COVERAGE_CHECKS = ("vad", "off")

# The official Qwen3-ForcedAligner accepts at most this much audio per input
# (qwen_asr.inference.utils.MAX_FORCE_ALIGN_INPUT_SECONDS). Longer shorts are
# split deterministically at low-energy points; the loaded provider re-checks
# the installed package's own limit.
ALIGNER_MAX_INPUT_SECONDS = 180.0


class CaptionStackConfigError(ValueError):
    """An operational setting has a value no code path supports."""


def _text(name: str, default: str = "") -> str:
    return " ".join(str(os.getenv(name, default) or "").split()) or default


def _choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = _text(name, default).lower()
    base = value
    if name.endswith("_DEVICE") and value.startswith("cuda:") and value[5:].isdigit():
        base = "cuda"                        # an explicit, user-chosen CUDA ordinal
    if base not in allowed:
        raise CaptionStackConfigError(f"{name}={value!r} is not supported; use one of: {', '.join(allowed)}")
    return value


def _number(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = _text(name, "")
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise CaptionStackConfigError(f"{name}={raw!r} is not a number") from error
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class CaptionStackSettings:
    primary_provider: str
    qwen_model: str
    qwen_reasoning_effort: str
    qwen_timeout_s: float
    qwen_max_retries: int
    dashscope_base_url: str
    transcribe_fallback_provider: str
    transcribe_fallback_model: str
    alignment_provider: str
    alignment_fallback_provider: str
    aligner_model: str
    aligner_device: str
    aligner_dtype: str
    language: str
    dashscope_api_key: str = field(default="", repr=False)
    coverage_check: str = "vad"              # vad | off (report-only acoustic coverage flags)
    qwen_asr_model: str = DEFAULT_QWEN_ASR_MODEL
    qwen_asr_api_model: str = DEFAULT_QWEN_ASR_API_MODEL
    qwen_asr_device: str = "auto"
    qwen_asr_dtype: str = "auto"

    @property
    def qwen_configured(self) -> bool:
        return bool(self.dashscope_api_key and self.dashscope_base_url)

    def public(self) -> dict[str, Any]:
        """Loggable / hashable view: every setting except the secret."""
        return {
            "primary_provider": self.primary_provider,
            "qwen_model": self.qwen_model,
            "qwen_reasoning_effort": self.qwen_reasoning_effort,
            "qwen_timeout_s": self.qwen_timeout_s,
            "qwen_max_retries": self.qwen_max_retries,
            "dashscope_base_url_set": bool(self.dashscope_base_url),
            "dashscope_api_key_set": bool(self.dashscope_api_key),
            "transcribe_fallback_provider": self.transcribe_fallback_provider,
            "transcribe_fallback_model": self.transcribe_fallback_model,
            "alignment_provider": self.alignment_provider,
            "alignment_fallback_provider": self.alignment_fallback_provider,
            "aligner_model": self.aligner_model,
            "aligner_device": self.aligner_device,
            "aligner_dtype": self.aligner_dtype,
            "language": self.language,
            "coverage_check": self.coverage_check,
            "qwen_asr_model": self.qwen_asr_model,
            "qwen_asr_api_model": self.qwen_asr_api_model,
            "qwen_asr_device": self.qwen_asr_device,
            "qwen_asr_dtype": self.qwen_asr_dtype,
        }


def load_settings() -> CaptionStackSettings:
    """Current settings; raises ``CaptionStackConfigError`` on unsupported values."""
    alignment = _choice("MIMIR_WORD_ALIGNMENT_PROVIDER", "qwen3_forced_aligner", ALIGNMENT_PROVIDERS)
    alignment_fallback = _choice("MIMIR_WORD_ALIGNMENT_FALLBACK_PROVIDER", "legacy_whisper",
                                 ALIGNMENT_FALLBACK_PROVIDERS)
    if alignment_fallback == alignment:
        alignment_fallback = "none"          # a provider is never its own fallback
    return CaptionStackSettings(
        primary_provider=_choice("MIMIR_CAPTION_PRIMARY_PROVIDER", "qwen_omni", PRIMARY_PROVIDERS),
        qwen_model=_text("MIMIR_QWEN_OMNI_MODEL", DEFAULT_QWEN_OMNI_MODEL),
        qwen_reasoning_effort=_choice("MIMIR_QWEN_REASONING_EFFORT", "none", REASONING_EFFORTS),
        qwen_timeout_s=_number("MIMIR_QWEN_TIMEOUT_SECONDS", 180.0, 10.0, 1800.0),
        qwen_max_retries=int(_number("MIMIR_QWEN_MAX_RETRIES", 1, 0, 5)),
        dashscope_base_url=_text("DASHSCOPE_BASE_URL", "").rstrip("/"),
        dashscope_api_key=str(os.getenv("DASHSCOPE_API_KEY", "") or "").strip(),
        transcribe_fallback_provider=_choice("MIMIR_CAPTION_TRANSCRIBE_FALLBACK_PROVIDER", "openai_transcribe",
                                             TRANSCRIBE_FALLBACK_PROVIDERS),
        transcribe_fallback_model=_text("MIMIR_CAPTION_TRANSCRIBE_FALLBACK_MODEL", DEFAULT_TRANSCRIBE_FALLBACK_MODEL),
        alignment_provider=alignment,
        alignment_fallback_provider=alignment_fallback,
        aligner_model=_text("MIMIR_QWEN_ALIGNER_MODEL", DEFAULT_QWEN_ALIGNER_MODEL),
        aligner_device=_choice("MIMIR_QWEN_ALIGNER_DEVICE", "auto", DEVICES),
        aligner_dtype=_choice("MIMIR_QWEN_ALIGNER_DTYPE", "auto", DTYPES),
        language=_text("CAPTION_LANGUAGE", _text("MIMIR_TRANSCRIPTION_LANGUAGE", "en")),
        coverage_check=_choice("MIMIR_CAPTION_COVERAGE_CHECK", "vad", COVERAGE_CHECKS),
        qwen_asr_model=_text("MIMIR_QWEN_ASR_MODEL", DEFAULT_QWEN_ASR_MODEL),
        qwen_asr_api_model=_text("MIMIR_QWEN_ASR_API_MODEL", DEFAULT_QWEN_ASR_API_MODEL),
        qwen_asr_device=_choice("MIMIR_QWEN_ASR_DEVICE", "auto", DEVICES),
        qwen_asr_dtype=_choice("MIMIR_QWEN_ASR_DTYPE", "auto", DTYPES),
    )


# ============================================================
# LANGUAGE
# ============================================================

# ISO 639-1 -> the language names Qwen3-ASR recognizes (qwen_asr SUPPORTED_LANGUAGES).
_ASR_LANGUAGES = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic", "de": "German", "fr": "French",
    "es": "Spanish", "pt": "Portuguese", "id": "Indonesian", "it": "Italian", "ko": "Korean", "ru": "Russian",
    "th": "Thai", "vi": "Vietnamese", "ja": "Japanese", "tr": "Turkish", "hi": "Hindi", "ms": "Malay",
    "nl": "Dutch", "sv": "Swedish", "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "fil": "Filipino", "tl": "Filipino", "fa": "Persian", "el": "Greek", "ro": "Romanian", "hu": "Hungarian",
    "mk": "Macedonian",
}
# The forced aligner is trained on only 11 of them (Qwen3-ASR README / technical report).
# qwen_asr does not reject the others: it would time them with an untrained model, so
# MIMIR refuses them here and the configured fallback clock times the transcript instead.
_ALIGNER_LANGUAGES = {code: name for code, name in _ASR_LANGUAGES.items()
                      if name in {"Chinese", "English", "Cantonese", "French", "German", "Italian", "Japanese",
                                  "Korean", "Portuguese", "Russian", "Spanish"}}


def _language_name(language: str, table: dict[str, str], what: str) -> str:
    value = str(language or "").strip()
    if not value:
        raise CaptionStackConfigError("CAPTION_LANGUAGE is empty")
    code = value.replace("_", "-").split("-", 1)[0].casefold()
    if code in table:
        return table[code]
    named = value[:1].upper() + value[1:].lower()
    if named in set(table.values()):
        return named
    raise CaptionStackConfigError(f"{what} does not support language {value!r} "
                                  f"(supported: {', '.join(sorted(set(table.values())))})")


def aligner_language(language: str) -> str:
    """``en`` / ``en-US`` / ``English`` -> ``English`` (the aligner's language names)."""
    return _language_name(language, _ALIGNER_LANGUAGES, "the Qwen forced aligner")


def asr_language(language: str) -> str:
    """The Qwen3-ASR language name of ``language`` (30 languages)."""
    return _language_name(language, _ASR_LANGUAGES, "Qwen3-ASR")


# ============================================================
# DEVICE / DTYPE (resolved only when the aligner is actually loaded)
# ============================================================

def resolve_device(requested: str, torch: Any) -> tuple[str, str]:
    """(device, note). ``auto`` prefers CUDA, then CPU; a missing CUDA runtime never crashes."""
    requested = str(requested or "auto").strip().lower()
    cuda_ok = bool(getattr(getattr(torch, "cuda", None), "is_available", lambda: False)())
    if requested == "auto":
        return ("cuda", "auto: CUDA available") if cuda_ok else ("cpu", "auto: CUDA unavailable, using CPU")
    if requested.startswith("cuda"):
        return (requested, "") if cuda_ok else ("cpu", f"{requested} requested but CUDA is unavailable; using CPU")
    if requested == "mps":
        backend = getattr(getattr(torch, "backends", None), "mps", None)
        available = bool(getattr(backend, "is_available", lambda: False)()) if backend is not None else False
        return ("mps", "") if available else ("cpu", "mps requested but unavailable; using CPU")
    return "cpu", ""


def resolve_dtype(requested: str, device: str, torch: Any) -> tuple[Any, str]:
    """(torch dtype, name). ``auto``: bf16 on CUDA when supported, else fp16 on CUDA, fp32 elsewhere."""
    requested = str(requested or "auto").strip().lower()
    if requested == "auto":
        if device.startswith("cuda"):
            bf16 = getattr(getattr(torch, "cuda", None), "is_bf16_supported", lambda: False)
            name = "bfloat16" if bool(bf16()) else "float16"
        else:
            name = "float32"
    else:
        name = requested
    return getattr(torch, name), name
