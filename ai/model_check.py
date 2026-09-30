from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import OpenAI

from ai import model_config
from ai.caption_stack import config as caption_stack_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _check_external_tools() -> None:
    print()
    print("🧰 EXTERNAL TOOLS")
    print("-" * 60)
    missing: list[str] = []
    for command in ("ffmpeg", "ffprobe", "curl"):
        path = shutil.which(command)
        if path:
            print(f"✅ {command}: {path}")
        else:
            print(f"❌ {command}: bulunamadı")
            missing.append(command)
    if missing:
        raise RuntimeError("Eksik sistem aracı: " + ", ".join(missing))


def _print_caption_stack() -> None:
    """Final-caption routing: WHAT (Qwen ears) and WHEN (word aligner). Never prints a key."""
    import importlib.util

    settings = caption_stack_config.load_settings()
    print()
    print("🎙️ FINAL CAPTION STACK")
    print("-" * 60)
    print(f"LEXICAL EAR       : {settings.primary_provider} -> {settings.qwen_model} "
          f"[reasoning {settings.qwen_reasoning_effort}] (DashScope key/url set: {settings.qwen_configured})")
    print(f"TRANSCRIBE BACKUP : {settings.transcribe_fallback_provider} -> {settings.transcribe_fallback_model}")
    print(f"WORD ALIGNMENT    : {settings.alignment_provider} -> {settings.aligner_model} "
          f"[device {settings.aligner_device}, dtype {settings.aligner_dtype}]")
    print(f"ALIGNMENT BACKUP  : {settings.alignment_fallback_provider}")
    installed = importlib.util.find_spec("qwen_asr") is not None
    print(f"qwen-asr installed: {installed} (the aligner loads lazily, only for final caption timing)")


def _live_qwen_check() -> None:
    settings = caption_stack_config.load_settings()
    if settings.primary_provider != "qwen_omni":
        return
    if not settings.qwen_configured:
        raise RuntimeError("DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL bulunamadı (final caption Qwen ear).")
    client = OpenAI(api_key=settings.dashscope_api_key, base_url=settings.dashscope_base_url,
                    timeout=settings.qwen_timeout_s, max_retries=settings.qwen_max_retries)
    print(f"⏳ {settings.qwen_model} (DashScope) ...", end=" ", flush=True)
    try:
        stream = client.chat.completions.create(
            model=settings.qwen_model, messages=[{"role": "user", "content": "Reply exactly with: OK"}],
            modalities=["text"], reasoning_effort=settings.qwen_reasoning_effort, stream=True)
        text = "".join(str(getattr(getattr(c, "delta", None), "content", "") or "")
                       for chunk in stream for c in (getattr(chunk, "choices", None) or []))
        print("✅" if text.strip() else "✅ response received")
    except Exception as error:
        print("❌")
        message = str(error).replace(settings.dashscope_api_key, "***")[:300]
        raise RuntimeError(f"{settings.qwen_model} live check başarısız: {type(error).__name__}: {message}") from None


def _live_check() -> None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY bulunamadı. .env dosyasını kontrol et.")

    client = OpenAI(api_key=api_key)
    plan = model_config.model_plan()
    unique_models: list[str] = []

    for role in plan.values():
        model = role["model"]
        if model not in unique_models:
            unique_models.append(model)

    print()
    print("🌐 LIVE API MODEL CHECK")
    print("-" * 60)

    for model in unique_models:
        print(f"⏳ {model} ...", end=" ", flush=True)
        try:
            response = client.responses.create(
                model=model,
                reasoning={"effort": "low"},
                input="Reply exactly with: OK",
            )
            print("✅" if (response.output_text or "").strip() else "✅ response received")
        except TypeError as error:
            print("❌")
            raise RuntimeError(
                "OpenAI Python SDK reasoning parametresini kabul etmiyor. "
                "Önce: python -m pip install -U openai"
            ) from error
        except Exception as error:
            print("❌")
            raise RuntimeError(f"{model} live check başarısız: {error}") from error
    _live_qwen_check()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIMIR model routing ve sistem gereksinimlerini kontrol eder."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Her benzersiz OpenAI modeline ve Qwen Omni'ye küçük bir gerçek API isteği gönder.",
    )
    args = parser.parse_args()

    print()
    print("MIMIR OPTIMAL V8 MODEL CHECK")
    print("=" * 60)
    print(f"OpenAI Python SDK: {openai.__version__}")
    print(f".env mevcut: {(PROJECT_ROOT / '.env').exists()}")

    model_config.validate_model_plan()
    model_config.print_model_plan()
    _print_caption_stack()
    _check_external_tools()

    if args.live:
        _live_check()
    else:
        print()
        print("✅ Config ve sistem araçları kontrol edildi.")
        print("Gerçek API testi için: python -m ai.model_check --live")


if __name__ == "__main__":
    main()
