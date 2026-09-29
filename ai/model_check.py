from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import OpenAI

from ai import model_config


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIMIR model routing ve sistem gereksinimlerini kontrol eder."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Her benzersiz OpenAI modeline küçük bir gerçek API isteği gönder.",
    )
    args = parser.parse_args()

    print()
    print("MIMIR OPTIMAL V8 MODEL CHECK")
    print("=" * 60)
    print(f"OpenAI Python SDK: {openai.__version__}")
    print(f".env mevcut: {(PROJECT_ROOT / '.env').exists()}")

    model_config.validate_model_plan()
    model_config.print_model_plan()
    _check_external_tools()

    if args.live:
        _live_check()
    else:
        print()
        print("✅ Config ve sistem araçları kontrol edildi.")
        print("Gerçek API testi için: python -m ai.model_check --live")


if __name__ == "__main__":
    main()
