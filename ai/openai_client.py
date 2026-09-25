"""Shared, lazy OpenAI client for the MIMIR Shorts pipeline."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH, override=False)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Create the OpenAI client only when an API call is actually needed."""
    global _client

    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY bulunamadı. .env dosyasını kontrol et."
        )

    _client = OpenAI(api_key=api_key)
    return _client


class _LazyOpenAIProxy:
    """Keep existing ``client.responses`` / ``client.audio`` call sites simple."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_client(), name)


client = _LazyOpenAIProxy()
