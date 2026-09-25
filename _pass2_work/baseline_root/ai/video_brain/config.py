from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError as error:
    raise RuntimeError(
        "python-dotenv kurulu değil.\n"
        "Aktif .venv içinde çalıştır:\n"
        "pip install -U python-dotenv"
    ) from error


# ============================================================
# PATHS
# ============================================================

# ai/video_brain/config.py
# parents[0] = video_brain
# parents[1] = ai
# parents[2] = mimir
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(
    dotenv_path=ENV_PATH,
    override=False,
)


# ============================================================
# HELPERS
# ============================================================

def _env_bool(
    name: str,
    default: bool = False,
) -> bool:

    raw = os.getenv(
        name,
        "true" if default else "false",
    )

    return (
        str(raw)
        .strip()
        .lower()
        in {
            "1",
            "true",
            "yes",
            "on",
            "y",
            "evet",
        }
    )


def _env_float(
    name: str,
    default: float,
) -> float:

    raw = os.getenv(
        name,
        str(default),
    )

    try:
        return float(
            raw
        )
    except (
        TypeError,
        ValueError,
    ):
        return float(
            default
        )


# ============================================================
# VIDEO BRAIN FEATURE FLAG
# ============================================================

# Support-only layer.
# Existing MIMIR pipeline stays independent from this.
VIDEO_BRAIN_ENABLED = _env_bool(
    "VIDEO_BRAIN_ENABLED",
    True,
)


# ============================================================
# GEMINI
# ============================================================

GEMINI_API_KEY = (
    os.getenv(
        "GEMINI_API_KEY",
        "",
    )
    .strip()
)

# Read model from .env.
# If missing, use the V8 default.
VIDEO_BRAIN_MODEL = (
    os.getenv(
        "VIDEO_BRAIN_MODEL",
        "gemini-3.7-flash",
    )
    .strip()
    or "gemini-3.7-flash"
)


# ============================================================
# FILE / VIDEO PROCESSING
# ============================================================

VIDEO_UPLOAD_POLL_SECONDS = _env_float(
    "VIDEO_UPLOAD_POLL_SECONDS",
    3.0,
)

VIDEO_UPLOAD_TIMEOUT_SECONDS = _env_float(
    "VIDEO_UPLOAD_TIMEOUT_SECONDS",
    300.0,
)


# ============================================================
# OUTPUT
# ============================================================

VIDEO_BRAIN_OUTPUT_DIR = (
    PROJECT_ROOT
    / "vod_output"
    / "video_brain"
)


# ============================================================
# VALIDATION
# ============================================================

def validate_config() -> None:

    if not VIDEO_BRAIN_ENABLED:

        raise RuntimeError(
            "Video Brain devre dışı.\n"
            ".env:\n"
            "VIDEO_BRAIN_ENABLED=true"
        )

    if not ENV_PATH.exists():

        raise RuntimeError(
            f".env dosyası bulunamadı:\n{ENV_PATH}"
        )

    if not GEMINI_API_KEY:

        raise RuntimeError(
            "GEMINI_API_KEY bulunamadı.\n\n"
            f"Kontrol edilen .env:\n{ENV_PATH}\n\n"
            "Şu satırı ekle:\n"
            "GEMINI_API_KEY=BURAYA_KEY"
        )

    if not VIDEO_BRAIN_MODEL:

        raise RuntimeError(
            "VIDEO_BRAIN_MODEL boş."
        )

    if VIDEO_UPLOAD_POLL_SECONDS <= 0:

        raise RuntimeError(
            "VIDEO_UPLOAD_POLL_SECONDS 0'dan büyük olmalı."
        )

    if VIDEO_UPLOAD_TIMEOUT_SECONDS <= 0:

        raise RuntimeError(
            "VIDEO_UPLOAD_TIMEOUT_SECONDS 0'dan büyük olmalı."
        )


# ============================================================
# DEBUG
# ============================================================

def debug_config() -> None:

    print()
    print(
        "MIMIR Video Brain Config"
    )
    print(
        f"PROJECT_ROOT: {PROJECT_ROOT}"
    )
    print(
        f"ENV_PATH: {ENV_PATH}"
    )
    print(
        f"ENV exists: {ENV_PATH.exists()}"
    )
    print(
        f"VIDEO_BRAIN_ENABLED: {VIDEO_BRAIN_ENABLED}"
    )
    print(
        f"VIDEO_BRAIN_MODEL: {VIDEO_BRAIN_MODEL}"
    )
    print(
        "GEMINI_API_KEY: "
        + (
            "SET ✅"
            if GEMINI_API_KEY
            else "MISSING ❌"
        )
    )
    print(
        f"UPLOAD POLL: {VIDEO_UPLOAD_POLL_SECONDS}s"
    )
    print(
        f"UPLOAD TIMEOUT: {VIDEO_UPLOAD_TIMEOUT_SECONDS}s"
    )


if __name__ == "__main__":

    try:

        debug_config()
        validate_config()

        print()
        print(
            "✅ VIDEO BRAIN CONFIG OK"
        )

    except Exception as error:

        print()
        print(
            "❌ VIDEO BRAIN CONFIG HATASI:"
        )
        print(
            error
        )

        raise SystemExit(
            1
        )
