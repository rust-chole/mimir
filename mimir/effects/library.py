"""User-curated local SFX / reaction library (category folders)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from mimir.core.jsonio import content_hash

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"}
IMAGE_EXTENSIONS = {".png", ".webp"}
ACCENT_CATEGORIES = ("reversal", "disbelief", "confusion", "absurdity", "fail", "awkward", "hype", "fear_shock",
                     "wholesome_ironic", "impact")
TRANSITION_CATEGORY = "transition"


def scan(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    audio: dict[str, list[str]] = {}
    visual: dict[str, list[str]] = {}
    if root.is_dir():
        for category in (*ACCENT_CATEGORIES, TRANSITION_CATEGORY):
            folder = root / category
            if folder.is_dir():
                files = sorted(str(p.resolve()) for p in folder.iterdir()
                               if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
                if files:
                    audio[category] = files
            vfolder = root / "visual" / category
            if vfolder.is_dir():
                files = sorted(str(p.resolve()) for p in vfolder.iterdir()
                               if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
                if files:
                    visual[category] = files
    return {"root": str(root), "audio": audio, "visual": visual}


def fingerprint(root: str | Path) -> str:
    library = scan(root)
    rows = []
    for kind in ("audio", "visual"):
        for category, files in library[kind].items():
            for name in files:
                stat = Path(name).stat()
                rows.append([kind, category, Path(name).name, stat.st_size, stat.st_mtime_ns])
    return content_hash(rows)


def pick(files: list[str], seed: str) -> str:
    """Deterministic choice (stable across runs for the same story)."""
    if not files:
        raise ValueError("empty category")
    index = int(content_hash(seed)[:8], 16) % len(files)
    return files[index]
