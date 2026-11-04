"""Variants: what the clip evaluation can run, defined as environment overrides only (no code change).

    baseline                        what ``python main.py`` runs today: the code defaults plus your .env
    <root>/variants/<name>.env      ``KEY=VALUE`` lines applied on top of that (``#`` starts a comment)

Example ``vod_output/clip_eval/variants/judge-high.env``:

    # Terra judges the candidates at high effort from the start
    MIMIR_CLIP_JUDGE_REASONING=high

Keys Stage 2 (story / clip selection) reads, with their defaults:

    MIMIR_CLIP_SCOUT_MODEL=gpt-5.6-luna            MIMIR_CLIP_SCOUT_REASONING=medium
    MIMIR_CLIP_JUDGE_MODEL=<MIMIR_EDITOR_MODEL>     MIMIR_CLIP_JUDGE_REASONING=medium
    MIMIR_EDITOR_MODEL=gpt-5.6-terra                MIMIR_CLIP_JUDGE_ESCALATION_REASONING=high
    MIMIR_CLIP_HIGH_ESCALATION_MARGIN=0.35          MIMIR_STORY_VISUAL_ORIGIN=1
    MIMIR_BOUNDARY_POLISH_MAX_EXTEND=3.0            MIMIR_BOUNDARY_POLISH_MAX_TRIM=1.5
    MIMIR_SOURCE_PEAK_MIN_AUDIO_SCORE=0.82          MIMIR_SOURCE_PEAK_MIN_PEAK_DBFS=-13.0
    MIMIR_SOURCE_PEAK_MAX=2

Model routing is read when ``ai.model_config`` is imported, so every variant runs in its own
subprocess (``ai.clip_eval.worker``). Secrets are refused: they stay in .env and are never recorded.
With ``--stage full`` any pipeline key applies (captions, editing, Video Brain, ...).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ai.caption_eval.runner import env_from_pairs
from ai.clip_eval import cases as case_store

VARIANT_HELP = __doc__
BASELINE = "baseline"
STAGE2_KEYS = frozenset({
    "MIMIR_CLIP_SCOUT_MODEL", "MIMIR_CLIP_SCOUT_REASONING", "MIMIR_CLIP_JUDGE_MODEL", "MIMIR_CLIP_JUDGE_REASONING",
    "MIMIR_CLIP_JUDGE_ESCALATION_REASONING", "MIMIR_EDITOR_MODEL", "MIMIR_CLIP_HIGH_ESCALATION_MARGIN",
    "MIMIR_BOUNDARY_POLISH_MAX_EXTEND", "MIMIR_BOUNDARY_POLISH_MAX_TRIM", "MIMIR_STORY_VISUAL_ORIGIN",
    "MIMIR_SOURCE_PEAK_MIN_AUDIO_SCORE", "MIMIR_SOURCE_PEAK_MIN_PEAK_DBFS", "MIMIR_SOURCE_PEAK_MAX",
})
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


@dataclass(frozen=True)
class Variant:
    name: str
    env: dict[str, str]
    path: Path | None = None

    @property
    def digest(self) -> str:
        blob = "\n".join(f"{k}={v}" for k, v in sorted(self.env.items()))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def ignored_keys(self, stage: str) -> list[str]:
        """Overrides that cannot change a ``--stage clip`` run (Stage 2 never reads them)."""
        return [] if stage == "full" else sorted(k for k in self.env if k not in STAGE2_KEYS)


def variants_dir(root: str | Path | None = None) -> Path:
    return case_store.eval_root(root) / "variants"


def parse_env_text(text: str, *, source: str = "variant") -> dict[str, str]:
    """``KEY=VALUE`` lines (``#`` comments, blank lines, optional ``export``, optional quotes)."""
    pairs: list[str] = []
    for number, raw in enumerate(str(text).replace("\r\n", "\n").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            raise ValueError(f"{source} line {number}: expected KEY=VALUE, got {raw!r}")
        key, value = line.split("=", 1)
        value = value.strip()
        if value[:1] in ("'", '"'):                            # quoted: up to the closing quote
            closing = value.find(value[0], 1)
            if closing < 0:
                raise ValueError(f"{source} line {number}: unclosed quote in {raw!r}")
            value = value[1:closing]
        elif " #" in value:                                    # unquoted: a ' #' starts a comment
            value = value.split(" #", 1)[0].strip()
        pairs.append(f"{key.strip()}={value}")
    try:
        return env_from_pairs(pairs)
    except ValueError as error:
        raise ValueError(f"{source}: {error}") from None


def load(name: str, root: str | Path | None = None) -> Variant:
    if not _NAME.fullmatch(str(name or "")):
        raise ValueError(f"variant name {name!r}: use letters, digits, '.', '_' or '-'")
    if name == BASELINE:
        return Variant(BASELINE, {})
    path = variants_dir(root) / f"{name}.env"
    if not path.is_file():
        known = ", ".join(v.name for v in list_variants(root))
        raise FileNotFoundError(f"no variant {name!r}: create {path} (KEY=VALUE lines); known: {known}")
    return Variant(name, parse_env_text(path.read_text(encoding="utf-8-sig"), source=str(path)), path)


def list_variants(root: str | Path | None = None) -> list[Variant]:
    found = [Variant(BASELINE, {})]
    folder = variants_dir(root)
    if folder.is_dir():
        for path in sorted(folder.glob("*.env")):
            if path.stem == BASELINE or not _NAME.fullmatch(path.stem):
                continue
            try:
                found.append(load(path.stem, root))
            except ValueError as error:
                found.append(Variant(path.stem, {"!error": str(error)}, path))
    return found


def describe(variants: list[Variant]) -> str:
    lines = [VARIANT_HELP.strip(), "", "Defined here:"]
    for variant in variants:
        if "!error" in variant.env:
            lines.append(f"  {variant.name:<24} INVALID: {variant.env['!error']}")
            continue
        body = ", ".join(f"{k}={v}" for k, v in sorted(variant.env.items())) or "(no overrides)"
        lines.append(f"  {variant.name:<24} {body}")
        ignored = variant.ignored_keys("clip")
        if ignored:
            lines.append(f"  {'':<24} note: {', '.join(ignored)} only matter(s) with --stage full")
    return "\n".join(lines)


def merged_env(base: Mapping[str, str], variant: Variant) -> dict[str, str]:
    env = dict(base)
    env.update(variant.env)
    return env
