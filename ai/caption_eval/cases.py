"""Evaluation cases: the audio of one short, what MIMIR captioned, and the human gold.

One directory per case under ``<root>/cases/<case_id>/``:

    audio.wav              the caption stack's analysis audio (mono 16 kHz PCM16) of the short
    meta.json              language, verified names/entities, where the case came from
    draft.txt              what MIMIR captioned (read-only reference for the human)
    published_words.json   the captioned words with times on the case clock (the "published" system)
    gold.txt               THE HUMAN GOLD: starts as a copy of the draft with a TODO line; a person
                           listens to audio.wav, corrects it word for word and deletes the TODO line
    gold_words.json        optional human word times (or gold_words.auto.json from ``align-gold``)

``prepare`` builds a case from an edited clip (``--clip``) or from a published short
(``--final``: the story part of the published MP4, mapped through its final timeline;
the edited clip is usually deleted by the temp cleanup). Nothing here calls a model.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai.caption_eval import metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "vod_output" / "caption_eval"
GOLD_TODO = "# TODO"
GOLD_HEADER = (
    "# TODO: listen to audio.wav and correct this transcript WORD FOR WORD, then delete this TODO line.\n"
    "# Write exactly what is said (slang, profanity, repeats, unfinished words). Spell names correctly.\n"
    "# Fillers (uh/um), case and punctuation do not matter. Write [inaudible] where even you cannot tell.\n"
    "# Lines starting with # are ignored.\n"
)


def eval_root(root: str | Path | None = None) -> Path:
    return Path(root or os.getenv("MIMIR_CAPTION_EVAL_DIR", "") or DEFAULT_ROOT).resolve()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower()
    return value[:60] or "case"


@dataclass
class Case:
    case_id: str
    directory: Path
    meta: dict[str, Any]

    @property
    def audio_path(self) -> Path:
        return self.directory / "audio.wav"

    @property
    def gold_path(self) -> Path:
        return self.directory / "gold.txt"

    @property
    def language(self) -> str:
        return str(self.meta.get("language") or "en")

    @property
    def names(self) -> list[str]:
        return [str(x) for x in self.meta.get("verified_names", []) or [] if str(x).strip()]

    @property
    def entities(self) -> list[str]:
        return [str(x) for x in self.meta.get("entities", []) or [] if str(x).strip()]

    @property
    def duration(self) -> float:
        return float((self.meta.get("audio") or {}).get("duration", 0.0) or 0.0)

    @property
    def audio_sha256(self) -> str:
        return str((self.meta.get("audio") or {}).get("sha256", ""))

    def gold_status(self) -> str:
        """missing | todo (still the unedited draft) | empty | ready"""
        if not self.gold_path.is_file():
            return "missing"
        text = self.gold_path.read_text(encoding="utf-8-sig")
        if any(line.strip().upper().startswith(GOLD_TODO) for line in text.splitlines()):
            return "todo"
        return "ready" if metrics.gold_display_tokens(text) else "empty"

    def gold_text(self) -> str:
        return self.gold_path.read_text(encoding="utf-8-sig")

    def gold_times(self) -> tuple[list[tuple[float, float]] | None, str]:
        """Gold word times + their source (human | aligner), when they match the current gold words."""
        tokens = [t for t in metrics.gold_display_tokens(self.gold_text()) if t != metrics._WILDCARD_TOKEN]
        for name, source in (("gold_words.json", "human"), ("gold_words.auto.json", "aligner")):
            data = _read_json(self.directory / name)
            rows = data.get("words") if isinstance(data, Mapping) else data
            if not isinstance(rows, list) or len(rows) != len(tokens):
                continue
            try:
                times = [(float(r["start"]), float(r["end"])) for r in rows]
            except (KeyError, TypeError, ValueError):
                continue
            if [_plain(r.get("word", "")) for r in rows] != [_plain(t) for t in tokens]:
                continue                                   # stale: the gold text changed after timing
            return times, source
        return None, ""

    def published_words(self) -> list[dict[str, Any]] | None:
        data = _read_json(self.directory / "published_words.json")
        rows = data.get("words") if isinstance(data, Mapping) else None
        return rows if isinstance(rows, list) else None


def _plain(text: str) -> str:
    return "".join(ch for ch in str(text).casefold() if ch.isalnum())


def list_cases(root: str | Path | None = None, only: Sequence[str] | None = None) -> list[Case]:
    base = eval_root(root) / "cases"
    if not base.is_dir():
        return []
    wanted = {str(x) for x in only} if only else None
    cases: list[Case] = []
    for directory in sorted(p for p in base.iterdir() if p.is_dir()):
        meta = _read_json(directory / "meta.json")
        if not isinstance(meta, dict):
            continue
        case = Case(str(meta.get("case_id") or directory.name), directory, meta)
        if wanted is None or case.case_id in wanted:
            cases.append(case)
    return cases


# ============================================================
# PREPARE
# ============================================================

def _profile_words(profile: Mapping[str, Any], *, offset: float = 0.0, minimum: float = -1e9
                   ) -> list[dict[str, Any]]:
    """Captioned words of a speaker profile on the case clock (``paced - offset``)."""
    rows: list[dict[str, Any]] = []
    for word in profile.get("words", []) or []:
        if not isinstance(word, Mapping):
            continue
        try:
            start, end = float(word.get("edited_start")), float(word.get("edited_end"))
        except (TypeError, ValueError):
            continue
        if start < minimum - 1e-3:
            continue
        rows.append({"word": str(word.get("word", "")), "start": round(start - offset, 3),
                     "end": round(end - offset, 3), "status": str(word.get("lexical_status", "")),
                     "uncertain": bool(word.get("caption_uncertain")) or None})
    return rows


def _profile_names(profile: Mapping[str, Any]) -> list[str]:
    quality = profile.get("caption_quality") if isinstance(profile.get("caption_quality"), Mapping) else {}
    lexical = quality.get("lexical") if isinstance(quality.get("lexical"), Mapping) else {}
    return [str(x) for x in lexical.get("verified_terms", []) or [] if str(x).strip()]


def _profile_language(profile: Mapping[str, Any]) -> str:
    quality = profile.get("caption_quality") if isinstance(profile.get("caption_quality"), Mapping) else {}
    settings = quality.get("settings") if isinstance(quality.get("settings"), Mapping) else {}
    return str(settings.get("language") or "")


def discover_profile(edited_clip: str | Path) -> Path | None:
    """The final speaker profile the pipeline wrote for an edited clip (newest profile version)."""
    from ai.editor import speaker_caption_support as scs

    clip = Path(edited_clip)
    match = re.match(r"^clip_(\d+)_", clip.name)
    if not match:
        return None
    directory = scs.SPEAKER_OUTPUT_DIR / scs._safe_name(clip.parent.name)
    found = sorted(directory.glob(f"clip_{int(match.group(1)):02d}_speakers_v*.json"),
                   key=lambda p: (int(re.findall(r"_v(\d+)", p.stem)[-1]) if re.findall(r"_v(\d+)", p.stem) else 0,
                                  p.stat().st_mtime))
    return found[-1] if found else None


def discover_timeline(published: str | Path, search_root: str | Path | None = None) -> Path | None:
    """The final timeline document of a published short (``<stem>_short.mp4``), newest first."""
    published = Path(published)
    stem = published.stem[:-len("_short")] if published.stem.endswith("_short") else published.stem
    wanted = slug(stem)
    root = Path(search_root) if search_root else PROJECT_ROOT / "vod_output"
    matches: list[Path] = []
    for path in root.rglob("*.timeline.json"):
        data = _read_json(path)
        if not isinstance(data, Mapping) or data.get("kind") != "mimir_final_timeline":
            continue
        source = Path(str((data.get("intro") or {}).get("source_path", "")))
        if slug(source.parent.name) == wanted:
            matches.append(path)
    matches.sort(key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _extract(media: Path, workdir: Path) -> Any:
    from ai.caption_stack import audio as analysis_audio

    return analysis_audio.extract_analysis_audio(media, directory=workdir)


def _write_case(directory: Path, audio_path: Path, *, meta: dict[str, Any], words: list[dict[str, Any]] | None,
                force: bool) -> None:
    from ai.caption_stack import audio as analysis_audio

    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "audio.wav"
    if target.exists() and not force:
        raise FileExistsError(f"case already exists: {directory} (use --force to rebuild audio/draft; "
                              "gold.txt is never overwritten)")
    shutil.copyfile(audio_path, target)
    loaded = analysis_audio.load_analysis_audio(target)
    meta["audio"] = {"duration": round(loaded.duration, 3), "sha256": loaded.sha256, "sample_rate": loaded.sample_rate}
    _write_json(directory / "meta.json", meta)
    text = " ".join(w["word"] for w in words or [] if str(w.get("word", "")).strip())
    if words is not None:
        _write_json(directory / "published_words.json", {"source": meta.get("source"), "words": words})
        (directory / "draft.txt").write_text(textwrap.fill(text, 100) + "\n", encoding="utf-8")
    gold = directory / "gold.txt"
    if not gold.exists():                                   # a human's gold is never overwritten
        body = textwrap.fill(text, 100) if text else "(write the transcript here)"
        gold.write_text(GOLD_HEADER + body + "\n", encoding="utf-8")


def prepare_from_clip(clip: str | Path, *, root: str | Path | None = None, case_id: str | None = None,
                      profile: str | Path | None = None, names: Sequence[str] = (), entities: Sequence[str] = (),
                      language: str | None = None, force: bool = False) -> Case:
    """A case from an edited clip (or any audio/video file); the draft comes from its speaker profile."""
    clip = Path(clip).resolve()
    if not clip.is_file():
        raise FileNotFoundError(f"clip not found: {clip}")
    profile_path = Path(profile).resolve() if profile else discover_profile(clip)
    data = _read_json(profile_path) if profile_path else None
    data = data if isinstance(data, Mapping) else {}
    case_id = case_id or f"{slug(clip.parent.name)}__{slug(clip.stem)}"[:80]
    directory = eval_root(root) / "cases" / case_id
    meta = {"case_id": case_id, "created_at": _now(),
            "language": language or _profile_language(data) or os.getenv("CAPTION_LANGUAGE", "") or "en",
            "verified_names": list(names) or _profile_names(data), "entities": list(entities),
            "source": {"kind": "clip", "clip": str(clip), "profile": str(profile_path) if profile_path else None,
                       "clock": "edited clip"},
            "notes": ""}
    with tempfile.TemporaryDirectory(prefix="mimir_eval_") as tmp:
        audio = _extract(clip, Path(tmp))
        _write_case(directory, audio.path, meta=meta, words=_profile_words(data) if data else None, force=force)
    return Case(case_id, directory, _read_json(directory / "meta.json"))


def prepare_from_final(published: str | Path, *, root: str | Path | None = None, case_id: str | None = None,
                       timeline: str | Path | None = None, profile: str | Path | None = None,
                       names: Sequence[str] = (), entities: Sequence[str] = (), language: str | None = None,
                       force: bool = False) -> Case:
    """A case from a published short: the story part of its audio, on the paced clock minus the restart."""
    from ai.caption_stack import audio as analysis_audio

    published = Path(published).resolve()
    if not published.is_file():
        raise FileNotFoundError(f"published short not found: {published}")
    timeline_path = Path(timeline).resolve() if timeline else discover_timeline(published)
    doc = _read_json(timeline_path) if timeline_path else None
    if not isinstance(doc, Mapping):
        raise FileNotFoundError(f"no final timeline (*.timeline.json) found for {published.name}; pass --timeline, "
                                "or prepare from the edited clip with --clip (run the pipeline with --keep-temp)")
    intro = float((doc.get("intro") or {}).get("duration", 0.0) or 0.0)
    restart = float((doc.get("restart") or {}).get("main_restart_paced", 0.0) or 0.0)
    edited = Path(str((doc.get("intro") or {}).get("source_path", "")))
    profile_path = Path(profile).resolve() if profile else discover_profile(edited)
    data = _read_json(profile_path) if profile_path else None
    data = data if isinstance(data, Mapping) else {}
    case_id = case_id or slug(published.stem)
    directory = eval_root(root) / "cases" / case_id
    meta = {"case_id": case_id, "created_at": _now(),
            "language": language or _profile_language(data) or os.getenv("CAPTION_LANGUAGE", "") or "en",
            "verified_names": list(names) or _profile_names(data), "entities": list(entities),
            "source": {"kind": "final", "published": str(published), "timeline": str(timeline_path),
                       "profile": str(profile_path) if profile_path else None, "intro_duration": intro,
                       "restart_paced": restart,
                       "clock": "published story part: case t = paced t - restart (effects audio included)"},
            "notes": ""}
    with tempfile.TemporaryDirectory(prefix="mimir_eval_") as tmp:
        full = _extract(published, Path(tmp))
        story, _a, _b = analysis_audio.cut_window(full, intro, full.duration, Path(tmp) / "story.wav")
        words = _profile_words(data, offset=restart, minimum=restart) if data else None
        _write_case(directory, story, meta=meta, words=words, force=force)
    return Case(case_id, directory, _read_json(directory / "meta.json"))
