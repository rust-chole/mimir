from __future__ import annotations

import copy
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VOD_OUTPUT_DIR = PROJECT_ROOT / "vod_output"
MEME_OUTPUT_DIR = VOD_OUTPUT_DIR / "memes"
MEME_LIBRARY_DIR = PROJECT_ROOT / "meme_library"
LOCAL_SFX_DIR = MEME_LIBRARY_DIR / "local_sfx"

# Keep V3 compatibility with the existing smooth-audio Meme Renderer V3.
DISCOVERY_VERSION = 4
MODE = "local_category_reaction_audio_only"

SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".webm",
    ".mp4", ".mov", ".mkv",
}

# Curated semantic folders. Flat legacy files are still supported and inferred
# from their filename, but explicit category folders win deterministically.
MEME_CATEGORIES = {
    "reversal",
    "disbelief",
    "confusion",
    "absurdity",
    "fail",
    "awkward",
    "hype",
    "fear_shock",
    "wholesome_ironic",
    "impact",
}

REACTION_CATEGORIES = {
    "reversal", "disbelief", "confusion", "absurdity", "fail",
    "awkward", "fear_shock", "wholesome_ironic",
}

# Keep the source payoff readable. Long reaction assets should land after it,
# not cover it. Tiny impact/hype stings get a very narrow exception.
PAYOFF_PRE_GAP = 0.10
PAYOFF_POST_GAP = 0.12
REACTION_PAYOFF_RECOVERY = 0.22
TINY_IMPACT_MAX_DURATION = 0.95
TINY_IMPACT_MAX_OVERLAP = 0.28
LONG_REACTION_SECONDS = 2.0
LONG_REACTION_MIN_QUIET_FRACTION = 0.72
LONG_REACTION_MIN_QUIET_SCORE = 0.58

# Generic edit functions already emitted by the existing Meme Analyzer V2.
# Filenames are enough: e.g. vine_boom.mp3, bruh.mp3, record_scratch.mp3.
FUNCTION_HINTS: dict[str, tuple[str, ...]] = {
    "record_scratch": (
        "record scratch", "scratch", "rewind", "stop", "wait", "hold up",
    ),
    "confused_voice": (
        "bruh", "huh", "what", "confused", "excuse me", "ayo", "what the",
        "fah", "fahhh", "fahhhh",
    ),
    "disbelief_sting": (
        "fah", "fahhh", "fahhhh", "bruh", "what", "no way", "disbelief",
    ),
    "impact_boom": (
        "vine boom", "boom", "impact", "bass drop", "hit", "thud",
    ),
    "dramatic_hit": (
        "dramatic", "hit", "boom", "sting", "impact", "suspense",
    ),
    "error_buzzer": (
        "error", "buzzer", "wrong", "incorrect", "fail buzzer", "denied",
    ),
    "fail_sting": (
        "fail", "sad trombone", "trombone", "lose", "loss", "womp", "wah",
        "fah", "fahhh", "fahhhh",
    ),
    "awkward_cricket": (
        "cricket", "awkward", "silence", "dead silence", "quiet",
    ),
    "crowd_gasp": (
        "gasp", "crowd gasp", "shock", "surprise", "audience gasp",
    ),
    "hype_sting": (
        "hype", "airhorn", "air horn", "horn", "crowd", "lets go", "cheer",
    ),
    "cartoon_pop": (
        "pop", "cartoon", "boing", "click",
    ),
    "comedic_pause": (
        "fah", "fahhh", "fahhhh", "bruh", "cricket", "awkward",
    ),
    "laugh_sting": (
        "laugh", "laughter", "laughing", "giggle", "comedy",
    ),
    "other": (),
    "none": (),
}

CATEGORY_HINTS: dict[str, tuple[str, ...]] = {
    "reversal": (
        "light work", "never see it coming", "plot twist", "reversal", "backfire",
        "instant regret", "aged badly",
    ),
    "disbelief": (
        "what the hell", "no way", "bruh", "disbelief", "fah", "fahhh", "fahhhh",
    ),
    "confusion": (
        "huh", "what", "confused", "ayo", "excuse me", "what is going on",
    ),
    "absurdity": (
        "cursed", "absurd", "weird", "goofy", "what the", "what is this",
    ),
    "fail": (
        "fail", "womp", "sad trombone", "error", "buzzer", "fah", "fahhhh",
    ),
    "awkward": (
        "awkward", "cricket", "silence", "dead air", "stare",
    ),
    "hype": (
        "lets go", "let's go", "hype", "scream", "celebration", "hype reaction",
    ),
    "fear_shock": (
        "jumpscare", "jump scare", "shock", "scream", "scared",
    ),
    "wholesome_ironic": (
        "aww so cute", "so cute", "cute", "wholesome", "aww",
    ),
    "impact": (
        "vine boom", "impact", "boom", "dramatic hit", "bass hit",
    ),
}

GENERIC_HINTS = (
    "meme", "sound", "sfx", "effect", "audio", "reaction",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON root object değil: {target}")
    return data


def _save_json(path: str | Path, data: dict[str, Any]) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(target)
    return target


def _normalize(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[_\-.]+", " ", text)
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _tokens(value: Any) -> set[str]:
    return {token for token in _normalize(value).split() if len(token) >= 2}


def _probe_audio(path: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type:format=duration",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=8,
        )
        if completed.returncode != 0:
            return {"valid": False, "duration": 0.0}
        payload = json.loads(completed.stdout or "{}")
        streams = payload.get("streams", [])
        has_audio = isinstance(streams, list) and any(
            isinstance(item, dict) and item.get("codec_type") == "audio"
            for item in streams
        )
        duration = float((payload.get("format") or {}).get("duration") or 0.0)
        return {"valid": bool(has_audio), "duration": max(0.0, duration)}
    except Exception:
        return {"valid": False, "duration": 0.0}


def _canonical_category(value: Any) -> str:
    text = _normalize(value).replace(" ", "_")
    aliases = {
        "fear": "fear_shock",
        "shock": "fear_shock",
        "jumpscare": "fear_shock",
        "wholesome": "wholesome_ironic",
        "cute": "wholesome_ironic",
        "ironic": "wholesome_ironic",
        "boom": "impact",
        "reaction": "disbelief",
    }
    text = aliases.get(text, text)
    return text if text in MEME_CATEGORIES else ""


def _infer_category(path: Path, normalized_name: str) -> str:
    try:
        relative = path.relative_to(LOCAL_SFX_DIR)
        if len(relative.parts) > 1:
            folder_category = _canonical_category(relative.parts[0])
            if folder_category:
                return folder_category
    except ValueError:
        pass

    tokens = _tokens(normalized_name)
    best_category = ""
    best_score = 0.0
    for category, hints in CATEGORY_HINTS.items():
        score = 0.0
        for hint in hints:
            hint_n = _normalize(hint)
            if hint_n and hint_n in normalized_name:
                score += 3.0
            hint_tokens = _tokens(hint_n)
            if hint_tokens and hint_tokens.issubset(tokens):
                score += 1.5
        if score > best_score:
            best_score = score
            best_category = category
    return best_category


def _scan_library() -> list[dict[str, Any]]:
    LOCAL_SFX_DIR.mkdir(parents=True, exist_ok=True)
    result: list[dict[str, Any]] = []
    for path in sorted(LOCAL_SFX_DIR.rglob("*"), key=lambda p: str(p.relative_to(LOCAL_SFX_DIR)).casefold()):
        if not path.is_file() or path.suffix.casefold() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        info = _probe_audio(path)
        if not info["valid"]:
            continue
        normalized_name = _normalize(path.stem)
        category = _infer_category(path, normalized_name)
        result.append(
            {
                "path": path.resolve(),
                "name": path.stem,
                "normalized_name": normalized_name,
                "tokens": _tokens(path.stem),
                "duration": float(info["duration"]),
                "category": category,
                "relative_path": str(path.relative_to(LOCAL_SFX_DIR)),
            }
        )
    return result


def _validate_slot_package(package: dict[str, Any]) -> None:
    from ai.editor import meme_analyzer

    if package.get("version") != meme_analyzer.MEME_ANALYZER_VERSION:
        raise RuntimeError(
            f"Local SFX Discovery, Meme Analyzer V{meme_analyzer.MEME_ANALYZER_VERSION} çıktısı bekliyor.")
    if not isinstance(package.get("clips"), list):
        raise RuntimeError("Meme slot JSON içinde 'clips' listesi yok.")


def _get_clip(package: dict[str, Any], clip_index: int) -> dict[str, Any]:
    for position, clip in enumerate(package.get("clips", []), start=1):
        if not isinstance(clip, dict):
            continue
        try:
            current = int(clip.get("clip_index", position))
        except (TypeError, ValueError):
            current = position
        if current == clip_index:
            return clip
    raise IndexError(f"clip_index={clip_index} meme slot paketinde bulunamadı.")


def _slot_text(slot: dict[str, Any]) -> str:
    parts: list[str] = [
        str(slot.get("sound_function", "")),
        str(slot.get("intent", "")),
        str(slot.get("why_here", "")),
        str(slot.get("decision_reason", "")),
    ]
    queries = slot.get("search_queries", [])
    if isinstance(queries, list):
        parts.extend(str(item) for item in queries if isinstance(item, str))
    return _normalize(" ".join(parts))


def _candidate_score(slot: dict[str, Any], candidate: dict[str, Any]) -> float:
    sound_function = _normalize(slot.get("sound_function", "other")).replace(" ", "_")
    requested_category = _canonical_category(slot.get("meme_category", ""))
    candidate_category = _canonical_category(candidate.get("category", ""))
    slot_text = _slot_text(slot)
    slot_tokens = _tokens(slot_text)
    name = candidate["normalized_name"]
    name_tokens: set[str] = candidate["tokens"]

    score = 0.0

    # Strongest signal: explicit curated category folder. This prevents a random
    # funny sound from beating the right reaction class merely by filename overlap.
    if requested_category and candidate_category == requested_category:
        score += 22.0
    elif requested_category and candidate_category:
        score -= 6.0

    category_aliases = CATEGORY_HINTS.get(requested_category, ())
    for alias in category_aliases:
        alias_n = _normalize(alias)
        if alias_n and alias_n in name:
            score += 8.0
        alias_tokens = _tokens(alias_n)
        if alias_tokens and alias_tokens.issubset(name_tokens):
            score += 4.0

    # Strong signal: known function aliases in the user's filename.
    aliases = FUNCTION_HINTS.get(sound_function, ())
    for alias in aliases:
        alias_n = _normalize(alias)
        if alias_n and alias_n in name:
            score += 12.0
        alias_tokens = _tokens(alias_n)
        if alias_tokens and alias_tokens.issubset(name_tokens):
            score += 6.0

    # Existing analyzer already produces intent/search terms. Reuse them locally.
    overlap = slot_tokens & name_tokens
    score += 3.0 * len(overlap)

    if name and name in slot_text:
        score += 6.0

    # Generic words should not dominate matching.
    score -= 0.5 * len(name_tokens & set(GENERIC_HINTS))

    # Tiny preference for short SFX; very long files are rarely ideal reaction accents.
    duration = float(candidate.get("duration", 0.0))
    if 0.15 <= duration <= 2.8:
        score += 1.2
    elif duration <= 6.5:
        score += 0.3
    elif duration > 8.0:
        score -= 2.0

    return score


def _select_local_sound(slot: dict[str, Any], library: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    if not library:
        return None, 0.0

    ranked = sorted(
        ((candidate, _candidate_score(slot, candidate)) for candidate in library),
        key=lambda item: (-item[1], item[0]["name"].casefold()),
    )
    best, score = ranked[0]

    # With a tiny 4-5 sound curated library we can safely use the best local choice,
    # but only when the upstream Meme Analyzer itself approved the slot.
    return best, score



def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _range_overlap(start: float, end: float, protected: list[float] | tuple[float, float] | None) -> float:
    if not protected or len(protected) < 2:
        return 0.0
    try:
        p0 = float(protected[0])
        p1 = float(protected[1])
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(end, p1) - max(start, p0))


def _payoff_bounds(clip: dict[str, Any]) -> tuple[float, float] | None:
    payoff = clip.get("payoff_edited_range")
    if not isinstance(payoff, (list, tuple)) or len(payoff) < 2:
        return None
    try:
        start = float(payoff[0])
        end = float(payoff[1])
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    return start, end


def _timing_is_payoff_safe(
    start: float,
    duration: float,
    clip: dict[str, Any],
    slot: dict[str, Any],
) -> bool:
    payoff = _payoff_bounds(clip)
    if payoff is None:
        return True
    end = start + duration
    overlap = _range_overlap(start, end, payoff)
    category = _canonical_category(slot.get("meme_category", ""))
    if overlap <= 1e-6:
        # Do not end a long reaction directly on top of the payoff onset or begin
        # directly on its final syllable/reaction tail.
        if category in REACTION_CATEGORIES:
            if end <= payoff[0] and payoff[0] - end < PAYOFF_PRE_GAP:
                return False
            if start >= payoff[1] and start - payoff[1] < PAYOFF_POST_GAP:
                return False
        return True

    # Only a tiny impact/hype accent may touch a payoff, and only very briefly.
    if category in {"impact", "hype"} and duration <= TINY_IMPACT_MAX_DURATION:
        return overlap <= TINY_IMPACT_MAX_OVERLAP
    return False


def _refine_full_asset_timing(
    slot: dict[str, Any],
    clip: dict[str, Any],
    asset_duration: float,
) -> tuple[dict[str, Any] | None, str]:
    """Keep local audio intact and move only its onset, never trim the asset."""
    refined = copy.deepcopy(slot)
    timing = refined.get("timing")
    if not isinstance(timing, dict):
        return None, "Slot timing bilgisi eksik."

    try:
        main_duration = float(clip.get("main_clip_duration", 0.0))
        teaser_duration = float(clip.get("opening_teaser_duration", 0.0))
        target_start = float(timing.get("main_edited_start", 0.0))
    except (TypeError, ValueError):
        return None, "Clip/timing süreleri geçersiz."

    duration = max(0.0, float(asset_duration))
    earliest = 0.70
    if duration <= 0.0 or main_duration <= earliest:
        return None, "Local SFX süresi veya main clip süresi geçersiz."
    if duration > (main_duration - earliest) + 1e-6:
        return None, (
            f"Local SFX tam süresi ({duration:.3f}s) main clip içine kesmeden sığmıyor "
            f"({main_duration:.3f}s)."
        )

    latest = max(earliest, main_duration - duration)
    target_start = _clamp(target_start, earliest, latest)

    candidates: list[tuple[float, str]] = [(target_start, "semantic_anchor")]
    support = clip.get("audio_support")
    if isinstance(support, dict):
        pauses = support.get("pause_windows", [])
        if isinstance(pauses, list):
            for pause in pauses:
                if not isinstance(pause, dict):
                    continue
                try:
                    p_start = float(pause.get("main_start", 0.0)) + 0.025
                    p_score = float(pause.get("pause_score", 0.0))
                except (TypeError, ValueError):
                    continue
                if abs(p_start - target_start) <= 1.35:
                    candidates.append((_clamp(p_start, earliest, latest), f"pause:{p_score:.3f}"))

        hints = support.get("anchor_hints", [])
        anchor = refined.get("anchor") if isinstance(refined.get("anchor"), dict) else {}
        try:
            anchor_word_id = int(anchor.get("word_id"))
        except (TypeError, ValueError):
            anchor_word_id = -1
        if isinstance(hints, list):
            for hint in hints:
                if not isinstance(hint, dict):
                    continue
                try:
                    if int(hint.get("word_id", -2)) != anchor_word_id:
                        continue
                    word_end = float(hint.get("main_word_end", target_start))
                    delay = float(hint.get("recommended_delay_ms", 45)) / 1000.0
                    h_start = _clamp(word_end + delay, earliest, latest)
                    candidates.append((h_start, "anchor_headroom"))
                except (TypeError, ValueError):
                    continue

    payoff = _payoff_bounds(clip)
    slot_inside_payoff = bool(refined.get("inside_payoff"))
    category = _canonical_category(refined.get("meme_category", ""))
    long_reaction = category in REACTION_CATEGORIES and duration > LONG_REACTION_SECONDS

    # Long spoken/reaction memes are extremely intrusive. They may not sit on top
    # of normal source dialogue just because an AI semantic anchor looked funny.
    # Require a factual quiet window large enough for most of the full asset.
    if long_reaction:
        support = clip.get("audio_support") if isinstance(clip.get("audio_support"), dict) else {}
        pauses = support.get("pause_windows", []) if isinstance(support, dict) else []
        required_quiet = max(1.20, duration * LONG_REACTION_MIN_QUIET_FRACTION)
        long_candidates: list[tuple[float, str]] = []
        if isinstance(pauses, list):
            for pause in pauses:
                if not isinstance(pause, dict):
                    continue
                try:
                    p_start = float(pause.get("main_start", 0.0)) + 0.025
                    p_duration = float(pause.get("duration", 0.0))
                    p_quiet = float(pause.get("quiet_score", 0.0))
                except (TypeError, ValueError):
                    continue
                if p_duration + 1e-6 < required_quiet or p_quiet < LONG_REACTION_MIN_QUIET_SCORE:
                    continue
                if payoff is not None and p_start < payoff[1] + REACTION_PAYOFF_RECOVERY - 0.10:
                    continue
                if abs(p_start - target_start) > 3.0 and payoff is None:
                    continue
                long_candidates.append((_clamp(p_start, earliest, latest), f"long_quiet:{p_quiet:.3f}"))

        if payoff is not None:
            post_start = _clamp(payoff[1] + REACTION_PAYOFF_RECOVERY, earliest, latest)
            # A post-payoff start is only valid for a long reaction when factual
            # pause evidence covers it; do not manufacture headroom.
            long_candidates = [item for item in long_candidates if item[0] >= post_start - 0.12]

        if not long_candidates:
            return None, (
                f"Uzun reaction meme ({duration:.2f}s) için payoff sonrası yeterli sessiz headroom yok; "
                "source dialogue üzerine bindirilmedi."
            )
        candidates = long_candidates

    # Reaction memes that semantically belong to the payoff should normally land
    # AFTER the source payoff, preserving the original surprise first.
    if payoff is not None and category in REACTION_CATEGORIES and not long_reaction:
        post_payoff = _clamp(payoff[1] + REACTION_PAYOFF_RECOVERY, earliest, latest)
        if abs(post_payoff - target_start) <= 1.80:
            candidates.append((post_payoff, "post_payoff_reaction"))

    # De-duplicate close candidates and score: semantic proximity first, then clean onset.
    # Unsafe payoff collisions are removed, not merely penalized.
    unique: list[tuple[float, str]] = []
    for start, source in candidates:
        if any(abs(start - existing[0]) < 0.035 for existing in unique):
            continue
        unique.append((start, source))

    safe_unique = [
        item for item in unique
        if _timing_is_payoff_safe(item[0], duration, clip, refined)
    ]
    if not safe_unique:
        return None, "Local meme için payoff'u bozmayan tam-süre yerleşim bulunamadı."

    def score(item: tuple[float, str]) -> float:
        start, source = item
        proximity = 1.0 - _clamp(abs(start - target_start) / 1.35, 0.0, 1.0)
        clean_bonus = 0.0
        if source.startswith("pause:"):
            try:
                clean_bonus = 0.50 + 0.55 * float(source.split(":", 1)[1])
            except (TypeError, ValueError):
                clean_bonus = 0.50
        elif source == "anchor_headroom":
            clean_bonus = 0.42
        elif source.startswith("long_quiet:"):
            try:
                clean_bonus = 0.85 + 0.55 * float(source.split(":", 1)[1])
            except (TypeError, ValueError):
                clean_bonus = 0.85
        if source == "post_payoff_reaction":
            clean_bonus += 0.65
        return 1.35 * proximity + clean_bonus

    best_start, source = max(safe_unique, key=score)
    best_end = best_start + duration

    timing["main_edited_start"] = round(best_start, 3)
    timing["main_edited_end"] = round(best_end, 3)
    timing["final_start"] = round(teaser_duration + best_start, 3)
    timing["final_end"] = round(teaser_duration + best_end, 3)
    timing["max_duration"] = round(duration, 3)
    timing["asset_full_duration"] = round(duration, 3)
    timing["full_asset_preserved"] = True
    timing["timing_refinement"] = source
    if payoff is not None:
        timing["protected_payoff_main_range"] = [round(payoff[0], 3), round(payoff[1], 3)]
        timing["protected_payoff_final_range"] = [
            round(teaser_duration + payoff[0], 3),
            round(teaser_duration + payoff[1], 3),
        ]
    refined["timing"] = timing
    refined["meme_category"] = category or str(refined.get("meme_category", ""))
    return refined, source


def _discover_slot(slot: dict[str, Any], library: list[dict[str, Any]], clip: dict[str, Any]) -> dict[str, Any]:
    try:
        slot_index = int(slot.get("slot_index", 1))
    except (TypeError, ValueError):
        slot_index = 1

    preferred = str(slot.get("preferred_media", "either")).strip().casefold()
    if preferred == "visual":
        return {
            "slot_index": slot_index,
            "selected": False,
            "reason": "Local SFX modu yalnız ses meme'i kullanır; visual-only slot atlandı.",
            "candidate_count": len(library),
            "warnings": [],
        }

    if not library:
        return {
            "slot_index": slot_index,
            "selected": False,
            "reason": f"Local SFX klasörü boş: {LOCAL_SFX_DIR}",
            "candidate_count": 0,
            "warnings": [],
        }

    selected, local_score = _select_local_sound(slot, library)
    if selected is None:
        return {
            "slot_index": slot_index,
            "selected": False,
            "reason": "Uygun local SFX bulunamadı.",
            "candidate_count": len(library),
            "warnings": [],
        }

    path: Path = selected["path"]
    refined_slot, timing_source = _refine_full_asset_timing(
        slot,
        clip,
        float(selected.get("duration", 0.0)),
    )
    if refined_slot is None:
        return {
            "slot_index": slot_index,
            "selected": False,
            "reason": timing_source,
            "candidate_count": len(library),
            "warnings": [],
        }
    slot = refined_slot

    try:
        fit_score = float(slot.get("fit_score", 8.0))
    except (TypeError, ValueError):
        fit_score = 8.0
    try:
        confidence = float(slot.get("confidence", 0.75))
    except (TypeError, ValueError):
        confidence = 0.75

    relative = path.relative_to(MEME_LIBRARY_DIR)
    candidate = {
        "candidate_key": f"local:{path.name.casefold()}",
        "name": selected["name"],
        "provider": "local_sfx",
        "media_type": "audio",
        "source_url": "",
        "asset_url": "",
        "external_id": path.stem,
        "notes": (
            f"Local-only deterministic match score={local_score:.2f}; "
            f"category={selected.get('category') or 'legacy_inferred'}"
        ),
        "category": selected.get("category", ""),
    }
    asset = {
        "provider": "local_sfx",
        "media_type": "audio",
        "name": selected["name"],
        "absolute_path": str(path),
        "local_file": str(relative),
        "source_url": "",
        "asset_url": "",
        "duration": round(float(selected.get("duration", 0.0)), 3),
        "category": selected.get("category", ""),
        "cached": True,
        "downloaded": False,
    }

    return {
        "slot_index": slot_index,
        "selected": True,
        "fit_score": round(max(0.0, min(10.0, fit_score)), 1),
        "confidence": round(max(0.0, min(1.0, confidence)), 2),
        "reason": (
            f"Web kullanılmadan local SFX seçildi: {path.name} "
            f"(category={slot.get('meme_category', 'unknown')}, "
            f"sound_function={slot.get('sound_function', 'other')}, "
            f"full_duration={float(selected.get('duration', 0.0)):.3f}s, timing={timing_source})."
        ),
        "candidate_count": len(library),
        "candidate": candidate,
        "asset": asset,
        "slot": slot,
        "warnings": [],
        "local_match_score": round(local_score, 2),
    }


def _discover_clip(clip: dict[str, Any], library: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        clip_index = int(clip.get("clip_index", 1))
    except (TypeError, ValueError):
        clip_index = 1
    slots = clip.get("slots", [])
    if not isinstance(slots, list):
        slots = []

    if not slots:
        return {
            "clip_index": clip_index,
            "title": str(clip.get("title", "")),
            "searched": False,
            "selected_asset_count": 0,
            "discoveries": [],
            "reason": "Meme Analyzer bu klip için slot açmadı.",
        }

    discoveries = [
        _discover_slot(slot, library, clip)
        for slot in slots[:1]
        if isinstance(slot, dict)
    ]
    selected_count = sum(1 for item in discoveries if item.get("selected") is True)
    return {
        "clip_index": clip_index,
        "title": str(clip.get("title", "")),
        "searched": False,
        "local_only": True,
        "selected_asset_count": selected_count,
        "discoveries": discoveries,
    }


def get_output_path(slot_json_path: str | Path) -> Path:
    path = Path(slot_json_path)
    base = path.stem
    suffix = "_meme_slots"
    if base.endswith(suffix):
        base = base[:-len(suffix)]
    MEME_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return (MEME_OUTPUT_DIR / f"{base}_meme_discovery.json").resolve()


def _save_output(slot_json_path: str | Path, clips: list[dict[str, Any]], library: list[dict[str, Any]]) -> Path:
    output = get_output_path(slot_json_path)
    package = {
        "version": DISCOVERY_VERSION,
        "mode": MODE,
        "model": "none",
        "reasoning_effort": "none",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": {"meme_slots": str(Path(slot_json_path).resolve())},
        "cache": {
            "root": str(MEME_LIBRARY_DIR.resolve()),
            "local_sfx": str(LOCAL_SFX_DIR.resolve()),
            "web_enabled": False,
            "download_enabled": False,
        },
        "library": {
            "sound_count": len(library),
            "sounds": [item.get("relative_path", item["path"].name) for item in library],
            "categories": sorted({item.get("category", "") for item in library if item.get("category")}),
        },
        "clip_count": len(clips),
        "clips": clips,
    }
    return _save_json(output, package)


def discover_memes(slot_json_path: str | Path, clip_index: int | None = None) -> dict[str, Any]:
    slot_path = Path(slot_json_path).resolve()
    package = _load_json(slot_path)
    _validate_slot_package(package)
    library = _scan_library()

    print()
    print("=" * 74)
    print("🔊 MIMIR LOCAL SFX DISCOVERY")
    print("=" * 74)
    print(f"📚 Local sound count: {len(library)}")
    print("🌐 Web: KAPALI | Download: KAPALI | Discovery AI: YOK")

    clips: list[dict[str, Any]] = []
    if clip_index is not None:
        clips.append(_discover_clip(_get_clip(package, clip_index), library))
    else:
        for raw in package.get("clips", []):
            if isinstance(raw, dict):
                clips.append(_discover_clip(raw, library))

    output = _save_output(slot_path, clips, library)
    selected = sum(int(item.get("selected_asset_count", 0)) for item in clips)
    print(f"✅ Local SFX discovery tamamlandı. selected={selected}")
    print(f"📂 {output}")
    return {"output_path": str(output), "clips": clips}


if __name__ == "__main__":
    raw = input("Meme slot JSON: ").strip().strip('"')
    discover_memes(raw)
