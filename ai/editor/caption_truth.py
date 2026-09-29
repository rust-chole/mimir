"""Caption Truth V6: resolve, escalate and FREEZE final caption truth.

Four separate authorities own the final caption words:

    LEXICAL  what was spoken      -> the caption judge (strongest model) over the
                                     evidence of every ear (gpt-transcribe primary,
                                     model-diverse cross-check, micro votes, word clock);
                                     strict acoustic votes when no judge is available
    TIMING   when it was spoken   -> the immutable Whisper word clock (+ local PCM guard)
    SPEAKER  who spoke it         -> diarization + human identity checkpoint
    IDENTITY canonical spelling of verified people/entities
                                  -> participant_name_lock (evidence-gated), then the
                                     caption judge for what the lock could not decide
                                     (closed choice: verified spelling | keep | uncertain)

This stage runs once per selected short, AFTER the caption stage, and never
re-transcribes the clip. It:

1. canonicalizes verified entity names (spelling only; id/time/speaker kept);
2. spends extra inference only on ambiguous high-impact name spans: one
   UNPROMPTED model-diverse ear (independent of the names, logprobs when the
   backend offers them) + one ear biased with the small verified vocabulary,
   on a few seconds of audio around the word (bounded, cached);
3. asks the caption judge to decide the verified-name candidates the lock
   could not (spelling only), and records who decided every disputed word
   (``lexical_decisions``, ``entity_decisions``) in the truth document;
4. marks words the evidence could not settle as ``caption_uncertain`` (shown,
   never emphasized) instead of guessing;
5. validates timing/speaker invariants and FREEZES the result: a truth
   document with a signature over (id, text, start, end, speaker, label).
   Downstream presentation may only consume it; the final quality gate proves
   the published captions still equal the frozen truth.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ai.editor import caption_judge
from ai.editor import participant_name_lock as name_lock

CAPTION_TRUTH_VERSION = 1
TRUTH_KIND = "mimir_caption_truth"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEMP_DIR = PROJECT_ROOT / "vod_output" / "temp" / "caption_truth"

MAX_ESCALATIONS = max(0, min(6, int(os.getenv("MIMIR_V6_CAPTION_ESCALATIONS", "3") or 3)))
ESCALATION_PAD_S = 1.5
ESCALATION_MIN_WINDOW_S = 3.0
MAX_VOCABULARY = 8
LOW_CONFIDENCE_PROB = 0.5
MIN_TOKEN_LEVEL_EARS = 3        # recorded micro ears needed to judge an unresolved span word by word
TIMING_EPS = 1e-3
ESCALATION_REASONS = ("no_reference_evidence", "score_below_threshold", "insufficient_support",
                      "ordinary_word_without_acoustic_evidence")
ESCALATION_HINTS = frozenset({"direct_address", "name_syntax", "co_participant_speaker", "known_name_flag"})
UNCERTAIN_FLAG = name_lock.TRUTH_V6_UNCERTAIN      # consumed by caption presentation: shown, never emphasized
MARKER = name_lock.TRUTH_V6_MARKER


# ============================================================
# SMALL HELPERS
# ============================================================

def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _canon(token: str) -> str:
    """Casefold letters AND digits: numbers ("12", "12s", "$500") are high-impact truth tokens."""
    return "".join(ch for ch in str(token).replace("’", "'").replace("'", "").casefold() if ch.isalnum())


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"\S+", str(text)) if _canon(t)]


def _write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)
    return path


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not Path(path).is_file():
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"name": path.name, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def truth_path_for(profile_path: str | Path) -> Path:
    """The frozen truth document lives next to the final speaker profile."""
    profile = Path(profile_path)
    stem = profile.stem.split("_speakers_v")[0]
    return profile.with_name(f"{stem}_caption_truth_v{CAPTION_TRUTH_VERSION}.json")


def parse_entities(raw: str | None) -> list[str]:
    """Comma-separated user-verified names/terms (``MIMIR_CAPTION_ENTITIES``)."""
    items: list[str] = []
    for part in str(raw or "").split(","):
        value = " ".join(part.split())
        if value and value.casefold() not in {i.casefold() for i in items}:
            items.append(value)
    return items[:MAX_VOCABULARY]


def verified_extra_entities(creator_name: str | None, entities: Sequence[str] = ()) -> list[name_lock.ExtraEntity]:
    """Only explicit user input is verified: the creator name and configured entities."""
    extras: list[name_lock.ExtraEntity] = []
    if creator_name and " ".join(str(creator_name).split()):
        extras.append(name_lock.ExtraEntity(" ".join(str(creator_name).split()), "creator", "user_verified_creator"))
    for value in entities:
        extras.append(name_lock.ExtraEntity(value, "entity", "user_verified_entity"))
    return extras


# ============================================================
# TRUTH SIGNATURE / FREEZE
# ============================================================

def truth_rows(profile: Mapping[str, Any] | None) -> list[list[Any]]:
    """(id, text, start, end, speaker_raw, speaker_label) of every profile word."""
    rows: list[list[Any]] = []
    for index, word in enumerate((profile or {}).get("words", []) or []):
        if not isinstance(word, Mapping):
            continue
        try:
            start = round(float(word.get("edited_start", 0.0)), 4)
            end = round(float(word.get("edited_end", 0.0)), 4)
        except (TypeError, ValueError):
            start, end = float("nan"), float("nan")
        rows.append([index, str(word.get("word", "")), start, end, str(word.get("speaker_raw") or ""),
                     str(word.get("speaker_label") or "")])
    return rows


def truth_signature(profile: Mapping[str, Any] | None) -> str:
    rows = truth_rows(profile)
    blob = json.dumps(rows, ensure_ascii=False, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def verify_frozen_truth(profile_path: str | Path | None, truth_path: str | Path | None) -> tuple[bool, str]:
    """(ok, detail): the profile words still equal the frozen truth."""
    truth = _load_json(Path(truth_path)) if truth_path else None
    if truth is None or truth.get("kind") != TRUTH_KIND:
        return False, "frozen caption truth document missing"
    profile = _load_json(Path(profile_path)) if profile_path else None
    if profile is None:
        return False, "final speaker profile missing"
    current = truth_signature(profile)
    if current != truth.get("signature"):
        return False, f"caption truth changed after freeze ({str(truth.get('signature'))[:12]} -> {current[:12]})"
    return True, "caption truth unchanged since freeze"


# ============================================================
# TIMING / SPEAKER VALIDATION (no mutation)
# ============================================================

def timing_issues(profile: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    try:
        duration = float(profile.get("clip_duration", 0.0) or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    previous_start = -math.inf
    for index, word in enumerate(profile.get("words", []) or []):
        if not isinstance(word, Mapping):
            continue
        try:
            start, end = float(word.get("edited_start")), float(word.get("edited_end"))
        except (TypeError, ValueError):
            issues.append(f"word {index}: non-numeric timing")
            continue
        if not (math.isfinite(start) and math.isfinite(end)):
            issues.append(f"word {index}: non-finite timing")
            continue
        if start < -TIMING_EPS or end < start - TIMING_EPS:
            issues.append(f"word {index}: invalid interval {start:.3f}-{end:.3f}")
        if duration > 0 and end > duration + 0.05:
            issues.append(f"word {index}: ends after the clip ({end:.3f} > {duration:.3f})")
        if start < previous_start - 0.05:
            issues.append(f"word {index}: start {start:.3f} precedes the previous word ({previous_start:.3f})")
        previous_start = max(previous_start, start)
    return issues


def speaker_issues(profile: Mapping[str, Any]) -> list[str]:
    """Speaker ownership: raw ids from the profile's speaker set; printed labels only
    from the human-confirmed display map and consistent with the raw id."""
    from ai.editor import captions

    trusted = captions._trusted_human_display_map(dict(profile))
    allowed = {str(x) for key in ("participant_speakers", "kept_speakers") for x in (profile.get(key) or []) if str(x)}
    allowed |= {str(k) for k in trusted}
    for segment in profile.get("segments", []) or []:
        if isinstance(segment, Mapping) and str(segment.get("speaker") or ""):
            allowed.add(str(segment.get("speaker")))
    issues: list[str] = []
    for index, word in enumerate(profile.get("words", []) or []):
        if not isinstance(word, Mapping):
            continue
        raw = str(word.get("speaker_raw") or "")
        label = str(word.get("speaker_label") or "")
        if raw and allowed and raw not in allowed:
            issues.append(f"word {index}: unknown speaker id {raw!r}")
        if label and trusted.get(raw) != label:
            issues.append(f"word {index}: label {label!r} is not the confirmed name of speaker {raw!r}")
    return issues


# ============================================================
# UNCERTAINTY (ASR ears could not agree)
# ============================================================

def _ear_phrases(detail: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    """(source, canonical tokens) of every micro-ASR ear of one span.

    An empty phrase is kept: that ear heard nothing where the caption shows
    words, which is evidence AGAINST them."""
    ears: list[tuple[str, list[str]]] = []
    for row in detail.get("candidate_votes", []) or []:
        if not isinstance(row, Mapping):
            continue
        phrase = [_canon(t) for t in _tokens(str(row.get("phrase", "")))]
        for source in row.get("sources", []) or []:
            ears.append((str(source), phrase))
    return ears


def disputed_positions(texts: Sequence[str], ears: Sequence[tuple[str, Sequence[str]]],
                       center: float | None = None) -> list[int] | None:
    """Window positions whose displayed token is NOT confirmed by the ears (None: nothing mappable).

    Each ear's phrase (the caption stage's mapping of that ear to the disputed
    core) is anchored where it matches the window best. The core is where the
    most ears anchor (overlapping anchors form one location; a tie goes to the
    location nearest ``center``, the expected core position); an ear anchored
    elsewhere was mis-mapped and confirms nothing there. Inside the core a
    displayed token is confirmed when a strict majority of ALL ears of the span
    (an ear that heard nothing counts against) heard exactly it, including at
    least one unprompted acoustic ear (a context-prompted majority alone can
    echo the caption's wording). The context padding around the core is not
    part of the dispute.
    """
    anchors: list[tuple[str, Sequence[str], tuple[int, int]]] = []
    for source, phrase in ears:
        region = _best_contiguous_alignment(texts, phrase) if phrase else None
        if region is not None:
            anchors.append((source, phrase, region))
    if not anchors:
        return None
    locations: list[list[tuple[str, Sequence[str], tuple[int, int]]]] = []
    for row in sorted(anchors, key=lambda r: r[2]):
        if locations and row[2][0] < max(r[2][1] for r in locations[-1]):
            locations[-1].append(row)
        else:
            locations.append([row])
    target = (len(texts) - 1) / 2.0 if center is None else center

    def rank(rows: list[tuple[str, Sequence[str], tuple[int, int]]]) -> tuple[int, float]:
        lo, hi = min(r[2][0] for r in rows), max(r[2][1] for r in rows)
        return len({r[0] for r in rows}), -abs((lo + hi - 1) / 2.0 - target)

    core = max(locations, key=rank)
    lo, hi = min(r[2][0] for r in core), max(r[2][1] for r in core)
    confirmations: list[set[str]] = [set() for _ in texts]
    for source, phrase, region in core:
        a, b = max(0, region[0] - 1), min(len(texts), region[1] + 1)   # boundary substitutions
        for block in SequenceMatcher(None, list(texts[a:b]), list(phrase), autojunk=False).get_matching_blocks():
            for k in range(block.size):
                confirmations[a + block.a + k].add(source)
    unprompted = {source for source, _phrase in ears if not name_lock._micro_source_prompted(source)}
    disputed = []
    for index in range(lo, hi):
        agree = confirmations[index]
        if len(agree) * 2 <= len(ears) or (unprompted and not agree & unprompted):
            disputed.append(index)
    return disputed


def _core_center(inside_times: Sequence[tuple[float, float]], window: tuple[float, float], duration: float) -> float:
    """Expected index of the disputed core inside the micro window.

    The caption stage centres the window on the core and clamps it at the clip
    edges, so a window touching the clip start/end holds its core near that edge."""
    if not inside_times:
        return 0.0
    w0, w1 = window
    if w0 <= 1e-3 and not (duration > 0 and w1 >= duration - 1e-3):
        t = w0 + 0.25 * (w1 - w0)
    elif duration > 0 and w1 >= duration - 1e-3 and w0 > 1e-3:
        t = w0 + 0.75 * (w1 - w0)
    else:
        t = (w0 + w1) / 2.0
    return float(min(range(len(inside_times)), key=lambda i: abs(sum(inside_times[i]) / 2.0 - t)))


def unresolved_span_word_ids(profile: Mapping[str, Any], words: Sequence[Any]) -> list[tuple[list[int], str]]:
    """Profile word ids the caption-stage micro ears could not confirm (unresolved spans).

    Mapping is by audio time + lexical anchor (never by the stage's token
    index, which refers to the pre-alignment text). With at least
    ``MIN_TOKEN_LEVEL_EARS`` recorded ears the dispute is judged per displayed
    word (``disputed_positions``): words the independent majority heard stay
    certain, only unconfirmed ones are flagged. With fewer ears the whole
    region matching the top candidate phrase is flagged. Nothing mappable ->
    the span is reported as unmapped and nothing is flagged.
    """
    quality = profile.get("caption_quality") if isinstance(profile.get("caption_quality"), Mapping) else {}
    micro = quality.get("micro_accuracy") if isinstance(quality.get("micro_accuracy"), Mapping) else {}
    try:
        duration = float(profile.get("clip_duration", 0.0) or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    found: list[tuple[list[int], str]] = []
    for detail in micro.get("details", []) or []:
        if not isinstance(detail, Mapping) or detail.get("resolved") is not False:
            continue
        window = detail.get("audio_window") or []
        try:
            w0, w1 = float(window[0]), float(window[1])
        except (TypeError, ValueError, IndexError):
            continue
        votes = sorted((row for row in detail.get("candidate_votes", []) or []
                        if isinstance(row, Mapping) and _tokens(str(row.get("phrase", "")))),
                       key=lambda row: -int(row.get("votes", 0) or 0))
        if not votes:
            continue
        phrase = [_canon(t) for t in _tokens(str(votes[0].get("phrase", "")))]
        # Token level: one profile word may hold several spoken tokens ("mall ever."), and the
        # ears' phrases are tokens; every token keeps the id of the word that shows it.
        owners: list[int] = []
        texts: list[str] = []
        times: list[tuple[float, float]] = []
        for index, word in enumerate(words):
            if not isinstance(word, Mapping):
                continue
            try:
                start, end = float(word.get("edited_start", -1)), float(word.get("edited_end", -1))
            except (TypeError, ValueError):
                continue
            if w0 - 1e-3 <= start and end <= w1 + 1e-3:
                for token in _tokens(str(word.get("word", ""))):
                    owners.append(index)
                    texts.append(_canon(token))
                    times.append((start, end))
        ears = _ear_phrases(detail)
        if len(ears) >= MIN_TOKEN_LEVEL_EARS:
            disputed = disputed_positions(texts, ears, _core_center(times, (w0, w1), duration))
            if disputed is None:
                found.append(([], "unmapped"))
            elif disputed:
                found.append((sorted({owners[k] for k in disputed}), "asr_ears_disagree"))
            continue
        best = _best_contiguous_alignment(texts, phrase)
        if best is None:
            found.append(([], "unmapped"))
            continue
        lo, hi = best
        found.append((sorted({owners[k] for k in range(lo, hi)}), "asr_ears_disagree"))
    return found


def _best_contiguous_alignment(texts: Sequence[str], phrase: Sequence[str]) -> tuple[int, int] | None:
    """Tightest contiguous run of window words that spells the disputed phrase.

    Slides the phrase over the window (a repeated phrase can never stretch the
    region across both occurrences); the run needs at least half of the phrase
    tokens in order. Ties prefer the run closest to the window centre.
    """
    if not texts or not phrase:
        return None
    size = min(len(phrase), len(texts))
    centre = (len(texts) - size) / 2.0
    best: tuple[float, float, int] | None = None
    for offset in range(0, len(texts) - size + 1):
        window = texts[offset:offset + size]
        matched = sum(block.size for block in SequenceMatcher(None, window, phrase, autojunk=False)
                      .get_matching_blocks())
        score = matched / float(len(phrase))
        key = (score, -abs(offset - centre), offset)
        if best is None or key > best:
            best = key
    if best is None or best[0] < 0.5:
        return None
    offset = best[2]
    return offset, offset + size


# ============================================================
# TARGETED ESCALATION (bounded, cached)
# ============================================================

@dataclass(frozen=True)
class EarResult:
    text: str
    model: str
    prompted: bool
    token_probs: tuple[tuple[str, float], ...] = ()   # (token, probability) when the backend reports logprobs


# ear(audio_path, start, end, *, prompted, vocabulary, context_before, context_after) -> EarResult
Ear = Callable[..., EarResult]


@dataclass
class EscalationSpan:
    word_id: int
    token: str
    entity: str
    start: float
    end: float
    reason: str
    score: float
    key: str = ""


def plan_escalations(lock_audit: Mapping[str, Any], words: Sequence[Any], *,
                     max_spans: int = MAX_ESCALATIONS) -> list[EscalationSpan]:
    """Entity candidates the lock could not decide on text evidence alone.

    Only spans with a hint that the token may refer to the verified entity are
    escalated (ordinary-word candidates need direct address or name syntax),
    ranked by evidence score; at most ``max_spans`` per short.
    """
    rows: list[EscalationSpan] = []
    for row in lock_audit.get("rejected", []) or []:
        if not isinstance(row, Mapping):
            continue
        reason = str(row.get("reason", ""))
        if not reason.startswith(ESCALATION_REASONS):
            continue
        evidence = set(row.get("evidence") or [])
        needed = {"direct_address", "name_syntax"} if reason.startswith("ordinary_word") else ESCALATION_HINTS
        if not evidence & needed:
            continue
        index = int(row.get("index", -1))
        if not 0 <= index < len(words) or not isinstance(words[index], Mapping):
            continue
        word = words[index]
        try:
            start, end = float(word.get("edited_start")), float(word.get("edited_end"))
        except (TypeError, ValueError):
            continue
        rows.append(EscalationSpan(index, str(row.get("token", "")), str(row.get("near", "")), start, end, reason,
                                   float(row.get("score", 0.0) or 0.0)))
    rows.sort(key=lambda span: (-span.score, span.start))
    return rows[:max(0, int(max_spans))]


def escalation_window(span: EscalationSpan, duration: float) -> tuple[float, float]:
    start = max(0.0, span.start - ESCALATION_PAD_S)
    end = min(duration if duration > 0 else span.end + ESCALATION_PAD_S, span.end + ESCALATION_PAD_S)
    missing = ESCALATION_MIN_WINDOW_S - (end - start)
    if missing > 0:
        start = max(0.0, start - missing / 2.0)
        end = end + missing / 2.0 if duration <= 0 else min(duration, end + missing / 2.0)
    return round(start, 3), round(end, 3)


def aligned_token(window_words: Sequence[str], heard: Sequence[str], position: int) -> str | None:
    """The token an ear heard at ``position`` of the window (None: dropped/unalignable)."""
    source = [_canon(t) for t in window_words]
    target = [_canon(t) for t in heard]
    if not source or not target or not 0 <= position < len(source):
        return None
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, source, target, autojunk=False).get_opcodes():
        if not i1 <= position < i2:
            continue
        if tag == "equal":
            return heard[j1 + (position - i1)]
        if tag == "replace" and j2 > j1:
            offset = min(j2 - j1 - 1, int((position - i1) * (j2 - j1) / max(1, i2 - i1)))
            return heard[j1 + offset]
        return None
    return None


def _token_probability(result: EarResult, token: str | None) -> float | None:
    if not token or not result.token_probs:
        return None
    wanted = _canon(token)
    probs = [p for t, p in result.token_probs if _canon(t) and (_canon(t) in wanted or wanted.startswith(_canon(t)))]
    return round(min(probs), 4) if probs else None


def _escalation_key(clip_fingerprint: Mapping[str, Any], span: EscalationSpan, window: tuple[float, float],
                    vocabulary: Sequence[str], ear_id: str) -> str:
    blob = json.dumps({"clip": dict(clip_fingerprint), "word": span.word_id, "token": span.token,
                       "window": list(window), "vocabulary": list(vocabulary), "ear": ear_id,
                       "version": CAPTION_TRUTH_VERSION}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def run_escalations(
    spans: Sequence[EscalationSpan],
    *,
    words: Sequence[Any],
    duration: float,
    audio_source: Callable[[], Path] | None,
    ear: Ear | None,
    vocabulary: Sequence[str],
    clip_fingerprint: Mapping[str, Any],
    cached: Mapping[str, Mapping[str, Any]],
    ear_id: str,
) -> tuple[dict[int, list[name_lock.AltObservation]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run (or reuse) the two local ears per span.

    Returns (alternatives per word id, escalation records, fallbacks).
    """
    alternatives: dict[int, list[name_lock.AltObservation]] = {}
    records: list[dict[str, Any]] = []
    fallbacks: list[dict[str, Any]] = []
    audio: Path | None = None
    texts = [str(w.get("word", "")) if isinstance(w, Mapping) else "" for w in words]
    try:
        for span in spans:
            window = escalation_window(span, duration)
            key = _escalation_key(clip_fingerprint, span, window, vocabulary, ear_id)
            span.key = key
            inside = [i for i, w in enumerate(words) if isinstance(w, Mapping)
                      and window[0] - 1e-3 <= float(w.get("edited_start", -1)) and
                      float(w.get("edited_end", 1e9)) <= window[1] + 1e-3]
            if span.word_id not in inside:
                inside = sorted(set(inside) | {span.word_id})
            position = inside.index(span.word_id)
            window_words = [texts[i] for i in inside]
            record = dict(cached.get(key) or {})
            if not record:
                if ear is None or audio_source is None:
                    fallbacks.append({"subsystem": "caption_escalation", "level": "text_evidence_only",
                                      "reason": "no escalation ear available", "word_id": span.word_id})
                    continue
                try:
                    if audio is None:
                        audio = audio_source()
                    before = " ".join(texts[max(0, inside[0] - 12):inside[0]])
                    after = " ".join(texts[inside[-1] + 1:inside[-1] + 13])
                    ears: list[dict[str, Any]] = []
                    for prompted in (False, True):
                        result = ear(audio, window[0], window[1], prompted=prompted, vocabulary=list(vocabulary),
                                     context_before=before, context_after=after)
                        token = aligned_token(window_words, _tokens(result.text), position)
                        ears.append({"prompted": prompted, "model": result.model, "text": result.text[:400],
                                     "aligned_token": token, "token_probability": _token_probability(result, token)})
                    record = {"key": key, "word_id": span.word_id, "token": span.token, "entity": span.entity,
                              "window": list(window), "reason": span.reason, "ears": ears, "at": _now()}
                except Exception as error:  # fail closed: text evidence only
                    fallbacks.append({"subsystem": "caption_escalation", "level": "text_evidence_only",
                                      "reason": f"{type(error).__name__}: {str(error)[:200]}",
                                      "word_id": span.word_id})
                    continue
            records.append(record)
            for row in record.get("ears", []) or []:
                token = row.get("aligned_token")
                if token:
                    probability = row.get("token_probability")
                    alternatives.setdefault(span.word_id, []).append(name_lock.AltObservation(
                        str(token), f"v6_{'prompted' if row.get('prompted') else 'independent'}",
                        bool(row.get("prompted")), float(probability) if probability is not None else None))
    finally:
        if audio is not None:
            try:
                audio.unlink(missing_ok=True)
            except OSError:
                pass
    return alternatives, records, fallbacks


# ============================================================
# REAL EARS (existing MIMIR caption ASR functions; no new provider)
# ============================================================

def extract_clip_audio(edited_clip_path: Path) -> Path:
    """Exact-final 48 kHz mono PCM of the edited clip (same master as the caption stage)."""
    from ai import vod_processor

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(Path(edited_clip_path).resolve()).encode("utf-8")).hexdigest()[:12]
    output = TEMP_DIR / f"truth_{digest}.wav"
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(edited_clip_path), "-vn", "-ac", "1",
               "-ar", str(getattr(vod_processor, "CAPTION_MASTER_SAMPLE_RATE", 48000)), "-c:a", "pcm_s16le",
               str(output)]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               check=False)
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size < 1024:
        raise RuntimeError("caption truth audio extraction failed: " + (completed.stderr.strip()[-300:] or "ffmpeg"))
    return output


def _probabilities(response: Any) -> tuple[tuple[str, float], ...]:
    raw = getattr(response, "logprobs", None)
    if raw is None and isinstance(response, Mapping):
        raw = response.get("logprobs")
    rows: list[tuple[str, float]] = []
    for item in raw or []:
        token = getattr(item, "token", None) if not isinstance(item, Mapping) else item.get("token")
        logprob = getattr(item, "logprob", None) if not isinstance(item, Mapping) else item.get("logprob")
        try:
            rows.append((str(token), math.exp(float(logprob))))
        except (TypeError, ValueError, OverflowError):
            continue
    return tuple(rows)


def openai_micro_ear(audio_path: Path, start: float, end: float, *, prompted: bool, vocabulary: Sequence[str],
                     context_before: str = "", context_after: str = "") -> EarResult:
    """Unprompted ear: the model-diverse cross-check model, no names, logprobs when
    supported. Prompted ear: the primary caption model with the verified vocabulary
    as recognition bias (reference only; the audio always wins)."""
    from ai import vod_processor

    clip = vod_processor.extract_caption_micro_audio(audio_path, start=start, end=end, label="truth_v6",
                                                     enhanced=False)
    try:
        if prompted:
            instruction = ("Verified names that may occur (reference only): " + ", ".join(vocabulary)
                           + ". If the audio clearly says one of these names, spell it exactly. Never force a name "
                             "the audio does not contain.") if vocabulary else ""
            text = vod_processor.transcribe_caption_micro_pass(
                clip, model=vod_processor.CAPTION_ACCURATE_MODEL, label="v6 verified-vocabulary listen",
                context_before=context_before, context_after=context_after, extra_instruction=instruction,
                use_context=True, known_names=list(vocabulary))
            return EarResult(str(text or "").strip(), vod_processor.CAPTION_ACCURATE_MODEL, True)
        from ai.openai_client import client

        model = vod_processor.CAPTION_CROSSCHECK_MODEL
        kwargs: dict[str, Any] = {"model": model}
        if vod_processor.TRANSCRIPTION_LANGUAGE:
            kwargs["language"] = vod_processor.TRANSCRIPTION_LANGUAGE
        with clip.open("rb") as handle:
            try:
                response = client.audio.transcriptions.create(file=handle, response_format="json",
                                                              include=["logprobs"], **kwargs)
            except Exception:  # backend/SDK without logprobs: same ear, text only
                handle.seek(0)
                response = client.audio.transcriptions.create(file=handle, **kwargs)
        text = getattr(response, "text", None)
        if text is None and isinstance(response, Mapping):
            text = response.get("text")
        return EarResult(str(text or "").strip(), model, False, _probabilities(response))
    finally:
        try:
            clip.unlink(missing_ok=True)
        except OSError:
            pass


# ============================================================
# WHO DECIDED EACH DISPUTED WORD (lexical provenance)
# ============================================================

def lexical_decision_rows(profile: Mapping[str, Any], words: Sequence[Any]) -> list[dict[str, Any]]:
    """The caption stage's per-span lexical decisions, mapped to profile word ids by audio time."""
    quality = profile.get("caption_quality") if isinstance(profile.get("caption_quality"), Mapping) else {}
    micro = quality.get("micro_accuracy") if isinstance(quality.get("micro_accuracy"), Mapping) else {}
    rows: list[dict[str, Any]] = []
    for detail in micro.get("details", []) or []:
        decision = detail.get("lexical_decision") if isinstance(detail, Mapping) else None
        if not isinstance(decision, Mapping):
            continue
        try:
            w0, w1 = (float(v) for v in (detail.get("audio_window") or [])[:2])
        except (TypeError, ValueError):
            continue
        before = str(detail.get("primary_phrase", ""))
        after = detail.get("selected_phrase")
        shown = str(after) if after is not None else before
        owners, texts = [], []
        for index, word in enumerate(words):
            if not isinstance(word, Mapping):
                continue
            try:
                start, end = float(word.get("edited_start", -1)), float(word.get("edited_end", -1))
            except (TypeError, ValueError):
                continue
            if w0 - 1e-3 <= start and end <= w1 + 1e-3:
                for token in _tokens(str(word.get("word", ""))):
                    owners.append(index)
                    texts.append(_canon(token))
        found = _best_contiguous_alignment(texts, [_canon(t) for t in _tokens(shown)]) if _tokens(shown) else None
        ids = sorted({owners[k] for k in range(*found)}) if found else []
        paced = float(words[ids[0]].get("edited_start", w0)) if ids else w0
        rows.append({
            "span_id": decision.get("span_id"), "from": before, "to": shown,
            "changed": after is not None and [_canon(t) for t in _tokens(before)] != [_canon(t) for t in _tokens(shown)],
            "resolved": bool(decision.get("resolved")), "decided_by": decision.get("source"),
            "confidence": decision.get("confidence"), "supporting_ears": decision.get("supporting_ears", []),
            "guard": decision.get("guard", ""), "word_ids": ids, "paced_start": round(paced, 3),
        })
    return rows


def _entity_payload(index: int, row: Mapping[str, Any], words: Sequence[Any],
                    escalations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    texts = [str(w.get("word", "")) if isinstance(w, Mapping) else "" for w in words]
    word = words[index] if isinstance(words[index], Mapping) else {}
    ears = [{"prompted_with_names": bool(e.get("prompted")), "model": e.get("model"), "heard": e.get("text"),
             "heard_at_word": e.get("aligned_token"), "token_probability": e.get("token_probability")}
            for record in escalations if int(record.get("word_id", -1)) == index
            for e in record.get("ears", []) or []]
    return {"word_id": index, "caption_word": texts[index], "verified_name": row.get("near"),
            "entity_kind": row.get("entity_kind"), "similarity": row.get("confusion"),
            "lock_evidence": list(row.get("evidence") or []), "lock_reason": row.get("reason"),
            "speaker": str(word.get("speaker_label") or word.get("speaker_raw") or ""),
            "context_before": " ".join(texts[max(0, index - 12):index]),
            "context_after": " ".join(texts[index + 1:index + 13]), "local_ears": ears}


def _judgment_key(payload: Any, judge: Any) -> str:
    blob = json.dumps({"payload": payload, "model": getattr(judge, "model", "custom"),
                       "effort": getattr(judge, "effort", ""), "version": caption_judge.JUDGE_VERSION},
                      sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


# ============================================================
# STAGE
# ============================================================

@dataclass
class TruthResult:
    audit: dict[str, Any]
    changed_text: bool
    truth_path: Path
    fallbacks: list[dict[str, Any]] = field(default_factory=list)


def recognition_vocabulary(roster: Sequence[name_lock.Participant]) -> list[str]:
    """Small ASR bias list: the verified full names (display spelling), nothing else."""
    names: dict[str, list[str]] = {}
    for entity in roster:
        names.setdefault(entity.identity or entity.key, []).append(entity.name)
    return [" ".join(tokens) for tokens in names.values()][:MAX_VOCABULARY]


def verify_truth_invariants(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> None:
    """Only spelling (and truth-side flags) may change: id, time, speaker and order are identical."""
    ignored = {"word", "name_lock", UNCERTAIN_FLAG, MARKER}
    if len(before) != len(after):
        raise ValueError("caption truth changed the word count")
    for index, (a, b) in enumerate(zip(before, after)):
        for key in (set(a) | set(b)) - ignored:
            if a.get(key) != b.get(key):
                raise ValueError(f"caption truth changed {key} of word {index}")


def _clear_previous_flags(words: list[Any]) -> None:
    for word in words:
        if isinstance(word, dict) and MARKER in word:
            word.pop(MARKER, None)
            word.pop(UNCERTAIN_FLAG, None)


def run_caption_truth(
    profile_path: str | Path,
    *,
    edited_clip_path: str | Path | None,
    creator_name: str | None = None,
    entities: Sequence[str] = (),
    ear: Ear | None = None,
    audio_source: Callable[[], Path] | None = None,
    max_escalations: int = MAX_ESCALATIONS,
    truth_path: str | Path | None = None,
    judge: Any = "default",
) -> TruthResult:
    """Resolve + escalate + judge + freeze. Rewrites the profile only when it changes.

    ``judge``: the caption judge for undecided verified-name candidates
    ("default" = caption_judge.default_judge(); None = no judge)."""
    profile_file = Path(profile_path)
    target = Path(truth_path) if truth_path else truth_path_for(profile_file)
    profile = json.loads(profile_file.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise ValueError("speaker profile root is not an object")
    original_json = json.dumps(profile, ensure_ascii=False, sort_keys=True)
    before_words = [dict(w) for w in profile.get("words", []) or [] if isinstance(w, dict)]
    try:
        duration = float(profile.get("clip_duration", 0.0) or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    extras = verified_extra_entities(creator_name, entities)
    vocabulary = recognition_vocabulary(name_lock.confirmed_participants(profile, extras))
    previous = _load_json(target) or {}
    cached = {str(row.get("key")): row for row in previous.get("escalations", []) or []
              if isinstance(row, Mapping) and row.get("key")}
    fallbacks: list[dict[str, Any]] = []

    first = name_lock.lock_participant_names(profile, extra_entities=extras)
    spans = plan_escalations(first.audit, first.words, max_spans=max_escalations)
    clip_fp: dict[str, Any] = {}
    if edited_clip_path is not None and Path(edited_clip_path).is_file():
        clip_fp = _file_fingerprint(Path(edited_clip_path))
    if audio_source is None and edited_clip_path is not None and Path(edited_clip_path).is_file():
        clip = Path(edited_clip_path)
        audio_source = lambda: extract_clip_audio(clip)  # noqa: E731
    ear_id = "openai_micro_v1" if ear is None else getattr(ear, "ear_id", getattr(ear, "__name__", "custom"))
    if ear is None and spans:
        ear = openai_micro_ear
    alternatives, escalations, escalation_fallbacks = run_escalations(
        spans, words=first.words, duration=duration, audio_source=audio_source, ear=ear, vocabulary=vocabulary,
        clip_fingerprint=clip_fp, cached=cached, ear_id=ear_id) if spans else ({}, [], [])
    fallbacks.extend(escalation_fallbacks)
    result = name_lock.lock_participant_names(profile, extra_entities=extras, alternatives=alternatives) \
        if alternatives else first
    for record in escalations:
        decided = next((c for c in result.audit.get("corrections", []) if int(c.get("index", -1)) == record["word_id"]),
                       None)
        record["outcome"] = "canonicalized" if decided else "kept_asr_spelling"
    words = [dict(w) if isinstance(w, Mapping) else w for w in result.words]
    _clear_previous_flags(words)
    verify_truth_invariants(before_words, [w for w in words if isinstance(w, dict)])

    uncertain: list[dict[str, Any]] = []
    for ids, reason in unresolved_span_word_ids(profile, words):
        if not ids:
            uncertain.append({"word_ids": [], "reason": reason})
            continue
        uncertain.append({"word_ids": ids, "reason": reason, "text": " ".join(str(words[i].get("word", ""))
                                                                              for i in ids)})
        for index in ids:
            words[index][UNCERTAIN_FLAG] = True
            words[index][MARKER] = {"uncertain": reason}
    decided_ids = {int(c.get("index", -1)) for c in result.audit.get("corrections", [])}
    open_names: list[tuple[int, Mapping[str, Any]]] = []
    for row in result.audit.get("rejected", []) or []:
        index = int(row.get("index", -1))
        if index in decided_ids or not 0 <= index < len(words) or not isinstance(words[index], dict):
            continue
        if str(row.get("reason", "")).startswith(ESCALATION_REASONS) and set(row.get("evidence") or []) & ESCALATION_HINTS:
            open_names.append((index, row))

    # What the evidence-gated lock could not decide goes to the caption judge (closed choice).
    entity_decisions: list[dict[str, Any]] = []
    judgments: list[dict[str, Any]] = []
    extra_corrections: list[dict[str, Any]] = []
    if open_names:
        entity_judge = caption_judge.default_judge() if judge == "default" else judge
        payload = [_entity_payload(index, row, words, escalations) for index, row in open_names]
        key = _judgment_key(payload, entity_judge)
        cached_judgment = next((r for r in previous.get("judgments", []) or []
                                if isinstance(r, Mapping) and r.get("key") == key), None)
        if cached_judgment is not None:
            decided = {int(k): dict(v) for k, v in (cached_judgment.get("decisions") or {}).items()}
            entity_meta = dict(cached_judgment.get("meta") or {})
        else:
            decided, entity_meta = caption_judge.resolve_entities(payload, judge=entity_judge)
        if entity_meta.get("status") == "judged":
            judgments.append({"key": key, "decisions": {str(k): v for k, v in decided.items()}, "meta": entity_meta})
        elif entity_meta.get("status") in ("unavailable", "failed"):
            fallbacks.append({"subsystem": "caption_entity_judge", "level": "uncertain_marked",
                              "reason": str(entity_meta.get("reason", entity_meta.get("status")))})
        for index, row in open_names:
            decision = decided.get(index) or {"verdict": "uncertain", "confidence": None, "reason": "no decision"}
            token = str(words[index].get("word", ""))
            entity_decisions.append({"word_id": index, "token": token, "near": row.get("near"), **decision})
            if decision["verdict"] == "canonical":
                replacement = name_lock._replacement(token, str(row.get("near", "")))
                if replacement and replacement != token:
                    provenance = {"from": token, "to": replacement, "canonical": row.get("near"),
                                  "entity_kind": row.get("entity_kind"), "confusion": row.get("confusion"),
                                  "evidence": [*list(row.get("evidence") or []), "caption_judge"],
                                  "decided_by": "caption_judge", "confidence": decision.get("confidence"),
                                  "version": name_lock.NAME_LOCK_VERSION}
                    words[index]["word"] = replacement
                    words[index]["name_lock"] = provenance
                    extra_corrections.append(provenance | {"index": index})
                    continue
            if decision["verdict"] == "keep":
                continue
            words[index][UNCERTAIN_FLAG] = True
            words[index][MARKER] = {"uncertain": "entity_spelling_unverified", "near": row.get("near")}
            uncertain.append({"word_ids": [index], "reason": "entity_spelling_unverified",
                              "text": str(words[index].get("word", "")), "near": row.get("near")})
        verify_truth_invariants(before_words, [w for w in words if isinstance(w, dict)])

    lexical_judge = {}
    quality = profile.get("caption_quality") if isinstance(profile.get("caption_quality"), Mapping) else {}
    if isinstance(quality.get("lexical_judge"), Mapping):
        lexical_judge = dict(quality["lexical_judge"])
    if lexical_judge.get("status") in ("unavailable", "failed"):
        fallbacks.append({"subsystem": "caption_lexical_judge", "level": "strict_acoustic_vote",
                          "reason": str(lexical_judge.get("reason", lexical_judge.get("status")))})
    lexical_rows = lexical_decision_rows(profile, words)

    profile["words"] = words
    lock_audit = {k: v for k, v in result.audit.items() if k != "changed"}
    if extra_corrections:
        lock_audit["corrections"] = [*lock_audit.get("corrections", []), *extra_corrections]
        judged = {int(c["index"]) for c in extra_corrections}
        lock_audit["rejected"] = [r for r in lock_audit.get("rejected", []) if int(r.get("index", -1)) not in judged]
        lock_audit["status"] = "corrected"
    profile["participant_name_lock"] = lock_audit
    t_issues = timing_issues(profile)
    s_issues = speaker_issues(profile)
    signature = truth_signature(profile)
    profile["caption_truth"] = {"version": CAPTION_TRUTH_VERSION, "signature": signature, "path": str(target)}
    changed_profile = json.dumps(profile, ensure_ascii=False, sort_keys=True) != original_json
    if changed_profile:
        temp = profile_file.with_name(profile_file.name + ".tmp")
        temp.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, profile_file)
    changed_text = [w.get("word") for w in before_words] != [w.get("word") for w in words if isinstance(w, dict)]

    document = {
        "kind": TRUTH_KIND, "version": CAPTION_TRUTH_VERSION,
        "profile": {"path": str(profile_file.resolve()), "words": len(truth_rows(profile))},
        "signature": signature,
        "authorities": {
            "lexical": "caption judge over every ear's evidence (grounding-guarded); strict micro votes without it",
            "timing": "immutable Whisper word clock (+ backward-only exact-PCM guard); never moved by spelling",
            "speaker": "diarization + human identity checkpoint; never changed by text",
            "identity": "verified roster (human-confirmed speakers + user-verified entities); evidence-gated spelling",
        },
        "roster": lock_audit.get("roster", []),
        "vocabulary": vocabulary,
        "entity_lock": {k: lock_audit.get(k) for k in ("status", "corrections", "rejected", "policy")},
        "escalations": escalations,
        "lexical_judge": lexical_judge,
        "lexical_decisions": lexical_rows,
        "entity_decisions": entity_decisions,
        "judgments": judgments,
        "uncertain": uncertain,
        "timing_issues": t_issues,
        "speaker_issues": s_issues,
        "fallbacks": fallbacks,
        "words": truth_rows(profile),
    }
    previous_document = {k: v for k, v in previous.items() if k != "created_at"}
    if previous_document != document or not target.is_file():
        _write_json(target, {**document, "created_at": _now()})
    audit = {
        "status": "frozen", "version": CAPTION_TRUTH_VERSION, "signature": signature, "truth_path": str(target),
        "corrections": lock_audit.get("corrections", []), "rejected": lock_audit.get("rejected", []),
        "escalations": len(escalations), "escalation_calls": sum(len(r.get("ears", [])) for r in escalations
                                                                 if r.get("key") not in cached),
        "uncertain_words": sum(len(u.get("word_ids", [])) for u in uncertain),
        "judge_changed_spans": sum(1 for r in lexical_rows if r["changed"] and r["decided_by"] == "caption_judge"),
        "judge_named_words": len(extra_corrections),
        "timing_issues": len(t_issues), "speaker_issues": len(s_issues), "fallbacks": fallbacks,
        "changed": changed_text,
    }
    return TruthResult(audit, changed_text, target, fallbacks)
