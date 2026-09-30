"""Lexical truth of the final short: listen, compare, escalate locally, judge, FREEZE.

    pass A  primary ear   (Qwen Omni; minimal domain context, no names)
    pass B  precision ear (Qwen Omni; verified spellings as REFERENCE ONLY)
    compare A vs B        harmless differences (case, punctuation, apostrophe glyph,
                          hyphenation, number word vs digit) are ignored; a
                          different / missing / inserted word is a local dispute
    escalate              each dispute is settled by deterministic agreement of an
                          independent model-diverse ear, else by the Astra caption
                          judge over the smallest local evidence package (one
                          batched call), else it keeps the primary words marked
                          uncertain. Nothing is guessed.
    freeze                ``FrozenTranscript``: immutable tokens + provenance +
                          signature. Timing is not part of it.

The model-diverse OpenAI ear runs only when genuinely needed: Qwen unavailable
or invalid (it then provides both passes), A and B materially disagree (one
full-short listen), or local disputes (a few seconds of audio each).
"""
from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from ai.caption_stack import audio as analysis_audio
from ai.caption_stack import openai_ears, qwen_omni
from ai.caption_stack.config import CaptionStackSettings

LEXICAL_VERSION = 1
MAX_DISPUTES = 10               # escalated disputes per short; the rest keep primary words, marked uncertain
MAX_DISPUTE_TOKENS = 8          # a dispute is a local phrase, never a transcript-sized span
MERGE_GAP_TOKENS = 1
CONTEXT_TOKENS = 8
LOCAL_PAD_S = 0.6
LOCAL_MIN_WINDOW_S = 3.0
LOCAL_MAX_WINDOW_S = 12.0
MATERIAL_AGREEMENT = 0.80       # below this A/B key agreement one full-short model-diverse ear replaces local ears
MIN_LEXICAL_COVERAGE = 0.78
PARALLEL_EARS = 4

PRIMARY = "qwen_primary"
PRECISION = "qwen_precision"
FALLBACK_LOCAL = "fallback_local"
FALLBACK_FULL = "fallback_full"
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})
_SEPARATORS = str.maketrans({"-": " ", "–": " ", "—": " ", "/": " "})
_NEGATIONS = frozenset({"not", "no", "never", "nobody", "nothing", "none", "neither", "nor", "nowhere", "cannot"})
_PLACEHOLDER_NAMES = {"speaker", "unknown", "main", "secondary"}


class LexicalUnavailable(RuntimeError):
    """No configured ear produced valid lexical evidence for the final short."""


# ============================================================
# TOKENS + COMPARISON KEYS
# ============================================================

def _lexical(token: str) -> bool:
    return any(ch.isalnum() for ch in str(token))


def display_tokens(text: str) -> list[str]:
    """Whitespace tokens that carry a word; punctuation-only tokens attach to a neighbor."""
    tokens: list[str] = []
    prefix = ""
    for raw in str(text or "").split():
        if not _lexical(raw):
            if tokens:
                tokens[-1] += raw
            else:
                prefix += raw
            continue
        tokens.append(prefix + raw)
        prefix = ""
    return tokens


def token_keys(token: str) -> list[str]:
    """Comparison keys of one token: case, punctuation and apostrophe glyph are harmless;
    hyphen/slash compounds compare by their parts; number words compare as digits."""
    from ai import vod_processor

    keys: list[str] = []
    for part in str(token).translate(_APOSTROPHES).translate(_SEPARATORS).split():
        key = vod_processor.canonical_word(part).strip("'").replace("_", "")
        if key:
            keys.append(key)
    return keys


def key_sequence(tokens: Sequence[str]) -> tuple[list[str], list[int]]:
    keys: list[str] = []
    owners: list[int] = []
    for index, token in enumerate(tokens):
        for key in token_keys(token):
            keys.append(key)
            owners.append(index)
    return keys, owners


def keys_of(tokens: Sequence[str]) -> list[str]:
    return key_sequence(tokens)[0]


def agreement(left: Sequence[str], right: Sequence[str]) -> float:
    a, b = keys_of(left), keys_of(right)
    if not a and not b:
        return 1.0
    return float(SequenceMatcher(None, a, b, autojunk=False).ratio())


def aligned_range(base: Sequence[str], other: Sequence[str], core: tuple[int, int]) -> tuple[int, int] | None:
    """The ``other`` token range heard where ``base[core]`` is (None: other heard nothing there)."""
    bk, bo = key_sequence(base)
    ok, oo = key_sequence(other)
    inside = [k for k, owner in enumerate(bo) if core[0] <= owner < core[1]]
    if not inside or not ok:
        return None
    k0, k1 = inside[0], inside[-1] + 1
    picked: list[int] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, bk, ok, autojunk=False).get_opcodes():
        if tag == "insert":
            if k0 <= i1 <= k1:
                picked.extend(range(j1, j2))
            continue
        lo, hi = max(k0, i1), min(k1, i2)
        if lo >= hi:
            continue
        if tag == "equal":
            picked.extend(range(j1 + (lo - i1), j1 + (hi - i1)))
        elif tag == "replace":
            picked.extend(range(j1, j2))
    if not picked:
        return None
    return oo[min(picked)], oo[max(picked)] + 1


def phrase_at(base: Sequence[str], other: Sequence[str], core: tuple[int, int]) -> tuple[str, ...]:
    found = aligned_range(base, other, core)
    return tuple(other[found[0]:found[1]]) if found else ()


def locate_phrase(tokens: Sequence[str], phrase: str, taken: Sequence[tuple[int, int]] = ()) -> tuple[int, int] | None:
    """First exact (key-level) occurrence of ``phrase`` in ``tokens`` not already covered."""
    wanted = keys_of(display_tokens(phrase))
    if not wanted:
        return None
    keys, owners = key_sequence(tokens)
    size = len(wanted)
    for start in range(0, len(keys) - size + 1):
        if keys[start:start + size] != wanted:
            continue
        found = (owners[start], owners[start + size - 1] + 1)
        if not any(found[0] < b and a < found[1] for a, b in taken):
            return found
    return None


# ============================================================
# EARS
# ============================================================

@dataclass(frozen=True)
class EarTranscript:
    ear: str
    provider: str
    model: str
    prompted: bool
    tokens: tuple[str, ...]
    uncertain: tuple[qwen_omni.UncertainSpan, ...] = ()
    language: str = ""
    notes: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    def summary(self) -> dict[str, Any]:
        return {"ear": self.ear, "provider": self.provider, "model": self.model, "prompted": self.prompted,
                "words": len(self.tokens), "language": self.language,
                "uncertain_spans": [span.to_dict() for span in self.uncertain], "notes": list(self.notes)}


def _qwen_listen(path: Path, instructions: str, settings: CaptionStackSettings) -> tuple[Any, dict[str, Any]]:
    return qwen_omni.listen(analysis_audio.wav_base64(path), instructions=instructions, settings=settings)


def _openai_listen(path: Path, *, prompted: bool, model: str, language: str, verified_terms: Sequence[str]) -> str:
    if prompted:
        return openai_ears.precision_listen(path, model=model, language=language, verified_terms=verified_terms)
    return openai_ears.acoustic_listen(path, model=model, language=language)


@dataclass
class Ears:
    """Injectable ear functions (defaults call the real providers)."""
    qwen: Callable[[Path, str, CaptionStackSettings], tuple[Any, dict[str, Any]]] = _qwen_listen
    openai: Callable[..., str] = _openai_listen


def clean_verified_terms(values: Sequence[str] | None) -> list[str]:
    return openai_ears.clean_terms(values, limit=12)


# ============================================================
# DISPUTES
# ============================================================

@dataclass
class Dispute:
    span_id: str
    chunk: int
    core: tuple[int, int]                          # primary (pass A) token range, chunk-local
    kinds: list[str]
    flags: list[str] = field(default_factory=list)
    candidates: dict[str, dict[str, Any]] = field(default_factory=dict)
    window: tuple[float, float] | None = None
    window_source: str = ""
    decision: Any = None                           # caption_judge.SpanDecision


def _flags(core_tokens: Sequence[str], other_tokens: Sequence[str], verified_keys: set[str],
           sentence_initial: bool) -> list[str]:
    from ai import vod_processor

    a, b = keys_of(core_tokens), keys_of(other_tokens)
    differing = set(a) ^ set(b)
    flags: list[str] = []
    if any(k in _NEGATIONS or k.endswith("n't") for k in differing):
        flags.append("negation")
    if any(any(ch.isdigit() for ch in k) or k in vod_processor.NUMBER_WORDS for k in differing):
        flags.append("number")
    if any("'" in k for k in differing):
        flags.append("contraction")
    capitalized = [t for i, t in enumerate([*core_tokens, *other_tokens])
                   if t[:1].isupper() and not (i == 0 and sentence_initial)]
    if differing & verified_keys or any(set(token_keys(t)) & differing for t in capitalized):
        flags.append("name")
    return flags


def find_disputes(primary: EarTranscript, precision: EarTranscript, *, chunk: int,
                  verified_terms: Sequence[str] = ()) -> list[Dispute]:
    """Meaningful A/B differences + spans either ear itself doubted, as local disputes."""
    a, b = list(primary.tokens), list(precision.tokens)
    ak, ao = key_sequence(a)
    bk, _bo = key_sequence(b)
    raw: list[tuple[int, int, str]] = []
    kind_names = {"replace": "different_word", "delete": "primary_only_words", "insert": "precision_only_words"}
    for tag, i1, i2, _j1, _j2 in SequenceMatcher(None, ak, bk, autojunk=False).get_opcodes():
        if tag == "equal" or not ao:
            continue
        if i1 < i2:
            c0, c1 = ao[i1], ao[i2 - 1] + 1
        else:   # B heard words A did not: the dispute is the A words around the gap
            left = ao[i1 - 1] if i1 > 0 else None
            right = ao[i1] if i1 < len(ao) else None
            c0 = left if left is not None else right
            c1 = (right if right is not None else left) + 1
        raw.append((c0, c1, kind_names[tag]))

    # Spans an ear itself doubted (even where A and B agree) are disputes too.
    doubted: list[tuple[str, int, qwen_omni.UncertainSpan, EarTranscript]] = []
    for ear, transcript in ((PRIMARY, primary), (PRECISION, precision)):
        for number, span in enumerate(transcript.uncertain, start=1):
            if ear == PRIMARY:
                found = locate_phrase(a, span.heard)
            else:
                in_b = locate_phrase(b, span.heard)
                found = aligned_range(b, a, in_b) if in_b else None
            if found:
                raw.append((found[0], found[1], f"self_uncertain:{ear}"))
                doubted.append((ear, number, span, transcript))

    # Overlapping ranges always merge (cores never overlap); adjacent ones only while local.
    merged: list[list[Any]] = []
    for c0, c1, kind in sorted(raw):
        last = merged[-1] if merged else None
        overlapping = bool(last) and c0 < last[1]
        adjacent = bool(last) and c0 <= last[1] + MERGE_GAP_TOKENS and max(c1, last[1]) - last[0] <= MAX_DISPUTE_TOKENS
        if overlapping or adjacent:
            last[1] = max(last[1], c1)
            if kind not in last[2]:
                last[2].append(kind)
        else:
            merged.append([c0, c1, [kind]])

    verified_keys = {k for term in verified_terms for k in keys_of(display_tokens(term))}
    disputes: list[Dispute] = []
    for index, (c0, c1, kinds) in enumerate(merged):
        dispute = Dispute(f"c{chunk}_d{index}", chunk, (c0, c1), list(kinds))
        _add_candidate(dispute, PRIMARY, primary, tuple(a[c0:c1]),
                       " ".join(a[max(0, c0 - CONTEXT_TOKENS):c1 + CONTEXT_TOKENS]))
        _add_candidate(dispute, PRECISION, precision, phrase_at(a, b, (c0, c1)),
                       " ".join(phrase_at(a, b, (max(0, c0 - CONTEXT_TOKENS), min(len(a), c1 + CONTEXT_TOKENS)))))
        initial = c0 == 0 or a[c0 - 1].rstrip().endswith((".", "!", "?"))
        dispute.flags = _flags(a[c0:c1], dispute.candidates[PRECISION]["tokens"], verified_keys, initial)
        disputes.append(dispute)

    # A doubting ear's alternatives become candidate readings of the whole core.
    for ear, number, span, transcript in doubted:
        base = a if ear == PRIMARY else b
        heard_at = locate_phrase(base, span.heard)
        for dispute in disputes:
            core_in_base = dispute.core if ear == PRIMARY else aligned_range(a, b, dispute.core)
            if not heard_at or not core_in_base or not (core_in_base[0] <= heard_at[0] and heard_at[1] <= core_in_base[1]):
                continue
            for alt_number, alternative in enumerate(span.alternatives, start=1):
                phrase = (*base[core_in_base[0]:heard_at[0]], *display_tokens(alternative),
                          *base[heard_at[1]:core_in_base[1]])
                _add_candidate(dispute, f"{ear}_alt{number}_{alt_number}", transcript, tuple(phrase), "",
                               view="self-reported alternative")
            break
    return disputes


def _add_candidate(dispute: Dispute, ear: str, transcript: EarTranscript, tokens: tuple[str, ...], window: str,
                   *, view: str = "") -> None:
    dispute.candidates[ear] = {"tokens": tuple(tokens), "model": transcript.model, "provider": transcript.provider,
                               "prompted": transcript.prompted, "view": view or "full short",
                               "window_text": window}


def _priority(dispute: Dispute) -> tuple[int, int, int]:
    impact = sum(1 for f in dispute.flags if f in {"negation", "number", "name"})
    acoustic = 0 if any(not k.startswith("self_uncertain") for k in dispute.kinds) else 1
    return (-impact, acoustic, dispute.core[0])


# ============================================================
# EVIDENCE WINDOWS + FALLBACK EARS
# ============================================================

Locator = Callable[[int, Sequence[str]], Sequence[tuple[float, float] | None] | None]


def _window_for(dispute: Dispute, tokens: Sequence[str], chunk_window: tuple[float, float],
                times: Sequence[tuple[float, float] | None] | None) -> tuple[tuple[float, float], str]:
    c0, c1 = dispute.core
    w0, w1 = max(0, c0 - CONTEXT_TOKENS), min(len(tokens), c1 + CONTEXT_TOKENS)
    lo, hi = chunk_window
    known = [times[i] for i in range(w0, w1) if times and i < len(times) and times[i]] if times else []
    if known:
        start, end, source = known[0][0] - LOCAL_PAD_S, known[-1][1] + LOCAL_PAD_S, "locator_alignment"
    else:   # proportional estimate inside the chunk (evidence window only, never caption timing)
        span = hi - lo
        start = lo + span * w0 / max(1, len(tokens)) - LOCAL_PAD_S
        end = lo + span * w1 / max(1, len(tokens)) + LOCAL_PAD_S
        source = "proportional_estimate"
    start, end = max(lo, start), min(hi, max(start + 0.2, end))
    if end - start < LOCAL_MIN_WINDOW_S:
        missing = LOCAL_MIN_WINDOW_S - (end - start)
        start, end = max(lo, start - missing / 2), min(hi, end + missing / 2)
    if end - start > LOCAL_MAX_WINDOW_S:
        centre = (start + end) / 2
        start, end = max(lo, centre - LOCAL_MAX_WINDOW_S / 2), min(hi, centre + LOCAL_MAX_WINDOW_S / 2)
    return (round(start, 3), round(end, 3)), source


def _settle(dispute: Dispute, fallback_ear: str) -> Any:
    """Deterministic agreement: the model-diverse ear heard exactly one Qwen candidate."""
    from ai.editor import caption_judge

    fallback = dispute.candidates.get(fallback_ear)
    if not fallback or not keys_of(fallback["tokens"]):
        return None                                 # a deletion is never settled without the judge
    wanted = keys_of(fallback["tokens"])
    for ear, candidate in dispute.candidates.items():
        if ear == fallback_ear or keys_of(candidate["tokens"]) != wanted:
            continue
        if ear == PRIMARY:
            return caption_judge.SpanDecision(dispute.span_id, None, True, f"agreement_{PRIMARY}+{fallback_ear}")
        return caption_judge.SpanDecision(dispute.span_id, " ".join(candidate["tokens"]), True,
                                          f"agreement_{ear}+{fallback_ear}", supporting_ears=(ear, fallback_ear))
    return None


def _span_evidence(dispute: Dispute, tokens: Sequence[str]) -> Any:
    from ai.editor import caption_judge

    c0, c1 = dispute.core
    w0, w1 = max(0, c0 - CONTEXT_TOKENS), min(len(tokens), c1 + CONTEXT_TOKENS)
    ears = [{"ear": ear, "model": cand["model"], "view": cand["view"], "prompted": bool(cand["prompted"]),
             "heard_window": cand.get("window_text", ""), "heard_core": " ".join(cand["tokens"])}
            for ear, cand in dispute.candidates.items()]
    return caption_judge.SpanEvidence(
        span_id=dispute.span_id, core=dispute.core, current=" ".join(tokens[c0:c1]),
        window_text=" ".join(tokens[w0:w1]), context_before=" ".join(tokens[max(0, w0 - 12):w0]),
        context_after=" ".join(tokens[w1:w1 + 12]), audio_window=dispute.window or (0.0, 0.0),
        suspicion=[*dispute.kinds, *dispute.flags], ears=ears, strict=None, clock=[], core_clock=[])


# ============================================================
# FROZEN TRANSCRIPT
# ============================================================

@dataclass(frozen=True)
class FrozenTranscript:
    """Immutable caption wording. Alignment consumes it; nothing rewrites it."""
    tokens: tuple[str, ...]
    status: tuple[str, ...]                      # agreed | resolved | uncertain (per token)
    chunk_of: tuple[int, ...]
    chunks: tuple[tuple[float, float], ...]
    spans: tuple[Mapping[str, Any], ...]
    language: str
    signature: str

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    def chunk_tokens(self, chunk: int) -> tuple[int, tuple[str, ...]]:
        indices = [i for i, c in enumerate(self.chunk_of) if c == chunk]
        if not indices:
            return 0, ()
        return indices[0], tuple(self.tokens[indices[0]:indices[-1] + 1])

    def span_of(self, token_index: int) -> Mapping[str, Any] | None:
        return next((s for s in self.spans if s["tokens"][0] <= token_index < s["tokens"][1]), None)

    def intact(self) -> bool:
        return lexical_signature(self.tokens, self.chunks) == self.signature


def lexical_signature(tokens: Sequence[str], chunks: Sequence[tuple[float, float]]) -> str:
    blob = json.dumps({"tokens": list(tokens), "chunks": [[round(a, 4), round(b, 4)] for a, b in chunks],
                       "version": LEXICAL_VERSION}, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _freeze_chunk(primary: Sequence[str], disputes: Sequence[Dispute], offset: int
                  ) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    tokens: list[str] = []
    status: list[str] = []
    spans: list[dict[str, Any]] = []
    position = 0
    for dispute in sorted(disputes, key=lambda d: d.core):
        c0, c1 = dispute.core
        for index in range(position, c0):
            tokens.append(primary[index])
            status.append("agreed")
        decision = dispute.decision
        new = list(primary[c0:c1]) if decision.phrase is None else display_tokens(decision.phrase)
        f0 = offset + len(tokens)
        tokens.extend(new)
        status.extend(["resolved" if decision.resolved else "uncertain"] * len(new))
        before = list(primary[c0:c1])
        alternatives = []
        if len(new) == 1:
            for ear, cand in dispute.candidates.items():
                if len(cand["tokens"]) == 1:
                    alternatives.append({"token": cand["tokens"][0], "source": ear, "prompted": bool(cand["prompted"])})
        spans.append({
            "span_id": dispute.span_id, "chunk": dispute.chunk, "tokens": [f0, offset + len(tokens)],
            "from": " ".join(before), "to": " ".join(new), "changed": keys_of(before) != keys_of(new),
            "resolved": bool(decision.resolved), "decided_by": decision.source,
            "confidence": decision.confidence, "guard": decision.guard, "reasons": [*dispute.kinds, *dispute.flags],
            "supporting_ears": list(decision.supporting_ears),
            "audio_window": list(dispute.window) if dispute.window else None, "window_source": dispute.window_source,
            "ears": [{"ear": ear, "model": c["model"], "prompted": bool(c["prompted"]), "view": c["view"],
                      "heard": " ".join(c["tokens"])} for ear, c in dispute.candidates.items()],
            "alternatives": alternatives,
        })
        position = c1
    for index in range(position, len(primary)):
        tokens.append(primary[index])
        status.append("agreed")
    return tokens, status, spans


# ============================================================
# VERIFIED-NAME ORTHOGRAPHY (spelling only, before the freeze)
# ============================================================

def _name_display_map(names: Sequence[str]) -> dict[str, str]:
    from ai import vod_processor

    result: dict[str, str] = {}
    for raw in names or []:
        parts = display_tokens(" ".join(str(raw).split()))
        if len(parts) != 1:
            continue
        key = vod_processor.canonical_word(parts[0])
        if len(key) >= 3 and key not in _PLACEHOLDER_NAMES:
            result.setdefault(key, parts[0].strip(".,!?;:"))
    return result


def apply_vocative_name_orthography(tokens: Sequence[str], names: Sequence[str], *, similarity: float
                                    ) -> tuple[list[str], dict[str, Any]]:
    """Normalize a phonetic near-miss of a human-verified name only in a direct address.

    Orthographic, not semantic: the token is comma/colon-delimited and the next
    words address "you". ``Jon, would you`` + verified ``JOHN`` -> ``John, would you``;
    ``I watched Jon yesterday`` is untouched. Token count never changes.
    """
    from ai import vod_processor

    out = list(tokens)
    display = _name_display_map(names)
    if not out or not display:
        return out, {"status": "not_needed", "corrections": []}
    corrections: list[dict[str, Any]] = []
    for index, token in enumerate(out):
        current = vod_processor.canonical_word(token)
        if not current or current in display or len(current) < 3:
            continue
        ranked = sorted(((float(SequenceMatcher(None, current, target, autojunk=False).ratio()), target, shown)
                         for target, shown in display.items()
                         if current[:2] == target[:2] and abs(len(current) - len(target)) <= 2), reverse=True)
        ranked = [row for row in ranked if row[0] >= similarity]
        if not ranked or (len(ranked) > 1 and ranked[0][0] < ranked[1][0] + 0.10):
            continue
        following = [vod_processor.canonical_word(x) for x in out[index + 1:index + 5]]
        if not (token.rstrip().endswith((",", ":")) and ("you" in following or "your" in following)):
            continue
        score, target, shown = ranked[0]
        match = re.match(r"^([^A-Za-z0-9']*)([A-Za-z0-9']+)(.*)$", token)
        if not match:
            continue
        prefix, core, suffix = match.groups()
        replacement_core = shown.upper() if core.isupper() else (
            shown[:1].upper() + shown[1:].lower() if core[:1].isupper() else shown.lower())
        replacement = prefix + replacement_core + suffix
        if replacement != token:
            out[index] = replacement
            corrections.append({"index": index, "from": token, "to": replacement, "target_name": target,
                                "similarity": round(score, 4), "reason": "human_verified_name+vocative_second_person"})
    return out, {"status": "corrected" if corrections else "clean", "corrections": corrections,
                 "policy": "verified spelling only in direct-address vocatives; token count unchanged"}


def name_keys(names: Sequence[str]) -> list[str]:
    """Canonical tokens of human-verified names (3+ letters, no speaker placeholders)."""
    from ai import vod_processor

    result: list[str] = []
    for raw in names or []:
        for token in re.findall(r"[A-Za-z0-9']+", str(raw)):
            value = vod_processor.canonical_word(token)
            if len(value) >= 3 and value not in _PLACEHOLDER_NAMES and value not in result:
                result.append(value)
    return result


def known_name_suspicions(tokens: Sequence[str], names: Sequence[str], *, similarity: float) -> list[dict[str, Any]]:
    """Near-miss spellings of verified names. Evidence for the caption-truth name lock only;
    never a text change and never an escalation of its own."""
    from ai import vod_processor

    known = name_keys(names)
    rows: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        value = vod_processor.canonical_word(token)
        if not known or len(value) < 3 or value in known:
            continue
        for name in known:
            if value[:2] != name[:2] or abs(len(value) - len(name)) > 2:
                continue
            score = float(SequenceMatcher(None, value, name, autojunk=False).ratio())
            if score >= similarity:
                rows.append({"start": index, "end": index + 1, "severity": round(min(1.0, 0.82 + 0.18 * score), 4),
                             "reason": f"possible known-name spelling mismatch: {token} vs {name}",
                             "source": "known_name"})
                break
    return rows


# ============================================================
# STAGE
# ============================================================

def _chunk_audio(audio: analysis_audio.AnalysisAudio, chunks: Sequence[tuple[float, float]], directory: Path
                 ) -> list[Path]:
    if len(chunks) == 1:
        return [audio.path]
    return [analysis_audio.cut_window(audio, a, b, directory / f"chunk_{i:02d}.wav")[0]
            for i, (a, b) in enumerate(chunks)]


def _listen_role(role: str, path: Path, *, settings: CaptionStackSettings, terms: Sequence[str], ears: Ears,
                 degradations: list[dict[str, str]]) -> EarTranscript:
    prompted = role == PRECISION
    if settings.primary_provider == "qwen_omni":
        domain = openai_ears.domain_keywords()
        instructions = (qwen_omni.precision_instructions(settings.language, domain, terms) if prompted
                        else qwen_omni.primary_instructions(settings.language, domain))
        try:
            evidence, meta = ears.qwen(path, instructions, settings)
            return EarTranscript(role, "qwen_omni", str(meta.get("model", settings.qwen_model)), prompted,
                                 tuple(display_tokens(evidence.text)), tuple(evidence.uncertain_spans),
                                 evidence.language, tuple([*evidence.notes, *meta.get("invalid_attempts", [])]))
        except (qwen_omni.QwenUnavailable, qwen_omni.QwenInvalidResponse) as error:
            degradations.append({"subsystem": "caption_lexical_primary", "level": settings.transcribe_fallback_provider,
                                 "reason": f"{role}: {type(error).__name__}: {str(error)[:240]}"})
            if settings.transcribe_fallback_provider == "none":
                raise LexicalUnavailable(f"Qwen {role} failed and no transcription fallback is configured: {error}")
    model = settings.transcribe_fallback_model
    text = ears.openai(path, prompted=prompted, model=model, language=settings.language, verified_terms=terms)
    tokens = tuple(display_tokens(text))
    if not tokens:
        raise LexicalUnavailable(f"{model} ({role}) returned no words")
    return EarTranscript(role, "openai_transcribe", model, prompted, tokens)


def listen_passes(audio: analysis_audio.AnalysisAudio, chunks: Sequence[tuple[float, float]], *,
                  settings: CaptionStackSettings, verified_terms: Sequence[str], ears: Ears, workdir: Path
                  ) -> tuple[list[tuple[EarTranscript, EarTranscript]], list[dict[str, str]]]:
    """Pass A + pass B for every chunk (concurrently; results consumed in fixed order)."""
    paths = _chunk_audio(audio, chunks, workdir)
    degradations: list[dict[str, str]] = []
    terms = clean_verified_terms(verified_terms)
    with ThreadPoolExecutor(max_workers=min(PARALLEL_EARS, 2 * len(paths)), thread_name_prefix="mimir-lexical") as pool:
        futures = [(pool.submit(_listen_role, PRIMARY, p, settings=settings, terms=terms, ears=ears,
                                degradations=degradations),
                    pool.submit(_listen_role, PRECISION, p, settings=settings, terms=terms, ears=ears,
                                degradations=degradations)) for p in paths]
        passes = [(a.result(), b.result()) for a, b in futures]
    return passes, degradations


def resolve_and_freeze(
    audio: analysis_audio.AnalysisAudio,
    chunks: Sequence[tuple[float, float]],
    passes: Sequence[tuple[EarTranscript, EarTranscript]],
    *,
    settings: CaptionStackSettings,
    verified_terms: Sequence[str] = (),
    orthography_names: Sequence[str] | None = None,
    locator: Locator | None = None,
    judge: Any = None,
    ears: Ears | None = None,
    workdir: Path,
    name_similarity: float = 0.64,
) -> tuple[FrozenTranscript, dict[str, Any]]:
    """Disputes -> deterministic agreement / Astra -> frozen transcript (+ report).

    ``orthography_names``: the human-verified participant names allowed to fix a
    direct-address spelling (defaults to ``verified_terms``).
    """
    from ai.editor import caption_judge

    ears = ears or Ears()
    terms = clean_verified_terms(verified_terms)
    use_fallback = settings.transcribe_fallback_provider == "openai_transcribe"
    per_chunk: list[list[Dispute]] = []
    agreements: list[float] = []
    for index, (primary, precision) in enumerate(passes):
        per_chunk.append(find_disputes(primary, precision, chunk=index, verified_terms=terms))
        agreements.append(agreement(primary.tokens, precision.tokens))
    all_disputes = [d for chunk in per_chunk for d in chunk]
    escalated = {id(d) for d in sorted(all_disputes, key=_priority)[:MAX_DISPUTES]}
    overflow = [d for d in all_disputes if id(d) not in escalated]
    material = any(score < MATERIAL_AGREEMENT for score in agreements) or bool(overflow)
    calls = {"full": 0, "local": 0}
    errors: list[str] = []

    # Evidence windows (the locator aligns the PRIMARY words: evidence only, never caption timing).
    for index, disputes in enumerate(per_chunk):
        if not disputes:
            continue
        tokens = passes[index][0].tokens
        times = None
        if locator is not None:
            try:
                times = locator(index, tokens)
            except Exception as error:  # locating is best-effort; the proportional window still works
                errors.append(f"locator chunk {index}: {type(error).__name__}: {str(error)[:160]}")
        for dispute in disputes:
            dispute.window, dispute.window_source = _window_for(dispute, tokens, chunks[index], times)

    fallback_ear = FALLBACK_FULL if material else FALLBACK_LOCAL
    if use_fallback and all_disputes:
        jobs: list[tuple[Dispute | None, int, Path]] = []
        if material:
            paths = _chunk_audio(audio, chunks, workdir)
            jobs = [(None, index, paths[index]) for index, disputes in enumerate(per_chunk) if disputes]
        else:
            for dispute in all_disputes:
                if id(dispute) in escalated and dispute.window:
                    try:
                        path = analysis_audio.cut_window(audio, *dispute.window,
                                                         workdir / f"{dispute.span_id}.wav")[0]
                    except (OSError, ValueError) as error:
                        errors.append(f"window {dispute.span_id}: {type(error).__name__}: {str(error)[:160]}")
                        continue
                    jobs.append((dispute, dispute.chunk, path))
        model = settings.transcribe_fallback_model
        with ThreadPoolExecutor(max_workers=max(1, min(PARALLEL_EARS, len(jobs))),
                                thread_name_prefix="mimir-lexical-fallback") as pool:
            futures = [pool.submit(ears.openai, path, prompted=False, model=model, language=settings.language,
                                   verified_terms=()) for _dispute, _chunk, path in jobs]
            for (dispute, chunk_index, _path), future in zip(jobs, futures):
                try:
                    heard = display_tokens(future.result())
                except Exception as error:
                    errors.append(f"{model} {'full' if dispute is None else dispute.span_id}: "
                                  f"{type(error).__name__}: {str(error)[:160]}")
                    continue
                calls["full" if dispute is None else "local"] += 1
                primary_tokens = list(passes[chunk_index][0].tokens)
                transcript = EarTranscript(fallback_ear, "openai_transcribe", model, False, tuple(heard))
                targets = per_chunk[chunk_index] if dispute is None else [dispute]
                for target in targets:
                    c0, c1 = target.core
                    if dispute is None:
                        phrase = phrase_at(primary_tokens, heard, target.core)
                        window_text = " ".join(phrase_at(primary_tokens, heard, (
                            max(0, c0 - CONTEXT_TOKENS), min(len(primary_tokens), c1 + CONTEXT_TOKENS))))
                    else:
                        w0 = max(0, c0 - CONTEXT_TOKENS)
                        window = primary_tokens[w0:min(len(primary_tokens), c1 + CONTEXT_TOKENS)]
                        phrase = phrase_at(window, heard, (c0 - w0, c1 - w0))
                        window_text = " ".join(heard)
                    _add_candidate(target, fallback_ear, transcript, phrase, window_text,
                                   view="full short" if dispute is None else "local window")

    open_spans: list[Any] = []
    by_span: dict[str, Dispute] = {}
    for dispute in all_disputes:
        if id(dispute) not in escalated:
            dispute.decision = caption_judge.SpanDecision(dispute.span_id, None, False, "not_escalated_over_limit")
            continue
        dispute.decision = _settle(dispute, fallback_ear)
        if dispute.decision is None:
            open_spans.append(_span_evidence(dispute, passes[dispute.chunk][0].tokens))
            by_span[dispute.span_id] = dispute
    decisions, judge_meta = caption_judge.resolve_spans(
        open_spans, primary_text=" ".join(p.text for p, _ in passes),
        precision_text=" ".join(b.text for _, b in passes), verified_names=terms,
        clip_duration=audio.duration, judge=judge)
    for span_id, dispute in by_span.items():
        dispute.decision = decisions[span_id]

    tokens: list[str] = []
    status: list[str] = []
    chunk_of: list[int] = []
    spans: list[dict[str, Any]] = []
    for index, (primary, _precision) in enumerate(passes):
        t, s, sp = _freeze_chunk(primary.tokens, per_chunk[index], len(tokens))
        tokens.extend(t)
        status.extend(s)
        chunk_of.extend([index] * len(t))
        spans.extend(sp)

    primary_count = sum(len(p.tokens) for p, _ in passes)
    safety: dict[str, Any] = {"reverted": False}
    if primary_count >= 6 and len(tokens) / max(1, primary_count) < MIN_LEXICAL_COVERAGE:
        safety = {"reverted": True, "reason": "lexical_coverage",
                  "coverage": round(len(tokens) / max(1, primary_count), 4)}
        for dispute in all_disputes:
            dispute.decision = caption_judge.SpanDecision(dispute.span_id, None, False, "safety_revert_primary")
        tokens, status, chunk_of, spans = [], [], [], []
        for index, (primary, _precision) in enumerate(passes):
            t, s, sp = _freeze_chunk(primary.tokens, per_chunk[index], len(tokens))
            tokens.extend(t)
            status.extend(s)
            chunk_of.extend([index] * len(t))
            spans.extend(sp)
    if not tokens:
        raise LexicalUnavailable("the resolved caption transcript has no words")

    names = clean_verified_terms(orthography_names) if orthography_names is not None else terms
    tokens, orthography = apply_vocative_name_orthography(tokens, names, similarity=name_similarity)
    frozen_spans = tuple(MappingProxyType(dict(s)) for s in spans)
    language = next((p.language for p, _ in passes if p.language), settings.language)
    frozen = FrozenTranscript(tuple(tokens), tuple(status), tuple(chunk_of), tuple(tuple(c) for c in chunks),
                              frozen_spans, language, lexical_signature(tokens, chunks))
    report = {
        "version": LEXICAL_VERSION,
        "passes": [ear.summary() for pair in passes for ear in pair],
        "agreement": [round(score, 4) for score in agreements],
        "disputes": len(all_disputes),
        "escalated": len(escalated),
        "not_escalated": len(overflow),
        "material_disagreement": bool(material),
        "fallback_ear": fallback_ear if use_fallback and all_disputes else "none",
        "fallback_calls": dict(calls),
        "settled_by_agreement": sum(1 for d in all_disputes if str(d.decision.source).startswith("agreement_")),
        "judge": judge_meta,
        "changed_spans": sum(1 for s in spans if s["changed"]),
        "uncertain_spans": sum(1 for s in spans if not s["resolved"]),
        "uncertain_words": sum(1 for s in status if s == "uncertain"),
        "orthography": orthography,
        "safety": safety,
        "errors": errors,
        "verified_terms": terms,
        "spans": [dict(span) for span in spans],
        "frozen_tokens": list(frozen.tokens),
        "chunks": [list(c) for c in frozen.chunks],
        "signature": frozen.signature,
    }
    return frozen, report


def listen_window(audio: analysis_audio.AnalysisAudio, start: float, end: float, *, prompted: bool,
                  vocabulary: Sequence[str], settings: CaptionStackSettings, ears: Ears | None = None,
                  workdir: Path) -> tuple[str, str, str]:
    """One local ear on a few seconds of the analysis audio -> (text, provider, model).

    Unprompted: the model-diverse fallback ear (no names) when configured, else the
    Qwen primary ear. Prompted: the Qwen precision ear with ``vocabulary`` as spelling
    references, else the fallback precision ear.
    """
    ears = ears or Ears()
    path = analysis_audio.cut_window(audio, start, end, workdir / f"window_{int(start * 1000):08d}_"
                                                                  f"{int(end * 1000):08d}_{int(prompted)}.wav")[0]
    fallback = settings.transcribe_fallback_provider == "openai_transcribe"
    use_qwen = settings.primary_provider == "qwen_omni" and (prompted or not fallback)
    if use_qwen:
        domain = openai_ears.domain_keywords()
        instructions = (qwen_omni.precision_instructions(settings.language, domain, vocabulary) if prompted
                        else qwen_omni.primary_instructions(settings.language, domain))
        try:
            evidence, meta = ears.qwen(path, instructions, settings)
            return evidence.text, "qwen_omni", str(meta.get("model", settings.qwen_model))
        except (qwen_omni.QwenUnavailable, qwen_omni.QwenInvalidResponse):
            if not fallback:
                raise
    model = settings.transcribe_fallback_model
    text = ears.openai(path, prompted=prompted, model=model, language=settings.language,
                       verified_terms=list(vocabulary))
    return text, "openai_transcribe", model
