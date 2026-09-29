"""Caption judge: lexical truth decided from evidence by the strongest model (Astra).

Four authorities, never merged:

    LEXICAL  WHAT was said  -> this judge, over the evidence of every ear
    TIMING   WHEN           -> the measured word clock only; the judge never emits
                               or edits a time (text goes through the fixed-clock aligner)
    SPEAKER  WHO            -> diarization + the human identity checkpoint
    DISPLAY  HOW            -> the deterministic caption renderer

Two bounded decisions:

1. Disputed spans (caption stage). ONE batched call per short covering every
   span the independent ears did not settle: the primary transcript, the
   model-diverse cross-check with its token confidence, every local micro-ASR
   ear (model, audio view, prompted or not, what it heard in the window and in
   the disputed core), the verified names, why the span is suspect, and the
   measured clock words inside the window. Verdicts: ``keep_primary`` |
   ``use_heard`` | ``unresolved``. A ``use_heard`` text is accepted only when
   every word was heard by some ear or the word clock (or is a verified name),
   the change stays local, and a deletion is backed by a majority of ears that
   heard nothing where the clock has no word. A rejected or missing decision
   falls back to the strict acoustic vote for that span.

2. Verified-name spelling (caption truth freeze). A closed choice per candidate
   word: ``canonical`` (the verified spelling) | ``keep`` | ``uncertain``.
   Word count, id, time and speaker never change.

Spans that >=3 independent ears heard identically are already settled and are
not sent (no strong-model budget where the acoustic evidence is unanimous).
No judge (no key, API error) -> the strict acoustic vote (3/3, then 4/5)
decides, and the run records that degradation. Nothing here guesses.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ai import model_config

JUDGE_VERSION = 1
MIN_CONFIDENCE = float(os.getenv("MIMIR_CAPTION_JUDGE_MIN_CONFIDENCE", "0.6") or 0.6)
MAX_EXTRA_TOKENS = 3             # a span decision is a local repair, never a rewrite
JUDGE_TIMEOUT_S = float(os.getenv("MIMIR_CAPTION_JUDGE_TIMEOUT", "300") or 300)

# judge(name=..., instructions=..., payload=..., schema=...) -> parsed JSON object
Judge = Callable[..., Mapping[str, Any]]

SPAN_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "span_id": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["keep_primary", "use_heard", "unresolved"]},
                    "text": {"type": "string"},
                    "supporting_ears": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["span_id", "verdict", "text", "supporting_ears", "confidence", "reason"],
            },
        },
    },
    "required": ["decisions"],
}

ENTITY_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "word_id": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["canonical", "keep", "uncertain"]},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["word_id", "verdict", "confidence", "reason"],
            },
        },
    },
    "required": ["decisions"],
}

SPAN_INSTRUCTIONS = """
You are the lexical judge for the captions of ONE finished short video. You decide WHAT was said in each
disputed span, using ONLY the evidence given: several independent speech-recognition "ears" listened to the
same audio (full clip and a few seconds around each span), with their models, audio views and whether they
were prompted with context. You also get the measured word clock (words a timing model heard, with times).

Decide per span:
- "keep_primary": the primary transcript's core words are what was said.
- "use_heard": a different wording was said. "text" = the exact core words, taken from what the ears / the
  word clock actually heard (you may pick the best-supported variant or combine words the ears heard; you may
  use a verified name when an ear heard a near-spelling of it). List the ear ids that support it.
- "unresolved": the evidence does not settle it. The primary words stay and are marked uncertain.

Rules:
- The audio evidence decides. Never invent a word no ear heard. Never "improve" grammar, style or slang.
- Unprompted acoustic ears weigh more than context-prompted ones (a prompted ear can echo the context).
- Keep tiny function words and fillers when the ears heard them. Deleting the core requires ears that heard
  nothing there.
- You do NOT decide timing, speakers or display. Never output times.
- Prefer "unresolved" over a fluent guess. confidence = how strongly the evidence supports your verdict (0..1).
- Return one decision for every span_id.
""".strip()

ENTITY_INSTRUCTIONS = """
You decide the SPELLING of possible verified names in the frozen captions of ONE short. Each candidate is a
caption word that sounds like a verified person/entity name. Evidence: the surrounding words, who is speaking,
why the name lock flagged it (direct address, name syntax, co-participant speaker, ...), its similarity score,
and what independent local ears heard at that word (with token confidence when available).

Per candidate choose:
- "canonical": the word IS the verified name -> spell it exactly as verified.
- "keep": the word is an ordinary word (or another name), not the verified entity -> keep the caption word.
- "uncertain": the evidence does not settle it -> the word stays as heard and is marked uncertain.

Only the spelling of that one word can change; timing, speakers and every other word never change.
Prefer "uncertain" over forcing a name. Return one decision per word_id.
""".strip()


def _canon(token: str) -> str:
    from ai import vod_processor

    return vod_processor.canonical_word(token)


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"\S+", str(text or "")) if _canon(t)]


# ============================================================
# EVIDENCE
# ============================================================

@dataclass
class SpanEvidence:
    span_id: str
    core: tuple[int, int]                      # primary-transcript token range [start, end)
    current: str                               # primary core words
    window_text: str
    context_before: str
    context_after: str
    audio_window: tuple[float, float]
    suspicion: list[str]
    ears: list[dict[str, Any]]                 # ear, model, view, prompted, heard_window, heard_core
    strict: dict[str, Any] | None              # strict acoustic vote result (phrase, votes, of) or None
    clock: list[tuple[str, float, float]]      # measured clock words inside the audio window
    core_clock: list[str]                      # measured clock words overlapping the core
    low_confidence: list[tuple[str, float]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "primary_core": self.current,
            "primary_window": self.window_text,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "why_suspect": list(self.suspicion),
            "audio_window_s": [round(self.audio_window[0], 2), round(self.audio_window[1], 2)],
            "ears": [{"ear": e["ear"], "model": e.get("model", ""), "view": e.get("view", ""),
                      "prompted_with_context": bool(e.get("prompted")), "heard_in_window": e.get("heard_window", ""),
                      "heard_at_core": e.get("heard_core", "")} for e in self.ears],
            "strict_acoustic_vote": self.strict,
            "measured_clock_words": [[w, round(a, 2), round(b, 2)] for w, a, b in self.clock],
            "measured_clock_at_core": list(self.core_clock),
            "crosscheck_low_confidence": [[w, round(p, 3)] for w, p in self.low_confidence],
        }

    def heard_vocabulary(self, verified_names: Sequence[str] = ()) -> set[str]:
        """Every canonical word some evidence source actually heard in this window."""
        heard: set[str] = set()
        for text in [self.window_text, self.current, *(e.get("heard_window", "") for e in self.ears),
                     *(e.get("heard_core", "") for e in self.ears), *(w for w, _a, _b in self.clock)]:
            heard.update(_canon(t) for t in _tokens(text))
        for name in verified_names:
            heard.update(_canon(t) for t in _tokens(name))
        heard.discard("")
        return heard

    def ear_ids(self) -> set[str]:
        return {str(e["ear"]) for e in self.ears} | {"clock", "primary"}


@dataclass(frozen=True)
class SpanDecision:
    span_id: str
    phrase: str | None          # replacement core words; None = keep the primary words
    resolved: bool              # False -> primary kept AND marked uncertain downstream
    source: str                 # caption_judge | strict_vote | settled_unanimous
    confidence: float | None = None
    reason: str = ""
    supporting_ears: tuple[str, ...] = ()
    guard: str = ""             # why a judge decision was not accepted (then strict decided)

    def to_dict(self) -> dict[str, Any]:
        return {"span_id": self.span_id, "phrase": self.phrase, "resolved": self.resolved, "source": self.source,
                "confidence": self.confidence, "reason": self.reason[:300],
                "supporting_ears": list(self.supporting_ears), "guard": self.guard}


def strict_decision(span: SpanEvidence, *, guard: str = "") -> SpanDecision:
    """The deterministic acoustic rule: a non-empty strict-majority phrase, else unresolved."""
    strict = span.strict or {}
    phrase = str(strict.get("phrase") or "").strip()
    if phrase:
        return SpanDecision(span.span_id, phrase, True,
                            f"strict_vote_{int(strict.get('votes', 0))}_of_{int(strict.get('of', 0))}", guard=guard)
    return SpanDecision(span.span_id, None, False, "strict_vote", guard=guard)


def check_span_decision(span: SpanEvidence, row: Mapping[str, Any], verified_names: Sequence[str] = ()
                        ) -> tuple[SpanDecision | None, str]:
    """Deterministic guard on one judge decision -> (accepted decision, "") or (None, problem)."""
    verdict = str(row.get("verdict", ""))
    try:
        confidence = float(row.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    reason = str(row.get("reason", ""))
    supporters = tuple(str(e) for e in row.get("supporting_ears", []) or [] if str(e) in span.ear_ids())
    if verdict == "unresolved" or (verdict in ("keep_primary", "use_heard") and confidence < MIN_CONFIDENCE):
        return SpanDecision(span.span_id, None, False, "caption_judge", confidence, reason, supporters), ""
    if verdict == "keep_primary":
        return SpanDecision(span.span_id, None, True, "caption_judge", confidence, reason, supporters), ""
    if verdict != "use_heard":
        return None, f"unknown verdict {verdict!r}"
    new = _tokens(str(row.get("text", "")))
    old = _tokens(span.current)
    if not supporters:
        return None, "use_heard without a supporting ear"
    if len(new) > len(old) + MAX_EXTRA_TOKENS:
        return None, f"not a local repair ({len(old)} -> {len(new)} words)"
    if not new:
        silent = sum(1 for e in span.ears if not _tokens(e.get("heard_core", "")))
        if silent * 2 <= len(span.ears) or span.core_clock:
            return None, "deletion without a majority of ears (and the clock) hearing nothing"
        return SpanDecision(span.span_id, "", True, "caption_judge", confidence, reason, supporters), ""
    vocabulary = span.heard_vocabulary(verified_names)
    unheard = [t for t in new if _canon(t) not in vocabulary]
    if unheard:
        return None, "word(s) no ear heard: " + ", ".join(unheard[:4])
    return SpanDecision(span.span_id, " ".join(new), True, "caption_judge", confidence, reason, supporters), ""


# ============================================================
# MODEL CALL
# ============================================================

def default_judge() -> Judge | None:
    """Astra through the Responses API (strict JSON schema); None without a key."""
    if not os.getenv("OPENAI_API_KEY", "").strip():
        return None
    model = model_config.CAPTION_JUDGE_MODEL
    effort = model_config.CAPTION_JUDGE_REASONING_EFFORT

    def judge(*, name: str, instructions: str, payload: Mapping[str, Any], schema: Mapping[str, Any]
              ) -> Mapping[str, Any]:
        from ai.openai_client import get_client

        client = get_client()
        with_options = getattr(client, "with_options", None)
        client = with_options(timeout=JUDGE_TIMEOUT_S) if callable(with_options) else client
        response = client.responses.create(
            model=model, reasoning={"effort": effort}, instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            text={"format": {"type": "json_schema", "name": name, "strict": True, "schema": dict(schema)}})
        status = str(getattr(response, "status", "completed") or "completed")
        if status in {"failed", "cancelled", "incomplete"}:
            raise RuntimeError(f"caption judge response status {status}")
        data = json.loads(str(getattr(response, "output_text", "") or ""))
        if not isinstance(data, dict):
            raise ValueError("caption judge returned a non-object")
        return data

    judge.model = model            # type: ignore[attr-defined]
    judge.effort = effort          # type: ignore[attr-defined]
    return judge


def _judge_identity(judge: Judge | None) -> dict[str, str]:
    return {"model": str(getattr(judge, "model", "custom")), "effort": str(getattr(judge, "effort", ""))}


def resolve_spans(spans: Sequence[SpanEvidence], *, primary_text: str, crosscheck_text: str,
                  verified_names: Sequence[str] = (), clip_duration: float = 0.0,
                  judge: Judge | None = None) -> tuple[dict[str, SpanDecision], dict[str, Any]]:
    """One judge call for every unsettled span; guard each decision; strict vote on any gap."""
    if not spans:
        return {}, {"status": "not_needed", "version": JUDGE_VERSION}
    meta: dict[str, Any] = {"version": JUDGE_VERSION, "spans": len(spans), **_judge_identity(judge)}
    if judge is None:
        meta.update(status="unavailable", reason="no caption judge (OPENAI_API_KEY missing); strict acoustic vote")
        return {s.span_id: strict_decision(s) for s in spans}, meta
    payload = {
        "clip_duration_s": round(float(clip_duration), 2),
        "primary_transcript": {"role": "primary wording (whole clip)", "text": primary_text},
        "crosscheck_transcript": {"role": "independent model, whole clip", "text": crosscheck_text},
        "verified_names": list(verified_names),
        "spans": [s.to_payload() for s in spans],
    }
    try:
        answer = judge(name="mimir_caption_spans", instructions=SPAN_INSTRUCTIONS, payload=payload,
                       schema=SPAN_JUDGE_SCHEMA)
    except Exception as error:  # the judge is unavailable -> the deterministic rule decides everything
        meta.update(status="failed", reason=f"{type(error).__name__}: {str(error)[:300]}")
        return {s.span_id: strict_decision(s) for s in spans}, meta
    by_id = {str(r.get("span_id")): r for r in answer.get("decisions", []) or [] if isinstance(r, Mapping)}
    decisions: dict[str, SpanDecision] = {}
    rejected: list[dict[str, str]] = []
    for span in spans:
        row = by_id.get(span.span_id)
        if row is None:
            decisions[span.span_id] = strict_decision(span, guard="judge gave no decision")
            rejected.append({"span_id": span.span_id, "problem": "missing"})
            continue
        accepted, problem = check_span_decision(span, row, verified_names)
        if accepted is None:
            decisions[span.span_id] = strict_decision(span, guard=problem)
            rejected.append({"span_id": span.span_id, "problem": problem})
        else:
            decisions[span.span_id] = accepted
    meta.update(status="judged", accepted=len(spans) - len(rejected), guard_rejected=rejected)
    return decisions, meta


# ============================================================
# VERIFIED-NAME SPELLING (caption truth)
# ============================================================

def resolve_entities(candidates: Sequence[Mapping[str, Any]], *, judge: Judge | None
                     ) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Closed-choice spelling decisions -> ({word_id: decision}, meta). Never touches time/speaker/count."""
    if not candidates:
        return {}, {"status": "not_needed", "version": JUDGE_VERSION}
    meta: dict[str, Any] = {"version": JUDGE_VERSION, "candidates": len(candidates), **_judge_identity(judge)}
    if judge is None:
        meta.update(status="unavailable", reason="no caption judge; unverified names stay marked uncertain")
        return {}, meta
    try:
        answer = judge(name="mimir_caption_entities", instructions=ENTITY_INSTRUCTIONS,
                       payload={"candidates": [dict(c) for c in candidates]}, schema=ENTITY_JUDGE_SCHEMA)
    except Exception as error:
        meta.update(status="failed", reason=f"{type(error).__name__}: {str(error)[:300]}")
        return {}, meta
    known = {int(c["word_id"]) for c in candidates}
    decisions: dict[int, dict[str, Any]] = {}
    for row in answer.get("decisions", []) or []:
        if not isinstance(row, Mapping):
            continue
        try:
            word_id = int(row.get("word_id"))
            confidence = float(row.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        verdict = str(row.get("verdict", ""))
        if word_id not in known or verdict not in ("canonical", "keep", "uncertain"):
            continue
        if verdict != "uncertain" and confidence < MIN_CONFIDENCE:
            verdict = "uncertain"
        decisions[word_id] = {"verdict": verdict, "confidence": round(confidence, 3),
                              "reason": str(row.get("reason", ""))[:300]}
    meta.update(status="judged", decided=len(decisions))
    return decisions, meta
