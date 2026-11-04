"""Tier 2: a pairwise LLM judge from another model family, behind ``compare --judge``.

For one video the judge sees the published clip of run A and of run B: its duration, its transcript
with times relative to the clip start, and the few words right outside each cut (marked as NOT in the
clip). It answers which makes the better Short, or a tie, after checking four narrow criteria. No
absolute scores. Each pair is asked in both orders; when the two answers disagree, the pair is a tie
(position-bias control). Identical clips (tIoU >= 0.9) are a tie without a call.

* Model: ``MIMIR_CLIP_EVAL_JUDGE_MODEL`` (default: ``VIDEO_BRAIN_MODEL``, else gemini-3.7-flash). MIMIR's
  generators are OpenAI models (Luna, Terra, Astra), so an OpenAI model is refused as judge.
* Only dev videos are ever sent (``guard.require_dev``): the judge is an AI reading transcripts.
* Verdicts are cached in ``<root>/judge/cache`` by (model, prompt version, both clips): re-running is free.
* Trust: ``calibrate`` measures Cohen's kappa between this judge and a human on blind pairs; below 0.6
  (or never calibrated) reports mark the verdicts UNTRUSTED and the gate ignores them.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ai.clip_eval import cases as case_store
from ai.clip_eval import guard, metrics, pricing

PROMPT_VERSION = 1
DEFAULT_MODEL = "gemini-3.7-flash"
MODEL_ENV = "MIMIR_CLIP_EVAL_JUDGE_MODEL"
IDENTICAL_TIOU = 0.9
CONTEXT_WORDS = 8
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt", "text-", "davinci", "omni-")

RUBRIC = """You compare two candidate clips cut from the same livestream / video. Each is meant to be
published on its own as a vertical Short. Decide which one makes the better Short, or say tie.

You only get each clip's transcript (times in seconds from the clip's first frame), its duration, and a
few words right before and after it that the viewer will NOT see. Judge only what the viewer would get.

Check these four criteria for EACH clip, answering yes / no / unclear:

1. hook_3s - Something in the first 3 seconds makes a scrolling viewer stay: a surprising line, a
   question, a loud reaction, the action already starting.
   good: "[0.4] WAIT he's actually going for it" | bad: "[0.3] so um yeah anyway let me check my settings"
2. standalone - It makes sense to someone who never saw the stream: the setup it needs is inside it.
   good: "[1.0] I bet you can't hit this ... [9.5] NO WAY he hit it" | bad: "[0.5] that's exactly what I
   said about him earlier" (refers to something outside the clip)
3. payoff_inside - The moment the clip builds up to (the punchline, the result, the reveal) happens
   inside the clip, with a beat of reaction after it.
   good: ends 1-2 s after the punchline and the laugh | bad: ends on "and the winner is-" (the payoff is
   in the words AFTER the clip)
4. clean_cut - It starts at the beginning of a sentence and ends after one; it does not cut into a word
   or a sentence.
   good: starts "[0.0] Okay watch this" and ends "...let's go!" | bad: starts "...ing the boss and then"
   or ends "and then he"

Then decide: the clip that satisfies more of these criteria, and more convincingly, is the better Short.
If they are genuinely equal, answer tie. Do not prefer a clip for being first or second, longer or
shorter. Do not give scores. Reply with JSON only."""

_CRITERION = {"type": "STRING", "enum": ["yes", "no", "unclear"]}
_CLIP = {"type": "OBJECT", "properties": {k: _CRITERION for k in ("hook_3s", "standalone", "payoff_inside",
                                                                     "clean_cut")},
         "required": ["hook_3s", "standalone", "payoff_inside", "clean_cut"]}
SCHEMA = {"type": "OBJECT",
          "properties": {"first": _CLIP, "second": _CLIP,
                         "better": {"type": "STRING", "enum": ["first", "second", "tie"]},
                         "reason": {"type": "STRING"}},
          "required": ["first", "second", "better", "reason"]}


def judge_model() -> str:
    model = (os.getenv(MODEL_ENV) or os.getenv("VIDEO_BRAIN_MODEL") or DEFAULT_MODEL).strip()
    if model.lower().startswith(_OPENAI_PREFIXES):
        raise ValueError(f"{MODEL_ENV}={model}: the judge must come from another model family than MIMIR's "
                         "OpenAI generators (use a Gemini model)")
    return model


def clip_view(case: case_store.Case, top: Sequence[float]) -> dict[str, Any]:
    """What the judge sees of one clip: relative timestamps, duration, the words outside each cut."""
    start, end = float(top[0]), float(top[1])
    words = case.words()
    inside = [(s, w) for s, e, w in words if w and start <= (s + e) / 2.0 <= end]
    before = [w for s, e, w in words if w and (s + e) / 2.0 < start][-CONTEXT_WORDS:]
    after = [w for s, e, w in words if w and (s + e) / 2.0 > end][:CONTEXT_WORDS]
    lines: list[str] = []
    current: list[str] = []
    stamp = None
    for s, w in inside:                                    # one line per ~3 s of speech
        rel = max(0.0, s - start)
        if stamp is None or rel - stamp >= 3.0:
            if current:
                lines.append(f"[{stamp:.1f}] " + " ".join(current))
            current, stamp = [], rel
        current.append(w)
    if current:
        lines.append(f"[{stamp:.1f}] " + " ".join(current))
    return {"duration_s": round(end - start, 1), "transcript": "\n".join(lines) or "(no speech)",
            "words_before_not_shown": " ".join(before), "words_after_not_shown": " ".join(after)}


def build_prompt(first: Mapping[str, Any], second: Mapping[str, Any]) -> str:
    def block(name: str, view: Mapping[str, Any]) -> str:
        return (f"=== {name} clip ({view['duration_s']} s) ===\n"
                f"(before the clip, NOT shown: ...{view['words_before_not_shown']})\n{view['transcript']}\n"
                f"(after the clip, NOT shown: {view['words_after_not_shown']}...)")
    return block("first", first) + "\n\n" + block("second", second) + "\n\nWhich makes the better Short?"


Client = Callable[[str, str, str, Mapping[str, Any]], tuple[dict[str, Any], dict[str, Any]]]


def gemini_client(model: str, system: str, prompt: str, schema: Mapping[str, Any]
                  ) -> tuple[dict[str, Any], dict[str, Any]]:
    """One Gemini REST call (standard library only) -> (parsed JSON, usage)."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        env_file = case_store.PROJECT_ROOT / ".env"
        if env_file.is_file():
            from dotenv import dotenv_values

            key = str(dotenv_values(env_file).get("GEMINI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (.env)")
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "responseSchema": schema}}
    request = urllib.request.Request(GEMINI_URL.format(model=model), data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key},
                                     method="POST")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"Gemini HTTP {error.code}: {detail}") from None
    parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [{}])
    text = "".join(str(p.get("text", "")) for p in parts)
    usage = data.get("usageMetadata") or {}
    return json.loads(text), {"input_tokens": int(usage.get("promptTokenCount") or 0),
                              "output_tokens": int(usage.get("candidatesTokenCount") or 0)
                              + int(usage.get("thoughtsTokenCount") or 0)}


def _cache_path(root: str | Path | None, model: str, first: Mapping[str, Any], second: Mapping[str, Any]) -> Path:
    key = hashlib.sha256(json.dumps([model, PROMPT_VERSION, first, second], ensure_ascii=False,
                                    sort_keys=True).encode("utf-8")).hexdigest()
    return case_store.eval_root(root) / "judge" / "cache" / f"{key}.json"


class Session:
    """Judge calls with a hard budget: refused before a call the remaining budget cannot cover."""

    def __init__(self, *, root: str | Path | None, max_usd: float | None, dry_run: bool,
                 client: Client | None = None, model: str | None = None) -> None:
        self.root = root
        self.model = model or judge_model()
        self.prices = pricing.load_prices()
        self.max_usd = max_usd
        self.dry_run = dry_run
        self.client = client or gemini_client
        self.spent = 0.0
        self.calls = 0
        self.planned = 0
        self.planned_usd = 0.0
        self.refused = ""

    def estimate(self, prompt: str) -> float | None:
        return pricing.call_estimate(self.prices, self.model, input_chars=len(RUBRIC) + len(prompt), effort="low",
                                     output_tokens=600)

    def ask(self, first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any] | None:
        """'first' | 'second' | 'tie' verdict (cached), or None when not asked (dry run / no budget)."""
        target = _cache_path(self.root, self.model, first, second)
        cached = case_store._read_json(target)
        if isinstance(cached, Mapping) and cached.get("better") in ("first", "second", "tie"):
            return dict(cached)
        prompt = build_prompt(first, second)
        estimate = self.estimate(prompt)
        if estimate is None:
            self.refused = f"{self.model} has no price in pricing.json; judge calls cannot be capped"
            return None
        self.planned += 1
        self.planned_usd += estimate
        if self.dry_run:
            return None
        if self.max_usd is None:
            self.refused = "uncached judge pairs need --max-usd (or --dry-run for the estimate)"
            return None
        if self.spent + estimate > float(self.max_usd) + 1e-9:
            self.refused = f"budget: next judge call ~${estimate:.3f}, spent ${self.spent:.3f} of ${self.max_usd:.2f}"
            return None
        if self.client is gemini_client and os.getenv(pricing.OFFLINE_ENV, "").strip().lower() in {"1", "true",
                                                                                                    "yes", "on"}:
            raise pricing.OfflineCallBlocked(f"judge call to {self.model} attempted with {pricing.OFFLINE_ENV}=1")
        answer, usage = self.client(self.model, RUBRIC, prompt, SCHEMA)
        cost = pricing.token_cost(self.prices, self.model, **usage)
        self.spent += float(cost if cost is not None else estimate)
        self.calls += 1
        if answer.get("better") not in ("first", "second", "tie"):
            raise RuntimeError(f"judge answered {answer.get('better')!r}, not first / second / tie")
        record = {**answer, "model": self.model, "prompt_version": PROMPT_VERSION, "usage": usage,
                  "usd": cost, "at": case_store._now()}
        case_store._write_json(target, record)
        return record


def _to_ab(answer: Mapping[str, Any] | None, first_is: str) -> str | None:
    if answer is None:
        return None
    better = answer["better"]
    if better == "tie":
        return "tie"
    return first_is if better == "first" else ("B" if first_is == "A" else "A")


def judge_pair(session: Session, case: case_store.Case, top_a: Sequence[float], top_b: Sequence[float]
               ) -> dict[str, Any]:
    """A vs B for one video, both orders -> {verdict: A | B | tie | None, ab, ba, reason}."""
    guard.require_dev([case])
    if metrics.tiou((top_a[0], top_a[1]), (top_b[0], top_b[1])) >= IDENTICAL_TIOU:
        return {"verdict": "tie", "identical": True, "ab": "tie", "ba": "tie", "reason": "same clip"}
    view_a, view_b = clip_view(case, top_a), clip_view(case, top_b)
    first = session.ask(view_a, view_b)                      # A shown first
    second = session.ask(view_b, view_a)                     # B shown first
    ab, ba = _to_ab(first, "A"), _to_ab(second, "B")
    if ab is None or ba is None:
        return {"verdict": None, "ab": ab, "ba": ba, "reason": ""}
    verdict = ab if ab == ba else "tie"
    return {"verdict": verdict, "ab": ab, "ba": ba, "order_disagreement": ab != ba,
            "reason": str((first or {}).get("reason", ""))[:300]}


def compare_runs(result_a: Mapping[str, Any], result_b: Mapping[str, Any], *, root: str | Path | None = None,
                 max_usd: float | None = None, dry_run: bool = False, client: Client | None = None
                 ) -> dict[str, Any]:
    """Judge the published clips of two scored runs on every dev video (paired repeats)."""
    from ai.clip_eval import calibrate

    session = Session(root=root, max_usd=max_usd, dry_run=dry_run, client=client)
    cases = {c.case_id: c for c in guard.optimization_cases(root)}
    rows_a = {c["case_id"]: c for c in (result_a["splits"]["dev"]["cases"])}
    rows_b = {c["case_id"]: c for c in (result_b["splits"]["dev"]["cases"])}
    rows: list[dict[str, Any]] = []
    for case_id in sorted(set(rows_a) & set(rows_b) & set(cases)):
        tops_a = [r.get("top") for r in rows_a[case_id].get("repeat_rows") or []]
        tops_b = [r.get("top") for r in rows_b[case_id].get("repeat_rows") or []]
        for index, (top_a, top_b) in enumerate(zip(tops_a, tops_b), start=1):
            if not top_a or not top_b:
                continue
            row = judge_pair(session, cases[case_id], top_a, top_b)
            rows.append({"case_id": case_id, "repeat": index, **row})
    decided = [r for r in rows if r.get("verdict")]
    trust = calibrate.trust(session.model, root=root)
    return {"model": session.model, "prompt_version": PROMPT_VERSION, "pairs": len(decided),
            "wins": sum(1 for r in decided if r["verdict"] == "B"),
            "losses": sum(1 for r in decided if r["verdict"] == "A"),
            "ties": sum(1 for r in decided if r["verdict"] == "tie"),
            "identical": sum(1 for r in decided if r.get("identical")),
            "order_disagreements": sum(1 for r in decided if r.get("order_disagreement")),
            "undecided": len(rows) - len(decided), "rows": rows, "spent_usd": session.spent, "calls": session.calls,
            "planned_calls": session.planned, "planned_usd": session.planned_usd, "refused": session.refused,
            "dry_run": dry_run, "calibration": trust, "trusted": bool(trust and trust.get("trusted"))}
