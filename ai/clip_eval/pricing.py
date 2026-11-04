"""What a run will cost, what it did cost, and the hard ``--max-usd`` cap.

Prices live in ``pricing.json`` (override: ``MIMIR_CLIP_EVAL_PRICING=<file>``, merged over it).

* ``estimate_*``: cost before anything runs. A case that was already run uses its MEASURED cost
  (same models and efforts); otherwise a transcript-length heuristic, rounded up.
* ``Meter``: installed inside the worker subprocess around the OpenAI SDK
  (``responses.create``, ``chat.completions.create``, ``audio.transcriptions.create``). Before every
  call it estimates that call and refuses it (``BudgetExceeded``) if the remaining budget cannot
  cover it; after the call it books the real ``usage``. One call can therefore overshoot the cap by
  at most the difference between its estimate and its real cost. ``MIMIR_CLIP_EVAL_OFFLINE=1``
  refuses every call (the tests set it).
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

PRICING_PATH = Path(__file__).with_name("pricing.json")
OFFLINE_ENV = "MIMIR_CLIP_EVAL_OFFLINE"


class BudgetExceeded(BaseException):
    """A model call the remaining budget cannot cover was refused.

    A BaseException on purpose: Stage 2 wraps its optional passes (story expansion, boundary
    polish) in ``except Exception`` and would otherwise carry on with a fallback, producing output
    that is not what the variant really does. The run records the job as ``budget_stopped``."""


class OfflineCallBlocked(BudgetExceeded):
    """A model call was attempted while ``MIMIR_CLIP_EVAL_OFFLINE=1``."""


def load_prices(path: str | Path | None = None) -> dict[str, Any]:
    data = json.loads(PRICING_PATH.read_text(encoding="utf-8"))
    override = path or os.getenv("MIMIR_CLIP_EVAL_PRICING", "")
    if override:
        extra = json.loads(Path(override).read_text(encoding="utf-8-sig"))
        for key, value in extra.items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key] = {**data[key], **value}
            else:
                data[key] = value
        data["override"] = str(override)
    return data


def _token_price(prices: Mapping[str, Any], model: str) -> Mapping[str, float] | None:
    table = prices.get("per_million_tokens") or {}
    if model in table:
        return table[model]
    for name, row in table.items():                 # dated snapshots: gpt-5.6-terra-2026-07-30 -> gpt-5.6-terra
        if model.startswith(name + "-"):
            return row
    return None


def is_priced(prices: Mapping[str, Any], model: str) -> bool:
    return _token_price(prices, model) is not None or model in (prices.get("per_audio_minute") or {})


def token_cost(prices: Mapping[str, Any], model: str, *, input_tokens: float = 0, cached_tokens: float = 0,
               output_tokens: float = 0) -> float | None:
    row = _token_price(prices, model)
    if row is None:
        return None
    cached = min(float(cached_tokens or 0), float(input_tokens or 0))
    fresh = float(input_tokens or 0) - cached
    return (fresh * float(row["input"]) + cached * float(row.get("cached_input", row["input"]))
            + float(output_tokens or 0) * float(row["output"])) / 1_000_000.0


def audio_cost(prices: Mapping[str, Any], model: str, seconds: float) -> float | None:
    rate = (prices.get("per_audio_minute") or {}).get(model)
    return None if rate is None else float(rate) * max(0.0, float(seconds)) / 60.0


def _heuristics(prices: Mapping[str, Any]) -> Mapping[str, Any]:
    return prices.get("estimate") or {}


def reasoning_tokens(prices: Mapping[str, Any], effort: str) -> int:
    table = _heuristics(prices).get("reasoning_tokens") or {}
    return int(table.get(str(effort or "medium").lower(), table.get("medium", 6144)))


def call_estimate(prices: Mapping[str, Any], model: str, *, input_chars: int, effort: str, output_tokens: int
                  ) -> float | None:
    """Conservative cost of one Responses call: chars -> tokens, reasoning by effort, safety factor."""
    per_char = float(_heuristics(prices).get("tokens_per_char", 0.32))
    safety = float(_heuristics(prices).get("safety_factor", 1.25))
    cost = token_cost(prices, model, input_tokens=input_chars * per_char,
                      output_tokens=reasoning_tokens(prices, effort) + output_tokens)
    return None if cost is None else cost * safety


# ============================================================
# ESTIMATES BEFORE A RUN
# ============================================================

# (call, role, probability it happens, extra input chars beyond the transcript, JSON output tokens,
#  whether the call carries the whole timestamped transcript) - mirrors analyze_transcript()
CLIP_CALLS = (
    ("money_moments", "scout", 1.0, 3_000, 3_000, True),
    ("visual_audit_repair", "scout", 0.3, 3_000, 1_500, True),
    ("clips", "scout", 1.0, 14_000, 5_000, True),
    ("final_judge", "judge", 1.0, 18_000, 2_500, True),
    ("final_judge_escalation", "escalation", 0.35, 18_000, 2_500, True),
    ("story_expansion", "judge", 0.5, 9_000, 2_000, True),
    ("boundary_polish", "scout", 1.0, 8_000, 800, False),
)


def estimate_clip_job(prices: Mapping[str, Any], configuration: Mapping[str, Any], stats: Mapping[str, Any]
                      ) -> dict[str, Any]:
    """Heuristic cost of ONE Stage 2 run on one case -> {expected, worst, unpriced}."""
    routing = configuration.get("routing") or {}
    instructions = configuration.get("instruction_chars") or {}
    transcript_chars = int(stats.get("transcript_chars", 0) or 0)
    visual_chars = int(stats.get("visual_chars", 0) or 0)
    expected = worst = 0.0
    unpriced: set[str] = set()
    rows = []
    for call, role, probability, extra, out_tokens, full in CLIP_CALLS:
        if call == "visual_audit_repair" and not stats.get("has_visual_report"):
            continue
        route = routing.get(role) or {}
        model, effort = str(route.get("model", "")), str(route.get("effort", "medium"))
        chars = extra + int(instructions.get(call, 4_000)) + (transcript_chars + visual_chars if full else 0)
        cost = call_estimate(prices, model, input_chars=chars, effort=effort, output_tokens=out_tokens)
        if cost is None:
            unpriced.add(model)
            continue
        expected += probability * cost
        worst += cost
        rows.append({"call": call, "model": model, "effort": effort, "input_chars": chars, "usd": round(cost, 4),
                     "probability": probability})
    return {"expected": expected, "worst": worst, "unpriced": sorted(unpriced), "calls": rows, "basis": "heuristic"}


def estimate_transcribe(prices: Mapping[str, Any], duration_s: float, *, text_model: str = "gpt-4o-mini-transcribe",
                        timing_model: str = "whisper-1") -> dict[str, Any]:
    """Stage 1: the whole VOD through the text model and the word-timing model (per audio minute)."""
    parts = {model: audio_cost(prices, model, duration_s) for model in (text_model, timing_model)}
    unpriced = sorted(m for m, v in parts.items() if v is None)
    total = sum(v for v in parts.values() if v is not None)
    return {"expected": total, "worst": total * 1.1, "unpriced": unpriced, "basis": "per audio minute",
            "calls": [{"call": "transcription", "model": m, "usd": None if v is None else round(v, 4)}
                      for m, v in parts.items()]}


def estimate_full_job(prices: Mapping[str, Any], configuration: Mapping[str, Any], stats: Mapping[str, Any],
                      clip: Mapping[str, Any]) -> dict[str, Any]:
    """The whole pipeline on one VOD: Stage 1 + whole-VOD Gemini facts + Stage 2 + one short."""
    duration = float(stats.get("duration", 0.0) or 0.0)
    full = (_heuristics(prices).get("full_stage") or {})
    stage1 = estimate_transcribe(prices, duration, text_model=str(configuration.get("vod_text_model")
                                                                    or "gpt-4o-mini-transcribe"))
    gemini_model = str(configuration.get("video_brain_model") or "gemini-3.7-flash")
    gemini = 0.0
    unpriced = set(stage1["unpriced"]) | set(clip.get("unpriced") or [])
    if configuration.get("video_brain_enabled", True):
        cost = token_cost(prices, gemini_model, input_tokens=duration * float(full.get("gemini_video_tokens_per_second",
                                                                                       300)),
                          output_tokens=float(full.get("gemini_output_tokens", 8000)))
        if cost is None:
            unpriced.add(gemini_model)
        else:
            gemini = cost * float(_heuristics(prices).get("safety_factor", 1.25))
    after = float(full.get("after_selection_usd", 2.5))
    expected = stage1["expected"] + gemini + float(clip.get("expected", 0.0)) + after
    worst = stage1["worst"] + gemini + float(clip.get("worst", 0.0)) + after * 2.0
    return {"expected": expected, "worst": worst, "unpriced": sorted(unpriced), "basis": "heuristic (full pipeline)",
            "unmetered_usd": stage1["worst"] + gemini,
            "calls": [{"call": "stage1_transcription", "usd": round(stage1["expected"], 4)},
                      {"call": f"video_brain:{gemini_model}", "usd": round(gemini, 4)},
                      {"call": "stage2", "usd": round(float(clip.get("expected", 0.0)), 4)},
                      {"call": "after_selection_allowance", "usd": after}]}


# ============================================================
# METER (inside the worker process)
# ============================================================

def _text_len(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return len(str(value))


def _get(obj: Any, *names: str) -> Any:
    for name in names:
        if obj is None:
            return None
        obj = obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)
    return obj


class Meter:
    """Books every OpenAI SDK call of this process and enforces the budget before each one."""

    def __init__(self, prices: Mapping[str, Any], *, budget_usd: float | None, reserved_usd: float = 0.0,
                 strict_unpriced: bool = True, offline: bool | None = None) -> None:
        self.prices = prices
        self.budget = None if budget_usd is None else float(budget_usd)
        self.reserved = float(reserved_usd or 0.0)
        self.strict_unpriced = strict_unpriced
        self.offline = (os.getenv(OFFLINE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
                        if offline is None else offline)
        self.calls: list[dict[str, Any]] = []
        self.spent = 0.0
        self.refused: str | None = None
        self._lock = threading.Lock()
        self._originals: list[tuple[Any, str, Any]] = []

    # -- policy ----------------------------------------------------------
    def check(self, kind: str, model: str, estimate: float | None, *, reserved: bool = False) -> None:
        with self._lock:
            if self.offline:
                self.refused = f"{kind} call to {model or '?'} attempted with {OFFLINE_ENV}=1"
                raise OfflineCallBlocked(self.refused)
            if self.refused:
                raise BudgetExceeded(self.refused)
            if reserved:
                return
            if estimate is None:
                if self.strict_unpriced:
                    self.refused = f"{model or '?'} has no price in pricing.json; its cost cannot be capped"
                    raise BudgetExceeded(self.refused)
                return
            if self.budget is not None and self.spent + self.reserved + estimate > self.budget + 1e-9:
                self.refused = (f"next {kind} call to {model} (~${estimate:.3f}) would pass the budget: spent "
                                f"${self.spent:.3f} + reserved ${self.reserved:.3f} of ${self.budget:.2f}")
                raise BudgetExceeded(self.refused)

    def book(self, row: dict[str, Any]) -> None:
        with self._lock:
            self.calls.append(row)
            if row.get("usd") is not None and not row.get("reserved"):
                self.spent += float(row["usd"])

    def summary(self) -> dict[str, Any]:
        totals = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
        by_model: dict[str, dict[str, Any]] = {}
        for row in self.calls:
            for key in totals:
                totals[key] += int(row.get(key) or 0)
            entry = by_model.setdefault(row.get("model", "?"), {"calls": 0, "usd": 0.0})
            entry["calls"] += 1
            entry["usd"] = round(entry["usd"] + float(row.get("usd") or 0.0), 6)
        return {"calls": len(self.calls), "spent_usd": round(self.spent, 6), **totals, "by_model": by_model,
                "unpriced_calls": sum(1 for r in self.calls if r.get("usd") is None and not r.get("reserved")),
                "estimated_calls": sum(1 for r in self.calls if r.get("basis") == "estimate"),
                "budget_usd": self.budget, "reserved_usd": self.reserved, "refused": self.refused,
                "detail": self.calls}

    # -- SDK patches -----------------------------------------------------
    def install(self) -> "Meter":
        from openai.resources.audio.transcriptions import Transcriptions
        from openai.resources.chat.completions import Completions
        from openai.resources.responses import Responses

        self._patch(Responses, "create", self._wrap_responses)
        self._patch(Completions, "create", self._wrap_chat)
        self._patch(Transcriptions, "create", self._wrap_transcription)
        return self

    def uninstall(self) -> None:
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)
        self._originals.clear()

    def _patch(self, owner: Any, name: str, factory: Callable[[Any], Any]) -> None:
        original = getattr(owner, name)
        self._originals.append((owner, name, original))
        setattr(owner, name, factory(original))

    def _wrap_responses(self, original: Any) -> Any:
        meter = self

        def create(resource: Any, *args: Any, **kwargs: Any) -> Any:
            model = str(kwargs.get("model", ""))
            effort = str(_get(kwargs.get("reasoning"), "effort") or "medium")
            chars = _text_len(kwargs.get("instructions")) + _text_len(kwargs.get("input")) + _text_len(
                kwargs.get("text"))
            estimate = call_estimate(meter.prices, model, input_chars=chars, effort=effort, output_tokens=3_000)
            meter.check("responses", model, estimate)
            started = time.perf_counter()
            response = original(resource, *args, **kwargs)
            usage = _get(response, "usage")
            row: dict[str, Any] = {"kind": "responses", "model": model, "effort": effort, "input_chars": chars,
                                   "seconds": round(time.perf_counter() - started, 2)}
            if usage is not None:
                row.update(input_tokens=int(_get(usage, "input_tokens") or 0),
                           cached_tokens=int(_get(usage, "input_tokens_details", "cached_tokens") or 0),
                           output_tokens=int(_get(usage, "output_tokens") or 0),
                           reasoning_tokens=int(_get(usage, "output_tokens_details", "reasoning_tokens") or 0))
                row["usd"] = token_cost(meter.prices, model, input_tokens=row["input_tokens"],
                                        cached_tokens=row["cached_tokens"], output_tokens=row["output_tokens"])
                row["basis"] = "usage"
            else:
                row.update(usd=estimate, basis="estimate")
            meter.book(row)
            return response
        return create

    def _wrap_chat(self, original: Any) -> Any:
        meter = self

        def create(resource: Any, *args: Any, **kwargs: Any) -> Any:
            model = str(kwargs.get("model", ""))
            chars = _text_len(kwargs.get("messages"))
            estimate = call_estimate(meter.prices, model, input_chars=chars,
                                     effort=str(kwargs.get("reasoning_effort") or "none"), output_tokens=3_000)
            meter.check("chat", model, estimate)
            started = time.perf_counter()
            result = original(resource, *args, **kwargs)
            row: dict[str, Any] = {"kind": "chat", "model": model, "input_chars": chars}

            def finish(usage: Any) -> None:
                row["seconds"] = round(time.perf_counter() - started, 2)
                if usage is not None:
                    row.update(input_tokens=int(_get(usage, "prompt_tokens") or 0),
                               output_tokens=int(_get(usage, "completion_tokens") or 0))
                    row["usd"] = token_cost(meter.prices, model, input_tokens=row["input_tokens"],
                                            output_tokens=row["output_tokens"])
                    row["basis"] = "usage"
                else:
                    row.update(usd=estimate, basis="estimate")
                meter.book(row)

            if kwargs.get("stream"):
                def chunks() -> Iterator[Any]:
                    usage = None
                    try:
                        for chunk in result:
                            usage = _get(chunk, "usage") or usage
                            yield chunk
                    finally:
                        finish(usage)
                return chunks()
            finish(_get(result, "usage"))
            return result
        return create

    def _wrap_transcription(self, original: Any) -> Any:
        meter = self

        def create(resource: Any, *args: Any, **kwargs: Any) -> Any:
            model = str(kwargs.get("model", ""))
            # Billed per audio minute: the caller reserved the whole job's audio up front.
            meter.check("transcription", model, None, reserved=True)
            started = time.perf_counter()
            result = original(resource, *args, **kwargs)
            meter.book({"kind": "transcription", "model": model, "usd": None, "reserved": True,
                        "seconds": round(time.perf_counter() - started, 2), "basis": "reserved"})
            return result
        return create
