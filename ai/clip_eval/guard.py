"""The holdout never reaches an AI.

Holdout videos exist to catch overfitting. Once their transcripts, labels or per-video scores reach an
optimizer, a prompt, or any report an AI reads, they are no longer held out. Everything meant for an
AI goes through this module:

    optimization_cases(root)    the cases an optimizer may use: dev only
    require_dev(cases)          refuses holdout cases (the LLM judge and the calibration set use it)
    ai_view(score)              a dev-only copy of a scored run: per-video scores + textual feedback
    assert_ai_safe(payload)     refuses a payload that mentions a holdout case id, video file, hash,
                                label note or a stretch of a holdout transcript
    write_feedback(run)         <run>/feedback.dev.json: ``ai_view`` checked by ``assert_ai_safe``

Human reports (report.md, compare.md) contain holdout results and carry a banner saying so.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ai.clip_eval import cases as case_store

FEEDBACK_NAME = "feedback.dev.json"
SHINGLE_WORDS = 8
SHINGLE_STEP = 40


class HoldoutLeak(RuntimeError):
    """Holdout data was about to reach something an AI reads."""


def holdout_cases(root: str | Path | None = None) -> list[case_store.Case]:
    return case_store.list_cases(root, split="holdout")


def optimization_cases(root: str | Path | None = None) -> list[case_store.Case]:
    """The only cases an optimization loop may read, run or learn from."""
    return case_store.list_cases(root, split="dev")


def require_dev(cases: Iterable[case_store.Case]) -> list[case_store.Case]:
    cases = list(cases)
    held = [c.case_id for c in cases if c.split != "dev"]
    if held:
        raise HoldoutLeak(f"holdout case(s) {', '.join(held)} may not be shown to an AI (judge / optimizer)")
    return cases


def _shingles(words: Sequence[str]) -> list[str]:
    out = []
    for start in range(0, max(0, len(words) - SHINGLE_WORDS + 1), SHINGLE_STEP):
        text = " ".join(words[start:start + SHINGLE_WORDS]).strip()
        if len(text) >= 30:
            out.append(text.casefold())
    return out


def holdout_markers(root: str | Path | None = None) -> dict[str, list[str]]:
    """Strings that identify holdout data: ids / names / hashes (whole-token match) and text (substring)."""
    tokens: set[str] = set()
    texts: set[str] = set()
    for case in holdout_cases(root):
        source = case.meta.get("source") or {}
        tokens.update(x for x in (case.case_id, source.get("sha256"), (case.meta.get("transcript") or {}).get("sha256"))
                      if x)
        stem = Path(str(source.get("video", ""))).stem
        if len(stem) >= 6:
            tokens.add(stem)
        for value in (source.get("video"), source.get("name")):
            if value and len(str(value)) >= 8:
                texts.add(str(value).casefold())
        for row in _notes(case):
            if len(row) >= 20:
                texts.add(row.casefold())
        texts.update(_shingles([w for _s, _e, w in case.words() if w]))
    # A marker that also occurs in dev data does not identify the holdout (common phrases, shared notes).
    dev_text = "\n".join(" ".join(w for _s, _e, w in c.words() if w) + "\n" + "\n".join(_notes(c))
                         for c in optimization_cases(root)).casefold()
    return {"tokens": sorted(tokens), "texts": sorted(t for t in texts if t not in dev_text)}


def _notes(case: case_store.Case) -> list[str]:
    out: list[str] = []
    for name in case.annotators():
        labels = case.labels(name)
        if labels:
            out += [row.note for row in [*labels.moments, *labels.negatives] if row.note]
    return out


def assert_ai_safe(payload: Any, root: str | Path | None = None) -> None:
    """Raise HoldoutLeak when the payload mentions any holdout case, video, hash, note or transcript text."""
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, default=str)
    folded = text.casefold()
    markers = holdout_markers(root)
    for token in markers["tokens"]:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", text):
            raise HoldoutLeak(f"holdout identifier {token[:16]!r} found in a payload meant for an AI")
    for snippet in markers["texts"]:
        if snippet in folded:
            raise HoldoutLeak(f"holdout text ({snippet[:24]!r}...) found in a payload meant for an AI")


def ai_view(score: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a scored run an AI may read: dev videos only, with their textual feedback."""
    dev = (score.get("splits") or {}).get("dev") or {}
    configuration = score.get("configuration") or {}
    cases = []
    for case in dev.get("cases") or []:
        if case.get("split") != "dev":
            raise HoldoutLeak(f"case {case.get('case_id')} in the dev section is not a dev case")
        cases.append({"case_id": case["case_id"], "metrics": case.get("metrics"), "feedback": case.get("feedback"),
                      "repeats": [{"repeat": r.get("repeat"), "top": r.get("top"), "feedback": r.get("feedback")}
                                  for r in case.get("repeat_rows") or []]})
    return {"kind": "mimir clip eval feedback (dev split only)", "run": score.get("run"),
            "variant": score.get("variant"), "variant_env": score.get("variant_env") or {},
            "routing": configuration.get("routing"), "settings": configuration.get("settings"),
            "metrics_version": score.get("metrics_version"), "split": "dev",
            "aggregate": dev.get("aggregate"), "cases": cases}


def write_feedback(name: str, *, root: str | Path | None = None, out: str | Path | None = None) -> Path:
    from ai.clip_eval import runner

    view = ai_view(runner.score_run(name, root=root))
    assert_ai_safe(view, root)
    target = Path(out) if out else runner.run_dir(name, root) / FEEDBACK_NAME
    case_store._write_json(target, view)
    return target
