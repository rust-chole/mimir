"""Is the LLM judge worth listening to? Blind human verdicts on judge pairs -> agreement and Cohen's kappa.

    python -m ai.clip_eval calibrate <run_a> <run_b>         add the two runs' dev pairs to the blind set and
                                                              label them in the browser (``--text``: terminal)
    python -m ai.clip_eval calibrate --judge --dry-run       what the judge would cost on the same pairs
    python -m ai.clip_eval calibrate --judge --max-usd 1     the judge answers them (both orders, cached) -> kappa

Everything lives in ``<root>/calibration/``: ``pairs.json`` (which clip is shown as "1" is drawn at random
and never shown; neither are run names), ``human.json`` (your answers: 1 / 2 / tie) and ``result.json``
(agreement and kappa per judge model and prompt version). kappa >= 0.6 makes the judge trusted; reports
and the gate read it from here. Only dev videos and pairs whose two clips differ enter the set.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Callable, Mapping

from ai.clip_eval import cases as case_store
from ai.clip_eval import guard, judge, metrics

TRUST_KAPPA = 0.6
TARGET_PAIRS = 30
ANSWERS = ("1", "2", "tie")


def calib_dir(root: str | Path | None = None) -> Path:
    return case_store.eval_root(root) / "calibration"


def load_pairs(root: str | Path | None = None) -> list[dict[str, Any]]:
    data = case_store._read_json(calib_dir(root) / "pairs.json")
    return [p for p in (data.get("pairs") if isinstance(data, Mapping) else None) or [] if isinstance(p, dict)]


def human_labels(root: str | Path | None = None) -> dict[str, str]:
    data = case_store._read_json(calib_dir(root) / "human.json")
    return {str(k): str(v) for k, v in (data or {}).items() if str(v) in ANSWERS} if isinstance(data, Mapping) else {}


def save_human(pair_id: str, answer: str, *, root: str | Path | None = None) -> None:
    if answer not in ANSWERS:
        raise ValueError(f"answer must be one of {', '.join(ANSWERS)}")
    if pair_id not in {p["id"] for p in load_pairs(root)}:
        raise ValueError(f"no calibration pair {pair_id!r}")
    labels = human_labels(root)
    labels[pair_id] = answer
    case_store._write_json(calib_dir(root) / "human.json", labels)


def add_pairs(result_a: Mapping[str, Any], result_b: Mapping[str, Any], *, root: str | Path | None = None,
              seed: int | None = None) -> int:
    """Add every dev video x paired repeat whose published clips differ; returns how many are new."""
    rng = random.Random(seed)
    pairs = load_pairs(root)
    known = {p["id"] for p in pairs}
    dev = {c.case_id for c in guard.optimization_cases(root)}
    rows_a = {c["case_id"]: c for c in result_a["splits"]["dev"]["cases"]}
    rows_b = {c["case_id"]: c for c in result_b["splits"]["dev"]["cases"]}
    added = 0
    for case_id in sorted(set(rows_a) & set(rows_b) & dev):
        tops_a = [r.get("top") for r in rows_a[case_id].get("repeat_rows") or []]
        tops_b = [r.get("top") for r in rows_b[case_id].get("repeat_rows") or []]
        for top_a, top_b in zip(tops_a, tops_b):
            if not top_a or not top_b or metrics.tiou(tuple(top_a), tuple(top_b)) >= judge.IDENTICAL_TIOU:
                continue
            pair_id = hashlib.sha256(f"{case_id}|{top_a}|{top_b}".encode("utf-8")).hexdigest()[:12]
            if pair_id in known:
                continue
            known.add(pair_id)
            pairs.append({"id": pair_id, "case_id": case_id, "clips": {"A": list(top_a), "B": list(top_b)},
                          "first": rng.choice(("A", "B")), "runs": {"A": result_a.get("run"), "B": result_b.get("run")},
                          "added_at": case_store._now()})
            added += 1
    case_store._write_json(calib_dir(root) / "pairs.json", {"pairs": pairs})
    return added


def shown(pair: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    """(clip shown as 1, clip shown as 2): the side assignment the human never sees."""
    first = pair["first"]
    other = "B" if first == "A" else "A"
    return list(pair["clips"][first]), list(pair["clips"][other])


def human_ab(pair: Mapping[str, Any], answer: str) -> str:
    if answer == "tie":
        return "tie"
    first = pair["first"]
    return first if answer == "1" else ("B" if first == "A" else "A")


def run_judge(*, root: str | Path | None = None, max_usd: float | None = None, dry_run: bool = False,
              client: judge.Client | None = None) -> dict[str, Any]:
    """The judge's verdict on every calibration pair (cached), then the agreement with the human."""
    session = judge.Session(root=root, max_usd=max_usd, dry_run=dry_run, client=client)
    cases = {c.case_id: c for c in guard.optimization_cases(root)}
    verdicts: dict[str, str | None] = {}
    for pair in load_pairs(root):
        case = cases.get(pair["case_id"])
        if case is None:                                 # moved to holdout or deleted since: never judged
            continue
        verdicts[pair["id"]] = judge.judge_pair(session, case, pair["clips"]["A"], pair["clips"]["B"])["verdict"]
    result = compute(session.model, verdicts, root=root)
    result.update(spent_usd=session.spent, calls=session.calls, planned_calls=session.planned,
                  planned_usd=session.planned_usd, refused=session.refused, dry_run=dry_run)
    return result


def compute(model: str, verdicts: Mapping[str, str | None], *, root: str | Path | None = None) -> dict[str, Any]:
    pairs = {p["id"]: p for p in load_pairs(root)}
    human = human_labels(root)
    both = [(str(verdicts[pid]), human_ab(pairs[pid], answer)) for pid, answer in human.items()
            if pid in pairs and verdicts.get(pid)]
    stats = metrics.cohen_kappa(both)
    decisive = [(j, h) for j, h in both if j != "tie" and h != "tie"]
    result = {"model": model, "prompt_version": judge.PROMPT_VERSION, **stats,
              "agreement_without_ties": (sum(1 for j, h in decisive if j == h) / len(decisive)) if decisive else None,
              "human_labeled": len(human), "pairs": len(pairs), "at": case_store._now(),
              "trusted": bool(stats["kappa"] is not None and stats["kappa"] >= TRUST_KAPPA and stats["n"] >= 10)}
    if stats["n"]:
        store = case_store._read_json(calib_dir(root) / "result.json")
        store = dict(store) if isinstance(store, Mapping) else {}
        store[f"{model}|v{judge.PROMPT_VERSION}"] = result
        case_store._write_json(calib_dir(root) / "result.json", store)
    return result


def trust(model: str, *, root: str | Path | None = None) -> dict[str, Any] | None:
    store = case_store._read_json(calib_dir(root) / "result.json")
    if not isinstance(store, Mapping):
        return None
    row = store.get(f"{model}|v{judge.PROMPT_VERSION}")
    return dict(row) if isinstance(row, Mapping) else None


def text_session(*, root: str | Path | None = None, ask: Callable[[str], str] = input,
                 say: Callable[[str], None] = print) -> int:
    """Label unlabeled pairs in the terminal: transcript + time range of clip 1 and clip 2."""
    cases = {c.case_id: c for c in guard.optimization_cases(root)}
    done = human_labels(root)
    todo = [p for p in load_pairs(root) if p["id"] not in done and p["case_id"] in cases]
    labeled = 0
    for number, pair in enumerate(todo, start=1):
        case = cases[pair["case_id"]]
        one, two = shown(pair)
        say(f"\n=== pair {number}/{len(todo)}: {case.video_path.name}")
        for title, top in (("Clip 1", one), ("Clip 2", two)):
            view = judge.clip_view(case, top)
            say(f"--- {title}: {case_store.clock(top[0])} - {case_store.clock(top[1])} ({view['duration_s']} s)")
            say(view["transcript"])
        answer = ask("Which makes the better Short? [1 / 2 / t(ie) / s(kip) / q(uit)] ").strip().lower()
        if answer in ("q", "quit"):
            break
        if answer in ("t", "tie"):
            answer = "tie"
        if answer in ANSWERS:
            save_human(pair["id"], answer, root=root)
            labeled += 1
    return labeled
