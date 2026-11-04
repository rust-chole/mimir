"""Weak labels: what a person decided about shorts MIMIR already published (Accept / Reject).

``harvest-reviews`` reads every ``<name>_short.review.md`` (the Decision checkboxes) with its
``.qc.json`` in ``vod_output/final`` and the QC rejects in ``vod_output/rejected``, finds the case of
the source video, and recovers the published clip's range on the source clock from the pipeline
state (``vod_output/pipeline/<video>_pipeline.json`` -> selected clip index) and the Stage 2 analysis.

These are WEAK labels: one decision about one clip, made after seeing MIMIR's pick. They are stored
apart from the golden labels (``<root>/weak_labels/<case_id>.json``), reported in their own lines, and
never shown by the labeling editor or the sheet (they would reveal MIMIR's pick and anchor the
annotator). The range is marked uncertain when the analysis file changed after the short was published.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai.clip_eval import cases as case_store
from ai.clip_eval import metrics

WEAK_DIR = "weak_labels"
_CHECKED = re.compile(r"^\s*[-*]\s*\[[xX]\]\s*(Accept|Reject)\b(.*)$")


def parse_decision(text: str) -> tuple[str, str]:
    """accept | reject | undecided | conflict, plus the reject reason, from a review sheet."""
    section = str(text).split("## Decision", 1)
    if len(section) < 2:
        return "undecided", ""
    found: dict[str, str] = {}
    for line in section[1].splitlines():
        if line.startswith("## "):
            break
        match = _CHECKED.match(line)
        if match:
            reason = match.group(2).strip()
            reason = re.sub(r"^\.?\s*Reason:\s*", "", reason, flags=re.IGNORECASE).strip()
            found[match.group(1).lower()] = reason
    if len(found) > 1:
        return "conflict", ""
    if not found:
        return "undecided", ""
    decision = next(iter(found))
    return decision, found[decision] if decision == "reject" else ""


def _safe_name(value: str) -> str:            # mirrors shorts_pipeline._safe_name (pipeline state file names)
    text = str(value).strip()
    for char in '<>:"/\\|?*':
        text = text.replace(char, "_")
    return " ".join(text.split()).strip(" ._") or "video"


def picked_range(source: Path, published: Path | None, *, vod_output: Path) -> dict[str, Any] | None:
    """The published clip's [start, end] on the source clock, from the pipeline state + analysis."""
    state = case_store._read_json(vod_output / "pipeline" / f"{_safe_name(source.stem)}_pipeline.json")
    if not isinstance(state, Mapping):
        return None
    selected = state.get("selected_clip") if isinstance(state.get("selected_clip"), Mapping) else {}
    try:
        index = int(selected.get("clip_index"))
    except (TypeError, ValueError):
        return None
    analysis_path = vod_output / "analysis" / f"{source.stem}_clips.json"
    analysis = case_store._read_json(analysis_path)
    clips = analysis.get("clips") if isinstance(analysis, Mapping) else None
    if not isinstance(clips, list) or not (1 <= index <= len(clips)) or not isinstance(clips[index - 1], Mapping):
        return None
    clip = clips[index - 1]
    start, end = metrics._num(clip.get("start")), metrics._num(clip.get("end"))
    if start is None or end is None or end <= start:
        return None
    uncertain = bool(published and published.is_file() and analysis_path.stat().st_mtime > published.stat().st_mtime)
    return {"start": round(start, 3), "end": round(end, 3), "clip_index": index,
            "title": str(clip.get("title", ""))[:120], "uncertain": uncertain}


def harvest(*, root: str | Path | None = None, final_dir: str | Path | None = None,
            rejected_dir: str | Path | None = None, vod_output: str | Path | None = None,
            log: Any = print) -> list[dict[str, Any]]:
    """Collect every review decision into the weak labels of the matching cases."""
    base = Path(vod_output) if vod_output else case_store.PROJECT_ROOT / "vod_output"
    final_dir = Path(final_dir) if final_dir else base / "final"
    rejected_dir = Path(rejected_dir) if rejected_dir else base / "rejected"
    cases = case_store.list_cases(root)
    found: list[dict[str, Any]] = []
    for qc_path in sorted(final_dir.glob("*_short.qc.json")) + sorted(rejected_dir.glob("*_short_REJECTED.qc.json")):
        qc = case_store._read_json(qc_path)
        if not isinstance(qc, Mapping) or not qc.get("source"):
            continue
        source = Path(str(qc["source"]))
        rejected_by_qc = qc_path.parent == rejected_dir or qc.get("status") == "rejected"
        published = (Path(str(qc["published"])) if qc.get("published")
                     else qc_path.with_name(qc_path.name[:-len(".qc.json")] + ".mp4"))
        review = qc_path.with_name(qc_path.name.replace(".qc.json", ".review.md"))
        if rejected_by_qc:
            decision, reason = "qc_rejected", ", ".join(qc.get("failed") or [])
        elif review.is_file():
            decision, reason = parse_decision(review.read_text(encoding="utf-8-sig"))
        else:
            decision, reason = "undecided", ""
        size = source.stat().st_size if source.is_file() else None
        case = case_store.find_case_for_video(source, cases, size=size)
        row = {"case_id": case.case_id if case else None, "source": str(source), "qc_file": str(qc_path),
               "review_file": str(review) if review.is_file() else None, "decision": decision, "reason": reason,
               "qc_status": qc.get("status"), "at": qc.get("at"), "harvested_at": case_store._now(),
               "range": picked_range(source, published, vod_output=base)}
        found.append(row)
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in found:
        if row["case_id"]:
            by_case.setdefault(row["case_id"], []).append(row)
    for case_id, rows in by_case.items():
        case_store._write_json(case_store.eval_root(root) / WEAK_DIR / f"{case_id}.json",
                               {"case_id": case_id, "kind": "weak labels from human review sheets", "rows": rows})
    unmatched = [r for r in found if not r["case_id"]]
    if unmatched:
        log(f"{len(unmatched)} review(s) belong to videos with no case (prepare them first): "
            + ", ".join(sorted({Path(r['source']).name for r in unmatched})))
    return found


def load_weak(case_id: str, root: str | Path | None = None) -> list[dict[str, Any]]:
    data = case_store._read_json(case_store.eval_root(root) / WEAK_DIR / f"{case_id}.json")
    rows = data.get("rows") if isinstance(data, Mapping) else None
    return [r for r in rows or [] if isinstance(r, Mapping)]


def count(root: str | Path | None = None) -> int:
    folder = case_store.eval_root(root) / WEAK_DIR
    return sum(len(load_weak(p.stem, root)) for p in folder.glob("*.json")) if folder.is_dir() else 0


def check_against(case: case_store.Case, tops: Sequence[Sequence[float] | None], *,
                  root: str | Path | None = None) -> dict[str, Any] | None:
    """Did a run publish the same clip a person accepted / rejected before? (tIoU >= 0.5, any repeat)"""
    rows = [r for r in load_weak(case.case_id, root) if r.get("range") and r.get("decision") in ("accept", "reject")]
    if not rows:
        return None
    spans = [(float(t[0]), float(t[1])) for t in tops if t]
    out = {"accepted": 0, "accepted_repicked": 0, "rejected": 0, "rejected_repicked": 0, "uncertain_ranges": 0}
    for row in rows:
        rng = (float(row["range"]["start"]), float(row["range"]["end"]))
        hit = any(metrics.tiou(span, rng) >= metrics.MATCH_TIOU for span in spans)
        key = "accepted" if row["decision"] == "accept" else "rejected"
        out[key] += 1
        out[f"{key}_repicked"] += int(hit)
        out["uncertain_ranges"] += int(bool(row["range"].get("uncertain")))
    return out


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = ("accepted", "accepted_repicked", "rejected", "rejected_repicked", "uncertain_ranges")
    return {"cases": len(rows), **{k: sum(int(r.get(k, 0)) for r in rows) for k in keys}}
