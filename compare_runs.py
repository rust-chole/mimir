"""Side-by-side comparison of two MIMIR runs of the same VOD (real-VOD A/B validation).

    python compare_runs.py A B [--out report.md]

A and B are each either an artifact index (``<name>_short.artifacts.json`` /
``<name>_short_REJECTED.artifacts.json``, written by every run of this branch) or a
pipeline state file (``vod_output/state/<stem>_pipeline.json``, written by every
branch, including the CCR baseline). The report lists the facts a human needs
to judge the two shorts; it never decides which one is better.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).is_file():
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _duration(path: str | None) -> str:
    if not path or not Path(path).is_file():
        return "n/a"
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                              "default=nw=1:nk=1", path], capture_output=True, text=True, check=True).stdout
        return f"{float(out.strip()):.2f}s"
    except Exception:
        return "n/a"


def from_state(state_path: Path) -> dict[str, Any]:
    """Rebuild the comparable facts from a pipeline state file (any branch)."""
    state = _load(state_path)
    stages = state.get("stages", {}) or {}

    def stage_path(name: str) -> str | None:
        return (stages.get(name) or {}).get("path")

    truth = _load(stage_path("caption_truth_v6"))
    base = stage_path("intro_final_base")
    timeline = _load(Path(base).with_name(Path(base).stem + ".timeline.json")) if base else {}
    lexical = truth.get("lexical_decisions") or []
    return {
        "status": state.get("publish_status") or state.get("run_status"),
        "final_video": state.get("final_output"),
        "story": {"selected_clip": state.get("selected_clip"), "integrity": state.get("story_integrity"),
                  "protected_paced": (timeline.get("restart") or {}).get("protected_paced")},
        "captions": {"truth": stage_path("caption_truth_v6"), "judge": truth.get("lexical_judge"),
                     "clock": truth.get("clock_health"), "uncertain": truth.get("uncertain"),
                     "judge_changed_spans": sum(1 for r in lexical if r.get("changed")
                                                and r.get("decided_by") == "caption_judge"),
                     "judge_answers_set_aside": sum(1 for r in lexical if r.get("guard")),
                     "entity_decisions": truth.get("entity_decisions")},
        "speakers": {"roster": truth.get("roster")},
        "intro": {"cold_open": timeline.get("intro")},
        "edit": {"pro_edit": state.get("pro_edit")},
        "qc": {"gate": (state.get("v6") or {}).get("gate"), "repairs": (state.get("final_qc") or {}).get("repairs")},
        "human_review": state.get("human_review"),
    }


def facts(path: Path) -> dict[str, Any]:
    data = _load(path)
    if "stages" in data and "pipeline_version" in data:
        return from_state(path)
    return data


def rows(run: dict[str, Any]) -> dict[str, str]:
    story = run.get("story") or {}
    clip = story.get("selected_clip") or {}
    integrity = story.get("integrity") or {}
    captions = run.get("captions") or {}
    clock = captions.get("clock") or {}
    judge = captions.get("judge") or {}
    intro = (run.get("intro") or {}).get("cold_open") or {}
    edit = run.get("edit") or {}
    pro = edit.get("pro_edit") or {}
    gate = (run.get("qc") or {}).get("gate") or {}
    roster = (run.get("speakers") or {}).get("roster") or []
    return {
        "status": str(run.get("status")),
        "final duration": _duration(run.get("final_video")),
        "selected clip": f"#{clip.get('clip_index')} {clip.get('title', '')!s}"[:80],
        "clip score / duration": f"{clip.get('score')} / {clip.get('duration_seconds')}s",
        "story integrity": f"{integrity.get('status', 'n/a')}: {integrity.get('reason', '')}"[:160],
        "story bounds (source s)": f"{integrity.get('original')} -> {integrity.get('final')}",
        "protected ranges (paced)": str(len(story.get("protected_paced") or [])),
        "caption judge": f"{judge.get('status', 'n/a')} ({judge.get('model', '')})",
        "judge-changed spans": str(captions.get("judge_changed_spans", "n/a")),
        "judge answers set aside": str(captions.get("judge_answers_set_aside", "n/a")),
        "uncertain caption spans": str(len(captions.get("uncertain") or [])),
        "judge name decisions": str(len(captions.get("entity_decisions") or [])),
        "clock": f"{clock.get('status', 'n/a')} ({clock.get('source', '')})",
        "speakers": ", ".join(str(r.get("name", r)) if isinstance(r, dict) else str(r) for r in roster)[:80] or "anonymous",
        "cold open": f"{intro.get('duration', 'n/a')}s, headline {intro.get('headline', '')!r}"[:100],
        "edit": f"planner {pro.get('planner', 'n/a')}, camera ops {(pro.get('direction') or {}).get('camera_ops', 'n/a')}",
        "gate": f"{gate.get('status', 'n/a')}; failed {gate.get('failed', [])}; warnings {gate.get('warnings', [])}"[:200],
        "repairs": str((run.get("qc") or {}).get("repairs")),
        "human review": str(run.get("human_review")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    left, right = rows(facts(args.a)), rows(facts(args.b))
    lines = [f"# MIMIR run comparison", "", f"- A: `{args.a}`", f"- B: `{args.b}`", "",
             "| fact | A | B |", "| --- | --- | --- |"]
    for key in left:
        a, b = left[key].replace("|", "/"), right.get(key, "").replace("|", "/")
        lines.append(f"| {key}{' **(differs)**' if a != b else ''} | {a} | {b} |")
    lines += ["", "The comparison lists facts only; the human decides which short is better."]
    report = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(report, encoding="utf-8")
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
