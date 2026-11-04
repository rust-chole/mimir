"""Stand-ins for Stage 2 in the clip-eval tests (loaded via MIMIR_CLIP_EVAL_TEST_ANALYZER in a worker process).

The fake writes its analysis with the REAL ``clip_analyzer.save_analysis`` so the tests prove that the
worker redirected Stage 2's output away from vod_output/analysis. It never calls a model unless
MIMIR_CLIP_EVAL_FAKE_MODE=paid, which attempts one call inside ``except Exception`` (exactly how Stage 2
wraps its optional passes) to prove the budget meter refuses it and the refusal cannot be swallowed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# case id -> published clip, then other candidates (source seconds); overridable per process
DEFAULT_PICKS = {"dev_one": [[100.0, 130.0], [300.0, 330.0], [500.0, 520.0]],
                 "dev_two": [[40.0, 70.0], [200.0, 228.0]],
                 "hold_one": [[600.0, 640.0], [10.0, 40.0]]}


def analysis_for(case_id: str, picks: dict[str, list[list[float]]], *, version: int, model: str) -> dict[str, Any]:
    clips = []
    for rank, (start, end) in enumerate(picks.get(case_id) or [], start=1):
        clips.append({"start": start, "end": end, "duration": round(end - start, 3), "rank": rank, "score": 9.0 - rank,
                      "terra_selected": rank == 1, "title": f"{case_id} clip {rank} by {model}"})
    return {"version": version, "model": model, "clips": clips}


def fake_create_clip_analysis(transcript_path: Path, video_report_path: Path | None = None) -> dict[str, Any]:
    from ai.editor import clip_analyzer as ca

    mode = os.getenv("MIMIR_CLIP_EVAL_FAKE_MODE", "good")
    if mode == "paid":
        try:                                   # Stage 2 style: an optional pass that must not hide a refusal
            ca.client.responses.create(model=ca.CLIP_SCOUT_MODEL, input="x", reasoning={"effort": "low"})
        except Exception:
            pass
    picks = json.loads(os.getenv("MIMIR_CLIP_EVAL_FAKE_PICKS", "") or "null") or DEFAULT_PICKS
    analysis = analysis_for(Path(transcript_path).stem, picks, version=ca.ANALYZER_VERSION, model=ca.CLIP_JUDGE_MODEL)
    analysis["seen"] = {"scout": ca.CLIP_SCOUT_MODEL, "judge_effort": ca.CLIP_JUDGE_REASONING_EFFORT,
                        "visual_report": str(video_report_path) if video_report_path else None}
    ca.save_analysis(transcript_path, analysis, video_report_path)
    return analysis
