"""Stage ``publish``: copy the QC-approved Short to the output folder with its manifest."""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from mimir.config import Settings
from mimir.core.jsonio import write_json
from mimir.core.stage import StageContext, StageOutput
from mimir.errors import QualityGateError


def slug(text: str, limit: int = 60) -> str:
    value = re.sub(r"[^\w\-]+", "_", str(text).strip(), flags=re.UNICODE).strip("_").lower()
    return (value or "short")[:limit]


class PublishStage:
    name = "publish"
    version = 1
    deps = ("source", "story", "cold_open", "timeline", "captions", "render", "qc")

    def params(self, settings: Settings) -> Any:
        return {"output_dir": str(settings.output_dir)}

    def run(self, ctx: StageContext) -> StageOutput:
        qc = ctx.dep("qc").json("qc")
        if not qc["passed"]:
            raise QualityGateError(f"quality gate failed: {qc['failed']}")
        story = ctx.dep("story").json("story")
        cold = ctx.dep("cold_open").json("cold_open")
        timeline = ctx.dep("timeline").json("timeline")
        render = ctx.dep("render").json("render")
        folder = Path(ctx.settings.output_dir) / slug(ctx.source.path.stem)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{slug(story['title'])}.mp4"
        shutil.copy2(ctx.dep("render").path("video"), target)
        manifest: dict[str, Any] = {
            "video": str(target), "source": str(ctx.source.path), "job_id": ctx.source.job_id,
            "title": story["title"], "hook_text": cold.get("hook", {}).get("text", ""),
            "story": {"source_range": [story["start"], story["end"]], "beats": story["beats"],
                      "primary_moment": story["primary_moment"]},
            "cold_open": {"source_window": cold["window"], "peak": cold["peak"]},
            "timeline": {"fps": timeline["fps"], "frames": timeline["frame_count"], "duration": timeline["duration"],
                         "segments": timeline["segments"]},
            "render": render, "qc": {"passed": qc["passed"], "checks": [(c["name"], c["passed"]) for c in qc["checks"]],
                                     "warnings": [c["name"] for c in qc["checks"]
                                                  if not c["passed"] and c["severity"] == "warn"],
                                     "repair_round": qc["repair_round"]},
            "speakers": next((c["details"] for c in qc["checks"] if c["name"] == "speaker_resolution"), {}),
        }
        manifest_path = write_json(folder / f"{slug(story['title'])}.json", manifest)
        return StageOutput(data={"published": {"video": str(target), "manifest": str(manifest_path)}})
