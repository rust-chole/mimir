"""Stage ``qc``: final quality gate over the rendered MP4 and every truth it must honor.

Produces a report and, when it fails, deterministic repair directives. The
pipeline driver applies at most ONE controlled repair round; there is no
open-ended AI re-render loop.
"""
from __future__ import annotations

from typing import Any

from mimir.config import Settings, routes_for, section
from mimir.core.stage import StageContext, StageOutput
from mimir.media.frames import frame_at, frames_at_indices
from mimir.media.probe import MediaInfo
from mimir.qc.checks import QCInputs, run_checks
from mimir.qc.reviewer import review
from mimir.timeline.schema import Timeline

ALL_PRIOR = ("source", "probe", "transcript", "vod_evidence", "story", "caption_verify", "speakers", "identity",
             "caption_truth", "vision", "cold_open", "timeline", "edit_context", "edit_direction", "edit_validation",
             "edit_compile", "captions", "effects", "render_base", "render")
REPAIRABLE = {"required_content_visible", "no_broken_faces", "camera_stability", "plan_reached_pixels",
              "captions_reached_pixels", "av_sync", "file_validity", "crop_validity", "final_review"}


class QCStage:
    name = "qc"
    version = 1
    deps = ALL_PRIOR

    def params(self, settings: Settings) -> Any:
        return {"qc": section(settings, "qc"), "repair_round": settings.repair.round,
                "routes": routes_for(settings, "final_reviewer") if settings.qc.reviewer else {}}

    def run(self, ctx: StageContext) -> StageOutput:
        ledger = [{"stage": name, **note} for name, artifact in ctx.deps.items() for note in artifact.notes]
        inputs = QCInputs(
            settings=ctx.settings, source=ctx.source.path,
            media=MediaInfo.from_dict(ctx.dep("probe").json("media")),
            story=ctx.dep("story").json("story"), cold_open=ctx.dep("cold_open").json("cold_open"),
            truth=ctx.dep("caption_truth").json("caption_truth"), captions=ctx.dep("captions").json("captions"),
            ass_path=ctx.dep("captions").path("ass"),
            timeline=Timeline.from_dict(ctx.dep("timeline").json("timeline")),
            pacing=ctx.dep("timeline").json("pacing"), vision=ctx.dep("vision").json("vision"),
            context=ctx.dep("edit_context").json("edit_context"), plan=ctx.dep("edit_compile").json("render_plan"),
            effects=ctx.dep("effects").json("effects"), base_video=ctx.dep("render_base").path("video"),
            final_video=ctx.dep("render").path("video"), ledger=ledger)
        checks = [c.to_dict() for c in run_checks(inputs)]
        if ctx.settings.qc.reviewer:
            checks.append(self._review(ctx, inputs))
        failed = [c for c in checks if not c["passed"] and c["severity"] == "fail"]
        repair: dict[str, Any] = {"widen_spans": [], "conservative_camera": False, "rerender": False}
        repairable = bool(failed) and all(c["name"] in REPAIRABLE for c in failed)
        for check in failed:
            directives = check.get("repair", {})
            repair["widen_spans"] = sorted(set(repair["widen_spans"]) | set(directives.get("widen_spans", [])))
            repair["conservative_camera"] |= bool(directives.get("conservative_camera"))
            repair["rerender"] |= bool(directives.get("rerender"))
        for check in failed:
            ctx.ledger.warning("qc_failed", f"{check['name']}: {check['details']}")
        return StageOutput(data={"qc": {"passed": not failed, "checks": checks,
                                        "failed": [c["name"] for c in failed],
                                        "repairable": repairable, "repair": repair if failed else None,
                                        "repair_round": ctx.settings.repair.round}})

    def _review(self, ctx: StageContext, inputs: QCInputs) -> dict[str, Any]:
        tl = inputs.timeline
        n = inputs.plan["frame_count"]
        picks = sorted({1, tl.main_start_frame + 5, n // 2, (3 * n) // 4, n - 3})
        width, height = inputs.plan["output"]
        rendered = frames_at_indices(inputs.final_video, picks, fps=tl.fps, src_width=width, src_height=height)
        pairs = [(i, frame_at(inputs.source, tl.source_time(i), src_width=inputs.media.width,
                              src_height=inputs.media.height, width=640), rendered[i]) for i in picks if i in rendered]
        verdict = review(ctx.provider, ctx.settings.route("final_reviewer"), pairs, inputs.story)
        spans = []
        for issue in verdict.get("issues", []):
            if issue["type"] in ("cropped_subject", "missing_action") and issue["severity"] == "high":
                try:
                    frame = int(issue["frame_id"].lstrip("f"))
                except ValueError:
                    continue
                spans += [s["id"] for s in inputs.context["spans"] if s["frames"][0] <= frame < s["frames"][1]]
        high = any(i["severity"] == "high" for i in verdict.get("issues", []))
        return {"name": "final_review", "passed": verdict.get("verdict") == "pass" or not high,
                "severity": "fail" if high else "warn", "details": verdict,
                "repair": {"widen_spans": sorted(set(spans)), "conservative_camera": bool(spans)}}
