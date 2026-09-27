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
from mimir.qc.reviewer import moment_evidence, pick_moments, review, verdict_to_check
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
        repairable = bool(failed) and all(c["name"] in REPAIRABLE and c.get("repairable", True) for c in failed)
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
        moments = pick_moments(tl, inputs.story, inputs.cold_open, n)
        width, height = inputs.plan["output"]
        rendered = frames_at_indices(inputs.final_video, [m.frame for m in moments], fps=tl.fps,
                                     src_width=width, src_height=height)
        pairs = []
        for moment in moments:
            if moment.frame not in rendered:
                continue
            evidence = moment_evidence(moment, tl, inputs.story, inputs.context, inputs.plan, inputs.captions,
                                       inputs.truth)
            source = frame_at(inputs.source, tl.source_time(moment.frame), src_width=inputs.media.width,
                              src_height=inputs.media.height, width=640)
            pairs.append((evidence, source, rendered[moment.frame]))
        verdict = review(ctx.provider, ctx.settings.route("final_reviewer"), pairs, inputs.story, inputs.cold_open)
        check = verdict_to_check(verdict, inputs.context)
        check["details"]["frames"] = [e["frame_id"] + ":" + e["moment"] for e, _, _ in pairs]
        return check
