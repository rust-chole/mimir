"""The one production pipeline: stage graph + job driver (at most one QC repair round)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mimir.captions.stage import CaptionStage
from mimir.coldopen.planner import ColdOpenStage
from mimir.config import RepairDirectives, Settings
from mimir.core import log
from mimir.core.artifacts import ArtifactStore
from mimir.core.jsonio import sampled_file_hash, write_json
from mimir.core.runner import PipelineRunner, RunReport
from mimir.core.stage import SourceInfo
from mimir.edit.compiler import EditCompileStage
from mimir.edit.context import EditContextStage
from mimir.edit.director import EditDirectionStage
from mimir.edit.validator import EditValidationStage
from mimir.effects.stage import EffectsStage
from mimir.errors import MediaError, QualityGateError
from mimir.media.probe_stage import ProbeStage
from mimir.models.provider import ModelProvider
from mimir.publish import PublishStage
from mimir.qc.stage import QCStage
from mimir.render.base import BaseRenderStage
from mimir.render.stage import RenderStage
from mimir.speakers.identity import IdentityStage, Prompt
from mimir.speakers.stage import SpeakerStage
from mimir.story.discovery import StoryStage
from mimir.story.evidence import VodEvidenceStage
from mimir.timeline.builder import TimelineStage
from mimir.transcript.truth import CaptionTruthStage
from mimir.transcript.verify import CaptionVerifyStage
from mimir.transcript.whole import TranscriptStage
from mimir.vision.stage import VisionStage

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".ts"}


def build_stages(identity_prompt: Prompt | None = None) -> list:
    """Source Video -> ... -> Publish. The only production path."""
    return [
        ProbeStage(),               # media probe
        TranscriptStage(),          # transcript truth (whole VOD: lexical + timing)
        VodEvidenceStage(),         # cheap whole-VOD audio/motion peaks + bounded peak probe
        StoryStage(),               # story discovery -> StoryPackage
        CaptionVerifyStage(),       # caption-grade lexical/timing truth of the story window
        SpeakerStage(),             # speaker truth (story window)
        IdentityStage(identity_prompt),  # optional identity truth (+ speaker_preview audio)
        CaptionTruthStage(),        # frozen caption truth
        VisionStage(),              # selected-short visual analysis
        ColdOpenStage(),            # mandatory peak cold open
        TimelineStage(),            # canonical timeline (pacing, source->output mapping)
        EditContextStage(),         # edit context builder
        EditDirectionStage(),       # AI edit director (intent only)
        EditValidationStage(),      # edit plan validator
        EditCompileStage(),         # deterministic edit compiler
        CaptionStage(),             # captions (from frozen truth)
        EffectsStage(),             # effects / SFX
        BaseRenderStage(),          # base story timeline render
        RenderStage(),              # camera + captions + effects + audio -> MP4
        QCStage(),                  # final quality control
        PublishStage(),             # publish
    ]


@dataclass
class JobResult:
    published_video: Path
    manifest: Path
    qc: dict[str, Any]
    job_dir: Path
    report: RunReport


def job_id_for(path: Path) -> tuple[str, str]:
    identity = sampled_file_hash(path)
    stem = re.sub(r"[^\w\-]+", "_", path.stem, flags=re.UNICODE).strip("_")[:40] or "video"
    return f"{stem}_{identity[:10]}", identity


def resolve_source(video: str | Path) -> Path:
    path = Path(video).expanduser().resolve()
    if not path.is_file():
        raise MediaError(f"source video not found: {path}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise MediaError(f"unsupported source extension {path.suffix!r}")
    return path


def run_job(video: str | Path, settings: Settings, provider: ModelProvider, *, force: bool = False,
            rerun: Iterable[str] = (), identity_prompt: Prompt | None = None) -> JobResult:
    path = resolve_source(video)
    job_id, identity = job_id_for(path)
    source = SourceInfo(path, identity, job_id)
    store = ArtifactStore(Path(settings.workspace) / "jobs" / job_id)
    stages = build_stages(identity_prompt)
    log.info(f"MIMIR job {job_id} ({path.name}) provider={provider.name}")
    runner = PipelineRunner(stages, settings=settings, provider=provider, store=store, source=source, force=force,
                            rerun=rerun)
    report = runner.run(until="qc")
    qc = report.artifacts["qc"].json("qc")
    if not qc["passed"] and qc["repairable"] and settings.qc.repair_passes > 0:
        directives = qc["repair"]
        log.warn(f"QC failed {qc['failed']}; one controlled repair round: {directives}")
        settings = settings.with_(repair=RepairDirectives(
            round=1, widen_spans=tuple(directives["widen_spans"]),
            conservative_camera=bool(directives["conservative_camera"]), rerender=bool(directives["rerender"])))
        runner = PipelineRunner(stages, settings=settings, provider=provider, store=store, source=source)
        report = runner.run(until="qc", report=RunReport())
        qc = report.artifacts["qc"].json("qc")
    summary_path = write_json(store.job_dir / "run_summary.json", {
        "job_id": job_id, "source": str(path), "executed": report.executed, "reused": report.reused,
        "timings": report.timings, "notes": report.notes(), "qc_passed": qc["passed"], "qc_failed": qc["failed"],
        "repair_round": qc["repair_round"]})
    if not qc["passed"]:
        raise QualityGateError(f"final quality gate failed after {qc['repair_round']} repair round(s): {qc['failed']}",
                               str(report.artifacts["qc"].path("qc")))
    runner = PipelineRunner(stages, settings=settings, provider=provider, store=store, source=source)
    report = runner.run()
    published = report.artifacts["publish"].json("published")
    if not Path(published["video"]).is_file():  # the output folder was cleaned: publish again
        runner = PipelineRunner(stages, settings=settings, provider=provider, store=store, source=source,
                                rerun=("publish",))
        report = runner.run()
        published = report.artifacts["publish"].json("published")
    log.info(f"published {published['video']} (summary {summary_path})")
    return JobResult(Path(published["video"]), Path(published["manifest"]), qc, store.job_dir, report)
