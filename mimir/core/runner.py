"""Pipeline runner: orchestration, signatures, caching. No business logic lives here."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from mimir.core import log
from mimir.core.artifacts import Artifact, ArtifactStore
from mimir.core.jsonio import content_hash, to_jsonable
from mimir.core.stage import Ledger, SourceInfo, Stage, StageContext
from mimir.errors import MimirError, StageError

if TYPE_CHECKING:
    from mimir.config import Settings
    from mimir.models.provider import ModelProvider


SOURCE = "source"


@dataclass
class RunReport:
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    executed: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def notes(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name, artifact in self.artifacts.items():
            for note in artifact.notes:
                rows.append({"stage": name, **note})
        return rows

    def degraded(self) -> list[dict[str, Any]]:
        return [note for note in self.notes() if note.get("level") == "degraded"]


def stage_signature(stage: Stage, params: Any, dep_ids: dict[str, str]) -> str:
    return content_hash({
        "stage": stage.name,
        "version": int(stage.version),
        "params": to_jsonable(params),
        "deps": dict(sorted(dep_ids.items())),
    })


class PipelineRunner:
    def __init__(
        self,
        stages: Sequence[Stage],
        *,
        settings: "Settings",
        provider: "ModelProvider",
        store: ArtifactStore,
        source: SourceInfo,
        force: bool = False,
        rerun: Iterable[str] = (),
    ) -> None:
        names = [stage.name for stage in stages]
        if len(set(names)) != len(names):
            raise ValueError("duplicate stage names")
        known = {SOURCE}
        for stage in stages:
            missing = [dep for dep in stage.deps if dep not in known]
            if missing:
                raise ValueError(f"stage {stage.name!r} depends on unknown/later stages {missing}")
            known.add(stage.name)
        self.stages = list(stages)
        self.settings = settings
        self.provider = provider
        self.store = store
        self.source = source
        self.force = bool(force)
        self.rerun = set(rerun)
        unknown = self.rerun - set(names)
        if unknown:
            raise ValueError(f"--rerun names unknown stages: {sorted(unknown)}")

    def run(self, until: str | None = None, report: RunReport | None = None) -> RunReport:
        report = report or RunReport()
        dep_ids: dict[str, str] = {SOURCE: self.source.identity}
        for artifact_name, artifact in report.artifacts.items():
            dep_ids[artifact_name] = artifact.content_id
        for stage in self.stages:
            params = stage.params(self.settings)
            signature = stage_signature(stage, params, {dep: dep_ids[dep] for dep in stage.deps})
            artifact = None
            if not self.force and stage.name not in self.rerun:
                artifact = self.store.lookup(stage.name, signature)
            if artifact is not None:
                log.info(f"{stage.name}: reused ({signature[:10]})")
                report.reused.append(stage.name)
            else:
                artifact = self._execute(stage, signature, params, report)
                report.executed.append(stage.name)
            report.artifacts[stage.name] = artifact
            report.timings[stage.name] = artifact.elapsed
            dep_ids[stage.name] = artifact.content_id
            if until is not None and stage.name == until:
                break
        return report

    def _execute(self, stage: Stage, signature: str, params: Any, report: RunReport) -> Artifact:
        log.info(f"{stage.name}: running ({signature[:10]})")
        out_dir = self.store.scratch_dir(stage.name, signature)
        ledger = Ledger()
        deps = {name: report.artifacts[name] for name in stage.deps if name != SOURCE}
        ctx = StageContext(stage.name, self.settings, self.provider, self.source, deps, out_dir, ledger)
        started = time.perf_counter()
        try:
            output = stage.run(ctx)
        except MimirError:
            raise
        except Exception as error:  # unexpected defects surface with the stage name
            raise StageError(stage.name, f"{type(error).__name__}: {error}") from error
        elapsed = time.perf_counter() - started
        for entry in ledger.entries:
            if entry["level"] != "info":
                log.warn(f"{stage.name}: {entry['code']}: {entry['message']}")
        artifact = self.store.commit(
            stage.name,
            signature,
            version=int(stage.version),
            data=output.data,
            files=output.files,
            inputs={name: artifact.content_id for name, artifact in deps.items()},
            params=to_jsonable(params),
            notes=ledger.entries,
            elapsed=elapsed,
        )
        log.info(f"{stage.name}: done in {elapsed:.1f}s")
        return artifact
