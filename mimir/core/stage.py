"""Stage contract.

A stage owns one piece of truth. It declares its dependencies (other stage
names or the special ``source``), the subset of settings that affect its
output (``params``) and an explicit ``version`` bumped whenever its behavior
changes. The runner derives the stage signature from exactly those three
things, so a caption-style change never re-runs transcription and a camera
change never re-runs whole-VOD analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from mimir.core.artifacts import Artifact

if TYPE_CHECKING:
    from mimir.config import Settings
    from mimir.models.provider import ModelProvider


@dataclass
class Ledger:
    """Recorded decisions of one stage run. ``degraded`` entries fail final QC."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def info(self, code: str, message: str, **data: Any) -> None:
        self.entries.append({"level": "info", "code": code, "message": message, **data})

    def warning(self, code: str, message: str, **data: Any) -> None:
        self.entries.append({"level": "warning", "code": code, "message": message, **data})

    def degraded(self, code: str, message: str, **data: Any) -> None:
        self.entries.append({"level": "degraded", "code": code, "message": message, **data})


@dataclass
class SourceInfo:
    path: Path
    identity: str          # sampled content hash of the source file
    job_id: str


@dataclass
class StageOutput:
    data: dict[str, Any] = field(default_factory=dict)
    files: dict[str, Path] = field(default_factory=dict)


@dataclass
class StageContext:
    stage: str
    settings: "Settings"
    provider: "ModelProvider"
    source: SourceInfo
    deps: dict[str, Artifact]
    out_dir: Path
    ledger: Ledger

    def dep(self, name: str) -> Artifact:
        try:
            return self.deps[name]
        except KeyError as error:
            raise KeyError(f"stage {self.stage!r} did not declare dependency {name!r}") from error


class Stage(Protocol):
    name: str
    version: int
    deps: tuple[str, ...]

    def params(self, settings: "Settings") -> Any:
        ...

    def run(self, ctx: StageContext) -> StageOutput:
        ...
