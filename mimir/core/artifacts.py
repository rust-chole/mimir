"""Versioned, hash-verified stage artifacts.

Layout (one job per source video)::

    <workspace>/jobs/<job_id>/stages/<stage>/<signature[:16]>/
        record.json          signature, inputs, params, output hashes, ledger notes
        <outputs...>         JSON documents and media files owned by the stage

A stage result is reusable only when a record with the exact signature exists
and every output still matches the recorded size and content hash. Several
signatures per stage are kept (bounded), so toggling a caption style back and
forth never re-renders unrelated work.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from mimir.core.jsonio import content_hash, read_json, sha256_file, write_json

RECORD_NAME = "record.json"
KEEP_SIGNATURES_PER_STAGE = 4


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class OutputEntry:
    kind: str            # "json" | "file"
    relpath: str
    sha256: str
    size: int
    mtime_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "relpath": self.relpath, "sha256": self.sha256, "size": self.size,
                "mtime_ns": self.mtime_ns}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputEntry":
        return cls(str(data["kind"]), str(data["relpath"]), str(data["sha256"]), int(data["size"]),
                   int(data["mtime_ns"]))


@dataclass
class Artifact:
    """A committed stage result (read-only view)."""

    stage: str
    signature: str
    directory: Path
    outputs: dict[str, OutputEntry]
    notes: list[dict[str, Any]] = field(default_factory=list)
    reused: bool = False
    elapsed: float = 0.0
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def content_id(self) -> str:
        """Identity of this artifact's content, used by dependent stage signatures."""
        return content_hash({name: entry.sha256 for name, entry in sorted(self.outputs.items())})

    def path(self, name: str) -> Path:
        try:
            return self.directory / self.outputs[name].relpath
        except KeyError as error:
            raise KeyError(f"stage {self.stage!r} has no output {name!r}") from error

    def json(self, name: str) -> Any:
        if name not in self._cache:
            entry = self.outputs.get(name)
            if entry is None or entry.kind != "json":
                raise KeyError(f"stage {self.stage!r} has no JSON output {name!r}")
            self._cache[name] = read_json(self.directory / entry.relpath)
        return self._cache[name]

    def has(self, name: str) -> bool:
        return name in self.outputs


class ArtifactStore:
    def __init__(self, job_dir: str | Path) -> None:
        self.job_dir = Path(job_dir).resolve()
        self.stages_dir = self.job_dir / "stages"
        self.stages_dir.mkdir(parents=True, exist_ok=True)

    def stage_dir(self, stage: str, signature: str) -> Path:
        return self.stages_dir / stage / signature[:16]

    def scratch_dir(self, stage: str, signature: str) -> Path:
        """Fresh, empty directory a running stage writes its outputs into."""
        directory = self.stage_dir(stage, signature)
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        return directory

    def lookup(self, stage: str, signature: str) -> Artifact | None:
        directory = self.stage_dir(stage, signature)
        record_path = directory / RECORD_NAME
        if not record_path.is_file():
            return None
        try:
            record = read_json(record_path)
        except (OSError, ValueError):
            return None
        if record.get("signature") != signature or record.get("stage") != stage:
            return None
        outputs = {name: OutputEntry.from_dict(raw) for name, raw in dict(record.get("outputs", {})).items()}
        changed = False
        for entry in outputs.values():
            path = directory / entry.relpath
            if not path.is_file():
                return None
            stat = path.stat()
            if stat.st_size != entry.size:
                return None
            if stat.st_mtime_ns != entry.mtime_ns:
                if sha256_file(path) != entry.sha256:
                    return None
                entry.mtime_ns = stat.st_mtime_ns
                changed = True
        if changed:
            record["outputs"] = {name: entry.to_dict() for name, entry in outputs.items()}
            write_json(record_path, record)
        self._touch(directory)
        return Artifact(stage, signature, directory, outputs, list(record.get("notes", [])), reused=True)

    def commit(
        self,
        stage: str,
        signature: str,
        *,
        version: int,
        data: dict[str, Any],
        files: dict[str, Path],
        inputs: dict[str, str],
        params: Any,
        notes: list[dict[str, Any]],
        elapsed: float,
    ) -> Artifact:
        directory = self.stage_dir(stage, signature)
        directory.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, OutputEntry] = {}
        for name, document in data.items():
            relpath = f"{name}.json"
            path = write_json(directory / relpath, document)
            stat = path.stat()
            outputs[name] = OutputEntry("json", relpath, sha256_file(path), stat.st_size, stat.st_mtime_ns)
        for name, file_path in files.items():
            file_path = Path(file_path).resolve()
            if not file_path.is_file():
                raise FileNotFoundError(f"stage {stage!r} declared missing output {name!r}: {file_path}")
            try:
                relpath = str(file_path.relative_to(directory))
            except ValueError:
                relpath = file_path.name
                target = directory / relpath
                if target.resolve() != file_path:
                    shutil.copy2(file_path, target)
                file_path = target
            stat = file_path.stat()
            outputs[name] = OutputEntry("file", relpath, sha256_file(file_path), stat.st_size, stat.st_mtime_ns)
        write_json(directory / RECORD_NAME, {
            "stage": stage,
            "version": version,
            "signature": signature,
            "created": _now(),
            "elapsed_seconds": round(elapsed, 3),
            "inputs": inputs,
            "params": params,
            "outputs": {name: entry.to_dict() for name, entry in outputs.items()},
            "notes": notes,
        })
        self._prune(stage, keep=directory)
        return Artifact(stage, signature, directory, outputs, list(notes), reused=False, elapsed=elapsed)

    def _touch(self, directory: Path) -> None:
        try:
            os.utime(directory / RECORD_NAME)
        except OSError:
            pass

    def _prune(self, stage: str, keep: Path) -> None:
        stage_root = self.stages_dir / stage
        entries = [p for p in stage_root.iterdir() if p.is_dir() and p != keep]
        entries.sort(key=lambda p: (p / RECORD_NAME).stat().st_mtime if (p / RECORD_NAME).exists() else 0.0,
                     reverse=True)
        for stale in entries[KEEP_SIGNATURES_PER_STAGE - 1:]:
            shutil.rmtree(stale, ignore_errors=True)
