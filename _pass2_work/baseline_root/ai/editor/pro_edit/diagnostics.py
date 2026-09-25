"""Structured Pro Edit diagnostics and versioned artifacts (no per-frame spam)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from ai.editor.pro_edit.presets import ResolvedPlan
from ai.editor.pro_edit.schema import EditPlan
from ai.editor.pro_edit.validator import ValidationReport

ARTIFACT_VERSION = 1


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json_atomic(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)
    return path


@dataclass(frozen=True)
class ArtifactPaths:
    directory: Path
    clip_index: int

    def _name(self, kind: str, suffix: str = "json") -> Path:
        return self.directory / f"clip_{self.clip_index:02d}_pro_edit_{kind}_v{ARTIFACT_VERSION}.{suffix}"

    @property
    def context(self) -> Path:
        return self._name("context")

    @property
    def plan(self) -> Path:
        """Validated plan (+ planner metadata, validation report, cache key)."""
        return self._name("validated_plan")

    @property
    def resolved(self) -> Path:
        return self._name("resolved_plan")

    @property
    def request(self) -> Path:
        return self._name("request")

    @property
    def raw_response(self) -> Path:
        return self._name("raw_response")

    @property
    def subjects(self) -> Path:
        return self._name("subjects")

    @property
    def filter_script(self) -> Path:
        return self._name("camera", "txt")

    @property
    def intro_filter_script(self) -> Path:
        return self._name("intro_camera", "txt")

    @property
    def intro_resolved(self) -> Path:
        return self._name("intro_resolved_plan")

    @property
    def caption_ass(self) -> Path:
        """Presentation ASS (render input, not only a debug artifact)."""
        return self._name("captions", "ass")

    @property
    def caption_occupancy(self) -> Path:
        """Cached visual-activity map (motion evidence, not object detection)."""
        return self._name("caption_visual_occupancy")

    @property
    def caption_background(self) -> Path:
        """Cached caption background map (luminance/edges + TEXT_LIKE occupancy; no OCR)."""
        return self._name("caption_background")

    @property
    def energy(self) -> Path:
        """Editorial energy report (what was degraded and why)."""
        return self._name("editorial_energy")

    @property
    def caption_presentation(self) -> Path:
        """Presentation manifest: pages, word ids, lines, styles, safe regions."""
        return self._name("caption_presentation")


@dataclass
class Diagnostics:
    lines: list[str] = field(default_factory=list)

    def block(self, tag: str, **values: Any) -> None:
        self.lines.append(f"[{tag}]")
        for key, value in values.items():
            self.lines.append(f"{key}={value}")

    def emit(self) -> None:
        if self.lines:
            print("\n".join(self.lines))


def plan_block(diag: Diagnostics, *, enabled: bool, plan: EditPlan, style: str, style_version: int,
               duration: float, report: ValidationReport, planner: str, model_calls: int, plan_id: str) -> None:
    diag.block(
        "PRO_EDIT",
        enabled=int(enabled),
        schema=plan.schema_version,
        style=f"{style}@v{style_version}",
        planner=planner,
        model_calls=model_calls,
        duration=f"{duration:.2f}",
        events_received=report.events_received,
        events_valid=report.events_valid,
        events_sanitized=report.events_sanitized,
        events_rejected=report.events_rejected,
        plan_id=plan_id[:16],
    )


def event_blocks(diag: Diagnostics, resolved: ResolvedPlan, plan: EditPlan, *, debug: bool) -> None:
    by_id = {op.source_event_id: op for op in resolved.ops}
    for event in plan.camera_events:
        op = by_id.get(event.event_id)
        values: dict[str, Any] = {
            "id": event.event_id,
            "range": f"{event.start:.3f}-{event.end:.3f}",
            "role": event.role.value,
            "camera": event.camera.value,
            "motion": event.motion.value,
            "target": (op.params.target_subject or "center_safe") if op else "dropped",
            "intensity": f"{event.intensity:.2f}",
        }
        if op and debug:
            values["resolved"] = json.dumps(op.params.to_dict(), separators=(",", ":"))
        diag.block("PRO_EDIT_EVENT", **values)
    for event_id, reason in resolved.dropped:
        diag.block("PRO_EDIT_DROPPED", id=event_id, reason=reason)


def summarize_issues(report: ValidationReport, limit: int = 12) -> list[str]:
    return [f"{i.severity.value}:{i.event_id or 'plan'}:{i.code}" for i in report.issues[:limit]]


def write_artifacts(paths: ArtifactPaths, *, context: dict[str, Any] | None = None,
                    plan: dict[str, Any] | None = None, resolved: dict[str, Any] | None = None) -> list[Path]:
    written: list[Path] = []
    for path, data in ((paths.context, context), (paths.plan, plan), (paths.resolved, resolved)):
        if data is not None:
            written.append(write_json_atomic(path, data))
    return written


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def join_warnings(items: Iterable[str]) -> str:
    return " | ".join(str(i) for i in items if i)
