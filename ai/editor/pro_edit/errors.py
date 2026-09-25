"""Explicit error taxonomy for the Pro Edit presentation layer.

Every failure inside Pro Edit is typed so the pipeline can degrade to the
existing MIMIR output deliberately instead of catching anonymous errors.
"""
from __future__ import annotations

from typing import Any


class ProEditError(RuntimeError):
    """Base class. Any ProEditError must leave existing MIMIR output intact."""


class EditContextError(ProEditError):
    """Required MIMIR evidence for building an EditContext is missing/invalid."""


class EditPlanValidationError(ProEditError):
    """A plan (or raw planner payload) failed validation fatally."""

    def __init__(self, message: str, issues: list[Any] | None = None) -> None:
        super().__init__(message)
        self.issues = list(issues or [])


class EditPlanTimelineError(ProEditError):
    """A timestamp was used in (or converted to) the wrong timeline domain."""


class PresetResolutionError(ProEditError):
    """A semantic preset could not be resolved into bounded render geometry."""


class SubjectResolutionError(ProEditError):
    """A camera target could not be resolved from subject evidence."""


class PlannerUnavailableError(ProEditError):
    """The planner backend could not be reached or is not configured."""


class PlannerOutputError(ProEditError):
    """The planner answered, but not with parseable machine-readable JSON."""


class EditRenderError(ProEditError):
    """FFmpeg/ffprobe execution or post-render validation failed."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        stderr: str = "",
        command: list[str] | None = None,
    ) -> None:
        tail = (stderr or "").strip()[-2000:]
        detail = message if not tail else f"{message}\n--- ffmpeg stderr (tail) ---\n{tail}"
        super().__init__(detail)
        self.returncode = returncode
        self.stderr = stderr or ""
        self.command = list(command or [])
