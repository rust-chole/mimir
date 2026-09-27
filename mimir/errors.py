"""Error types. A stage either produces its artifact or raises; there is no silent fallback."""
from __future__ import annotations


class MimirError(RuntimeError):
    """Base class for every expected MIMIR failure."""


class ConfigError(MimirError):
    """Invalid or missing configuration (API key, model route, settings file)."""


class MediaError(MimirError):
    """FFmpeg/ffprobe failure or unusable media."""


class ModelError(MimirError):
    """A model call failed or returned output that violates its contract."""


class StageError(MimirError):
    """A pipeline stage could not produce a valid artifact."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"[{stage}] {message}")
        self.stage = stage


class NoStoryError(StageError):
    """Story discovery found no money moment that can carry a complete Short."""


class QualityGateError(MimirError):
    """The rendered Short failed final quality control (after at most one repair)."""

    def __init__(self, message: str, report_path: str | None = None) -> None:
        super().__init__(message)
        self.report_path = report_path
