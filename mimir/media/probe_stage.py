"""Stage ``probe``: technical facts about the source."""
from __future__ import annotations

from typing import Any

from mimir.config import Settings
from mimir.core.stage import StageContext, StageOutput
from mimir.errors import StageError
from mimir.media.probe import output_fps_for, probe


class ProbeStage:
    name = "probe"
    version = 1
    deps = ("source",)

    def params(self, settings: Settings) -> Any:
        return {"output_fps": settings.output.fps}

    def run(self, ctx: StageContext) -> StageOutput:
        info = probe(ctx.source.path)
        if info.width < 64 or info.height < 64 or info.fps <= 0:
            raise StageError(self.name, f"unsupported video geometry/rate: {info.width}x{info.height}@{info.fps}")
        return StageOutput(data={"media": {**info.to_dict(),
                                           "output_fps": output_fps_for(info, ctx.settings.output.fps)}})
