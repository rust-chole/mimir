"""Stage ``render_base``: the canonical timeline rendered at source geometry, frame exact.

Every segment is trimmed from the SOURCE with exactly ``frames`` video frames
(``fps`` resampling + clone padding + frame trim) and exactly
``frames * rate / fps`` audio samples, with 8 ms fades at each join so jump
cuts never click. Audio stays PCM (no encoder priming offsets) until the
final encode, which guarantees audio/video sync by construction.
"""
from __future__ import annotations

from typing import Any

from mimir.config import Settings
from mimir.core.stage import StageContext, StageOutput
from mimir.errors import StageError
from mimir.media.ffmpeg import ffmpeg, filter_complex_script
from mimir.media.probe import probe
from mimir.timeline.schema import Timeline

JOIN_FADE = 0.008


def filter_graph(timeline: Timeline, rate: int) -> str:
    parts = []
    labels = []
    fps = timeline.fps
    for segment in timeline.segments:
        n = segment.frames
        samples = n * rate // fps
        s, e = segment.source_start, segment.source_end
        duration = n / fps
        parts.append(f"[0:v]trim=start={s:.6f}:end={e + 0.25:.6f},setpts=PTS-STARTPTS,fps={fps}:round=near,"
                     f"tpad=stop_mode=clone:stop=4,trim=end_frame={n},setpts=PTS-STARTPTS[v{segment.index}]")
        parts.append(f"[0:a]atrim=start={s:.6f}:end={e + 0.05:.6f},asetpts=PTS-STARTPTS,aresample={rate},"
                     f"aformat=sample_fmts=fltp:channel_layouts=stereo,apad,atrim=end_sample={samples},"
                     f"afade=t=in:st=0:d={JOIN_FADE},afade=t=out:st={max(0.0, duration - JOIN_FADE):.6f}:d={JOIN_FADE},"
                     f"asetpts=PTS-STARTPTS[a{segment.index}]")
        labels.append(f"[v{segment.index}][a{segment.index}]")
    parts.append("".join(labels) + f"concat=n={len(timeline.segments)}:v=1:a=1[vout][aout]")
    return ";".join(parts)


class BaseRenderStage:
    name = "render_base"
    version = 1
    deps = ("source", "timeline")

    def params(self, settings: Settings) -> Any:
        return {"audio_rate": settings.output.audio_rate}

    def run(self, ctx: StageContext) -> StageOutput:
        timeline = Timeline.from_dict(ctx.dep("timeline").json("timeline"))
        rate = ctx.settings.output.audio_rate
        output = ctx.out_dir / "base.mkv"
        script = ctx.out_dir / "base_filter.txt"
        script.write_text(filter_graph(timeline, rate), encoding="utf-8")
        ffmpeg(["-v", "error", "-i", str(ctx.source.path), *filter_complex_script(script),
                "-map", "[vout]", "-map", "[aout]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "12",
                "-pix_fmt", "yuv420p", "-g", str(timeline.fps), "-c:a", "pcm_s16le", "-ar", str(rate),
                str(output)], timeout=7200)
        info = probe(output, count_frames=True)
        if info.frame_count != timeline.frame_count:
            raise StageError(self.name, f"base render has {info.frame_count} frames, timeline {timeline.frame_count}")
        if not info.has_audio:
            raise StageError(self.name, "base render lost the audio stream")
        return StageOutput(data={"base": {"frames": info.frame_count, "fps": timeline.fps, "width": info.width,
                                          "height": info.height, "duration": round(timeline.duration, 6)}},
                           files={"video": output})
