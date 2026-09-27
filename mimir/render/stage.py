"""Stage ``render``: base timeline + render plan + captions + effects -> final MP4 (one encode)."""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Any

from mimir.config import Settings, section
from mimir.core.stage import StageContext, StageOutput
from mimir.errors import MediaError, StageError
from mimir.media.ffmpeg import escape_filter_path, ffmpeg_bin
from mimir.media.frames import iter_frames
from mimir.media.probe import probe
from mimir.render.audio import render_audio
from mimir.render.compositor import Compositor, apply_flash, flash_alpha, top_boxes


def subtitle_filter(ass_path: Path, font_file: str) -> str:
    value = f"ass=filename='{escape_filter_path(ass_path)}'"
    if font_file:
        value += f":fontsdir='{escape_filter_path(Path(font_file).parent)}'"
    return value


def encode(base_video: Path, audio: Path, ass_path: Path, output: Path, plan: dict[str, Any], effects: dict[str, Any],
           settings: Settings) -> None:
    out = settings.output
    width, height = plan["output"]
    fps = plan["fps"]
    src_w, src_h = plan["source"]
    compositor = Compositor(src_w, src_h, width, height, plan["geometry"]["stack_split"])
    tops = top_boxes(plan)
    flash = effects.get("flash", {})
    args = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
            "-i", str(audio), "-vf", subtitle_filter(ass_path, settings.captions.font_file),
            "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", out.preset, "-crf", str(out.crf),
            "-pix_fmt", "yuv420p", "-profile:v", "high", "-r", str(fps), "-g", str(fps * 2),
            "-c:a", "aac", "-b:a", out.audio_bitrate, "-ar", str(out.audio_rate), "-movflags", "+faststart",
            str(output)]
    process = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr_chunks: list[bytes] = []
    reader = threading.Thread(target=lambda: stderr_chunks.append(process.stderr.read()), daemon=True)
    reader.start()
    written = 0
    try:
        for index, frame in enumerate(iter_frames(base_video, src_width=src_w, src_height=src_h)):
            if index >= plan["frame_count"]:
                break
            image = compositor.frame(frame.image, plan["layout"][index], plan["windows"][index], tops.get(index))
            if flash:
                image = apply_flash(image, flash_alpha(index, flash["frame"], flash["frames"]))
            process.stdin.write(image.tobytes())
            written += 1
    except BrokenPipeError:
        pass
    finally:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        code = process.wait()
        reader.join(timeout=5)
    if code != 0:
        raise MediaError("final encode failed: " + b"".join(stderr_chunks).decode("utf-8", "replace")[-1500:])
    if written != plan["frame_count"]:
        raise MediaError(f"compositor wrote {written} frames, plan has {plan['frame_count']}")


class RenderStage:
    name = "render"
    version = 1
    deps = ("timeline", "render_base", "edit_compile", "captions", "effects")

    def params(self, settings: Settings) -> Any:
        return {"output": section(settings, "output"), "font_file": settings.captions.font_file,
                "duck": settings.effects.duck_threshold, "repair_round": settings.repair.round}

    def run(self, ctx: StageContext) -> StageOutput:
        base = ctx.dep("render_base")
        plan = ctx.dep("edit_compile").json("render_plan")
        effects = ctx.dep("effects").json("effects")
        ass_path = ctx.dep("captions").path("ass")
        frames, fps = plan["frame_count"], plan["fps"]
        audio = render_audio(base.path("video"), effects.get("sfx", []), ctx.out_dir / "final_audio.wav",
                             frames=frames, fps=fps, out=ctx.settings.output, effects=ctx.settings.effects,
                             work=ctx.out_dir)
        output = ctx.out_dir / "final.mp4"
        encode(base.path("video"), audio, ass_path, output, plan, effects, ctx.settings)
        info = probe(output, count_frames=True)
        if (info.width, info.height) != tuple(plan["output"]) or info.frame_count != frames:
            raise StageError(self.name, f"final render geometry/frames mismatch: {info.width}x{info.height} "
                                        f"{info.frame_count} frames (expected {plan['output']} {frames})")
        audio.unlink(missing_ok=True)
        return StageOutput(data={"render": {"frames": info.frame_count, "fps": fps, "duration": info.duration,
                                            "width": info.width, "height": info.height,
                                            "video_codec": info.video_codec, "audio_codec": info.audio_codec}},
                           files={"video": output})
