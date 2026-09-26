"""Audio layer: story audio + restrained SFX (ducked) -> two-pass loudness normalization.

The output has exactly ``frames * rate / fps`` samples, so the final mux can
never drift from the picture.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from mimir.config import OutputSettings, EffectsSettings
from mimir.errors import MediaError
from mimir.media.ffmpeg import ffmpeg, filter_complex_script

SFX_FADE_IN, SFX_FADE_OUT = 0.04, 0.18


def mix_graph(sfx: Sequence[dict[str, Any]], total_samples: int, rate: int, duck_threshold: float) -> str:
    parts = [f"[0:a]aresample={rate},aformat=sample_fmts=fltp:channel_layouts=stereo,"
             f"apad,atrim=end_sample={total_samples}[base]"]
    duck_keys, plain = [], []
    for index, cue in enumerate(sfx, start=1):
        duration = float(cue["duration"])
        delay = int(round(float(cue["start"]) * 1000))
        label = f"s{index}"
        parts.append(f"[{index}:a]atrim=start=0:end={duration:.6f},asetpts=PTS-STARTPTS,aresample={rate},"
                     f"aformat=sample_fmts=fltp:channel_layouts=stereo,highpass=f=35,volume={float(cue['volume']):.4f},"
                     f"afade=t=in:st=0:d={SFX_FADE_IN},afade=t=out:st={max(0.0, duration - SFX_FADE_OUT):.6f}:"
                     f"d={SFX_FADE_OUT},adelay={delay}|{delay},apad,atrim=end_sample={total_samples}[{label}]")
        (duck_keys if cue.get("duck") else plain).append(label)
    base = "base"
    if duck_keys:
        if len(duck_keys) == 1:
            parts.append(f"[{duck_keys[0]}]asplit=2[key][duckmix]")
        else:
            parts.append("".join(f"[{k}]" for k in duck_keys) + f"amix=inputs={len(duck_keys)}:normalize=0,"
                         "asplit=2[key][duckmix]")
        parts.append(f"[base][key]sidechaincompress=threshold={duck_threshold}:ratio=4:attack=12:release=135[ducked]")
        base = "ducked"
        plain.append("duckmix")
    if plain:
        inputs = f"[{base}]" + "".join(f"[{p}]" for p in plain)
        parts.append(f"{inputs}amix=inputs={1 + len(plain)}:duration=first:dropout_transition=0:normalize=0[mix]")
    else:
        parts.append(f"[{base}]anull[mix]")
    return ";".join(parts)


def _loudnorm_stats(stderr: str) -> dict[str, str]:
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr, flags=re.S)
    if not match:
        raise MediaError("loudnorm measurement returned no statistics")
    return json.loads(match.group(0))


def render_audio(base: Path, sfx: Sequence[dict[str, Any]], output: Path, *, frames: int, fps: int,
                 out: OutputSettings, effects: EffectsSettings, work: Path) -> Path:
    rate = out.audio_rate
    total = frames * rate // fps
    inputs = ["-i", str(base)]
    for cue in sfx:
        inputs += ["-i", str(cue["path"])]
    graph = mix_graph(sfx, total, rate, effects.duck_threshold)
    measure_script = work / "audio_measure.txt"
    measure_script.write_text(graph + f";[mix]loudnorm=I={out.loudness_lufs}:TP={out.true_peak_db}:LRA=11:"
                                      "print_format=json[m]", encoding="utf-8")
    result = ffmpeg(["-v", "info", *inputs, *filter_complex_script(measure_script), "-map", "[m]", "-f", "null", "-"],
                    timeout=1800)
    stats = _loudnorm_stats(result.stderr.decode("utf-8", "replace"))
    measured = float(stats["input_i"])
    if measured < -60.0:  # (near) silent story audio: normalization would only amplify noise
        norm = "anull"
    else:
        norm = (f"loudnorm=I={out.loudness_lufs}:TP={out.true_peak_db}:LRA=11:measured_I={stats['input_i']}:"
                f"measured_TP={stats['input_tp']}:measured_LRA={stats['input_lra']}:"
                f"measured_thresh={stats['input_thresh']}:offset={stats['target_offset']}:linear=true")
    apply_script = work / "audio_apply.txt"
    apply_script.write_text(graph + f";[mix]{norm},aresample={rate},alimiter=limit=0.95:level=disabled,"
                                    f"apad,atrim=end_sample={total}[final]", encoding="utf-8")
    ffmpeg(["-v", "error", *inputs, *filter_complex_script(apply_script), "-map", "[final]", "-ac", "2",
            "-ar", str(rate), "-c:a", "pcm_s16le", str(output)], timeout=1800)
    return output
