from __future__ import annotations

import base64
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


TERRA_VISUAL_FALLBACK_VERSION = 3
MAX_FRAME_SAMPLES = 32
MAX_SCENE_SAMPLES = 10
MAX_AUDIO_GUIDED_SAMPLES = 12
AUDIO_GUIDED_PEAK_LIMIT = 4
AUDIO_BURST_PRE_SECONDS = 0.55
AUDIO_BURST_POST_SECONDS = 0.55
BATCH_SIZE = 7
FRAME_WIDTH = 640
JPEG_QUALITY = 7


EVENT_TYPES = [
    "reaction", "visual_payoff", "scene_change", "gameplay_event",
    "object_break", "destruction", "impact", "crash", "fall",
    "explosion", "physical_action", "entrance_exit", "reveal",
    "popup", "sudden_change", "other",
]
PEAK_SIGNALS = [
    "strong_visible_reaction", "sudden_motion", "impact", "reveal",
    "chaos", "multi_person_reaction", "expression_change", "state_change",
]


BATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "layout": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "orientation": {
                    "type": "string",
                    "enum": ["portrait", "landscape", "square", "unknown"],
                },
                "content_type": {
                    "type": "string",
                    "enum": ["streamer", "gameplay", "streamer_gameplay", "other"],
                },
                "notes": {"type": "string"},
            },
            "required": ["orientation", "content_type", "notes"],
        },
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "start": {"type": "number", "minimum": 0},
                    "end": {"type": "number", "minimum": 0},
                    "type": {"type": "string", "enum": EVENT_TYPES},
                    "description": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "observable_intensity": {
                        "type": "number", "minimum": 0, "maximum": 1
                    },
                    "signals": {
                        "type": "array",
                        "items": {"type": "string", "enum": PEAK_SIGNALS},
                    },
                },
                "required": [
                    "start", "end", "type", "description", "confidence",
                    "observable_intensity", "signals",
                ],
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["layout", "observations", "warnings"],
}


TERRA_VISUAL_INSTRUCTIONS = """
You are MIMIR's OpenAI visual-support observer. Gemini is intentionally bypassed.
Analyze sampled frames factually; do not make final editorial decisions.

For THIS pass only, perform factual visual observation. Do not decide whether a
moment is viral, funny, boring, clip-worthy, or the best intro. Later Terra
editor passes will make editorial decisions from this report.

You receive a SMALL CHRONOLOGICAL BATCH of timestamp-labelled still frames from
one source video. The timestamp immediately before each image belongs to that
image.

Report only visually supported facts. Prioritize:
- strong human reaction or synchronized multi-person reaction
- sudden motion, impact, fall, crash, destruction, object break, explosion
- gameplay death/elimination/score-changing action
- reveal, entrance/exit, popup, abrupt scene/state change
- visible payoff that changes what is happening

Rules:
1. This is sampled-frame analysis, not continuous native video. Never pretend
   you saw motion that the supplied frames do not support.
2. Use tight timestamp ranges around evidence; lower confidence when timing is
   uncertain.
3. A visible change between adjacent samples may be sudden_change/scene_change.
4. observable_intensity describes visible intensity only, NOT editorial value.
5. signals are factual visual signals only.
6. Never identify a real person.
7. Do not OCR or transcribe on-screen text.
8. Avoid duplicates inside the batch.
9. Closely spaced samples may intentionally bracket a strong source transient.
   Compare BEFORE -> PEAK -> AFTER state. If a door/object/person state visibly
   changes across those samples, report that change explicitly instead of treating
   the frames as unrelated stills.
""".strip()


def _run(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=timeout,
    )


def _probe_video(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("Terra visual fallback için ffprobe bulunamadı.")

    result = _run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height:format=duration",
            "-of", "json", str(path),
        ],
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError("ffprobe video bilgisi alınamadı: " + result.stderr[:1200])

    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as error:
        raise RuntimeError("ffprobe geçersiz JSON döndürdü.") from error

    try:
        duration = max(0.01, float(data.get("format", {}).get("duration", 0.0)))
    except (TypeError, ValueError):
        duration = 0.01

    streams = data.get("streams", [])
    stream = streams[0] if isinstance(streams, list) and streams else {}
    try:
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
    except (TypeError, ValueError):
        width = height = 0

    return {"duration": duration, "width": width, "height": height}


def _uniform_times(duration: float) -> list[float]:
    if duration <= 45:
        target = min(22, max(10, int(math.ceil(duration / 1.7))))
    elif duration <= 120:
        target = 24
    elif duration <= 300:
        target = 27
    else:
        target = 28

    if target <= 1:
        return [0.0]

    usable_end = max(0.0, duration - 0.08)
    return [round((usable_end * i) / (target - 1), 3) for i in range(target)]


def _scene_change_times(path: Path, duration: float) -> list[float]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or duration > 1800:
        return []

    command = [
        ffmpeg, "-hide_banner", "-loglevel", "info", "-i", str(path),
        "-an", "-sn", "-dn",
        "-vf", "scale=320:-2,select='gt(scene,0.34)',showinfo",
        "-fps_mode", "vfr", "-f", "null", "-",
    ]
    try:
        result = _run(command, timeout=max(45.0, min(220.0, duration * 0.45 + 30.0)))
    except Exception:
        return []

    raw = (result.stderr or "") + "\n" + (result.stdout or "")
    values: list[float] = []
    for match in re.finditer(r"pts_time:([0-9]+(?:\.[0-9]+)?)", raw):
        try:
            value = float(match.group(1))
        except (TypeError, ValueError):
            continue
        if 0 <= value <= duration:
            values.append(value)

    values = sorted(set(round(v, 3) for v in values))
    if len(values) <= MAX_SCENE_SAMPLES:
        return values

    result_times: list[float] = []
    for i in range(MAX_SCENE_SAMPLES):
        index = round(i * (len(values) - 1) / max(1, MAX_SCENE_SAMPLES - 1))
        value = values[index]
        if value not in result_times:
            result_times.append(value)
    return result_times


def _audio_peak_items(path: Path, duration: float) -> list[dict[str, Any]]:
    if duration > 300:
        return []
    try:
        from ai.editor.intro_peak_support import analyze_audio_peaks
        raw = analyze_audio_peaks(path, clip_duration=duration)
    except Exception:
        return []

    result: list[dict[str, Any]] = []
    for item in raw[:MAX_AUDIO_GUIDED_SAMPLES]:
        if not isinstance(item, dict):
            continue
        try:
            center = float(item.get("center", item.get("start", 0.0)))
            score = max(0.0, min(1.0, float(item.get("audio_score", 0.0))))
            dbfs = float(item.get("dbfs", -120.0))
            peak_dbfs = float(item.get("peak_dbfs", -120.0))
        except (TypeError, ValueError):
            continue
        if not 0 <= center <= duration:
            continue
        result.append({
            "center": round(center, 3),
            "start": round(max(0.0, float(item.get("start", center - 0.38))), 3),
            "end": round(min(duration, float(item.get("end", center + 0.48))), 3),
            "audio_score": round(score, 3),
            "dbfs": round(dbfs, 2),
            "peak_dbfs": round(peak_dbfs, 2),
        })
    result.sort(key=lambda x: (-float(x["audio_score"]), float(x["center"])))
    return result


def _audio_guided_times(
    path: Path,
    duration: float,
    audio_peaks: list[dict[str, Any]] | None = None,
) -> list[float]:
    peaks = audio_peaks if isinstance(audio_peaks, list) else _audio_peak_items(path, duration)
    result: list[float] = []
    for item in peaks[:AUDIO_GUIDED_PEAK_LIMIT]:
        try:
            center = float(item.get("center", item.get("start", 0.0)))
        except (TypeError, ValueError):
            continue
        # V26: a single frame at the exact audio peak is not enough to prove a
        # door/object state change. Deliberately bracket the transient so the
        # visual observer sees BEFORE -> PEAK -> AFTER evidence.
        for timestamp in (
            center - AUDIO_BURST_PRE_SECONDS,
            center,
            center + AUDIO_BURST_POST_SECONDS,
        ):
            timestamp = max(0.0, min(max(0.0, duration - 0.03), timestamp))
            if all(abs(timestamp - old) >= 0.18 for old in result):
                result.append(round(timestamp, 3))
    result.sort()
    return result[:MAX_AUDIO_GUIDED_SAMPLES]


def _merge_times(duration: float, *groups: list[float]) -> list[float]:
    candidates: list[float] = []
    for group in groups:
        for value in group:
            try:
                timestamp = max(0.0, min(duration - 0.03, float(value)))
            except (TypeError, ValueError):
                continue
            if all(abs(timestamp - old) >= 0.24 for old in candidates):
                candidates.append(round(timestamp, 3))

    candidates.sort()
    if len(candidates) <= MAX_FRAME_SAMPLES:
        return candidates

    indices = {
        round(i * (len(candidates) - 1) / max(1, MAX_FRAME_SAMPLES - 1))
        for i in range(MAX_FRAME_SAMPLES)
    }
    return [candidates[i] for i in sorted(indices)]


def _spread_pick(values: list[float], count: int) -> list[float]:
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return list(values)
    indices = {
        round(i * (len(values) - 1) / max(1, count - 1))
        for i in range(count)
    }
    return [values[i] for i in sorted(indices)]


def collect_sample_times(
    path: Path,
    duration: float,
    audio_peaks: list[dict[str, Any]] | None = None,
) -> list[float]:
    # V26: audio-burst bracket frames get reserved slots. Previously the final
    # 32-frame cap could uniformly thin them back out, recreating the exact
    # one-frame blind spot this guard is meant to solve.
    audio_times = _audio_guided_times(path, duration, audio_peaks)
    base = _merge_times(
        duration,
        _uniform_times(duration),
        _scene_change_times(path, duration),
    )
    base = [
        t for t in base
        if all(abs(float(t) - float(a)) >= 0.18 for a in audio_times)
    ]
    remaining = max(0, MAX_FRAME_SAMPLES - len(audio_times))
    chosen = sorted(audio_times + _spread_pick(base, remaining))
    return chosen[:MAX_FRAME_SAMPLES]


def _extract_frame(path: Path, timestamp: float, output: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Terra visual fallback için ffmpeg bulunamadı.")

    result = subprocess.run(
        [
            ffmpeg, "-v", "error", "-ss", f"{timestamp:.3f}", "-i", str(path),
            "-frames:v", "1", "-vf", f"scale='min({FRAME_WIDTH},iw)':-2",
            "-q:v", str(JPEG_QUALITY), "-y", str(output),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
        timeout=35,
    )
    return result.returncode == 0 and output.is_file() and output.stat().st_size > 500


def _data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def _response_text(response: Any) -> str:
    direct = str(getattr(response, "output_text", "") or "").strip()
    if direct:
        return direct
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = str(getattr(content, "text", "") or "").strip()
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _normalize_observation(item: Any, duration: float) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    try:
        start = max(0.0, min(duration, float(item.get("start", 0.0))))
        end = max(start, min(duration, float(item.get("end", start))))
    except (TypeError, ValueError):
        return None

    kind = str(item.get("type", "other")).strip().lower()
    if kind not in EVENT_TYPES:
        kind = "other"
    description = str(item.get("description", "")).strip()
    if not description:
        return None

    signals: list[str] = []
    raw_signals = item.get("signals", [])
    if isinstance(raw_signals, list):
        for value in raw_signals:
            signal = str(value).strip().lower()
            if signal in PEAK_SIGNALS and signal not in signals:
                signals.append(signal)

    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "type": kind,
        "description": description,
        "confidence": _score(item.get("confidence", 0.0)),
        "observable_intensity": _score(item.get("observable_intensity", 0.0)),
        "signals": signals,
    }


def _merge_observations(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = sorted(items, key=lambda x: (float(x["start"]), float(x["end"]), str(x["type"])))
    merged: list[dict[str, Any]] = []
    for item in items:
        if not merged:
            merged.append(item)
            continue
        prev = merged[-1]
        same_type = str(prev["type"]) == str(item["type"])
        close = float(item["start"]) - float(prev["end"]) <= 0.75
        if same_type and close:
            prev["end"] = round(max(float(prev["end"]), float(item["end"])), 3)
            if float(item["confidence"]) > float(prev["confidence"]):
                prev["description"] = item["description"]
            prev["confidence"] = max(float(prev["confidence"]), float(item["confidence"]))
            prev["observable_intensity"] = max(
                float(prev["observable_intensity"]), float(item["observable_intensity"])
            )
            prev["signals"] = list(dict.fromkeys(list(prev["signals"]) + list(item["signals"])))
        else:
            merged.append(item)
    return merged[:80]


def _majority(values: list[str], default: str) -> str:
    cleaned = [v for v in values if v and v != "unknown"]
    if not cleaned:
        return default
    return Counter(cleaned).most_common(1)[0][0]


def _build_report(
    *,
    duration: float,
    frame_count: int,
    layouts: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    events = _merge_observations(observations)

    reaction_times = sorted({float(e["start"]) for e in events if e["type"] == "reaction"})
    payoff_times = sorted({
        float(e["start"]) for e in events
        if e["type"] in {"visual_payoff", "impact", "crash", "fall", "explosion", "reveal"}
    })
    scene_times = sorted({
        float(e["start"]) for e in events if e["type"] in {"scene_change", "sudden_change"}
    })

    peak_candidates = [
        e for e in events
        if float(e["observable_intensity"]) >= 0.48 or e["signals"]
    ]
    peak_candidates.sort(
        key=lambda e: (
            -(0.72 * float(e["observable_intensity"]) + 0.28 * float(e["confidence"])),
            float(e["start"]),
        )
    )
    peaks: list[dict[str, Any]] = []
    for e in peak_candidates:
        if any(abs(float(e["start"]) - float(p["start"])) < 0.7 for p in peaks):
            continue
        peaks.append({
            "start": float(e["start"]),
            "end": float(e["end"]),
            "signals": list(e["signals"]),
            "description": str(e["description"]),
            "observable_intensity": float(e["observable_intensity"]),
            "confidence": float(e["confidence"]),
        })
        if len(peaks) >= 8:
            break

    orientations = [str(x.get("orientation", "unknown")) for x in layouts if isinstance(x, dict)]
    content_types = [str(x.get("content_type", "other")) for x in layouts if isinstance(x, dict)]
    notes = [str(x.get("notes", "")).strip() for x in layouts if isinstance(x, dict)]
    layout_notes = next((n for n in notes if n), "Sampled-frame layout estimate.")

    confs = [float(e["confidence"]) for e in events]
    mean_conf = sum(confs) / len(confs) if confs else 0.55
    coverage = min(1.0, frame_count / max(12.0, min(32.0, duration / 2.0 + 8.0)))
    analysis_conf = max(0.35, min(0.90, mean_conf * (0.72 + 0.22 * coverage)))

    return {
        "summary": (
            f"Terra sampled-frame fallback inspected {frame_count} timestamped frames "
            f"and recorded {len(events)} factual visual events across the source."
        ),
        "layout": {
            "orientation": _majority(orientations, "unknown"),
            "content_type": _majority(content_types, "other"),
            "notes": layout_notes,
        },
        "visual_events": [
            {
                "start": float(e["start"]),
                "end": float(e["end"]),
                "type": str(e["type"]),
                "description": str(e["description"]),
                "confidence": float(e["confidence"]),
            }
            for e in events
        ],
        "editing_support": {
            "reaction_times": reaction_times,
            "visual_payoff_times": payoff_times,
            "scene_change_times": scene_times,
            "strong_visual_moments": [],
            "notes": "Facts only. Terra decides editorial significance later.",
        },
        "intro_visual_support": {
            "visual_peak_regions": peaks,
            "notes": "Observable visual intensity from Terra sampled-frame fallback only.",
        },
        "quality": {
            "visual_clarity": 0.72 if frame_count >= 12 else 0.55,
            "analysis_confidence": round(analysis_conf, 3),
            "limitations": [
                "Gemini bypassed; OpenAI visual support analyzed timestamped sampled frames instead of continuous native video.",
                "Events occurring entirely between sampled frames may be missed.",
            ],
        },
        "warnings": list(dict.fromkeys([w for w in warnings if w]))[:20],
    }


def analyze_local_video_with_terra(
    video_path: str | Path,
    *,
    reason: str = "Gemini quota exhausted",
) -> dict[str, Any]:
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Terra visual fallback video bulamadı: {path}")

    probe = _probe_video(path)
    duration = float(probe["duration"])
    audio_peaks = _audio_peak_items(path, duration)
    timestamps = collect_sample_times(path, duration, audio_peaks)
    if not timestamps:
        raise RuntimeError("Terra visual fallback frame örneği seçemedi.")

    from ai import model_config
    from ai.openai_client import client

    visual_model = getattr(model_config, "VISUAL_SUPPORT_MODEL", model_config.EDITOR_MODEL)
    default_effort = getattr(model_config, "VISUAL_SUPPORT_REASONING_EFFORT", "medium")
    effort = str(os.getenv("MIMIR_VISUAL_SUPPORT_REASONING", default_effort)).strip().lower()
    if effort == "max":
        effort = "high"
    if effort not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
        effort = "medium"

    all_observations: list[dict[str, Any]] = []
    layouts: list[dict[str, Any]] = []
    warnings: list[str] = []
    response_ids: list[str] = []
    response_models: list[str] = []

    with tempfile.TemporaryDirectory(prefix="mimir_terra_visual_") as temp_dir:
        temp = Path(temp_dir)
        samples: list[tuple[float, Path]] = []
        for index, timestamp in enumerate(timestamps, start=1):
            frame = temp / f"frame_{index:03d}_{timestamp:010.3f}.jpg"
            if _extract_frame(path, timestamp, frame):
                samples.append((timestamp, frame))

        if len(samples) < 3:
            raise RuntimeError(
                f"Terra visual fallback yeterli frame çıkaramadı ({len(samples)})."
            )

        detail = "auto" if duration <= 90 else "low"
        batches = [samples[i:i + BATCH_SIZE] for i in range(0, len(samples), BATCH_SIZE)]

        for batch_index, batch in enumerate(batches, start=1):
            content: list[dict[str, Any]] = [
                {
                    "type": "input_text",
                    "text": (
                        f"VIDEO DURATION: {duration:.3f}s\n"
                        f"SOURCE SIZE: {probe['width']}x{probe['height']}\n"
                        f"BATCH: {batch_index}/{len(batches)}\n"
                        f"BATCH FRAME COUNT: {len(batch)}\n"
                        "Analyze only the supplied frames and return the required factual batch JSON."
                    ),
                }
            ]

            for local_index, (timestamp, frame) in enumerate(batch, start=1):
                content.append({
                    "type": "input_text",
                    "text": (
                        f"FRAME {local_index:02d}/{len(batch):02d} "
                        f"— exact sample timestamp t={timestamp:.3f}s"
                    ),
                })
                content.append({
                    "type": "input_image",
                    "image_url": _data_url(frame),
                    "detail": detail,
                })

            try:
                response = client.responses.create(
                    model=visual_model,
                    reasoning={"effort": effort},
                    instructions=TERRA_VISUAL_INSTRUCTIONS,
                    input=[{"role": "user", "content": content}],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "mimir_terra_visual_batch_v2",
                            "strict": True,
                            "schema": BATCH_SCHEMA,
                        }
                    },
                )
            except Exception as error:
                raise RuntimeError(
                    f"Terra visual fallback batch {batch_index}/{len(batches)} API hatası: {error}"
                ) from error

            output_text = _response_text(response)
            if not output_text:
                raise RuntimeError(
                    f"Terra visual fallback batch {batch_index}/{len(batches)} boş çıktı döndürdü."
                )

            try:
                payload = json.loads(output_text)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Terra visual fallback batch {batch_index}/{len(batches)} geçersiz JSON döndürdü: "
                    + output_text[:1200]
                ) from error

            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"Terra visual fallback batch {batch_index}/{len(batches)} root object değil."
                )

            layout = payload.get("layout", {})
            if isinstance(layout, dict):
                layouts.append(layout)

            raw_observations = payload.get("observations", [])
            if isinstance(raw_observations, list):
                for item in raw_observations:
                    normalized = _normalize_observation(item, duration)
                    if normalized is not None:
                        all_observations.append(normalized)

            raw_warnings = payload.get("warnings", [])
            if isinstance(raw_warnings, list):
                warnings.extend(str(w).strip() for w in raw_warnings if str(w).strip())

            response_id = str(getattr(response, "id", "") or "").strip()
            if response_id:
                response_ids.append(response_id)
            response_model = str(getattr(response, "model", "") or "").strip()
            if response_model:
                response_models.append(response_model)

    report = _build_report(
        duration=duration,
        frame_count=len(samples),
        layouts=layouts,
        observations=all_observations,
        warnings=warnings,
    )

    return {
        "backend": "terra_frame_fallback",
        "model": visual_model,
        "model_version": response_models[-1] if response_models else visual_model,
        "response_id": ",".join(response_ids[-4:]),
        "usage_metadata": {},
        "output_text": json.dumps(report, ensure_ascii=False),
        "remote_file": {},
        "fallback_reason": str(reason),
        "sampling": {
            "version": TERRA_VISUAL_FALLBACK_VERSION,
            "duration": round(duration, 3),
            "frame_count": len(samples),
            "batch_count": math.ceil(len(samples) / BATCH_SIZE),
            "batch_size": BATCH_SIZE,
            "image_detail": "auto" if duration <= 90 else "low",
            "timestamps": [round(t, 3) for t, _ in samples],
            "audio_peak_regions": audio_peaks[:MAX_AUDIO_GUIDED_SAMPLES],
            "audio_burst_bracketing": {
                "enabled": True,
                "peak_limit": AUDIO_GUIDED_PEAK_LIMIT,
                "pre_seconds": AUDIO_BURST_PRE_SECONDS,
                "post_seconds": AUDIO_BURST_POST_SECONDS,
            },
            "width": int(probe["width"]),
            "height": int(probe["height"]),
        },
    }
