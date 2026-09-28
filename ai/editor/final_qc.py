"""Final QC on the RENDERED short (the file that would be published).

Plans and JSON describe intent; this module re-derives facts from the actual
MP4 and its direct inputs, deterministically (FFmpeg decode + numpy):

    media_integrity           decodable video + audio, CFR, audio length == video length
    composition_clock         duration == cold open + main from the restart (hard cut)
    intro_structure           a real cold open exists, inside the generic safety bounds
    intro_is_source_footage   intro pixels == the clean peak footage (moving video, the real peak)
    no_captions_in_intro      the speech-caption band of every intro frame is untouched source
    intro_moving              the intro is not a frozen frame when the source moves
    intro_audio               intro audio is the source audio of the peak (present, in sync)
    intro_headline            an approved headline is visibly burned; no headline -> no text
    main_matches_render       the main part == the verified caption/camera render at the mapped time
    main_av_sync              main audio == render audio at the mapped time (no drift)
    caption_words_timing      every burned word == a frozen truth word at its measured onset,
                              nothing extra, nothing lost to the restart trim
    caption_collision         no two caption lines overlap in the same place at the same time
    protected_story_present   every protected story range survives the restart trim
    peak_recurs               the peak shown in the cold open is also inside the main story
    effects_clear             memes/effects never cover the caption band

The AI reviewer (final_review.py) may add bounded findings; it never overrides
these facts. A check that cannot run reports ``fail`` with the reason; nothing
here returns ``pass`` without having measured it.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

FINAL_QC_VERSION = 1

ANALYSIS_WIDTH = 320
DECODE_TIMEOUT_S = 600
AUDIO_RATE = 8000

INTRO_SAMPLES = 6
MAIN_SAMPLES = 8
INK_DELTA = 48                 # |final - source| above this is "drawn ink", not encoder noise
INK_SHARE_LIMIT = 0.004        # share of band pixels allowed to exceed INK_DELTA (noise / aliasing)
SOURCE_MATCH_MAD = 9.0         # mean |diff| of two encodes of the same picture
SOURCE_MATCH_CORR = 0.90
FROZEN_MOTION = 0.6            # mean |frame_n - frame_{n+1}| below this is a still picture
SOURCE_MOTION = 2.5            # ... and above this the source clearly moves
AUDIO_SILENT_DB = -55.0
AUDIO_LEVEL_TOLERANCE_DB = 6.0
AUDIO_MIN_CORRELATION = 0.60
AUDIO_MAX_LAG_S = 0.045
CAPTION_ONSET_TOLERANCE_S = 0.011   # one ASS centisecond (+ rounding)
BLOCKING = "blocking"


@dataclass
class QcContext:
    candidate: Path                       # the file that would be published
    timeline_doc: Mapping[str, Any]       # intro_renderer final timeline document
    intro_source: Path                    # the (clean / intro-camera) paced clip the cold open was cut from
    main_render: Path                     # verified caption (+camera) render of the paced clip
    burned_ass: Path | None               # captions burned into main_render
    truth_path: Path | None = None        # frozen caption truth document
    hook_ass: Path | None = None
    protected_paced: Sequence[Sequence[float]] = ()
    effect_windows: Sequence[Sequence[float]] = ()     # (start, end) in FINAL seconds
    display_labels: Sequence[str] = ()
    notes: list[str] = field(default_factory=list)


def _check(name: str, status: str, detail: str = "", **data: Any) -> dict[str, Any]:
    return {"check": name, "status": status, "detail": detail, "severity": BLOCKING, **data}


# ============================================================
# MEDIA ACCESS
# ============================================================

def _numpy() -> Any:
    import numpy as np

    return np


def probe_streams(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,"
         "sample_rate,channels", "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=120)
    if completed.returncode != 0:
        raise RuntimeError("ffprobe failed: " + completed.stderr.strip()[-300:])
    data = json.loads(completed.stdout or "{}")
    result: dict[str, Any] = {"format_duration": float((data.get("format") or {}).get("duration") or 0.0)}
    for stream in data.get("streams", []) or []:
        kind = stream.get("codec_type")
        if kind in ("video", "audio") and kind not in result:
            result[kind] = stream
    return result


def _rate(text: str) -> float:
    try:
        num, _, den = str(text).partition("/")
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def video_facts(path: Path) -> dict[str, Any]:
    streams = probe_streams(path)
    video = streams.get("video")
    if video is None:
        raise RuntimeError(f"no video stream: {path}")
    fps = _rate(video.get("r_frame_rate", "0/1")) or _rate(video.get("avg_frame_rate", "0/1"))
    duration = float(video.get("duration") or streams["format_duration"] or 0.0)
    try:
        frames = int(video.get("nb_frames") or 0)
    except (TypeError, ValueError):
        frames = 0
    audio = streams.get("audio")
    return {
        "width": int(video.get("width") or 0), "height": int(video.get("height") or 0), "fps": fps,
        "avg_fps": _rate(video.get("avg_frame_rate", "0/1")), "duration": duration,
        "frames": frames or int(round(duration * fps)),
        "audio_duration": float(audio.get("duration") or 0.0) if audio else None,
        "format_duration": streams["format_duration"],
    }


def decode_frames(path: Path, frames: Iterable[int], width: int, height: int) -> dict[int, Any]:
    """Exact decode-order frames as grayscale float arrays of (height, width)."""
    np = _numpy()
    wanted = sorted({int(f) for f in frames if int(f) >= 0})
    if not wanted:
        return {}
    select = "+".join(f"eq(n,{f})" for f in wanted)
    command = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path), "-an", "-sn",
               "-vf", f"select='{select}',scale={width}:{height}:flags=area,format=gray", "-fps_mode", "passthrough",
               "-frames:v", str(len(wanted)), "-f", "rawvideo", "-"]   # stop after the last wanted frame
    completed = subprocess.run(command, capture_output=True, timeout=DECODE_TIMEOUT_S, check=False)
    if completed.returncode != 0:
        raise RuntimeError("frame decode failed: " + completed.stderr.decode("utf-8", "replace")[-300:])
    size = width * height
    result = {}
    for index, frame in enumerate(wanted[: len(completed.stdout) // size]):
        chunk = completed.stdout[index * size:(index + 1) * size]
        result[frame] = np.frombuffer(chunk, dtype=np.uint8).reshape((height, width)).astype(np.float32)
    return result


def decode_audio(path: Path, start: float, duration: float) -> Any:
    np = _numpy()
    command = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-ss", f"{max(0.0, start):.4f}",
               "-i", str(path), "-t", f"{max(0.01, duration):.4f}", "-vn", "-ac", "1", "-ar", str(AUDIO_RATE),
               "-f", "s16le", "-"]
    completed = subprocess.run(command, capture_output=True, timeout=DECODE_TIMEOUT_S, check=False)
    if completed.returncode != 0:
        raise RuntimeError("audio decode failed: " + completed.stderr.decode("utf-8", "replace")[-300:])
    raw = completed.stdout[: len(completed.stdout) - len(completed.stdout) % 2]
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0


def _db(samples: Any) -> float:
    np = _numpy()
    if samples is None or len(samples) == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(samples * samples)))
    return 20.0 * math.log10(max(rms, 1e-6))


def best_lag(reference: Any, other: Any, max_lag_s: float = 0.12) -> tuple[float, float]:
    """(lag seconds, normalized correlation) of ``other`` against ``reference``."""
    np = _numpy()
    n = min(len(reference), len(other))
    if n < AUDIO_RATE // 10:
        return 0.0, 0.0
    a = reference[:n] - reference[:n].mean()
    b = other[:n] - other[:n].mean()
    limit = int(max_lag_s * AUDIO_RATE)
    norm = float(np.sqrt((a * a).sum() * (b * b).sum())) or 1.0
    scores = []
    for lag in range(-limit, limit + 1):
        if lag >= 0:
            value = float((a[lag:] * b[:n - lag]).sum())
        else:
            value = float((a[:n + lag] * b[-lag:]).sum())
        scores.append((lag, value / norm))
    peak = max(corr for _, corr in scores)
    # Periodic sound (a held tone, a beat) correlates equally at every period:
    # among (near-)equal peaks the smallest shift is the true alignment.
    lag, corr = min(((lag, corr) for lag, corr in scores if corr >= peak - 0.01), key=lambda row: abs(row[0]))
    return lag / AUDIO_RATE, corr


def _rows(height: int, band: tuple[float, float]) -> slice:
    return slice(max(0, int(math.floor(band[0] * height))), min(height, int(math.ceil(band[1] * height))))


def _mask_rows(shape: tuple[int, int], bands: Iterable[tuple[float, float]]) -> Any:
    np = _numpy()
    mask = np.ones(shape, dtype=bool)
    for band in bands:
        mask[_rows(shape[0], band), :] = False
    return mask


def _corr(a: Any, b: Any) -> float:
    np = _numpy()
    x, y = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / denominator) if denominator > 1e-6 else (1.0 if float(abs(a - b).mean()) < 1 else 0.0)


# ============================================================
# ASS PARSING (burned captions)
# ============================================================

_OVERRIDE = re.compile(r"(\{[^}]*\})")
_POS = re.compile(r"\\pos\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)")
_FS = re.compile(r"\\fs(-?[\d.]+)")
_TIME = re.compile(r"(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)")


def _ass_seconds(value: str) -> float:
    match = _TIME.fullmatch(value.strip())
    if not match:
        raise ValueError(f"bad ASS time {value!r}")
    return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))


def _plain(text: str) -> str:
    return text.replace("\\N", " ").replace("\\n", " ").replace("\\h", " ").replace("\\{", "{").replace(
        "\\}", "}").replace("\u2060", "").strip()


def parse_caption_lines(ass_path: Path) -> list[dict[str, Any]]:
    """Every text Dialogue as (start, end, style, pos, font, [(text, visible)]) - vector drawings skipped."""
    lines = []
    style_size: dict[str, float] = {}
    for raw in Path(ass_path).read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if raw.startswith("Style:"):
            parts = raw.split(":", 1)[1].split(",")
            if len(parts) > 2:
                try:
                    style_size[parts[0].strip()] = float(parts[2])
                except ValueError:
                    pass
            continue
        if not raw.startswith("Dialogue:"):
            continue
        parts = raw.split(":", 1)[1].split(",", 9)
        if len(parts) < 10:
            continue
        start, end, style, text = _ass_seconds(parts[1]), _ass_seconds(parts[2]), parts[3].strip(), parts[9]
        if re.search(r"\\p[1-9]", text):
            continue
        visible = True
        pieces: list[tuple[str, bool]] = []
        pos = None
        size = style_size.get(style, 0.0)
        for piece in _OVERRIDE.split(text):
            if piece.startswith("{") and piece.endswith("}"):
                if "\\1a&HFF&" in piece:
                    visible = False
                elif "\\1a&H00&" in piece:
                    visible = True
                match = _POS.search(piece)
                if match:
                    pos = (float(match.group(1)), float(match.group(2)))
                match = _FS.search(piece)
                if match:
                    size = float(match.group(1))
                continue
            word = _plain(piece)
            if word:
                pieces.append((word, visible))
        if pieces:
            lines.append({"start": start, "end": end, "style": style, "pos": pos, "font": size, "pieces": pieces})
    return lines


def burned_word_onsets(ass_path: Path, labels: Sequence[str] = ()) -> list[tuple[float, str]]:
    """(first-visible time, token) of every burned caption token, re-derived from the events.

    A caption line is shown as successive events (one per spoken word) over the
    same text; tags are only emitted on a state change, so adjacent words with
    the same state share one text piece: tokens are whitespace-separated. A
    token's onset is the first event in which it is visible."""
    label_tokens = {token for label in labels if label for token in f"{label}:".split()}
    groups: dict[tuple[Any, ...], list[list[tuple[str, bool]]]] = {}
    starts: dict[tuple[Any, ...], list[float]] = {}
    for line in parse_caption_lines(ass_path):
        tokens = [(token, visible) for text, visible in line["pieces"] for token in text.split()]
        key = (line["style"], line["pos"], tuple(token for token, _ in tokens))
        groups.setdefault(key, []).append(tokens)
        starts.setdefault(key, []).append(line["start"])
    onsets: list[tuple[float, str]] = []
    for key, events in groups.items():
        order = sorted(range(len(events)), key=lambda i: starts[key][i])
        for index, token in enumerate(key[2]):
            if token in label_tokens and index == 0:
                continue
            first = next((starts[key][i] for i in order if events[i][index][1]), None)
            if first is not None:
                onsets.append((round(first, 3), token))
    onsets.sort()
    return onsets


def _canon(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKC", str(text)).casefold() if ch.isalnum())


def truth_words(truth_path: Path) -> list[tuple[float, str]]:
    data = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    rows = []
    for row in data.get("words", []) or []:
        if isinstance(row, list) and len(row) >= 3 and str(row[1]).strip():
            # A truth word can hold several spoken tokens ("date with") sharing one onset.
            rows.extend((float(row[2]), token) for token in str(row[1]).split())
    rows.sort()
    return rows


# ============================================================
# CHECKS
# ============================================================

def check_media_integrity(ctx: QcContext, facts: dict[str, Any], main: dict[str, Any]) -> dict[str, Any]:
    problems = []
    if facts["frames"] <= 0 or facts["duration"] <= 0:
        problems.append("no decodable video frames")
    if facts["audio_duration"] is None:
        problems.append("no audio stream")
    elif abs(facts["audio_duration"] - facts["duration"]) > 0.10:
        problems.append(f"audio {facts['audio_duration']:.3f}s vs video {facts['duration']:.3f}s")
    if facts["fps"] <= 0 or (facts["avg_fps"] and abs(facts["avg_fps"] - facts["fps"]) / facts["fps"] > 0.002):
        problems.append(f"not constant frame rate ({facts['avg_fps']:.3f} vs {facts['fps']:.3f})")
    if (facts["width"], facts["height"]) != (main["width"], main["height"]):
        problems.append(f"size {facts['width']}x{facts['height']} != main render {main['width']}x{main['height']}")
    if problems:
        return _check("media_integrity", "fail", "; ".join(problems))
    return _check("media_integrity", "pass", f"{facts['width']}x{facts['height']} @ {facts['fps']:.3f} fps, "
                  f"{facts['frames']} frames, audio {facts['audio_duration']:.3f}s")


def check_composition_clock(ctx: QcContext, facts: dict[str, Any]) -> dict[str, Any]:
    expected = float(ctx.timeline_doc.get("expected_final_duration", 0.0) or 0.0)
    tolerance = 2.0 / max(1.0, facts["fps"]) + 0.03
    delta = abs(facts["duration"] - expected)
    if expected <= 0 or delta > tolerance:
        return _check("composition_clock", "fail",
                      f"rendered {facts['duration']:.3f}s vs cold open + main = {expected:.3f}s (delta {delta:.3f}s)")
    return _check("composition_clock", "pass", f"{facts['duration']:.3f}s == cold open + main (delta {delta:.3f}s)")


def check_intro_structure(ctx: QcContext) -> dict[str, Any]:
    from ai.editor import intro_bounds

    intro = ctx.timeline_doc.get("intro", {}) or {}
    restart = ctx.timeline_doc.get("restart", {}) or {}
    duration = float(intro.get("duration", 0.0) or 0.0)
    problems = []
    if duration < intro_bounds.MIN_INTRO_S - 0.05:
        problems.append(f"cold open {duration:.3f}s shorter than {intro_bounds.MIN_INTRO_S:.1f}s")
    if duration > intro_bounds.MAX_INTRO_S + 0.05:
        problems.append(f"cold open {duration:.3f}s longer than {intro_bounds.MAX_INTRO_S:.1f}s")
    if intro.get("normal_captions") is not False:
        problems.append("cold open source is not the clean clip")
    if restart.get("transition") != "hard_cut":
        problems.append(f"restart is {restart.get('transition')!r}, not a hard cut")
    if problems:
        return _check("intro_structure", "fail", "; ".join(problems))
    return _check("intro_structure", "pass", f"{duration:.3f}s cold open from clean footage, hard restart")


def _intro_samples(ctx: QcContext, fps: float) -> list[tuple[int, int, float]]:
    intro = ctx.timeline_doc.get("intro", {}) or {}
    duration = float(intro.get("duration", 0.0) or 0.0)
    paced_start = float((intro.get("paced") or [0.0])[0])
    frames = max(1, int(round(duration * fps)))
    picks = sorted({min(frames - 1, max(0, int(round(frames * (k + 0.5) / INTRO_SAMPLES)))) for k in range(INTRO_SAMPLES)})
    source_first = int(round(paced_start * fps))
    return [(n, source_first + n, n / fps) for n in picks]


def _hook_band(ctx: QcContext, width: int, height: int) -> tuple[tuple[float, float] | None, float]:
    """(band, display seconds) of the burned headline, from the renderer's own layout functions."""
    intro = ctx.timeline_doc.get("intro", {}) or {}
    text = str(intro.get("headline", "") or "")
    window = intro.get("headline_final")
    if not text or not window:
        return None, 0.0
    from ai.editor import intro_renderer as ir

    line_1, line_2 = ir.split_balanced_two_lines(text)
    font = ir.estimate_font_size(line_1=line_1, line_2=line_2, width=width, height=height)
    ys = [height * ir.LINE_1_Y_RATIO, height * ir.LINE_2_Y_RATIO] if line_2 else [height * ir.SINGLE_LINE_Y_RATIO]
    half = font * max(ir.START_SCALE, ir.BOUNCE_SCALE, ir.PULSE_SCALE) / 100.0 / 2.0 + ir.MAIN_OUTLINE
    band = (max(0.0, (min(ys) - half) / height - 0.02), min(1.0, (max(ys) + half) / height + 0.02))
    return band, float(window[1])


def _caption_bands(ctx: QcContext) -> list[tuple[float, float]]:
    if ctx.burned_ass is None or not Path(ctx.burned_ass).is_file():
        return []
    from ai.editor.pro_edit.caption_guard import caption_safe_region

    region = caption_safe_region(ctx.burned_ass)
    return sorted({(round(b.y0, 4), round(b.y1, 4)) for b in region.bands})


def check_intro_pixels(ctx: QcContext, facts: dict[str, Any]) -> list[dict[str, Any]]:
    np = _numpy()
    fps = facts["fps"]
    width = ANALYSIS_WIDTH
    height = max(16, int(round(facts["height"] * width / max(1, facts["width"]))))
    height -= height % 2
    samples = _intro_samples(ctx, fps)
    final = decode_frames(ctx.candidate, [s[0] for s in samples] + [s[0] + 1 for s in samples], width, height)
    source = decode_frames(ctx.intro_source, [s[1] for s in samples] + [s[1] + 1 for s in samples], width, height)
    hook, hook_until = _hook_band(ctx, facts["width"], facts["height"])
    caption_bands = _caption_bands(ctx)
    rows: list[dict[str, Any]] = []
    worst_ink, worst_mad, worst_corr = 0.0, 0.0, 1.0
    final_motion, source_motion = [], []
    headline_ink = []
    missing = []
    for final_n, source_n, t in samples:
        a, b = final.get(final_n), source.get(source_n)
        if a is None or b is None:
            missing.append(final_n)
            continue
        excluded = [hook] if hook is not None and t < hook_until else []
        mask = _mask_rows(a.shape, excluded)
        diff = np.abs(a - b)
        worst_mad = max(worst_mad, float(diff[mask].mean()))
        worst_corr = min(worst_corr, _corr(a[mask], b[mask]))
        for band in caption_bands:
            rows_ = _rows(a.shape[0], band)
            if hook is not None and t < hook_until and band[0] < hook[1] and hook[0] < band[1]:
                continue
            worst_ink = max(worst_ink, float((diff[rows_, :] > INK_DELTA).mean()))
        if hook is not None and t < hook_until:
            headline_ink.append(float((diff[_rows(a.shape[0], hook), :] > INK_DELTA).mean()))
        a2, b2 = final.get(final_n + 1), source.get(source_n + 1)
        if a2 is not None and b2 is not None:
            final_motion.append(float(np.abs(a2 - a)[mask].mean()))
            source_motion.append(float(np.abs(b2 - b)[mask].mean()))
    if missing:
        fail = f"could not decode intro frames {missing[:4]}"
        return [_check(name, "fail", fail) for name in ("intro_is_source_footage", "no_captions_in_intro",
                                                          "intro_moving", "intro_headline")]
    if worst_mad <= SOURCE_MATCH_MAD and worst_corr >= SOURCE_MATCH_CORR:
        rows.append(_check("intro_is_source_footage", "pass",
                           f"{len(samples)} frames == clean peak footage (MAD {worst_mad:.2f}, corr {worst_corr:.3f})"))
    else:
        rows.append(_check("intro_is_source_footage", "fail",
                           f"intro frames differ from the clean peak footage (MAD {worst_mad:.2f}, corr {worst_corr:.3f})"))
    if not caption_bands:
        rows.append(_check("no_captions_in_intro", "fail", "burned caption band unknown; cannot prove a caption-free intro"))
    elif worst_ink <= INK_SHARE_LIMIT:
        rows.append(_check("no_captions_in_intro", "pass",
                           f"caption band untouched in every intro frame (ink share {worst_ink:.4f})"))
    else:
        rows.append(_check("no_captions_in_intro", "fail",
                           f"text-like ink in the speech-caption band during the cold open (share {worst_ink:.4f})"))
    moving_source = [m for m in source_motion if m >= SOURCE_MOTION]
    frozen = [f for f, s in zip(final_motion, source_motion) if s >= SOURCE_MOTION and f < FROZEN_MOTION]
    if not final_motion:
        rows.append(_check("intro_moving", "fail", "no consecutive intro frames decoded"))
    elif frozen:
        rows.append(_check("intro_moving", "fail", f"{len(frozen)} intro sample(s) frozen while the source moves"))
    else:
        rows.append(_check("intro_moving", "pass", f"moving footage (final motion {max(final_motion):.2f}, "
                           f"source {max(source_motion):.2f}; {len(moving_source)} moving sample(s))"))
    headline = str((ctx.timeline_doc.get("intro", {}) or {}).get("headline", "") or "")
    if headline:
        if headline_ink and max(headline_ink) > INK_SHARE_LIMIT * 5:
            rows.append(_check("intro_headline", "pass", f"headline {headline!r} burned (ink {max(headline_ink):.3f})"))
        else:
            rows.append(_check("intro_headline", "fail", f"approved headline {headline!r} not visible in the intro"))
    else:
        rows.append(_check("intro_headline", "pass", "no headline approved; cold open carries no text"))
    return rows


def check_intro_audio(ctx: QcContext) -> dict[str, Any]:
    intro = ctx.timeline_doc.get("intro", {}) or {}
    duration = float(intro.get("duration", 0.0) or 0.0)
    paced_start = float((intro.get("paced") or [0.0])[0])
    edge = min(0.05, duration / 10.0)
    final = decode_audio(ctx.candidate, edge, duration - 2 * edge)
    source = decode_audio(ctx.intro_source, paced_start + edge, duration - 2 * edge)
    final_db, source_db = _db(final), _db(source)
    if source_db < AUDIO_SILENT_DB:
        if final_db < AUDIO_SILENT_DB + AUDIO_LEVEL_TOLERANCE_DB:
            return _check("intro_audio", "pass", f"source peak is silent ({source_db:.1f} dB) and so is the intro")
        return _check("intro_audio", "fail", f"intro has sound ({final_db:.1f} dB) the silent source does not")
    if final_db < AUDIO_SILENT_DB:
        return _check("intro_audio", "fail", f"intro is silent ({final_db:.1f} dB); source {source_db:.1f} dB")
    if abs(final_db - source_db) > AUDIO_LEVEL_TOLERANCE_DB:
        return _check("intro_audio", "fail", f"intro level {final_db:.1f} dB vs source {source_db:.1f} dB")
    lag, corr = best_lag(source, final)
    if corr < AUDIO_MIN_CORRELATION or abs(lag) > AUDIO_MAX_LAG_S:
        return _check("intro_audio", "fail", f"intro audio is not the peak's audio (corr {corr:.2f}, lag {lag * 1000:.0f} ms)")
    return _check("intro_audio", "pass", f"real peak audio, {final_db:.1f} dB, corr {corr:.2f}, lag {lag * 1000:.0f} ms")


def _in_windows(t: float, windows: Sequence[Sequence[float]], pad: float = 0.3) -> bool:
    return any(float(a) - pad <= t <= float(b) + pad for a, b in windows)


def check_main_pixels(ctx: QcContext, facts: dict[str, Any], main: dict[str, Any]) -> dict[str, Any]:
    np = _numpy()
    fps = facts["fps"]
    intro_frames = int(round(float(ctx.timeline_doc["intro"]["duration"]) * fps))
    restart_frames = int(round(float(ctx.timeline_doc["restart"]["main_restart_paced"]) * fps))
    main_frames = int(main["frames"])
    usable = main_frames - restart_frames - 3
    if usable <= 2:
        return _check("main_matches_render", "fail", "main part too short to verify")
    pairs = []
    for k in range(MAIN_SAMPLES):
        offset = 1 + int(round(usable * (k + 0.5) / MAIN_SAMPLES))
        final_n = intro_frames + offset
        if _in_windows(final_n / fps, ctx.effect_windows):
            continue
        pairs.append((final_n, restart_frames + offset))
    width = ANALYSIS_WIDTH
    height = max(16, int(round(facts["height"] * width / max(1, facts["width"]))))
    height -= height % 2
    final = decode_frames(ctx.candidate, [a for a, _ in pairs], width, height)
    render = decode_frames(ctx.main_render, [b for _, b in pairs], width, height)
    worst_mad, worst_corr, checked = 0.0, 1.0, 0
    for a_n, b_n in pairs:
        a, b = final.get(a_n), render.get(b_n)
        if a is None or b is None:
            return _check("main_matches_render", "fail", f"could not decode main frame pair {a_n}/{b_n}")
        worst_mad = max(worst_mad, float(np.abs(a - b).mean()))
        worst_corr = min(worst_corr, _corr(a, b))
        checked += 1
    if checked == 0:
        return _check("main_matches_render", "fail", "no main frame outside effect windows to verify")
    if worst_mad <= SOURCE_MATCH_MAD and worst_corr >= SOURCE_MATCH_CORR:
        return _check("main_matches_render", "pass", f"{checked} frames == verified caption/camera render at the "
                      f"mapped time (MAD {worst_mad:.2f}, corr {worst_corr:.3f})")
    return _check("main_matches_render", "fail", f"main part differs from the verified render at the mapped time "
                  f"(MAD {worst_mad:.2f}, corr {worst_corr:.3f}) - offset, drift or overpaint")


def check_main_av_sync(ctx: QcContext, facts: dict[str, Any], main: dict[str, Any]) -> dict[str, Any]:
    intro_duration = float(ctx.timeline_doc["intro"]["duration"])
    restart = float(ctx.timeline_doc["restart"]["main_restart_paced"])
    main_span = float(main["duration"]) - restart
    windows = []
    for fraction in (0.2, 0.5, 0.8):
        paced = restart + main_span * fraction
        final_t = intro_duration + (paced - restart)
        if main_span < 2.5 or _in_windows(final_t, ctx.effect_windows, pad=1.2):
            continue
        windows.append((paced, final_t))
    if not windows:
        return _check("main_av_sync", "fail", "no main audio window outside effect windows to verify")
    worst_lag, worst_corr = 0.0, 1.0
    silent = 0
    for paced, final_t in windows:
        reference = decode_audio(ctx.main_render, paced - 1.0, 2.0)
        final = decode_audio(ctx.candidate, final_t - 1.0, 2.0)
        if _db(reference) < AUDIO_SILENT_DB:
            silent += 1
            if _db(final) >= AUDIO_SILENT_DB + AUDIO_LEVEL_TOLERANCE_DB:
                return _check("main_av_sync", "fail", f"sound at {final_t:.2f}s where the story is silent")
            continue
        lag, corr = best_lag(reference, final)
        worst_lag = max(worst_lag, abs(lag))
        worst_corr = min(worst_corr, corr)
    if worst_corr < AUDIO_MIN_CORRELATION or worst_lag > AUDIO_MAX_LAG_S:
        return _check("main_av_sync", "fail", f"main audio drift/mismatch (corr {worst_corr:.2f}, lag {worst_lag * 1000:.0f} ms)")
    return _check("main_av_sync", "pass", f"{len(windows) - silent} window(s) in sync (corr >= {worst_corr:.2f}, "
                  f"|lag| <= {worst_lag * 1000:.0f} ms)" + (f"; {silent} silent" if silent else ""))


def check_caption_words(ctx: QcContext, facts: dict[str, Any]) -> dict[str, Any]:
    if ctx.burned_ass is None or not Path(ctx.burned_ass).is_file():
        return _check("caption_words_timing", "fail", "burned caption ASS missing")
    if ctx.truth_path is None or not Path(ctx.truth_path).is_file():
        return _check("caption_words_timing", "fail", "frozen caption truth missing")
    burned = burned_word_onsets(Path(ctx.burned_ass), ctx.display_labels)
    truth = truth_words(Path(ctx.truth_path))
    restart = float(ctx.timeline_doc["restart"]["main_restart_paced"])
    intro_duration = float(ctx.timeline_doc["intro"]["duration"])
    pool = [[t, _canon(text), text, False] for t, text in burned]
    missing, mistimed = [], []
    for start, text in truth:
        key = _canon(text)
        expected = round(start + 1e-9, 2)
        candidates = [row for row in pool if not row[3] and row[1] == key and abs(row[0] - expected) <= 0.5]
        if not candidates:
            missing.append(f"{text}@{start:.2f}")
            continue
        row = min(candidates, key=lambda r: abs(r[0] - expected))
        row[3] = True
        if abs(row[0] - expected) > CAPTION_ONSET_TOLERANCE_S:
            mistimed.append(f"{text}: burned {row[0]:.2f}s vs spoken {start:.3f}s")
    extra = [f"{row[2]}@{row[0]:.2f}" for row in pool if not row[3]]
    trimmed = [f"{text}@{start:.2f}" for start, text in truth if start < restart - 1e-3]
    beyond = [f"{text}@{start:.2f}" for start, text in truth
              if intro_duration + (start - restart) > facts["duration"] + 1e-3]
    problems = []
    if missing:
        problems.append(f"{len(missing)} truth word(s) never burned: {', '.join(missing[:5])}")
    if mistimed:
        problems.append(f"{len(mistimed)} word(s) mistimed: {'; '.join(mistimed[:4])}")
    if extra:
        problems.append(f"{len(extra)} burned word(s) not in the frozen truth: {', '.join(extra[:5])}")
    if trimmed:
        problems.append(f"{len(trimmed)} spoken word(s) cut by the restart: {', '.join(trimmed[:5])}")
    if beyond:
        problems.append(f"{len(beyond)} word(s) after the end of the short")
    if problems:
        return _check("caption_words_timing", "fail", "; ".join(problems))
    return _check("caption_words_timing", "pass", f"{len(truth)} burned tokens == frozen truth text at their "
                  f"measured onsets (±{CAPTION_ONSET_TOLERANCE_S * 1000:.0f} ms), none lost to the restart")


def check_caption_collision(ctx: QcContext) -> dict[str, Any]:
    if ctx.burned_ass is None or not Path(ctx.burned_ass).is_file():
        return _check("caption_collision", "fail", "burned caption ASS missing")
    lines = [line for line in parse_caption_lines(Path(ctx.burned_ass)) if line["pos"] is not None]
    collisions = []
    for i, a in enumerate(lines):
        for b in lines[i + 1:]:
            if b["start"] >= a["end"]:
                continue
            if a["start"] >= b["end"] or tuple(p for p, _ in a["pieces"]) == tuple(p for p, _ in b["pieces"]):
                continue
            gap = abs(a["pos"][1] - b["pos"][1])
            if gap < 0.85 * max(a["font"], b["font"], 1.0):
                collisions.append(f"{a['start']:.2f}s y={a['pos'][1]:.0f}/{b['pos'][1]:.0f}")
    if collisions:
        return _check("caption_collision", "fail", f"{len(collisions)} overlapping caption line(s): "
                      + ", ".join(collisions[:4]))
    return _check("caption_collision", "pass", f"{len(lines)} positioned caption events, no overlap")


def check_protected_story(ctx: QcContext, main: dict[str, Any]) -> dict[str, Any]:
    restart = float(ctx.timeline_doc["restart"]["main_restart_paced"])
    protected = list(ctx.protected_paced) or list((ctx.timeline_doc.get("restart") or {}).get("protected_paced") or [])
    lost = [(round(float(a), 3), round(float(b), 3)) for a, b in protected
            if float(a) < restart - 1e-3 or float(b) > float(main["duration"]) + 0.05]
    if lost:
        return _check("protected_story_present", "fail", f"protected story range(s) cut from the short: {lost[:4]}")
    return _check("protected_story_present", "pass", f"{len(protected)} protected range(s) inside the main story")


def check_peak_recurs(ctx: QcContext, main: dict[str, Any]) -> dict[str, Any]:
    intro = ctx.timeline_doc.get("intro", {}) or {}
    restart = float(ctx.timeline_doc["restart"]["main_restart_paced"])
    paced = intro.get("paced") or [0.0, 0.0]
    peak = intro.get("peak") or {}
    core = (float(peak.get("peak_start", paced[0]) or paced[0]), float(peak.get("peak_end", paced[1]) or paced[1]))
    if core[0] < restart - 1e-3:
        return _check("peak_recurs", "fail", f"the peak ({core[0]:.2f}s) lies before the story restart ({restart:.2f}s)")
    if core[1] > float(main["duration"]) + 0.05:
        return _check("peak_recurs", "fail", "the peak lies after the end of the main story")
    return _check("peak_recurs", "pass", f"peak {core[0]:.2f}-{core[1]:.2f}s recurs inside the main story")


def check_effects_clear(ctx: QcContext, facts: dict[str, Any]) -> dict[str, Any]:
    if not ctx.effect_windows:
        return _check("effects_clear", "pass", "no visual effect in this short")
    np = _numpy()
    fps = facts["fps"]
    intro_duration = float(ctx.timeline_doc["intro"]["duration"])
    restart = float(ctx.timeline_doc["restart"]["main_restart_paced"])
    bands = _caption_bands(ctx)
    width = ANALYSIS_WIDTH
    height = max(16, int(round(facts["height"] * width / max(1, facts["width"]))))
    height -= height % 2
    worst = 0.0
    for a, b in ctx.effect_windows:
        if float(a) < intro_duration:
            return _check("effects_clear", "fail", f"effect at {float(a):.2f}s inside the cold open")
        t = (float(a) + float(b)) / 2.0
        final_n = int(round(t * fps))
        render_n = int(round((restart + t - intro_duration) * fps))
        final = decode_frames(ctx.candidate, [final_n], width, height).get(final_n)
        render = decode_frames(ctx.main_render, [render_n], width, height).get(render_n)
        if final is None or render is None:
            return _check("effects_clear", "fail", "could not decode an effect frame")
        diff = np.abs(final - render)
        for band in bands:
            worst = max(worst, float((diff[_rows(height, band), :] > INK_DELTA).mean()))
    if worst > INK_SHARE_LIMIT * 3:
        return _check("effects_clear", "fail", f"an effect covers the caption band (share {worst:.3f})")
    return _check("effects_clear", "pass", f"{len(ctx.effect_windows)} effect(s) clear of captions and cold open")


def run_final_qc(ctx: QcContext) -> list[dict[str, Any]]:
    """Every check, each isolated: a QC bug becomes a failed check, never a crash or a pass."""
    rows: list[dict[str, Any]] = []

    def guarded(name: str, function: Any, *args: Any) -> None:
        try:
            result = function(*args)
            rows.extend(result if isinstance(result, list) else [result])
        except Exception as error:
            rows.append(_check(name, "fail", f"QC could not measure: {type(error).__name__}: {str(error)[:300]}"))

    try:
        facts = video_facts(Path(ctx.candidate))
        main = video_facts(Path(ctx.main_render))
    except Exception as error:
        return [_check("media_integrity", "fail", f"unreadable media: {type(error).__name__}: {error}")]
    guarded("media_integrity", check_media_integrity, ctx, facts, main)
    guarded("composition_clock", check_composition_clock, ctx, facts)
    guarded("intro_structure", check_intro_structure, ctx)
    guarded("intro_pixels", check_intro_pixels, ctx, facts)
    guarded("intro_audio", check_intro_audio, ctx)
    guarded("main_matches_render", check_main_pixels, ctx, facts, main)
    guarded("main_av_sync", check_main_av_sync, ctx, facts, main)
    guarded("caption_words_timing", check_caption_words, ctx, facts)
    guarded("caption_collision", check_caption_collision, ctx)
    guarded("protected_story_present", check_protected_story, ctx, main)
    guarded("peak_recurs", check_peak_recurs, ctx, main)
    guarded("effects_clear", check_effects_clear, ctx, facts)
    return rows
