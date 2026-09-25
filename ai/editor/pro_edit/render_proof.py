"""Pixel proof: the planned camera must be visible in the rendered frames.

A resolved camera plan (or a filter script) existing is not evidence that the
render applied it. For sampled frames this module decodes the SOURCE (clean
paced clip) and the RENDERED output, predicts the output from the source
with the planned crop, and scores both the prediction and the untouched
source against what was actually rendered:

    camera frame   -> the planned crop explains the pixels clearly better
                      than the untouched source ("reached")
    control frame  -> outside every camera op the pixels equal the source

Scores are structure-only (high-pass image, normalized cross-correlation) so
encoder noise and colour grading do not matter; burned caption / hook-text
bands are masked. The same comparison proves that the FINAL published short
still contains the camera-rendered main (intro + xfade offset from the intro
timeline) and the intro camera.

Deterministic and local (FFmpeg decode + OpenCV/numpy); no model, no
network. Without OpenCV/numpy the proof reports ``unavailable`` (explicitly,
never as a pass).
"""
from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ai.editor.pro_edit.camera import IDENTITY, CameraState, crop_window
from ai.editor.pro_edit.framing import Box, crop_contains

RENDER_PROOF_VERSION = 1
ANALYSIS_WIDTH = 320
SOURCE_OVERSAMPLE = 2
MIN_MATCH = 0.55                 # structural correlation that counts as "the same picture"
MIN_MARGIN = 0.03                # planned crop must beat the untouched source by this much
MIN_PROOF_ZOOM = 1.02            # smaller planned changes are not provable at analysis size
MIN_TEXTURE = 1.5                # high-pass std below this: frame too flat to prove anything
MAX_CAMERA_SAMPLES = 8
CONTROL_SAMPLES = 2
BAND_PAD = 0.03
EDGE_PX = 4
DECODE_TIMEOUT_S = 600


class ProofUnavailable(RuntimeError):
    """Pixel analysis cannot run here (no OpenCV/numpy or undecodable media)."""


def _numpy() -> tuple[Any, Any]:
    from ai.editor.pro_edit.vision.cv_runtime import OpenCVUnavailable, load_opencv

    try:
        import numpy as np

        cv2 = load_opencv()
    except OpenCVUnavailable as error:  # exact cause + remedy (e.g. blocked by Windows Smart App Control)
        raise ProofUnavailable(f"OpenCV unavailable: {error.reason}") from error
    except Exception as error:  # optional runtime dependency
        raise ProofUnavailable(f"OpenCV/numpy unavailable: {type(error).__name__}: {error}") from error
    return cv2, np


# ============================================================
# FRAME ACCESS
# ============================================================

def analysis_size(width: int, height: int, target_width: int = ANALYSIS_WIDTH) -> tuple[int, int]:
    w = max(16, int(target_width))
    h = max(16, int(round(height * w / max(1, width))))
    return w - w % 2, h - h % 2


def decode_gray_frames(path: str | Path, frames: Iterable[int], size: tuple[int, int]) -> dict[int, Any]:
    """Exact frames (0-based decode index) as grayscale arrays of ``size`` (w, h)."""
    _cv2, np = _numpy()
    wanted = sorted({int(f) for f in frames if int(f) >= 0})
    if not wanted:
        return {}
    w, h = size
    select = "+".join(f"eq(n,{f})" for f in wanted)
    command = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(Path(path)), "-an", "-sn",
               "-vf", f"select='{select}',scale={w}:{h}:flags=area,format=gray", "-fps_mode", "passthrough",
               "-f", "rawvideo", "-"]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=DECODE_TIMEOUT_S, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ProofUnavailable(f"frame decode failed: {type(error).__name__}: {error}") from error
    if completed.returncode != 0:
        raise ProofUnavailable("frame decode failed: " + completed.stderr.decode("utf-8", "replace")[-300:])
    frame_bytes = w * h
    count = len(completed.stdout) // frame_bytes
    result: dict[int, Any] = {}
    for index, frame in enumerate(wanted[:count]):
        chunk = completed.stdout[index * frame_bytes:(index + 1) * frame_bytes]
        result[frame] = np.frombuffer(chunk, dtype=np.uint8).reshape((h, w))
    return result


# ============================================================
# COMPARISON
# ============================================================

def predict_view(source: Any, state: CameraState, base: tuple[int, int, int, int], source_size: tuple[int, int],
                 out_size: tuple[int, int]) -> Any:
    """What the renderer should show: base window, then the planned crop, scaled to the output."""
    cv2, _np = _numpy()
    sh, sw = source.shape[:2]
    fx, fy = sw / float(source_size[0]), sh / float(source_size[1])
    bx, by, bw, bh = base
    window = source[int(round(by * fy)):int(round((by + bh) * fy)), int(round(bx * fx)):int(round((bx + bw) * fx))]
    wh, ww = window.shape[:2]
    x, y, cw, ch = crop_window(state, float(ww), float(wh))
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(ww, int(math.ceil(x + cw))), min(wh, int(math.ceil(y + ch)))
    crop = window[y0:y1, x0:x1]
    return cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA)


def structure(image: Any) -> Any:
    cv2, np = _numpy()
    value = image.astype(np.float32)
    return value - cv2.GaussianBlur(value, (0, 0), 2.0)


def band_mask(size: tuple[int, int], bands: Sequence[tuple[float, float]]) -> Any:
    """True = compared pixel. Bands are output-normalized (y0, y1) rows to ignore."""
    _cv2, np = _numpy()
    w, h = size
    mask = np.ones((h, w), dtype=bool)
    mask[:EDGE_PX, :] = mask[-EDGE_PX:, :] = False
    mask[:, :EDGE_PX] = mask[:, -EDGE_PX:] = False
    for y0, y1 in bands:
        top = max(0, int(math.floor((y0 - BAND_PAD) * h)))
        bottom = min(h, int(math.ceil((y1 + BAND_PAD) * h)))
        mask[top:bottom, :] = False
    return mask


def masked_ncc(a: Any, b: Any, mask: Any) -> float | None:
    _cv2, np = _numpy()
    va = a[mask].astype(np.float64)
    vb = b[mask].astype(np.float64)
    if va.size < 64:
        return None
    va -= va.mean()
    vb -= vb.mean()
    denominator = math.sqrt(float((va * va).sum()) * float((vb * vb).sum()))
    if denominator <= 1e-9:
        return None
    return float((va * vb).sum() / denominator)


@dataclass(frozen=True)
class ProofTarget:
    frame: int
    state: CameraState
    kind: str          # "camera" | "control"
    event_id: str = ""


def select_targets(resolved: Any, *, first_frame: int = 0, max_camera: int = MAX_CAMERA_SAMPLES,
                   controls: int = CONTROL_SAMPLES) -> list[ProofTarget]:
    """Peak frame of each camera op (strongest change) + untouched control frames."""
    path = resolved.path
    targets: list[ProofTarget] = []
    for op in sorted(resolved.ops, key=lambda o: o.start_frame):
        lo = max(op.start_frame, first_frame)
        if lo >= op.end_frame:
            continue
        best: tuple[float, int] | None = None
        step = max(1, (op.end_frame - lo) // 24)
        for frame in range(lo, op.end_frame, step):
            state = path.state_at(frame)
            strength = max(state.zoom - 1.0, abs(state.cx - 0.5) * 0.2, abs(state.cy - 0.5) * 0.2)
            if best is None or strength > best[0]:
                best = (strength, frame)
        if best is None:
            continue
        state = path.state_at(best[1])
        if state.zoom < MIN_PROOF_ZOOM:
            continue
        targets.append(ProofTarget(best[1], state, "camera", op.source_event_id))
    if len(targets) > max_camera:
        ranked = sorted(targets, key=lambda t: -t.state.zoom)[:max_camera]
        targets = sorted(ranked, key=lambda t: t.frame)
    active = path.active_ranges()
    gaps: list[tuple[int, int]] = []
    cursor = first_frame
    for a, b in active + [(resolved.frame_count, resolved.frame_count)]:
        if a - cursor >= 6:
            gaps.append((cursor, a))
        cursor = max(cursor, b)
    for a, b in sorted(gaps, key=lambda g: g[0] - g[1])[:controls]:
        targets.append(ProofTarget((a + b) // 2, IDENTITY, "control"))
    return sorted(targets, key=lambda t: t.frame)


def _verdict(target: ProofTarget, planned: float | None, identity: float | None, texture: float) -> str:
    if planned is None or identity is None or texture < MIN_TEXTURE:
        return "inconclusive"
    if target.kind == "control":
        return "identity_ok" if identity >= MIN_MATCH else "inconclusive"
    if planned >= MIN_MATCH and planned - identity >= MIN_MARGIN:
        return "reached"
    if identity >= MIN_MATCH and identity - planned >= MIN_MARGIN:
        return "missing"
    return "inconclusive"


def _summarize(samples: list[dict[str, Any]], *, what: str) -> tuple[str, str]:
    camera = [s for s in samples if s["kind"] == "camera"]
    if not camera:
        return "no_camera_ops", f"{what}: the plan has no provable camera change"
    missing = [s for s in camera if s["verdict"] == "missing"]
    reached = [s for s in camera if s["verdict"] == "reached"]
    if missing:
        return "failed", (f"{what}: {len(missing)}/{len(camera)} planned camera frame(s) look like the untouched "
                          f"source ({', '.join(str(s['frame']) for s in missing[:6])})")
    if reached:
        return "passed", f"{what}: {len(reached)}/{len(camera)} planned camera frame(s) verified in pixels"
    return "inconclusive", f"{what}: no sampled frame had enough structure to prove the camera"


def prove_camera(*, source_path: str | Path, rendered_path: str | Path, resolved: Any,
                 source_size: tuple[int, int], caption_region: Any = None, hook_band: Any = None,
                 first_frame: int = 0, fps: float | None = None) -> dict[str, Any]:
    """Main (or intro) render proof: planned camera vs rendered frames vs untouched source."""
    try:
        targets = select_targets(resolved, first_frame=first_frame)
        if not targets:
            return {"version": RENDER_PROOF_VERSION, "status": "no_camera_ops", "samples": [],
                    "reason": "nothing to sample"}
        out_w, out_h = resolved.output_size
        size = analysis_size(out_w, out_h)
        src_size = analysis_size(source_size[0], source_size[1], ANALYSIS_WIDTH * SOURCE_OVERSAMPLE)
        frames = [t.frame for t in targets]
        rendered = decode_gray_frames(rendered_path, frames, size)
        source = decode_gray_frames(source_path, frames, src_size)
        rate = float(fps or resolved.fps.fps)
        samples: list[dict[str, Any]] = []
        for target in targets:
            if target.frame not in rendered or target.frame not in source:
                samples.append({"frame": target.frame, "kind": target.kind, "event_id": target.event_id,
                                "verdict": "inconclusive", "reason": "frame not decoded"})
                continue
            t = target.frame / rate
            bands: list[tuple[float, float]] = []
            if caption_region is not None:
                bands.extend((b.y0, b.y1) for b in caption_region.active_bands(t - 0.3, t + 0.3))
            if hook_band is not None and hook_band.active(t - 0.1, t + 0.1):
                bands.append((hook_band.y0, hook_band.y1))
            mask = band_mask(size, bands)
            actual = structure(rendered[target.frame])
            planned_view = structure(predict_view(source[target.frame], target.state, resolved.base, source_size,
                                                  size))
            identity_view = structure(predict_view(source[target.frame], IDENTITY, resolved.base, source_size, size))
            planned = masked_ncc(actual, planned_view, mask)
            identity = masked_ncc(actual, identity_view, mask)
            texture = float(actual[mask].std()) if mask.any() else 0.0
            verdict = _verdict(target, planned, identity, texture)
            samples.append({"frame": target.frame, "t": round(t, 3), "kind": target.kind,
                            "event_id": target.event_id, "zoom": round(target.state.zoom, 4),
                            "center": [round(target.state.cx, 4), round(target.state.cy, 4)],
                            "ncc_planned": None if planned is None else round(planned, 4),
                            "ncc_identity": None if identity is None else round(identity, 4),
                            "texture": round(texture, 2), "masked_bands": len(bands), "verdict": verdict})
        status, reason = _summarize(samples, what="render")
        controls = [s for s in samples if s["kind"] == "control"]
        return {"version": RENDER_PROOF_VERSION, "status": status, "reason": reason,
                "controls_ok": sum(1 for s in controls if s["verdict"] == "identity_ok"),
                "controls": len(controls), "samples": samples,
                "thresholds": {"min_match": MIN_MATCH, "min_margin": MIN_MARGIN, "min_zoom": MIN_PROOF_ZOOM}}
    except ProofUnavailable as error:
        return {"version": RENDER_PROOF_VERSION, "status": "unavailable", "reason": str(error), "samples": []}


@dataclass(frozen=True)
class FinalPair:
    """One verified camera frame of a render and where the final short shows it."""

    render_path: str
    frame: int                 # frame index in the render (== clean paced clip frame)
    final_frame: int           # frame index in the published final
    bands: tuple[tuple[float, float], ...] = ()   # burned text only in the final / render (masked vs source)
    segment: str = "main"      # "main" | "intro"
    event_id: str = ""


def prove_final(*, final_path: str | Path, pairs: Sequence[FinalPair], source_path: str | Path,
                source_size: tuple[int, int], output_size: tuple[int, int],
                base: tuple[int, int, int, int]) -> dict[str, Any]:
    """The FINAL short contains the camera-rendered frames at the mapped times.

    Each final frame must match the camera render (burned captions are in both)
    and look more like the camera render than like the untouched source.
    """
    try:
        if not pairs:
            return {"version": RENDER_PROOF_VERSION, "status": "no_camera_ops", "samples": [],
                    "reason": "no verified camera frame is shown in the final"}
        size = analysis_size(*output_size)
        src_size = analysis_size(source_size[0], source_size[1], ANALYSIS_WIDTH * SOURCE_OVERSAMPLE)
        final_frames = decode_gray_frames(final_path, [p.final_frame for p in pairs], size)
        source_frames = decode_gray_frames(source_path, [p.frame for p in pairs], src_size)
        renders: dict[str, dict[int, Any]] = {}
        for render_path in sorted({p.render_path for p in pairs}):
            renders[render_path] = decode_gray_frames(render_path, [p.frame for p in pairs
                                                                    if p.render_path == render_path], size)
        samples: list[dict[str, Any]] = []
        for pair in pairs:
            rendered_frames = renders.get(pair.render_path, {})
            row: dict[str, Any] = {"segment": pair.segment, "frame": pair.frame, "final_frame": pair.final_frame,
                                   "kind": "camera", "event_id": pair.event_id}
            if (pair.final_frame not in final_frames or pair.frame not in rendered_frames
                    or pair.frame not in source_frames):
                samples.append({**row, "verdict": "inconclusive", "reason": "frame not decoded"})
                continue
            actual = structure(final_frames[pair.final_frame])
            rendered = structure(rendered_frames[pair.frame])
            identity_view = structure(predict_view(source_frames[pair.frame], IDENTITY, base, source_size, size))
            masked = band_mask(size, pair.bands)
            same = masked_ncc(actual, rendered, masked)
            identity = masked_ncc(actual, identity_view, masked)
            texture = float(actual[masked].std()) if masked.any() else 0.0
            if same is None or identity is None or texture < MIN_TEXTURE:
                verdict = "inconclusive"
            elif same >= MIN_MATCH and same - identity >= MIN_MARGIN:
                verdict = "reached"
            elif identity >= MIN_MATCH and identity - same >= MIN_MARGIN:
                verdict = "missing"
            else:
                verdict = "inconclusive"
            samples.append({**row, "ncc_final_vs_render": None if same is None else round(same, 4),
                            "ncc_final_vs_source": round(identity, 4), "texture": round(texture, 2),
                            "verdict": verdict})
        status, reason = _summarize(samples, what="final")
        return {"version": RENDER_PROOF_VERSION, "status": status, "reason": reason, "samples": samples}
    except ProofUnavailable as error:
        return {"version": RENDER_PROOF_VERSION, "status": "unavailable", "reason": str(error), "samples": []}


# ============================================================
# STORY GEOMETRY (every rendered frame, not only the samples)
# ============================================================

def story_geometry_check(resolved: Any, context: Any, style: Any, spans: Sequence[Any] | None = None,
                         step: int = 3) -> dict[str, Any]:
    """Re-verify story geometry on the FINAL resolved path (after energy softening).

    Uses the resolver's own constraint model (``presets.story_constraints``:
    required regions, required subjects as their window median box, minimum
    visible fraction) for every camera op, on every ``step``-th frame."""
    from ai.editor.pro_edit.presets import story_constraints

    rate = float(resolved.fps.fps)
    checked = 0
    violations: list[dict[str, Any]] = []
    span_list = tuple(context.spans if spans is None else spans)
    for op in resolved.ops:
        constraints = story_constraints(context, span_list, op.start_frame / rate, op.end_frame / rate,
                                        resolved.base, style, captions=False)
        if not constraints.required and constraints.min_visible_fraction <= 0:
            checked += len(range(op.start_frame, op.end_frame, max(1, step)))
            continue
        for frame in list(range(op.start_frame, op.end_frame, max(1, step))) + [op.end_frame - 1]:
            state = resolved.path.state_at(frame)
            checked += 1
            if constraints.min_visible_fraction > 0 and 1.0 / state.zoom < constraints.min_visible_fraction - 1e-6:
                violations.append({"frame": frame, "event_id": op.source_event_id, "issue": "min_visible_fraction",
                                   "visible": round(1.0 / state.zoom, 4)})
            for box in constraints.required:
                if not crop_contains(state.zoom, (state.cx, state.cy), box, tol=1e-4):
                    violations.append({"frame": frame, "event_id": op.source_event_id,
                                       "issue": "required_region_cropped", "box": box.to_list()})
    return {"status": "passed" if not violations else "failed", "frames_checked": checked,
            "violations": violations[:20], "violation_count": len(violations)}
