"""Low-cost visual context for the Astra edit director.

This is NOT a vision authority and never emits geometry. It samples a tiny,
deterministic storyboard from the already-selected paced clip so the edit
director can understand what the structured evidence refers to. The compiler
still owns every crop/zoom/pixel.

Budget policy:
- low-detail images only
- bounded frame count
- one extra local decode, no extra model call
- failure => JSON-only director (never blocks the short)
"""
from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ai.editor.pro_edit.vision.cv_runtime import load_opencv
from ai.editor.pro_edit.vision.frames import iterate_frames

VISION_VERSION = 1
SAMPLE_FPS = 2.0
MAX_WIDTH = 384
JPEG_QUALITY = 72
MIN_TARGET_GAP_S = 0.55

_ROLE_PRIORITY = {
    "payoff": 100,
    "reaction": 92,
    "escalation": 82,
    "setup": 72,
    "hook": 66,
    "bridge": 48,
    "neutral": 35,
}


@dataclass(frozen=True)
class DirectorFrame:
    order: int
    requested_t: float
    sampled_t: float
    reason: str
    width: int
    height: int
    sha256: str
    data_url: str
    detail: str = "low"

    def manifest(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "requested_t": round(self.requested_t, 3),
            "sampled_t": round(self.sampled_t, 3),
            "reason": self.reason,
            "width": self.width,
            "height": self.height,
            "sha256": self.sha256,
            "detail": self.detail,
        }

    def api_item(self) -> dict[str, Any]:
        return {"type": "input_image", "image_url": self.data_url, "detail": self.detail}


def _clip(value: float, duration: float) -> float:
    return max(0.0, min(float(duration), float(value)))


def _dedupe(candidates: Iterable[tuple[int, float, str]], *, duration: float,
            limit: int, min_gap: float = MIN_TARGET_GAP_S) -> list[tuple[float, str]]:
    chosen: list[tuple[float, str]] = []
    for _priority, raw_t, reason in sorted(candidates, key=lambda row: (-row[0], row[1], row[2])):
        t = _clip(raw_t, duration)
        if any(abs(t - previous) < min_gap for previous, _reason in chosen):
            continue
        chosen.append((t, reason))
        if len(chosen) >= limit:
            break
    return sorted(chosen, key=lambda row: row[0])


def select_targets(context: Any, max_frames: int) -> list[tuple[float, str]]:
    """Deterministic semantic + coverage targets in PACED_CLIP seconds."""
    duration = max(0.0, float(context.clip.duration_s))
    if duration <= 0 or max_frames <= 0:
        return []
    visible = _clip(float(context.clip.visible_start_s), duration)
    candidates: list[tuple[int, float, str]] = []

    intro = getattr(context, "intro", None)
    if intro is not None:
        candidates.append((98, (float(intro.teaser_start) + float(intro.teaser_end)) / 2.0, "intro_mid"))
        peak = getattr(intro, "peak_focus", None)
        if peak is not None:
            candidates.append((99, float(peak), "intro_peak"))

    for span in getattr(context, "spans", ()) or ():
        role = str(getattr(getattr(span, "role", ""), "value", getattr(span, "role", "")))
        start, end = float(span.start), float(span.end)
        if end <= visible:
            continue
        midpoint = (max(start, visible) + end) / 2.0
        candidates.append((_ROLE_PRIORITY.get(role, 40), midpoint, f"story:{role}"))

    for event in getattr(context, "visual_events", ()) or ():
        start, end = float(event.start), float(event.end)
        if end <= visible:
            continue
        confidence = max(0.0, min(1.0, float(getattr(event, "confidence", 0.0))))
        candidates.append((76 + int(12 * confidence), (max(start, visible) + end) / 2.0,
                           f"visual:{getattr(event, 'type', 'event')}"))

    for cut in getattr(context, "scene_changes", ()) or ():
        t = float(cut)
        if t >= visible:
            candidates.append((58, min(duration, t + 0.08), "after_shot_cut"))

    # Reserve broad coverage so Astra actually sees the whole selected short,
    # not only the loudest/payoff moments.
    main_length = max(0.0, duration - visible)
    coverage = min(3, max_frames)
    for index in range(coverage):
        fraction = (index + 1) / (coverage + 1)
        candidates.append((52 - index, visible + main_length * fraction, f"coverage:{index + 1}/{coverage}"))
    candidates.append((54, min(duration, visible + 0.12), "main_start"))
    candidates.append((50, max(visible, duration - 0.12), "main_end"))

    return _dedupe(candidates, duration=duration, limit=max_frames)


def _encode_jpeg(image: Any) -> tuple[bytes, int, int]:
    cv2 = load_opencv()
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("OpenCV could not encode director frame")
    height, width = image.shape[:2]
    return bytes(encoded), int(width), int(height)


def build_visual_context(path: str, context: Any, *, max_frames: int) -> tuple[tuple[DirectorFrame, ...], dict[str, Any]]:
    """Return bounded low-detail image inputs + a JSON-safe manifest. Never raises."""
    targets = select_targets(context, max_frames)
    if not targets:
        return (), {"version": VISION_VERSION, "status": "empty", "frames": []}
    try:
        nearest: list[tuple[float, Any, float] | None] = [None] * len(targets)
        for sample in iterate_frames(path, context.clip.width, context.clip.height,
                                     sample_fps=SAMPLE_FPS, max_width=MAX_WIDTH):
            for index, (target_t, _reason) in enumerate(targets):
                delta = abs(float(sample.t) - target_t)
                previous = nearest[index]
                if previous is None or delta < previous[0]:
                    nearest[index] = (delta, sample.image.copy(), float(sample.t))
        frames: list[DirectorFrame] = []
        for index, ((target_t, reason), found) in enumerate(zip(targets, nearest), start=1):
            if found is None:
                continue
            _delta, image, sampled_t = found
            jpeg, width, height = _encode_jpeg(image)
            digest = hashlib.sha256(jpeg).hexdigest()
            data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
            frames.append(DirectorFrame(index, target_t, sampled_t, reason, width, height, digest, data_url))
        manifest = {
            "version": VISION_VERSION,
            "status": "ready" if frames else "empty",
            "detail": "low",
            "policy": "bounded adaptive storyboard; read-only visual evidence; engine owns geometry",
            "requested_frames": len(targets),
            "frames": [frame.manifest() for frame in frames],
        }
        return tuple(frames), manifest
    except Exception as error:
        return (), {
            "version": VISION_VERSION,
            "status": "unavailable",
            "detail": "low",
            "frames": [],
            "reason": f"{type(error).__name__}: {str(error)[:220]}",
        }
