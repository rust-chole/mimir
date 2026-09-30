"""Explicit intro / main / final clocks for the mandatory MIMIR cold-open.

Traced from ai/editor/intro_renderer.py (V10):

* intro source  = CLEAN paced clip [teaser_start, teaser_end]  (PACED_CLIP)
* intro-local   = [0, teaser_duration]                          (INTRO)
* main source   = captioned preview [main_restart, main_end]    (PACED_CLIP)
* final         = intro-local, then a HARD CUT into main (``transition`` = 0)
  final_t = teaser_duration - transition + (paced_t - main_restart)
* captions: none on the teaser (only the optional hook ASS); main keeps its
  burned captions, trimmed together with video+audio at ``main_restart``.

Every value comes from ``intro_renderer.compute_intro_clock`` (frame-snapped
cut points, protected-range-aware restart) so Pro Edit and the renderer can
never disagree. Pro Edit never selects the intro: the
teaser record is read-only input and its signature is checked afterwards.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ai.editor.pro_edit.errors import EditPlanTimelineError, ProEditError
from ai.editor.pro_edit.timebase import ClipTimelineMap, TimelineDomain, Timestamp

HANDOFF_FRAME_TOLERANCE = 2


class IntroIntegrityError(ProEditError):
    """The selected intro or its handoff changed across Pro Edit."""


def teaser_signature(record: Mapping[str, Any] | None) -> str:
    if not isinstance(record, Mapping):
        return "absent"
    edited = record.get("edited", {}) if isinstance(record.get("edited"), Mapping) else {}
    payload = {
        "clip_index": record.get("clip_index"),
        "teaser_start": edited.get("teaser_start"),
        "teaser_end": edited.get("teaser_end"),
        "duration": edited.get("duration"),
        "peak_id": record.get("peak_id", edited.get("peak_id")),
        "source": record.get("source"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IntroTimeline:
    teaser_start: float
    teaser_end: float
    transition: float
    main_restart: float
    first_caption: float | None
    main_duration: float
    teaser_record_signature: str
    vod_start: float | None = None
    vod_end: float | None = None
    peak_focus: float | None = None

    @property
    def teaser_duration(self) -> float:
        return self.teaser_end - self.teaser_start

    @property
    def main_effective_duration(self) -> float:
        return max(0.0, self.main_duration - self.main_restart)

    @property
    def expected_final_duration(self) -> float:
        return self.teaser_duration + self.main_effective_duration - self.transition

    # --- domain conversions (explicit, never implicit) -------------------------

    def intro_to_source(self, ts: Timestamp) -> Timestamp:
        t = ts.require(TimelineDomain.INTRO)
        if not -1e-6 <= t <= self.teaser_duration + 1e-6:
            raise EditPlanTimelineError(f"intro time {t:.3f}s outside [0, {self.teaser_duration:.3f}]")
        return Timestamp(self.teaser_start + t, TimelineDomain.PACED_CLIP)

    def intro_to_final(self, ts: Timestamp) -> Timestamp:
        return Timestamp(ts.require(TimelineDomain.INTRO), TimelineDomain.FINAL)

    def main_to_final(self, ts: Timestamp) -> Timestamp | None:
        paced = ts.require(TimelineDomain.PACED_CLIP)
        if paced < self.main_restart:
            return None
        return Timestamp(self.teaser_duration - self.transition + (paced - self.main_restart), TimelineDomain.FINAL)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intro_source_paced": [round(self.teaser_start, 4), round(self.teaser_end, 4)],
            "intro_source_vod": None if self.vod_start is None else [round(self.vod_start, 4),
                                                                     round(self.vod_end or 0.0, 4)],
            "intro_local": [0.0, round(self.teaser_duration, 4)],
            "transition_s": round(self.transition, 4),
            "main_restart_paced": round(self.main_restart, 4),
            "first_caption_paced": None if self.first_caption is None else round(self.first_caption, 4),
            "main_duration": round(self.main_duration, 4),
            "expected_final_duration": round(self.expected_final_duration, 4),
            "peak_focus_paced": self.peak_focus,
            "teaser_record_signature": self.teaser_record_signature,
            "order": "intro -> hard cut -> main (intentional repeated peak; not deduplicated)",
        }


def build_intro_timeline(
    teaser_record: Mapping[str, Any],
    *,
    caption_path: str | Path | None,
    clean_duration: float,
    main_duration: float,
    clip_map: ClipTimelineMap | None = None,
    clip_timeline: Mapping[str, Any] | None = None,
    fps: float = 0.0,
) -> IntroTimeline:
    from ai.editor import intro_renderer

    clock = intro_renderer.compute_intro_clock(
        dict(teaser_record), clip_timeline=dict(clip_timeline) if clip_timeline is not None else None,
        caption_path=caption_path, clean_duration=float(clean_duration), main_duration=float(main_duration),
        fps=float(fps))
    start, end = float(clock["teaser_start"]), float(clock["teaser_end"])
    restart, first_caption = float(clock["main_restart"]), clock["first_caption"]
    transition = float(clock["transition"])
    vod_start = vod_end = None
    if clip_map is not None:
        vod_start = clip_map.convert(Timestamp(start, TimelineDomain.PACED_CLIP), TimelineDomain.VOD).seconds
        vod_end = clip_map.convert(Timestamp(end, TimelineDomain.PACED_CLIP), TimelineDomain.VOD).seconds
    focus = teaser_record.get("focus_time")
    try:
        peak_focus = round(float(focus), 4) if focus is not None else None
    except (TypeError, ValueError):
        peak_focus = None
    return IntroTimeline(
        teaser_start=float(start), teaser_end=float(end), transition=float(transition),
        main_restart=float(restart), first_caption=first_caption, main_duration=float(main_duration),
        teaser_record_signature=teaser_signature(teaser_record), vod_start=vod_start, vod_end=vod_end,
        peak_focus=peak_focus,
    )


def verify_handoff(before: IntroTimeline, after: IntroTimeline, *, frame_duration: float) -> None:
    """The intro selection and post-intro handoff must be identical to what MIMIR
    would compute without Pro Edit (no double offset, no shifted restart)."""
    tol = frame_duration * 0.5 + 1e-6
    problems = []
    if before.teaser_record_signature != after.teaser_record_signature:
        problems.append("teaser record changed")
    for name in ("teaser_start", "teaser_end", "transition", "main_restart"):
        if abs(getattr(before, name) - getattr(after, name)) > tol:
            problems.append(f"{name} {getattr(before, name):.4f} -> {getattr(after, name):.4f}")
    if abs(before.main_duration - after.main_duration) > frame_duration + 1e-3:
        problems.append(f"main duration {before.main_duration:.4f} -> {after.main_duration:.4f}")
    if problems:
        raise IntroIntegrityError("intro/handoff changed: " + "; ".join(problems))


def verify_final_duration(timeline: IntroTimeline, final_duration: float, *, frame_duration: float) -> float:
    delta = abs(float(final_duration) - timeline.expected_final_duration)
    if delta > HANDOFF_FRAME_TOLERANCE * frame_duration + 0.05:
        raise IntroIntegrityError(
            f"final duration {final_duration:.3f}s != intro {timeline.teaser_duration:.3f} + main "
            f"{timeline.main_effective_duration:.3f} - transition {timeline.transition:.3f} "
            f"(= {timeline.expected_final_duration:.3f}s)")
    return delta


# ============================================================
# HOOK TEXT SAFE BAND (the neon hook burned over the teaser)
# ============================================================

HOOK_BAND_PADDING = 0.01


@dataclass(frozen=True)
class HookTextBand:
    """Where the intro renderer's hook text sits on the OUTPUT frame, and when.

    Geometry comes from intro_renderer's own layout functions (same text
    cleaning, line split, font size and line positions), so the camera avoids
    exactly the text that will be burned. Times are PACED_CLIP seconds inside
    the selected teaser range.
    """

    start: float
    end: float
    y0: float
    y1: float
    lines: tuple[str, ...]
    font_px: int

    def active(self, start: float, end: float) -> bool:
        return self.start < end and start < self.end

    def to_dict(self) -> dict[str, Any]:
        return {"paced": [round(self.start, 4), round(self.end, 4)], "band_y": [round(self.y0, 4), round(self.y1, 4)],
                "lines": list(self.lines), "font_px": self.font_px}


def hook_text_band(intro_record: Mapping[str, Any] | None, intro: IntroTimeline | None, *, width: int,
                   height: int) -> HookTextBand | None:
    if not isinstance(intro_record, Mapping) or intro is None or width <= 0 or height <= 0:
        return None
    from ai.editor import intro_renderer as ir

    text = ir.clean_hook_text(str(intro_record.get("intro_text", "")))
    if not text:
        return None
    line_1, line_2 = ir.split_balanced_two_lines(text)
    font = ir.estimate_font_size(line_1=line_1, line_2=line_2, width=int(width), height=int(height))
    display = ir.calculate_display_duration(intro=dict(intro_record), teaser_duration=intro.teaser_duration)
    display = min(display, max(0.25, intro.teaser_duration - 0.03))
    if line_2:
        ys = [round(height * ir.LINE_1_Y_RATIO), round(height * ir.LINE_2_Y_RATIO)]
    else:
        ys = [round(height * ir.SINGLE_LINE_Y_RATIO)]
    # \an5 centre anchor; settled scale peaks at BOUNCE_SCALE (the 142% slam
    # lasts one landing and fades in from transparent).
    half = font * max(ir.BOUNCE_SCALE, ir.PULSE_SCALE, 100) / 100.0 / 2.0 + ir.MAIN_OUTLINE
    return HookTextBand(
        start=intro.teaser_start + min(ir.LINE_1_DELAY, ir.LINE_2_DELAY),
        end=intro.teaser_start + display,
        y0=max(0.0, (min(ys) - half) / height - HOOK_BAND_PADDING),
        y1=min(1.0, (max(ys) + half) / height + HOOK_BAND_PADDING),
        lines=tuple(line for line in (line_1, line_2) if line),
        font_px=int(font),
    )
