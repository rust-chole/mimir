"""Synthetic MIMIR-shaped artifacts for Pro Edit tests (no network, no media)."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.editor.pro_edit.context import ContextInputs, EditContext, build_edit_context  # noqa: E402
from ai.editor.pro_edit.media import MediaInfo  # noqa: E402
from ai.editor.pro_edit.subjects import NullSubjectProvider, SubjectTrackProvider  # noqa: E402
from ai.editor.pro_edit.timebase import FrameRate  # noqa: E402

# OpenCV + numpy are OPTIONAL at runtime (requirements.txt): without them Pro Edit's
# pixel evidence (activity, background/HUD, tracking) degrades through the fallback
# ladder. Tests that measure those pixels need them and are reported as skipped.
HAVE_OPENCV = all(importlib.util.find_spec(name) is not None for name in ("cv2", "numpy"))
needs_opencv = unittest.skipUnless(
    HAVE_OPENCV, "OpenCV/numpy not installed (optional Pro Edit evidence: pip install -r requirements.txt)")

# Raw clip: VOD 80..110 (30 s). Two 1 s cuts -> paced 28 s.
#   escalation anchor VOD 88-90  -> raw 8-10    -> paced 7-9
#   payoff raw 14-17                            -> paced 13-16
#   reaction anchor VOD 97.2-99 -> raw 17.2-19  -> paced 16.2-18
#   protected raw 13.8-18.5                     -> paced 12.8-17.5
BASE_TIMELINE: dict[str, Any] = {
    "version": 3,
    "source": {"video_stem": "demo_vod", "video_path": "C:/Videos/demo vod.mp4"},
    "timelines": [
        {
            "clip_index": 1,
            "title": "demo",
            "source": {"absolute_start": 80.0, "absolute_end": 110.0, "duration": 30.0},
            "edited": {"estimated_duration": 28.0},
            "cut_ranges": [{"start": 5.0, "end": 6.0}, {"start": 20.0, "end": 21.0}],
            "payoff": {"type": "payoff", "source_start": 14.0, "source_end": 17.0,
                       "absolute_start": 94.0, "absolute_end": 97.0},
            "protected_ranges": [{"start": 13.8, "end": 18.5, "duration": 4.7, "reason": "money"}],
            "hook": {"type": "curiosity", "text": "wait for it"},
            "editorial": {
                "anchor_moments": [
                    {"start": 88.0, "end": 90.0, "type": "escalation", "strength": 7.0},
                    {"start": 97.2, "end": 99.0, "type": "reaction", "strength": 8.0},
                ]
            },
            "events": [{"type": "sfx_suggestion", "source_time": 14.2, "sfx": "boom"}],
        }
    ],
}

WORDS = [
    ("So", 0.40, 0.60, "A"), ("today", 0.62, 0.95, "A"), ("we", 1.0, 1.1, "A"),
    ("try", 1.12, 1.4, "A"), ("this", 1.42, 1.7, "A"),
    ("no", 7.2, 7.4, "B"), ("way", 7.45, 7.8, "B"),
    ("watch", 12.9, 13.2, "A"), ("THIS", 13.25, 13.6, "A"), ("OH", 16.3, 16.7, "B"),
    ("my", 16.75, 16.9, "B"), ("god", 16.95, 17.4, "B"),
]


def speaker_profile(words: list[tuple[str, float, float, str]] | None = None) -> dict[str, Any]:
    rows = words if words is not None else WORDS
    return {
        "version": 25,
        "status": "ok",
        "timing_basis": "exact_final_48k_audio",
        "display_labels": {"A": "KAI", "B": "TYLA"},
        "words": [
            {"word": w, "edited_start": s, "edited_end": e, "speaker_raw": spk, "speaker_label": ""}
            for (w, s, e, spk) in rows
        ],
    }


VIDEO_REPORT: dict[str, Any] = {
    "layout": {"orientation": "landscape", "content_type": "streamer"},
    "visual_events": [
        {"start": 16.3, "end": 17.6, "type": "reaction", "description": "big laugh", "confidence": 0.9},
        {"start": 13.1, "end": 13.9, "type": "impact", "description": "cup falls", "confidence": 0.8},
    ],
    "editing_support": {"reaction_times": [16.4], "scene_change_times": [], "visual_payoff_times": [13.2]},
    "intro_visual_support": {"visual_peak_regions": [{"start": 13.0, "end": 14.2}]},
}


def media(duration: float = 28.0, width: int = 1920, height: int = 1080, fps: str = "30000/1001",
          has_audio: bool = True) -> MediaInfo:
    rate = FrameRate.parse(fps)
    return MediaInfo(
        path="C:/fake/edited.mp4", width=width, height=height, fps=rate, avg_fps=rate,
        duration_s=duration, video_duration_s=duration, frame_count=int(round(duration * rate.fps)),
        has_audio=has_audio, video_start_s=0.0, audio_start_s=0.0 if has_audio else None,
        pix_fmt="yuv420p", is_cfr=True,
    )


class Workspace:
    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="mimir_pro_edit_test_")
        self.root = Path(self._tmp.name)

    def write_json(self, name: str, data: Any) -> Path:
        path = self.root / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def cleanup(self) -> None:
        self._tmp.cleanup()


def make_context(
    workspace: Workspace,
    *,
    timeline: dict[str, Any] | None = None,
    words: list[tuple[str, float, float, str]] | None = None,
    report: dict[str, Any] | None = VIDEO_REPORT,
    visible_start: float = 0.0,
    provider: SubjectTrackProvider | None = None,
    media_info: MediaInfo | None = None,
) -> EditContext:
    timeline_data = copy.deepcopy(timeline if timeline is not None else BASE_TIMELINE)
    profile_path = workspace.write_json("speakers.json", speaker_profile(words))
    report_path = workspace.write_json("report.json", report) if report is not None else None
    return build_edit_context(ContextInputs(
        timeline_data=timeline_data,
        clip_index=1,
        media=media_info or media(),
        speaker_profile_path=profile_path,
        video_report_path=report_path,
        visible_start_s=visible_start,
        subject_provider=provider or NullSubjectProvider(),
    ))


def event(event_id: str, start: float, end: float, **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "event_id": event_id, "start": start, "end": end, "role": "payoff", "camera": "preserve",
        "motion": "punch_in", "target": {"type": "center_safe", "id": None}, "intensity": 0.7,
        "caption_style": "default", "emphasis_word_ids": [], "sfx": "none", "support_visual": "none",
        "confidence": 0.9, "reason_code": "payoff_hit",
    }
    data.update(overrides)
    return data


def plan(*events: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"schema_version": 2, "style_pack": "pro_stream_v1", "timeline_domain": "paced_clip", "intro": None,
                            "events": list(events)}
    data.update(overrides)
    return data
