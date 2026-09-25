from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from ai.video_brain.config import (
    VIDEO_BRAIN_MODEL,
    VIDEO_BRAIN_OUTPUT_DIR,
)
from ai.video_brain.video_client import (
    GeminiQuotaError,
    analyze_local_video,
)


# ============================================================
# MIMIR VISUAL FACT OBSERVER V6
# ============================================================
#
# SUPPORT-ONLY visual intelligence layer.
#
# This module DOES NOT:
# - transcribe
# - select clips
# - cut video
# - render captions
# - render intros
# - render memes
# - modify the Shorts pipeline
#
# It only:
# - sends the clean source/edited video to Video Brain client
# - asks for visual understanding
# - normalizes the model output
# - saves a stable JSON report
#
# Compatible public API:
#   analyze_video(...)
#   try_analyze_video(...)
#   get_output_path(...)
# ============================================================


VIDEO_ANALYZER_VERSION = 6
REPORT_SCHEMA_VERSION = 4

# Once a Gemini quota error is seen in this process, all later visual passes
# go straight to Terra so the same run does not waste time retrying Gemini.
_GEMINI_QUOTA_EXHAUSTED = False


def _visual_backend_mode() -> str:
    """Return auto/gemini/terra. Default is OpenAI-frame hard bypass; Gemini is opt-in."""
    value = str(os.getenv("MIMIR_VISUAL_BACKEND", "terra")).strip().lower()
    aliases = {
        "openai": "terra",
        "terra_frames": "terra",
        "gemini_only": "gemini",
    }
    value = aliases.get(value, value)
    if value not in {"auto", "gemini", "terra"}:
        return "terra"
    return value


def _looks_like_gemini_quota_error(error: BaseException) -> bool:
    """Catch quota exhaustion even if an older client wrapped the 429."""
    if isinstance(error, GeminiQuotaError):
        return True
    text = str(error).casefold()
    needles = (
        "resource_exhausted",
        "quota exceeded",
        "quota_exceeded",
        "free_tier_requests",
        "generaterequestsperdayperprojectpermodel-freetier",
        "gemini api error 429",
        "http 429",
        "code\": 429",
        "code': 429",
    )
    return any(item in text for item in needles)


def _run_terra_visual_fallback(path: Path, reason: str) -> dict[str, Any]:
    from ai.video_brain.terra_visual_fallback import analyze_local_video_with_terra

    print()
    print("🧠 Visual backend: TERRA frame fallback")
    print(f"   Sebep: {reason[:220]}")
    return analyze_local_video_with_terra(path, reason=reason)


# ============================================================
# PROMPT
# ============================================================

VIDEO_ANALYSIS_PROMPT = r"""
You are MIMIR Visual Observer.

Your ONLY job is to watch the ENTIRE uploaded video and create a factual,
timestamped inventory of visually observable events.

You are NOT the editor.
You do NOT decide what is viral, funny, important, compelling, boring, or
worth clipping.
You do NOT recommend clips.
You do NOT score moments for retention.

TERRA will make every editorial / compellingness decision later.

This distinction is critical:

    GEMINI / VISUAL OBSERVER = WHAT VISIBLY HAPPENED
    TERRA                   = WHETHER IT IS A GREAT CLIP MOMENT

Analyze the full video from beginning to end. Do not focus only on faces or
spoken moments. Visual-only events matter even when the transcript says
nothing about them.

You MUST explicitly report sudden physical or state-changing events such as:
- a door/window/object breaking
- destruction or visible damage
- impact / collision
- fall / knockdown
- crash
- explosion
- object being thrown, dropped, opened, closed, kicked, hit, shattered
- sudden entrance or exit
- major gameplay action
- death / elimination / score-changing event in gameplay
- abrupt scene or camera-state change
- visible reveal
- strong visible reaction
- sudden movement that changes the situation
- object appearing/disappearing in a meaningful way

Example: if a door breaks on screen, that event MUST appear in visual_events
with a tight timestamp even if nobody says "the door broke".

Do not omit a concrete event merely because dialogue is happening at the
same time.

Return ONLY one valid JSON object, no markdown.

Use this exact structure:

{
  "summary": "Neutral factual visual summary.",
  "layout": {
    "orientation": "portrait|landscape|square|unknown",
    "content_type": "streamer|gameplay|streamer_gameplay|other",
    "notes": "Persistent layout facts only."
  },
  "visual_events": [
    {
      "start": 0.0,
      "end": 0.0,
      "type": "reaction|visual_payoff|scene_change|gameplay_event|object_break|destruction|impact|crash|fall|explosion|physical_action|entrance_exit|reveal|popup|sudden_change|other",
      "description": "Concrete visible fact; say exactly what changes on screen.",
      "confidence": 0.0
    }
  ],
  "editing_support": {
    "reaction_times": [],
    "visual_payoff_times": [],
    "scene_change_times": [],
    "strong_visual_moments": [],
    "notes": "No editorial recommendations. Terra decides significance."
  },
  "intro_visual_support": {
    "visual_peak_regions": [
      {
        "start": 0.0,
        "end": 0.0,
        "signals": ["strong_visible_reaction|sudden_motion|impact|reveal|chaos|multi_person_reaction|expression_change|state_change"],
        "description": "Exactly what visibly changes in this short high-intensity region.",
        "observable_intensity": 0.0,
        "confidence": 0.0
      }
    ],
    "notes": "Observable visual intensity only; this is not a viral/editorial ranking."
  },
  "quality": {
    "visual_clarity": 0.0,
    "analysis_confidence": 0.0,
    "limitations": []
  },
  "warnings": []
}

Rules:
1. All timestamps are seconds from the START of this uploaded video.
2. Inspect the WHOLE video, including quiet/no-dialogue stretches.
   Treat visible state changes as first-class facts even when there is no speech.
   Scan before, during, and after dialogue; do not stop analysis at transcript-like moments.
3. Never identify or guess the real identity of a person.
4. Never invent an event. Lower confidence when uncertain.
5. Prefer several precise events over a vague summary.
6. For instantaneous events, use a tight range around the visible action.
7. For multi-step events, include the full visible action window.
8. If destruction/impact/breakage occurs, name the object/action plainly.
9. Do not call something "viral", "best", "strong", "boring" or "clip-worthy".
10. Keep visual_events non-duplicative but DO NOT collapse distinct actions.
11. confidence and quality values are 0.0 to 1.0.
12. Use empty lists rather than inventing uncertain facts.
13. Do not perform OCR/text-region analysis; source-text detection is not part of this system.
14. MIMIR-added captions/hooks/memes are not source-video events.
15. For intro_visual_support, scan the WHOLE video and report up to 8 short regions
    with the strongest OBSERVABLE visual intensity: big visible reactions, sudden
    motion, impact, reveal, chaos, synchronized/multi-person reaction, strong
    expression change, or abrupt state change. This is factual visual support,
    NOT a recommendation that the region is viral or should be the intro.
16. Prefer tight peak regions (roughly 0.3-2.5 seconds) and include the actual
    visible peak, not a long surrounding scene.
17. Output JSON only.
""".strip()


# ============================================================
# BASIC HELPERS
# ============================================================

def _safe_filename(value: str) -> str:

    text = str(value).strip()

    for char in '<>:"/\\|?*':
        text = text.replace(char, "_")

    text = " ".join(text.split()).strip(" ._")

    return text or "video"


def resolve_video_input(
    video_path: str | Path,
) -> Path:

    path = Path(
        str(video_path).strip().strip('"')
    ).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Video bulunamadı:\n{path}"
        )

    if not path.is_file():
        raise RuntimeError(
            f"Video yolu dosya değil:\n{path}"
        )

    if path.stat().st_size <= 0:
        raise RuntimeError(
            f"Video dosyası boş:\n{path}"
        )

    return path


def _source_fingerprint(
    path: Path,
) -> dict[str, Any]:

    stat = path.stat()

    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def get_output_path(
    video_path: str | Path,
) -> Path:

    path = resolve_video_input(video_path)

    output_dir = Path(
        VIDEO_BRAIN_OUTPUT_DIR
    ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        output_dir
        / f"{_safe_filename(path.stem)}_video_report_v6.json"
    )


# ============================================================
# MODEL JSON PARSING
# ============================================================

def _extract_json_object(
    text: str,
) -> dict[str, Any]:

    raw = str(text or "").strip()

    if not raw:
        raise RuntimeError(
            "Video Brain model çıktısı boş."
        )

    # Tolerate accidental markdown fences.
    raw = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        raw,
        flags=re.IGNORECASE,
    )

    raw = re.sub(
        r"\s*```\s*$",
        "",
        raw,
    ).strip()

    try:
        parsed = json.loads(raw)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    # Last-resort extraction if the model adds a tiny prefix/suffix.
    start = raw.find("{")
    end = raw.rfind("}")

    if start < 0 or end <= start:
        raise RuntimeError(
            "Video Brain çıktısında JSON object bulunamadı:\n"
            + raw[:3000]
        )

    candidate = raw[start:end + 1]

    try:
        parsed = json.loads(candidate)

    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Video Brain geçersiz JSON döndürdü:\n"
            + candidate[:5000]
        ) from error

    if not isinstance(parsed, dict):
        raise RuntimeError(
            "Video Brain JSON root object değil."
        )

    return parsed


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def _text(value: Any) -> str:

    if value is None:
        return ""

    return str(value).strip()



def _score(value: Any) -> float:

    try:
        number = float(value)

    except (TypeError, ValueError):
        return 0.0

    return round(
        max(0.0, min(1.0, number)),
        3,
    )


def _nonnegative_time(
    value: Any,
) -> float | None:

    try:
        number = float(value)

    except (TypeError, ValueError):
        return None

    if number < 0:
        return None

    return round(number, 3)


def _time_list(
    value: Any,
    *,
    limit: int = 100,
) -> list[float]:

    if not isinstance(value, list):
        return []

    result: list[float] = []

    for item in value:

        timestamp = _nonnegative_time(item)

        if timestamp is not None:
            result.append(timestamp)

    return sorted(set(result))[:limit]


def _string_list(
    value: Any,
    *,
    limit: int = 50,
) -> list[str]:

    if not isinstance(value, list):
        return []

    result: list[str] = []

    for item in value:

        text = _text(item)

        if text and text not in result:
            result.append(text)

    return result[:limit]


# ============================================================
# REPORT NORMALIZATION
# ============================================================

def normalize_report(
    raw: dict[str, Any],
) -> dict[str, Any]:

    if not isinstance(raw, dict):
        raw = {}

    # --------------------------------------------------------
    # LAYOUT
    # --------------------------------------------------------

    layout_raw = raw.get("layout", {})

    if not isinstance(layout_raw, dict):
        layout_raw = {}

    orientation = _text(
        layout_raw.get("orientation", "unknown")
    ).lower()

    allowed_orientations = {
        "portrait",
        "landscape",
        "square",
        "unknown",
    }

    if orientation not in allowed_orientations:
        orientation = "unknown"

    content_type = _text(
        layout_raw.get("content_type", "other")
    ).lower()

    allowed_content_types = {
        "streamer",
        "gameplay",
        "streamer_gameplay",
        "other",
    }

    if content_type not in allowed_content_types:
        content_type = "other"

    # --------------------------------------------------------
    # VISUAL EVENTS
    # --------------------------------------------------------

    events_raw = raw.get(
        "visual_events",
        [],
    )

    visual_events: list[
        dict[str, Any]
    ] = []

    allowed_event_types = {
        "reaction",
        "visual_payoff",
        "scene_change",
        "gameplay_event",
        "object_break",
        "destruction",
        "impact",
        "crash",
        "fall",
        "explosion",
        "physical_action",
        "entrance_exit",
        "reveal",
        "popup",
        "sudden_change",
        "other",
    }

    if isinstance(events_raw, list):

        for item in events_raw[:120]:

            if not isinstance(item, dict):
                continue

            start = _nonnegative_time(
                item.get("start", 0)
            )

            end = _nonnegative_time(
                item.get(
                    "end",
                    item.get("start", 0),
                )
            )

            if start is None:
                continue

            if end is None:
                end = start

            if end < start:
                end = start

            event_type = _text(
                item.get("type", "other")
            ).lower()

            if event_type not in allowed_event_types:
                event_type = "other"

            description = _text(
                item.get("description", "")
            )

            visual_events.append(
                {
                    "start": start,
                    "end": end,
                    "type": event_type,
                    "description": description,
                    "confidence": _score(
                        item.get("confidence", 0)
                    ),
                }
            )

    visual_events.sort(
        key=lambda event: (
            event["start"],
            event["end"],
            event["type"],
        )
    )

    # Compatibility timing hints are derived from FACTUAL event types only.
    # No "interestingness" judgement happens here.
    derived_reaction_times = sorted(
        {
            float(event["start"])
            for event in visual_events
            if event["type"] == "reaction"
        }
    )

    derived_payoff_times = sorted(
        {
            float(event["start"])
            for event in visual_events
            if event["type"] == "visual_payoff"
        }
    )

    derived_scene_change_times = sorted(
        {
            float(event["start"])
            for event in visual_events
            if event["type"] == "scene_change"
        }
    )

    # --------------------------------------------------------
    # EDITING SUPPORT
    # --------------------------------------------------------

    editing_raw = raw.get(
        "editing_support",
        {},
    )

    if not isinstance(editing_raw, dict):
        editing_raw = {}

    strong_raw = editing_raw.get(
        "strong_visual_moments",
        [],
    )

    strong_visual_moments: list[
        dict[str, Any]
    ] = []

    if isinstance(strong_raw, list):

        for item in strong_raw[:50]:

            if not isinstance(item, dict):
                continue

            timestamp = _nonnegative_time(
                item.get("time")
            )

            reason = _text(
                item.get("reason", "")
            )

            if timestamp is None or not reason:
                continue

            strong_visual_moments.append(
                {
                    "time": timestamp,
                    "reason": reason,
                }
            )

    strong_visual_moments.sort(
        key=lambda item: item["time"]
    )

    # --------------------------------------------------------
    # INTRO VISUAL SUPPORT
    # --------------------------------------------------------

    intro_support_raw = raw.get("intro_visual_support", {})
    if not isinstance(intro_support_raw, dict):
        intro_support_raw = {}

    visual_peak_regions: list[dict[str, Any]] = []
    peak_raw = intro_support_raw.get("visual_peak_regions", [])
    allowed_peak_signals = {
        "strong_visible_reaction",
        "sudden_motion",
        "impact",
        "reveal",
        "chaos",
        "multi_person_reaction",
        "expression_change",
        "state_change",
    }

    if isinstance(peak_raw, list):
        for item in peak_raw[:24]:
            if not isinstance(item, dict):
                continue
            start = _nonnegative_time(item.get("start", 0))
            end = _nonnegative_time(item.get("end", item.get("start", 0)))
            if start is None:
                continue
            if end is None or end < start:
                end = start
            raw_signals = item.get("signals", [])
            signals: list[str] = []
            if isinstance(raw_signals, list):
                for value in raw_signals:
                    signal = _text(value).lower()
                    if signal in allowed_peak_signals and signal not in signals:
                        signals.append(signal)
            confidence = _score(item.get("confidence", 0))
            intensity = _score(item.get("observable_intensity", 0))
            description = _text(item.get("description", ""))
            if confidence <= 0 and intensity <= 0 and not description:
                continue
            visual_peak_regions.append({
                "start": start,
                "end": end,
                "signals": signals,
                "description": description,
                "observable_intensity": intensity,
                "confidence": confidence,
            })

    visual_peak_regions.sort(
        key=lambda item: (
            -float(item["observable_intensity"]),
            -float(item["confidence"]),
            float(item["start"]),
        )
    )
    visual_peak_regions = visual_peak_regions[:8]

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    quality_raw = raw.get("quality", {})

    if not isinstance(quality_raw, dict):
        quality_raw = {}

    # --------------------------------------------------------
    # FINAL STABLE REPORT
    # --------------------------------------------------------

    return {
        "summary": _text(
            raw.get("summary", "")
        ),
        "layout": {
            "orientation": orientation,
            "content_type": content_type,
            "notes": _text(
                layout_raw.get("notes", "")
            ),
        },
        "visual_events": visual_events,
        "editing_support": {
            "reaction_times": sorted(
                set(
                    _time_list(
                        editing_raw.get(
                            "reaction_times",
                            [],
                        )
                    )
                    + derived_reaction_times
                )
            ),
            "visual_payoff_times": sorted(
                set(
                    _time_list(
                        editing_raw.get(
                            "visual_payoff_times",
                            [],
                        )
                    )
                    + derived_payoff_times
                )
            ),
            "scene_change_times": sorted(
                set(
                    _time_list(
                        editing_raw.get(
                            "scene_change_times",
                            [],
                        )
                    )
                    + derived_scene_change_times
                )
            ),
            # Deliberately empty: Visual Observer does not rank or recommend.
            "strong_visual_moments": [],
            "notes": (
                "Facts only. Terra decides editorial significance."
            ),
        },
        "intro_visual_support": {
            "visual_peak_regions": visual_peak_regions,
            "notes": (
                "Observable visual intensity only. Terra decides whether any peak "
                "belongs in the intro."
            ),
        },
        "quality": {
            "visual_clarity": _score(
                quality_raw.get(
                    "visual_clarity",
                    0,
                )
            ),
            "analysis_confidence": _score(
                quality_raw.get(
                    "analysis_confidence",
                    0,
                )
            ),
            "limitations": _string_list(
                quality_raw.get(
                    "limitations",
                    [],
                )
            ),
        },
        "warnings": _string_list(
            raw.get("warnings", [])
        ),
    }


# ============================================================
# CACHE
# ============================================================

def _cache_is_compatible(
    package: dict[str, Any],
    fingerprint: dict[str, Any],
) -> bool:

    if not isinstance(package, dict):
        return False

    if (
        package.get("analyzer_version")
        != VIDEO_ANALYZER_VERSION
    ):
        return False

    if (
        package.get("report_schema_version")
        != REPORT_SCHEMA_VERSION
    ):
        return False

    backend = _text(package.get("visual_backend", package.get("backend", "gemini"))).lower()
    package_model = _text(package.get("model"))

    if backend == "terra_frame_fallback":
        try:
            from ai import model_config
            if package_model != _text(model_config.EDITOR_MODEL):
                return False
        except Exception:
            return False
    elif package_model != _text(VIDEO_BRAIN_MODEL):
        return False

    source = package.get("source", {})

    if not isinstance(source, dict):
        return False

    try:

        return (
            source.get("path")
            == fingerprint.get("path")
            and int(source.get("size", -1))
            == int(fingerprint.get("size", -2))
            and int(source.get("mtime_ns", -1))
            == int(fingerprint.get("mtime_ns", -2))
        )

    except (TypeError, ValueError):
        return False


# ============================================================
# PUBLIC API
# ============================================================

def analyze_video(
    video_path: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:

    path = resolve_video_input(
        video_path
    )

    output_path = get_output_path(
        path
    )

    fingerprint = _source_fingerprint(
        path
    )

    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    if not force and output_path.exists():

        try:

            cached = json.loads(
                output_path.read_text(
                    encoding="utf-8"
                )
            )

            if _cache_is_compatible(
                cached,
                fingerprint,
            ):

                print()
                print(
                    "⏭️ Video Brain cache kullanılıyor."
                )
                print(
                    f"📂 {output_path}"
                )

                return cached

        except Exception:
            # Broken cache should never break Video Brain.
            pass

    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    print()
    print("=" * 76)
    print("👁️ MIMIR VISUAL FACT OBSERVER V6")
    print("=" * 76)
    print("🧩 Role: FACTS ONLY — no clip judgement")
    print(f"🎞️ Video: {path.name}")
    print(f"🤖 Model: {VIDEO_BRAIN_MODEL}")

    global _GEMINI_QUOTA_EXHAUSTED

    backend_mode = _visual_backend_mode()

    # Hard bypass: this mode never touches Gemini. Useful when the free-tier
    # quota is already known to be exhausted.
    if backend_mode == "terra":
        raw_response = _run_terra_visual_fallback(
            path,
            "MIMIR_VISUAL_BACKEND=terra (hard Gemini bypass)",
        )

    elif _GEMINI_QUOTA_EXHAUSTED and backend_mode != "gemini":
        raw_response = _run_terra_visual_fallback(
            path,
            "Gemini quota already exhausted earlier in this run",
        )

    else:
        try:
            raw_response = analyze_local_video(
                video_path=path,
                prompt=VIDEO_ANALYSIS_PROMPT,
                temperature=0.10,
                max_output_tokens=8192,
            )
        except Exception as error:
            # Older/newer Gemini clients can wrap HTTP 429 differently.
            # In auto mode we detect the quota by both type and message, then
            # immediately continue with Terra instead of returning None.
            if backend_mode != "gemini" and _looks_like_gemini_quota_error(error):
                _GEMINI_QUOTA_EXHAUSTED = True
                print()
                print("⚠️ Gemini kotası dolu. MIMIR durmuyor; görsel analiz Terra'ya devredildi.")
                raw_response = _run_terra_visual_fallback(path, str(error))
            else:
                raise

    if not isinstance(raw_response, dict):
        raise RuntimeError(
            "video_client beklenmeyen response tipi döndürdü."
        )

    output_text = _text(
        raw_response.get("output_text")
    )

    if not output_text:
        raise RuntimeError(
            "video_client output_text döndürmedi."
        )

    raw_report = _extract_json_object(
        output_text
    )

    report = normalize_report(
        raw_report
    )

    remote_file = raw_response.get(
        "remote_file",
        {},
    )

    if not isinstance(remote_file, dict):
        remote_file = {}

    usage_metadata = raw_response.get(
        "usage_metadata",
        {},
    )

    if not isinstance(usage_metadata, dict):
        usage_metadata = {}

    backend = _text(raw_response.get("backend", "gemini")) or "gemini"
    sampling = raw_response.get("sampling", {})
    if not isinstance(sampling, dict):
        sampling = {}

    package = {
        "analyzer_version": VIDEO_ANALYZER_VERSION,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "mode": "visual_fact_observer_no_editorial_judgement",
        "visual_backend": backend,
        "fallback_reason": _text(raw_response.get("fallback_reason", "")),
        "sampling": sampling,
        "source": fingerprint,
        "model": _text(
            raw_response.get(
                "model",
                VIDEO_BRAIN_MODEL,
            )
        ),
        "model_version": _text(
            raw_response.get(
                "model_version",
                "",
            )
        ),
        "response_id": _text(
            raw_response.get(
                "response_id",
                "",
            )
        ),
        "usage_metadata": usage_metadata,
        "remote_file": {
            "name": _text(
                remote_file.get("name")
            ),
            "uri": _text(
                remote_file.get("uri")
            ),
            "state": _text(
                remote_file.get("state")
            ),
            "mime_type": _text(
                remote_file.get("mime_type")
            ),
            "display_name": _text(
                remote_file.get("display_name")
            ),
        },
        "report": report,
    }

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_output = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    temp_output.write_text(
        json.dumps(
            package,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temp_output.replace(
        output_path
    )

    print()
    print("✅ Video Brain report hazır.")
    if backend == "terra_frame_fallback":
        print(
            "🧠 Visual backend: Terra frame fallback "
            f"({int(sampling.get('frame_count', 0) or 0)} kare)"
        )
    else:
        print(f"👁️ Visual backend: Gemini / {VIDEO_BRAIN_MODEL}")
    print(
        "📝 Summary: "
        + (
            report["summary"][:240]
            or "<boş>"
        )
    )
    print(
        "🎬 Visual events: "
        f"{len(report['visual_events'])}"
    )
    print(
        "🎯 Analysis confidence: "
        f"{report['quality']['analysis_confidence']:.2f}"
    )
    print(f"📂 {output_path}")
    print("=" * 76)

    return package


def try_analyze_video(
    video_path: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any] | None:
    """
    Fail-open wrapper for future Shorts pipeline integration.

    Video Brain is support-only. Gemini quota exhaustion is handled inside
    analyze_video by switching to Terra visual frame fallback. Other failures
    remain fail-open for selected-clip support callers.
    """

    try:

        return analyze_video(
            video_path,
            force=force,
        )

    except Exception as error:

        print()
        print(
            "⚠️ Video Brain destek analizi başarısız."
        )
        print(
            "   Ana MIMIR pipeline bu hata nedeniyle durmamalı."
        )
        print(
            f"   Sebep: {error}"
        )

        return None


# ============================================================
# CLI
# ============================================================

def main() -> int:

    args = sys.argv[1:]

    if not args:

        print(
            "Kullanım:\n"
            'python -m ai.video_brain.video_analyzer '
            '"C:\\path\\video.mp4" [--force]'
        )

        return 2

    force = "--force" in args

    positional = [
        arg
        for arg in args
        if arg != "--force"
    ]

    if not positional:

        print(
            "Video yolu eksik."
        )

        return 2

    video_path = positional[0]

    try:

        analyze_video(
            video_path,
            force=force,
        )

        return 0

    except Exception as error:

        print()
        print(
            "❌ VIDEO ANALYZER HATASI:"
        )
        print(
            error
        )

        return 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
