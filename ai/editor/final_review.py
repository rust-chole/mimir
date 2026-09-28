"""Bounded multimodal review of the final short (after deterministic QC passed).

The reviewer sees what a human editor would glance at: sampled frames of the
RENDERED short side by side with the same moments of the clean SOURCE, the
headline, the transcript and the story beats. It answers a fixed checklist.

Authority is bounded:
* it can never pass anything deterministic QC failed (it only runs after QC);
* a high-confidence finding of a repairable kind triggers at most ONE
  deterministic repair (drop the headline, drop the effect, hold a static
  camera) - the repaired short is re-checked by deterministic QC, never
  re-reviewed in a loop;
* every other finding is published as an explicit warning.
No API key / reviewer disabled -> "not_run" (recorded), never a silent pass.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

FINAL_REVIEW_VERSION = 1
MIN_REPAIR_CONFIDENCE = 0.80
REVIEW_FRAME_WIDTH = 432
MAX_MAIN_FRAMES = 5

FINDING_TYPES = (
    "headline_contradicts_video",
    "headline_spoils_or_misleads",
    "important_subject_cropped",
    "half_cut_face",
    "effect_obscures_content",
    "caption_unreadable",
    "cold_open_not_the_peak",
    "other",
)
REPAIRS = {
    "headline_contradicts_video": "drop_headline",
    "headline_spoils_or_misleads": "drop_headline",
    "effect_obscures_content": "drop_effects",
    "important_subject_cropped": "static_camera",
    "half_cut_face": "static_camera",
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(FINDING_TYPES)},
                    "frames": {"type": "array", "items": {"type": "integer"}},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "explanation": {"type": "string"},
                },
                "required": ["type", "frames", "confidence", "explanation"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["findings", "summary"],
    "additionalProperties": False,
}

REVIEW_INSTRUCTIONS = """
You are the final reviewer of an automatically edited short-form video.

You see numbered frames. For every number there is the FINAL frame (left) and
the same moment in the clean SOURCE footage (right). Frame 0.. are the cold
open (the peak, played first; it carries only the headline, never speech
captions). Later frames are the main story after the hard restart.

Report ONLY concrete problems you can point at in specific frames:
- headline_contradicts_video: the headline claims something the footage/transcript does not show
- headline_spoils_or_misleads: the headline gives away the payoff or misleads about what happens
- important_subject_cropped: the crop cuts away what the viewer needs to see (the action, a reacting person)
- half_cut_face: a face that matters is cut in half by the frame edge
- effect_obscures_content: an overlay/effect hides the action, a face or captions
- caption_unreadable: captions cannot be read against the background
- cold_open_not_the_peak: the cold open does not show the peak moment
- other: anything else that would embarrass a professional editor

Do not report taste preferences. Do not report problems you are not sure of.
Confidence is your probability that a professional editor would agree.
An empty findings list is the correct answer for a clean short.
""".strip()


@dataclass
class ReviewInput:
    candidate: Path
    intro_source: Path
    timeline_doc: Mapping[str, Any]
    headline: str
    transcript: str
    story_beats: Sequence[float] = ()          # paced seconds worth showing (peak, payoff, protected starts)
    effect_windows: Sequence[Sequence[float]] = ()


@dataclass
class ReviewResult:
    status: str                                # reviewed | not_run | failed
    findings: list[dict[str, Any]] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"version": FINAL_REVIEW_VERSION, "status": self.status, "findings": self.findings,
                "repairs": self.repairs, "warnings": self.warnings, "summary": self.summary, "reason": self.reason}


def review_enabled() -> bool:
    return str(os.getenv("MIMIR_FINAL_REVIEW", "1")).strip().casefold() not in {"0", "false", "no", "off"}


def decide(findings: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    """(repairs, warnings): the ONLY authority the reviewer has."""
    repairs: list[str] = []
    warnings: list[str] = []
    for row in findings:
        kind = str(row.get("type", "other"))
        try:
            confidence = float(row.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        text = f"reviewer: {kind} ({confidence:.2f}) {str(row.get('explanation', ''))[:200]}"
        repair = REPAIRS.get(kind)
        if repair and confidence >= MIN_REPAIR_CONFIDENCE:
            if repair not in repairs:
                repairs.append(repair)
        else:
            warnings.append(text)
    return repairs, warnings


# ============================================================
# EVIDENCE (frames)
# ============================================================

def _frame_times(data: ReviewInput) -> list[tuple[float, float]]:
    """(final seconds, source paced seconds) pairs to show."""
    intro = data.timeline_doc.get("intro", {}) or {}
    restart = float((data.timeline_doc.get("restart", {}) or {}).get("main_restart_paced", 0.0) or 0.0)
    intro_duration = float(intro.get("duration", 0.0) or 0.0)
    paced_start = float((intro.get("paced") or [0.0])[0])
    rows = [(intro_duration * f, paced_start + intro_duration * f) for f in (0.25, 0.7)]
    beats = sorted({round(float(b), 2) for b in data.story_beats if float(b) >= restart})
    expected = float(data.timeline_doc.get("expected_final_duration", 0.0) or 0.0)
    main_span = max(0.0, expected - intro_duration)
    if len(beats) < MAX_MAIN_FRAMES and main_span > 0:
        beats += [restart + main_span * f for f in (0.1, 0.35, 0.6, 0.85)]
    for effect in data.effect_windows:
        middle = (float(effect[0]) + float(effect[1])) / 2.0
        beats.append(restart + middle - intro_duration)
    for paced in sorted(set(round(b, 2) for b in beats))[:MAX_MAIN_FRAMES + len(data.effect_windows)]:
        rows.append((intro_duration + paced - restart, paced))
    return rows


def _grab(path: Path, seconds: float, target: Path) -> bool:
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, seconds):.3f}", "-i",
               str(path), "-frames:v", "1", "-vf", f"scale={REVIEW_FRAME_WIDTH}:-2", "-q:v", "4", str(target)]
    completed = subprocess.run(command, capture_output=True, check=False, timeout=120)
    return completed.returncode == 0 and target.is_file() and target.stat().st_size > 0


def _side_by_side(final: Path, source: Path, target: Path) -> bool:
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(final), "-i", str(source),
               "-filter_complex", "[0:v][1:v]hstack=inputs=2", "-q:v", "4", str(target)]
    completed = subprocess.run(command, capture_output=True, check=False, timeout=120)
    return completed.returncode == 0 and target.is_file()


def build_frames(data: ReviewInput, workdir: Path) -> list[tuple[int, float, Path]]:
    frames = []
    for index, (final_t, paced_t) in enumerate(_frame_times(data)):
        a, b, pair = workdir / f"f{index}.jpg", workdir / f"s{index}.jpg", workdir / f"pair{index}.jpg"
        if _grab(data.candidate, final_t, a) and _grab(data.intro_source, paced_t, b) and _side_by_side(a, b, pair):
            frames.append((index, final_t, pair))
    return frames


# ============================================================
# MODEL CALL
# ============================================================

def openai_reviewer(model: str, effort: str) -> Callable[[str, list[tuple[int, float, Path]]], dict[str, Any]]:
    def call(prompt: str, frames: list[tuple[int, float, Path]]) -> dict[str, Any]:
        from ai.openai_client import client

        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for index, seconds, path in frames:
            content.append({"type": "input_text", "text": f"FRAME {index} (final {seconds:.2f}s): final | source"})
            content.append({"type": "input_image",
                            "image_url": "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()})
        response = client.responses.create(
            model=model, reasoning={"effort": effort}, instructions=REVIEW_INSTRUCTIONS,
            input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_schema", "name": "final_review", "strict": True, "schema": REVIEW_SCHEMA}},
        )
        return json.loads(str(response.output_text or "{}"))

    return call


def default_reviewer() -> Callable[[str, list[tuple[int, float, Path]]], dict[str, Any]] | None:
    if not review_enabled() or not str(os.getenv("OPENAI_API_KEY", "")).strip():
        return None
    from ai import model_config

    model = str(os.getenv("MIMIR_FINAL_REVIEW_MODEL", "")).strip() or model_config.EDITOR_MODEL
    effort = str(os.getenv("MIMIR_FINAL_REVIEW_REASONING", "medium")).strip() or "medium"
    return openai_reviewer(model, effort)


def review_final(data: ReviewInput,
                 reviewer: Callable[[str, list[tuple[int, float, Path]]], dict[str, Any]] | None = None,
                 ) -> ReviewResult:
    reviewer = reviewer if reviewer is not None else default_reviewer()
    if reviewer is None:
        reason = "disabled (MIMIR_FINAL_REVIEW=0)" if not review_enabled() else "no OPENAI_API_KEY"
        return ReviewResult("not_run", reason=reason, warnings=[f"final AI review not run: {reason}"])
    with tempfile.TemporaryDirectory(prefix="mimir_review_") as tmp:
        frames = build_frames(data, Path(tmp))
        if not frames:
            return ReviewResult("failed", reason="no review frames could be extracted",
                                warnings=["final AI review failed: no frames"])
        prompt = (f"HEADLINE: {data.headline or '(none - the cold open carries no text)'}\n"
                  f"TRANSCRIPT (story):\n{data.transcript[:6000]}")
        try:
            answer = reviewer(prompt, frames)
        except Exception as error:  # reviewer is advisory: failure is recorded, never blocks, never passes silently
            return ReviewResult("failed", reason=f"{type(error).__name__}: {str(error)[:300]}",
                                warnings=[f"final AI review failed: {type(error).__name__}"])
    findings = [row for row in (answer.get("findings") or []) if isinstance(row, Mapping)]
    repairs, warnings = decide(findings)
    return ReviewResult("reviewed", findings=[dict(r) for r in findings], repairs=repairs, warnings=warnings,
                        summary=str(answer.get("summary", ""))[:500])
