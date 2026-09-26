"""Deterministic story rules: moment/candidate validation and causal-flow enforcement.

Nothing here selects a new highlight or calls a model. It can only make the
selected story complete: expand toward its cause and immediate consequence,
align edges to real speech, and protect the causal handles so pacing cannot
turn the story back into a peak montage.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from mimir.config import StorySettings
from mimir.media.frames import iter_frames
from mimir.media.motion import motion_series
from mimir.story.prompts import EMOTIONS, HOOK_TYPES, MOMENT_TYPES

MAX_SETUP_SECONDS = 20.0
MAX_REACTION_SECONDS = 6.0
MIN_MAIN_DURATION = 18.0
GOOD_METRIC = 7.6
RECOVERY_METRIC = 6.6
MAX_TRANSCRIPT_GAP = 1.90
INTENSE_STRENGTH = 8.0
EXTREME_STRENGTH = 9.0
INTENSE_TYPES = {"physical_payoff", "visual_impact", "destruction", "escalation", "conflict", "failure", "win"}
FLOW_DEFAULTS: dict[str, tuple[float, float]] = {
    "physical_payoff": (6.5, 3.0), "visual_impact": (5.5, 2.6), "destruction": (6.5, 3.2),
    "escalation": (6.0, 3.0), "conflict": (5.0, 3.0), "reaction": (4.2, 2.2), "failure": (5.2, 3.0),
    "win": (4.8, 2.6), "reveal": (4.8, 2.5), "reversal": (4.5, 2.5), "unexpected_answer": (3.8, 2.0),
    "awkward": (3.6, 2.2), "ridiculous_claim": (3.8, 2.0), "punchline": (3.4, 1.8), "tension": (4.5, 2.4),
    "quotable": (2.8, 1.5), "other": (3.8, 2.0),
}
MAX_PROTECTED_RANGE = 6.0
MAX_PROTECTED_RANGES = 14
VISUAL_ORIGIN_LOOKBACK = 20.0
VISUAL_ORIGIN_FPS = 4.0
VISUAL_ORIGIN_WIDTH = 256
VISUAL_ORIGIN_MIN_LEAD = 1.0
VISUAL_ORIGIN_KEEP_MAX = 4.8


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ============================================================ moments

def validate_moments(raw: Sequence[dict[str, Any]], duration: float, limit: int) -> list[dict[str, Any]]:
    rows = []
    for item in raw:
        start = clamp(_f(item.get("start")), 0.0, duration)
        end = clamp(_f(item.get("end"), start), start, duration)
        if end - start < 0.15:
            end = min(duration, start + 0.6)
        if end <= start or end - start > 25.0:
            continue
        kind = str(item.get("type", "other"))
        rows.append({
            "start": round(start, 3), "end": round(end, 3),
            "strength": round(clamp(_f(item.get("strength")), 0.0, 10.0), 2),
            "type": kind if kind in MOMENT_TYPES else "other",
            "source": str(item.get("source", "transcript")) if item.get("source") in ("transcript", "visual", "both")
            else "transcript",
            "visual_event_ids": [str(v) for v in item.get("visual_event_ids", [])][:8],
            "label": " ".join(str(item.get("label", "")).split())[:160],
            "why_compelling": " ".join(str(item.get("why_compelling", "")).split())[:400],
            "context_before_seconds": round(clamp(_f(item.get("context_before_seconds")), 0.0, MAX_SETUP_SECONDS), 2),
            "context_after_seconds": round(clamp(_f(item.get("context_after_seconds")), 0.0, 12.0), 2),
            "preserve_pause_after": bool(item.get("preserve_pause_after", False)),
        })
    rows.sort(key=lambda r: -r["strength"])
    kept: list[dict[str, Any]] = []
    for row in rows:  # drop near-duplicates (heavy overlap with a stronger moment)
        duplicate = any(min(row["end"], k["end"]) - max(row["start"], k["start"])
                        > 0.6 * min(row["end"] - row["start"], k["end"] - k["start"]) for k in kept)
        if not duplicate:
            kept.append(row)
        if len(kept) >= limit:
            break
    kept.sort(key=lambda r: r["start"])
    for index, row in enumerate(kept):
        row["moment_id"] = f"m{index + 1:02d}"
    return kept


def moment_map(moments: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {m["moment_id"]: m for m in moments}


# ============================================================ candidates

def _clean_ranges(ranges: Sequence[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    rows = []
    for item in ranges:
        a = clamp(_f(item.get("start")), start, end)
        b = clamp(_f(item.get("end"), a), a, end)
        if b - a < 0.05:
            continue
        if b - a > MAX_PROTECTED_RANGE:
            center = (a + b) / 2  # protect the timing-sensitive core, never a whole clip
            a, b = max(start, center - MAX_PROTECTED_RANGE / 2), min(end, center + MAX_PROTECTED_RANGE / 2)
        rows.append({"start": round(a, 3), "end": round(b, 3),
                     "reason": " ".join(str(item.get("reason", "")).split())[:200] or "protected beat",
                     "kind": str(item.get("kind", "model"))})
    rows.sort(key=lambda r: (r["start"], r["end"]))
    return rows[:MAX_PROTECTED_RANGES]


def validate_candidate(raw: dict[str, Any], *, duration: float, moments: Sequence[dict[str, Any]],
                       settings: StorySettings) -> tuple[dict[str, Any] | None, str]:
    """Return (clean candidate, "") or (None, reason)."""
    mmap = moment_map(moments)
    start = clamp(_f(raw.get("start")), 0.0, duration)
    end = clamp(_f(raw.get("end"), start), start, duration)
    primary_id = str(raw.get("primary_moment_id", "")).strip()
    primary = mmap.get(primary_id)
    if primary is None:
        # the composer must build around a scouted moment; pick the strongest one it covers
        covered = [mmap[m] for m in raw.get("covered_moment_ids", []) if m in mmap]
        if not covered:
            return None, f"primary_moment_id {primary_id!r} is not a scouted money moment"
        primary = max(covered, key=lambda m: m["strength"])
        primary_id = primary["moment_id"]
    # never exclude the anchor the clip is built around
    start = min(start, max(0.0, primary["start"] - 0.5))
    end = max(end, min(duration, primary["end"] + 0.5))
    payoff_start = clamp(_f(raw.get("payoff_start"), primary["start"]), start, end)
    payoff_end = clamp(_f(raw.get("payoff_end"), primary["end"]), payoff_start, end)
    if payoff_end - payoff_start < 0.15 or not (payoff_start <= primary["end"] and payoff_end >= primary["start"]):
        payoff_start, payoff_end = primary["start"], primary["end"]
    if end - start > settings.max_duration + 0.05:
        return None, f"duration {end - start:.1f}s exceeds {settings.max_duration}s"
    if end - start < settings.min_duration and duration >= settings.min_duration:
        return None, f"duration {end - start:.1f}s below {settings.min_duration}s"
    covered_ids = [m for m in dict.fromkeys([primary_id, *[str(v) for v in raw.get("covered_moment_ids", [])]])
                   if m in mmap and mmap[m]["start"] < end and mmap[m]["end"] > start]
    must_keep = _clean_ranges(raw.get("must_keep_ranges", []), start, end)
    must_keep.append({"start": round(max(start, payoff_start - 0.12), 3), "end": round(min(end, payoff_end + 0.28), 3),
                      "reason": "primary payoff", "kind": "payoff"})
    beats = raw.get("beats") if isinstance(raw.get("beats"), dict) else {}
    hook_type = str(raw.get("hook_type", "curiosity"))
    emotion = str(raw.get("emotion", "surprise"))
    return {
        "start": round(start, 3), "end": round(end, 3), "duration": round(end - start, 3),
        "payoff_start": round(payoff_start, 3), "payoff_end": round(payoff_end, 3),
        "score": round(clamp(_f(raw.get("score")), 0.0, 10.0), 2),
        "title": " ".join(str(raw.get("title", "")).split())[:120] or primary["label"][:120],
        "hook_text": " ".join(str(raw.get("hook_text", "")).split())[:80],
        "hook_type": hook_type if hook_type in HOOK_TYPES else "curiosity",
        "emotion": emotion if emotion in EMOTIONS else "surprise",
        "reason": " ".join(str(raw.get("reason", "")).split())[:600],
        "context": " ".join(str(raw.get("context", "")).split())[:600],
        "primary_moment_id": primary_id,
        "covered_moment_ids": covered_ids,
        "coverage_reason": " ".join(str(raw.get("coverage_reason", "")).split())[:400],
        "length_exception_reason": " ".join(str(raw.get("length_exception_reason", "")).split())[:300],
        "visual_event_ids": [str(v) for v in raw.get("visual_event_ids", [])][:16],
        "must_keep_ranges": must_keep,
        "caption_highlights": [" ".join(str(v).split())[:60] for v in raw.get("caption_highlights", [])][:6],
        "beat_notes": {k: " ".join(str(beats.get(k, "")).split())[:200]
                       for k in ("setup", "escalation", "payoff", "reaction")},
    }, ""


# ============================================================ causal flow

def story_segments(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [{"start": _f(s["start"]), "end": _f(s["end"]), "text": s["text"]}
            for s in transcript.get("segments", []) if _f(s["end"]) > _f(s["start"]) and s.get("text")]
    return sorted(rows, key=lambda r: (r["start"], r["end"]))


def align_to_speech(segments: Sequence[dict[str, Any]], core_start: float, core_end: float, desired_start: float,
                    desired_end: float, duration: float) -> tuple[float, float]:
    """Expand to nearby contiguous speech without crossing large scene-like gaps."""
    if not segments:
        return clamp(desired_start, 0.0, duration), clamp(desired_end, 0.0, duration)
    center = (core_start + core_end) / 2
    nearest = min(range(len(segments)), key=lambda i: 0.0 if segments[i]["start"] <= center <= segments[i]["end"]
                  else min(abs(center - segments[i]["start"]), abs(center - segments[i]["end"])))
    left = right = nearest
    while left > 0:
        prev, current = segments[left - 1], segments[left]
        if current["start"] - prev["end"] > MAX_TRANSCRIPT_GAP or core_start - prev["start"] > MAX_SETUP_SECONDS + 0.5:
            break
        left -= 1
        if segments[left]["start"] <= desired_start:
            break
    while right + 1 < len(segments):
        current, nxt = segments[right], segments[right + 1]
        if nxt["start"] - current["end"] > MAX_TRANSCRIPT_GAP or nxt["end"] - core_end > MAX_REACTION_SECONDS + 0.5:
            break
        right += 1
        if segments[right]["end"] >= desired_end:
            break
    return (clamp(min(desired_start, segments[left]["start"]), 0.0, duration),
            clamp(max(desired_end, segments[right]["end"]), 0.0, duration))


def extend_to_minimum(segments: Sequence[dict[str, Any]], start: float, end: float, core_start: float,
                      core_end: float, minimum: float, duration: float, left_floor: float | None) -> tuple[float, float]:
    """Use real nearby speech (setup first) to reach a readable length; never pads blindly."""
    if end - start >= minimum or not segments:
        return start, end
    covered = [i for i, s in enumerate(segments) if s["end"] >= start - 0.05 and s["start"] <= end + 0.05]
    if not covered:
        return start, end
    left, right = min(covered), max(covered)
    changed = True
    while end - start < minimum and changed:
        changed = False
        if left > 0:
            prev, current = segments[left - 1], segments[left]
            if (current["start"] - prev["end"] <= MAX_TRANSCRIPT_GAP
                    and core_start - prev["start"] <= MAX_SETUP_SECONDS + 0.5
                    and (left_floor is None or prev["start"] >= left_floor - 0.05)):
                left -= 1
                start = min(start, segments[left]["start"])
                changed = True
                if end - start >= minimum:
                    break
        if right + 1 < len(segments):
            current, nxt = segments[right], segments[right + 1]
            if nxt["start"] - current["end"] <= MAX_TRANSCRIPT_GAP and nxt["end"] - core_end <= MAX_REACTION_SECONDS + 0.5:
                right += 1
                end = max(end, segments[right]["end"])
                changed = True
    return clamp(start, 0.0, duration), clamp(end, start, duration)


def detect_visual_origin(video: Path, width: int, height: int, moment_start: float) -> dict[str, float] | None:
    """First sustained (or impact-style) motion burst leading into a payoff: an earlier-boundary HINT only."""
    if moment_start <= VISUAL_ORIGIN_MIN_LEAD:
        return None
    scan_start = max(0.0, moment_start - VISUAL_ORIGIN_LOOKBACK)
    frames = list(iter_frames(video, start=scan_start, duration=moment_start + 0.35 - scan_start,
                              fps=VISUAL_ORIGIN_FPS, width=VISUAL_ORIGIN_WIDTH, src_width=width, src_height=height))
    series = [(s.t, s.energy) for s in motion_series(frames)[1:] if s.t <= moment_start + 0.05]
    if len(series) < 12:
        return None
    values = sorted(v for _, v in series)
    base_values = values[:max(4, int(len(values) * 0.55))]
    baseline = statistics.median(base_values)
    mad = statistics.median(abs(v - baseline) for v in base_values)
    threshold = max(baseline * 1.50, baseline + 2.50 * max(1.0, mad))
    strong = max(baseline * 2.00, threshold * 1.25)
    onset: int | None = None
    for i in range(len(series) - 3):  # sustained struggle / chain reaction
        window = series[i:i + 4]
        if sum(v >= threshold for _, v in window) < 3:
            continue
        t = window[0][0]
        if moment_start - t < VISUAL_ORIGIN_MIN_LEAD:
            continue
        future = [v for tt, v in series if t <= tt <= min(moment_start, t + 5.5)]
        remaining = [v for tt, v in series if t <= tt <= moment_start]
        if future and remaining and max(future) >= strong and \
                sum(v >= threshold for v in remaining) / len(remaining) >= 0.25:
            onset = i
            break
    if onset is None:  # impact style: one huge isolated spike rising from calm
        for i in range(1, len(series)):
            t, v = series[i]
            if moment_start - t < VISUAL_ORIGIN_MIN_LEAD or v < strong:
                continue
            before = [x for _, x in series[max(0, i - 3):i]]
            if before and max(before) < threshold:
                onset = i
                break
    if onset is None:
        return None
    onset_time = max(scan_start, series[onset][0] - 0.35)
    early = [(t, v) for t, v in series if onset_time <= t <= min(moment_start, onset_time + VISUAL_ORIGIN_KEEP_MAX)]
    crest_time, crest = max(early, key=lambda r: r[1]) if early else (onset_time, threshold)
    keep_end = min(max(onset_time + 2.2, crest_time + 0.65), min(moment_start - 0.35, onset_time + VISUAL_ORIGIN_KEEP_MAX))
    return {"start": round(onset_time, 3), "keep_end": round(max(onset_time + 0.6, keep_end), 3),
            "baseline": round(baseline, 3), "threshold": round(threshold, 3), "crest": round(crest, 3),
            "crest_ratio": round(crest / max(1.0, baseline), 3)}


@dataclass
class FlowResult:
    candidate: dict[str, Any]
    report: dict[str, Any]


def enforce_causal_flow(candidate: dict[str, Any], *, moments: Sequence[dict[str, Any]],
                        transcript: dict[str, Any], duration: float, video: Path | None, width: int, height: int,
                        judge_metrics: dict[str, float], settings: StorySettings) -> FlowResult:
    mmap = moment_map(moments)
    primary = mmap[candidate["primary_moment_id"]]
    original_start, original_end = candidate["start"], candidate["end"]
    m_start, m_end = primary["start"], primary["end"]
    kind, strength, source = primary["type"], primary["strength"], primary["source"]
    default_before, default_after = FLOW_DEFAULTS.get(kind, FLOW_DEFAULTS["other"])
    visual_or_causal = kind in INTENSE_TYPES or source in ("visual", "both")
    intense = visual_or_causal and strength >= INTENSE_STRENGTH
    extreme = visual_or_causal and strength >= EXTREME_STRENGTH
    before = min(MAX_SETUP_SECONDS, max(default_before, primary["context_before_seconds"]))
    after = min(MAX_REACTION_SECONDS, max(default_after, primary["context_after_seconds"]))
    target = MIN_MAIN_DURATION
    if extreme:
        before, after, target = min(MAX_SETUP_SECONDS, max(before, 16.0)), min(MAX_REACTION_SECONDS, max(after, 3.8)), 26.0
    elif intense:
        before, after, target = min(MAX_SETUP_SECONDS, max(before, 12.0)), min(MAX_REACTION_SECONDS, max(after, 3.0)), 22.0
    weakest = min(judge_metrics.get("context_completeness", 10.0), judge_metrics.get("story_coverage", 10.0))
    if weakest < RECOVERY_METRIC:
        before, after = min(MAX_SETUP_SECONDS, before + 2.4), min(MAX_REACTION_SECONDS, after + 1.0)
    elif original_end - original_start < MIN_MAIN_DURATION and duration > MIN_MAIN_DURATION + 0.05:
        before, after = min(MAX_SETUP_SECONDS, before + 1.2), min(MAX_REACTION_SECONDS, after + 0.45)
    if primary["preserve_pause_after"]:
        after = min(MAX_REACTION_SECONDS, after + 0.55)
    desired_start, desired_end = max(0.0, m_start - before), min(duration, m_end + after)

    origin = None
    if video is not None and (intense or (visual_or_causal and strength >= 7.0)
                              or (original_end - original_start < MIN_MAIN_DURATION and weakest < RECOVERY_METRIC)):
        found = detect_visual_origin(video, width, height, m_start)
        if found and (intense or (found["crest"] >= 42.0 and found["crest_ratio"] >= 2.7)):
            origin = found
            desired_start = found["start"]
    for moment_id in candidate["covered_moment_ids"]:
        moment = mmap.get(moment_id)
        if moment is None:
            continue
        b0, a0 = FLOW_DEFAULTS.get(moment["type"], FLOW_DEFAULTS["other"])
        desired_start = min(desired_start, max(0.0, moment["start"] - min(MAX_SETUP_SECONDS,
                                                                          max(b0, moment["context_before_seconds"]))))
        desired_end = max(desired_end, min(duration, moment["end"] + min(MAX_REACTION_SECONDS,
                                                                          max(a0, moment["context_after_seconds"]))))
    floor = desired_start if origin else None
    segments = story_segments(transcript)
    flow_start, flow_end = align_to_speech(segments, m_start, m_end, desired_start, desired_end, duration)
    if floor is not None:
        flow_start = max(flow_start, floor)
    if duration > target + 0.05 and (original_end - original_start < target or weakest < RECOVERY_METRIC or intense):
        flow_start, flow_end = extend_to_minimum(segments, flow_start, flow_end, m_start, m_end, target, duration, floor)
    start, end = min(original_start, flow_start), max(original_end, flow_end)
    report = {"applied": False, "primary_type": kind, "primary_strength": strength, "intense": intense,
              "extreme": extreme, "context_before": round(before, 2), "context_after": round(after, 2),
              "visual_origin": origin}
    if end - start > settings.max_duration + 0.02:
        # keep the causal core: trim the far edges symmetrically around the payoff
        excess = end - start - settings.max_duration
        start = min(m_start - 2.0, start + excess * 0.4)
        end = max(m_end + 1.5, end - excess * 0.6)
        if end - start > settings.max_duration + 0.02:
            return FlowResult(candidate, {**report, "reason": "cannot fit causal flow in max duration"})
    result = dict(candidate)
    result["start"], result["end"] = round(start, 3), round(end, 3)
    result["duration"] = round(end - start, 3)
    keep = list(candidate["must_keep_ranges"])
    if origin:
        vs, ve = max(start, origin["start"]), min(end, origin["keep_end"])
        if ve > vs + 0.35:
            keep.append({"start": round(vs, 3), "end": round(ve, 3), "kind": "causal_origin",
                         "reason": "visual causal origin: the silent physical setup that starts the chain"})
    elif intense and m_start - start > 4.0:
        oe = min(m_start - 2.0, start + (3.2 if extreme else 2.6))
        if oe > start + 0.35:
            keep.append({"start": round(start, 3), "end": round(oe, 3), "kind": "causal_origin",
                         "reason": "causal origin: how the high-intensity event started"})
    guard = 5.2 if extreme else (4.5 if intense else 3.8)
    setup_start, setup_end = max(start, m_start - min(before, guard)), min(end, m_start + 0.10)
    if setup_end > setup_start:
        keep.append({"start": round(setup_start, 3), "end": round(setup_end, 3), "kind": "escalation",
                     "reason": "causal escalation bridge into the payoff"})
    reaction_start, reaction_end = max(start, m_end - 0.08), min(end, m_end + min(after, 2.8))
    if reaction_end > reaction_start:
        keep.append({"start": round(reaction_start, 3), "end": round(reaction_end, 3), "kind": "reaction",
                     "reason": "reaction / consequence after the payoff"})
    result["must_keep_ranges"] = sorted(keep, key=lambda r: (r["start"], r["end"]))[:MAX_PROTECTED_RANGES + 4]
    report.update({"applied": True, "original": [original_start, original_end], "final": [result["start"], result["end"]]})
    return FlowResult(result, report)
