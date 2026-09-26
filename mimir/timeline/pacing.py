"""Dead-air pacing (ported CLEAN V3 thresholds) with story protection.

Obvious dead air is compressed; comedic/tension pauses around the payoff are
kept; protected story ranges are never cut; a cut budget stops catastrophic
over-cutting; and a story-integrity guard keeps the bridges between causal
beats (origin -> escalation -> payoff -> reaction) readable.
"""
from __future__ import annotations

from typing import Any, Sequence

from mimir.config import PacingSettings

PROTECTED_PAD = 0.04
BRIDGE_RULES = (  # (max bridge seconds, max single cut, max removal ratio)
    (3.0, 0.45, 0.18),
    (7.0, 0.85, 0.24),
    (float("inf"), 1.20, 0.32),
)
CAUSAL_KINDS = {"causal_origin", "escalation", "payoff", "reaction"}


def _priority(removable: float) -> int:
    for limit, value in ((1.50, 5), (1.00, 4), (0.65, 3), (0.35, 2)):
        if removable >= limit:
            return value
    return 1


def suggest_cuts(words: Sequence[dict[str, Any]], start: float, end: float, payoff: tuple[float, float],
                 cfg: PacingSettings) -> list[dict[str, Any]]:
    """Dead-air suggestions in source time: action ``cut`` (automatic) or ``review`` (kept)."""
    inside = [w for w in words if float(w["end"]) > start and float(w["start"]) < end]
    rows: list[dict[str, Any]] = []
    for left, right in zip(inside, inside[1:]):
        gap_start, gap_end = float(left["end"]), float(right["start"])
        gap = gap_end - gap_start
        if gap < cfg.min_gap:
            continue
        cut_start, cut_end = gap_start + cfg.keep_after_word, gap_end - cfg.keep_before_word
        removable = cut_end - cut_start
        if removable < cfg.min_removable:
            continue
        near_payoff = gap_start < payoff[1] + cfg.payoff_protect_after and gap_end > payoff[0] - cfg.payoff_protect_before
        if near_payoff and gap < 1.35:
            action, reason = "review", "short pause around the payoff may carry timing"
        elif gap >= cfg.auto_cut_gap and removable >= cfg.auto_cut_min_removable:
            action, reason = "cut", "clear dead air between words"
        else:
            action, reason = "review", "short/medium pause kept"
        rows.append({"kind": "dead_air", "action": action, "start": round(cut_start, 3), "end": round(cut_end, 3),
                     "duration": round(removable, 3), "priority": _priority(removable), "reason": reason})
    if inside:
        lead = float(inside[0]["start"]) - start
        if lead >= cfg.leading_trim_threshold:
            trim_end = float(inside[0]["start"]) - cfg.leading_padding
            rows.append({"kind": "leading_dead_air", "action": "cut", "start": round(start, 3),
                         "end": round(trim_end, 3), "duration": round(trim_end - start, 3), "priority": 4,
                         "reason": "leading dead air before the first word"})
        tail = end - float(inside[-1]["end"])
        if tail >= cfg.trailing_trim_threshold:
            trim_start = float(inside[-1]["end"]) + cfg.trailing_padding
            rows.append({"kind": "trailing_dead_air", "action": "cut", "start": round(trim_start, 3),
                         "end": round(end, 3), "duration": round(end - trim_start, 3), "priority": 4,
                         "reason": "trailing dead air after the last word"})
    return sorted(rows, key=lambda r: r["start"])


def minimum_retained(duration: float) -> float:
    if duration >= 30.0:
        return max(18.0, duration * 0.55)
    if duration >= 20.0:
        return max(14.0, duration * 0.60)
    if duration >= 14.0:
        return max(11.0, duration * 0.70)
    return duration * 0.80


def apply_protection(cuts: list[dict[str, Any]], protected: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    for cut in cuts:
        if cut["action"] != "cut":
            continue
        for rng in protected:
            if cut["start"] < rng["end"] + PROTECTED_PAD and cut["end"] > rng["start"] - PROTECTED_PAD:
                cut["action"] = "review"
                cut["blocked_by"] = f"protected: {rng.get('reason', '')}"[:120]
                break
    return cuts


def apply_budget(cuts: list[dict[str, Any]], duration: float, cfg: PacingSettings) -> float:
    budget = min(duration * cfg.max_cut_ratio, duration - minimum_retained(duration), cfg.max_cut_seconds)
    spent = 0.0
    for cut in sorted((c for c in cuts if c["action"] == "cut"), key=lambda c: (-c["priority"], -c["duration"], c["start"])):
        if spent + cut["duration"] <= budget + 1e-9:
            spent += cut["duration"]
        else:
            cut["action"] = "review"
            cut["blocked_by"] = "cut budget (emergency minimum-duration guard)"
    return round(budget, 3)


def apply_story_integrity(cuts: list[dict[str, Any]], protected: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    handles = sorted(({"start": r["start"], "end": r["end"]} for r in protected if r.get("kind") in CAUSAL_KINDS),
                     key=lambda r: r["start"])
    merged: list[dict[str, float]] = []
    for handle in handles:
        if merged and handle["start"] <= merged[-1]["end"] + 0.08:
            merged[-1]["end"] = max(merged[-1]["end"], handle["end"])
        else:
            merged.append(dict(handle))
    reports = []
    for left, right in zip(merged, merged[1:]):
        bridge_start, bridge_end = left["end"], right["start"]
        bridge = bridge_end - bridge_start
        if bridge <= 0.08:
            continue
        max_single, max_ratio = next((s, r) for limit, s, r in BRIDGE_RULES if bridge <= limit)
        inside = [(c, max(0.0, min(c["end"], bridge_end) - max(c["start"], bridge_start)))
                  for c in cuts if c["action"] == "cut"]
        inside = [(c, o) for c, o in inside if o > 0.001]
        kept, blocked = 0.0, 0
        for cut, overlap in sorted(inside, key=lambda row: (-row[0]["priority"], row[1], row[0]["start"])):
            if overlap > max_single + 1e-9 or kept + overlap > bridge * max_ratio + 1e-9:
                cut["action"] = "review"
                cut["blocked_by"] = "story integrity: keeps the causal bridge readable"
                blocked += 1
            else:
                kept += overlap
        reports.append({"start": round(bridge_start, 3), "end": round(bridge_end, 3), "blocked": blocked})
    return reports
