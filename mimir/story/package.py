"""StoryPackage: the selected story as explicit beats + protected ranges (source time)."""
from __future__ import annotations

from typing import Any, Sequence

MIN_BEAT = {"setup": 1.0, "escalation": 0.5, "payoff": 0.2, "reaction": 0.6}
BEAT_ORDER = ("setup", "escalation", "payoff", "reaction")


def build_beats(candidate: dict[str, Any], primary: dict[str, Any]) -> list[dict[str, Any]]:
    start, end = float(candidate["start"]), float(candidate["end"])
    payoff_start, payoff_end = float(candidate["payoff_start"]), float(candidate["payoff_end"])
    after = max(1.5, min(6.0, float(primary.get("context_after_seconds", 2.0)) or 2.0))
    reaction_ranges = [r for r in candidate["must_keep_ranges"] if r.get("kind") == "reaction"]
    reaction_end = min(end, max([payoff_end + after] + [float(r["end"]) for r in reaction_ranges]))
    lead = payoff_start - start
    escalation_len = min(5.2, max(1.0, 0.35 * lead)) if lead > 1.5 else max(0.0, lead * 0.5)
    escalation_start = max(start, payoff_start - escalation_len)
    notes = candidate.get("beat_notes", {})
    beats = [
        {"role": "setup", "start": start, "end": escalation_start},
        {"role": "escalation", "start": escalation_start, "end": payoff_start},
        {"role": "payoff", "start": payoff_start, "end": payoff_end},
        {"role": "reaction", "start": payoff_end, "end": reaction_end},
    ]
    if reaction_end < end - 0.05:
        beats.append({"role": "aftermath", "start": reaction_end, "end": end})
    for beat in beats:
        beat["start"], beat["end"] = round(beat["start"], 3), round(beat["end"], 3)
        beat["duration"] = round(beat["end"] - beat["start"], 3)
        beat["note"] = notes.get(beat["role"], "")
    return beats


def completeness(beats: Sequence[dict[str, Any]]) -> list[str]:
    by_role = {b["role"]: b for b in beats}
    problems = []
    for role in BEAT_ORDER:
        beat = by_role.get(role)
        if beat is None or beat["duration"] + 1e-6 < MIN_BEAT[role]:
            problems.append(f"{role} beat missing or shorter than {MIN_BEAT[role]}s")
    return problems


def build_package(candidate: dict[str, Any], moments: Sequence[dict[str, Any]], *, judge: dict[str, Any],
                  flow: dict[str, Any], boundary: dict[str, Any], rank: int) -> dict[str, Any]:
    mmap = {m["moment_id"]: m for m in moments}
    primary = mmap[candidate["primary_moment_id"]]
    beats = build_beats(candidate, primary)
    protected = sorted(({**r, "kind": r.get("kind", "model")} for r in candidate["must_keep_ranges"]),
                       key=lambda r: (r["start"], r["end"]))
    return {
        "story_id": f"story_{int(candidate['start'] * 1000):09d}_{int(candidate['end'] * 1000):09d}",
        "start": candidate["start"],
        "end": candidate["end"],
        "duration": round(candidate["end"] - candidate["start"], 3),
        "title": candidate["title"],
        "hook_text": candidate["hook_text"],
        "hook_type": candidate["hook_type"],
        "emotion": candidate["emotion"],
        "reason": candidate["reason"],
        "context": candidate["context"],
        "primary_moment": primary,
        "covered_moments": [mmap[m] for m in candidate["covered_moment_ids"] if m in mmap],
        "payoff": {"start": candidate["payoff_start"], "end": candidate["payoff_end"]},
        "beats": beats,
        "protected_ranges": protected,
        "caption_highlights": candidate["caption_highlights"],
        "visual_event_ids": candidate["visual_event_ids"],
        "selection": {"rank": rank, "judge": judge},
        "causal_flow": flow,
        "boundary_polish": boundary,
        "completeness_problems": completeness(beats),
    }
