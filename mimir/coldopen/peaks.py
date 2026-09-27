"""Multimodal peak candidates inside the story (measured audio + visual evidence + the payoff)."""
from __future__ import annotations

from typing import Any, Sequence

from mimir.config import ColdOpenSettings
from mimir.media.audio import audio_peaks, decode_pcm, envelope

MERGE_DISTANCE = 0.65
MAX_CANDIDATES = 8
WEAK_PEAK = 0.62
STRONG_PEAK = 0.78
# EXTREME PEAK = VERY SHORT COLD OPEN: an unmistakable, compact moment needs no context
EXTREME_PEAK = 0.90          # combined strength when both channels agree
EXTREME_CHANNEL = 0.85       # one decisive channel is enough (visual-only physical event, vocal reaction)
EXTREME_MAX_CORE = 1.35      # the whole event must be compact (core + breathing room stays ~2 s)
EXTREME_PRE, EXTREME_POST = 0.25, 0.40   # breathing room around the core (never padding to a quota)


def story_audio_peaks(source: str, start: float, end: float) -> list[dict[str, Any]]:
    samples = decode_pcm(source, start=start, duration=end - start, rate=16000)
    peaks = audio_peaks(envelope(samples, 16000), max_peaks=10)
    return [{**p, "center": round(p["center"] + start, 3), "start": round(p["start"] + start, 3),
             "end": round(p["end"] + start, 3)} for p in peaks]


def build_candidates(story: dict[str, Any], audio: Sequence[dict[str, Any]], vision: dict[str, Any]
                     ) -> list[dict[str, Any]]:
    start, end = float(story["start"]), float(story["end"])
    payoff_start, payoff_end = float(story["payoff"]["start"]), float(story["payoff"]["end"])
    reactions = [(float(b["start"]), float(b["end"])) for b in story.get("beats", []) if b["role"] == "reaction"]
    rows: list[dict[str, Any]] = []
    for peak in audio:
        if start <= peak["center"] <= end:
            rows.append({"start": peak["start"], "end": peak["end"], "center": peak["center"],
                         "audio_score": peak["score"], "visual_score": 0.0, "signals": ["audio_energy_peak"]})
    for peak in vision.get("visual_peaks", []):
        if start <= peak["t"] <= end:
            rows.append({"start": peak["t"] - 0.35, "end": peak["t"] + 0.45, "center": peak["t"], "audio_score": 0.0,
                         "visual_score": peak["score"], "signals": ["visual_motion_burst"]})
    for region in vision.get("action_regions", []):
        center = (region["t0"] + region["t1"]) / 2
        if start <= center <= end:
            rows.append({"start": region["t0"], "end": region["t1"], "center": center, "audio_score": 0.0,
                         "visual_score": min(1.0, 0.45 + 0.2 * region["relative_intensity"]),
                         "signals": ["action_region"]})
    merged: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: r["center"]):
        near = next((m for m in merged if abs(m["center"] - row["center"]) <= MERGE_DISTANCE), None)
        if near is None:
            merged.append(dict(row))
            continue
        near["start"], near["end"] = min(near["start"], row["start"]), max(near["end"], row["end"])
        near["audio_score"] = max(near["audio_score"], row["audio_score"])
        near["visual_score"] = max(near["visual_score"], row["visual_score"])
        near["signals"] = sorted(set(near["signals"]) | set(row["signals"]))
    for row in merged:
        row["multimodal"] = row["audio_score"] > 0 and row["visual_score"] > 0
        row["in_payoff"] = payoff_start - 0.5 <= row["center"] <= payoff_end + 0.5
        row["in_reaction"] = any(a - 0.3 <= row["center"] <= b for a, b in reactions)
        row["combined_score"] = round(min(1.0, max(row["audio_score"], row["visual_score"])
                                          + (0.15 if row["multimodal"] else 0.0)
                                          + (0.10 if row["in_payoff"] else 0.0)), 3)
    payoff = {"start": payoff_start, "end": payoff_end, "center": (payoff_start + payoff_end) / 2,
              "audio_score": max([r["audio_score"] for r in merged if r["in_payoff"]], default=0.0),
              "visual_score": max([r["visual_score"] for r in merged if r["in_payoff"]], default=0.0),
              "signals": ["story_payoff"], "multimodal": False, "in_payoff": True, "in_reaction": False}
    payoff["combined_score"] = round(max(0.6, min(1.0, 0.25 + max(payoff["audio_score"], payoff["visual_score"]))), 3)
    candidates = [payoff] + sorted(merged, key=lambda r: -r["combined_score"])[:MAX_CANDIDATES - 1]
    for index, row in enumerate(candidates):
        row["peak_id"] = f"p{index:02d}"
        row["start"], row["end"], row["center"] = round(row["start"], 3), round(row["end"], 3), round(row["center"], 3)
    return candidates


def is_extreme(peak: dict[str, Any]) -> bool:
    """An unmistakable compact peak: both channels strongly agree, OR one channel is decisive at the
    story's payoff/reaction (a visual-only physical event or a vocal reaction qualifies on its own)."""
    if float(peak["end"]) - float(peak["start"]) > EXTREME_MAX_CORE + 1e-6:
        return False
    audio, visual = float(peak.get("audio_score", 0.0)), float(peak.get("visual_score", 0.0))
    if peak.get("multimodal") and audio >= 0.72 and visual >= 0.72 \
            and float(peak.get("combined_score", 0.0)) >= EXTREME_PEAK:
        return True
    return max(audio, visual) >= EXTREME_CHANNEL and bool(peak.get("in_payoff") or peak.get("in_reaction"))


def minimum_duration(peak: dict[str, Any], cfg: ColdOpenSettings) -> tuple[float, float, str]:
    """(minimum seconds, lead fraction, policy): stronger compact peaks get shorter cold opens."""
    if is_extreme(peak):
        return cfg.min_understandable, 0.35, "extreme_peak"
    strength = float(peak.get("combined_score", 0.0))
    if strength < WEAK_PEAK:
        return cfg.weak_peak_min, 0.62, "weak_peak_more_context"
    compact = (strength >= STRONG_PEAK and peak.get("multimodal") and peak.get("audio_score", 0) >= 0.72
               and peak.get("visual_score", 0) >= 0.72 and peak["end"] - peak["start"] <= 1.55)
    if compact:
        return cfg.strong_peak_min, 0.42, "compact_multimodal_spike"
    return cfg.moderate_peak_min, 0.55, "contextual_peak"


def snap_to_words(start: float, end: float, words: Sequence[dict[str, Any]], max_duration: float
                  ) -> tuple[float, float]:
    """Never start or end the cold open in the middle of a spoken word."""
    for word in words:
        ws, we = float(word["start"]), float(word["end"])
        if ws < start < we:
            start = ws - 0.05 if end - (ws - 0.05) <= max_duration else we + 0.02
        if ws < end < we:
            end = we + 0.10 if (we + 0.10) - start <= max_duration else ws - 0.02
    return start, end


def compute_window(peak: dict[str, Any], story: dict[str, Any], words: Sequence[dict[str, Any]],
                   cfg: ColdOpenSettings, phrase: tuple[float, float] | None, cuts: Sequence[float]
                   ) -> tuple[float, float, str]:
    s_start, s_end = float(story["start"]), float(story["end"])
    core_start, core_end = float(peak["start"]), float(peak["end"])
    if core_end - core_start > cfg.max_duration - 0.5:  # very long cores: keep the most intense center
        center = float(peak["center"])
        core_start, core_end = center - (cfg.max_duration - 0.5) / 2, center + (cfg.max_duration - 0.5) / 2
    minimum, lead, policy = minimum_duration(peak, cfg)
    start, end = core_start, core_end
    if policy == "extreme_peak":
        start, end = core_start - EXTREME_PRE, core_end + EXTREME_POST
    if phrase is not None:
        start, end = min(start, phrase[0] - 0.08), max(end, phrase[1] + 0.18)
    if end - start < minimum:
        extra = minimum - (end - start)
        start -= extra * lead
        end += extra * (1 - lead)
    if end - start > cfg.max_duration:
        start = max(start, core_end - cfg.max_duration)
        end = start + cfg.max_duration
    start, end = max(s_start, start), min(s_end, end)
    for cut in cuts:  # avoid a one-frame flash of the previous shot at the edges, never inside the action
        if start < cut < min(start + 0.35, core_start) and end - cut >= minimum * 0.8:
            start = cut + 0.02
        if max(end - 0.35, core_end) < cut < end and cut - start >= minimum * 0.8:
            end = cut - 0.02
    start, end = snap_to_words(start, end, words, cfg.max_duration)
    start, end = max(s_start, start), min(s_end, end)
    if end - start < cfg.min_understandable - 1e-6:
        # never too short to understand: grow forward first (the reaction), then back, inside the story
        missing = cfg.min_understandable - (end - start)
        end = min(s_end, end + missing)
        start = max(s_start, end - cfg.min_understandable)
        start, end = snap_to_words(start, end, words, cfg.max_duration)
        start, end = max(s_start, start), min(s_end, end)
        policy += "+understandable_floor"
    if not (start <= float(peak["start"]) + 0.05 and end >= min(float(peak["end"]), start + cfg.max_duration) - 0.05):
        # the peak must be inside the cold open: re-centre on the core itself, bounded (recorded)
        if policy.startswith("extreme_peak"):
            start = max(s_start, float(peak["start"]) - EXTREME_PRE)
            end = min(s_end, max(float(peak["end"]) + EXTREME_POST, start + minimum))
        else:
            start = max(s_start, float(peak["center"]) - cfg.max_duration * lead)
            end = min(s_end, start + max(minimum, min(cfg.max_duration, float(peak["end"]) - start + 0.4)))
        policy += "+recentred_on_core"
    return round(start, 3), round(end, 3), policy
