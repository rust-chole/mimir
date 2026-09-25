"""Global EDITORIAL ENERGY coordinator (cross-channel load, deterministic).

Local budgets already exist per channel (caption emphasis/impact ratios,
primitive budgets, camera merge gaps, SFX min gap). This pass prevents
SIMULTANEOUS overload across channels:

    CAMERA   resolved camera ops (strong punch / snap vs mild push / focus)
    CAPTION  page decoration (impact outline, outline swell, decorative plates/badges)
    SUPPORT_VISUAL / SFX   planner recommendations (Pro Edit never executes them)
    TRANSITION  the existing intro -> main crossfade (fixed, never degraded)

Energy values are one central, configurable table (ENERGY_TABLE; override via
MIMIR_PRO_EDIT_ENERGY_TABLE). The window (default 800 ms) is about one short
caption page (the presentation's SHORT_PAGE_S is 0.75 s): editorial events
closer than one page read as simultaneous. Budget default 4.

Resolution runs in fixed passes, so no circular dependency exists:

    1. captions are built (placement + legibility)          [caption pass]
    2. the camera is resolved against the captions' band     [camera pass]
    3. this pass reads both, degrades lowest priority first:
         optional SFX recommendation  -> dropped
         optional support visual rec. -> dropped
         caption decoration           -> decorative plate/badge off -> impact outline off -> calm
         camera emphasis (not PAYOFF) -> strong move becomes a slow push / speaker focus
    4. captions are re-rendered with calmer effects (geometry never grows, so the
       camera's caption band stays valid); the camera is re-resolved ONCE with
       the SAME caption band. Nothing feeds back into step 1.

Never altered: story, caption truth, word timing, intro identity, legibility
treatments (readability outranks camera emphasis), PAYOFF camera ops (payoff
source action), the intro transition.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from ai.editor.pro_edit.caption_primitives import (
    LEGIBILITY_PRIMITIVES,
    REQUIRED_PRIMITIVES,
    PageEffects,
    PrimitiveId,
)
from ai.editor.pro_edit.schema import STRONG_MOTIONS, CaptionStyle, MotionPreset, StoryRole

ENERGY_VERSION = 1
DEFAULT_WINDOW_MS = 800
DEFAULT_BUDGET = 4
ENERGY_TABLE: Mapping[str, int] = {
    "caption_default": 0,
    "caption_emphasis": 1,          # outline swell on an EMPHASIS page
    "caption_impact": 2,            # impact outline accent
    "caption_decoration": 1,        # decorative plate / badge (legibility plates are free)
    "camera_strong": 2,             # punch_in / punch_in_fast / snap_reframe
    "camera_mild": 1,               # slow push / speaker focus / pull-out
    "sfx": 1,
    "support_visual": 3,            # meme / b-roll recommendation
    "transition": 2,                # intro -> main crossfade
}
# Degrade order: lower value first. Transition is fixed.
CHANNEL_PRIORITY = {"sfx": 0, "support_visual": 1, "caption": 2, "camera": 3, "transition": 99}
CAMERA_SOFTER = {MotionPreset.PUNCH_IN_FAST: MotionPreset.SLOW_PUSH, MotionPreset.PUNCH_IN: MotionPreset.SLOW_PUSH,
                 MotionPreset.SNAP_REFRAME: MotionPreset.SPEAKER_FOCUS}
CAPTION_LADDER = ("full", "no_decoration", "no_impact", "calm")


def parse_energy_table(raw: str | None) -> tuple[dict[str, int], str]:
    """Strict JSON override of ENERGY_TABLE (known keys, integers 0..5). (table, problem)."""
    table = dict(ENERGY_TABLE)
    if not raw or not str(raw).strip():
        return table, ""
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or set(data) - set(ENERGY_TABLE):
            raise ValueError("unknown keys")
        for key, value in data.items():
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 5:
                raise ValueError(f"{key} must be an integer 0..5")
            table[key] = value
    except ValueError as error:
        return dict(ENERGY_TABLE), f"MIMIR_PRO_EDIT_ENERGY_TABLE invalid ({error}); defaults used"
    return table, ""


@dataclass
class EnergyEvent:
    event_id: str
    channel: str                  # caption | camera | sfx | support_visual | transition
    start: float
    end: float
    energy: int
    protected: bool = False
    state: str = "full"
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def degradable(self) -> bool:
        return not self.protected and self.energy > 0 and self.channel != "transition"


@dataclass
class EnergyDecision:
    event_id: str
    channel: str
    before: int
    after: int
    action: str
    window: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {"event_id": self.event_id, "channel": self.channel, "energy": [self.before, self.after],
                "action": self.action, "window": [round(self.window[0], 3), round(self.window[1], 3)]}


@dataclass
class EnergyResult:
    table: Mapping[str, int]
    window_s: float
    budget: int
    max_load_before: int = 0
    max_load_after: int = 0
    decisions: list[EnergyDecision] = field(default_factory=list)
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    caption_overrides: dict[int, PageEffects] = field(default_factory=dict)
    caption_notes: dict[int, str] = field(default_factory=dict)
    camera_motions: dict[str, MotionPreset] = field(default_factory=dict)
    dropped_recommendations: set[tuple[str, str]] = field(default_factory=set)    # (channel, event_id)
    applied: dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.decisions)

    def to_dict(self) -> dict[str, Any]:
        return {"version": ENERGY_VERSION, "table": dict(self.table), "window_ms": int(round(self.window_s * 1000)),
                "budget": self.budget, "max_load_before": self.max_load_before,
                "max_load_after": self.max_load_after, "decisions": [d.to_dict() for d in self.decisions],
                "unresolved": list(self.unresolved), "applied": dict(self.applied)}


# ============================================================
# EVENT EXTRACTION
# ============================================================

def caption_state_effects(effects: PageEffects, state: str) -> PageEffects:
    """Effects of a page at a caption ladder state (readability kept at every state)."""
    if state == "full":
        return effects
    legibility_plate = effects.plate_reason == "legibility"
    keep = set(effects.primitives)
    keep.discard(PrimitiveId.STATIC_NUMBER_BADGE)
    keep.discard(PrimitiveId.STATIC_UNDERLINE_ACCENT)
    keep.discard(PrimitiveId.PAGE_ALPHA_IN)
    if not legibility_plate:
        keep.discard(PrimitiveId.STATIC_ACCENT_BACKPLATE)
    if state in ("no_impact", "calm"):
        keep.discard(PrimitiveId.SOFT_IMPACT_OUTLINE)
    if state == "calm":
        keep.discard(PrimitiveId.ACTIVE_OUTLINE_EASE)
    keep |= REQUIRED_PRIMITIVES & set(effects.primitives)
    keep |= LEGIBILITY_PRIMITIVES & set(effects.primitives)
    return PageEffects(effects.page_id, frozenset(keep), 0, legibility_plate, frozenset(), frozenset(),
                       frozenset(), effects.contrast_word_ids, "legibility" if legibility_plate else "")


def caption_energy(effects: PageEffects, style: CaptionStyle, table: Mapping[str, int]) -> int:
    energy = 0
    if PrimitiveId.SOFT_IMPACT_OUTLINE in effects.primitives:
        energy += table["caption_impact"]
    elif PrimitiveId.ACTIVE_OUTLINE_EASE in effects.primitives and style is not CaptionStyle.DEFAULT:
        energy += table["caption_emphasis"]
    else:
        energy += table["caption_default"]
    decorative_plate = effects.plate_lines and effects.plate_reason != "legibility"
    if decorative_plate or effects.plate_word_ids or effects.badge_word_ids:
        energy += table["caption_decoration"]
    return energy


def collect_events(*, presentation: Any = None, resolved: Any = None, plan: Any = None,
                   recommendations: Iterable[Mapping[str, Any]] = (), transition: tuple[float, float] | None = None,
                   table: Mapping[str, int] = ENERGY_TABLE) -> list[EnergyEvent]:
    events: list[EnergyEvent] = []
    if presentation is not None and presentation.settings is not None:
        for page in presentation.pages:
            effects = presentation.settings.effects_for(page)
            energy = caption_energy(effects, page.style, table)
            if energy > 0:
                events.append(EnergyEvent(f"page:{page.page_id}", "caption", page.start,
                                          min(page.end, page.start + 0.3), energy,
                                          detail={"page_id": page.page_id, "style": page.style.value}))
    roles = {e.event_id: e.role for e in getattr(plan, "events", ())} if plan is not None else {}
    motions = {e.event_id: e.motion for e in getattr(plan, "events", ())} if plan is not None else {}
    if resolved is not None:
        fps = resolved.fps.fps
        for op in resolved.ops:
            motion = motions.get(op.source_event_id)
            strong = motion in STRONG_MOTIONS if motion is not None else op.params.preset in {m.value for m in
                                                                                              STRONG_MOTIONS}
            start = op.start_frame / fps
            end = (op.start_frame + max(1, op.attack_frames)) / fps
            events.append(EnergyEvent(
                f"camera:{op.source_event_id}", "camera", start, end,
                table["camera_strong"] if strong else table["camera_mild"],
                protected=roles.get(op.source_event_id) is StoryRole.PAYOFF,
                detail={"event_id": op.source_event_id, "motion": getattr(motion, "value", op.params.preset)}))
    for row in recommendations:
        channel = str(row.get("channel", ""))
        if channel not in ("sfx", "support_visual"):
            continue
        fps = resolved.fps.fps if resolved is not None else 30.0
        t = float(row.get("frame", 0)) / fps
        events.append(EnergyEvent(f"{channel}:{row.get('event_id')}", channel, t, t + 0.05, table[channel],
                                  detail={"event_id": row.get("event_id"), "value": row.get("value")}))
    if transition is not None and transition[1] > transition[0]:
        events.append(EnergyEvent("transition:intro", "transition", transition[0], transition[1],
                                  table["transition"], protected=True))
    return sorted(events, key=lambda e: (e.start, e.channel, e.event_id))


# ============================================================
# RESOLUTION
# ============================================================

def _windows(events: Sequence[EnergyEvent], window: float) -> list[tuple[int, float, float, list[EnergyEvent]]]:
    rows = []
    for anchor in sorted({e.start for e in events}):
        lo, hi = anchor - window / 2.0, anchor + window / 2.0
        inside = [e for e in events if e.energy > 0 and e.start < hi and e.end > lo]
        rows.append((sum(e.energy for e in inside), lo, hi, inside))
    return rows


def _degrade(event: EnergyEvent, presentation: Any, table: Mapping[str, int]) -> str:
    """One step down the event's ladder; returns the action (energy updated in place)."""
    if event.channel in ("sfx", "support_visual"):
        event.energy = 0
        event.state = "dropped"
        return "optional recommendation dropped"
    if event.channel == "camera":
        event.energy = table["camera_mild"] if event.energy > table["camera_mild"] else event.energy
        event.state = "softened"
        event.protected = True             # one step only: a camera move is never removed
        return "strong camera move -> gentle move"
    page = next(p for p in presentation.pages if p.page_id == event.detail["page_id"])
    base = presentation.settings.effects_for(page)
    index = CAPTION_LADDER.index(event.state)
    while index + 1 < len(CAPTION_LADDER):
        index += 1
        effects = caption_state_effects(base, CAPTION_LADDER[index])
        energy = caption_energy(effects, page.style, table)
        if energy < event.energy or index == len(CAPTION_LADDER) - 1:
            event.state = CAPTION_LADDER[index]
            event.energy = energy
            event.detail["effects"] = effects
            return f"caption decoration -> {CAPTION_LADDER[index]}"
    event.protected = True
    return "no further caption degradation"


def coordinate(events: list[EnergyEvent], *, presentation: Any = None, window_ms: int = DEFAULT_WINDOW_MS,
               budget: int = DEFAULT_BUDGET, table: Mapping[str, int] = ENERGY_TABLE) -> EnergyResult:
    """Deterministic degradation until no window exceeds the budget (or nothing is degradable)."""
    window = window_ms / 1000.0
    result = EnergyResult(dict(table), window, budget)
    rows = _windows(events, window)
    result.max_load_before = max((r[0] for r in rows), default=0)
    settled: set[tuple[float, float]] = set()
    for _step in range(10_000):
        rows = _windows(events, window)
        over = [r for r in rows if r[0] > budget and (r[1], r[2]) not in settled]
        if not over:
            break
        load, lo, hi, inside = max(over, key=lambda r: (r[0], -r[1]))
        candidates = [e for e in inside if e.degradable and (e.channel != "caption" or presentation is not None)]
        if not candidates:
            result.unresolved.append({"window": [round(lo, 3), round(hi, 3)], "load": load,
                                      "reason": "only protected events (transition / payoff camera / readability)"})
            settled.add((lo, hi))
            continue
        candidates.sort(key=lambda e: (CHANNEL_PRIORITY[e.channel], -e.start, e.event_id))
        chosen = candidates[0]
        before = chosen.energy
        action = _degrade(chosen, presentation, table)
        result.decisions.append(EnergyDecision(chosen.event_id, chosen.channel, before, chosen.energy, action,
                                               (lo, hi)))
    result.max_load_after = max((r[0] for r in _windows(events, window)), default=0)
    for event in events:
        if event.state == "full":
            continue
        if event.channel == "caption":
            page_id = int(event.detail["page_id"])
            result.caption_overrides[page_id] = event.detail["effects"]
            result.caption_notes[page_id] = event.state
        elif event.channel == "camera":
            motion = event.detail.get("motion")
            try:
                softer = CAMERA_SOFTER.get(MotionPreset(motion))
            except ValueError:
                softer = None
            if softer is not None:
                result.camera_motions[str(event.detail["event_id"])] = softer
        elif event.state == "dropped":
            result.dropped_recommendations.add((event.channel, str(event.detail.get("event_id"))))
    return result


def soften_plan(plan: Any, motions: Mapping[str, MotionPreset]) -> Any:
    """Plan copy with the energy-softened camera motions (other fields untouched)."""
    if not motions:
        return plan
    events = tuple(replace(e, motion=motions[e.event_id]) if e.event_id in motions else e for e in plan.events)
    return replace(plan, events=events)


def mark_recommendations(rows: Iterable[Mapping[str, Any]], dropped: set[tuple[str, str]]
                         ) -> tuple[dict[str, Any], ...]:
    marked = []
    for row in rows:
        row = dict(row)
        if (str(row.get("channel")), str(row.get("event_id"))) in dropped:
            row["executed"] = False
            row["energy_dropped"] = True
        marked.append(row)
    return tuple(marked)
