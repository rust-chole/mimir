"""Story protection map: MIMIR's narrative structure as read-only spans.

The story system owns setup/escalation/payoff/reaction/bridge, must-keep
ranges and order. Pro Edit V1 only presents. Each ``StorySpan`` says what a
presentation layer may do inside it AND which image geometry must stay
visible, because preserving a timestamp is not enough: a face-only crop can
keep the payoff frames while cropping the payoff action out.

Protection semantics (combinable flags; LOCKED = no flag):

* SOURCE_REQUIRED        original footage must stay on screen (no replacement)
* CAMERA_EDIT_ALLOWED    camera framing/motion may change
* SUPPORT_VISUAL_ALLOWED meme / b-roll may be recommended
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import IntFlag
from typing import Iterable, Mapping, Sequence

from ai.editor.pro_edit.schema import StoryRole
from ai.editor.pro_edit.timebase import TimelineDomain, TimeRange


class Protection(IntFlag):
    LOCKED = 0
    SOURCE_REQUIRED = 1
    CAMERA_EDIT_ALLOWED = 2
    SUPPORT_VISUAL_ALLOWED = 4


ROLE_PROTECTION: Mapping[StoryRole, Protection] = {
    StoryRole.HOOK: Protection.SOURCE_REQUIRED | Protection.CAMERA_EDIT_ALLOWED,
    StoryRole.SETUP: Protection.SOURCE_REQUIRED | Protection.CAMERA_EDIT_ALLOWED,
    StoryRole.ESCALATION: Protection.SOURCE_REQUIRED | Protection.CAMERA_EDIT_ALLOWED,
    StoryRole.PAYOFF: Protection.SOURCE_REQUIRED | Protection.CAMERA_EDIT_ALLOWED,
    StoryRole.REACTION: Protection.SOURCE_REQUIRED | Protection.CAMERA_EDIT_ALLOWED,
    StoryRole.BRIDGE: Protection.CAMERA_EDIT_ALLOWED | Protection.SUPPORT_VISUAL_ALLOWED,
    StoryRole.NEUTRAL: Protection.CAMERA_EDIT_ALLOWED | Protection.SUPPORT_VISUAL_ALLOWED,
}
# When the location of an on-screen action is unknown, a crop inside an
# action span must keep at least this fraction of each frame dimension.
ACTION_MIN_VISIBLE_FRACTION = 0.85
ACTION_ROLES = frozenset({StoryRole.PAYOFF, StoryRole.ESCALATION, StoryRole.HOOK})
INTRO_SPAN_ID = "intro_01"


@dataclass(frozen=True)
class RequiredRegion:
    """Normalized box (source frame) that must stay inside the crop."""

    x0: float
    y0: float
    x1: float
    y1: float
    source: str

    def to_list(self) -> list[float]:
        return [round(self.x0, 4), round(self.y0, 4), round(self.x1, 4), round(self.y1, 4)]


@dataclass(frozen=True)
class StorySpan:
    span_id: str
    role: StoryRole
    start: float
    end: float
    domain: TimelineDomain
    protection: Protection
    visual_importance: str = "normal"
    required_subject_ids: tuple[str, ...] = ()
    required_regions: tuple[RequiredRegion, ...] = ()
    min_visible_fraction: float = 0.0
    must_keep: bool = False
    evidence: tuple[str, ...] = ()

    @property
    def camera_allowed(self) -> bool:
        return bool(self.protection & Protection.CAMERA_EDIT_ALLOWED)

    @property
    def support_visual_allowed(self) -> bool:
        return bool(self.protection & Protection.SUPPORT_VISUAL_ALLOWED) and not self.source_required

    @property
    def source_required(self) -> bool:
        return bool(self.protection & Protection.SOURCE_REQUIRED)

    def overlaps(self, start: float, end: float) -> bool:
        return self.start < end and start < self.end

    def to_dict(self) -> dict[str, object]:
        flags = [f.name for f in (Protection.SOURCE_REQUIRED, Protection.CAMERA_EDIT_ALLOWED,
                                  Protection.SUPPORT_VISUAL_ALLOWED) if self.protection & f] or ["LOCKED"]
        return {
            "span_id": self.span_id, "role": self.role.value, "domain": self.domain.value,
            "start": round(self.start, 3), "end": round(self.end, 3), "protection": flags,
            "visual_importance": self.visual_importance,
            "required_subject_ids": list(self.required_subject_ids),
            "required_regions": [r.to_list() for r in self.required_regions],
            "min_visible_fraction": self.min_visible_fraction, "must_keep": self.must_keep,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class RegionEvidence:
    """Important region over time (e.g. from a tracker sidecar ``regions`` list)."""

    start: float
    end: float
    region: RequiredRegion


def _overlapping(ranges: Iterable[TimeRange], start: float, end: float) -> list[TimeRange]:
    return [r for r in ranges if r.start < end and start < r.end]


def build_story_spans(
    *,
    segments: Sequence[tuple[float, float, StoryRole]],
    must_keep: Sequence[TimeRange],
    visual_peaks: Sequence[TimeRange],
    reactions: Sequence[TimeRange],
    speech: Sequence[TimeRange],
    visible_start: float,
    action_subjects: Mapping[str, tuple[float, float, str]] | None = None,
    regions: Sequence[RegionEvidence] = (),
) -> tuple[StorySpan, ...]:
    """One span per story segment of the paced main, numbered per role.

    ``action_subjects`` maps subject_id -> (first_t, last_t, kind) for reliable
    person/object tracks; they become required subjects of action spans.
    """
    counters: dict[StoryRole, int] = {}
    spans: list[StorySpan] = []
    for start, end, role in segments:
        counters[role] = counters.get(role, 0) + 1
        span_id = f"{role.value}_{counters[role]:02d}"
        protection = ROLE_PROTECTION[role]
        evidence: list[str] = []
        keep = bool(_overlapping(must_keep, start, end))
        if keep:
            protection = (protection | Protection.SOURCE_REQUIRED) & ~Protection.SUPPORT_VISUAL_ALLOWED
            evidence.append("must_keep")
        if end <= visible_start:
            protection = Protection.LOCKED  # never shown after the intro handoff
            evidence.append("before_visible_main")
        peaks = _overlapping(visual_peaks, start, end)
        if peaks:
            importance = "action"
            evidence.extend(sorted({p.label for p in peaks if p.label}))
        elif _overlapping(reactions, start, end):
            importance = "reaction"
        elif _overlapping(speech, start, end):
            importance = "speech"
        else:
            importance = "normal"
        span_regions = tuple(e.region for e in regions if e.start < end and start < e.end)
        required_subjects: tuple[str, ...] = ()
        min_visible = 0.0
        if importance == "action" and role in ACTION_ROLES:
            required_subjects = tuple(sorted(
                sid for sid, (t0, t1, kind) in (action_subjects or {}).items()
                if kind in {"person", "object"} and t0 < end and start < t1))
            # Unknown action location -> keep most of the frame on screen.
            if not span_regions and not required_subjects:
                min_visible = ACTION_MIN_VISIBLE_FRACTION
        spans.append(StorySpan(
            span_id=span_id, role=role, start=start, end=end, domain=TimelineDomain.PACED_CLIP,
            protection=protection, visual_importance=importance, required_subject_ids=required_subjects,
            required_regions=span_regions, min_visible_fraction=min_visible, must_keep=keep,
            evidence=tuple(evidence),
        ))
    return tuple(spans)


def intro_span(teaser_start: float, teaser_end: float, *, has_action: bool,
               regions: Sequence[RegionEvidence] = ()) -> StorySpan:
    """The mandatory cold-open (already selected by MIMIR). PACED source range."""
    span_regions = tuple(e.region for e in regions if e.start < teaser_end and teaser_start < e.end)
    return StorySpan(
        span_id=INTRO_SPAN_ID, role=StoryRole.PAYOFF, start=teaser_start, end=teaser_end,
        domain=TimelineDomain.INTRO, protection=Protection.SOURCE_REQUIRED | Protection.CAMERA_EDIT_ALLOWED,
        visual_importance="action" if has_action else "peak", required_regions=span_regions,
        min_visible_fraction=ACTION_MIN_VISIBLE_FRACTION if not span_regions else 0.0, must_keep=True,
        evidence=("mandatory_intro_peak",),
    )


def story_signature(spans: Iterable[StorySpan], source_identity: str) -> str:
    rows = [[s.span_id, s.role.value, s.domain.value, round(s.start, 3), round(s.end, 3)] for s in spans]
    payload = {"source": source_identity, "spans": rows}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def source_story_signature(clip_timeline: Mapping[str, object]) -> str:
    """Signature of the story-system inputs themselves (timeline truth).

    Recomputed before and after Pro Edit to catch any cross-stage mutation of
    clip boundaries, cut ranges, payoff or protected ranges.
    """
    keys = ("clip_index", "source", "cut_ranges", "payoff", "protected_ranges", "hook")
    payload = {key: clip_timeline.get(key) for key in keys}
    editorial = clip_timeline.get("editorial")
    if isinstance(editorial, dict):
        payload["anchors"] = editorial.get("anchor_moments")
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()
