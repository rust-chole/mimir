"""EditContext: one read-only, PACED_CLIP-domain view of existing MIMIR truth.

Built from artifacts MIMIR already produced for the selected short:

* timeline JSON (final cut ranges persisted by pacing_cutter, payoff,
  protected must-keep ranges, hook text, anchors)      -> story
* selected clip analysis (anchors / editor notes, VOD time) -> story hints
* final speaker profile (exact-final 48 kHz word clock)  -> words/speakers
* selected-clip visual report (paced-clip seconds)       -> visual evidence
* caption ASS + intro renderer policy                    -> visible main start
* optional subject sidecar                                -> subject tracks

Nothing here is fabricated: absent evidence stays empty. No model call, no
whole-VOD analysis. Source structures are never mutated (copies only).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ai.editor import timeline as mimir_timeline
from ai.editor.pro_edit.caption_guard import (
    EMPTY_CAPTION_REGION,
    CaptionSafeRegion,
    CaptionWordRef,
    caption_safe_region,
    caption_signature,
    caption_words,
    trusted_display_names,
)
from ai.editor.pro_edit.errors import EditContextError
from ai.editor.pro_edit.intro_timeline import HookTextBand, IntroTimeline
from ai.editor.pro_edit.speaker_link import associate_speakers
from ai.editor.pro_edit.story import (
    RegionEvidence,
    RequiredRegion,
    StorySpan,
    build_story_spans,
    intro_span,
    source_story_signature,
    story_signature,
)
from ai.editor.pro_edit.media import MediaInfo
from ai.editor.pro_edit.schema import ROLE_PRIORITY, StoryRole
from ai.editor.pro_edit.subjects import NullSubjectProvider, SubjectTrack, SubjectTrackProvider
from ai.editor.pro_edit.timebase import ClipTimelineMap, FrameRate, TimelineDomain, TimeRange, clip_ranges

EDIT_CONTEXT_SCHEMA_VERSION = 2
SPEECH_GAP_MIN_S = 0.60
REACTION_POINT_WINDOW_S = 0.80
MAX_PLANNER_WORDS = 360
MAX_PLANNER_VISUAL_EVENTS = 24
MAX_DESCRIPTION_CHARS = 100

ESCALATION_ANCHOR_TYPES = frozenset({"escalation", "conflict", "tension", "awkward", "ridiculous_claim"})
REACTION_ANCHOR_TYPES = frozenset({"reaction"})
PEAK_ANCHOR_TYPES = frozenset({
    "punchline", "reveal", "reversal", "unexpected_answer", "failure", "win",
    "physical_payoff", "visual_impact", "destruction", "quotable",
})
VISUAL_PEAK_TYPES = frozenset({
    "visual_payoff", "impact", "crash", "fall", "explosion", "destruction",
    "object_break", "reveal", "physical_action",
})
MOTION_EVENT_TYPES = frozenset({"physical_action", "sudden_change", "gameplay_event", "entrance_exit"})

P = TimelineDomain.PACED_CLIP


@dataclass(frozen=True)
class ClipInfo:
    duration_s: float
    fps: FrameRate
    frame_count: int
    width: int
    height: int
    has_audio: bool
    visible_start_s: float = 0.0
    timeline_domain: TimelineDomain = P

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def frame_tolerance_s(self) -> float:
        return max(self.fps.frame_duration, 1e-3)


# Pro Edit holds caption words by reference only (see caption_guard).
WordRef = CaptionWordRef


@dataclass(frozen=True)
class StorySegment:
    start: float
    end: float
    role: StoryRole


@dataclass(frozen=True)
class StoryEvidence:
    hook_ranges: tuple[TimeRange, ...] = ()
    setup_ranges: tuple[TimeRange, ...] = ()
    escalation_ranges: tuple[TimeRange, ...] = ()
    payoff_ranges: tuple[TimeRange, ...] = ()
    reaction_ranges: tuple[TimeRange, ...] = ()
    bridge_ranges: tuple[TimeRange, ...] = ()
    protected_ranges: tuple[TimeRange, ...] = ()
    must_keep_ranges: tuple[TimeRange, ...] = ()
    visual_peak_ranges: tuple[TimeRange, ...] = ()
    segments: tuple[StorySegment, ...] = ()


@dataclass(frozen=True)
class VisualEvent:
    start: float
    end: float
    type: str
    description: str
    confidence: float


@dataclass(frozen=True)
class EditContext:
    schema_version: int
    clip_identity: str
    clip: ClipInfo
    story: StoryEvidence
    words: tuple[WordRef, ...] = ()
    speaker_segments: tuple[tuple[float, float, str], ...] = ()
    speaker_identities: Mapping[str, str] = field(default_factory=dict)
    subject_tracks: tuple[SubjectTrack, ...] = ()
    subject_provider: str = "none"
    speech_gap_ranges: tuple[TimeRange, ...] = ()
    energy_events: tuple[TimeRange, ...] = ()
    laughter_events: tuple[TimeRange, ...] = ()
    transient_events: tuple[TimeRange, ...] = ()
    scene_changes: tuple[float, ...] = ()
    motion_events: tuple[TimeRange, ...] = ()
    visual_events: tuple[VisualEvent, ...] = ()
    intro_metadata: Mapping[str, Any] = field(default_factory=dict)
    meme_candidates: tuple[Mapping[str, Any], ...] = ()
    sfx_metadata: tuple[Mapping[str, Any], ...] = ()
    layout_content_type: str = ""
    evidence_sources: tuple[str, ...] = ()
    spans: tuple[StorySpan, ...] = ()
    intro_span: StorySpan | None = None
    intro: IntroTimeline | None = None
    story_signature: str = ""
    source_story_signature: str = ""
    caption_signature: str = ""
    caption_region: CaptionSafeRegion = EMPTY_CAPTION_REGION
    hook_band: HookTextBand | None = None

    # --- lookups -----------------------------------------------------------

    def span(self, span_id: str | None) -> StorySpan | None:
        if span_id is None:
            return None
        if self.intro_span is not None and span_id == self.intro_span.span_id:
            return self.intro_span
        return next((s for s in self.spans if s.span_id == span_id), None)

    def spans_overlapping(self, start: float, end: float) -> tuple[StorySpan, ...]:
        return tuple(s for s in self.spans if s.overlaps(start, end))

    def dominant_span(self, start: float, end: float) -> StorySpan | None:
        best = None
        best_key = (0.0, -1)
        for span in self.spans_overlapping(start, end):
            key = (round(min(end, span.end) - max(start, span.start), 6), ROLE_PRIORITY[span.role])
            if key > best_key:
                best, best_key = span, key
        return best

    def word_index(self) -> dict[int, WordRef]:
        return {word.id: word for word in self.words}

    def subject_ids(self) -> frozenset[str]:
        return frozenset(track.subject_id for track in self.subject_tracks)

    def speaker_ids(self) -> frozenset[str]:
        return frozenset(seg[2] for seg in self.speaker_segments)

    def dominant_role(self, start: float, end: float) -> StoryRole:
        """Story role owning most of [start, end]; ties -> higher priority."""
        best_role = StoryRole.NEUTRAL
        best_key = (0.0, -1)
        for segment in self.story.segments:
            overlap = max(0.0, min(end, segment.end) - max(start, segment.start))
            if overlap <= 0:
                continue
            key = (round(overlap, 6), ROLE_PRIORITY[segment.role])
            if key > best_key:
                best_key, best_role = key, segment.role
        return best_role

    def overlaps_protected(self, start: float, end: float) -> bool:
        return any(r.start < end and start < r.end for r in self.story.protected_ranges)

    def active_speaker(self, start: float, end: float) -> str | None:
        totals: dict[str, float] = {}
        for seg_start, seg_end, speaker in self.speaker_segments:
            overlap = max(0.0, min(end, seg_end) - max(start, seg_start))
            if overlap > 0:
                totals[speaker] = totals.get(speaker, 0.0) + overlap
        if not totals:
            return None
        return sorted(totals.items(), key=lambda item: (-item[1], item[0]))[0][0]

    # --- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        def ranges(items: Iterable[TimeRange]) -> list[dict[str, object]]:
            return [item.to_dict() for item in items]

        return {
            "schema_version": self.schema_version,
            "clip_identity": self.clip_identity,
            "clip": {
                "duration_s": round(self.clip.duration_s, 6),
                "fps": str(self.clip.fps),
                "frame_count": self.clip.frame_count,
                "width": self.clip.width,
                "height": self.clip.height,
                "aspect_ratio": round(self.clip.aspect_ratio, 6),
                "has_audio": self.clip.has_audio,
                "visible_start_s": round(self.clip.visible_start_s, 3),
                "timeline_domain": self.clip.timeline_domain.value,
            },
            "story": {
                "hook_ranges": ranges(self.story.hook_ranges),
                "setup_ranges": ranges(self.story.setup_ranges),
                "escalation_ranges": ranges(self.story.escalation_ranges),
                "payoff_ranges": ranges(self.story.payoff_ranges),
                "reaction_ranges": ranges(self.story.reaction_ranges),
                "bridge_ranges": ranges(self.story.bridge_ranges),
                "protected_ranges": ranges(self.story.protected_ranges),
                "must_keep_ranges": ranges(self.story.must_keep_ranges),
                "visual_peak_ranges": ranges(self.story.visual_peak_ranges),
                "segments": [
                    {"start": round(s.start, 3), "end": round(s.end, 3), "role": s.role.value}
                    for s in self.story.segments
                ],
            },
            "transcript": {
                "ownership": "read-only references to the final speaker profile (caption system owns truth)",
                "words": [
                    {"id": w.id, "text": w.text, "start": round(w.start, 3), "end": round(w.end, 3),
                     "speaker_id": w.speaker_id, "confidence": round(w.confidence, 3)}
                    for w in self.words
                ],
            },
            "speakers": {
                "segments": [
                    {"start": round(a, 3), "end": round(b, 3), "speaker_id": s} for a, b, s in self.speaker_segments
                ],
                "identities": dict(self.speaker_identities),
            },
            "subjects": {
                "provider": self.subject_provider,
                "tracks": [track.summary() for track in self.subject_tracks],
            },
            "audio": {
                "speech_gap_ranges": ranges(self.speech_gap_ranges),
                "energy_events": ranges(self.energy_events),
                "laughter_events": ranges(self.laughter_events),
                "transient_events": ranges(self.transient_events),
            },
            "visual": {
                "scene_changes": [round(t, 3) for t in self.scene_changes],
                "motion_events": ranges(self.motion_events),
                "important_actions": [
                    {"start": round(v.start, 3), "end": round(v.end, 3), "type": v.type,
                     "description": v.description, "confidence": round(v.confidence, 3)}
                    for v in self.visual_events
                ],
            },
            "existing": {
                "intro_metadata": dict(self.intro_metadata),
                "meme_candidates": [dict(x) for x in self.meme_candidates],
                "sfx_metadata": [dict(x) for x in self.sfx_metadata],
            },
            "layout_content_type": self.layout_content_type,
            "evidence_sources": list(self.evidence_sources),
            "story_spans": [span.to_dict() for span in self.spans],
            "intro_span": self.intro_span.to_dict() if self.intro_span is not None else None,
            "intro_timeline": self.intro.to_dict() if self.intro is not None else None,
            "caption_safe_region": self.caption_region.to_dict(),
            "intro_hook_text_band": self.hook_band.to_dict() if self.hook_band is not None else None,
            "signatures": {
                "story": self.story_signature,
                "source_story": self.source_story_signature,
                "caption_words": self.caption_signature,
            },
        }


# ============================================================
# BUILDER HELPERS
# ============================================================

def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    target = Path(path)
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _paced(ranges: Iterable[TimeRange | None], duration: float) -> tuple[TimeRange, ...]:
    result: list[TimeRange] = []
    for item in ranges:
        if item is None:
            continue
        lo = max(0.0, min(duration, item.start))
        hi = max(0.0, min(duration, item.end))
        if hi - lo > 1e-6:
            result.append(TimeRange(lo, hi, P, item.label))
    result.sort(key=lambda r: (r.start, r.end, r.label))
    return tuple(result)


def _range(start: Any, end: Any, domain: TimelineDomain, label: str) -> TimeRange | None:
    lo, hi = _num(start), _num(end)
    if lo is None or hi is None or hi <= lo:
        return None
    return TimeRange(lo, hi, domain, label)


def _find_clip(timeline_data: Mapping[str, Any], clip_index: int) -> dict[str, Any]:
    for position, item in enumerate(timeline_data.get("timelines", []) or [], start=1):
        if not isinstance(item, dict):
            continue
        try:
            current = int(item.get("clip_index", position))
        except (TypeError, ValueError):
            current = position
        if current == int(clip_index):
            return copy.deepcopy(item)
    raise EditContextError(f"timeline has no clip_index={clip_index}")


def _words_from_profile(profile: Mapping[str, Any] | None, duration: float) -> tuple[WordRef, ...]:
    """Authoritative caption words by reference (ids = index in the final profile).

    Values are NOT clamped or rewritten: the caption signature must describe
    exactly what the caption system produced.
    """
    del duration
    return tuple(sorted(caption_words(profile), key=lambda w: (w.start, w.end, w.id)))


def _speaker_segments(words: Sequence[WordRef], gap: float = 0.6) -> tuple[tuple[float, float, str], ...]:
    segments: list[list[Any]] = []
    for word in words:
        if word.speaker_id is None:
            continue
        if segments and segments[-1][2] == word.speaker_id and word.start - segments[-1][1] <= gap:
            segments[-1][1] = max(segments[-1][1], word.end)
        else:
            segments.append([word.start, word.end, word.speaker_id])
    return tuple((float(a), float(b), str(s)) for a, b, s in segments)


def _speech_gaps(words: Sequence[WordRef], duration: float) -> tuple[TimeRange, ...]:
    if not words:
        return ()
    gaps: list[TimeRange] = []
    cursor = 0.0
    for word in sorted(words, key=lambda w: w.start):
        if word.start - cursor >= SPEECH_GAP_MIN_S:
            gaps.append(TimeRange(cursor, word.start, P, "derived_from_caption_words"))
        cursor = max(cursor, word.end)
    if duration - cursor >= SPEECH_GAP_MIN_S:
        gaps.append(TimeRange(cursor, duration, P, "derived_from_caption_words"))
    return tuple(gaps)


def _segment_story(
    duration: float,
    layers: Mapping[StoryRole, Sequence[TimeRange]],
    visible_start: float,
) -> tuple[StorySegment, ...]:
    """Deterministic role partition of [0, duration] from evidence layers.

    Highest ROLE_PRIORITY wins per elementary interval. Uncovered time before
    the first story evidence is SETUP, later uncovered time is BRIDGE; with no
    story evidence at all everything is NEUTRAL.
    """
    points = {0.0, float(duration), float(visible_start)}
    for ranges in layers.values():
        for item in ranges:
            points.add(max(0.0, min(duration, item.start)))
            points.add(max(0.0, min(duration, item.end)))
    ordered = sorted(points)
    story_ranges = [r for role, rs in layers.items() if role is not StoryRole.HOOK for r in rs]
    first_story = min((r.start for r in story_ranges), default=None)
    segments: list[StorySegment] = []
    for lo, hi in zip(ordered, ordered[1:]):
        if hi - lo <= 1e-6:
            continue
        mid = (lo + hi) / 2.0
        covering = [role for role, rs in layers.items() if any(r.start <= mid < r.end for r in rs)]
        if covering:
            role = max(covering, key=lambda r: ROLE_PRIORITY[r])
        elif first_story is None:
            role = StoryRole.NEUTRAL
        elif mid < first_story:
            role = StoryRole.SETUP
        else:
            role = StoryRole.BRIDGE
        if segments and segments[-1].role is role and abs(segments[-1].end - lo) <= 1e-6:
            segments[-1] = StorySegment(segments[-1].start, hi, role)
        else:
            segments.append(StorySegment(lo, hi, role))
    return tuple(segments)


def clip_identity_for(video_stem: str, clip_index: int, clip_map: ClipTimelineMap, frame_count: int) -> str:
    payload = {
        "video_stem": video_stem,
        "clip_index": int(clip_index),
        "vod_start": round(clip_map.clip_vod_start, 6),
        "raw_duration": round(clip_map.raw_duration, 6),
        "cuts": [[round(a, 6), round(b, 6)] for a, b in clip_map.cut_ranges],
        "frames": int(frame_count),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


# ============================================================
# BUILDER
# ============================================================

@dataclass
class ContextInputs:
    timeline_data: Mapping[str, Any]
    clip_index: int
    media: MediaInfo
    analysis_clip: Mapping[str, Any] | None = None
    speaker_profile_path: str | Path | None = None
    video_report_path: str | Path | None = None
    visible_start_s: float = 0.0
    intro_metadata: Mapping[str, Any] = field(default_factory=dict)
    subject_provider: SubjectTrackProvider = field(default_factory=NullSubjectProvider)
    caption_path: str | Path | None = None
    intro: IntroTimeline | None = None
    hook_band: HookTextBand | None = None


def build_edit_context(inputs: ContextInputs) -> EditContext:
    media = inputs.media
    if media.frame_count <= 0 or media.duration_s <= 0:
        raise EditContextError("paced clip media has no frames/duration")
    duration = media.video_duration_s if media.video_duration_s > 0 else media.duration_s
    clip_timeline = _find_clip(inputs.timeline_data, inputs.clip_index)
    clip_map = ClipTimelineMap.from_timeline_clip(clip_timeline)
    sources: list[str] = ["timeline", "paced_clip_probe"]

    # Consistency guard: final cut ranges must explain the rendered duration.
    if abs(clip_map.paced_duration - duration) > max(0.25, 3 * media.fps.frame_duration):
        raise EditContextError(
            "timeline cut ranges do not match the rendered paced clip "
            f"(timeline={clip_map.paced_duration:.3f}s, media={duration:.3f}s); refusing to guess"
        )

    V, R = TimelineDomain.VOD, TimelineDomain.RAW_CLIP

    def to_paced(items: Iterable[TimeRange | None]) -> tuple[TimeRange, ...]:
        return _paced((clip_map.range_to_paced(i) if i is not None else None for i in items), duration)

    payoff_raw = clip_timeline.get("payoff", {}) if isinstance(clip_timeline.get("payoff"), dict) else {}
    payoff = to_paced([_range(payoff_raw.get("source_start"), payoff_raw.get("source_end"), R, "payoff")])
    must_keep = to_paced(
        _range(item.get("start"), item.get("end"), R, "must_keep")
        for item in clip_timeline.get("protected_ranges", []) or [] if isinstance(item, dict)
    )

    editorial = clip_timeline.get("editorial", {}) if isinstance(clip_timeline.get("editorial"), dict) else {}
    anchors = [a for a in (editorial.get("anchor_moments") or []) if isinstance(a, dict)]
    if inputs.analysis_clip and not anchors:
        anchors = [a for a in (inputs.analysis_clip.get("anchor_moments") or []) if isinstance(a, dict)]
    escalation: list[TimeRange | None] = []
    reaction: list[TimeRange | None] = []
    peaks: list[TimeRange | None] = []
    for anchor in anchors:
        kind = str(anchor.get("type", "other")).strip().casefold()
        item = _range(anchor.get("start"), anchor.get("end"), V, f"anchor:{kind}")
        if kind in REACTION_ANCHOR_TYPES:
            reaction.append(item)
        elif kind in ESCALATION_ANCHOR_TYPES:
            escalation.append(item)
        elif kind in PEAK_ANCHOR_TYPES:
            peaks.append(item)
    if anchors:
        sources.append("terra_anchor_moments")

    if inputs.analysis_clip:
        for note in inputs.analysis_clip.get("editor_notes", []) or []:
            if isinstance(note, dict) and str(note.get("type")) == "reaction_hold":
                t = _num(note.get("time"))
                if t is not None:
                    reaction.append(TimeRange(t, t + REACTION_POINT_WINDOW_S, V, "terra_note:reaction_hold"))

    paced_payoff = payoff
    escalation_paced = list(to_paced(escalation))
    peak_paced = list(to_paced(peaks))
    # Peaks overlapping the payoff reinforce it; earlier peaks are escalation.
    for item in peak_paced:
        if not any(item.overlaps(p) for p in paced_payoff):
            escalation_paced.append(TimeRange(item.start, item.end, P, item.label))
    reaction_paced = list(to_paced(reaction))

    visual_events: list[VisualEvent] = []
    scene_changes: list[float] = []
    motion_events: list[TimeRange] = []
    visual_peaks: list[TimeRange] = []
    report = _load_json(inputs.video_report_path)
    layout_content_type = ""
    if report is not None:
        sources.append("selected_clip_visual_report")
        layout = report.get("layout", {})
        if isinstance(layout, dict):
            layout_content_type = str(layout.get("content_type", "")).strip().casefold()
        for raw in report.get("visual_events", []) or []:
            if not isinstance(raw, dict):
                continue
            item = _range(raw.get("start"), raw.get("end"), P, f"visual:{raw.get('type', 'other')}")
            if item is None:
                t = _num(raw.get("start"))
                item = TimeRange(t, t + 0.3, P, f"visual:{raw.get('type', 'other')}") if t is not None else None
            if item is None:
                continue
            kind = str(raw.get("type", "other"))
            conf = _num(raw.get("confidence")) or 0.0
            desc = " ".join(str(raw.get("description", "")).split())[:MAX_DESCRIPTION_CHARS]
            visual_events.append(VisualEvent(item.start, item.end, kind, desc, max(0.0, min(1.0, conf))))
            if kind == "reaction":
                reaction_paced.append(item)
            if kind in VISUAL_PEAK_TYPES:
                visual_peaks.append(item)
            if kind in MOTION_EVENT_TYPES:
                motion_events.append(item)
            if kind == "scene_change":
                scene_changes.append(item.start)
        support = report.get("editing_support", {}) if isinstance(report.get("editing_support"), dict) else {}
        for t in support.get("scene_change_times", []) or []:
            value = _num(t)
            if value is not None:
                scene_changes.append(value)
        for t in support.get("reaction_times", []) or []:
            value = _num(t)
            if value is not None:
                reaction_paced.append(TimeRange(value, value + REACTION_POINT_WINDOW_S, P, "visual:reaction_time"))
        intro_support = report.get("intro_visual_support", {})
        if isinstance(intro_support, dict):
            for raw in intro_support.get("visual_peak_regions", []) or []:
                if isinstance(raw, dict):
                    item = _range(raw.get("start"), raw.get("end"), P, "visual:peak_region")
                    if item is not None:
                        visual_peaks.append(item)

    visible_start = max(0.0, min(duration, float(inputs.visible_start_s or 0.0)))
    hook: list[TimeRange] = []
    hook_block = clip_timeline.get("hook", {}) if isinstance(clip_timeline.get("hook"), dict) else {}
    if str(hook_block.get("text", "")).strip():
        span = min(float(getattr(mimir_timeline, "HOOK_TEXT_DURATION", 2.2)), 0.15 * max(0.0, duration - visible_start))
        if span > 0.2:
            hook.append(TimeRange(visible_start, visible_start + span, P, "timeline_hook_text"))

    layers: dict[StoryRole, tuple[TimeRange, ...]] = {
        StoryRole.HOOK: _paced(hook, duration),
        StoryRole.ESCALATION: _paced(escalation_paced, duration),
        StoryRole.PAYOFF: paced_payoff,
        StoryRole.REACTION: _paced(reaction_paced, duration),
    }
    segments = _segment_story(duration, layers, visible_start)
    setup = tuple(TimeRange(s.start, s.end, P, "derived_setup") for s in segments if s.role is StoryRole.SETUP)
    bridge = tuple(TimeRange(s.start, s.end, P, "derived_bridge") for s in segments if s.role is StoryRole.BRIDGE)
    protected = _paced(list(must_keep) + list(paced_payoff), duration)

    profile = _load_json(inputs.speaker_profile_path)
    words = _words_from_profile(profile, duration)
    if words:
        sources.append("final_speaker_profile_words")
    # Only identities the caption renderer may display (captions V24: human-confirmed).
    identities: dict[str, str] = trusted_display_names(profile)

    sfx_meta: list[dict[str, Any]] = []
    for event in clip_timeline.get("events", []) or []:
        if isinstance(event, dict) and str(event.get("type", "")) == "sfx_suggestion":
            raw_t = _num(event.get("source_time"))
            if raw_t is None:
                continue
            paced_t = clip_map.range_to_paced(TimeRange(raw_t, raw_t + 1e-3, R, "sfx"))
            if paced_t is not None:
                sfx_meta.append({"time": round(paced_t.start, 3), "sfx": str(event.get("sfx", ""))})

    tracks = inputs.subject_provider.load(duration)
    if tracks:
        sources.append(f"subjects:{getattr(inputs.subject_provider, 'name', 'provider')}")
        if words:
            tracks = associate_speakers(tracks, words)
            if any(t.speaker_id for t in tracks):
                sources.append("speaker_visual_association")
    region_rows = getattr(inputs.subject_provider, "load_regions", lambda _d: ())(duration)
    regions = tuple(RegionEvidence(a, b, RequiredRegion(*box, source=label)) for a, b, box, label in region_rows)

    action_subjects: dict[str, tuple[float, float, str]] = {}
    for track in tracks:
        good = [x.t for x in track.samples if x.confidence >= 0.5]
        if good and track.kind in {"person", "object"}:
            action_subjects[track.subject_id] = (min(good), max(good), track.kind)
    speech = tuple(TimeRange(w.start, w.end, P, "speech") for w in words)
    spans = build_story_spans(
        segments=[(seg.start, seg.end, seg.role) for seg in segments],
        must_keep=must_keep,
        visual_peaks=_paced(visual_peaks, duration),
        reactions=layers[StoryRole.REACTION],
        speech=speech,
        visible_start=visible_start,
        action_subjects=action_subjects,
        regions=regions,
    )
    the_intro_span = None
    if inputs.intro is not None:
        peaks_in_intro = [r for r in _paced(visual_peaks, duration)
                          if r.start < inputs.intro.teaser_end and inputs.intro.teaser_start < r.end]
        the_intro_span = intro_span(inputs.intro.teaser_start, inputs.intro.teaser_end,
                                    has_action=bool(peaks_in_intro), regions=regions)
        sources.append("intro_timeline")
    caption_region = caption_safe_region(inputs.caption_path)
    if not caption_region.empty:
        sources.append("caption_safe_region")

    source_block = inputs.timeline_data.get("source", {})
    stem = str(source_block.get("video_stem", "vod")) if isinstance(source_block, dict) else "vod"
    identity = clip_identity_for(stem, inputs.clip_index, clip_map, media.frame_count)
    all_spans = spans + ((the_intro_span,) if the_intro_span is not None else ())

    return EditContext(
        schema_version=EDIT_CONTEXT_SCHEMA_VERSION,
        clip_identity=identity,
        clip=ClipInfo(
            duration_s=duration,
            fps=media.fps,
            frame_count=media.frame_count,
            width=media.width,
            height=media.height,
            has_audio=media.has_audio,
            visible_start_s=visible_start,
        ),
        story=StoryEvidence(
            hook_ranges=layers[StoryRole.HOOK],
            setup_ranges=setup,
            escalation_ranges=layers[StoryRole.ESCALATION],
            payoff_ranges=paced_payoff,
            reaction_ranges=layers[StoryRole.REACTION],
            bridge_ranges=bridge,
            protected_ranges=protected,
            must_keep_ranges=must_keep,
            visual_peak_ranges=_paced(visual_peaks, duration),
            segments=segments,
        ),
        words=words,
        speaker_segments=_speaker_segments(words),
        speaker_identities=identities,
        subject_tracks=tuple(tracks),
        subject_provider=str(getattr(inputs.subject_provider, "name", "provider")),
        speech_gap_ranges=_speech_gaps(words, duration),
        scene_changes=tuple(sorted({round(t, 3) for t in scene_changes if 0.0 <= t <= duration})),
        motion_events=_paced(motion_events, duration),
        visual_events=tuple(sorted(visual_events, key=lambda v: (v.start, v.end, v.type))),
        intro_metadata=dict(inputs.intro_metadata),
        sfx_metadata=tuple(sfx_meta),
        layout_content_type=layout_content_type,
        evidence_sources=tuple(sources),
        spans=spans,
        intro_span=the_intro_span,
        intro=inputs.intro,
        story_signature=story_signature(all_spans, identity),
        source_story_signature=source_story_signature(clip_timeline),
        caption_signature=caption_signature(words),
        caption_region=caption_region,
        hook_band=inputs.hook_band,
    )
