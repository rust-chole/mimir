"""Causal story integrity: does the SELECTED short keep what its payoff needs?

CCR/V3 story selection stays authoritative; this layer never ranks candidates
and never calls a model. After selection it asks one question over evidence
MIMIR already has (transcript words, visual events, payoff, anchors, protected
ranges, current boundaries):

    SETUP -> (ESCALATION) -> PAYOFF -> REACTION

Semantic necessity, not duration buckets: a one-word reaction is a reaction,
a silent visual setup is a setup, a story may have no escalation beat. The
checks are structural facts:

* the payoff the candidate claims lies inside its boundaries;
* no boundary cuts through a running utterance (a cut sentence loses its cause
  or its consequence);
* something precedes the payoff (speech or a visual event), something follows it;
* no protected range crosses a boundary.

Outcomes (recorded with the reason):

    PASS      causally understandable as selected
    REPAIR    ONE bounded boundary expansion to the nearest phrase edge / the
              adjacent phrase, never across a scene-like pause, never beyond the
              analyzer's maximum clip length; the added setup / reaction phrase is
              protected so pacing cannot cut it again; then re-checked (no loop)
    WARN      understandable, but an optional beat is weak / not provable
    RESELECT  only when the candidate lacks the payoff it claims AND another
              already-generated candidate exists (and the user did not pin a clip)

Ideas borrowed from the V6 causal layer (speech-aligned expansion that never
crosses a large pause, protected causal handles); V6's per-type duration
defaults are deliberately NOT ported.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

STORY_INTEGRITY_VERSION = 2
# Generic speech/safety bounds (not tuned to any video, not beat-length targets).
PHRASE_GAP_S = 0.45          # a pause this long separates phrases
SCENE_GAP_S = 1.90           # never expand across a pause this long (scene/topic break)
CONTINUOUS_S = 0.25          # neighbours closer than this across a boundary = one running utterance
MAX_SETUP_REACH_S = 20.0     # a missing setup is looked for at most this far before the current start
MAX_UTTERANCE_REACH_S = 8.0  # a cut utterance is completed only if its edge is at most this far away
MAX_REACTION_REACH_S = 6.0   # a reaction is looked for at most this far after the current end
PAYOFF_TOLERANCE_S = 0.15
MAX_ADDED_PROTECTIONS = 2


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class Evidence:
    words: list[tuple[float, float, str]]                  # absolute VOD time, sorted
    visual: list[tuple[float, float, str]] = field(default_factory=list)

    @classmethod
    def build(cls, transcript: Mapping[str, Any] | None, visual_events: Sequence[Mapping[str, Any]] = ()
              ) -> "Evidence":
        words = []
        for item in (transcript or {}).get("words", []) or []:
            if not isinstance(item, Mapping) or not str(item.get("word", "")).strip():
                continue
            a, b = _f(item.get("start"), -1.0), _f(item.get("end"), -1.0)
            if a >= 0 and b >= a:
                words.append((a, b, str(item["word"]).strip()))
        visual = [(_f(e.get("start")), _f(e.get("end")), str(e.get("type", "visual"))) for e in visual_events
                  if isinstance(e, Mapping) and _f(e.get("end")) > _f(e.get("start"))]
        return cls(sorted(words), sorted(visual))

    def words_in(self, a: float, b: float) -> list[tuple[float, float, str]]:
        return [w for w in self.words if w[0] < b and w[1] > a]

    def visual_in(self, a: float, b: float) -> list[tuple[float, float, str]]:
        return [v for v in self.visual if v[0] < b and v[1] > a]

    # --- phrase geometry ------------------------------------------------------

    def cut_through(self, t: float) -> bool:
        """True when a running utterance crosses ``t`` (a word straddles it, or the words on
        both sides are closer than CONTINUOUS_S)."""
        before = [w for w in self.words if w[0] < t]
        after = [w for w in self.words if w[0] >= t]
        if any(w[0] < t - 0.02 and w[1] > t + 0.02 for w in self.words):
            return True
        return bool(before and after and after[0][0] - before[-1][1] < CONTINUOUS_S
                    and t - before[-1][1] < CONTINUOUS_S and after[0][0] - t < CONTINUOUS_S)

    def phrase_start_before(self, t: float, floor: float) -> float | None:
        """Start of the phrase running into ``t`` (walk back over sub-phrase pauses); None when
        that start lies beyond ``floor`` (a point inside the phrase would still cut it)."""
        before = [w for w in self.words if w[0] < t]
        if not before:
            return None
        index = len(before) - 1
        while index > 0 and before[index][0] - before[index - 1][1] < PHRASE_GAP_S:
            index -= 1
        return before[index][0] if before[index][0] >= floor else None

    def phrase_end_after(self, t: float, ceiling: float) -> float | None:
        after = [w for w in self.words if w[1] > t]
        if not after:
            return None
        index = 0
        while index + 1 < len(after) and after[index + 1][0] - after[index][1] < PHRASE_GAP_S:
            index += 1
        return after[index][1] if after[index][1] <= ceiling else None

    def previous_phrase(self, t: float, floor: float) -> tuple[float, float] | None:
        """The phrase ending right before ``t`` if no scene-like pause separates it."""
        before = [w for w in self.words if w[1] <= t + 0.02]
        if not before or t - before[-1][1] > SCENE_GAP_S:
            return None
        start = self.phrase_start_before(before[-1][1] + 1e-3, floor)
        return (start, before[-1][1]) if start is not None else None

    def next_phrase(self, t: float, ceiling: float) -> tuple[float, float] | None:
        after = [w for w in self.words if w[0] >= t - 0.02]
        if not after or after[0][0] - t > SCENE_GAP_S:
            return None
        end = self.phrase_end_after(after[0][0], ceiling)
        return (after[0][0], end) if end is not None else None


@dataclass
class IntegrityResult:
    status: str                                  # pass | repaired | warn | reselect
    clip: dict[str, Any]
    checks: list[dict[str, Any]]
    reason: str
    original: tuple[float, float]
    final: tuple[float, float]
    recheck: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"version": STORY_INTEGRITY_VERSION, "status": self.status, "reason": self.reason,
                "original": [round(v, 3) for v in self.original], "final": [round(v, 3) for v in self.final],
                "checks": self.checks, "recheck": self.recheck}


def _primary_money_moment(clip: Mapping[str, Any],
                          money_moments: Sequence[Mapping[str, Any]] = ()) -> Mapping[str, Any] | None:
    wanted = str(clip.get("primary_moment_id", "")).strip()
    if not wanted:
        return None
    for row in money_moments:
        if isinstance(row, Mapping) and str(row.get("moment_id", "")).strip() == wanted:
            a, b = _f(row.get("start"), -1.0), _f(row.get("end"), -1.0)
            if a >= 0 and b > a:
                return row
    return None


def _payoff(clip: Mapping[str, Any], money_moments: Sequence[Mapping[str, Any]] = ()
            ) -> tuple[float, float] | None:
    primary = _primary_money_moment(clip, money_moments)
    if primary is not None:
        return _f(primary.get("start")), _f(primary.get("end"))
    a, b = _f(clip.get("payoff_start"), -1.0), _f(clip.get("payoff_end"), -1.0)
    if a >= 0 and b > a:
        return a, b
    anchor = clip.get("strongest_anchor") if isinstance(clip.get("strongest_anchor"), Mapping) else {}
    a, b = _f(anchor.get("start"), -1.0), _f(anchor.get("end"), -1.0)
    return (a, b) if a >= 0 and b > a else None


def _protected(clip: Mapping[str, Any]) -> list[tuple[float, float]]:
    rows = []
    for item in clip.get("must_keep_ranges", []) or []:
        if isinstance(item, Mapping) and _f(item.get("end")) > _f(item.get("start")):
            rows.append((_f(item.get("start")), _f(item.get("end"))))
    return rows


def assess(clip: Mapping[str, Any], evidence: Evidence, *, source_duration: float, max_duration: float,
           money_moments: Sequence[Mapping[str, Any]] = ()
           ) -> tuple[list[dict[str, Any]], float, float, list[dict[str, Any]]]:
    """(checks, wanted_start, wanted_end, protections to add). Pure: nothing is changed here."""
    start, end = _f(clip.get("start")), _f(clip.get("end"))
    checks: list[dict[str, Any]] = []
    want_start, want_end = start, end
    protect: list[dict[str, Any]] = []
    floor = max(0.0, start - MAX_SETUP_REACH_S)
    ceiling = min(source_duration or end + MAX_REACTION_REACH_S, end + MAX_REACTION_REACH_S)

    def check(name: str, status: str, detail: str) -> None:
        checks.append({"check": name, "status": status, "detail": detail})

    payoff = _payoff(clip, money_moments)
    if payoff is None:
        check("payoff_present", "warn", "the candidate records no payoff; nothing to verify")
    else:
        pa, pb = payoff
        inside = max(0.0, min(end, pb) - max(start, pa)) / max(1e-6, pb - pa)
        if inside < 0.5:
            check("payoff_present", "fail", f"claimed payoff {pa:.2f}-{pb:.2f}s lies outside {start:.2f}-{end:.2f}s")
        elif pa < start - PAYOFF_TOLERANCE_S or pb > end + PAYOFF_TOLERANCE_S:
            want_start, want_end = min(want_start, pa), max(want_end, pb)
            check("payoff_present", "repair", f"payoff {pa:.2f}-{pb:.2f}s is cut by a boundary")
        else:
            check("payoff_present", "pass", f"payoff {pa:.2f}-{pb:.2f}s inside the story")

    for a, b in _protected(clip):
        if (a < start - 0.05 < b) or (a < end + 0.05 < b):
            want_start, want_end = min(want_start, a), max(want_end, b)
            check("protected_ranges", "repair", f"protected {a:.2f}-{b:.2f}s crosses a boundary")

    pa, pb = payoff if payoff else (start, start)
    # SETUP: the cause must be in the story (speech or a visual event before the payoff).
    if evidence.cut_through(start):
        phrase = evidence.phrase_start_before(start + 1e-3, max(0.0, start - MAX_UTTERANCE_REACH_S))
        if phrase is not None and phrase < want_start:
            want_start = phrase
            check("setup", "repair", f"start {start:.2f}s cuts a running utterance; phrase starts at {phrase:.2f}s")
        else:
            check("setup", "warn", f"start {start:.2f}s cuts an utterance whose start is not reachable")
    elif payoff and not evidence.words_in(start, pa) and not evidence.visual_in(start, pa):
        previous = evidence.previous_phrase(start, floor)
        if previous is not None:
            want_start = min(want_start, previous[0])
            protect.append({"start": round(previous[0], 3), "end": round(min(previous[1], pa), 3),
                            "reason": "causal setup restored by story integrity", "kind": "causal_setup"})
            check("setup", "repair", f"nothing precedes the payoff; the adjacent phrase {previous[0]:.2f}-"
                                     f"{previous[1]:.2f}s is its setup")
        else:
            check("setup", "warn", "no speech or visual setup before the payoff is provable (may be visual-only)")
    else:
        check("setup", "pass", "setup evidence precedes the payoff")

    # REACTION: a consequence after the payoff (a very short one is fine).
    if evidence.cut_through(end):
        phrase = evidence.phrase_end_after(end - 1e-3, min(ceiling, end + MAX_UTTERANCE_REACH_S))
        if phrase is not None and phrase > want_end:
            want_end = phrase
            check("reaction", "repair", f"end {end:.2f}s cuts a running utterance; phrase ends at {phrase:.2f}s")
        else:
            check("reaction", "warn", f"end {end:.2f}s cuts an utterance whose end is not reachable")
    elif payoff and not evidence.words_in(pb, end) and not evidence.visual_in(pb, end) \
            and not any(_f(a.get("start")) >= pb - 0.05 for a in clip.get("anchor_moments", []) or []
                        if isinstance(a, Mapping) and str(a.get("type", "")) == "reaction"):
        following = evidence.next_phrase(end, ceiling)
        if following is not None:
            want_end = max(want_end, following[1])
            protect.append({"start": round(max(following[0], pb), 3), "end": round(following[1], 3),
                            "reason": "reaction restored by story integrity", "kind": "causal_reaction"})
            check("reaction", "repair", f"nothing follows the payoff; the adjacent phrase {following[0]:.2f}-"
                                        f"{following[1]:.2f}s is its reaction")
        else:
            check("reaction", "warn", "no reaction after the payoff is provable (the story may end on the payoff)")
    else:
        check("reaction", "pass", "a reaction/consequence follows the payoff")

    # Never beyond the analyzer's maximum: keep payoff/protected coverage first, drop the optional reach.
    if want_end - want_start > max_duration + 1e-6:
        for name in ("reaction", "setup"):
            if want_end - want_start <= max_duration + 1e-6:
                break
            row = next((c for c in checks if c["check"] == name and c["status"] == "repair"), None)
            if row is None:
                continue
            if name == "reaction":
                want_end = max(end, *(b for _a, b in ([payoff] if payoff else [])), *(b for _a, b in _protected(clip)))
                protect = [p for p in protect if p["kind"] != "causal_reaction"]
            else:
                want_start = min(start, *(a for a, _b in ([payoff] if payoff else [])),
                                 *(a for a, _b in _protected(clip)))
                protect = [p for p in protect if p["kind"] != "causal_setup"]
            row.update(status="warn", detail=row["detail"] + f"; not applied (would exceed {max_duration:.0f}s)")
    # Expansion only: a boundary is never moved inward, and only an expansion is clamped to the source.
    final_start = min(start, max(0.0, want_start))
    final_end = max(end, min(source_duration, want_end) if source_duration > 0 else want_end)
    return checks, final_start, final_end, protect[:MAX_ADDED_PROTECTIONS]


def review(clip: Mapping[str, Any], evidence: Evidence, *, source_duration: float, max_duration: float,
           money_moments: Sequence[Mapping[str, Any]] = ()) -> IntegrityResult:
    start, end = _f(clip.get("start")), _f(clip.get("end"))
    checks, want_start, want_end, protect = assess(
        clip, evidence, source_duration=source_duration, max_duration=max_duration,
        money_moments=money_moments)
    failed = [c for c in checks if c["status"] == "fail"]
    if failed:
        reason = "; ".join(f"{c['check']}: {c['detail']}" for c in failed)
        return IntegrityResult("reselect", dict(clip), checks, reason, (start, end), (start, end))
    repaired = dict(clip)
    status = "pass"
    if want_start < start - 1e-6 or want_end > end + 1e-6 or protect:
        repaired = copy.deepcopy(dict(clip))
        repaired["start"], repaired["end"] = round(want_start, 3), round(want_end, 3)
        repaired["duration"] = round(want_end - want_start, 3)
        repaired["must_keep_ranges"] = [*list(clip.get("must_keep_ranges", []) or []), *protect]
        repaired["story_integrity"] = {"expanded_from": [round(start, 3), round(end, 3)]}
        status = "repaired"
    # Re-check the repaired story once (no second repair round).
    recheck, *_ = assess(repaired, evidence, source_duration=source_duration, max_duration=max_duration,
                         money_moments=money_moments)
    if status == "pass" and any(c["status"] == "warn" for c in checks):
        status = "warn"
    reasons = [f"{c['check']}: {c['detail']}" for c in checks if c["status"] != "pass"]
    return IntegrityResult(status, repaired, checks, "; ".join(reasons) or "causally understandable as selected",
                           (start, end), (_f(repaired.get("start")), _f(repaired.get("end"))), recheck)


def review_selection(clips: Sequence[Any], selected_index: int, *, evidence: Evidence, source_duration: float,
                     max_duration: float, rank: Sequence[int], user_pinned: bool,
                     money_moments: Sequence[Mapping[str, Any]] = ()
                     ) -> tuple[int, IntegrityResult, list[dict[str, Any]]]:
    """(index to use, its result, alternates considered). ``rank`` = candidate positions in selection order."""
    first = review(clips[selected_index - 1], evidence, source_duration=source_duration, max_duration=max_duration,
                   money_moments=money_moments)
    if first.status != "reselect" or user_pinned:
        if first.status == "reselect":
            first.status, first.reason = "warn", first.reason + " (clip pinned by the user: kept)"
        return selected_index, first, []
    considered: list[dict[str, Any]] = []
    for position in rank:
        if position == selected_index or not isinstance(clips[position - 1], Mapping):
            continue
        result = review(clips[position - 1], evidence, source_duration=source_duration, max_duration=max_duration,
                        money_moments=money_moments)
        considered.append({"clip_index": position, "status": result.status})
        if result.status != "reselect":
            result.reason = (f"reselected from clip {selected_index} ({first.reason}); " + result.reason)
            result.status = "reselected" if result.status == "pass" else f"reselected+{result.status}"
            return position, result, considered
    first.status, first.reason = "warn", first.reason + " (no alternate candidate keeps its payoff: kept)"
    return selected_index, first, considered
