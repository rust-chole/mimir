"""Stage ``story``: money moments -> candidates -> final judgement -> complete StoryPackage.

Model calls (routed): scout (+ at most one visual-audit repair), composer (+ at
most one repair when no candidate validates), judge (+ one escalation when the
decision is close or ignores a decisive visual payoff), story expansion (only
when the story is short or judged incomplete) and local boundary polish.
Everything a model returns is validated deterministically; the causal-flow
rules then guarantee setup -> escalation -> payoff -> reaction coverage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from mimir.config import Settings, routes_for, section
from mimir.core.stage import StageContext, StageOutput
from mimir.errors import NoStoryError
from mimir.media.probe import MediaInfo
from mimir.story import prompts as P
from mimir.story.causal import (
    align_to_speech,
    clamp,
    enforce_causal_flow,
    moment_map,
    story_segments,
    validate_candidate,
    validate_moments,
)
from mimir.story.package import build_package
from mimir.transcript.segments import timestamped_lines

DECISIVE_TYPES = {"object_break", "destruction", "impact", "crash", "fall", "explosion", "visual_payoff"}
DECISIVE_MIN_CONFIDENCE = 0.68
MAX_INJECTIONS = 4
EXPANSION_GOOD_METRIC = 7.6


def visual_evidence_text(events: Sequence[dict[str, Any]]) -> str:
    if not events:
        return "FACTUAL VISUAL EVENTS: none reported."
    lines = ["FACTUAL VISUAL EVENTS (observer facts, not editorial judgement):"]
    for event in events:
        flag = " [REQUIRED REVIEW]" if event.get("required_review") else ""
        lines.append(f"- {event['event_id']} [{event['start']:.2f}-{event['end']:.2f}] {event['type']} "
                     f"(confidence {event['confidence']:.2f}){flag}: {event['description']}")
    return "\n".join(lines)


def moments_text(moments: Sequence[dict[str, Any]]) -> str:
    return "\n".join(f"- {m['moment_id']} [{m['start']:.2f}-{m['end']:.2f}] {m['type']} strength={m['strength']} "
                     f"source={m['source']} before={m['context_before_seconds']} after={m['context_after_seconds']}: "
                     f"{m['label']} | {m['why_compelling']}" for m in moments)


def candidates_text(candidates: Sequence[dict[str, Any]]) -> str:
    rows = []
    for index, c in enumerate(candidates, start=1):
        rows.append(f"CANDIDATE {index}: [{c['start']:.2f}-{c['end']:.2f}] ({c['duration']:.1f}s) payoff "
                    f"[{c['payoff_start']:.2f}-{c['payoff_end']:.2f}] primary={c['primary_moment_id']} covers="
                    f"{','.join(c['covered_moment_ids'])} title={c['title']!r}\n  reason: {c['reason']}\n"
                    f"  beats: {json.dumps(c['beat_notes'], ensure_ascii=False)}")
    return "\n".join(rows)


def excerpt(segments: Sequence[dict[str, Any]], start: float, end: float) -> str:
    return timestamped_lines([s for s in segments if s["end"] >= start and s["start"] <= end])


class StoryStage:
    name = "story"
    version = 1
    deps = ("source", "probe", "transcript", "vod_evidence")

    def params(self, settings: Settings) -> Any:
        return {"story": section(settings, "story"),
                "routes": routes_for(settings, "story_scout", "story_composer", "story_judge",
                                     "story_judge_escalation", "story_expansion", "boundary_polish")}

    # ------------------------------------------------------------------ run

    def run(self, ctx: StageContext) -> StageOutput:
        info = MediaInfo.from_dict(ctx.dep("probe").json("media"))
        transcript = ctx.dep("transcript").json("transcript")
        evidence = ctx.dep("vod_evidence").json("vod_evidence")
        events = list(evidence["visual_events"])
        duration = info.duration
        segments = story_segments(transcript)
        transcript_text = timestamped_lines(transcript["segments"])
        settings = ctx.settings.story

        scout = self._scout(ctx, transcript_text, events, duration)
        moments = validate_moments(scout.get("moments", []), duration, settings.max_money_moments)
        audit = {row["event_id"]: row for row in scout.get("visual_event_audit", []) if isinstance(row, dict)}
        required = {e["event_id"] for e in events if e.get("required_review")}
        missing = sorted(required - set(audit))
        if missing:
            repair = self._scout(ctx, transcript_text, events, duration, missing=missing, moments=moments)
            moments = validate_moments([*moments, *repair.get("moments", [])], duration, settings.max_money_moments)
            audit.update({row["event_id"]: row for row in repair.get("visual_event_audit", [])})
            still = sorted(required - set(audit))
            if still:
                ctx.ledger.warning("visual_audit_incomplete", "scout did not audit required events after one repair; "
                                   "they are treated as secondary evidence", events=still)
                for event_id in still:
                    audit[event_id] = {"event_id": event_id, "decision": "secondary",
                                       "reason": "not audited by the scout (recorded)"}
        moments, injected = self._inject_decisive(moments, events, duration, settings.max_money_moments + MAX_INJECTIONS)
        if injected:
            ctx.ledger.info("decisive_visual_candidates", "decisive visual events reach the composer", events=injected)
        if not moments:
            raise NoStoryError(self.name, "no money moment was found in the VOD")

        candidates = self._compose(ctx, transcript_text, events, moments, duration)
        candidates = self._ensure_decisive_candidates(candidates, moments, injected, segments, duration, ctx)
        judge = self._judge(ctx, transcript_text, events, moments, candidates, injected)
        order = self._ranked_indices(judge, len(candidates))
        if settings.story_index is not None:
            forced = settings.story_index - 1
            if not 0 <= forced < len(candidates):
                raise NoStoryError(self.name, f"--story-index {settings.story_index} out of range (1..{len(candidates)})")
            order = [forced] + [i for i in order if i != forced]
        metrics_by_index = {int(r["candidate_index"]) - 1: r for r in judge.get("rankings", [])}
        attempts = []
        for rank, index in enumerate(order[:3], start=1):
            package = self._finalize(ctx, candidates[index], moments, transcript, segments, info,
                                     metrics_by_index.get(index, {}), judge, rank)
            attempts.append({"candidate": index + 1, "problems": package["completeness_problems"]})
            if not package["completeness_problems"]:
                return StageOutput(data={"story": package, "discovery": {
                    "moments": moments, "visual_event_audit": list(audit.values()), "candidates": candidates,
                    "judge": judge, "injected_visual_events": injected, "attempts": attempts}})
            ctx.ledger.warning("incomplete_story", f"candidate {index + 1} cannot form a complete causal story",
                               problems=package["completeness_problems"])
        raise NoStoryError(self.name, f"no candidate forms a complete setup->escalation->payoff->reaction story: {attempts}")

    # ------------------------------------------------------------- scouting

    def _scout(self, ctx: StageContext, transcript_text: str, events: list[dict[str, Any]], duration: float, *,
               missing: Sequence[str] = (), moments: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
        text = f"ORIGINAL VOD DURATION: {duration:.2f} seconds\n\nTIMESTAMPED TRANSCRIPT:\n{transcript_text}\n\n" \
               f"{visual_evidence_text(events)}\n\n"
        if missing:
            text += ("REPAIR: these REQUIRED REVIEW events were not audited: " + ", ".join(missing) +
                     ". Audit each of them now (anchor / secondary / reject with a concrete reason) and add money "
                     "moments for any that are genuine payoffs.\nCURRENT MOMENTS:\n" + moments_text(moments))
        else:
            text += "Find the strongest money moments across the whole VOD. Every REQUIRED REVIEW event must appear " \
                    "in visual_event_audit."
        return ctx.provider.json_task("story_scout", ctx.settings.route("story_scout"),
                                      instructions=P.SCOUT_INSTRUCTIONS, input_text=text, schema=P.MOMENT_SCHEMA,
                                      schema_name="mimir_money_moments_v1")

    @staticmethod
    def _inject_decisive(moments: list[dict[str, Any]], events: list[dict[str, Any]], duration: float, limit: int
                         ) -> tuple[list[dict[str, Any]], list[str]]:
        injected: list[str] = []
        extra = []
        for event in sorted(events, key=lambda e: -e["confidence"]):
            if event["type"] not in DECISIVE_TYPES or event["confidence"] < DECISIVE_MIN_CONFIDENCE:
                continue
            covered = any(m["start"] - 0.5 <= (event["start"] + event["end"]) / 2 <= m["end"] + 0.5 for m in moments)
            if covered:
                continue
            extra.append({"start": event["start"], "end": event["end"], "strength": 6.5 + 2.0 * event["confidence"],
                          "type": "destruction" if event["type"] in ("object_break", "destruction") else
                          ("visual_impact" if event["type"] != "visual_payoff" else "physical_payoff"),
                          "source": "visual", "visual_event_ids": [event["event_id"]],
                          "label": event["description"][:120], "why_compelling": "decisive visual event (observer fact)",
                          "context_before_seconds": 6.5, "context_after_seconds": 3.0, "preserve_pause_after": False})
            injected.append(event["event_id"])
            if len(injected) >= MAX_INJECTIONS:
                break
        if not extra:
            return moments, []
        merged = validate_moments([*moments, *extra], duration, limit)
        return merged, injected

    # ------------------------------------------------------------ composing

    def _compose(self, ctx: StageContext, transcript_text: str, events: list[dict[str, Any]],
                 moments: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
        base = (f"ORIGINAL VOD DURATION: {duration:.2f} seconds\n\nMONEY MOMENTS:\n{moments_text(moments)}\n\n"
                f"{visual_evidence_text(events)}\n\nTIMESTAMPED TRANSCRIPT:\n{transcript_text}\n\n"
                f"Build up to {ctx.settings.story.max_candidates} complete candidates.")
        errors: list[str] = []
        for attempt in range(2):
            text = base if not errors else base + "\n\nYOUR PREVIOUS CANDIDATES WERE INVALID:\n- " + "\n- ".join(errors)
            raw = ctx.provider.json_task("story_composer", ctx.settings.route("story_composer"),
                                         instructions=P.COMPOSER_INSTRUCTIONS, input_text=text,
                                         schema=P.CANDIDATES_SCHEMA, schema_name="mimir_story_candidates_v1")
            candidates, errors = [], []
            for item in raw.get("candidates", [])[:ctx.settings.story.max_candidates]:
                clean, error = validate_candidate(item, duration=duration, moments=moments,
                                                  settings=ctx.settings.story)
                if clean is None:
                    errors.append(error)
                else:
                    candidates.append(clean)
            if candidates:
                if errors:
                    ctx.ledger.info("candidates_dropped", "invalid candidates dropped", errors=errors)
                return candidates
        raise NoStoryError(self.name, "no valid story candidate after one repair: " + "; ".join(errors[:5]))

    def _ensure_decisive_candidates(self, candidates: list[dict[str, Any]], moments: list[dict[str, Any]],
                                    injected: list[str], segments: list[dict[str, Any]], duration: float,
                                    ctx: StageContext) -> list[dict[str, Any]]:
        """A decisive visual payoff always gets at least one candidate the judge can compare."""
        for moment in moments:
            if not set(moment["visual_event_ids"]) & set(injected):
                continue
            if any(c["start"] <= moment["start"] and c["end"] >= moment["end"] for c in candidates):
                continue
            start, end = align_to_speech(segments, moment["start"], moment["end"],
                                         max(0.0, moment["start"] - 12.0), min(duration, moment["end"] + 4.0),
                                         duration)
            raw = {"start": start, "end": end, "payoff_start": moment["start"], "payoff_end": moment["end"],
                   "score": moment["strength"], "title": moment["label"][:80], "hook_text": "",
                   "hook_type": "shock", "emotion": "shock", "reason": "decisive visual payoff candidate",
                   "context": "", "primary_moment_id": moment["moment_id"],
                   "covered_moment_ids": [moment["moment_id"]], "coverage_reason": "", "length_exception_reason": "",
                   "visual_event_ids": moment["visual_event_ids"], "caption_highlights": [],
                   "must_keep_ranges": [{"start": moment["start"], "end": moment["end"], "reason": "visual payoff"}],
                   "beats": {"setup": "", "escalation": "", "payoff": moment["label"], "reaction": ""}}
            clean, _ = validate_candidate(raw, duration=duration, moments=moments, settings=ctx.settings.story)
            if clean is not None and len(candidates) < ctx.settings.story.max_candidates + 2:
                candidates.append(clean)
        return candidates

    # ------------------------------------------------------------- judging

    def _judge(self, ctx: StageContext, transcript_text: str, events: list[dict[str, Any]],
               moments: list[dict[str, Any]], candidates: list[dict[str, Any]], injected: list[str]) -> dict[str, Any]:
        text = (f"MONEY MOMENTS:\n{moments_text(moments)}\n\n{visual_evidence_text(events)}\n\n"
                f"CANDIDATES:\n{candidates_text(candidates)}\n\nTIMESTAMPED TRANSCRIPT:\n{transcript_text}")
        judge = ctx.provider.json_task("story_judge", ctx.settings.route("story_judge"),
                                       instructions=P.JUDGE_INSTRUCTIONS, input_text=text, schema=P.JUDGE_SCHEMA,
                                       schema_name="mimir_story_judge_v1")
        judge = self._clean_judge(judge, len(candidates))
        margin = self._margin(judge)
        selected = candidates[judge["selected_candidate_index"] - 1]
        ignored = [m["moment_id"] for m in moments if set(m["visual_event_ids"]) & set(injected)
                   and not (selected["start"] <= m["start"] and selected["end"] >= m["end"])]
        escalated = False
        if len(candidates) > 1 and (margin is None or margin <= ctx.settings.story.judge_escalation_margin or ignored):
            judge = self._clean_judge(ctx.provider.json_task(
                "story_judge_escalation", ctx.settings.route("story_judge_escalation"),
                instructions=P.JUDGE_INSTRUCTIONS, input_text=text, schema=P.JUDGE_SCHEMA,
                schema_name="mimir_story_judge_v1"), len(candidates))
            escalated = True
        judge["escalated"] = escalated
        judge["margin"] = self._margin(judge)
        return judge

    @staticmethod
    def _clean_judge(judge: dict[str, Any], count: int) -> dict[str, Any]:
        index = int(judge.get("selected_candidate_index", 1) or 1)
        rankings = [r for r in judge.get("rankings", []) if 1 <= int(r.get("candidate_index", 0)) <= count]
        if not 1 <= index <= count:
            index = max(rankings, key=lambda r: r["overall_score"])["candidate_index"] if rankings else 1
        return {**judge, "selected_candidate_index": int(index), "rankings": rankings}

    @staticmethod
    def _margin(judge: dict[str, Any]) -> float | None:
        scores = sorted((float(r["overall_score"]) for r in judge.get("rankings", [])), reverse=True)
        return round(scores[0] - scores[1], 3) if len(scores) >= 2 else None

    @staticmethod
    def _ranked_indices(judge: dict[str, Any], count: int) -> list[int]:
        selected = judge["selected_candidate_index"] - 1
        by_score = sorted(judge.get("rankings", []), key=lambda r: -float(r["overall_score"]))
        order = [selected] + [int(r["candidate_index"]) - 1 for r in by_score if int(r["candidate_index"]) - 1 != selected]
        return order + [i for i in range(count) if i not in order]

    # ---------------------------------------------------------- finalizing

    def _finalize(self, ctx: StageContext, candidate: dict[str, Any], moments: list[dict[str, Any]],
                  transcript: dict[str, Any], segments: list[dict[str, Any]], info: MediaInfo,
                  metrics: dict[str, Any], judge: dict[str, Any], rank: int) -> dict[str, Any]:
        settings = ctx.settings.story
        metric_values = {k: float(metrics.get(k, 10.0)) for k in P.JUDGE_METRICS}
        expansion: dict[str, Any] = {"applied": False}
        needs = candidate["duration"] < settings.target_min or any(
            metric_values[k] < EXPANSION_GOOD_METRIC for k in ("context_completeness", "payoff_completeness",
                                                                "story_coverage"))
        if needs and info.duration > settings.short_exception_below:
            candidate, expansion = self._expand(ctx, candidate, moments, segments, info.duration)
        flow = enforce_causal_flow(candidate, moments=moments, transcript=transcript, duration=info.duration,
                                   video=ctx.source.path, width=info.width, height=info.height,
                                   judge_metrics=metric_values, settings=settings)
        candidate = flow.candidate
        candidate, boundary = self._polish(ctx, candidate, segments, info.duration)
        return build_package(candidate, moments, judge={**{k: v for k, v in metrics.items()}, "expansion": expansion,
                                                        "selection_reason": judge.get("selection_reason", ""),
                                                        "escalated": judge.get("escalated", False)},
                             flow=flow.report, boundary=boundary, rank=rank)

    def _expand(self, ctx: StageContext, candidate: dict[str, Any], moments: list[dict[str, Any]],
                segments: list[dict[str, Any]], duration: float) -> tuple[dict[str, Any], dict[str, Any]]:
        window_start, window_end = max(0.0, candidate["start"] - 30.0), min(duration, candidate["end"] + 20.0)
        text = (f"SELECTED CANDIDATE:\n{candidates_text([candidate])}\n\nMONEY MOMENTS:\n{moments_text(moments)}\n\n"
                f"LOCAL TRANSCRIPT [{window_start:.2f}-{window_end:.2f}]:\n{excerpt(segments, window_start, window_end)}")
        raw = ctx.provider.json_task("story_expansion", ctx.settings.route("story_expansion"),
                                     instructions=P.EXPANSION_INSTRUCTIONS, input_text=text,
                                     schema=P.EXPANSION_SCHEMA, schema_name="mimir_story_expansion_v1")
        proposal = {**candidate, **{k: raw[k] for k in ("start", "end", "payoff_start", "payoff_end",
                                                        "covered_moment_ids", "coverage_reason",
                                                        "length_exception_reason") if k in raw},
                    "must_keep_ranges": [*candidate["must_keep_ranges"], *raw.get("must_keep_ranges", [])],
                    "beats": candidate.get("beat_notes", {})}
        proposal["start"] = min(float(proposal["start"]), candidate["payoff_start"])  # never drop the core
        proposal["end"] = max(float(proposal["end"]), candidate["payoff_end"])
        clean, error = validate_candidate(proposal, duration=duration, moments=moments, settings=ctx.settings.story)
        if clean is None:
            ctx.ledger.info("expansion_rejected", f"story expansion proposal invalid: {error}")
            return candidate, {"applied": False, "rejected": error}
        clean["beat_notes"] = candidate["beat_notes"]
        return clean, {"applied": True, "from": [candidate["start"], candidate["end"]], "to": [clean["start"], clean["end"]],
                       "audit": raw.get("moment_coverage_audit", [])}

    def _polish(self, ctx: StageContext, candidate: dict[str, Any], segments: list[dict[str, Any]], duration: float
                ) -> tuple[dict[str, Any], dict[str, Any]]:
        settings = ctx.settings.story
        start, end = candidate["start"], candidate["end"]
        text = (f"CLIP: [{start:.2f}-{end:.2f}]\nLIMITS: extend up to {settings.boundary_max_extend:.1f}s, trim up to "
                f"{settings.boundary_max_trim:.1f}s per edge.\nPROTECTED RANGES: "
                + "; ".join(f"[{r['start']:.2f}-{r['end']:.2f}] {r['reason']}" for r in candidate["must_keep_ranges"])
                + f"\n\nSTART REGION:\n{excerpt(segments, start - 8.0, start + 8.0)}\n\n"
                  f"END REGION:\n{excerpt(segments, end - 8.0, end + 8.0)}")
        raw = ctx.provider.json_task("boundary_polish", ctx.settings.route("boundary_polish"),
                                     instructions=P.BOUNDARY_INSTRUCTIONS, input_text=text, schema=P.BOUNDARY_SCHEMA,
                                     schema_name="mimir_boundary_polish_v1")
        protected_start = min(r["start"] for r in candidate["must_keep_ranges"])
        protected_end = max(r["end"] for r in candidate["must_keep_ranges"])
        new_start, new_end = start, end
        if raw.get("start_action") in ("extend", "trim"):
            new_start = clamp(float(raw.get("proposed_start", start)), start - settings.boundary_max_extend,
                              start + settings.boundary_max_trim)
            new_start = clamp(new_start, 0.0, min(protected_start, candidate["payoff_start"]))
        if raw.get("end_action") in ("extend", "trim"):
            new_end = clamp(float(raw.get("proposed_end", end)), end - settings.boundary_max_trim,
                            end + settings.boundary_max_extend)
            new_end = clamp(new_end, max(protected_end, candidate["payoff_end"]), duration)
        if new_end - new_start > settings.max_duration or new_end - new_start < settings.min_duration:
            return candidate, {"applied": False, "reason": "proposal violates duration limits", "raw": raw}
        result = dict(candidate)
        result["start"], result["end"] = round(new_start, 3), round(new_end, 3)
        result["duration"] = round(new_end - new_start, 3)
        return result, {"applied": (new_start, new_end) != (start, end), "from": [start, end],
                        "to": [result["start"], result["end"]], "reasons": [raw.get("start_reason", ""),
                                                                            raw.get("end_reason", "")]}
