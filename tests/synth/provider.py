"""ScriptedProvider: deterministic stand-in for every model role in regression tests.

ASR ears hear the scenario's ground-truth words (optionally mis-hearing
scripted words, unless the ear's prompt carries the verified spelling);
editorial roles answer from the scenario's script by parsing the SAME input
text the production prompts contain. It exercises the real production
pipeline end to end without network access; it is not a production component.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from mimir.config import ModelRoute
from mimir.models.provider import AudioMeta, DiarizedSegment, ImageInput, ModelProvider, TimedWord, Transcription
from tests.synth.scenarios import DEFAULT_DIRECTOR, Scenario


def anchor_range(scenario: Scenario, truth: dict[str, Any], anchor: str) -> tuple[float, float]:
    kind, _, value = anchor.partition(":")
    if kind == "line":
        line = truth["lines"][int(value)]
        return line["start"], line["end"]
    if kind == "lines":
        a, b = (int(v) for v in value.split("-"))
        return truth["lines"][a]["start"], truth["lines"][b]["end"]
    if kind == "event":
        t = truth["events"][int(value)]["t"]
        return t - 0.1, t + 1.2
    raise ValueError(anchor)


class ScriptedProvider(ModelProvider):
    name = "scripted"

    def __init__(self, scenario: Scenario, truth: dict[str, Any]) -> None:
        self.scenario = scenario
        self.truth = truth
        self.calls: Counter[str] = Counter()
        self.inputs: dict[str, list[str]] = {}
        self.review_requests: list[dict[str, Any]] = []
        self.review_script: list[Any] = []

    # ------------------------------------------------------------------ ASR

    def _words(self, a: float, b: float) -> list[dict[str, Any]]:
        return [w for w in self.truth["words"] if a <= (w["start"] + w["end"]) / 2 < b]

    def _heard(self, role: str, view: str, token: str, prompt: str | None) -> str:
        mapping = {**self.scenario.asr.get(role, {}), **self.scenario.asr.get(f"{role}:{view}", {})}
        heard = mapping.get(token, token)
        if heard != token and prompt and "VERIFIED" in prompt:
            core = re.sub(r"[^\w]", "", token)
            if core and re.search(rf"\b{re.escape(core)}\b", prompt.split("VERIFIED", 1)[1]):
                return token  # the verified spelling was part of this ear's prompt
        return heard

    def transcribe(self, role, route, audio, *, language, prompt=None, keywords=(), word_timestamps=False,
                   logprobs=False, meta=AudioMeta()):
        self.calls[role] += 1
        words = self._words(meta.source_start, meta.source_end)
        heard = [self._heard(role, meta.view, w["text"], prompt) for w in words]
        timed = tuple(TimedWord(h, round(w["start"] - meta.source_start, 3), round(w["end"] - meta.source_start, 3))
                      for h, w in zip(heard, words)) if word_timestamps else ()
        probs = tuple((h, 0.9) for h in heard) if logprobs else ()
        return Transcription(" ".join(heard), timed, probs)

    def diarize(self, role, route, audio, *, language, known_speakers=(), meta=AudioMeta()):
        self.calls[role] += 1
        words = self._words(meta.source_start, meta.source_end)
        labels: dict[str, str] = {}
        segments: list[DiarizedSegment] = []
        current: list[dict[str, Any]] = []

        def flush() -> None:
            if current:
                key = current[0]["speaker"]
                label = labels.setdefault(key, "ABCDEFG"[len(labels)])
                segments.append(DiarizedSegment(label, round(current[0]["start"] - meta.source_start, 3),
                                                round(current[-1]["end"] - meta.source_start, 3),
                                                " ".join(w["text"] for w in current)))
                current.clear()

        for word in sorted(words, key=lambda w: (w["speaker"], w["start"])):
            if current and (word["speaker"] != current[-1]["speaker"] or word["start"] - current[-1]["end"] > 0.8):
                flush()
            current.append(word)
        flush()
        return sorted(segments, key=lambda s: s.start)

    # ------------------------------------------------------------ editorial

    def json_task(self, role, route, *, instructions, input_text, schema, schema_name, images=()):
        self.calls[role] += 1
        self.inputs.setdefault(role, []).append(input_text)
        handler = getattr(self, f"_{role}")
        return handler(input_text, schema, images)

    def _peak_probe(self, text: str, schema: dict[str, Any], images: Sequence[ImageInput]) -> dict[str, Any]:
        regions = []
        for rid, t in re.findall(r"^(r\d+): measured .* at ([\d.]+)s", text, flags=re.M):
            events = []
            for spec in self.scenario.visual_events:
                a, _ = anchor_range(self.scenario, self.truth, spec["at"])
                if abs(float(t) - (a + 0.1)) <= 1.5:
                    events.append({"type": spec["type"], "description": spec["description"],
                                   "confidence": spec["confidence"]})
            regions.append({"region_id": rid, "events": events})
        return {"regions": regions}

    def _story_scout(self, text: str, schema: dict[str, Any], images) -> dict[str, Any]:
        moments = []
        for moment in self.scenario.moments:
            a, b = anchor_range(self.scenario, self.truth, moment.anchor)
            moments.append({"start": a, "end": b, "strength": moment.strength, "type": moment.type,
                            "source": moment.source, "visual_event_ids": [], "label": moment.label,
                            "why_compelling": moment.label, "context_before_seconds": moment.before,
                            "context_after_seconds": moment.after, "preserve_pause_after": False})
        audit = []
        for event_id, a, b in re.findall(r"^- (\w+) \[([\d.]+)-([\d.]+)\].*REQUIRED REVIEW", text, flags=re.M):
            near = any(abs(m["start"] - float(a)) < 3.0 or abs(m["end"] - float(b)) < 3.0 for m in moments)
            audit.append({"event_id": event_id, "decision": "anchor" if near else "reject",
                          "reason": "part of the payoff" if near else "unrelated loud moment"})
            if near:
                for m in moments:
                    if abs(m["start"] - float(a)) < 3.0:
                        m["visual_event_ids"].append(event_id)
                        m["source"] = "both"
        return {"moments": moments, "visual_event_audit": audit}

    def _moment_ids(self, text: str) -> list[tuple[str, float, float]]:
        return [(mid, float(a), float(b)) for mid, a, b in re.findall(r"^- (m\d+) \[([\d.]+)-([\d.]+)\]", text,
                                                                         flags=re.M)]

    def _story_composer(self, text: str, schema: dict[str, Any], images) -> dict[str, Any]:
        story = self.scenario.story
        lines = self.truth["lines"]
        start = lines[story["start_line"]]["start"] - 0.3
        end = lines[story["end_line"]]["end"] + 0.8
        payoff = anchor_range(self.scenario, self.truth, story["payoff"])
        ids = self._moment_ids(text)
        primary = min(ids, key=lambda row: abs(row[1] - payoff[0]))[0] if ids else "m01"
        covered = [mid for mid, a, b in ids if start <= a and b <= end]
        candidates = [{
            "start": round(start, 3), "end": round(end, 3), "payoff_start": payoff[0], "payoff_end": payoff[1],
            "score": 8.6, "title": story["title"], "hook_text": story["hook"], "hook_type": "payoff_first",
            "emotion": story["emotion"], "reason": "complete causal story around the strongest moment",
            "context": "setup, escalation, payoff and reaction", "primary_moment_id": primary,
            "covered_moment_ids": covered, "coverage_reason": "one causal chain", "length_exception_reason": "",
            "visual_event_ids": [], "caption_highlights": story.get("highlights", []),
            "must_keep_ranges": [{"start": payoff[0], "end": payoff[1], "reason": "exact payoff"}],
            "beats": {"setup": "the situation is introduced", "escalation": "it builds toward the peak",
                      "payoff": "the peak happens", "reaction": "the reaction lands"},
        }]
        others = [row for row in ids if row[0] != primary]
        if others:
            mid, a, b = others[0]
            candidates.append({**candidates[0], "start": round(max(0.0, a - 4.0), 3), "end": round(b + 3.0, 3),
                               "payoff_start": a, "payoff_end": b, "score": 6.2, "title": "secondary beat",
                               "primary_moment_id": mid, "covered_moment_ids": [mid], "must_keep_ranges": [],
                               "length_exception_reason": "short secondary beat"})
        return {"candidates": candidates}

    def _judge(self, text: str) -> dict[str, Any]:
        count = len(re.findall(r"^CANDIDATE \d+:", text, flags=re.M))
        rankings = []
        for index in range(1, count + 1):
            good = index == 1
            score = 8.7 if good else 6.0
            rankings.append({"candidate_index": index, "overall_score": score,
                             **{k: (8.5 if good else 6.0) for k in (
                                 "money_moment_strength", "visual_event_value", "context_completeness",
                                 "payoff_completeness", "story_coverage", "duration_fit", "retention_shape")},
                             "reason": "complete causal chain" if good else "fragment"})
        return {"selected_candidate_index": 1, "rankings": rankings, "selection_reason": "strongest complete story"}

    def _story_judge(self, text: str, schema, images) -> dict[str, Any]:
        return self._judge(text)

    def _story_judge_escalation(self, text: str, schema, images) -> dict[str, Any]:
        return self._judge(text)

    def _story_expansion(self, text: str, schema, images) -> dict[str, Any]:
        match = re.search(r"CANDIDATE 1: \[([\d.]+)-([\d.]+)\].*payoff \[([\d.]+)-([\d.]+)\]", text)
        a, b, pa, pb = (float(v) for v in match.groups())
        ids = [row[0] for row in self._moment_ids(text)]
        return {"start": a, "end": b, "payoff_start": pa, "payoff_end": pb, "covered_moment_ids": [],
                "moment_coverage_audit": [{"moment_id": m, "decision": "omit", "reason": "unrelated"} for m in ids],
                "must_keep_ranges": [], "coverage_reason": "already complete", "length_exception_reason": ""}

    def _boundary_polish(self, text: str, schema, images) -> dict[str, Any]:
        a, b = (float(v) for v in re.search(r"CLIP: \[([\d.]+)-([\d.]+)\]", text).groups())
        return {"start_action": "keep", "end_action": "keep", "proposed_start": a, "proposed_end": b,
                "start_reason": "clean start", "end_reason": "complete reaction"}

    def _visual_observer(self, text: str, schema, images: Sequence[ImageInput]) -> dict[str, Any]:
        frames = []
        for image in images:
            match = re.match(r"frame (f\d+) at source ([\d.]+)s", image.label)
            if not match:
                continue
            fid, t = match.group(1), float(match.group(2))
            elements = self.scenario.observer(t) if self.scenario.observer else []
            frames.append({"frame_id": fid, "summary": f"scene at {t:.1f}s", "elements": elements})
        return {"frames": frames}

    def _cold_open(self, text: str, schema, images) -> dict[str, Any]:
        payoff = anchor_range(self.scenario, self.truth, self.scenario.story["payoff"])
        peaks = [(pid, float(a), float(b), float(c)) for pid, a, b, c in
                 re.findall(r"^PEAK (p\d+): core \[([\d.]+)-([\d.]+)\] combined=([\d.]+)", text, flags=re.M)]
        overlapping = [p for p in peaks if p[1] <= payoff[1] + 0.5 and p[2] >= payoff[0] - 0.5]
        chosen = max(overlapping or peaks, key=lambda p: (p[3], p[1]))
        return {"peak_id": chosen[0], "selection_mode": "peak_window", "start_word_id": "", "end_word_id": "",
                "reason": "strongest multimodal peak", "viewer_question": "how did this happen?",
                "spoiler_risk": "medium",
                "hooks": [{"text": h, "curiosity_target": "what caused the peak"} for h in self.scenario.hooks]}

    def _hook_judge(self, text: str, schema, images) -> dict[str, Any]:
        count = len(re.findall(r"^\d+\. ", text, flags=re.M))
        return {"scores": [{"index": i, "score": 8.6 if i == 1 else 7.0, "reason": "specific"}
                           for i in range(1, count + 1)], "selected_index": 1 if count else 0,
                "rejection_reason": ""}

    def _edit_director(self, text: str, schema, images) -> dict[str, Any]:
        prefs = {**DEFAULT_DIRECTOR, **self.scenario.director}
        decisions = []
        blocks = re.split(r"\n(?=s\d+ \[)", text.split("SPANS:\n", 1)[1])
        for block in blocks:
            head = re.match(r"(s\d+) \[.*?\] (\w+)/(\w+)", block)
            allowed = re.search(r"ALLOWED: (.*)", block)
            if not head or not allowed:
                continue
            span_id, segment, role = head.groups()
            options = [a.strip() for a in allowed.group(1).split(",")]
            wanted = prefs.get(role, ["WIDE_CONTEXT"])
            if self.scenario.extra.get("misbehave") and role == "payoff":
                intent = "SPEAKER_PUNCH"  # deliberately ignores the evidence: validation must correct it
            else:
                intent = next((w for w in wanted if w in options), "WIDE_CONTEXT")
            decisions.append({"span_id": span_id, "intent": intent, "target_id": "", "intensity": "normal",
                              "reason": f"{role}: {intent.lower()}"})
        return {"decisions": decisions, "notes": "scripted director"}

    def _effects(self, text: str, schema, images) -> dict[str, Any]:
        accent = self.scenario.accent
        if not accent:
            return {"use_accent": False, "candidates": [], "reason": "no expectation break"}
        line = self.truth["lines"][accent["line"]]
        word_id = line["words"][accent["word"]]
        target = next(w for w in self.truth["words"] if w["id"] == word_id)
        rows = re.findall(r"^(c\d+@story) \[([\d.]+)\] (\S+)", text, flags=re.M)
        key = next((k for k, _t, token in rows if token == target["text"]), None)
        if key is None:
            return {"use_accent": False, "candidates": [], "reason": "anchor not in main"}
        return {"use_accent": True, "reason": "real expectation break", "candidates": [{
            "anchor_word_key": key, "placement": "after_word", "category": accent["category"], "kind": "audio",
            "strength": "subtle", "unexpectedness": 0.82, "absurdity": 0.6, "reversal": 0.5, "fit_score": 7.9,
            "confidence": 0.8, "reason": "disbelief punctuation"}]}

    def _final_reviewer(self, text: str, schema, images) -> dict[str, Any]:
        """Checks the request is a real source-vs-render comparison; verdicts come from ``review_script``
        (one callable per QC round, fed the frame ids and evidence lines) or pass by default."""
        frame_ids = schema["properties"]["issues"]["items"]["properties"]["frame_id"]["enum"]
        labels = [image.label for image in images]
        assert labels == [f"{fid} {kind}" for fid in frame_ids for kind in ("SOURCE", "RENDERED")], labels
        evidence = {line.split(" | ")[0].split("=", 1)[1]: line for line in text.splitlines()
                    if line.startswith("frame_id=")}
        assert set(evidence) == set(frame_ids)
        self.review_requests.append({"frame_ids": frame_ids, "evidence": evidence, "images": len(images)})
        script = self.review_script[len(self.review_requests) - 1] if len(self.review_requests) <= len(
            self.review_script) else None
        return script(frame_ids, evidence) if script else {"verdict": "pass", "issues": []}
