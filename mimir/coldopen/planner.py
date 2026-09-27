"""Stage ``cold_open``: choose the peak, compute its window, write (and judge) the hook line.

The cold open is mandatory and is always the story's real peak (a supplied
candidate - never invented timestamps). The hook text is optional: a line the
judge scores below the acceptance bar is not rendered (NO HOOK IS BETTER THAN
A BAD HOOK); the cold open itself still plays.
"""
from __future__ import annotations

from typing import Any, Sequence

from mimir.config import Settings, routes_for, section
from mimir.core.stage import StageContext, StageOutput
from mimir.coldopen.peaks import build_candidates, compute_window, story_audio_peaks
from mimir.errors import StageError

BANNED_PHRASES = ("YOU WON'T BELIEVE", "YOU WONT BELIEVE", "WAIT FOR IT", "WATCH TILL THE END",
                  "WATCH UNTIL THE END", "KEEP WATCHING", "THIS IS CRAZY")

CHOICE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "peak_id": {"type": "string"},
        "selection_mode": {"type": "string", "enum": ["peak_window", "spoken_phrase"]},
        "start_word_id": {"type": "string"},
        "end_word_id": {"type": "string"},
        "reason": {"type": "string"},
        "viewer_question": {"type": "string"},
        "spoiler_risk": {"type": "string", "enum": ["low", "medium", "high"]},
        "hooks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"text": {"type": "string"}, "curiosity_target": {"type": "string"}},
                "required": ["text", "curiosity_target"],
            },
        },
    },
    "required": ["peak_id", "selection_mode", "start_word_id", "end_word_id", "reason", "viewer_question",
                 "spoiler_risk", "hooks"],
}

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"index": {"type": "integer"}, "score": {"type": "number"}, "reason": {"type": "string"}},
                "required": ["index", "score", "reason"],
            },
        },
        "selected_index": {"type": "integer"},
        "rejection_reason": {"type": "string"},
    },
    "required": ["scores", "selected_index", "rejection_reason"],
}

CHOICE_INSTRUCTIONS = """
You are a world-class short-form editor choosing the COLD OPEN of an already selected story. The final Short is:

    [COLD OPEN: the strongest peak, a few seconds]  ->  hard restart  ->  [MAIN STORY from its real beginning]

The peak appears first and again later when the story reaches it; that repetition is intentional.
Choose the strongest PEAK from the supplied multimodal candidates (never invent timestamps). Prefer a moment that
stops the scroll and makes the viewer ask how it happened: extreme reaction, disbelief, sudden impact, absurd claim,
conflict. Do not just pick the loudest line; profanity alone is not a hook. When two peaks are similar prefer the
LATER one. Avoid revealing the entire outcome when a partial reveal creates the same stopping power.
selection_mode=spoken_phrase when a complete spoken line is the reason it works (give its first/last word ids),
otherwise peak_window with empty word ids.
Then write up to 3 hook lines (2-7 words, specific to this clip, natural short-form English, no clickbait such as
"you won't believe", no spoiler of the payoff). Each hook names a concrete curiosity_target.
""".strip()

JUDGE_INSTRUCTIONS = """
You are the editor-in-chief and final gate for the cold-open hook line. NO HOOK IS BETTER THAN A BAD HOOK.
Score each candidate 0-10 for specificity to this clip, curiosity gap, pairing with the chosen peak, spoiler safety
and natural wording. Only a candidate scoring at least 8.0 may be selected; otherwise selected_index=0 and give an
actionable rejection_reason. Indices are 1-based.
""".strip()


def _words_text(words: Sequence[dict[str, Any]], start: float, end: float) -> str:
    return " ".join(f"[{w['id']}] {w['text']}" for w in words if w["end"] > start and w["start"] < end)


def clean_hook(text: str, cfg) -> tuple[str, str]:
    value = " ".join(str(text).replace("“", "").replace("”", "").replace('"', "").split())
    words = value.split()
    if not cfg.hook_min_words <= len(words) <= cfg.hook_max_words:
        return "", f"{len(words)} words (allowed {cfg.hook_min_words}-{cfg.hook_max_words})"
    if len(value) > cfg.hook_max_chars:
        return "", f"{len(value)} characters (max {cfg.hook_max_chars})"
    if any(phrase in value.upper().replace("’", "'") for phrase in BANNED_PHRASES):
        return "", "generic clickbait phrase"
    return value, ""


class ColdOpenStage:
    name = "cold_open"
    version = 1
    deps = ("source", "story", "caption_truth", "vision")

    def params(self, settings: Settings) -> Any:
        return {"cold_open": section(settings, "cold_open"), "routes": routes_for(settings, "cold_open", "hook_judge")}

    def run(self, ctx: StageContext) -> StageOutput:
        story = ctx.dep("story").json("story")
        truth = ctx.dep("caption_truth").json("caption_truth")
        vision = ctx.dep("vision").json("vision")
        cfg = ctx.settings.cold_open
        words = truth["words"]
        audio = story_audio_peaks(str(ctx.source.path), float(story["start"]), float(story["end"]))
        candidates = build_candidates(story, audio, vision)
        lines = []
        for c in candidates:
            lines.append(f"PEAK {c['peak_id']}: core [{c['start']:.2f}-{c['end']:.2f}] combined={c['combined_score']:.2f} "
                         f"audio={c['audio_score']:.2f} visual={c['visual_score']:.2f} signals={','.join(c['signals'])}"
                         f"{' IN_PAYOFF' if c['in_payoff'] else ''}\n  words: "
                         f"{_words_text(words, c['start'] - 1.2, c['end'] + 1.2) or '(no speech)'}")
        observations = [f"[{o['t']:.2f}] {o['summary']}" for o in vision.get("observations", [])]
        text = (f"STORY: {story['title']} | {story['reason']}\nBEATS: "
                + "; ".join(f"{b['role']} [{b['start']:.2f}-{b['end']:.2f}] {b['note']}" for b in story["beats"])
                + "\n\nPEAK CANDIDATES:\n" + "\n".join(lines)
                + ("\n\nVISUAL OBSERVATIONS:\n" + "\n".join(observations) if observations else ""))
        schema = {**CHOICE_SCHEMA, "properties": {**CHOICE_SCHEMA["properties"],
                                                  "peak_id": {"type": "string",
                                                              "enum": [c["peak_id"] for c in candidates]}}}
        choice = ctx.provider.json_task("cold_open", ctx.settings.route("cold_open"), instructions=CHOICE_INSTRUCTIONS,
                                        input_text=text, schema=schema, schema_name="mimir_cold_open_v1")
        by_id = {c["peak_id"]: c for c in candidates}
        peak = by_id.get(str(choice.get("peak_id", "")))
        if peak is None:
            raise StageError(self.name, f"cold open choice names no supplied peak ({choice.get('peak_id')!r})")
        phrase = None
        by_word = {w["id"]: w for w in words}
        if choice.get("selection_mode") == "spoken_phrase":
            first, last = by_word.get(choice.get("start_word_id", "")), by_word.get(choice.get("end_word_id", ""))
            if first and last and first["start"] <= last["end"] and last["end"] - first["start"] <= cfg.max_duration:
                phrase = (float(first["start"]), float(last["end"]))
        start, end, policy = compute_window(peak, story, words, cfg, phrase, vision.get("shot_cuts", []))
        if end - start < cfg.min_understandable - 0.05:
            raise StageError(self.name, f"cold open window too short to understand ({end - start:.2f}s)")
        hook = self._hook(ctx, choice, peak, story, words, start, end)
        return StageOutput(data={"cold_open": {
            "peak": {k: peak[k] for k in ("peak_id", "start", "end", "center", "combined_score", "audio_score",
                                          "visual_score", "signals", "multimodal", "in_payoff")},
            "window": {"start": start, "end": end, "duration": round(end - start, 3)},
            "policy": policy, "selection_mode": choice.get("selection_mode"), "phrase": phrase,
            "reason": choice.get("reason", ""), "viewer_question": choice.get("viewer_question", ""),
            "spoiler_risk": choice.get("spoiler_risk", ""), "hook": hook, "candidates": candidates,
        }})

    def _hook(self, ctx: StageContext, choice: dict[str, Any], peak: dict[str, Any], story: dict[str, Any],
              words: Sequence[dict[str, Any]], start: float, end: float) -> dict[str, Any]:
        cfg = ctx.settings.cold_open
        valid, rejected = [], []
        for row in choice.get("hooks", [])[:3]:
            text, problem = clean_hook(row.get("text", ""), cfg)
            (valid.append({"text": text, "target": row.get("curiosity_target", "")}) if text
             else rejected.append({"text": row.get("text", ""), "problem": problem}))
        if not valid:
            ctx.ledger.info("hook_none_valid", "no hook line passed deterministic checks", rejected=rejected)
            return {"text": "", "accepted": False, "rejected": rejected, "reason": "no valid candidate"}
        text = (f"STORY: {story['title']}\nCOLD OPEN [{start:.2f}-{end:.2f}] words: "
                f"{_words_text(words, start, end) or '(no speech)'}\nPEAK signals: {', '.join(peak['signals'])}\n"
                "HOOK CANDIDATES:\n" + "\n".join(f"{i}. {h['text']} (target: {h['target']})"
                                                 for i, h in enumerate(valid, start=1)))
        judge = ctx.provider.json_task("hook_judge", ctx.settings.route("hook_judge"), instructions=JUDGE_INSTRUCTIONS,
                                       input_text=text, schema=JUDGE_SCHEMA, schema_name="mimir_hook_judge_v1")
        scores = {int(s["index"]): float(s["score"]) for s in judge.get("scores", [])}
        index = int(judge.get("selected_index", 0) or 0)
        if 1 <= index <= len(valid) and scores.get(index, 0.0) >= cfg.hook_accept_score:
            return {"text": valid[index - 1]["text"], "accepted": True, "score": scores[index],
                    "candidates": valid, "rejected": rejected}
        ctx.ledger.info("hook_rejected", "hook judge rejected all lines; cold open plays without hook text",
                        reason=judge.get("rejection_reason", ""))
        return {"text": "", "accepted": False, "candidates": valid, "rejected": rejected,
                "reason": judge.get("rejection_reason", "")}
