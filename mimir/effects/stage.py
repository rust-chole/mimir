"""Stage ``effects``: restrained effects plan (never touches story, captions or camera truth).

* cold open -> main restart: a short flash and (if the library has one) a quiet whoosh;
* at most ONE meme/SFX accent in the whole Short, only on a real expectation break
  (unexpectedness >= 0.70), never in the cold open, never burying the payoff,
  only when the curated local library has an asset for that category.
NO ACCENT IS BETTER THAN A BAD ACCENT.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from mimir.config import Settings, routes_for, section
from mimir.core.stage import StageContext, StageOutput
from mimir.effects.library import ACCENT_CATEGORIES, TRANSITION_CATEGORY, fingerprint, pick, scan
from mimir.media.probe import audio_duration
from mimir.captions.layout import map_words
from mimir.timeline.schema import STORY, Timeline

MAIN_OPENING_PROTECTION = 0.70
PAYOFF_BEFORE, PAYOFF_AFTER = 0.30, 0.40
AFTER_WORD_DELAY = 0.08
STRENGTH_VOLUME = {"subtle": 0.85, "medium": 1.0, "strong": 1.15}

ACCENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "use_accent": {"type": "boolean"},
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "anchor_word_key": {"type": "string"},
                    "placement": {"type": "string", "enum": ["on_word", "after_word"]},
                    "category": {"type": "string", "enum": list(ACCENT_CATEGORIES)},
                    "kind": {"type": "string", "enum": ["audio", "visual"]},
                    "strength": {"type": "string", "enum": ["subtle", "medium", "strong"]},
                    "unexpectedness": {"type": "number"},
                    "absurdity": {"type": "number"},
                    "reversal": {"type": "number"},
                    "fit_score": {"type": "number"},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["anchor_word_key", "placement", "category", "kind", "strength", "unexpectedness",
                             "absurdity", "reversal", "fit_score", "confidence", "reason"],
            },
        },
        "reason": {"type": "string"},
    },
    "required": ["use_accent", "candidates", "reason"],
}

INSTRUCTIONS = """
You are a senior short-form editor deciding whether this Short deserves ONE meme/SFX accent. NO ACCENT IS BETTER
THAN A BAD ACCENT. An accent punctuates an EXPECTATION BREAK (a reversal, a bizarre line, an absurd physical result,
an awkward dead stop, a confident claim immediately failing) - not a merely loud, profane or funny moment.
Unexpectedness must be at least 0.70 or return use_accent=false. Never place an accent in the cold open, never
cover important dialogue (prefer a word followed by a real pause: large GAP_AFTER), keep the payoff itself clean.
Choose only categories available in the library. Scores are 0-1 (unexpectedness, absurdity, reversal, confidence)
and fit_score is 0-10. Up to 3 candidates; software keeps at most one.
""".strip()


class EffectsStage:
    name = "effects"
    version = 1
    deps = ("story", "caption_truth", "timeline")

    def params(self, settings: Settings) -> Any:
        return {"effects": section(settings, "effects"), "cold_open_frames": settings.cold_open.transition_frames,
                "library": fingerprint(settings.effects.sfx_library), "repair": section(settings, "repair"),
                "routes": routes_for(settings, "effects")}

    def run(self, ctx: StageContext) -> StageOutput:
        cfg = ctx.settings.effects
        timeline = Timeline.from_dict(ctx.dep("timeline").json("timeline"))
        story = ctx.dep("story").json("story")
        truth = ctx.dep("caption_truth").json("caption_truth")
        words = [{"key": w.key, "segment": w.segment, "start": w.start, "end": w.end, "text": w.text}
                 for w in map_words(truth["words"], timeline, truth.get("confirmed_names", {}))]
        library = scan(cfg.sfx_library)
        main_start = timeline.main_start_frame / timeline.fps
        plan: dict[str, Any] = {"flash": {"frame": timeline.main_start_frame,
                                          "frames": ctx.settings.cold_open.transition_frames},
                                "sfx": [], "overlays": [], "accent": {"used": False}}
        whooshes = library["audio"].get(TRANSITION_CATEGORY, [])
        if cfg.transition_whoosh and whooshes:
            path = pick(whooshes, story["story_id"] + ":whoosh")
            duration = audio_duration(path)
            plan["sfx"].append({"kind": "transition", "path": path, "start": round(max(0.0, main_start - 0.15), 3),
                                "duration": round(duration, 3), "volume": cfg.whoosh_volume, "duck": False,
                                "category": TRANSITION_CATEGORY})
        elif cfg.transition_whoosh:
            ctx.ledger.info("no_transition_sfx", "library has no transition sound; flash only")
        available = [c for c in ACCENT_CATEGORIES if c in library["audio"]]
        if not cfg.enable_accents:
            ctx.ledger.info("accents_disabled", "meme/SFX accents disabled by configuration")
        elif not available:
            ctx.ledger.info("accent_library_empty", "no curated accent assets; no accent considered")
        else:
            plan["accent"] = self._accent(ctx, story, words, timeline, library, available, main_start)
            if plan["accent"].get("used"):
                plan["sfx"].append(plan["accent"]["sfx"])
        return StageOutput(data={"effects": plan})

    def _accent(self, ctx: StageContext, story: dict[str, Any], all_words: list[dict[str, Any]], timeline: Timeline,
                library: dict[str, Any], available: list[str], main_start: float) -> dict[str, Any]:
        cfg = ctx.settings.effects
        words = [w for w in all_words if w["segment"] == STORY]
        payoff = [(a, b) for a, b, _ in timeline.map_interval(story["payoff"]["start"], story["payoff"]["end"])]
        lines = []
        for index, word in enumerate(words):
            gap = (words[index + 1]["start"] - word["end"]) if index + 1 < len(words) else 9.9
            lines.append(f"{word['key']} [{word['start']:.2f}] {word['text']} GAP_AFTER={gap:.2f}")
        text = (f"STORY: {story['title']} ({story['emotion']}): {story['reason']}\nPAYOFF (output seconds): "
                + ", ".join(f"{a:.2f}-{b:.2f}" for a, b in payoff)
                + f"\nLIBRARY CATEGORIES: {', '.join(available)}\n\nMAIN WORDS:\n" + "\n".join(lines))
        raw = ctx.provider.json_task("effects", ctx.settings.route("effects"), instructions=INSTRUCTIONS,
                                     input_text=text, schema=ACCENT_SCHEMA, schema_name="mimir_accent_v1")
        by_key = {w["key"]: (i, w) for i, w in enumerate(words)}
        best: tuple[float, dict[str, Any]] | None = None
        rejected = []
        for row in raw.get("candidates", [])[:3] if raw.get("use_accent") else []:
            found = by_key.get(row["anchor_word_key"])
            problem = ""
            if found is None:
                problem = "unknown anchor word"
            elif row["category"] not in available:
                problem = "category not in library"
            elif row["kind"] == "visual":
                problem = "visual accents need curated visual assets (audio only)"
            elif float(row["unexpectedness"]) < cfg.unexpectedness_min:
                problem = "unexpectedness below gate"
            elif float(row["fit_score"]) < cfg.audio_min_fit or float(row["confidence"]) < 0.68:
                problem = "fit/confidence below gate"
            if problem:
                rejected.append({"candidate": row, "problem": problem})
                continue
            index, word = found
            start = word["end"] + AFTER_WORD_DELAY if row["placement"] == "after_word" else word["start"]
            if start < main_start + MAIN_OPENING_PROTECTION:
                rejected.append({"candidate": row, "problem": "too close to the restart"})
                continue
            in_payoff = any(a - PAYOFF_BEFORE <= start <= b + PAYOFF_AFTER for a, b in payoff)
            if in_payoff and not (row["category"] == "impact" and float(row["fit_score"]) >= 8.4
                                  and float(row["confidence"]) >= 0.80):
                rejected.append({"candidate": row, "problem": "payoff stays clean"})
                continue
            gap = (words[index + 1]["start"] - word["end"]) if index + 1 < len(words) else 2.0
            headroom = max(0.0, min(1.0, gap / 1.2))
            score = (0.35 * float(row["unexpectedness"]) + 0.25 * float(row["absurdity"]) + 0.20 * float(row["reversal"])
                     + 0.10 * headroom + 0.10 * float(row["fit_score"]) / 10.0)
            if best is None or score > best[0]:
                best = (score, {**row, "start": round(start, 3)})
        if best is None:
            ctx.ledger.info("no_accent", "no accent passed the gates", reason=raw.get("reason", ""),
                            rejected=len(rejected))
            return {"used": False, "reason": raw.get("reason", ""), "rejected": rejected}
        score, row = best
        path = pick(library["audio"][row["category"]], story["story_id"] + ":" + row["category"])
        duration = audio_duration(path)
        volume = cfg.accent_volume * STRENGTH_VOLUME.get(row["strength"], 1.0)
        return {"used": True, "score": round(score, 3), "candidate": row, "rejected": rejected,
                "sfx": {"kind": "accent", "path": path, "start": row["start"], "duration": round(duration, 3),
                        "volume": round(volume, 3), "duck": True, "category": row["category"]}}
