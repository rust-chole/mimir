"""Stage ``speakers``: diarize the story window and assign every caption word to a speaker.

Diarizer labels become stable anonymous ids ordered by speaking time
(S1 = most speech). Background clusters (crowd, game audio, one-off shouts)
are kept as evidence but never become caption speakers.
"""
from __future__ import annotations

import shutil
from typing import Any

from mimir.config import Settings, routes_for
from mimir.core.stage import StageContext, StageOutput
from mimir.media.audio import extract_wav
from mimir.models.provider import AudioMeta
from mimir.speakers.assign import assign_speakers
from mimir.speakers.census import classify, normalize_segments, speaker_stats


class SpeakerStage:
    name = "speakers"
    version = 1
    deps = ("source", "probe", "caption_verify")

    def params(self, settings: Settings) -> Any:
        return {"language": settings.transcript.language, "routes": routes_for(settings, "diarize")}

    def run(self, ctx: StageContext) -> StageOutput:
        caption = ctx.dep("caption_verify").json("caption_words")
        window_start, window_end = (float(v) for v in caption["window"])
        words = caption["words"]
        if not words:
            return StageOutput(data={"speakers": {"mode": "silent", "participants": [], "background": [],
                                                  "segments": [], "assignment": {}, "metrics": {}}})
        temp = ctx.out_dir / "tmp"
        audio = extract_wav(ctx.source.path, temp / "window.wav", start=window_start,
                            duration=window_end - window_start)
        try:
            raw_segments = ctx.provider.diarize("diarize", ctx.settings.route("diarize"), audio,
                                                language=ctx.settings.transcript.language,
                                                meta=AudioMeta(window_start, window_end))
        finally:
            shutil.rmtree(temp, ignore_errors=True)
        segments = normalize_segments(raw_segments, window_start, window_end)
        stats = speaker_stats(segments)
        mode, participants, background = classify(stats)
        anon = {raw: f"S{index + 1}" for index, raw in enumerate(participants)}
        participant_segments = [{**s, "speaker": anon[s["raw_speaker"]]} for s in segments
                                if s["raw_speaker"] in anon]
        if mode == "silent" or not participant_segments:
            # diarization heard no participant: one anonymous speaker owns the words (recorded, not guessed away)
            ctx.ledger.warning("diarization_empty", "diarizer returned no participant speech; single speaker S1 assumed")
            mode, anon = "single", {}
            participant_segments = [{"raw_speaker": "", "speaker": "S1", "start": window_start, "end": window_end,
                                     "duration": window_end - window_start, "text": "", "word_count": 0}]
            rows = [{"speaker": "S1", "confidence": 0.5, "source": "single_speaker_default"} for _ in words]
            metrics: dict[str, Any] = {"unresolved_words": 0}
        elif mode == "single":
            rows = [{"speaker": "S1", "confidence": 0.95, "source": "single_participant"} for _ in words]
            metrics = {"unresolved_words": 0}
        else:
            rows, metrics = assign_speakers(words, participant_segments)
        by_id = {s["raw_speaker"]: s for s in stats}
        return StageOutput(data={"speakers": {
            "mode": mode,
            "participants": [{"id": anon[raw], "raw_speaker": raw, **{k: by_id[raw][k] for k in (
                "speaking_seconds", "share", "word_count", "substantial_segment_count")}}
                for raw in participants if raw in anon],
            "background": [{"raw_speaker": raw, **{k: by_id[raw][k] for k in ("speaking_seconds", "word_count")}}
                           for raw in background if raw in by_id],
            "segments": [{"speaker": s["speaker"], "start": s["start"], "end": s["end"], "text": s["text"]}
                         for s in participant_segments],
            "background_segments": [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segments
                                    if s["raw_speaker"] not in anon],
            "assignment": {word["id"]: row for word, row in zip(words, rows)},
            "metrics": metrics,
        }})
