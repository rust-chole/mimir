"""Stage ``speakers``: diarize the story window and assign every caption word to a speaker.

Diarizer labels become stable anonymous ids ordered by speaking time
(S1 = most speech). Background clusters (crowd, game audio, one-off shouts)
are kept as evidence but never become caption speakers.

``resolution.status`` separates evidence from absence of evidence:

* ``confirmed`` - usable participant segments; one voice counts as a single speaker
  only when it covers most of the transcribed speech;
* ``unresolved`` - the diarizer returned nothing, only unusable/noise segments, or a lone
  voice that covers too little of the speech. Words then keep NO speaker (never a
  fabricated S1), no identity can be attached, and the camera stays conservative.
"""
from __future__ import annotations

import shutil
from typing import Any

from mimir.config import Settings, routes_for
from mimir.core.stage import StageContext, StageOutput
from mimir.media.audio import extract_wav
from mimir.models.provider import AudioMeta
from mimir.speakers.assign import assign_speakers, measured_overlaps
from mimir.speakers.census import classify, normalize_segments, speaker_stats

SINGLE_MIN_COVERAGE = 0.6    # share of transcribed words a lone diarized voice must cover
COVERAGE_SLACK = 0.25        # seconds of diarization boundary tolerance around a word


def speech_coverage(words: list[dict[str, Any]], segments: list[dict[str, Any]]) -> float:
    """Share of words whose midpoint lies inside some participant segment (with boundary slack)."""
    if not words:
        return 0.0
    inside = 0
    for word in words:
        mid = (float(word["start"]) + float(word["end"])) / 2
        inside += any(s["start"] - COVERAGE_SLACK <= mid <= s["end"] + COVERAGE_SLACK for s in segments)
    return inside / len(words)


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
                                                  "segments": [], "assignment": {}, "metrics": {},
                                                  "overlaps": [], "resolution": {
                                                      "status": "confirmed", "reason": "no speech in the window",
                                                      "raw_segments": 0, "usable_segments": 0, "coverage": 0.0}}})
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
        coverage = speech_coverage(words, participant_segments)
        resolution = {"status": "confirmed", "reason": "", "raw_segments": len(raw_segments),
                      "usable_segments": len(segments), "coverage": round(coverage, 3)}
        if not raw_segments:
            resolution.update(status="unresolved", reason="diarizer returned no segments")
        elif not segments:
            resolution.update(status="unresolved", reason="diarizer returned only unusable segments "
                              "(no finite, positive, labelled time range inside the window)")
        elif mode in ("silent", "unresolved") or not participant_segments:
            resolution.update(status="unresolved", reason="no diarized voice is participant-like speech")
        elif mode == "single" and coverage < SINGLE_MIN_COVERAGE:
            resolution.update(status="unresolved", reason=f"the only diarized voice covers {coverage:.0%} of the "
                              f"transcribed words (needs {SINGLE_MIN_COVERAGE:.0%})")
        if resolution["status"] == "unresolved":
            ctx.ledger.warning("diarization_unresolved", f"{resolution['reason']}; caption words keep no speaker, "
                               "no identity is attached and framing stays conservative")
            return StageOutput(data={"speakers": {
                "mode": "unresolved", "resolution": resolution, "participants": [], "background": [],
                "segments": [], "overlaps": [],
                "background_segments": [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segments],
                "assignment": {w["id"]: {"speaker": "", "confidence": 0.0, "source": "diarization_unresolved"}
                               for w in words},
                "metrics": {"unresolved_words": len(words)},
            }})
        if mode == "single":
            rows = []
            for word in words:
                covered = speech_coverage([word], participant_segments) == 1.0
                rows.append({"speaker": "S1", "confidence": 0.95 if covered else 0.7,
                             "source": "single_participant" if covered else "single_participant_uncovered"})
            metrics: dict[str, Any] = {"unresolved_words": 0}
        else:
            rows, metrics = assign_speakers(words, participant_segments)
        by_id = {s["raw_speaker"]: s for s in stats}
        return StageOutput(data={"speakers": {
            "mode": mode,
            "resolution": resolution,
            "participants": [{"id": anon[raw], "raw_speaker": raw, **{k: by_id[raw][k] for k in (
                "speaking_seconds", "share", "word_count", "substantial_segment_count")}}
                for raw in participants if raw in anon],
            "background": [{"raw_speaker": raw, **{k: by_id[raw][k] for k in ("speaking_seconds", "word_count")}}
                           for raw in background if raw in by_id],
            "segments": [{"speaker": s["speaker"], "start": s["start"], "end": s["end"], "text": s["text"]}
                         for s in participant_segments],
            "background_segments": [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segments
                                    if s["raw_speaker"] not in anon],
            "overlaps": measured_overlaps(participant_segments) if mode != "single" else [],
            "assignment": {word["id"]: row for word, row in zip(words, rows)},
            "metrics": metrics,
        }})
