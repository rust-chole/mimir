"""Stage ``transcript``: whole-VOD lexical + timing truth used for story discovery.

Two ears per chunk: a fast lexical ear (wording) and the timing ear (word
clock), merged by the fixed-clock aligner. Caption-grade verification of the
selected story happens later (``mimir.transcript.verify``) on that window only.
"""
from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from mimir.config import Settings, routes_for, section
from mimir.core.stage import StageContext, StageOutput
from mimir.errors import StageError
from mimir.media.audio import extract_mp3
from mimir.media.probe import MediaInfo
from mimir.models.provider import AudioMeta
from mimir.transcript.align import AlignmentError, align_text_to_clock
from mimir.transcript.prompts import TRANSCRIPTION_PROMPT
from mimir.transcript.segments import build_segments

CHUNK_WORKERS = 3


class TranscriptStage:
    name = "transcript"
    version = 1
    deps = ("source", "probe")

    def params(self, settings: Settings) -> Any:
        return {"transcript": section(settings, "transcript"),
                "routes": routes_for(settings, "transcribe_fast", "timing")}

    def run(self, ctx: StageContext) -> StageOutput:
        info = MediaInfo.from_dict(ctx.dep("probe").json("media"))
        if not info.has_audio:
            raise StageError(self.name, "source has no audio stream; MIMIR needs speech/audio evidence")
        settings = ctx.settings.transcript
        temp = ctx.out_dir / "tmp"
        chunks: list[tuple[float, float, Path]] = []
        offset = 0.0
        index = 0
        while offset < info.duration - 0.05:
            length = min(settings.chunk_seconds, info.duration - offset)
            path = extract_mp3(ctx.source.path, temp / f"chunk_{index:03d}.mp3", start=offset, duration=length)
            chunks.append((offset, length, path))
            offset += length
            index += 1

        def work(chunk: tuple[float, float, Path]) -> dict[str, Any]:
            chunk_offset, length, path = chunk
            meta = AudioMeta(chunk_offset, chunk_offset + length)
            lexical_route = ctx.settings.route("transcribe_fast")
            lexical = ctx.provider.transcribe("transcribe_fast", lexical_route, path, language=settings.language,
                                              prompt=TRANSCRIPTION_PROMPT, meta=meta)
            text_source = "lexical"
            if not lexical.text.strip():
                lexical = ctx.provider.transcribe("transcribe_fast", lexical_route, path,
                                                  language=settings.language, meta=meta)
            timing = ctx.provider.transcribe("timing", ctx.settings.route("timing"), path,
                                             language=settings.language, word_timestamps=True, meta=meta)
            text = lexical.text.strip()
            if not text and timing.words:
                text = " ".join(w.text for w in timing.words)
                text_source = "timing_ear_text"
            words: list[dict[str, Any]] = []
            ratio = 1.0
            if text and timing.words:
                aligned, ratio = align_text_to_clock(text, timing.words, length)
                words = [{"text": w.text, "start": round(w.start + chunk_offset, 3),
                          "end": round(w.end + chunk_offset, 3), "timing_source": w.source} for w in aligned]
            elif text:
                raise AlignmentError(f"chunk at {chunk_offset:.1f}s has text but no word clock")
            return {"offset": chunk_offset, "duration": length, "words": words, "alignment_ratio": ratio,
                    "text_source": text_source}

        try:
            with ThreadPoolExecutor(max_workers=CHUNK_WORKERS) as pool:
                results = list(pool.map(work, chunks))
        except AlignmentError as error:
            raise StageError(self.name, str(error)) from error
        finally:
            shutil.rmtree(temp, ignore_errors=True)

        words: list[dict[str, Any]] = []
        for result in results:
            if result["text_source"] != "lexical":
                ctx.ledger.warning("lexical_ear_empty", "lexical ear returned no text; timing-ear wording used",
                                   chunk_offset=result["offset"])
            for word in result["words"]:
                if words and word["start"] < words[-1]["start"]:
                    continue
                words.append(word)
        for number, word in enumerate(words):
            word["id"] = f"v{number:06d}"
        if not words:
            raise StageError(self.name, "no speech was transcribed in the source")
        ratios = [r["alignment_ratio"] for r in results if r["words"]]
        return StageOutput(data={"transcript": {
            "duration": info.duration,
            "language": settings.language,
            "models": routes_for(ctx.settings, "transcribe_fast", "timing"),
            "alignment_ratio": round(sum(ratios) / len(ratios), 4) if ratios else 0.0,
            "chunks": [{k: r[k] for k in ("offset", "duration", "alignment_ratio", "text_source")} for r in results],
            "words": words,
            "segments": build_segments(words),
        }})
