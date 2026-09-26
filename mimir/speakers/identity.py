"""Stage ``identity``: optional real names for anonymous speakers.

* single confident speaker   -> nothing is asked, the run continues;
* several speakers           -> ``speaker_preview`` audio samples (audio only,
                               never part of the final video) are written; in an
                               interactive terminal the user may name each voice,
                               otherwise stable anonymous ids are kept;
* names given up front (``--speaker-names S1=Kai``) are user-confirmed.

Only confirmed names are ever printed on screen.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

from mimir.config import Settings, section
from mimir.core.stage import StageContext, StageOutput
from mimir.media.ffmpeg import ffmpeg

PREVIEW_TARGET = 3.0
PREVIEW_MIN_SEGMENT = 0.65
PREVIEW_TRIM = 0.06
OVERLAP_PAD = 0.20

Prompt = Callable[[str, Path], str]


def _console_prompt(speaker_id: str, preview: Path) -> str:
    sys.stdout.write(f"\nSpeaker preview for {speaker_id}: {preview}\n")
    sys.stdout.write(f"Who is {speaker_id}? (Enter to keep anonymous): ")
    sys.stdout.flush()
    try:
        return input().strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def preview_ranges(speaker_id: str, segments: list[dict[str, Any]]) -> list[tuple[float, float]]:
    own = [s for s in segments if s["speaker"] == speaker_id]
    others = [s for s in segments if s["speaker"] != speaker_id]
    clean = []
    for segment in sorted(own, key=lambda s: -(s["end"] - s["start"])):
        start, end = segment["start"] + PREVIEW_TRIM, segment["end"] - PREVIEW_TRIM
        if end - start < PREVIEW_MIN_SEGMENT:
            continue
        if any(o["start"] - OVERLAP_PAD < end and o["end"] + OVERLAP_PAD > start for o in others):
            continue
        clean.append((start, end))
    chosen: list[tuple[float, float]] = []
    total = 0.0
    for start, end in clean:
        take = min(end - start, PREVIEW_TARGET - total)
        if take < PREVIEW_MIN_SEGMENT * 0.8:
            break
        chosen.append((start, start + take))
        total += take
        if total >= PREVIEW_TARGET - 0.05:
            break
    return sorted(chosen)


def render_preview(source: Path, ranges: list[tuple[float, float]], output: Path) -> Path:
    parts = [f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS,afade=t=in:d=0.02,"
             f"afade=t=out:st={max(0.0, b - a - 0.03):.3f}:d=0.03[a{i}]" for i, (a, b) in enumerate(ranges)]
    graph = ";".join(parts) + ";" + "".join(f"[a{i}]" for i in range(len(ranges))) + \
        f"concat=n={len(ranges)}:v=0:a=1[out]"
    ffmpeg(["-v", "error", "-i", str(source), "-filter_complex", graph, "-map", "[out]", "-ac", "1", "-ar", "48000",
            "-c:a", "pcm_s16le", str(output)], timeout=600)
    return output


class IdentityStage:
    name = "identity"
    version = 1
    deps = ("source", "speakers")

    def __init__(self, prompt: Prompt | None = None) -> None:
        self._prompt = prompt

    def params(self, settings: Settings) -> Any:
        ident = section(settings, "identity")
        return {"speaker_names": ident["speaker_names"], "interactive": ident["interactive"]}

    def run(self, ctx: StageContext) -> StageOutput:
        speakers = ctx.dep("speakers").json("speakers")
        participants = [p["id"] for p in speakers["participants"]]
        identities: dict[str, dict[str, Any]] = {sid: {"name": "", "confirmed": False, "source": "anonymous"}
                                                 for sid in participants}
        for speaker_id, name in ctx.settings.identity.speaker_names:
            if speaker_id not in identities:
                ctx.ledger.warning("unknown_speaker_name", f"--speaker-names mentions {speaker_id}, which is not a "
                                   f"participant (have {participants})")
                continue
            identities[speaker_id] = {"name": name, "confirmed": True, "source": "user_declared"}
        ambiguous = len(participants) > 1
        files: dict[str, Path] = {}
        asked = False
        if ambiguous:
            for speaker_id in participants:
                ranges = preview_ranges(speaker_id, speakers["segments"])
                if not ranges:
                    ctx.ledger.info("no_clean_preview", f"no clean non-overlapping sample for {speaker_id}")
                    continue
                path = render_preview(ctx.source.path, ranges, ctx.out_dir / f"speaker_preview_{speaker_id}.wav")
                files[f"speaker_preview_{speaker_id}"] = path
            interactive = ctx.settings.identity.interactive and (self._prompt is not None or sys.stdin.isatty())
            if interactive:
                prompt = self._prompt or _console_prompt
                used = {row["name"].casefold() for row in identities.values() if row["name"]}
                for speaker_id in participants:
                    if identities[speaker_id]["confirmed"] or f"speaker_preview_{speaker_id}" not in files:
                        continue
                    answer = " ".join(prompt(speaker_id, files[f"speaker_preview_{speaker_id}"]).split())
                    asked = True
                    if answer and answer.casefold() not in used:
                        identities[speaker_id] = {"name": answer, "confirmed": True, "source": "user_preview"}
                        used.add(answer.casefold())
        return StageOutput(
            data={"identity": {"mode": speakers["mode"], "ambiguous": ambiguous, "asked": asked,
                               "speakers": identities,
                               "previews": {sid: f"speaker_preview_{sid}.wav" for sid in participants
                                            if f"speaker_preview_{sid}" in files}}},
            files=files,
        )
