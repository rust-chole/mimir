# MIMIR

Give MIMIR a long video or VOD; it produces one short-form video and **verifies the
rendered MP4 before publishing it**. If the final file does not pass, nothing is
published and you get a rejected copy plus a report explaining exactly what failed.

```powershell
python .\main.py "C:\path\to\vod.mp4"
```

| Outcome | Where | Exit code |
| --- | --- | --- |
| Published, every check passed | `vod_output/final/<name>_short.mp4` (+ `.qc.json`) | 0 |
| Published with explicit degradations (e.g. no face tracked -> stable wide camera) | same, warnings printed and in `.qc.json` | 3 |
| Rejected by final QC | `vod_output/rejected/<name>_short_REJECTED.mp4` + `.qc.json`; nothing in `final/` | 1 |
| No usable story in the VOD | — | 2 |

Useful options: `--creator "Name"` (a verified name the headline may use),
`--clip N` (pick a candidate), `--rerender` (re-run only presentation, render and QC;
transcription, story discovery and caption ASR stay cached), `--force` (ignore every
cache), `--no-memes`, `--verbose`.

## What happens

1. **Story** — whole-VOD transcript + visual facts; Luna scouts money moments, Terra
   judges the story, protected ranges keep setup / escalation / payoff / reaction.
2. **Pacing** — dead air is cut only where it does not touch protected story.
3. **Caption truth** — on the exact final audio: primary `gpt-transcribe`, a
   model-diverse cross-check (with token confidence) and the Whisper word clock.
   Disagreements, low-confidence words and near-miss names are re-heard in small
   local windows; text changes only on a strict independent majority (3/3, then 4/5),
   otherwise the primary word stays and is marked uncertain. Timing is the measured
   word clock (plus a backward-only acoustic guard for late phrase starts); speakers
   come from diarization + your voice-confirmed names. The result is **frozen**.
4. **Cold open** — the model chooses *which* moment; its length is measured from the
   event (sound onset and decay, whole phrases, shot changes). It is the real moving
   peak with its real audio, **no speech captions**, an optional grounded headline,
   then a **hard restart** into the story. No grounded headline -> no headline.
5. **Presentation** — caption layout over the frozen truth (the V3 look), an
   evidence-directed camera ("what must the viewer see now?"), optional single meme /
   SFX placed away from captions and never in the cold open.
6. **Final QC** on the rendered MP4 (see below) -> bounded review -> publish or reject.

## Final QC (on the actual file)

Stream integrity and A/V length; composition clock (cold open + story); cold open is
the clean source footage, moving, with the source audio, and has no speech captions;
headline burned only when approved; the story part equals the verified render at the
mapped time with audio in sync; every burned caption word equals the frozen truth at
its measured onset, nothing extra, nothing lost to the restart; no caption collisions;
protected story ranges present; the peak recurs in the story; effects clear of
captions; caption truth unchanged since freeze; camera plan proven in pixels.

A multimodal reviewer then looks at final-vs-source frames. It can only trigger one
deterministic repair (drop the headline, drop the effect, or hold a static camera) —
the repaired file is re-checked by QC — or add warnings. It can never pass something
QC failed.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env      # add OPENAI_API_KEY (and GEMINI_API_KEY for visual facts)
.\.venv\Scripts\python.exe .\verify_unified_clean.py
```

FFmpeg/ffprobe (with libass) must be on `PATH`. OpenCV (in `requirements.txt`) is
required: the camera and final QC are proven in pixels. Local SFX files go in
`meme_library/local_sfx/`.

Development: `python -m pytest tests` (the end-to-end suites render real media;
`MIMIR_VERIFY_E2E=1 python verify_unified_clean.py` runs everything).
