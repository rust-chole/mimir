# MIMIR

Give MIMIR a long video or VOD; it produces one short-form video and **verifies the
rendered MP4 before publishing it**. If the final file does not pass, nothing is
published and you get a rejected copy plus a report explaining exactly what failed.
A published short comes with a **human review sheet**: a person gives the final
acceptance, and the sheet says exactly where to look.

```powershell
python .\main.py "C:\path\to\vod.mp4"
```

| Outcome | Where | Exit code |
| --- | --- | --- |
| Published, every check passed | `vod_output/final/<name>_short.mp4` (+ `.qc.json`, `.review.md`) | 0 |
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
3. **Caption truth** — on the exact final short's audio (one mono 16 kHz analysis WAV
   of the edited clip). Three owners, never mixed:
   - **What was said** — `qwen3.8-omni-flash` listens twice: a primary ear with minimal
     domain context and no names, and a precision ear that gets verified names only as
     spelling references (never as proof a name was said). Harmless differences (case,
     punctuation, apostrophe glyph, hyphenation, "twelve"/"12") are ignored; a different,
     missing or inserted word, negation, number or name is a *local* dispute. One small
     model-diverse ear (`gpt-transcribe`, a few seconds of audio) settles it when it agrees
     with one reading; otherwise the **caption judge (Astra 6)** decides it in one batched
     call from the smallest evidence package (both Qwen readings, the independent ear,
     verified names, a little context). A deterministic guard rejects any word no ear
     heard; "not settled" keeps the primary words, marked uncertain. OpenAI transcription
     is only a bounded fallback (Qwen unavailable, invalid JSON after a retry, material
     disagreement), never the normal authority. The wording is then **frozen** (signed).
   - **When** — `Qwen3-ForcedAligner-0.6B` (official `qwen_asr`, loaded lazily, CPU or
     GPU) aligns the *frozen* words to the audio. Every result is validated (lexical
     parity, order, positive duration, monotonic, in range, full coverage, no overlap,
     no collapsed region); a small failed region is re-aligned locally with its trusted
     neighbours as anchors; otherwise the legacy Whisper clock re-times the whole
     transcript (disclosed). One timing authority per run; no global shifts, no invented
     times.
   - **Who** — diarization turns give each *voice* a caption colour (first voice A,
     next B, then C; a returning voice keeps its colour). Nobody is asked who is
     speaking and no name is printed; an uncertain word is neutral (a lone uncertain
     word inside one voice's turn is shown in that voice's colour instead of flashing
     neutral), and with fewer than two reliable voices the captions stay plain. Colours never move a caption
     to another lane (a second lane appears only during real overlapping speech) and
     never change a word or a time.
   Verified-name spelling the name lock cannot decide is a closed choice for the judge
   (verified spelling / keep / uncertain). Caption truth proves the words still equal the
   frozen transcript and freezes the final result with who decided each word.
4. **Cold open** — the model chooses *which* moment; its length is measured from the
   event (sound onset and decay, whole phrases, shot changes). It is the real moving
   peak with its real audio, **no speech captions**, an optional grounded headline,
   then a **hard restart** into the story. No grounded headline -> no headline.
5. **Presentation** — caption layout over the frozen truth (the V3 look). The **edit
   director (Astra 6)** decides editorial *intent* per moment ("what must the viewer
   see now?") from story beats, speakers, reactions, face tracks, shot cuts, layout
   (gameplay / facecam / screen share), UI/HUD regions, action regions and required
   visual content, given as region words. It cannot output coordinates: the
   deterministic compiler computes every crop/zoom and proves it in pixels. A stable
   wide shot is a valid choice. Optional single meme / SFX placed away from captions
   and never in the cold open.
6. **Final QC** on the rendered MP4 (see below) -> publish or reject -> **human review**.

## Final QC (on the actual file)

Stream integrity and A/V length; composition clock (cold open + story); cold open is
the clean source footage, moving, with the source audio, and has no speech captions;
headline burned only when approved; the story part equals the verified render at the
mapped time with audio in sync; every burned caption word equals the frozen truth at
its measured onset, nothing extra, nothing lost to the restart; no caption collisions;
protected story ranges present; the peak recurs in the story; effects clear of
captions; caption truth unchanged since freeze; camera plan proven in pixels.

Repairs are deterministic and happen at most once: an effect that makes the file fail
any check is dropped, a headline that fails its check is removed; the repaired file is
re-checked and the repair is disclosed (published as degraded).

## Human acceptance

No model reviews the final short. `<name>_short.review.md` lists, on the published
file's clock, every caption word the evidence could not settle, every word the caption
judge changed, judge-decided name spellings, the cold open and headline, effects,
the speaker colours and every disclosed degradation, plus Accept / Reject boxes.

## Model routing

| Role | Default | Why |
| --- | --- | --- |
| Caption ears (what was said, final short) | `qwen3.8-omni-flash`, reasoning none | a verbatim listener, twice |
| Caption word timing | `Qwen/Qwen3-ForcedAligner-0.6B` (local) | forced alignment of the frozen words |
| Caption judge (disputed words, name spelling) | `gpt-6-astra`, high | the words ARE the short |
| Edit director (editorial intent) | `gpt-6-astra`, high | the edit IS the short |
| Story judge, headline judge | `gpt-5.6-terra` | story authority |
| Scouting, teaser, headline drafts, speaker roles, memes, visual support | `gpt-5.6-luna` | cheap and sufficient |

Override any role in `.env` (see `.env.example`); `python -m ai.model_check --live`
checks that every configured model answers (OpenAI models and the DashScope Qwen ear).
The whole-VOD scouting transcript stays on `gpt-4o-mini-transcribe` + Whisper; only the
published short gets the caption stack (`ai/caption_stack/`).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env      # add OPENAI_API_KEY, DASHSCOPE_API_KEY + DASHSCOPE_BASE_URL (your
                            # Model Studio region's compatible-mode URL), GEMINI_API_KEY for visual facts
.\.venv\Scripts\python.exe .\verify_unified_clean.py
```

For an NVIDIA GPU, install the CUDA build of PyTorch from pytorch.org before
`requirements.txt` (the plain wheel runs the aligner on CPU). The aligner weights download
to the Hugging Face cache on first use of final-caption timing, never at import.

FFmpeg/ffprobe (with libass) must be on `PATH`. OpenCV (in `requirements.txt`) is
required: the camera and final QC are proven in pixels. Local SFX files go in
`meme_library/local_sfx/`.

Development: `python -m pytest tests` (the end-to-end suites render real media;
`MIMIR_VERIFY_E2E=1 python verify_unified_clean.py` runs everything).
