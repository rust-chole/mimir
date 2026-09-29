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
   A post-selection **causal integrity** check then asks whether the chosen short keeps
   what its payoff needs (payoff inside, no boundary through a running sentence, a cause
   before and a consequence after, no protected range cut). It never re-ranks: it passes,
   makes ONE bounded speech-aligned expansion (protecting what it adds), warns, or — only
   when the candidate lacks the payoff it claims and another candidate keeps its payoff —
   reselects. No beat-length quotas: a one-word reaction or a silent visual setup counts.
2. **Pacing** — dead air is cut only where it does not touch protected story.
3. **Caption truth** — on the exact final audio: primary `gpt-transcribe`, a
   model-diverse cross-check (with token confidence) and the Whisper word clock.
   Disagreements, low-confidence words and near-miss names are re-heard in small
   local windows by 3-5 independent ears. A span the ears hear unanimously is settled;
   every other span goes, in one call, to the **caption judge (Astra 6)** with all the
   evidence (both transcripts, every ear's model/view/wording, token confidence,
   verified names, measured clock words). It decides **what** was said; a deterministic
   guard checks every word it CHANGES has acoustic support at the disputed spot (an
   unprompted ear or the clock heard it, or two prompted ears did; the context never
   counts) and that the ears it cites carry the change — otherwise that span alone falls
   back to the strict 3/3-4/5 vote, else to the primary word marked uncertain. It never
   touches **when**: timing is the measured word clock (plus a backward-only acoustic
   guard for late phrase starts). The clock is health-checked (several symptoms, not one
   number); a damaged clock is re-measured on overlapping ~10 s chunks of the same audio
   and spliced in only where it was damaged (healthy anchors never move), with the
   whole-VOD clock as a second measured source; no synthetic timing exists, and only a
   clock that stays corrupt after recovery blocks. Speakers come from diarization + your
   voice-confirmed names; verified-name
   spelling the name lock cannot decide is a closed choice for the judge (verified
   spelling / keep / uncertain). The result is **frozen**, with who decided each word.
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
   wide shot is a valid choice. Motion is not automatically action: motion explained by
   UI/HUD/chat text, a person's own movement, a transient burst or an overlay-like corner
   is kept as evidence but never a framing target; a measured gameplay / screen-share
   layout gives the screen priority. Optional single meme / SFX placed away from captions
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

Three classes, on purpose:

- **Block** only objective corruption: unreadable media, wrong composition clock or
  intro source, captions in the cold open, wrong intro audio, proven A/V drift, burned
  words differing from the frozen truth, protected story removed, camera pixels not
  reached after the repair round, a published file that is not the QC-verified bytes, a
  word clock still corrupt after recovery.
- **Repair once** per subsystem, then re-check: a damaged word clock (re-measured), a cut
  sentence / missing setup-reaction (story expansion), an unsafe camera (stable static
  camera, cold open back to clean footage), a failing effect (dropped), a failing
  headline (removed). Repairs are disclosed.
- **Warn** (published as degraded, never rejected): no headline, no effect, weak face
  tracking, anonymous speakers, unresolved words shown as uncertain, a conservative wide
  camera, low director confidence, calm unexplained holds, a weak optional story beat.

## Cache integrity

Stage inputs and the source are identified by content (sha256, cached), so a moved or
re-extracted byte-identical project never re-runs a paid stage; critical outputs (paced
clip, timeline, frozen caption truth, presentation render, composed candidate, published
file) record their sha256 and an edited or stale file is recomputed, never reused.

## Human acceptance

No model reviews the final short. `<name>_short.review.md` lists, on the published
file's clock, every caption word the evidence could not settle, every word the caption
judge changed, judge-decided name spellings, the cold open and headline, effects,
speaker labels and every disclosed degradation, plus Accept / Reject boxes.

## Real-VOD A/B validation

Every run writes `<name>_short.artifacts.json` (or `_REJECTED.artifacts.json`) next to the
output: story + boundaries + integrity report, protected ranges, frozen caption truth,
lexical decisions and uncertain spans, clock health/recovery, speaker roster, cold-open
event and measured bounds, director plan, camera plan, pixel proof, QC report, review
sheet. `python compare_runs.py A B` puts two runs side by side (an index or any branch's
`vod_output/state/<stem>_pipeline.json`); it lists facts, the human decides.

## Model routing

| Role | Default | Why |
| --- | --- | --- |
| Caption judge (disputed words, name spelling) | `gpt-6-astra`, high | the words ARE the short |
| Edit director (editorial intent) | `gpt-6-astra`, high | the edit IS the short |
| Story judge, headline judge | `gpt-5.6-terra` | story authority |
| Scouting, teaser, headline drafts, speaker roles, memes, visual support | `gpt-5.6-luna` | cheap and sufficient |

Override any role in `.env` (see `.env.example`); `python -m ai.model_check --live`
checks that every configured model answers.

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
