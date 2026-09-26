# MIMIR architecture

One production path. `mimir/pipeline.py` declares the stage graph; `mimir/core/runner.py` only
orchestrates (signatures, caching, artifacts). Each subsystem package owns exactly one truth.

```
Source Video
 → probe            media/probe_stage.py      technical facts, output frame rate
 → transcript       transcript/whole.py       whole-VOD lexical + timing truth (for discovery)
 → vod_evidence     story/evidence.py         cheap whole-VOD audio/motion peaks + bounded peak probe
 → story            story/discovery.py        money moments → candidates → judge → StoryPackage
 → caption_verify   transcript/verify.py      caption-grade lexical/timing truth of the story window
 → speakers         speakers/stage.py         speaker truth (anonymous S1, S2, ...)
 → identity         speakers/identity.py      optional identity truth (+ speaker_preview audio)
 → caption_truth    transcript/truth.py       merged, name-locked, FROZEN caption truth (signed)
 → vision           vision/stage.py           selected-short visual evidence
 → cold_open        coldopen/planner.py       mandatory peak cold open (+ judged hook line)
 → timeline         timeline/builder.py       canonical source→output timeline (pacing, frame exact)
 → edit_context     edit/context.py           EditContextBuilder
 → edit_direction   edit/director.py          AI Edit Director (intent only)
 → edit_validation  edit/validator.py         evidence + restraint policy
 → edit_compile     edit/compiler.py          deterministic edit compiler (per-frame camera/layout)
 → captions         captions/stage.py         ASS from frozen truth through the timeline
 → effects          effects/stage.py          transition flash/whoosh, at most one accent
 → render_base      render/base.py            story timeline at source geometry (PCM audio)
 → render           render/stage.py           compositor + libass + audio mix → MP4
 → qc               qc/stage.py               final quality gate on the rendered MP4
 → publish          publish.py
```

## Truth ownership

| truth | owner | rule |
| --- | --- | --- |
| what was said | `transcript.verify` + `transcript.name_lock` | ASR consensus; micro-window votes (3/3, then 4/5); never auto-deletes; verified names only with evidence |
| when it was said | `transcript.align` (timing ear) + `transcript.clock_guard` | lexical corrections are re-aligned to the same immutable clock; one token per word, simultaneous speech keeps each word's measured interval; PCM guard only moves late phrase starts earlier |
| who said it | `speakers` | diarization census + per-segment text/timing evidence; unresolved stays unresolved; measured overlaps name the interrupter (their turn gets the second caption lane) |
| real identity | `speakers.identity` | only user-confirmed names; single confident speaker never asks |
| which story | `story` | complete causal chain; protected ranges; beats with minimum durations |
| source vs output time | `timeline` | explicit segments, frame quantized; source timestamps never rewritten |
| what the viewer sees | `edit.*` | AI chooses intent; code computes every crop, zoom, movement |

Speaker truth, visible face tracks and real identity stay separate: a face track is linked to a
speaker only when mouth activity correlates with that speaker's speech (Fisher-z, margin, one-to-one).

## Visual evidence (selected Short only)

`vision` analyses the story window at 1280 px / 10 fps: YuNet face detection (bundled, checksum
verified; a missing model falls back to Haar with a recorded `face_detector_fallback` warning) +
optical-flow tracking,
mouth activity, speaker linking, action regions (motion not explained by people), static UI/HUD
regions and a layout class (`talking_head`, `multi_person`, `facecam_gameplay`, `screen_content`,
`gameplay`, `scene`), plus one bounded observer call. A shot cut must persist: every picture shortly
before the jump differs from every picture 0.3-1.5 s after it. Flashes, explosions and strobes return
to the old picture, so they neither split face tracks nor force camera cuts at the payoff. A
pixel-stable corner face over moving content is a composited facecam.

## Edit architecture

`EditContextBuilder → AI Edit Director → EditPlan → Validation → Deterministic Compiler`

* Intents: `HOLD, WIDE_CONTEXT, TWO_SHOT, SPEAKER_MEDIUM, SPEAKER_PUNCH, REACTION, ACTION_REGION,
  GAMEPLAY_PRIORITY, SCREEN_PRIORITY`. The director never returns coordinates or commands.
* Evidence decides which intents a span allows: speaker shots only on a face whose mouth activity is
  linked to the voice (never a guessed lone face); no face-only framing of a facecam overlay; validation
  walks disallowed intents down a ladder (landing on gameplay/screen priority in those layouts),
  widens single-subject intents when several participants/actions are REQUIRED (the story outranks the
  voice), limits punches and records every correction.
* One virtual-camera window per frame: `(cx, cy, h)` in source-normalized units. Crop and "fit with
  blurred fill" are one continuum, so every change is a smooth zoom; the solver only ever widens to
  keep required content, keeps neighbouring faces and HUD fully in or out, keeps faces above captions.
* Camera paths: eased inside a shot; cuts at shot boundaries/timeline segments; switching to a subject
  that is not in frame is a cut (no whip pans); widening to required content is a cut; follow uses a
  dead zone, exponential smoothing and a speed cap. Paths are verified per frame.
* Facecam + gameplay renders as a stacked layout (facecam panel above, gameplay window below).

## Rendering

Layers stay separate: `render_base` (timeline, source geometry, PCM audio with 8 ms join fades) →
compositor (camera/layout per frame, flash) → libass captions (never zoomed) → audio mix (sidechain-ducked
SFX, two-pass loudnorm, exact sample count) → one H.264/AAC encode. Caption placement reads the render
plan: captions sit on the seam of stacked frames, and the cold-open hook moves into the blurred band
above fitted content instead of covering a screen or gameplay.

## Caching

Stage signature = hash(stage name, explicit stage version, the settings section the stage consumes,
content ids of its dependency artifacts). Artifacts are hash-verified on reuse; several signatures per
stage are kept. A caption-style change re-runs `captions → render → qc → publish` only; a camera change
re-runs the edit compiler onwards; a speaker-name change re-runs identity and caption truth onwards.

## Final quality control

Deterministic checks on the rendered MP4: file validity (codecs, geometry, exact frame count, clean
decode), story completeness (beats + protected ranges shown), mandatory cold open (window contains the
peak, the peak recurs in the story), main-story restart, caption text/timing vs frozen truth and the
burned ASS, speaker ownership/labels/lanes, required content in frame, crop validity, camera stability,
no half-cut faces, the plan reached the pixels (structure correlation against a re-composed prediction),
captions reached the pixels, A/V sync per segment, cold-open audio = peak audio, no degraded stage.
The optional multimodal reviewer (`--reviewer`, one call) gets a SOURCE/RENDERED frame pair for each
story moment (cold-open peak, main restart, setup, escalation, payoff, reaction; at most 8) with the
evidence for that moment: beat, planned intent, REQUIRED content drawn as boxes, linked speaker, burned
captions and caption truth. It reports only fixed failure types (missing beat, weak/wrong cold open,
causal damage, cropped required visual, speaker/camera mismatch, caption contradiction, plan not in
pixels). Visual failures map to deterministic repairs (widen that span); story and caption failures stop
the gate. The driver runs at most ONE repair round, then fails loudly.

## Model routing

Roles (`python -m mimir models`): audio ears (`transcribe_fast`, `transcribe_primary`,
`transcribe_crosscheck`, `timing`, `diarize`), editorial Luna roles (scouting, drafting, observers,
effects) and Terra roles (story judge + one high-effort escalation, story expansion, hook judge, edit
director, final reviewer). Only two bounded multimodal passes exist: the VOD peak probe (≤ 6 regions ×
3 frames) and the selected-Short observer (≤ 12 frames).
