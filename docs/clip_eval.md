# Measuring clip selection

MIMIR's story / clip selection (Stage 2: Luna scouts money moments and candidates, Terra
picks the clip) changes with every prompt, model or setting tweak, and so far nobody could
say whether a change made the picks better. `ai.clip_eval` fixes that: a set of real VODs
with **blind** human labels (the moments a person would publish as a Short), and one command
that runs Stage 2 on exactly the same frozen input and scores its picks with standard
highlight-detection metrics. A change is kept only when it wins on these numbers.

Everything lives in `vod_output/clip_eval/` (never committed: it points at your videos and
holds copies of their transcripts). The pipeline never imports `ai.clip_eval`.

**Money.** Only four commands can call a paid model: `transcribe`, `run`, `compare --judge`
and `calibrate --judge`. Each one prints its estimate with `--dry-run`, refuses to call
anything without `--max-usd`, refuses when the estimate is above the cap, and checks the
remaining budget before every single model call (see [Costs](#8-costs)). Everything else
is free. Prices are in `ai/clip_eval/pricing.json`; they are **unverified and set high on
purpose** (see the file), so real bills should come in below the estimates.

## 1. Build the golden set (a person, once)

Aim for **8-10 VODs**: different creators, games and lengths, a quiet one, a chaotic one,
and ones where MIMIR picked badly. Put **at least 3 in `holdout`**: holdout videos are only
used for the final yes/no (the gate) and never reach an AI (see [Holdout](#9-holdout-and-phase-2)).

```powershell
# one case per source VOD; the Stage 1 transcript (vod_output\transcripts\<name>.json) is frozen into it
python -m ai.clip_eval prepare "D:\VODs\Kai's stream 2026-10-01.mp4"
python -m ai.clip_eval prepare "D:\VODs\holdout\İzmir şaka gecesi.mp4" --split holdout
python -m ai.clip_eval prepare "D:\VODs\more"                      # every video in a folder
python -m ai.clip_eval status
```

`prepare` calls no model. It hashes the video (SHA256, cached), copies the Stage 1
transcript and, when production had one, the whole-VOD visual-facts report Stage 2 receives
(`vod_output\video_brain\<name>_video_report_v6.json`) into the case, so every run sees the
same bytes even after the pipeline re-transcribes. `--no-visual-report` makes a case
transcript-only; `--refresh` re-freezes the inputs (labels are never touched); `--proxy`
adds a small browser-playable copy (for MKV / TS or huge files).

**No transcript yet?** `status` and `prepare` print the one-time command. It runs the real
Stage 1 (`vod_processor.process_vod`) and is paid (~$0.54 per hour of video):

```powershell
python -m ai.clip_eval transcribe "D:\VODs\Kai's stream 2026-10-01.mp4" --dry-run
python -m ai.clip_eval transcribe "D:\VODs\Kai's stream 2026-10-01.mp4" --max-usd 2
python -m ai.clip_eval prepare "D:\VODs\Kai's stream 2026-10-01.mp4" --refresh
```

### Label blind

```powershell
python -m ai.clip_eval serve                     # opens http://127.0.0.1:8766
$env:MIMIR_CLIP_EVAL_ANNOTATOR = "ayse"          # optional: who is labeling (default "primary")
```

The editor plays the video with its timestamped transcript beside it (30-second windows,
click a line to jump). Mark In (`I`) and Out (`O`), pick a grade, write a short note, *Add
moment*. It shows **only the video, the transcript and your own labels**: never MIMIR's
picks, runs, published shorts, review sheets or another annotator's labels. That is on
purpose: an AI suggestion anchors the annotator, so labeling is blind (this is the big
difference from `caption_eval`, which starts from MIMIR's captions as a draft).

Rules:

- Mark each moment as **the Short you would publish**: from the first second you would
  keep (setup / hook) to the last (payoff + reaction), typically 15-60 s. The metrics
  compare whole clips (temporal IoU), so a 5-second punchline mark would make every good
  clip look wrong.
- **2-6 moments per video.** `great` = you would post it; `ok` = usable, not your first
  choice; `bad` = looks like a highlight but is not (a near miss MIMIR might fall for).
- **Never-pick** ranges: what must never become a Short (sponsor reads, private
  information, dead air, AFK, broken audio).
- Tick **Finished** after watching the whole video. Unfinished labels are never scored,
  and `run` only runs videos whose labels are finished.

**One text file instead of the editor:**

```powershell
python -m ai.clip_eval export                    # -> vod_output\clip_eval\label_sheet.txt
# watch each video; under its block write lines like
#   moment: 12:30 - 13:05 | great | chair breaks, chat explodes
#   never: 1:02:00 - 1:04:30 | sponsor read
# and change [ ] to [x]
python -m ai.clip_eval import vod_output\clip_eval\label_sheet.txt
```

**A second annotator** (optional) gives the human ceiling: `serve --annotator second` (or
`export --annotator second`) labels the same videos without ever seeing the first person's
labels. The first person to label a video owns its gold; the others are compared with it.

**Weak labels (optional).** Published shorts already carry a human decision:

```powershell
python -m ai.clip_eval harvest-reviews           # reads vod_output\final\*_short.review.md + .qc.json
```

It records Accept / Reject (and QC rejects) with the published clip's range per case, in
`vod_output\clip_eval\weak_labels\`. They are reported on their own line ("accepted picks
picked again"), never mixed into the golden metrics, and never shown while labeling.

## 2. Run the baseline

```powershell
python -m ai.clip_eval run --variant baseline --dry-run                    # what it would cost
python -m ai.clip_eval run --variant baseline --repeats 3 --max-usd 30     # dev videos, 3 runs each
python -m ai.clip_eval runs                                                 # find the run name
```

`--stage clip` (the default) runs **only Stage 2**: the real
`clip_analyzer.create_clip_analysis` on the case's frozen transcript and visual report, in a
separate process per job (model routing is read at import, so each variant needs its own
process). Before it starts, the worker points `clip_analyzer.ANALYSIS_DIR` at the job's own
folder: the production cache `vod_output\analysis\<name>_clips.json` (which the pipeline
reuses by signature without looking inside) is fingerprinted before and after, and a job
that changed it is an error. No file under `ai/` is modified for this.

`baseline` is exactly what `python main.py` runs today: the code defaults plus your `.env`
(any Stage 2 key your `.env` sets is printed and recorded). LLM output varies, so
`--repeats N` runs each video N times and the report shows the spread next to the mean.

Each run is stored under `vod_output\clip_eval\runs\<variant>__<configuration hash>\`:
`run.json` (variant, its overrides, the routing, Stage 2 settings and limits, code
fingerprints of `clip_analyzer` / `model_config` / `openai_client` exactly like the
pipeline's `_module_fingerprint`, git commit, money spent), and per video and repeat the
raw Stage 2 output, the worker log, token usage, cost and wall time. Results are cached per
video and repeat: running the same thing again costs nothing, `--repeats 5` after 3 runs
only the new two, and `score` / `compare` never call a model. Changing a prompt, a model, a
setting or the variant file gives a new hash, so a new run: the one you compare against is
never overwritten. `--force` re-runs.

`--split holdout` / `--split all` run the holdout videos (for the gate). `--stage full` runs
the whole unmodified pipeline from a byte-identical copy of the code in
`vod_output\clip_eval\workspaces\` (its own `vod_output`, so nothing of yours is touched and
every stage really runs: transcription, Gemini, captions, render). It is expensive (see
[Costs](#8-costs)) and only for when you need end-to-end numbers.

## 3. Run a variant

A variant is a file of environment overrides; no code changes:

```powershell
New-Item -ItemType Directory -Force vod_output\clip_eval\variants | Out-Null
Set-Content -Encoding utf8 vod_output\clip_eval\variants\judge-high.env "MIMIR_CLIP_JUDGE_REASONING=high"
Set-Content -Encoding utf8 vod_output\clip_eval\variants\scout-terra.env "MIMIR_CLIP_SCOUT_MODEL=gpt-5.6-terra"
python -m ai.clip_eval variants                  # lists them and the Stage 2 keys
python -m ai.clip_eval run --variant judge-high --repeats 3 --dry-run
python -m ai.clip_eval run --variant judge-high --repeats 3 --max-usd 30
```

Stage 2 reads `MIMIR_CLIP_SCOUT_MODEL`, `MIMIR_CLIP_SCOUT_REASONING`, `MIMIR_CLIP_JUDGE_MODEL`,
`MIMIR_CLIP_JUDGE_REASONING`, `MIMIR_CLIP_JUDGE_ESCALATION_REASONING`, `MIMIR_EDITOR_MODEL`,
`MIMIR_CLIP_HIGH_ESCALATION_MARGIN`, `MIMIR_BOUNDARY_POLISH_MAX_EXTEND` / `_TRIM`,
`MIMIR_STORY_VISUAL_ORIGIN` and `MIMIR_SOURCE_PEAK_*`; other keys only matter with
`--stage full` (the plan says so). Secrets are refused in variant files.

**A prompt change** is a code change: edit `ai/editor/clip_analyzer.py` on a branch, then
`run --variant baseline` again. The new code fingerprint makes it a new run next to the old
one.

## 4. Compare (and the gate)

```powershell
python -m ai.clip_eval compare baseline__1c41f2e2 judge-high__e488c163
python -m ai.clip_eval compare baseline__1c41f2e2 judge-high__e488c163 --gate
```

A is the reference, B the candidate (a unique prefix such as `judge-high` is enough). The
report (`vod_output\clip_eval\compare.md`) lists what differs between the two
configurations, then **dev and holdout separately**: every metric as mean ± spread across
repeats, B - A with a paired bootstrap 95% confidence interval over videos and a verdict
(`better`, `WORSE`, `may be noise`), per-video diffs, and the 3 most regressed videos with
their labeled moments, both published clips' transcript and B's feedback. With fewer than
20 videos every section says how much one video moves a rate.

`--gate` exits 1 when holdout HIT@1 or Recall@3 of B is lower than A's by more than
`--gate-threshold` (default 0.1), and also when no holdout video was scored by both runs
(no evidence, no pass). A trusted LLM judge (below) can also fail it; an untrusted one is
ignored.

## 5. LLM judge and calibration (optional)

```powershell
python -m ai.clip_eval compare baseline__1c41f2e2 judge-high__e488c163 --judge --dry-run
python -m ai.clip_eval compare baseline__1c41f2e2 judge-high__e488c163 --judge --max-usd 1
```

For each dev video and repeat, the judge gets the two published clips (transcript with
times from the clip start, duration, the words just outside each cut) and says which makes
the better Short, or tie. No absolute scores. Every pair is asked in both orders; if the two
answers disagree, it is a tie (position bias). The rubric is four narrow checks with good /
bad examples: a hook in the first 3 s, makes sense without the stream, the payoff is inside
the clip, a clean cut. The judge is Gemini (`MIMIR_CLIP_EVAL_JUDGE_MODEL`, default your
`VIDEO_BRAIN_MODEL`, else `gemini-3.7-flash`): another model family than MIMIR's OpenAI
generators (an OpenAI model is refused). Verdicts are cached; only dev videos are ever sent.

**Calibration is mandatory before the judge counts.** You answer ~30 pairs blind (which
side is which run is hidden), the judge answers the same pairs, and Cohen's kappa decides:

```powershell
python -m ai.clip_eval calibrate baseline__1c41f2e2 judge-high__e488c163   # adds the pairs, opens the browser
python -m ai.clip_eval calibrate --text                                    # or answer in the terminal
python -m ai.clip_eval calibrate --judge --dry-run
python -m ai.clip_eval calibrate --judge --max-usd 1                       # agreement + kappa
```

Below kappa 0.6 (or with fewer than 10 answered pairs) every report marks the judge
**UNTRUSTED** and the gate ignores it. Calibration is per judge model and prompt version.

## 6. What the numbers mean

The published clip is the one `shorts_pipeline._select_clip` would publish (Terra's
`terra_selected` clip); the top 3 are the next candidates in Stage 2's order. tIoU is the
temporal intersection over union of two time ranges. Rates are 0-1.

| metric | meaning |
| --- | --- |
| HIT@1 (0.3 / 0.5 / 0.7) | the published clip overlaps a `great` moment with tIoU >= threshold |
| Recall@1, Recall@3 | share of the `great` moments captured by the top 1 / 3 clips (tIoU >= 0.5) |
| mAP (0.3 / 0.5 / 0.7) | average precision of the ranked candidates, `great` = positive (QVHighlights style) |
| mean best tIoU | how well the published clip lines up with its best `great` moment |
| holds >= 90% of a great moment | the published clip contains the moment even if it is longer |
| start / end error | seconds between the published clip's ends and the matched moment's (median, p90; + = later) |
| never-pick violation | the published clip (or any candidate) overlaps a never-pick range by > 0.25 s |
| on a 'bad' moment | the published clip lands on a moment graded `bad` |
| starts / ends mid-word | a cut falls inside a transcript word (word timestamps, 50 ms tolerance) |
| duration inside bounds | inside Stage 2's own 12-55 s and inside the video; target 22-42 s reported apart |
| valid Stage 2 output | the analysis JSON is well-formed (version, clips with start < end, one selected clip) |
| changes between repeats | share of repeat pairs whose published clips differ (tIoU < 0.5), and their mean tIoU |
| cost / seconds | USD and wall time per video and repeat |
| human ceiling | tIoU between two annotators on the same great moment: the best a model can be expected to do |

Videos without a `great` moment have no HIT / Recall / mAP (shown as n/a, not 0). Every
video also gets a plain-text **feedback** line ("Published clip 8:32.1-9:07.7 matches great
moment 1 'chair breaks' with tIoU 0.82. It ends 4.3 s before the labeled end (the
payoff/reaction is cut).").

**Decision rule.** A change is better when, on the dev videos both runs scored, HIT@1 and
mean best tIoU go up, Recall@3 and never-pick violations do not get worse, the CI says it is
not noise (or the repeats agree), and `--gate` passes on holdout. With 8-10 videos one video
is a 0.1-0.125 swing in any rate: read the confidence intervals, not the third decimal.

## 7. Files

```
vod_output\clip_eval\
  cases\<case>\meta.json, transcript.json, visual_report.json, labels\<annotator>.json, proxy.mp4
  variants\<name>.env
  runs\<run>\run.json, hyp\<case>\r<k>.json / .analysis.json / .log, score.json, report.md, feedback.dev.json
  weak_labels\<case>.json      judge\cache\       calibration\pairs.json, human.json, result.json
  label_sheet.txt   compare.md
```

## 8. Costs

Estimates at the conservative prices in `pricing.json` (`--dry-run` prints the exact figure
for your videos; once a video has been run, its measured cost replaces the heuristic).

| step | command | estimated cost |
| --- | --- | --- |
| prepare, status, serve, export / import, split, variants, runs, score, compare, feedback, harvest-reviews | | $0 (no model call) |
| Stage 1 for a video without a transcript | `transcribe ... --max-usd 2` | ~$0.009 per minute of video (~$0.54 per hour) |
| Stage 2, one video, one repeat | `run ... --max-usd` | ~$0.70 (30 min VOD) to ~$1.30 (3 h VOD); up to ~$2 worst case |
| baseline on 7 dev videos x 3 repeats | `run --repeats 3 --max-usd 30` | ~$15-27 |
| holdout, 3 videos x 3 repeats | `run --split holdout --repeats 3 --max-usd 15` | ~$7-12 |
| LLM judge | `compare --judge --max-usd 1` | ~$0.015 per call, 2 calls per video and repeat: ~$0.65 for 7 x 3 |
| judge calibration | `calibrate --judge --max-usd 1` | ~$0.90 for 30 pairs |
| whole pipeline | `run --stage full` | ~$4 (30 min) to ~$9 (3 h) per video and repeat, of which Stage 1 + Gemini (~$0.8-5) are reserved up front because they cannot be metered call by call |

The cap is enforced inside the worker: before every OpenAI SDK call (Responses, chat,
transcription) the meter estimates that call and refuses it if the rest of the budget
cannot cover it. A refused call stops the job (recorded as `budget_stopped`, never scored)
and the run; finished jobs stay cached, so a higher `--max-usd` continues where it stopped.
A model that has no price in `pricing.json` is refused, because its cost cannot be capped.

## 9. Holdout and Phase 2

The holdout exists to catch overfitting, so it never reaches an AI. In code
(`ai/clip_eval/guard.py`): the judge and calibration refuse holdout videos; the only
AI-readable output is

```powershell
python -m ai.clip_eval feedback judge-high__e488c163   # -> runs\<run>\feedback.dev.json
```

which holds the dev videos' scores and feedback lines and is checked before it is written:
it fails if it mentions any holdout case id, video file, hash, label note or a stretch of a
holdout transcript. A future optimization loop (GEPA / DSPy-style reflective prompt search)
must read only `guard.optimization_cases()` and `feedback.dev.json`. `report.md` and
`compare.md` include holdout results and say at the top that they are for people only.

## 10. Your first run (one video)

```powershell
cd C:\path\to\mimir
.\.venv\Scripts\Activate.ps1
python -m pytest tests\test_clip_eval.py -q                 # hermetic, no model call
python -m ai.clip_eval prepare "D:\VODs\<one VOD>.mp4"      # needs its vod_output\transcripts\<name>.json
python -m ai.clip_eval serve                                 # label it blind, tick Finished, Save
python -m ai.clip_eval status                                # labels: ready
python -m ai.clip_eval run --variant baseline --dry-run      # read the estimate and the routing
python -m ai.clip_eval run --variant baseline --max-usd 2    # one Stage 2 run (~$0.7-1.3)
python -m ai.clip_eval runs
notepad vod_output\clip_eval\runs\<run name>\report.md
```
