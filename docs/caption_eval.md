# Measuring caption accuracy

MIMIR's caption stack changed several times without a way to tell whether a change
helped. `ai.caption_eval` fixes that: a set of real shorts with human-checked
transcripts (the **gold**), and one command per system that scores it on exactly
the same audio, gold and normalization. From now on a caption change is merged
only when it beats the current numbers.

Everything lives in `vod_output/caption_eval/` (never committed: it holds audio).

## 1. Build the gold set (a person, once)

Aim for **30-50 shorts**: different creators, games, noise levels, one- and
two-speaker clips, and the ones where captions went visibly wrong.

```powershell
# every published short (uses the final timeline + speaker profile the pipeline left on disk)
python -m ai.caption_eval prepare --final vod_output\final
# or one edited clip (run the pipeline with --keep-temp so it is not deleted)
python -m ai.caption_eval prepare --clip "vod_output\edited_clips\<vod>\clip_01_..._edited.mp4" --names "Kai,Tyla"
python -m ai.caption_eval status
```

Each case gets `audio.wav` (exactly what the caption stack hears), MIMIR's captions
as a draft, and `gold.txt` = that draft with a `# TODO` line. Then:

```powershell
python -m ai.caption_eval serve        # opens http://127.0.0.1:8765
```

The editor plays the audio (0.75x helps), shows MIMIR's words as chips (click one to
play from there; dashed = MIMIR itself was unsure), and saves the gold. Rules:

- Write **exactly what is said**: slang, profanity, repeats, cut-off words. Do not
  fix grammar. Do not trust the draft; it is what is being measured.
- Fillers (uh/um), capitalization and punctuation do not matter; spelling of
  **names** does (write them as they should appear on screen).
- `[inaudible]` where even you cannot tell; it matches anything at no cost.
- Put the names said in the short in the names field (they mark name errors).
- Tick **Finished** only after a full word-for-word pass. Unfinished cases are not scored.

A text editor works too: edit `gold.txt`, delete the `# TODO` line.

**One file for everything.** Instead of the editor, the whole set can be one text file:

```powershell
python -m ai.caption_eval export                    # -> vod_output\caption_eval\gold_sheet.txt
# correct each block under its "# MIMIR:" line, change [ ] to [x]
python -m ai.caption_eval import vod_output\caption_eval\gold_sheet.txt
python -m ai.caption_eval score-sheet vod_output\caption_eval\gold_sheet.txt
```

The sheet holds text only (no audio), so it can be sent to whoever analyses it;
`score-sheet` scores MIMIR's captions against it with nothing else on disk.

Optional timing reference: `python -m ai.caption_eval align-gold` times the finished
gold with the forced aligner (machine reference, labelled as such). A human
`gold_words.json` (`[{"word", "start", "end"}, ...]`, one row per gold word) wins.

## 2. Run and compare systems

```powershell
python -m ai.caption_eval systems                     # what can be run
python -m ai.caption_eval run published               # what MIMIR captioned: the baseline, no API cost
python -m ai.caption_eval run stack                   # today's stack, re-run on the same audio
python -m ai.caption_eval run ear:qwen_omni           # one ear alone
python -m ai.caption_eval run ear:qwen3_asr_flash:names
python -m ai.caption_eval run ear:openai:gpt-4o-transcribe
python -m ai.caption_eval run stack --name stack-asr --set MIMIR_CAPTION_PRIMARY_PROVIDER=qwen3_asr_flash
python -m ai.caption_eval compare published stack stack-asr
```

Hypotheses are cached per case (same system, overrides, audio, caption settings and
caption code), so `score` and `compare` after a gold edit cost nothing, and a changed
`.env` or caption code re-runs instead of reusing stale results. `--force` re-runs.
`--set KEY=VALUE` overrides a caption setting for one run (never a secret); model
routing such as `MIMIR_CAPTION_JUDGE_MODEL` is read at startup, so change it in `.env`.

Reports: `vod_output/caption_eval/runs/<name>/report.md`, and `compare.md`.

## 3. What the numbers mean

| metric | meaning |
| --- | --- |
| WER | (substituted + missing + extra words) / gold words, after normalization |
| critical word error rate | names, numbers and negations wrong or missing, plus invented numbers / negations |
| dropped phrases | 3+ consecutive gold words the system lost (omissions) |
| invented phrases | 3+ consecutive words nobody said (hallucinations) |
| word onset error | start-time error of correctly heard words (only with gold times) |
| 'uncertain' flag precision / recall | do the words MIMIR marks uncertain match its real errors? |
| coverage flag precision | do the VAD speech-coverage flags sit on real errors? |

Normalization is the caption stack's own definition of a harmless difference
(`ai/caption_stack/normalize.py`): case, punctuation, fillers, "ok"/"okay",
"gonna"/"going to", "don't"/"do not", "twenty-one"/"21" are never errors.

**Decision rule.** A change is better when, on the same scored cases, WER and the
critical word error rate both go down, dropped/invented phrases do not go up, and
no case regresses by more than 2 WER points without an explanation. Small sets are
noisy: with fewer than ~20 cases, treat differences under ~1 WER point as a tie.

## 4. The experiments, in order

1. **Baseline**: `run published` and `run stack`. Read the "Most frequent
   substitutions" and "Errors to listen to" sections: are the errors names,
   dropped phrases, numbers, or ordinary words? That decides what comes next.
2. **Which ear hears best** (one call per case each):
   `ear:qwen_omni`, `ear:qwen3_asr_flash` (or local `ear:qwen3_asr`),
   `ear:openai:gpt-4o-transcribe`, each also with `:names` where available.
3. **The stack with the best ear**: `run stack --set MIMIR_CAPTION_PRIMARY_PROVIDER=...`.
4. **Context**: `--set MIMIR_CAPTION_CONTEXT="<game, channel topic, recurring jargon>"`
   against the same run without it.
5. **Flags**: if the coverage / uncertain flags turn out precise, they can drive an
   automatic re-listen; if not, they stay review hints.

Each step is a `run` plus a `compare`; only what wins on the gold set becomes the default.
