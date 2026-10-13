"""Caption evaluation: measure caption accuracy on real shorts before changing the caption stack.

Every change to how MIMIR hears words has to beat the current numbers on the same
cases, with the same gold and the same normalization. The workflow:

    python -m ai.caption_eval prepare --final vod_output/final      # one case per published short
    (a person listens to each case's audio.wav and corrects gold.txt word for word)
    python -m ai.caption_eval status                                 # which golds are finished
    python -m ai.caption_eval run published                          # what MIMIR captioned: the baseline
    python -m ai.caption_eval run stack                              # the current stack, re-run
    python -m ai.caption_eval run stack --name stack-asr --set MIMIR_CAPTION_PRIMARY_PROVIDER=qwen3_asr_flash
    python -m ai.caption_eval run ear:openai:gpt-4o-transcribe       # one ear on its own
    python -m ai.caption_eval compare published stack stack-asr      # side by side

Modules: ``cases`` (case folders, prepare), ``metrics`` (normalization, alignment,
scores), ``systems`` (what can be run), ``runner`` (cached runs, scoring, compare),
``report`` (markdown), ``cli``. Data lives in ``vod_output/caption_eval`` (never
committed). See docs/caption_eval.md.
"""
