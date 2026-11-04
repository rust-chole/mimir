"""Clip evaluation: measure how good MIMIR's clip picks (Stage 2) are, against blind human labels.

Every change to how MIMIR finds the story (a prompt in ai/editor/clip_analyzer.py, a model or effort in
ai/model_config.py, a Stage 2 setting) has to beat the current numbers on the same videos, with the same
labels and the same metrics. The workflow:

    python -m ai.clip_eval prepare "D:\\VODs\\stream.mp4" [--split holdout]   # one case per source video
    python -m ai.clip_eval serve                    # a person labels each video BLIND (video + transcript)
    python -m ai.clip_eval status                   # which labels are finished
    python -m ai.clip_eval run --variant baseline --dry-run          # what it would cost
    python -m ai.clip_eval run --variant baseline --repeats 3 --max-usd 5
    python -m ai.clip_eval run --variant judge-high --repeats 3 --max-usd 5  # variants/judge-high.env
    python -m ai.clip_eval compare baseline__<hash> judge-high__<hash> [--gate] [--judge --max-usd 1]

Or as one text file: ``export`` -> fill it -> ``import``. Optional: ``harvest-reviews`` (weak labels from
review sheets), ``calibrate`` (is the LLM judge trustworthy?), ``feedback`` (dev-only AI-readable export).

Modules: ``cases`` (case folders, labels, prepare), ``metrics`` (Tier 1 scores), ``variants`` (env-file
overrides), ``pricing`` (estimates + the --max-usd meter), ``worker`` (one job per subprocess), ``runner``
(cached runs, scoring, compare, gate), ``report`` (markdown), ``editor`` (label editor), ``sheet`` (one-file
labels), ``reviews`` (weak labels), ``judge`` + ``calibrate`` (Tier 2), ``guard`` (holdout never reaches an
AI), ``cli``. Data lives in ``vod_output/clip_eval`` (never committed). The pipeline never imports this
package. See docs/clip_eval.md.
"""
from ai.clip_eval.cli import main

if __name__ == "__main__":         # importing the package (e.g. a module walk) never runs the CLI
    raise SystemExit(main())
