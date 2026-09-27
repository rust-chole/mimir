# MIMIR

VOD → vertical Short. MIMIR finds the strongest complete causal story in a long stream/VOD, opens the
Short with the story's real peak (mandatory cold open), then restarts and shows how that peak happened:

```
COLD OPEN (peak) → SETUP → ESCALATION → PAYOFF → REACTION
```

## Quick start

```bash
pip install -r requirements.txt          # plus FFmpeg with libass on PATH
cp .env.example .env                      # set OPENAI_API_KEY
python -m mimir doctor                    # checks ffmpeg filters, OpenCV, API key
python -m mimir run path/to/vod.mp4
```

Useful options:

| option | effect |
| --- | --- |
| `--speaker-names S1=Alex,S2=Sam` | confirmed names (only confirmed names are ever printed) |
| `--interactive` | when several voices are ambiguous, play `speaker_preview` audio and ask for names |
| `--entities "Alex,Valorant"` / `--creator NAME` | verified spellings for caption truth |
| `--story-index N` | use the N-th ranked story candidate |
| `--rerun captions,render` | re-run named stages (dependents follow through signatures) |
| `--record DIR` / `--replay DIR` | record every model response / replay them (regression runs, no live calls) |
| `--reviewer` | enable the bounded multimodal final reviewer |
| `--sfx-library DIR` | curated local SFX (`transition/`, `impact/`, `disbelief/`, ... category folders) |

Outputs: `output/<source>/<title>.mp4` + a JSON manifest. Every stage artifact lives under
`workspace/jobs/<job>/stages/<stage>/<signature>/` with a `record.json` (inputs, params, hashes, ledger notes).

`python -m mimir models` prints the model routing; override any role with `MIMIR_ROUTE_<ROLE>=model[:effort]`.

## Tests

```bash
python -m pytest -m "not render"      # fast unit tests
python -m pytest                      # + golden end-to-end renders (needs espeak-ng, scikit-image)
```

The golden set builds seven synthetic VODs with real synthesized speech, real face imagery, gameplay,
IRL action and UI footage, runs the full production pipeline with a scripted stand-in for the model
roles, renders real MP4s and requires the final quality gate to pass. See `ARCHITECTURE.md`.
