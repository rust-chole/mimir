# MIMIR current root → Pro Edit V5 rebase — validation report

**Result: DONE.** Pro Edit V5 has been semantically ported onto this exact current root. Deliverable: `.\mimir_current_root_to_pro_edit_v5_patch.zip` (70 files: 64 added, 6 replaced, 0 deleted).
Extracting the patch over a clean copy of the original root gives a tree SHA256-identical to the integrated root (`mismatch_count = 0` with Python `zipfile` and with Windows PowerShell `Expand-Archive -Force`).
All validation below ran natively on this machine: Windows 11 Pro 10.0.26200, Python 3.14.7 (`.venv`), FFmpeg 9.0.1 full build with libass.

Pro Edit stays **OFF by default** (`MIMIR_PRO_EDIT=0`). With it off, the pipeline does not even import the new package, and your captions, speakers, story, intro, memes, pacing and rendering behave as before. Caption ASS output is byte-identical on 18 real fixtures. FFmpeg filter paths are escaped identically, except paths containing an apostrophe, which previously failed and now render. One more exception: the patch also fixes a pre-existing crash that currently stops MIMIR from starting at all (§2).

| Area | Status |
| --- | --- |
| Critical pre-existing defect (`intro_analyzer.py` SyntaxError: pipeline could not start) | DONE — fixed (2 delimiters only) |
| Pro Edit V5 package (41 modules, examples, models README) | DONE — 37 modules byte-identical to V5, 4 bound to the current root |
| Caption truth / speaker policy of the current root (captions **V24**) | DONE — Pro Edit presentation bound to V24's own functions |
| Pipeline integration (stage 11B, fallbacks, CLI, request signature) | DONE — your newer structural intro guard kept |
| FFmpeg subtitle path escaping (apostrophes) | DONE — both renderers; real burns on Windows |
| Tests (V5 suite adapted + new current-root suite) | DONE — 248 run, 0 failed, 0 skipped (OpenCV venv); 248 run, 0 failed, 24 skipped (production `.venv`) |
| Production `.venv` (no OpenCV) behaviour | DONE — verify PASS; the 23 pixel-evidence tests + optional HarfBuzz test reported as skipped |
| Real footage (kai16), Pro Edit ON vs OFF | DONE — structure, audio, truth and intro handoff verified |
| Patch reproduction + SHA256 | DONE — `mismatch_count = 0` |
| Secret / excluded-file scan of the patch | DONE — 0 findings |
| Live paid planner call, full paid pipeline run | SKIPPED — no paid API calls were made (§9) |
| Exact TikTok / Reels / Shorts UI geometry | BLOCKED_DATA — same as V5; no geometry invented |

---

## 1. Deliverables and how to apply

| File (repository root) | Purpose |
| --- | --- |
| `mimir_current_root_to_pro_edit_v5_patch.zip` | Extract over this repository root with overwrite enabled |
| `MIMIR_V5_REBASE_REPORT.md` | This report |

```powershell
cd "C:\Users\yusuf\Desktop\mimir_unified_clean_v3"
Expand-Archive -LiteralPath .\mimir_current_root_to_pro_edit_v5_patch.zip -DestinationPath . -Force
.\.venv\Scripts\python.exe -m pip install -r requirements.txt   # optional: OpenCV for the pixel-evidence layers
.\.venv\Scripts\python.exe .\verify_unified_clean.py
```

Patch SHA256: `c0b09dc882964fa5fe2ac73b7550f0409928c5775fed965813c94da9c21534b1` (365,626 bytes, 70 members)

**Rollback:** `_v5_rebase_work\baseline_root\` is the untouched copy of the 43 original source files. To revert, copy the 6 replaced files back from it and delete `ai\editor\pro_edit\`, `tests\` and `.env.example`.

---

## 2. Critical pre-existing defect (independent of V5)

**Before this patch, the current root could not start.** On this machine, `import ai.shorts_pipeline` fails with `SyntaxError: unexpected character after line continuation character (ai/editor/intro_analyzer.py, line 236)`, so `main.py` crashes before stage 1.

- **Cause:** `INTRO_INSTRUCTIONS = \"\"\"` (line 236) and the closing `\"\"\"` (line 327) have escaped triple-quote delimiters.
- **Origin:** a 2026-09-17 22:51–22:56 edit that rewrote `intro_analyzer.py`, `intro_renderer.py` and `teaser_analyzer.py`. The same edit also added a UTF-8 BOM, stray CRLFs and cp1254 mojibake.
- **Fix in the patch:** exactly those two delimiters become `"""`. Only 6 bytes change; the BOM, CRLF and all other bytes are untouched. After the fix, all 72 `ai.*` modules import.
- **Reported, not changed** (pre-existing, outside V5's scope):
  - `intro_analyzer.py:1983-1985`: the curly-quote check now compares against mojibake (`â€œ`), so `“`/`”` pass.
  - `intro_renderer.py:805-806`: `clean_hook_text` no longer normalizes `’`/`‘`.
  - Turkish console strings and parts of model prompts contain mojibake (cosmetic).

---

## 3. Inputs and provenance

| Item | Identity |
| --- | --- |
| Current root | `C:\Users\yusuf\Desktop\mimir_unified_clean_v3`: 43 source files, 5,087 files in total; SHA256 manifests captured at session start |
| V5 reference | `_v5_reference\MIMIR_SHORTS_V7_1_PRO_EDIT_V5.zip`, 744,958 bytes, SHA256 `e39f5b038ccfd6bf04d3d1d8937dc71c118fc8086ec7218352ff399cd1d40e66`, 110 entries. Used as a feature reference only |
| V5 lineage | `unified_clean_v4` → speaker-gap / identity-lock / TYLA / V7.1 post-intro fixes → Pro Edit V1–V5. V5 ran on Linux (Python 3.11/3.12, FFmpeg 6.1.1) and had no Windows run |

The current root already contained V5's pre-Pro-Edit fixes, plus newer work V5 never saw: captions **V24** (V5 carries V16), the Sep 11 structural intro guard, the Sep 17 teaser/intro peak work, and a newer `speaker_naming.py`.

The regression baseline is **this current root's own output**, not the old V7.1 MD5. With the syntax fix applied, the current code regenerates all 10 live `captions_v24.ass` files **byte for byte** from the real profiles in `vod_output`.

---

## 4. Semantic conflict map (V5 vs current root)

| Class | Files | Decision |
| --- | --- | --- |
| Identical in both | 28 (incl. `speaker_caption_support.py`, `meme_*`, `pacing*`, `timeline.py`, `video_brain/*`, `model_config.py`, `vod_processor.py`, `clip_analyzer.py`) | nothing to do |
| Current root newer | `captions.py` (V24 vs V16), `speaker_naming.py`, `teaser_analyzer.py`, `intro_analyzer.py`, `requirements.txt` base | **kept**; V5 copies not used |
| Semantic merge | `shorts_pipeline.py`, `caption_renderer.py`, `intro_renderer.py`, `verify_unified_clean.py`, `requirements.txt` | V5 deltas ported at the smallest boundary (§5) |
| New in V5 | `ai/editor/pro_edit/**` (47), `tests/**` (15), `.env.example` | added; 4 modules and 6 test files adapted, 1 test file new (§6, §7) |
| V5 docs | `README*.txt`, `README.md`, `LATEST_STABLE_BUILD_PROVENANCE.json` | **SKIPPED**: they describe the V7.1 lineage, including MD5 and test claims that are false for this root. This report replaces them |
| Current-only files | `V5_REBASE_TASK.md`, `fix_intro_structural_guard.py`, `*.bak`, `teaser_analyzer.py.patch`, `python_local.exe`, `pyvenv.cfg` | untouched, nothing deleted |

---

## 5. Integration batches (each compiled and gated before the next)

| Batch | Change | Proof |
| --- | --- | --- |
| **B0** | `intro_analyzer.py`: two escaped delimiters | compile + full import |
| **B1** | `escape_filter_path` in `caption_renderer.py` (result == V5 file) and `intro_renderer.py` (function body only; BOM / CRLF / mojibake / LOCKED PEAK validation untouched). A `'` in a path now closes the quote, escapes it and reopens | output identical for every path without an apostrophe; real FFmpeg burns from stress paths on Windows |
| **B2** | `ai/editor/pro_edit/` added; captions-V24 binding (§6) | compile; import without OpenCV; real-profile binding check |
| **B3** | `shorts_pipeline.py` (+428 / −45 lines, CRLF kept): feature switch, stage 11B, caption-render job indirection, intro source, post-composition handoff check, `--pro-edit/--no-pro-edit`, `enable_pro_edit` | every removed line is a known feature-off-equivalent refactor line; your **Sep 11 effective-main structural guard is kept** (V5 carried the older one); feature-off request payload hashes identically to before |
| **B4** | `verify_unified_clean.py` (V5 Pro Edit block + `utf-8-sig` AST read for BOM files + current-root suite), `requirements.txt` (+`opencv-python-headless>=4.9,<5`, optional at runtime; commented `uharfbuzz`), `.env.example` (every key empty) | every `.env.example` key is read by the integrated code; the OpenCV wheel installs on Python 3.14 win_amd64 |
| **B5** | tests: V5 suite adapted (§7) + new `tests/test_pro_edit_current_root.py` (12 tests) | §8 |

---

## 6. Current-root binding: captions V24 speaker policy

V5's caption presentation was written against captions V16. It turns raw profile roles into **permanent A/B(/C) lanes** and prints word-level labels. Your root's captions V24 deliberately stopped doing that:
- Only human-confirmed names are printed.
- A second lane exists only inside measured speech overlap.
- There is never a third lane.
- Groups break only on a visible name change.
- In a diarization hard failure, one lane never shows two events at once.

Measured on your real profiles, **unmodified V5** would have put 24 / 27 / 16 / 23 words on a permanent secondary lane for `kai tyla` / `speedcuce` / `speedtakla` / `test`. Your V24 renderer shows 0 / 2 / 0 / 0.

Binding (4 modules; the other 37 are byte-identical to V5):
- `caption_presentation.py`: token truth (text, start, end, raw speaker) keeps exact parity with `captions._profile_edited_words`. The display lane and label come from V24's **own** `_prepare_adaptive_render_words` / `_trusted_human_display_map`, and parity is enforced for both. Hard breaks follow V24's visible-label rule, per lane. In V24 hard-failure mode, same-lane pages never overlap. The manifest records the policy used.
- `caption_guard.py`: new `trusted_display_names()`, which defers to V24's trust function.
- `context.py`: planner speaker identities contain only confirmed names.
- `stage.py`: "verified names" for emphasis reasons come only from confirmed names; the brand-font glyph check also covers displayed names.

Real-profile check (17 of 18 fixtures; the 18th has `status=fallback`, like the baseline):
- Printed labels ⊆ human-trusted names.
- Secondary-lane words exactly equal V24's.
- No third lane.
- No same-lane overlap in the 5 hard-failure profiles.
- First caption instant equals the baseline (intro restart math).

---

## 7. Test adaptations (why each expectation changed)

| Test | Change | Reason |
| --- | --- | --- |
| 5 libass raster call sites (7 tests) | `subtitles=filename='…'` built with production `escape_filter_path` | a bare `C:` drive colon splits FFmpeg filter options (EINVAL). V5 only ever ran on Linux paths |
| `test_hard_breaks_are_never_crossed` | "one visible identity per page"; confirmed-name change still breaks | V24 group rule |
| label width + lane collision | uses human-confirmed name + measured overlap | V24 opens lane 2 only there |
| tertiary-lane test → `test_third_speaker_never_gets_a_third_lane` | asserts no third lane / no unconfirmed name | V24 forbids a third lane |
| golden V3.1 parity | 3 single-speaker cases still **byte-exact V3.1**; 2 multi-speaker cases must be byte-identical to the same words with speaker fields cleared | unconfirmed raw metadata must have zero effect under V24 |
| two-line stability raster | colour-independent geometry check (strong-ink columns, extent, best alignment shift = 0) | on Windows (real Arial Bold) one anti-aliased inter-glyph column crosses the threshold with the fill colour (64 vs 46); no reflow |
| custom brand font | Liberation (Linux) or `%WINDIR%\Fonts\arial.ttf`; resolved-path compare | now **runs** on Windows (was skipped) |
| AST hygiene scan | reads sources as `utf-8-sig` | current-root BOM files |
| stress-path burn test | Unicode `fontsdir` branch uses Windows Arial Bold when Liberation is absent | the branch was gated on a Linux font path and silently did not run on Windows; it now burns with `fontsdir='C\:/…/yazı tipleri ş'` natively |
| E2E harness | child runs with `PYTHONUTF8=1`; copy ignores `_v5_*`, `venv*`, `.env` | piped stdout on Windows is cp1254 and cannot print the pipeline banner |
| 23 pixel-evidence tests | `@fx.needs_opencv` → reported **skipped** without OpenCV | they measure cv2/numpy pixels; the no-OpenCV path is covered by a dedicated test |

---

## 8. Validation (all Windows-native on this machine)

| Check | Result |
| --- | --- |
| Compile every `.py` of the integrated root | 0 syntax errors |
| Import all 72 `ai.*` modules (`.venv` and OpenCV venv) | 72/72 both |
| `verify_unified_clean.py`, `MIMIR_VERIFY_E2E=1`, OpenCV test venv | **PASS**: all legacy MIMIR contracts + Pro Edit 248 run, 0 failed, 0 errors, 0 skipped |
| `verify_unified_clean.py`, `MIMIR_VERIFY_E2E=1`, **your production `.venv`** (no OpenCV) | **PASS**: all legacy MIMIR contracts + Pro Edit 248 run, 0 failed, 0 errors, 24 skipped (23 need OpenCV, 1 needs optional uharfbuzz) |
| `verify_unified_clean.py` on the **patch-reproduced root** (the exact artifact) | **PASS** in both: production `.venv` 248 run, 0 failed, 0 errors, 24 skipped; OpenCV venv 248 run, 0 failed, 0 errors, 0 skipped (root rebuilt from the original + `Expand-Archive -Force`) |
| E2E pipeline (real orchestration + real FFmpeg; fake paid model stages): off / on / render failure / planner down / stage crash / captions off / caption crash | PASS in both venvs; feature-off never imports Pro Edit and writes no Pro Edit artifacts |
| Caption regression, 18 real fixtures | ASS SHA256, word-clock truth and events **18/18 identical** to baseline; 10/10 byte-equal to live `captions_v24.ass` |
| FFmpeg stress-path burns (spaces, `çğıİöşü`, apostrophes, `[ , ;]`, Unicode fontsdir) | PASS on Windows FFmpeg 9.0.1 |
| No-OpenCV degradation (forced and natural) | tracker "unavailable", activity/background "failed", presentation still `v5_placement`, real render keeps frames / fps / size / audio |

**Real footage: kai16** (one consistent run, Sep 9 23:32–23:37; the other edited clips in `vod_output` belong to different runs and were not mixed in). Pro Edit ON with the rules planner in the OpenCV venv:
- **Stage:** status `ready`, with 3 camera moves over 272 frames and 8 caption pages at the `v5_placement` level.
- **Evidence layers:** activity analyzed, background analyzed, layout TALKING_HEAD, legibility 3 clear / 5 needing a shadow (applied). Haar tracking (YuNet model absent) found 6 tracks with 2 reacquisitions. Energy stayed within budget. No warnings.
- **Structure vs feature OFF:** identical final (955 frames, 15.917 s, 60 fps, 1920×1080).
- **Integrity:** main audio bit-exact, caption truth unchanged, intro handoff verified (Δ 1.7 ms), first caption instant equal.
- **Visual check** (frames inspected): same words at the same instant in both. OFF renders the vertical-PlayRes ASS squashed mid-frame on this landscape clip. ON renders correctly proportioned bottom captions with a legibility shadow. The mandatory hook intro is present, with the camera applied beneath it.

---

## 9. V5 capability matrix

| Capability | Status | Notes |
| --- | --- | --- |
| Pro Edit context / schema / validation | DONE | planner identities = confirmed names only |
| Deterministic caption presentation | DONE | bound to captions V24 (§6) |
| BrandProfile | DONE | `mimir_default` == V3.1 look; custom font validated on Windows |
| Platform safe-zone architecture | DONE (architecture) / BLOCKED_DATA (geometry) | `tiktok/reels/shorts` → `generic_conservative`, flagged unverified; no coordinates invented |
| Activity evidence | DONE | needs OpenCV; degrades without it |
| Static HUD / TEXT_LIKE occupancy | DONE | no OCR; real-footage precision/recall MANUAL_REQUIRED |
| Caption background analysis + adaptive legibility | DONE | absolute thresholds MANUAL_REQUIRED (visual calibration) |
| Face / story / action-aware placement | DONE | Haar faces (YuNet optional, not bundled) |
| Content-layout classification | DONE | weights MANUAL_REQUIRED |
| ActionRegionEvidence | PARTIAL | payoff/reaction spans × motion hotspot DONE; optional HOG person OFF by default; hands/objects BLOCKED_DEPENDENCY (no licensed local model; nothing downloaded) |
| Semantic emphasis reasons | DONE | "name" reason only for human-confirmed names |
| Bounded primitive library (12) | DONE | geometry law raster-verified on Windows |
| Global editorial energy coordinator | DONE (memes PARTIAL) | MIMIR's meme stage runs after Pro Edit and keeps its own policy |
| Safe-region round trip | DONE | |
| Planner / presentation cache separation | DONE | |
| Windows path / UTF-8 / subtitle escaping | DONE | real burns on Windows |
| V5 fallback ladder | DONE | incl. forced no-OpenCV path |
| V5 diagnostics / config / tests | DONE | `.env.example`, `[PRO_EDIT_*]` blocks, 12 new current-root tests |
| HarfBuzz shaper (optional) | DONE in test venv | `uharfbuzz` 0.56.2 test passes on Windows; not added to requirements |
| Live model planner | SKIPPED | no paid API call made; replay/fake-provider paths tested |
| Full paid MIMIR run | SKIPPED | offline E2E with real FFmpeg covers orchestration |

---

## 10. How to enable Pro Edit

Pro Edit is OFF unless you set `MIMIR_PRO_EDIT=1` in `.env`, or pass `--pro-edit` for one run. `.env.example` lists every key. The most useful ones:
- `MIMIR_PRO_EDIT_PLANNER=model|rules|static`: `model` makes one planner call per short via your existing OpenAI client; `rules` is free and deterministic.
- `MIMIR_PRO_EDIT_CAPTIONS=1`
- `MIMIR_CAPTION_PLATFORM_PROFILE=generic`
- `MIMIR_CAPTION_BRAND_PROFILE=mimir_default`
- `MIMIR_CAPTION_LEGIBILITY / _UI_OCCUPANCY / _LAYOUT=1`
- `MIMIR_PRO_EDIT_ENERGY=1`

Without OpenCV, Pro Edit still runs; the pixel-evidence layers simply degrade. With OpenCV installed (`pip install -r requirements.txt`), every layer is active.

---

## 11. Known limitations and pre-existing issues (not changed)

- `setup_optimal_v8.ps1` still references `verify_optimal_v8.py`, which does not exist in this root (pre-existing).
- Piping the pipeline's console output on Windows needs `PYTHONUTF8=1`, because the banner cannot be encoded in cp1254 (pre-existing; an interactive console is fine).
- With Pro Edit ON, the caption render starts after intro analysis instead of overlapping it (V5 design). Feature-off scheduling is unchanged.
- **Cache effect for videos you already processed:** stage caches are keyed on each module's path + size + mtime.
  - Because `intro_analyzer.py`, `intro_renderer.py` and `caption_renderer.py` change, the first re-run of an OLD source re-executes: caption render and intro render (FFmpeg only), intro analysis (**paid** Luna/Terra), and the downstream meme analysis and discovery (**paid**).
  - Clip analysis, transcription, caption accuracy, speakers and teaser keep their caches.
  - This is unavoidable: `intro_analyzer.py` must change for MIMIR to start at all, and its 2026-09-17 edit had already invalidated the intro-analysis cache.
  - New sources have no extra cost. While the feature is off, Pro Edit modules never enter any signature.

Work directory: `_v5_rebase_work\` keeps the original-root copy (rollback), SHA256 manifests, logs, scripts and real-footage evidence. It is not part of the patch and can be deleted.
