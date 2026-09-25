"""Batch B3: port the Pro Edit V5 integration into the CURRENT ai/shorts_pipeline.py.

Facts established before this script (diff of LF-normalized files):
  * current pipeline == V5 base pipeline, EXCEPT the structural intro guard,
    where the current root is NEWER (Sep 11 fix: effective main duration after
    intro_renderer's main restart). V5 still carries the old V7.1 guard.
  * every other hunk is the additive, feature-flagged Pro Edit wiring.

Merge = V5 pipeline with its old guard block replaced by the CURRENT guard
block (V5's post-guard ``_verify_pro_edit_intro`` call is kept after it).
Proof obligations checked here:
  1. The current guard block survives byte-for-byte (LF-normalized).
  2. Every line REMOVED from the current file is one of the known lines that
     V5's feature-off-equivalent refactors replace (request signature dict,
     caption-render start block, direct render call, edited_clip_path ->
     intro_source_path, "done" -> caption_render_status).
  3. The result compiles; line endings are written back as CRLF (current style).
"""
from __future__ import annotations

import difflib
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
FINAL = WORK / "final_root" / "ai" / "shorts_pipeline.py"
BASE = WORK / "baseline_root" / "ai" / "shorts_pipeline.py"
V5 = WORK.parent / "_v5_reference" / "extracted" / "MIMIR_SHORTS_V7_1_PRO_EDIT_V5" / "ai" / "shorts_pipeline.py"

CUR_GUARD_START = "    # Structural contract: final base must be longer than the EFFECTIVE main\n"
CUR_GUARD_END = '            "Introsuz cikti yayinlanmadi."\n        )\n'
V5_GUARD_START = "    # Structural contract: final base must be longer than the captioned main,\n"
V5_GUARD_END = '            "İntrosuz çıktı yayınlanmadı."\n        )\n'

# Lines of the current file that V5's Pro Edit wiring legitimately replaces.
EXPECTED_REMOVED = {
    "    return _hash(",
    "        {",
    '            "pipeline_version": PIPELINE_VERSION,',
    '            "source": source,',
    '            "creator_name": creator_name or "",',
    '            "clip_index": clip_index,',
    '            "enable_memes": enable_memes,',
    '            "enable_video_brain": enable_video_brain,',
    '            "video_brain_model": video_brain_model if enable_video_brain else "",',
    '            "code_signature": _code_signature(enable_video_brain, video_brain_model),',
    "        }",
    "    )",
    "    caption_render_cached = book.reusable(",
    '        "caption_render",',
    "        caption_render_sig,",
    "        lambda: _valid_file(captioned_preview_path, MIN_VIDEO_BYTES),",
    "        output=captioned_preview_path,",
    "        freshness=[",
    "            edited_clip_path,",
    "            caption_path,",
    "            timeline_path,",
    "            *_module_paths([caption_renderer]),",
    "        ],",
    "    if SAFE_PIPELINE_PARALLEL and not caption_render_cached:",
    "        caption_render_executor = ThreadPoolExecutor(",
    "            max_workers=1,",
    '            thread_name_prefix="mimir-caption-render",',
    "        )",
    "        caption_render_future = caption_render_executor.submit(",
    "            runtime_profiler.timed,",
    '            "caption_render_worker",',
    "            caption_renderer.render_captioned_clip,",
    "            timeline_path=timeline_path,",
    "            clip_index=selected_clip_index,",
    "            caption_path=caption_path,",
    "                rendered_caption = caption_renderer.render_captioned_clip(",
    "                    timeline_path=timeline_path,",
    "                    clip_index=selected_clip_index,",
    "                    caption_path=caption_path,",
    "                )",
    '            "done",',
    "            edited_clip_path,",
    "                    edited_clip_path=edited_clip_path,",
}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cur_raw = BASE.read_bytes()
    if b"\r\n" not in cur_raw or cur_raw.count(b"\n") != cur_raw.count(b"\r\n"):
        raise SystemExit("expected a pure-CRLF current pipeline")
    cur = cur_raw.decode("utf-8").replace("\r\n", "\n")
    v5 = V5.read_bytes().decode("utf-8")
    if "\r" in v5:
        raise SystemExit("expected an LF V5 pipeline")

    c0, c1 = cur.index(CUR_GUARD_START), cur.index(CUR_GUARD_END) + len(CUR_GUARD_END)
    v0, v1 = v5.index(V5_GUARD_START), v5.index(V5_GUARD_END) + len(V5_GUARD_END)
    current_guard = cur[c0:c1]
    if "calculate_main_restart_seconds" not in current_guard or "main_effective" not in current_guard:
        raise SystemExit("current guard block is not the effective-main guard")
    merged = v5[:v0] + current_guard + v5[v1:]
    if merged.count(current_guard) != 1:
        raise SystemExit("guard block not unique after merge")
    after_guard = merged[merged.index(current_guard) + len(current_guard):]
    if not after_guard.startswith("\n    if pro_prep_ref is not None and pro_prep_ref.intro_timeline is not None:\n"):
        raise SystemExit("V5 post-composition intro verification is not directly after the guard")

    removed = [line[1:] for line in difflib.unified_diff(cur.split("\n"), merged.split("\n"), lineterm="", n=0)
               if line.startswith("-") and not line.startswith("---")]
    unexpected = [line for line in removed if line not in EXPECTED_REMOVED]
    if unexpected:
        raise SystemExit("unexpected current-root lines removed:\n" + "\n".join(unexpected))

    compile(merged, str(FINAL), "exec", dont_inherit=True)
    FINAL.write_bytes(merged.replace("\n", "\r\n").encode("utf-8"))
    added = sum(1 for line in difflib.unified_diff(cur.split("\n"), merged.split("\n"), lineterm="", n=0)
                if line.startswith("+") and not line.startswith("+++"))
    print(f"B3 shorts_pipeline.py merged: +{added} / -{len(removed)} lines vs current; "
          f"current structural guard kept; CRLF preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
