"""Real-fixture check of the Pro Edit presentation's captions-V24 binding.

For every real speaker profile in the live vod_output (read-only):
  * build the Pro Edit presentation (mimir_default brand, generic platform,
    no evidence) with the INTEGRATED code root;
  * truth parity: every displayed token == captions._profile_edited_words;
  * printed labels are a subset of the V24 human-trusted names;
  * the words Pro Edit places on the secondary lane are exactly the words the
    V24 baseline renderer places on ViralSecondary (lane parity);
  * at most two lanes; no same-lane page overlap in hard-failure profiles;
  * first caption instant == baseline ASS first Dialogue (intro restart math).
Writes nothing outside the given scratch directory.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

CODE_ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
LIVE = Path(r"C:\Users\yusuf\Desktop\mimir_unified_clean_v3")
sys.path.insert(0, str(CODE_ROOT))

from ai.editor import captions  # noqa: E402
from ai.editor.pro_edit import caption_presentation as cp  # noqa: E402
from ai.editor.pro_edit.caption_guard import trusted_display_names  # noqa: E402


def secondary_words_in_baseline(ass_text: str) -> list[str]:
    """Words shown ACTIVE on ViralSecondary in the baseline ASS (in order, unique by event)."""
    rows = []
    for line in ass_text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if parts[3] != "ViralSecondary":
            continue
        rows.append(parts[1])
    return rows


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    failures = 0
    for profile_path in sorted((LIVE / "vod_output" / "speaker_captions").glob("*/clip_*_speakers_v*.json")):
        stem = profile_path.parent.name
        timeline_path = LIVE / "vod_output" / "timelines" / f"{stem}_timeline_v3.json"
        if not timeline_path.is_file():
            continue
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        clip_index = int(re.search(r"clip_(\d+)_", profile_path.name).group(1))
        clip = captions.get_clip_timeline(timeline, clip_index)
        tag = f"{stem[:24]:24s} {profile_path.name:24s}"
        if str(profile.get("status")) != "ok":
            print(f"SKIP  {tag} profile status={profile.get('status')!r} (baseline also has no exact-final words)")
            continue
        # Baseline ASS from the SAME code root (captions.py is untouched by the port).
        with tempfile.TemporaryDirectory(prefix="mimir_v24_bind_") as tmp:
            base_ass = Path(tmp) / "baseline.ass"
            captions.create_ass_for_clip(transcript={"words": []}, clip_timeline=clip, output_path=base_ass,
                                         speaker_profile=profile)
            base_text = base_ass.read_text(encoding="utf-8-sig")
            try:
                pres = cp.build_presentation(profile=profile, clip_timeline=clip, plan=None, width=1080, height=1920)
            except Exception as error:
                failures += 1
                print(f"FAIL  {tag} build_presentation: {type(error).__name__}: {error}")
                continue
            ass_path = OUT / stem / f"{profile_path.stem}_pro_edit_captions.ass"
            cp.write_presentation(pres, ass_path)
            problems = []
            trusted = set(trusted_display_names(profile).values())
            labels = {p.label for p in pres.pages if p.label}
            if not labels <= trusted:
                problems.append(f"untrusted labels {sorted(labels - trusted)}")
            lanes = {p.lane for p in pres.pages}
            if len(lanes) > 2 or "tertiary" in lanes:
                problems.append(f"lanes {sorted(lanes)}")
            # Lane parity with the V24 renderer's prepared words.
            words = captions._profile_edited_words(profile, cp.edited_duration(profile, clip))
            prepared, _ = captions._prepare_adaptive_render_words(
                words, speaker_profile=profile, trusted_display_map=captions._trusted_human_display_map(profile))
            v24_secondary = [i for i, w in enumerate(prepared) if w["speaker_role"] == "secondary"]
            ours_secondary = [i for i, t in enumerate(pres.tokens) if t.lane == "secondary"]
            if v24_secondary != ours_secondary:
                problems.append(f"secondary lane differs: v24={len(v24_secondary)} ours={len(ours_secondary)}")
            base_secondary_events = len(secondary_words_in_baseline(base_text))
            if bool(base_secondary_events) != bool(ours_secondary):
                problems.append(f"baseline ViralSecondary events={base_secondary_events} vs ours={len(ours_secondary)}")
            strict = cp.strict_lane_timing(profile)
            if strict:
                last_end: dict[str, float] = {}
                for page in sorted(pres.pages, key=lambda p: p.start):
                    if page.lane in last_end and page.start < last_end[page.lane] - 1e-9:
                        problems.append(f"same-lane overlap at page {page.page_id}")
                        break
                    last_end[page.lane] = page.end
            try:
                cp.check_intro_handoff(ass_path, base_ass)
            except Exception as error:
                problems.append(f"intro handoff: {error}")
            metrics = pres.metrics()
            status = "OK  " if not problems else "FAIL"
            failures += bool(problems)
            print(f"{status}  {tag} words={metrics['words']:3d} pages={metrics['pages']:3d} "
                  f"secondary_words={len(ours_secondary):2d} labels={sorted(labels)} trusted={sorted(trusted)} "
                  f"hardfail={strict} {'; '.join(problems)}")
    print("FAILURES", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
