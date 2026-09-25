"""Caption regression harness (read-only over the live vod_output fixtures).

Regenerates clip ASS files from the REAL speaker profiles / timelines /
transcripts already present in the live root's vod_output, using the caption
code of an arbitrary code root (baseline or integrated final root). Output is
written ONLY under the given scratch output directory.

Records per fixture:
  * sha256 of the generated ASS
  * byte equality against the live vod_output ASS of the same caption version
  * the caption-truth word clock (word, start, end, printable label) that the
    renderer consumed -- presentation layers must never change this
  * the ASS event list with override tags stripped (timing + visible text)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path

LIVE_ROOT = Path(r"C:\Users\yusuf\Desktop\mimir_unified_clean_v3")
VOD = LIVE_ROOT / "vod_output"

TAG_RE = re.compile(r"\{[^}]*\}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ass_events(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        payload = line.split(":", 1)[1].lstrip()
        parts = payload.split(",", 9)
        if len(parts) < 10:
            continue
        body = parts[9]
        # Remove transparent (not-yet-spoken) words: {\alpha&HFF&}word{\r...}
        visible = re.sub(r"\{\\alpha&HFF&\}[^{]*", "", body)
        events.append({
            "start": parts[1],
            "end": parts[2],
            "style": parts[3],
            "visible_text": " ".join(TAG_RE.sub("", visible).split()),
        })
    return events


def fixtures() -> list[dict]:
    rows = []
    for profile in sorted((VOD / "speaker_captions").glob("*/clip_*_speakers_v*.json")):
        stem = profile.parent.name
        timeline = VOD / "timelines" / f"{stem}_timeline_v3.json"
        transcript = VOD / "transcripts" / f"{stem}.json"
        if not (timeline.is_file() and transcript.is_file()):
            continue
        clip_index = int(re.search(r"clip_(\d+)_", profile.name).group(1))
        rows.append({"stem": stem, "clip_index": clip_index, "profile": profile,
                     "timeline": timeline, "transcript": transcript})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("code_root")
    parser.add_argument("out_dir")
    args = parser.parse_args()

    code_root = Path(args.code_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(code_root))
    captions = importlib.import_module("ai.editor.captions")
    assert Path(captions.__file__).resolve().is_relative_to(code_root), captions.__file__

    results = []
    for fx in fixtures():
        stem, clip_index = fx["stem"], fx["clip_index"]
        record: dict = {"stem": stem, "clip_index": clip_index}
        try:
            transcript = json.loads(fx["transcript"].read_text(encoding="utf-8"))
            timeline_data = json.loads(fx["timeline"].read_text(encoding="utf-8"))
            profile = json.loads(fx["profile"].read_text(encoding="utf-8"))
            clip_timeline = captions.get_clip_timeline(timeline_data, clip_index)
            out_path = out_dir / stem / f"clip_{clip_index:02d}_captions.ass"
            captions.create_ass_for_clip(
                transcript=transcript,
                clip_timeline=clip_timeline,
                output_path=out_path,
                speaker_profile=profile,
            )
            data = out_path.read_bytes()
            record["ass_sha256"] = sha256_bytes(data)
            version = int(getattr(captions, "CAPTION_VERSION", 0))
            live = VOD / "captions" / stem / f"clip_{clip_index:02d}_captions_v{version}.ass"
            record["live_same_version_exists"] = live.is_file()
            record["byte_equal_live"] = live.is_file() and live.read_bytes() == data
            # Caption truth consumed by the renderer.
            duration = float(profile.get("clip_duration", 0.0) or 0.0) or float(
                clip_timeline.get("edited", {}).get("estimated_duration", 0.0))
            words = captions._profile_edited_words(profile, duration)
            trusted = captions._trusted_human_display_map(profile)
            prepared, windows = captions._prepare_adaptive_render_words(
                words, speaker_profile=profile, trusted_display_map=trusted)
            record["truth"] = [
                [w["word"], w["edited_start"], w["edited_end"], w.get("speaker_label", ""), w.get("speaker_role", "main")]
                for w in prepared
            ]
            record["truth_sha256"] = sha256_bytes(json.dumps(record["truth"], ensure_ascii=False).encode("utf-8"))
            record["events"] = ass_events(data.decode("utf-8-sig"))
            record["event_count"] = len(record["events"])
            record["status"] = "ok"
        except Exception as error:  # fixture-level isolation
            record["status"] = f"error: {type(error).__name__}: {error}"
        results.append(record)

    summary = {
        "code_root": str(code_root),
        "fixtures": len(results),
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "byte_equal_live": sum(1 for r in results if r.get("byte_equal_live")),
        "live_same_version": sum(1 for r in results if r.get("live_same_version_exists")),
    }
    (out_dir / "results.json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    for r in results:
        print(f"{r['status'][:60]:60s} ev={r.get('event_count','-')!s:>4} live_eq={r.get('byte_equal_live')!s:5s} {r['stem']}")
    return 0 if summary["ok"] == summary["fixtures"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
