"""Diagnostic (read-only): why does speaker<->face linking fail on real footage?

Runs the EXISTING tracker (Haar/YuNet via load_detector, LK flow, 10 fps samples)
on a real edited clip and prints per-track evidence and the existing Pearson
association scores against the real speaker profile words.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ai.editor.pro_edit import speaker_link  # noqa: E402
from ai.editor.pro_edit.caption_guard import caption_words, load_profile  # noqa: E402
from ai.editor.pro_edit.media import probe_media  # noqa: E402
from ai.editor.pro_edit.vision.detectors import load_detector  # noqa: E402
from ai.editor.pro_edit.vision.frames import DEFAULT_MAX_WIDTH, DEFAULT_SAMPLE_FPS, iterate_frames  # noqa: E402
from ai.editor.pro_edit.vision.tracker import MultiSubjectTracker, TrackerConfig  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    clip, profile_path = Path(sys.argv[1]), Path(sys.argv[2])
    media = probe_media(clip)
    detector, reason = load_detector()
    print("detector:", getattr(detector, "name", None), "|", reason)
    tracker = MultiSubjectTracker(detector, TrackerConfig())
    tracks = tracker.run(iterate_frames(clip, media.width, media.height, sample_fps=DEFAULT_SAMPLE_FPS,
                                        max_width=DEFAULT_MAX_WIDTH), ())
    print("stats:", tracker.stats, "| clip", media.width, "x", media.height, round(media.duration_s, 2), "s")
    words = caption_words(load_profile(profile_path))
    spk = speaker_link.speaking_intervals(words)
    print("speakers:", {k: round(sum(b - a for a, b in v), 2) for k, v in spk.items()}, "s of speech")
    for track in tracks:
        acts = [s.activity for s in track.samples if s.activity is not None]
        sources = {}
        for s in track.samples:
            sources[s.source] = sources.get(s.source, 0) + 1
        print(f"track {track.subject_id}: n={len(track.samples)} t=[{track.samples[0].t:.2f},{track.samples[-1].t:.2f}] "
              f"cx~{statistics.median(s.cx for s in track.samples):.2f} cy~{statistics.median(s.cy for s in track.samples):.2f} "
              f"w~{statistics.median(s.width for s in track.samples):.3f} conf~{statistics.median(s.confidence for s in track.samples):.2f} "
              f"activity n={len(acts)} mean={(sum(acts) / len(acts)) if acts else 0:.4f} sources={sources}")
    scores = speaker_link.association_scores(tracks, words)
    print("association scores (track, speaker) -> pearson:", {f"{k[0]}|{k[1]}": round(v, 3) for k, v in scores.items()})
    linked = speaker_link.associate_speakers(tracks, words)
    print("links:", [(t.subject_id, t.speaker_id, t.speaker_confidence, t.speaker_evidence) for t in linked if t.speaker_id])
    out = ROOT / "_pass2_work" / "evidence" / f"diag_tracks_{clip.stem[:20]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([t.summary() | {"samples": [[round(s.t, 2), round(s.cx, 3), round(s.cy, 3), round(s.width, 3),
                                                          round(s.confidence, 2), s.activity, s.source] for s in t.samples]}
                               for t in tracks], ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
