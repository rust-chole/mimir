"""Offline end-to-end harness for ai.shorts_pipeline.run_pipeline.

Runs the REAL pipeline orchestration and REAL FFmpeg stages (pacing cut,
audio pre-render, caption ASS + burn-in, intro render, publish) on a
synthetic source video. Only paid/interactive model stages are replaced by
deterministic fakes that write MIMIR-shaped artifacts.

Usage (always against a disposable COPY of the repo):
    python pipeline_harness.py --root <repo copy> --video <mp4> --mode <mode> --out result.json

Modes (there is ONE production path; modes only inject faults):
    run            full run (rules planner, memes off, fake no-finding reviewer)
    rerender       second run of the same source with --rerender (upstream caches kept)
    no_headline    the headline judge found no grounded line -> moving peak only
    broken_render  Pro Edit main render fails -> baseline render -> QC gate must REJECT
    stage_crash    Pro Edit preparation crashes -> baseline render -> QC gate must REJECT
    planner_down   planner unavailable (static camera) -> published as DEGRADED at worst
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


WORDS = [(f"word{i}", 0.5 + i * 0.45, 0.5 + i * 0.45 + 0.32) for i in range(58)]


def install_fakes(video: Path) -> None:
    from ai import vod_processor
    from ai.editor import (clip_analyzer, intro_analyzer, pacing, speaker_caption_support, speaker_naming,
                           teaser_analyzer, timeline)

    stem = video.stem

    def process_vod(video_path, *args, **kwargs):
        return write(Path(vod_processor.TRANSCRIPT_DIR) / f"{Path(video_path).stem}.json", {
            "version": 2, "source": {"video_path": str(video), "video_stem": stem},
            "words": [{"word": w, "start": s, "end": e} for w, s, e in WORDS], "segments": []})

    def create_clip_analysis(transcript_path, video_report_path=None, **kwargs):
        return write(Path(clip_analyzer.ANALYSIS_DIR) / f"{Path(transcript_path).stem}_clips.json", {
            "version": clip_analyzer.ANALYZER_VERSION,
            "clips": [{
                "title": "harness clip", "start": 0.0, "end": 30.0, "duration": 30.0, "score": 8.5,
                "terra_selected": True, "payoff_start": 14.0, "payoff_end": 17.0,
                "hook_text": "wait for it", "hook_type": "curiosity",
                "must_keep_ranges": [{"start": 13.8, "end": 18.5, "reason": "money"}],
                "anchor_moments": [{"start": 17.2, "end": 19.0, "type": "reaction", "strength": 8.0,
                                    "reason": "laugh"}],
                "strongest_anchor": {"start": 14.0, "end": 17.0, "strength": 8.0, "type": "punchline",
                                     "reason": "payoff"},
                "covered_moment_ids": [], "editor_notes": [],
            }]})

    def create_pacing_analysis(analysis_path, transcript_path, **kwargs):
        base = Path(analysis_path).stem.replace("_clips", "")
        return write(Path(pacing.PACING_OUTPUT_DIR) / f"{base}_pacing.json", {"version": 1, "clips": []})

    def create_edit_timeline(analysis_path, transcript_path, pacing_path=None, **kwargs):
        base = Path(analysis_path).stem.replace("_clips", "")
        return write(Path(timeline.TIMELINE_OUTPUT_DIR) / f"{base}_timeline_v{timeline.TIMELINE_VERSION}.json", {
            "version": 3, "source": {"video_path": str(video), "video_stem": stem},
            "timelines": [{
                "clip_index": 1, "title": "harness clip", "score": 8.5,
                "source": {"absolute_start": 0.0, "absolute_end": 30.0, "duration": 30.0},
                "edited": {"estimated_duration": 28.0, "minimum_story_duration": 14.0},
                "cut_ranges": [{"start": 5.0, "end": 6.0, "duration": 1.0},
                               {"start": 20.0, "end": 21.0, "duration": 1.0}],
                "payoff": {"type": "payoff", "source_start": 14.0, "source_end": 17.0,
                           "absolute_start": 14.0, "absolute_end": 17.0},
                "protected_ranges": [{"start": 13.8, "end": 18.5, "duration": 4.7, "reason": "money"}],
                "hook": {"type": "curiosity", "text": "wait for it"},
                "editorial": {"anchor_moments": [{"start": 17.2, "end": 19.0, "type": "reaction", "strength": 8.0}]},
                "events": [],
            }]})

    def create_speaker_scan_from_audio(audio_path, clip_index, reference_path=None):
        path = speaker_caption_support.get_speaker_scan_path(audio_path, clip_index)
        return write(Path(path), {"version": speaker_caption_support.SPEAKER_PROFILE_VERSION, "status": "ok",
                                  "mode": "single", "segments": [], "role_judge": {}})

    def resolve_interactive_speaker_names(speaker_profile_path, edited_clip_path, clip_index, creator_name=None):
        return {"profile_path": str(speaker_profile_path), "preview_path": None}

    def create_speaker_profile(edited_clip_path, clip_index, transcript_path=None, timeline_path=None,
                               speaker_scan_path=None):
        path = speaker_caption_support._output_path(edited_clip_path, clip_index)
        words = [(w, s, e) for w, s, e in WORDS if e < 27.9]
        return write(Path(path), {
            "version": speaker_caption_support.SPEAKER_PROFILE_VERSION, "status": "ok", "phase": "final_profile",
            "timing_basis": "exact_final_48k_audio", "clip_duration": 28.0, "mode": "single",
            "diarization_status": "ok", "display_labels": {}, "caption_quality": {},
            "words": [{"word": w, "edited_start": s, "edited_end": e, "speaker_raw": "A", "speaker_role": "main",
                       "speaker_label": ""} for w, s, e in words]})

    def analyze_teasers(timeline_path, transcript_path, clip_index, edited_video_path=None, video_report_path=None,
                        caption_profile_path=None):
        return write(Path(teaser_analyzer.get_output_path(timeline_path)), {
            "version": 1, "inputs": {"timeline": str(timeline_path)},
            "teasers": [{"clip_index": clip_index, "recommended": True,
                         "edited": {"teaser_start": 13.2, "teaser_end": 14.8, "duration": 1.6}}]})

    def analyze_intros(teaser_json_path, clip_index, manual_creator_name=None, video_report_path=None,
                       verified_names=(), caption_text=""):
        teaser = json.loads(Path(teaser_json_path).read_text(encoding="utf-8"))
        headline = "" if os.environ.get("HARNESS_NO_HEADLINE") == "1" else "HE SAID WORD TWELVE"
        gate = {"accepted": True, "no_headline": not headline}
        return write(Path(intro_analyzer.get_output_path(teaser_json_path)), {
            "version": 1, "inputs": {"timeline": teaser["inputs"]["timeline"], "teaser": str(teaser_json_path)},
            "intros": [{"clip_index": clip_index, "recommended": True, "score": 9.0 if headline else 0.0,
                        "title": "harness clip", "intro_text": headline, "quality_gate": gate}]})

    def fake_reviewer():
        return lambda prompt, frames: {"findings": [], "summary": f"harness reviewer saw {len(frames)} frames"}

    vod_processor.process_vod = process_vod
    clip_analyzer.create_clip_analysis = create_clip_analysis
    pacing.create_pacing_analysis = create_pacing_analysis
    timeline.create_edit_timeline = create_edit_timeline
    speaker_caption_support.create_speaker_scan_from_audio = create_speaker_scan_from_audio
    speaker_naming.resolve_interactive_speaker_names = resolve_interactive_speaker_names
    speaker_caption_support.create_speaker_profile = create_speaker_profile
    teaser_analyzer.analyze_teasers = analyze_teasers
    intro_analyzer.analyze_intros = analyze_intros
    from ai.editor import final_review

    final_review.default_reviewer = fake_reviewer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--mode", required=True, choices=["run", "rerender", "no_headline", "broken_render",
                                                          "stage_crash", "planner_down"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)
    for key in list(os.environ):
        if key.startswith("MIMIR_PRO_EDIT") or key in ("OPENAI_API_KEY", "MIMIR_V6", "MIMIR_CAPTION_ENTITIES"):
            os.environ.pop(key)
    os.environ["MIMIR_PRO_EDIT_PLANNER"] = "static" if args.mode == "planner_down" else "rules"
    if args.mode == "no_headline":
        os.environ["HARNESS_NO_HEADLINE"] = "1"
    video = Path(args.video).resolve()
    install_fakes(video)
    from ai import shorts_pipeline

    if args.mode == "broken_render":
        from ai.editor.pro_edit import stage
        from ai.editor.pro_edit.errors import EditRenderError

        def broken(**kwargs):
            raise EditRenderError("harness-injected render failure")

        stage.render_camera_captions = broken

    if args.mode == "stage_crash":
        from ai.editor.pro_edit import stage

        def crash(request):
            raise RuntimeError("harness-injected stage crash")

        stage.prepare_pro_edit = crash

    kwargs = dict(force=True, enable_memes=False, enable_video_brain=False, keep_temp=True)
    if args.mode == "rerender":
        kwargs.update(force=False, rerender=True)
    summary: dict = {"mode": args.mode}
    try:
        result = shorts_pipeline.run_pipeline(video, **kwargs)
    except shorts_pipeline.ShortsPipelineError as error:
        state_file = shorts_pipeline._state_path(video)
        state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.is_file() else {}
        summary.update({
            "error": str(error), "run_status": state.get("run_status"),
            "publish_status": state.get("publish_status"),
            "stages": {k: v.get("status") for k, v in state.get("stages", {}).items()},
            "v6_state": state.get("v6"), "final_qc": state.get("final_qc"),
            "published_exists": (shorts_pipeline.PUBLISHED_DIR / f"{video.stem}_short.mp4").exists(),
            "rejected": sorted(p.name for p in shorts_pipeline.REJECTED_DIR.glob("*"))
            if shorts_pipeline.REJECTED_DIR.exists() else [],
        })
        Path(args.out).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return 0
    final = Path(result["final_output"])
    state = json.loads(Path(result["state_file"]).read_text(encoding="utf-8"))
    summary.update({
        "status": result.get("status"),
        "final_output": str(final),
        "final_md5": md5(final),
        "warnings": result.get("warnings", []),
        "pro_edit": result.get("pro_edit"),
        "stages": {k: v.get("status") for k, v in state.get("stages", {}).items()},
        "pro_edit_artifacts": sorted(p.name for p in (root / "vod_output" / "pro_edit").rglob("*") if p.is_file())
        if (root / "vod_output" / "pro_edit").exists() else [],
        "v6": result.get("v6"),
        "v6_state": state.get("v6"),
        "final_qc": state.get("final_qc"),
        "final_timeline": json.loads(Path(state["stages"]["intro_final_base"]["path"]).with_name(
            Path(state["stages"]["intro_final_base"]["path"]).stem + ".timeline.json").read_text(encoding="utf-8")),
    })
    Path(args.out).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
