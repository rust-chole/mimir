"""Offline end-to-end harness for ai.shorts_pipeline.run_pipeline.

Runs the REAL pipeline orchestration and REAL FFmpeg stages (pacing cut,
audio pre-render, caption ASS + burn-in, intro render, publish) on a
synthetic source video. Only paid/interactive model stages are replaced by
deterministic fakes that write MIMIR-shaped artifacts.

Usage (always against a disposable COPY of the repo):
    python pipeline_harness.py --root <repo copy> --video <mp4> --mode off|on|on_broken|on_planner_down|on_stage_crash --out result.json
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

    def analyze_teasers(timeline_path, transcript_path, clip_index, edited_video_path=None, video_report_path=None):
        return write(Path(teaser_analyzer.get_output_path(timeline_path)), {
            "version": 1, "inputs": {"timeline": str(timeline_path)},
            "teasers": [{"clip_index": clip_index, "recommended": True,
                         "edited": {"teaser_start": 13.2, "teaser_end": 14.8, "duration": 1.6}}]})

    def analyze_intros(teaser_json_path, clip_index, manual_creator_name=None, video_report_path=None):
        teaser = json.loads(Path(teaser_json_path).read_text(encoding="utf-8"))
        return write(Path(intro_analyzer.get_output_path(teaser_json_path)), {
            "version": 1, "inputs": {"timeline": teaser["inputs"]["timeline"], "teaser": str(teaser_json_path)},
            "intros": [{"clip_index": clip_index, "recommended": True, "score": 9.0, "title": "harness clip",
                        "intro_text": "WAIT FOR IT", "quality_gate": {"accepted": True}}]})

    vod_processor.process_vod = process_vod
    clip_analyzer.create_clip_analysis = create_clip_analysis
    pacing.create_pacing_analysis = create_pacing_analysis
    timeline.create_edit_timeline = create_edit_timeline
    speaker_caption_support.create_speaker_scan_from_audio = create_speaker_scan_from_audio
    speaker_naming.resolve_interactive_speaker_names = resolve_interactive_speaker_names
    speaker_caption_support.create_speaker_profile = create_speaker_profile
    teaser_analyzer.analyze_teasers = analyze_teasers
    intro_analyzer.analyze_intros = analyze_intros


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--mode", required=True, choices=["baseline", "off", "on", "on_broken", "on_planner_down", "on_stage_crash",
                                                                "on_captions_off", "on_caption_crash"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)
    for key in list(os.environ):
        if key.startswith("MIMIR_PRO_EDIT") or key == "OPENAI_API_KEY":
            os.environ.pop(key)
    if args.mode == "on":
        os.environ["MIMIR_PRO_EDIT_PLANNER"] = "rules"
    if args.mode == "on_captions_off":
        # planner unavailable (no key) + caption presentation disabled
        os.environ["MIMIR_PRO_EDIT_CAPTIONS"] = "0"
    video = Path(args.video).resolve()
    install_fakes(video)
    from ai import shorts_pipeline

    if args.mode == "on_broken":
        from ai.editor.pro_edit import stage
        from ai.editor.pro_edit.errors import EditRenderError

        def broken(**kwargs):
            raise EditRenderError("harness-injected render failure")

        os.environ["MIMIR_PRO_EDIT_PLANNER"] = "rules"
        stage.render_camera_captions = broken

    if args.mode == "on_caption_crash":
        from ai.editor.pro_edit import stage

        def caption_crash(**kwargs):
            raise RuntimeError("harness-injected caption presentation failure")

        os.environ["MIMIR_PRO_EDIT_PLANNER"] = "rules"
        stage.build_presentation = caption_crash

    if args.mode == "on_stage_crash":
        from ai.editor.pro_edit import stage

        def crash(request):
            raise RuntimeError("harness-injected stage crash")

        stage.prepare_pro_edit = crash

    kwargs = dict(force=True, enable_memes=False, enable_video_brain=False, keep_temp=True)
    if args.mode != "baseline":
        kwargs["enable_pro_edit"] = args.mode != "off"
    result = shorts_pipeline.run_pipeline(video, **kwargs)
    final = Path(result["final_output"])
    state = json.loads(Path(result["state_file"]).read_text(encoding="utf-8"))
    summary = {
        "mode": args.mode,
        "final_output": str(final),
        "final_md5": md5(final),
        "warnings": result.get("warnings", []),
        "pro_edit": result.get("pro_edit"),
        "stages": {k: v.get("status") for k, v in state.get("stages", {}).items()},
        "pro_edit_artifacts": sorted(p.name for p in (root / "vod_output" / "pro_edit").rglob("*") if p.is_file())
        if (root / "vod_output" / "pro_edit").exists() else [],
        "pro_edit_imported": "ai.editor.pro_edit" in sys.modules,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
