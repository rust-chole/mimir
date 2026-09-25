"""Real-footage Windows validation of the integrated Pro Edit V5 (kai16 run, Sep 9 23:32-23:37).

Inputs (read-only, one consistent run): live vod_output transcript / timeline /
edited clip / speaker profile v21 / teaser / intro / video-brain report.
Sandbox: a copy of the integrated root (runtime_root); every output goes under
it or under an output folder whose name contains spaces + Turkish characters.

A) Pro Edit OFF: captions V24 ASS -> existing caption_renderer burn -> existing
   intro_renderer composition (mandatory intro)          = baseline final.
B) Pro Edit ON (rules planner, all V5 evidence layers, OpenCV venv):
   prepare_pro_edit -> render_with_fallback (camera + presentation ASS) ->
   render_intro_source -> the SAME intro_renderer composition -> handoff and
   final-duration verification exactly like the pipeline's _verify_pro_edit_intro.
Checks: caption truth untouched, audio stream bit-exact, frame count / fps /
resolution preserved, intro/main structure identical to A, first caption instant.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
RUNTIME = WORK / "runtime_root"
LIVE = WORK.parent / "vod_output"
OUT = WORK / "real_kai16" / "gerçek çıktı ş dir"
STEM = "kai16"
CLIP = LIVE / "edited_clips" / STEM / "clip_01_The Age Question Goes Very Wrong_edited.mp4"
PROFILE = LIVE / "speaker_captions" / STEM / "clip_01_speakers_v21.json"
TIMELINE = LIVE / "timelines" / f"{STEM}_timeline_v3.json"
TEASERS = LIVE / "teasers" / f"{STEM}_teasers.json"
INTROS = LIVE / "intros" / f"{STEM}_intros.json"
ANALYSIS = LIVE / "analysis" / f"{STEM}_clips.json"
REPORT = LIVE / "video_brain" / "clip_01_The Age Question Goes Very Wrong_edited_video_report_v6.json"


def md5_audio(path: Path) -> str:
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-c", "copy", "-f", "md5", "-"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    return out.stdout.strip()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    shutil.copytree(WORK / "final_root", RUNTIME)
    # The existing renderers resolve media under <root>/vod_output: stage the run's clip there.
    staged = RUNTIME / "vod_output" / "edited_clips" / STEM / CLIP.name
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CLIP, staged)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    sys.path.insert(0, str(RUNTIME))

    from ai import shorts_pipeline as sp
    from ai.editor import caption_renderer, captions, intro_renderer
    from ai.editor.pro_edit import config as pe_config
    from ai.editor.pro_edit import intro_timeline, media, stage
    from ai.editor.pro_edit.caption_guard import CaptionIntegrity

    report: dict = {"inputs": {k: str(v) for k, v in {"clip": CLIP, "profile": PROFILE, "timeline": TIMELINE,
                                                      "teasers": TEASERS, "intros": INTROS, "report": REPORT}.items()}}
    timeline_data = json.loads(TIMELINE.read_text(encoding="utf-8"))
    clip_timeline = captions.get_clip_timeline(timeline_data, 1)
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    ass = captions.create_ass_for_clip({}, clip_timeline, OUT / "clip_01_captions_v24.ass", profile)
    integrity_before = CaptionIntegrity.capture(PROFILE, ass)

    # ---------------- A) feature OFF (existing renderers) ----------------
    t0 = time.perf_counter()
    base_preview = Path(caption_renderer.render_captioned_clip(TIMELINE, 1, caption_path=ass))
    base_final = Path(intro_renderer.run_renderer(intro_json_path=INTROS, clip_index=1, edited_clip_path=staged,
                                                  captioned_preview_path=base_preview, caption_path=ass)[0])
    base_final_copy = OUT / "A_feature_off_final.mp4"
    shutil.copy2(base_final, base_final_copy)
    report["A_seconds"] = round(time.perf_counter() - t0, 1)

    # ---------------- B) feature ON (Pro Edit V5, rules planner) ----------------
    t0 = time.perf_counter()
    config = pe_config.load_config(override_enabled=True, environ={"MIMIR_PRO_EDIT_PLANNER": "rules"})
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    _index, selected_clip = sp._select_clip(analysis, 1)
    teaser_record = sp._package_clip(TEASERS, "teasers", 1)
    intro_record = sp._package_clip(INTROS, "intros", 1)
    prep = stage.prepare_pro_edit(stage.ProEditRequest(
        config=config, timeline_path=TIMELINE, clip_index=1, edited_clip_path=staged, caption_path=ass,
        output_path=OUT / "B_main_pro_edit.mp4", artifact_dir=OUT / "pro_edit artifacts", input_signature="kai16-real",
        analysis_clip=selected_clip, speaker_profile_path=PROFILE, video_report_path=REPORT,
        teaser_record=teaser_record, intro_record=intro_record, intro_output_path=OUT / "B_intro_source.mp4",
        force=True))
    prep.diagnostics.emit()
    report["B_prepare"] = {"status": prep.status, "reason": prep.reason, "warnings": prep.warnings,
                           "captions": {k: v for k, v in prep.captions.items() if k not in ("ass",)},
                           "tracking": prep.tracking, "energy": prep.energy, "intro_ready": prep.intro_ready,
                           "model_calls": prep.model_calls}
    outcome: dict = {}
    main = stage.render_with_fallback(prep, caption_file=ass, baseline=lambda: base_preview, outcome=outcome)
    intro_outcome: dict = {}
    intro_source = stage.render_intro_source(prep, clean_clip=staged, outcome=intro_outcome)
    pro_final = Path(intro_renderer.run_renderer(intro_json_path=INTROS, clip_index=1,
                                                 edited_clip_path=intro_source, captioned_preview_path=main,
                                                 caption_path=ass)[0])
    pro_final_copy = OUT / "B_feature_on_final.mp4"
    shutil.copy2(pro_final, pro_final_copy)
    report["B_render"] = {"main": outcome, "intro_source": intro_outcome}
    report["B_seconds"] = round(time.perf_counter() - t0, 1)

    # Handoff verification exactly like the pipeline (_verify_pro_edit_intro).
    clean_info, main_info = media.probe_media(staged), media.probe_media(main)
    after = intro_timeline.build_intro_timeline(teaser_record, caption_path=ass,
                                                clean_duration=media.probe_media(intro_source).duration_s,
                                                main_duration=main_info.duration_s)
    frame = clean_info.fps.frame_duration
    handoff = {"status": "not_checked"}
    if prep.intro_timeline is not None:
        intro_timeline.verify_handoff(prep.intro_timeline, after, frame_duration=frame)
        delta = intro_timeline.verify_final_duration(after, media.probe_media(pro_final_copy).duration_s,
                                                     frame_duration=frame)
        handoff = {"status": "verified", "final_duration_delta_s": round(delta, 4), "timeline": after.to_dict()}
    report["B_handoff"] = handoff

    # ---------------- invariants ----------------
    integrity_before.verify(PROFILE, ass, stage="real-footage validation")           # truth untouched
    a, b = media.probe_media(base_final_copy), media.probe_media(pro_final_copy)
    src, pm = media.probe_media(staged), media.probe_media(main)
    checks = {
        "caption_truth_unchanged": True,
        "main_frames_fps_size_preserved": (pm.frame_count, pm.fps.fraction, pm.width, pm.height)
        == (src.frame_count, src.fps.fraction, src.width, src.height),
        "main_audio_bit_exact": md5_audio(main) == md5_audio(staged),
        "final_structure_equal_to_feature_off": (a.frame_count, a.fps.fraction, a.width, a.height)
        == (b.frame_count, b.fps.fraction, b.width, b.height)
        and abs(a.duration_s - b.duration_s) <= b.fps.frame_duration + 0.01,
        "first_caption_instant_equal": prep.presentation_ass is None or abs(
            (intro_renderer.get_first_caption_start(prep.presentation_ass) or 0)
            - (intro_renderer.get_first_caption_start(ass) or 0)) <= 0.011,
    }
    report["media"] = {"feature_off_final": {"frames": a.frame_count, "duration": a.duration_s, "fps": a.fps.fraction,
                                             "size": [a.width, a.height]},
                       "feature_on_final": {"frames": b.frame_count, "duration": b.duration_s, "fps": b.fps.fraction,
                                            "size": [b.width, b.height]},
                       "main_pro_edit": {"frames": pm.frame_count, "duration": pm.duration_s}}
    report["checks"] = checks
    (OUT / "real_footage_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str),
                                                  encoding="utf-8")
    print(json.dumps({"B_prepare": {k: report["B_prepare"][k] for k in ("status", "reason", "intro_ready")},
                      "captions": report["B_prepare"]["captions"], "tracking": report["B_prepare"]["tracking"],
                      "energy": report["B_prepare"]["energy"], "render": report["B_render"],
                      "handoff": {k: v for k, v in handoff.items() if k != "timeline"}, "media": report["media"],
                      "checks": checks, "seconds": [report["A_seconds"], report["B_seconds"]]},
                     indent=1, ensure_ascii=False, default=str))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
