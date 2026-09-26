"""Golden regression set: real synthetic VODs through the ONE production pipeline to a real rendered MP4.

Every scenario must publish a Short that passes the final quality gate (which
itself inspects the rendered pixels, audio and burned captions); each scenario
then asserts the behaviour it exists to protect.
"""
from __future__ import annotations

import pytest

from mimir.edit.framing import Geometry
from mimir.media.probe import probe
from tests.e2e.harness import artifact, run_scenario
from tests.synth import scenarios

pytestmark = pytest.mark.render

RESULTS: dict[str, tuple] = {}


def run(name: str, root):
    if name not in RESULTS:
        scenario = next(make() for make in scenarios.ALL if make().name == name)
        RESULTS[name] = (scenario, *run_scenario(scenario, root / name))
    return RESULTS[name]


def common(scenario, result):
    assert result.qc["passed"], result.qc["failed"]
    info = probe(result.published_video, count_frames=True)
    assert (info.width, info.height) == (1080, 1920)
    assert info.video_codec == "h264" and info.audio_codec == "aac"
    timeline = artifact(result, "timeline", "timeline")
    assert info.frame_count == timeline["frame_count"]
    assert timeline["segments"][0]["kind"] == "cold_open"
    cold = artifact(result, "cold_open", "cold_open")
    first = timeline["segments"][0]
    assert first["source_start"] <= cold["peak"]["start"] + 0.05
    story = artifact(result, "story", "story")
    assert [b["role"] for b in story["beats"]][:4] == ["setup", "escalation", "payoff", "reaction"]
    assert not story["completeness_problems"]
    notes = [n for n in result.report.notes() if n.get("level") == "degraded"]
    assert notes == []
    return timeline, cold, story


@pytest.mark.parametrize("name", [make().name for make in scenarios.ALL])
def test_scenario_publishes_a_qc_approved_short(name, e2e_root):
    scenario, result, provider, truth = run(name, e2e_root)
    common(scenario, result)
    # bounded model use: one judge escalation at most, one observer call, one director call
    assert provider.calls["story_judge_escalation"] <= 1
    assert provider.calls["visual_observer"] <= 1
    assert provider.calls["edit_director"] <= 2


def test_single_speaker_story_restarts_and_accent_is_restrained(e2e_root):
    scenario, result, provider, truth = run("single_speaker", e2e_root)
    timeline, cold, story = common(scenario, result)
    assert artifact(result, "vision", "vision")["layout"]["class"] == "talking_head"
    assert cold["hook"]["accepted"] and cold["hook"]["text"]
    effects = artifact(result, "effects", "effects")
    accents = [s for s in effects["sfx"] if s["kind"] == "accent"]
    assert len(accents) <= 1
    main_story = sum(s["end_frame"] - s["start_frame"] for s in timeline["segments"] if s["kind"] == "story")
    assert main_story / timeline["fps"] >= 18.0


def test_two_speakers_link_faces_by_evidence_and_keep_both_at_the_payoff(e2e_root):
    scenario, result, provider, truth = run("two_speakers", e2e_root)
    common(scenario, result)
    vision = artifact(result, "vision", "vision")
    speakers = artifact(result, "speakers", "speakers")
    by_raw = {p["id"]: p for p in speakers["participants"]}
    assert speakers["mode"] == "dual" and len(by_raw) == 2
    linked = {f["speaker"] for f in vision["faces"] if f["speaker"]}
    assert linked == {"S1", "S2"}
    left = min(vision["faces"], key=lambda f: f["median_box"][0])
    words = artifact(result, "caption_truth", "caption_truth")["words"]
    first_speaker_left = next(w["speaker"] for w in words if w["text"].startswith("Did"))
    assert left["speaker"] == first_speaker_left      # the left person (A) asked "Did you try..."
    plan = artifact(result, "edit_validation", "validated_plan")
    context = artifact(result, "edit_context", "edit_context")
    roles = {s["id"]: s["role"] for s in context["spans"]}
    for row in plan["spans"]:
        if roles[row["id"]] in ("payoff", "reaction") and len(
                next(s for s in context["spans"] if s["id"] == row["id"])["visible"]) >= 2:
            assert row["intent"] in ("TWO_SHOT", "WIDE_CONTEXT", "HOLD")


def test_multi_speaker_interruption_gets_a_temporary_second_lane(e2e_root):
    scenario, result, provider, truth = run("multi_interruption", e2e_root)
    common(scenario, result)
    captions = artifact(result, "captions", "captions")
    assert captions["overlap_windows"]
    assert any(w["lane"] == "secondary" for w in captions["words"])
    assert artifact(result, "speakers", "speakers")["mode"] in ("triple", "crowd")


def test_gameplay_facecam_uses_the_stacked_layout(e2e_root):
    scenario, result, provider, truth = run("gameplay_facecam", e2e_root)
    common(scenario, result)
    vision = artifact(result, "vision", "vision")
    assert vision["layout"]["class"] == "facecam_gameplay"
    plan = artifact(result, "edit_compile", "render_plan")
    assert sum(plan["layout"]) > 0.5 * plan["frame_count"]


def test_irl_visual_only_payoff_is_found_and_kept_in_frame(e2e_root):
    scenario, result, provider, truth = run("irl_object", e2e_root)
    timeline, cold, story = common(scenario, result)
    fall = truth["events"][0]["t"]
    evidence = artifact(result, "vod_evidence", "vod_evidence")
    assert any(e["type"] == "object_break" and e["required_review"] for e in evidence["visual_events"])
    assert story["start"] < fall < story["end"]
    assert cold["peak"]["start"] - 1.0 <= fall <= cold["peak"]["end"] + 1.0
    context = artifact(result, "edit_context", "edit_context")
    assert any(r["kind"] in ("action", "element") for s in context["spans"] if s["role"] in ("payoff", "cold_open")
               for r in s["required"])


def test_ui_screen_content_is_never_cropped(e2e_root):
    scenario, result, provider, truth = run("ui_screen", e2e_root)
    common(scenario, result)
    assert artifact(result, "vision", "vision")["layout"]["class"] == "screen_content"
    plan = artifact(result, "edit_compile", "render_plan")
    geo = Geometry(1920, 1080, 1080, 1920, 1.45, 2.6)
    screen = [s for s in plan["spans"] if s["intent"] == "SCREEN_PRIORITY"]
    assert screen
    for span in screen:
        mid = (span["frames"][0] + span["frames"][1]) // 2
        assert plan["windows"][mid][2] >= geo.h_full - 1e-3


def test_difficult_entity_is_spelled_as_verified_without_breaking_timing(e2e_root):
    scenario, result, provider, truth = run("difficult_entity", e2e_root)
    common(scenario, result)
    caption_truth = artifact(result, "caption_truth", "caption_truth")
    locked = [w for w in caption_truth["words"] if w.get("name_lock")]
    assert locked and all(w["text"].startswith("Tyla") for w in locked)
    verify = artifact(result, "caption_verify", "caption_words")
    by_id = {w["id"]: w for w in verify["words"]}
    for word in locked:
        before = by_id[word["id"]]
        assert (before["start"], before["end"]) == (word["start"], word["end"])
    ass = result.report.artifacts["captions"].path("ass").read_text()
    assert "Tyla" in ass and "Tyler" not in ass
    assert verify["micro"]["checked_spans"] >= 1
    assert caption_truth["confirmed_names"] == {"S2": "Tyla"}
