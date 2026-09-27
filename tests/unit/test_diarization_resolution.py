"""Diarization evidence vs absence of evidence: an empty or unusable diarization is never a single speaker."""
import math
import types

import pytest

from mimir.config import Settings
from mimir.core.stage import Ledger
from mimir.edit.context import EditContextStage
from mimir.models.provider import DiarizedSegment
from mimir.qc.checks import check_speaker_resolution
from mimir.speakers import stage as speakers_module
from mimir.speakers.stage import SpeakerStage
from mimir.timeline.schema import COLD_OPEN, STORY, Timeline, quantize
from mimir.vision.tracking import Sample

TEXT = "so I finally tried the chef special and it was still moving on the plate".split()
WORDS = [{"id": f"c{i:05d}", "text": t, "start": round(10.0 + i * 0.4, 3), "end": round(10.0 + i * 0.4 + 0.3, 3)}
         for i, t in enumerate(TEXT)]
WINDOW = [9.5, 17.0]


class Diarizer:
    def __init__(self, segments):
        self.segments = segments

    def diarize(self, role, route, audio, *, language, known_speakers=(), meta=None):
        return self.segments


def run_speakers(segments, monkeypatch, tmp_path, words=WORDS):
    monkeypatch.setattr(speakers_module, "extract_wav", lambda *a, **k: tmp_path / "window.wav")
    caption = {"window": WINDOW, "words": words}
    ledger = Ledger()
    ctx = types.SimpleNamespace(dep=lambda name: types.SimpleNamespace(json=lambda key: caption),
                                settings=Settings(), provider=Diarizer(segments), ledger=ledger,
                                source=types.SimpleNamespace(path=tmp_path / "v.mp4"), out_dir=tmp_path)
    return SpeakerStage().run(ctx).data["speakers"], ledger


def seg(speaker, start, end, text):
    # diarizer times are relative to the window start
    return DiarizedSegment(speaker, start - WINDOW[0], end - WINDOW[0], text)


def test_confirmed_single_speaker_auto_continues(monkeypatch, tmp_path):
    speakers, ledger = run_speakers([seg("A", 9.9, 13.2, " ".join(TEXT[:8])),
                                     seg("A", 13.3, 16.2, " ".join(TEXT[8:]))], monkeypatch, tmp_path)
    assert speakers["mode"] == "single" and speakers["resolution"]["status"] == "confirmed"
    assert speakers["resolution"]["coverage"] == 1.0
    assert {row["speaker"] for row in speakers["assignment"].values()} == {"S1"}
    assert not [n for n in ledger.entries if n["level"] in ("warning", "degraded")]


def test_empty_diarization_is_unresolved_not_single(monkeypatch, tmp_path):
    speakers, ledger = run_speakers([], monkeypatch, tmp_path)
    assert speakers["mode"] == "unresolved"
    assert speakers["resolution"] == {"status": "unresolved", "reason": "diarizer returned no segments",
                                      "raw_segments": 0, "usable_segments": 0, "coverage": 0.0}
    assert speakers["participants"] == [] and speakers["segments"] == []
    assert all(row["speaker"] == "" and row["confidence"] == 0.0 for row in speakers["assignment"].values())
    assert [n["code"] for n in ledger.entries] == ["diarization_unresolved"]


@pytest.mark.parametrize("segments, reason", [
    ([seg("A", 12.0, 11.0, "reversed"), seg("A", 11.0, 11.0, "zero length"),
      DiarizedSegment("A", math.nan, 2.0, "nan start"), seg("", 10.0, 12.0, "no label"),
      seg("A", 40.0, 45.0, "outside the window")], "only unusable segments"),
    ([seg("A", 10.0, 10.3, "[music]"), seg("B", 12.0, 12.2, "(applause)")], "participant-like"),
])
def test_unusable_segments_are_unresolved(segments, reason, monkeypatch, tmp_path):
    speakers, _ = run_speakers(segments, monkeypatch, tmp_path)
    assert speakers["resolution"]["status"] == "unresolved" and reason in speakers["resolution"]["reason"]
    assert all(row["speaker"] == "" for row in speakers["assignment"].values())


def test_lone_voice_covering_little_speech_is_not_a_confirmed_single_speaker(monkeypatch, tmp_path):
    speakers, _ = run_speakers([seg("A", 9.9, 12.0, " ".join(TEXT[:5]))], monkeypatch, tmp_path)
    assert speakers["resolution"]["status"] == "unresolved" and "covers" in speakers["resolution"]["reason"]
    assert all(row["speaker"] == "" for row in speakers["assignment"].values())


def test_multi_speaker_ambiguity_stays_unresolved_per_word(monkeypatch, tmp_path):
    # A and B both claim "chef special and" in their own diarized text at the same time:
    # nobody can own those words, the rest of each turn stays with its voice
    speakers, _ = run_speakers([seg("A", 9.9, 13.2, " ".join(TEXT[:8])), seg("B", 12.1, 16.2, " ".join(TEXT[5:])),
                                seg("A", 16.3, 16.9, "yes"), seg("B", 9.5, 9.8, "hm hm")], monkeypatch, tmp_path)
    assert speakers["mode"] == "dual" and speakers["resolution"]["status"] == "confirmed"
    rows = [speakers["assignment"][w["id"]] for w in WORDS]
    assert [r["speaker"] for r in rows[5:8]] == ["", "", ""]
    assert all(r["source"] == "ambiguous_overlap" and r["confidence"] == 0.0 for r in rows[5:8])
    first, second = {r["speaker"] for r in rows[:5]}, {r["speaker"] for r in rows[8:]}
    assert len(first) == 1 and len(second) == 1 and first != second and "" not in first | second
    assert speakers["metrics"]["ambiguous_words"] == 3 and speakers["overlaps"]
    # several unnamed voices: identity is ambiguous and stays anonymous; speaker_preview (unchanged optional
    # behaviour) only uses speech nobody else overlaps, which these turns do not have
    from mimir.speakers import identity as identity_module
    monkeypatch.setattr(identity_module, "render_preview", lambda source, ranges, output: output)
    ledger = Ledger()
    ctx = types.SimpleNamespace(dep=lambda name: types.SimpleNamespace(json=lambda key: speakers), settings=Settings(),
                                source=types.SimpleNamespace(path=tmp_path / "v.mp4"), out_dir=tmp_path, ledger=ledger)
    identity = identity_module.IdentityStage(prompt=None).run(ctx).data["identity"]
    assert identity["ambiguous"] and identity["previews"] == {}
    assert [n["code"] for n in ledger.entries] == ["no_clean_preview", "no_clean_preview"]
    assert all(row == {"name": "", "confirmed": False, "source": "anonymous"} for row in identity["speakers"].values())


def test_unresolved_speakers_never_receive_an_identity(tmp_path, monkeypatch):
    from mimir.speakers import identity as identity_module

    monkeypatch.setattr(identity_module, "render_preview", lambda source, ranges, output: output)
    speakers = {"mode": "unresolved", "segments": [], "participants": []}
    settings = Settings()
    settings = settings.with_(identity=settings.identity.__class__(speaker_names=(("S1", "Alex"),)))
    ledger = Ledger()
    ctx = types.SimpleNamespace(dep=lambda name: types.SimpleNamespace(json=lambda key: speakers), settings=settings,
                                source=types.SimpleNamespace(path=tmp_path / "v.mp4"), out_dir=tmp_path, ledger=ledger)
    identity = identity_module.IdentityStage(prompt=None).run(ctx).data["identity"]
    assert identity["speakers"] == {} and identity["previews"] == {} and not identity["ambiguous"]
    assert [n["code"] for n in ledger.entries] == ["unknown_speaker_name"]


def context_for(resolution, speaker):
    timeline = Timeline(30, quantize([(COLD_OPEN, 14.0, 15.5), (STORY, 10.0, 17.0)], 30), 10.0, 17.0, 14.0, 15.5)
    samples = [Sample(10.0 + k * 0.1, 0.5, 0.4, 0.12, 0.2, 0.9, 0.01, "detected").to_list() for k in range(70)]
    face = {"id": "face_00", "samples": samples, "static_pattern": False, "speaker": speaker or None,
            "speaker_confidence": 0.9 if speaker else 0.0}
    artifacts = {
        "story": {"title": "t", "reason": "r", "emotion": "shock",
                  "beats": [{"role": "setup", "start": 10.0, "end": 12.0}, {"role": "escalation", "start": 12.0,
                            "end": 14.0}, {"role": "payoff", "start": 14.0, "end": 15.5},
                            {"role": "reaction", "start": 15.5, "end": 17.0}]},
        "caption_truth": {"words": [{**w, "speaker": speaker} for w in WORDS], "speaker_mode": "single",
                          "speaker_resolution": resolution},
        "vision": {"faces": [face], "shot_cuts": [], "layout": {"class": "talking_head"}, "observations": [],
                   "analysis_fps": 10.0, "action_regions": []},
        "timeline": timeline.to_dict(),
    }
    ctx = types.SimpleNamespace(dep=lambda name: types.SimpleNamespace(json=lambda key: artifacts[name]),
                                settings=Settings())
    return EditContextStage().run(ctx).data["edit_context"]


def test_camera_is_conservative_when_speakers_are_unresolved():
    confirmed = context_for({"status": "confirmed"}, "S1")
    unresolved = context_for({"status": "unresolved", "reason": "diarizer returned no segments"}, "")
    allowed_confirmed = set().union(*(s["allowed"] for s in confirmed["spans"]))
    allowed_unresolved = set().union(*(s["allowed"] for s in unresolved["spans"]))
    assert {"SPEAKER_MEDIUM", "SPEAKER_PUNCH", "REACTION"} <= allowed_confirmed
    assert not {"SPEAKER_MEDIUM", "SPEAKER_PUNCH", "REACTION"} & allowed_unresolved


def qc_inputs(resolution, speaker="", label="", intents=("WIDE_CONTEXT",), lane="main"):
    truth = {"words": [{**w, "speaker": speaker} for w in WORDS], "speaker_mode": "unresolved",
             "speaker_resolution": resolution, "confirmed_names": {}}
    captions = {"words": [{"key": w["id"], "word_id": w["id"], "speaker": speaker, "label": label, "lane": lane}
                          for w in WORDS]}
    plan = {"spans": [{"id": f"s{i}", "intent": intent} for i, intent in enumerate(intents)]}
    return types.SimpleNamespace(truth=truth, captions=captions, plan=plan)


def test_qc_distinguishes_unresolved_from_confirmed_and_catches_fabrication():
    ok = check_speaker_resolution(qc_inputs({"status": "confirmed"}, "S1"))
    assert ok.passed and ok.details["status"] == "confirmed"
    handled = check_speaker_resolution(qc_inputs({"status": "unresolved", "reason": "diarizer returned no segments"}))
    assert not handled.passed and handled.severity == "warn" and handled.details["violations"] == []
    fabricated = check_speaker_resolution(qc_inputs({"status": "unresolved"}, speaker="S1", label="Alex",
                                                    intents=("SPEAKER_PUNCH",), lane="secondary"))
    assert not fabricated.passed and fabricated.severity == "fail" and len(fabricated.details["violations"]) == 4
