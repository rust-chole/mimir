from mimir.models.provider import DiarizedSegment
from mimir.speakers.assign import assign_speakers, measured_overlaps
from mimir.speakers.census import classify, normalize_segments, speaker_stats
from mimir.speakers.identity import preview_ranges


def seg(speaker, start, end, text):
    return DiarizedSegment(speaker, start, end, text)


def test_census_single_dual_and_crowd_guard():
    single = speaker_stats(normalize_segments([seg("A", 0, 5, "one two three four five")], 0, 10))
    assert classify(single)[0] == "single"
    dual = speaker_stats(normalize_segments([seg("A", 0, 4, "a b c d e"), seg("B", 4.2, 6, "yes of course friend")],
                                            0, 10))
    assert classify(dual)[0] == "dual"
    shout = speaker_stats(normalize_segments([seg("A", 0, 4, "a b c d e"), seg("B", 4.2, 6, "yes of course friend"),
                                              seg("C", 6.1, 6.4, "[cheering]")], 0, 10))
    mode, participants, background = classify(shout)
    assert mode == "dual" and len(participants) == 2


def test_interleaved_overlap_keeps_each_word_with_its_voice():
    words = [{"id": f"w{i}", "text": t, "start": s, "end": s + 0.25} for i, (t, s) in enumerate(
        [("that", 0.0), ("was", 0.3), ("me", 0.6), ("no", 0.8), ("I", 0.9), ("burned", 1.2), ("that", 1.25),
         ("is", 1.5), ("the", 1.6), ("kitchen", 1.9), ("not", 2.2), ("fair", 2.5)])]
    segments = [{"speaker": "S1", "start": 0.0, "end": 2.15, "text": "that was me I burned the kitchen"},
                {"speaker": "S2", "start": 0.8, "end": 2.75, "text": "no that is not fair"}]
    rows, _ = assign_speakers(words, segments)
    speakers = {w["text"] + str(i): r["speaker"] for i, (w, r) in enumerate(zip(words, rows))}
    assert speakers["no3"] == "S2" and speakers["is7"] == "S2" and speakers["fair11"] == "S2"
    assert speakers["me2"] == "S1" and speakers["kitchen9"] == "S1"


def test_single_word_flip_is_absorbed_but_real_gaps_stay_unresolved():
    words = [{"id": f"w{i}", "text": f"t{i}", "start": i * 0.3, "end": i * 0.3 + 0.2} for i in range(7)]
    words.append({"id": "w7", "text": "late", "start": 6.0, "end": 6.2})
    segments = [{"speaker": "S1", "start": 0.0, "end": 0.85, "text": "t0 t1 t2"},
                {"speaker": "S2", "start": 0.86, "end": 1.05, "text": "uh"},   # boundary jitter, no text match
                {"speaker": "S1", "start": 1.1, "end": 2.0, "text": "t4 t5 t6"}]
    rows, metrics = assign_speakers(words, segments)
    assert rows[3]["speaker"] == "S1" and rows[3]["source"] == "turn_hysteresis"
    assert rows[7]["speaker"] == ""


def test_preview_uses_only_clean_non_overlapping_speech():
    segments = [{"speaker": "S1", "start": 0.0, "end": 2.0}, {"speaker": "S2", "start": 1.5, "end": 3.0},
                {"speaker": "S1", "start": 4.0, "end": 6.5}]
    ranges = preview_ranges("S1", segments)
    assert ranges and all(a >= 4.0 for a, _ in ranges)


def test_measured_overlaps_name_the_later_starter_and_ignore_boundary_jitter():
    segments = [{"speaker": "S1", "start": 15.3, "end": 18.15}, {"speaker": "S2", "start": 16.2, "end": 18.65},
                {"speaker": "S3", "start": 18.5, "end": 20.0},     # 0.15 s jitter with S2: not an overlap
                {"speaker": "S3", "start": 20.2, "end": 23.2}, {"speaker": "S1", "start": 21.4, "end": 24.5}]
    overlaps = measured_overlaps(segments)
    assert [(o["held_by"], o["interrupter"], o["turn"]) for o in overlaps] == [
        ("S1", "S2", [16.2, 18.65]), ("S3", "S1", [21.4, 24.5])]
    assert overlaps[0]["start"] == 16.2 and overlaps[0]["end"] == 18.15


def run_identity(participants, names, tmp_path, monkeypatch):
    import types
    from mimir.config import Settings
    from mimir.core.stage import Ledger
    from mimir.speakers import identity as module

    monkeypatch.setattr(module, "render_preview", lambda source, ranges, output: output)
    segments = [{"speaker": sid, "start": i * 5.0, "end": i * 5.0 + 4.0, "text": "clean speech sample here"}
                for i, sid in enumerate(participants)]
    speakers = {"mode": "dual" if len(participants) == 2 else "single", "segments": segments,
                "participants": [{"id": sid} for sid in participants]}
    settings = Settings()
    settings = settings.with_(identity=settings.identity.__class__(speaker_names=tuple(names.items())))
    ctx = types.SimpleNamespace(dep=lambda name: types.SimpleNamespace(json=lambda key: speakers),
                                settings=settings, source=types.SimpleNamespace(path=tmp_path / "v.mp4"),
                                out_dir=tmp_path, ledger=Ledger())
    return module.IdentityStage(prompt=None).run(ctx).data["identity"]


def test_speaker_preview_only_for_unnamed_voices_when_several_speak(tmp_path, monkeypatch):
    single = run_identity(["S1"], {}, tmp_path, monkeypatch)
    assert not single["ambiguous"] and single["previews"] == {} and not single["asked"]
    named = run_identity(["S1", "S2"], {"S1": "Kai", "S2": "Tyla"}, tmp_path, monkeypatch)
    assert not named["ambiguous"] and named["previews"] == {}
    partly = run_identity(["S1", "S2"], {"S1": "Kai"}, tmp_path, monkeypatch)
    assert partly["ambiguous"] and set(partly["previews"]) == {"S2"}
    assert partly["speakers"]["S2"] == {"name": "", "confirmed": False, "source": "anonymous"}
