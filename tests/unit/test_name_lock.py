from mimir.transcript.name_lock import build_roster, confusion_score, lock_names, phonetic_key


def words(texts, speakers=None):
    speakers = speakers or ["S1"] * len(texts)
    return [{"id": f"c{i:03d}", "text": t, "start": i * 0.4, "end": i * 0.4 + 0.3, "speaker": s}
            for i, (t, s) in enumerate(zip(texts, speakers))]


ROSTER = build_roster({"S2": {"name": "TYLA", "confirmed": True}}, (), "")


def test_phonetic_confusion_of_homophone_spellings():
    assert phonetic_key("Tyler") == phonetic_key("Tyla")
    assert confusion_score("Tyler", "Tyla") >= 0.85
    assert confusion_score("Taylor", "Kai") == 0.0


def test_direct_address_to_a_verified_participant_is_canonicalized():
    out, audit = lock_names(words(["Tyler,", "you", "said", "fifteen"]), ROSTER)
    assert out[0]["text"] == "Tyla,"
    assert out[0]["original_text"] == "Tyler,"
    assert audit["corrections"][0]["evidence"][0] == "direct_address"


def test_correction_keeps_ids_timing_and_speaker():
    before = words(["hey", "Tyler,", "look"], ["S1", "S1", "S1"])
    out, _ = lock_names(before, ROSTER)
    for a, b in zip(before, out):
        assert (a["id"], a["start"], a["end"], a["speaker"]) == (b["id"], b["start"], b["end"], b["speaker"])


def test_no_evidence_no_change_and_unknown_names_never_invented():
    out, _ = lock_names(words(["the", "Tyler", "report"]), ROSTER)
    assert out[1]["text"] == "Tyler"
    out, _ = lock_names(words(["Tyler,", "you"]), build_roster({}, (), ""))
    assert out[0]["text"] == "Tyler,"


def test_competing_verified_names_fail_closed():
    roster = build_roster({"S1": {"name": "Tyla", "confirmed": True}, "S2": {"name": "Tyler", "confirmed": True}},
                          (), "")
    out, audit = lock_names(words(["Tylar,", "you"]), roster)
    assert out[0]["text"] == "Tylar,"
    assert audit["rejected"][0]["reason"] == "competing_verified_names"


def test_ordinary_word_and_other_full_name_are_protected():
    roster = build_roster({}, ("Will",), "")
    out, _ = lock_names(words(["Well,", "you", "know"]), roster)
    assert out[0]["text"] == "Well,"
    roster = build_roster({}, ("Tyla",), "")
    out, _ = lock_names(words(["I", "met", "Tyler", "Johnson", "today"]), roster)
    assert out[2]["text"] == "Tyler"


def test_confident_independent_ear_hearing_another_word_vetoes():
    rows = words(["the", "Tyler", "said"])
    rows[1]["alternatives"] = [{"token": "Taylor", "prompted": False, "probability": 0.95},
                               {"token": "Taylor", "prompted": False, "probability": 0.9}]
    out, audit = lock_names(rows, build_roster({}, ("Tyla",), ""))
    assert out[1]["text"] == "Tyler"


def test_alias_confirmed_by_strong_evidence_resolves_later_mentions():
    out, _ = lock_names(words(["Tyler,", "you", "rock.", "Tyler", "said", "yes"]), ROSTER)
    assert out[0]["text"] == "Tyla," and out[3]["text"] == "Tyla"


def test_name_lock_invariants_reject_any_timing_or_speaker_change():
    import pytest as _pytest
    from mimir.transcript.name_lock import verify_invariants
    before = [{"id": "c1", "text": "Tyler", "start": 1.0, "end": 1.3, "speaker": "S2"}]
    verify_invariants(before, [{**before[0], "text": "Tyla"}])          # spelling may change
    for key, value in (("start", 1.01), ("end", 1.31), ("speaker", "S1"), ("id", "c2")):
        with _pytest.raises(AssertionError):
            verify_invariants(before, [{**before[0], "text": "Tyla", key: value}])
