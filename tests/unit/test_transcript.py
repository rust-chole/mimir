import numpy as np
import pytest

from mimir.models.provider import TimedWord, Transcription
from mimir.transcript.align import AlignmentError, align_text_to_clock, text_agreement
from mimir.transcript.clock_guard import apply_clock_guard
from mimir.transcript.segments import build_segments
from mimir.transcript.tokens import canonical_word, tokenize
from mimir.transcript.verify import (
    ROUND_ONE,
    disagreement_spans,
    known_name_spans,
    merge_spans,
    phrase_for_core,
    token_observations,
    vote,
)


def clock(*rows):
    return [TimedWord(t, a, b) for t, a, b in rows]


class TestTokens:
    def test_numbers_and_punctuation(self):
        assert canonical_word("Twelve!") == "12"
        assert canonical_word("don't") == "dont"
        assert tokenize("wait , what ?!") == ["wait,", "what?!"]


class TestFixedClockAlignment:
    def test_exact_anchors_keep_the_timing_ear_onsets(self):
        rows, ratio = align_text_to_clock("hello there chat", clock(("hello", 0.5, 0.8), ("there", 0.9, 1.2),
                                                                    ("chat", 1.3, 1.6)), 3.0)
        assert [(r.text, r.start) for r in rows] == [("hello", 0.5), ("there", 0.9), ("chat", 1.3)]
        assert ratio == 1.0

    def test_lexical_words_win_but_later_onsets_never_move(self):
        timing = clock(("hello", 0.5, 0.8), ("dare", 0.9, 1.2), ("chat", 1.3, 1.6))
        rows, _ = align_text_to_clock("hello there chat", timing, 3.0)
        assert [r.text for r in rows] == ["hello", "there", "chat"]
        assert rows[1].start == pytest.approx(0.9)
        assert rows[2].start == pytest.approx(1.3)

    def test_inserted_word_goes_into_the_local_gap(self):
        timing = clock(("I", 0.2, 0.4), ("it", 1.4, 1.6))
        rows, _ = align_text_to_clock("I ate it", timing, 3.0)
        assert [r.text for r in rows] == ["I", "ate", "it"]
        assert 0.4 <= rows[1].start < rows[1].end <= 1.4
        assert rows[2].start == pytest.approx(1.4)

    def test_coincident_onsets_are_grouped_not_dropped(self):
        timing = clock(("no", 1.0, 1.2), ("way", 1.0, 1.3))
        rows, _ = align_text_to_clock("no way", timing, 3.0)
        assert " ".join(r.text for r in rows) == "no way"

    def test_monotonic_and_positive(self):
        timing = clock(*[(f"w{i}", i * 0.3, i * 0.3 + 0.25) for i in range(20)])
        rows, _ = align_text_to_clock(" ".join(f"w{i}" for i in range(20)) + " extra words here", timing, 8.0)
        for a, b in zip(rows, rows[1:]):
            assert b.start >= a.start
        assert all(r.end > r.start for r in rows)

    def test_missing_clock_fails_closed(self):
        with pytest.raises(AlignmentError):
            align_text_to_clock("hello", [], 2.0)

    def test_agreement(self):
        assert text_agreement("a b c d", "a b c d") == 1.0
        assert text_agreement("a b c d", "a x c d") < 1.0


class TestClockGuard:
    def test_late_phrase_start_moves_back_to_the_real_onset(self):
        rate = 16000
        samples = np.random.default_rng(0).standard_normal(rate * 4).astype(np.float32) * 0.001
        t = np.arange(int(0.6 * rate)) / rate
        samples[int(2.0 * rate):int(2.6 * rate)] += 0.5 * np.sin(2 * np.pi * 220 * t)  # speech really starts at 2.0
        words = [{"id": "a", "text": "before", "start": 0.2, "end": 0.6},
                 {"id": "b", "text": "late", "start": 2.62, "end": 2.9}]   # clock claims 2.62 (silence there)
        out, report = apply_clock_guard(words, samples, rate, offset=0.0, duration=4.0)
        assert report["corrected_groups"] == 1
        assert out[1]["start"] == pytest.approx(2.0, abs=0.06)
        assert out[0] == words[0]

    def test_well_timed_words_are_untouched(self):
        rate = 16000
        samples = np.zeros(rate * 3, dtype=np.float32)
        t = np.arange(int(0.5 * rate)) / rate
        samples[int(1.5 * rate):int(2.0 * rate)] = 0.5 * np.sin(2 * np.pi * 200 * t)
        words = [{"id": "a", "text": "x", "start": 0.1, "end": 0.3}, {"id": "b", "text": "y", "start": 1.5, "end": 2.0}]
        out, report = apply_clock_guard(words, samples, rate, offset=0.0, duration=3.0)
        assert report["corrected_groups"] == 0
        assert out == words


class TestSuspectSpansAndVoting:
    def test_disagreement_and_known_name_spans(self):
        base = tokenize("hey Tyler you have fifteen cats")
        spans = disagreement_spans(base, [tokenize("hey Tyla you have fifty cats")])
        assert {(s["start"], s["end"]) for s in spans} == {(1, 2), (4, 5)}
        names = known_name_spans(base, ["Tyla"], 0.64)
        assert names and names[0]["start"] == 1

    def test_spans_never_grow_transcript_sized(self):
        rows = [{"start": i, "end": i + 2, "source": "asr_disagreement"} for i in range(0, 40)]
        merged = merge_spans(rows, 42, max_words=6, max_spans=10)
        assert all(s["end"] - s["start"] <= 6 for s in merged)
        assert len(merged) <= 10

    def test_phrase_for_core(self):
        window = tokenize("you said you have fifteen cats now")
        assert phrase_for_core(window, tokenize("you said you have fifty cats now"), (4, 5)) == ("fifty",)
        assert phrase_for_core(window, tokenize("you said you have 15 cats now"), (4, 5)) == ("15",)

    def test_vote_counts_identical_phrases(self):
        window = tokenize("hey Tyler you")
        ears = [(ROUND_ONE[0], Transcription("hey Tyla you")), (ROUND_ONE[1], Transcription("hey Tyla you")),
                (ROUND_ONE[2], Transcription("hey Tyler you"))]
        ranked = vote(window, ears, (1, 2))
        assert ranked[0]["phrase"] == ("Tyla",) and ranked[0]["votes"] == 2
        observations = token_observations(window, ears, (1, 2))
        assert [o["token"] for o in observations[0]] == ["Tyla", "Tyla", "Tyler"]
        assert observations[0][2]["prompted"] is True


def test_segments_break_on_pauses_and_sentences():
    words = [{"id": f"w{i}", "text": t, "start": s, "end": s + 0.2} for i, (t, s) in
             enumerate([("hi.", 0.0), ("you", 0.3), ("there.", 0.6), ("next", 0.9), ("one", 2.5)])]
    segments = build_segments(words)
    assert [s["text"] for s in segments][-1] == "one"


def tw(text, start, end):
    return TimedWord(text, start, end)


def test_simultaneous_speech_keeps_one_token_per_word_with_measured_times():
    # S3 "I burned the kitchen" and S1 "No, that is not" overlap on one timing clock;
    # "burned" (S3) and "that" (S1) share an onset, "is" starts inside "burned".
    clock = [tw("No,", 1.400, 1.503), tw("I", 1.508, 1.811), tw("burned", 1.881, 2.303), tw("that", 1.881, 2.052),
             tw("is", 2.200, 2.368), tw("the", 2.373, 2.688), tw("kitchen!", 2.758, 3.100), tw("not", 3.120, 3.300)]
    rows, ratio = align_text_to_clock("No, I burned that is the kitchen! not", clock, 10.0)
    assert ratio == 1.0
    assert [r.text for r in rows] == ["No,", "I", "burned", "that", "is", "the", "kitchen!", "not"]
    by_text = {r.text: r for r in rows}
    # measured intervals survive exactly: nothing merged, nothing trimmed
    assert (by_text["burned"].start, by_text["burned"].end) == (1.881, 2.303)
    assert (by_text["that"].start, by_text["that"].end) == (1.881, 2.052)
    assert (by_text["is"].start, by_text["is"].end) == (2.200, 2.368)
    assert by_text["that"].overlaps_previous and not by_text["burned"].overlaps_previous
    assert all(" " not in r.text for r in rows)


def test_partial_overlap_is_preserved_but_boundary_jitter_is_trimmed():
    rows, _ = align_text_to_clock("wait stop now", [tw("wait", 1.00, 1.40), tw("stop", 1.20, 1.60),
                                                   tw("now", 1.595, 1.90)], 5.0)
    assert (rows[0].start, rows[0].end) == (1.00, 1.40)          # "wait" keeps its measured end
    assert rows[1].start == 1.20 and rows[1].overlaps_previous    # "stop" overlaps it (two voices)
    # 5 ms of boundary jitter is not simultaneous speech: the tail yields, the onset stays
    assert rows[2].start == 1.595 and rows[1].end == 1.59 and not rows[2].overlaps_previous


def test_lexical_only_words_in_a_tight_span_stay_separate_tokens():
    # the lexical ear heard three words where the timing ear measured one short token
    rows, _ = align_text_to_clock("so I was", [tw("so", 0.50, 0.56)], 2.0)
    assert [r.text for r in rows] == ["so", "I", "was"]
    starts = [r.start for r in rows]
    assert starts == sorted(starts) and rows[0].start == 0.50
    assert all(r.end > r.start for r in rows)
