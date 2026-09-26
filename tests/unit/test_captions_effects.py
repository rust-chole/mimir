from mimir.captions.ass import escape, group_text, hook_events
from mimir.captions.layout import RenderWord, assign_lanes, event_end, group_words, map_words, mark_emphasis
from mimir.config import CaptionStyle
from mimir.timeline.schema import COLD_OPEN, STORY, Timeline, quantize

STYLE = CaptionStyle()


def rw(key, text, start, end, speaker="S1", label="", lane="main", segment=STORY, seg_index=1, uncertain=False):
    return RenderWord(key, key, segment, seg_index, text, start, end, speaker, label, uncertain, lane=lane)


def test_words_map_into_cold_open_and_story_and_skip_removed_media():
    timeline = Timeline(30, quantize([(COLD_OPEN, 10.0, 12.0), (STORY, 5.0, 8.0), (STORY, 9.0, 14.0)], 30),
                        5.0, 14.0, 10.5, 11.5)
    truth = [{"id": "c1", "text": "boom!", "start": 10.5, "end": 10.9, "speaker": "S1"},
             {"id": "c2", "text": "gone", "start": 8.2, "end": 8.6, "speaker": "S1"}]
    rows = map_words(truth, timeline, {})
    assert [r.key for r in rows] == ["c1@cold_open", "c1@story"]
    assert rows[0].start == 0.5


def test_grouping_limits_pauses_sentences_and_segments():
    words = [rw(f"w{i}", t, s, s + 0.2) for i, (t, s) in enumerate(
        [("one", 0.0), ("two", 0.25), ("three", 0.5), ("four", 0.75), ("five", 1.0), ("six.", 1.25),
         ("seven", 2.5)])]
    words[0].segment = COLD_OPEN
    groups = group_words(words, STYLE)
    texts = [[w.text for w in g.words] for g in groups]
    assert texts[0] == ["one"]                     # cold open never shares a group with the story
    assert all(len(t) <= STYLE.words_per_group for t in texts)
    assert ["seven"] in texts                      # a real pause starts a new group


def test_second_lane_only_for_the_interrupter():
    rows = [rw("a1", "so", 0.0, 0.4, "S1"), rw("a2", "then", 0.5, 0.9, "S1"), rw("b1", "no", 0.6, 0.8, "S2"),
            rw("a3", "later", 3.0, 3.3, "S1"), rw("b2", "reply", 4.0, 4.3, "S2")]
    windows = assign_lanes(rows)
    lanes = {r.key: r.lane for r in rows}
    assert lanes["b1"] == "secondary" and lanes["b2"] == "main" and lanes["a2"] == "main"
    assert windows


def test_measured_overlap_puts_the_whole_interrupting_turn_on_the_second_lane():
    # one ASR clock serialized the simultaneous words: no two word times overlap
    rows = [rw("a1", "please", 0.0, 0.5, "S1"), rw("b1", "the", 0.9, 1.0, "S2"), rw("a2", "me", 1.0, 1.2, "S1"),
            rw("b2", "one", 1.2, 1.4, "S2"), rw("b3", "kitchen.", 2.8, 3.2, "S2"), rw("a3", "later", 5.0, 5.4, "S1")]
    assert assign_lanes([rw(r.key, r.text, r.start, r.end, r.speaker) for r in rows]) == []
    windows = assign_lanes(rows, [(0.9, 3.2, "S2")])
    lanes = {r.key: r.lane for r in rows}
    assert lanes == {"a1": "main", "b1": "secondary", "a2": "main", "b2": "secondary", "b3": "secondary",
                     "a3": "main"}
    assert windows and windows[0][0] <= 0.9 and windows[0][1] >= 3.2


def test_display_holds_never_change_acoustic_times():
    words = [rw("a", "hey", 0.0, 0.05), rw("b", "you", 0.1, 0.2)]
    group = group_words(words, STYLE)[0]
    assert event_end(group, 0, STYLE, None, 10.0, False) == 0.1
    assert event_end(group, 1, STYLE, None, 10.0, False) >= 0.1 + STYLE.terminal_hold - 1e-9
    assert words[1].end == 0.2


def test_uncertain_words_are_never_emphasized_and_labels_only_for_confirmed_names():
    rows = [rw("a", "twenty", 0.0, 0.3, uncertain=True), rw("b", "cats", 0.4, 0.6)]
    mark_emphasis(rows, ["twenty cats"], [], {})
    assert rows[0].emphasis is False and rows[1].emphasis is True
    labelled = [rw("a", "hi", 0.0, 0.3, label="Kai")]
    text = group_text(group_words(labelled, STYLE)[0], 0, STYLE)
    assert "Kai:" in text


def test_ass_escaping_and_hook():
    assert escape("{\\bad}") == "(\\\\bad)"
    lines = hook_events("His face started melting", 0.0, 2.0, STYLE, 1080, 1920)
    assert lines and "HIS FACE" in lines[0] and "\\pos(540," in lines[0]


def test_hook_moves_into_band_above_fitted_content_only():
    from mimir.captions.stage import hook_y
    default = int(1920 * STYLE.hook_y_ratio)
    fitted = {"frame_count": 30, "layout": [0] * 30, "windows": [[0.5, 0.5, 1.8]] * 30}  # 16:9 screen fit
    y = hook_y(fitted, 30, STYLE, 1920)
    band = 1920 * (0.5 - 0.5 / 1.8)
    assert STYLE.hook_size < y < band - STYLE.hook_size  # inside the blurred band, clear of the content
    cropped = {"frame_count": 30, "layout": [0] * 30, "windows": [[0.5, 0.5, 1.0]] * 30}
    assert hook_y(cropped, 30, STYLE, 1920) == default
    stacked = {"frame_count": 30, "layout": [1] * 30, "windows": [[0.5, 0.5, 1.8]] * 30}
    assert hook_y(stacked, 30, STYLE, 1920) == default
    lines = hook_events("One click deleted everything", 0.0, 2.0, STYLE, 1080, 1920, y)
    assert f"\\pos(540,{y})" in lines[0]
