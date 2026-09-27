import pytest

from mimir.coldopen.peaks import build_candidates, compute_window, is_extreme, minimum_duration
from mimir.config import ColdOpenSettings, PacingSettings, StorySettings
from mimir.story.causal import enforce_causal_flow, validate_candidate, validate_moments
from mimir.story.package import build_package, completeness
from mimir.timeline.pacing import apply_budget, apply_protection, apply_story_integrity, suggest_cuts
from mimir.timeline.schema import COLD_OPEN, STORY, Timeline, quantize


def moment(start, end, kind="punchline", strength=8.0, before=4.0, after=2.0):
    return {"start": start, "end": end, "strength": strength, "type": kind, "source": "transcript",
            "visual_event_ids": [], "label": "x", "why_compelling": "y", "context_before_seconds": before,
            "context_after_seconds": after, "preserve_pause_after": False}


def transcript(lines):
    return {"segments": [{"start": a, "end": b, "text": t} for a, b, t in lines]}


class TestStoryRules:
    def test_moments_are_clamped_deduplicated_and_numbered(self):
        rows = validate_moments([moment(10, 12), moment(10.2, 12.1, strength=6), moment(30, 31)], 60.0, 12)
        assert [m["moment_id"] for m in rows] == ["m01", "m02"]
        assert rows[0]["strength"] == 8.0

    def test_candidate_always_contains_its_anchor_and_respects_limits(self):
        moments = validate_moments([moment(20, 22)], 100.0, 12)
        clean, error = validate_candidate({"start": 21.0, "end": 40.0, "primary_moment_id": "m01",
                                           "payoff_start": 20, "payoff_end": 22}, duration=100.0,
                                          moments=moments, settings=StorySettings())
        assert error == "" and clean["start"] <= 19.5
        assert any(r["kind"] == "payoff" for r in clean["must_keep_ranges"])
        _, error = validate_candidate({"start": 0.0, "end": 80.0, "primary_moment_id": "m01"}, duration=100.0,
                                      moments=moments, settings=StorySettings())
        assert "exceeds" in error
        _, error = validate_candidate({"start": 0, "end": 30, "primary_moment_id": "m99"}, duration=100.0,
                                      moments=moments, settings=StorySettings())
        assert "not a scouted" in error

    def test_causal_flow_expands_a_peak_fragment_into_a_story(self):
        moments = validate_moments([moment(40, 42, kind="reaction", strength=8.5, before=3, after=1)], 120.0, 12)
        speech = transcript([(t, t + 2.5, "words") for t in range(20, 60, 3)])
        fragment, _ = validate_candidate({"start": 38.0, "end": 51.0, "primary_moment_id": "m01",
                                          "payoff_start": 40, "payoff_end": 42}, duration=120.0,
                                         moments=moments, settings=StorySettings())
        result = enforce_causal_flow(fragment, moments=moments, transcript=speech, duration=120.0, video=None,
                                     width=1920, height=1080, judge_metrics={"context_completeness": 6.0},
                                     settings=StorySettings())
        story = result.candidate
        assert story["start"] < 38.0 and story["end"] - story["start"] >= 18.0
        kinds = {r["kind"] for r in story["must_keep_ranges"]}
        assert {"escalation", "reaction", "payoff"} <= kinds

    def test_package_beats_are_ordered_and_complete(self):
        moments = validate_moments([moment(30, 32)], 100.0, 12)
        candidate, _ = validate_candidate({"start": 15, "end": 38, "primary_moment_id": "m01", "payoff_start": 30,
                                           "payoff_end": 32}, duration=100.0, moments=moments,
                                          settings=StorySettings())
        package = build_package(candidate, moments, judge={}, flow={}, boundary={}, rank=1)
        roles = [b["role"] for b in package["beats"]][:4]
        assert roles == ["setup", "escalation", "payoff", "reaction"]
        assert package["completeness_problems"] == []
        broken = [dict(b) for b in package["beats"]]
        broken[3]["duration"] = 0.1
        assert completeness(broken)


class TestPacing:
    def words(self):
        times = [0.2, 0.6, 1.0, 3.5, 3.9, 4.3, 4.7, 9.0, 9.4]
        return [{"id": f"w{i}", "text": "w", "start": t, "end": t + 0.3} for i, t in enumerate(times)]

    def test_dead_air_is_cut_but_payoff_pauses_are_kept(self):
        cuts = suggest_cuts(self.words(), 0.0, 10.5, (8.8, 9.8), PacingSettings())
        auto = [c for c in cuts if c["action"] == "cut"]
        assert any(1.3 < c["start"] < 3.5 for c in auto)          # 2.2 s gap between beats
        assert all(not (c["start"] < 9.0 and c["end"] > 5.0 and c["kind"] == "dead_air" and c["action"] == "cut"
                        and c["end"] - c["start"] < 1.35) for c in cuts)

    def test_protected_ranges_block_cuts_and_budget_limits_removal(self):
        cuts = suggest_cuts(self.words(), 0.0, 10.5, (0.0, 0.1), PacingSettings())
        apply_protection(cuts, [{"start": 5.0, "end": 8.9, "reason": "visual origin"}])
        assert all(c["action"] != "cut" for c in cuts if c["start"] < 8.9 and c["end"] > 5.0)
        apply_budget(cuts, 10.5, PacingSettings(max_cut_seconds=0.5))
        assert sum(c["duration"] for c in cuts if c["action"] == "cut") <= 0.5 + 1e-9

    def test_story_integrity_keeps_bridges_between_causal_beats(self):
        cuts = [{"action": "cut", "start": 3.0, "end": 4.5, "duration": 1.5, "priority": 5}]
        protected = [{"start": 1.0, "end": 2.0, "kind": "escalation"}, {"start": 5.0, "end": 6.0, "kind": "payoff"}]
        apply_story_integrity(cuts, protected)
        assert cuts[0]["action"] == "review"


class TestCanonicalTimeline:
    def test_quantized_segments_map_source_to_output_exactly(self):
        segments = quantize([(COLD_OPEN, 30.0, 33.0), (STORY, 20.0, 25.0), (STORY, 26.0, 35.0)], 30)
        timeline = Timeline(30, segments, 20.0, 35.0, 31.0, 32.0)
        assert timeline.frame_count == 90 + 150 + 270
        assert timeline.main_start_frame == 90
        assert timeline.source_time(90) == pytest.approx(20.0)
        assert timeline.map_time(31.0, kinds=(COLD_OPEN, STORY)) == pytest.approx([1.0, 3.0 + 5.0 + 5.0])
        assert timeline.removed_ranges() == [(25.0, 26.0)]
        assert Timeline.from_dict(timeline.to_dict()) == timeline

    def test_interval_mapping_skips_removed_media(self):
        segments = quantize([(COLD_OPEN, 10.0, 12.0), (STORY, 0.0, 5.0), (STORY, 6.0, 12.0)], 30)
        timeline = Timeline(30, segments, 0.0, 12.0, 10.0, 11.0)
        pieces = timeline.map_interval(4.0, 7.0)
        assert sum(b - a for a, b, _ in pieces) == pytest.approx(2.0)


class TestColdOpen:
    story = {"start": 10.0, "end": 40.0, "payoff": {"start": 30.0, "end": 31.0}, "beats": []}

    def test_window_contains_the_peak_and_never_cuts_a_word(self):
        words = [{"id": "c1", "text": "whoa", "start": 28.9, "end": 29.3},
                 {"id": "c2", "text": "no", "start": 31.9, "end": 32.3}]
        peak = {"start": 30.0, "end": 31.0, "center": 30.5, "combined_score": 0.7, "multimodal": False,
                "audio_score": 0.7, "visual_score": 0.0}
        start, end, policy = compute_window(peak, self.story, words, ColdOpenSettings(), None, [])
        assert start <= 30.0 and end >= 31.0
        for word in words:
            assert not (word["start"] < start < word["end"]) and not (word["start"] < end < word["end"])
        assert end - start <= ColdOpenSettings().max_duration

    def test_stronger_compact_peaks_get_shorter_cold_opens(self):
        cfg = ColdOpenSettings()
        weak = minimum_duration({"combined_score": 0.4, "start": 0, "end": 1}, cfg)[0]
        compact = minimum_duration({"combined_score": 0.9, "multimodal": True, "audio_score": 0.8,
                                    "visual_score": 0.8, "start": 0, "end": 1}, cfg)[0]
        assert compact < weak

    def test_payoff_is_always_a_candidate(self):
        candidates = build_candidates(self.story, [{"center": 12.0, "start": 11.6, "end": 12.5, "score": 0.9}],
                                      {"visual_peaks": [{"t": 30.4, "score": 0.8}], "action_regions": []})
        assert candidates[0]["signals"] == ["story_payoff"]
        assert any(c["in_payoff"] and "visual_motion_burst" in c["signals"] for c in candidates)



class TestExtremeColdOpen:
    """EXTREME PEAK = VERY SHORT COLD OPEN (about 1.2-2.0 s), ordinary peaks keep their context."""
    story = {"start": 10.0, "end": 40.0, "payoff": {"start": 30.0, "end": 31.0},
             "beats": [{"role": "payoff", "start": 30.0, "end": 31.0}, {"role": "reaction", "start": 31.0,
                                                                          "end": 33.0}]}
    cfg = ColdOpenSettings()

    def window(self, peak, words=(), cuts=(), story=None):
        start, end, policy = compute_window(peak, story or self.story, list(words), self.cfg, None, list(cuts))
        assert start <= peak["start"] + 1e-6 and end >= peak["end"] - 1e-6, "the action must never be cut"
        return start, end, policy

    def peak(self, start, end, audio, visual, **flags):
        multimodal = audio > 0 and visual > 0
        combined = min(1.0, max(audio, visual) + (0.15 if multimodal else 0) + (0.1 if flags.get("in_payoff") else 0))
        return {"start": start, "end": end, "center": (start + end) / 2, "audio_score": audio, "visual_score": visual,
                "multimodal": multimodal, "combined_score": combined, "in_payoff": False, "in_reaction": False,
                **flags}

    def test_extreme_multimodal_peak_gets_a_very_short_cold_open(self):
        peak = self.peak(30.1, 31.0, 0.9, 0.88, in_payoff=True)
        start, end, policy = self.window(peak)
        assert is_extreme(peak) and policy == "extreme_peak"
        assert 1.2 <= end - start <= 2.0

    def test_extreme_visual_only_event_qualifies_without_loud_audio(self):
        peak = self.peak(30.2, 30.9, 0.0, 0.9, in_payoff=True)          # a silent fall / break
        start, end, policy = self.window(peak)
        assert policy == "extreme_peak" and 1.2 <= end - start <= 2.0
        reaction = self.peak(31.3, 32.1, 0.9, 0.0, in_reaction=True)   # a scream at the reaction
        assert is_extreme(reaction) and 1.2 <= self.window(reaction)[1] - self.window(reaction)[0] <= 2.0

    def test_a_payoff_span_of_exactly_the_compact_limit_is_extreme(self):
        # the story's own payoff (1.3 s, measured as 31.3 - 30.0 in floating point) with a decisive event
        payoff = {"start": 30.0, "end": 31.3, "center": 30.65, "audio_score": 0.99, "visual_score": 1.0,
                  "multimodal": False, "combined_score": 1.0, "in_payoff": True, "in_reaction": False}
        start, end, policy = self.window(payoff)
        assert policy == "extreme_peak" and 1.2 <= end - start <= 2.0

    def test_strong_but_off_story_or_long_peaks_are_not_extreme(self):
        assert not is_extreme(self.peak(20.0, 20.8, 0.0, 0.9))            # decisive, but not the payoff/reaction
        assert not is_extreme(self.peak(29.5, 31.5, 0.9, 0.9, in_payoff=True))  # not compact

    def test_normal_contextual_peak_keeps_a_longer_cold_open(self):
        peak = self.peak(30.2, 30.9, 0.7, 0.0, in_payoff=True)
        start, end, policy = self.window(peak)
        assert policy == "contextual_peak" and end - start >= self.cfg.moderate_peak_min - 1e-6

    def test_a_cold_open_is_never_too_short_to_understand(self):
        blip = self.peak(30.40, 30.50, 0.95, 0.9, in_payoff=True)      # a 0.1 s impact
        start, end, policy = self.window(blip)
        assert end - start >= self.cfg.min_understandable - 1e-6 and end - start <= 2.0
        at_end = {**self.story, "end": 30.6}                             # the story ends right after the impact
        start, end, policy = self.window(blip, story=at_end)
        assert end <= 30.6 + 1e-6 and end - start >= self.cfg.min_understandable - 1e-6

    def test_extreme_cold_open_never_cuts_a_word_or_the_action(self):
        peak = self.peak(30.1, 31.0, 0.9, 0.88, in_payoff=True)
        words = [{"id": "w1", "text": "NO!", "start": 31.2, "end": 31.8}]
        start, end, _ = self.window(peak, words=words, cuts=[29.95])     # a shot cut just before the action
        assert not (31.2 < end < 31.8)
        assert start >= 29.95                                            # no flash of the previous shot
