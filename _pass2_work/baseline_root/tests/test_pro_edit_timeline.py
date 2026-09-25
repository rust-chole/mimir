from __future__ import annotations

import copy
import unittest

import pro_edit_fixtures as fx  # noqa: F401  (sets sys.path)
from ai.editor.pro_edit.errors import EditContextError, EditPlanTimelineError
from ai.editor.pro_edit.schema import StoryRole
from ai.editor.pro_edit.timebase import (
    ClipTimelineMap,
    FrameRate,
    IntroTimelineMap,
    TimelineDomain as D,
    TimeRange,
    Timestamp,
)


def clip_map(cuts: list[tuple[float, float]] | None = None) -> ClipTimelineMap:
    return ClipTimelineMap.from_timeline_clip({
        "source": {"absolute_start": 80.0, "absolute_end": 100.0, "duration": 20.0},
        "cut_ranges": [{"start": a, "end": b} for a, b in (cuts or [])],
    })


class TimelineDomainTests(unittest.TestCase):
    def test_vod_to_raw_spec_example(self) -> None:
        # VOD 100 s, clip starts at VOD 80 s, event at VOD 86.5 s -> RAW 6.5 s
        m = clip_map()
        raw = m.vod_to_raw(Timestamp(86.5, D.VOD))
        self.assertEqual(raw.domain, D.RAW_CLIP)
        self.assertAlmostEqual(raw.seconds, 6.5)
        self.assertAlmostEqual(m.raw_to_vod(raw).seconds, 86.5)

    def test_mapping_around_removed_interval(self) -> None:
        m = clip_map([(5.0, 6.0), (10.0, 12.0)])
        self.assertAlmostEqual(m.paced_duration, 17.0)
        cases = {4.0: 4.0, 5.0: 5.0, 5.5: 5.0, 6.0: 5.0, 7.0: 6.0, 11.0: 9.0, 12.0: 9.0, 15.0: 12.0}
        for raw, paced in cases.items():
            got = m.raw_to_paced(Timestamp(raw, D.RAW_CLIP))
            self.assertAlmostEqual(got.seconds, paced, places=3, msg=f"raw {raw}")
        self.assertTrue(m.is_removed(Timestamp(5.5, D.RAW_CLIP)))
        self.assertFalse(m.is_removed(Timestamp(6.0, D.RAW_CLIP)))
        for paced in (0.0, 4.9, 5.0, 8.99, 9.0, 16.5):
            raw = m.paced_to_raw(Timestamp(paced, D.PACED_CLIP))
            back = m.raw_to_paced(raw)
            self.assertAlmostEqual(back.seconds, paced, places=3)

    def test_range_fully_removed_is_none(self) -> None:
        m = clip_map([(5.0, 6.0)])
        self.assertIsNone(m.range_to_paced(TimeRange(5.1, 5.9, D.RAW_CLIP)))
        partial = m.range_to_paced(TimeRange(4.5, 6.5, D.RAW_CLIP))
        self.assertIsNotNone(partial)
        assert partial is not None
        self.assertAlmostEqual(partial.start, 4.5)
        self.assertAlmostEqual(partial.end, 5.5)
        vod = m.range_to_paced(TimeRange(87.0, 88.0, D.VOD))
        assert vod is not None
        self.assertAlmostEqual(vod.start, 6.0)

    def test_no_silent_domain_mixing(self) -> None:
        m = clip_map()
        with self.assertRaises(EditPlanTimelineError):
            m.vod_to_raw(Timestamp(3.0, D.PACED_CLIP))
        with self.assertRaises(EditPlanTimelineError):
            m.convert(Timestamp(1.0, D.PACED_CLIP), D.FINAL)
        with self.assertRaises(EditPlanTimelineError):
            TimeRange(0, 1, D.VOD).overlaps(TimeRange(0, 1, D.PACED_CLIP))
        with self.assertRaises(EditPlanTimelineError):
            Timestamp(float("nan"), D.VOD)

    def test_intro_final_mapping(self) -> None:
        intro = IntroTimelineMap(teaser_duration=2.0, transition_duration=0.16, main_restart=1.5)
        self.assertIsNone(intro.paced_to_final(Timestamp(1.0, D.PACED_CLIP)))
        final = intro.paced_to_final(Timestamp(3.5, D.PACED_CLIP))
        assert final is not None
        self.assertEqual(final.domain, D.FINAL)
        self.assertAlmostEqual(final.seconds, 2.0 - 0.16 + 2.0)


class FrameRateTests(unittest.TestCase):
    def test_ntsc_rational_rates(self) -> None:
        for text, expected in (("30000/1001", 29.97002997), ("60000/1001", 59.94005994), ("25/1", 25.0)):
            rate = FrameRate.parse(text)
            self.assertAlmostEqual(rate.fps, expected, places=6)
        ntsc = FrameRate.parse("30000/1001")
        self.assertEqual(ntsc.frame_index(0.0), 0)
        self.assertEqual(ntsc.frame_index(1001 / 30000), 1)
        self.assertEqual(ntsc.frame_index(10.0), 300)  # 299.7 rounds to 300
        self.assertEqual(ntsc.frame_index(ntsc.frame_time(12345)), 12345)
        self.assertEqual(FrameRate.parse("60000/1001").frame_index(60.06), 3600)

    def test_invalid_rates(self) -> None:
        for bad in ("0/1", "abc", "-30", 0, float("inf")):
            with self.assertRaises(EditPlanTimelineError):
                FrameRate.parse(bad)


class ContextBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = fx.Workspace()

    def tearDown(self) -> None:
        self.ws.cleanup()

    def test_story_ranges_mapped_to_paced(self) -> None:
        ctx = fx.make_context(self.ws)
        story = ctx.story
        self.assertEqual([(r.start, r.end) for r in story.payoff_ranges], [(13.0, 16.0)])
        self.assertEqual([(r.start, r.end) for r in story.escalation_ranges], [(7.0, 9.0)])
        self.assertIn((16.2, 18.0), [(r.start, r.end) for r in story.reaction_ranges])
        self.assertEqual([(r.start, r.end) for r in story.must_keep_ranges], [(12.8, 17.5)])
        roles = {s.role for s in story.segments}
        self.assertTrue({StoryRole.HOOK, StoryRole.SETUP, StoryRole.PAYOFF, StoryRole.REACTION} <= roles)
        self.assertEqual(ctx.dominant_role(13.5, 15.0), StoryRole.PAYOFF)
        # Segments tile [0, duration] with no gaps.
        self.assertAlmostEqual(story.segments[0].start, 0.0)
        self.assertAlmostEqual(story.segments[-1].end, ctx.clip.duration_s)
        for a, b in zip(story.segments, story.segments[1:]):
            self.assertAlmostEqual(a.end, b.start)

    def test_words_are_referenced_not_copied_into_plan_space(self) -> None:
        ctx = fx.make_context(self.ws)
        self.assertEqual([w.id for w in ctx.words], list(range(len(fx.WORDS))))
        self.assertEqual(ctx.words[8].text, "THIS")
        self.assertEqual(ctx.speaker_ids(), frozenset({"A", "B"}))

    def test_context_does_not_mutate_source_timeline(self) -> None:
        timeline = copy.deepcopy(fx.BASE_TIMELINE)
        snapshot = copy.deepcopy(timeline)
        fx.make_context(self.ws, timeline=timeline)
        self.assertEqual(timeline, snapshot)

    def test_missing_evidence_stays_empty(self) -> None:
        ctx = fx.make_context(self.ws, report=None, words=[])
        self.assertEqual(ctx.words, ())
        self.assertEqual(ctx.visual_events, ())
        self.assertEqual(ctx.subject_tracks, ())
        self.assertEqual(ctx.laughter_events, ())

    def test_stale_cut_ranges_refused(self) -> None:
        with self.assertRaises(EditContextError):
            fx.make_context(self.ws, media_info=fx.media(duration=24.0))


if __name__ == "__main__":
    unittest.main()
