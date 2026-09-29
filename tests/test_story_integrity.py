"""Causal story integrity after selection: pass / one bounded repair / warn / reselect (no duration buckets)."""
from __future__ import annotations

import unittest

from ai.editor import story_integrity as si

MAX = 55.0


def phrase(start: float, n: int, step: float = 0.3, word: str = "w") -> list[dict]:
    return [{"word": f"{word}{i}", "start": round(start + i * step, 3), "end": round(start + i * step + 0.25, 3)}
            for i in range(n)]


def evidence(*phrases: list[dict], visual=()) -> si.Evidence:
    return si.Evidence.build({"words": [w for p in phrases for w in p]}, visual)


def clip(start: float, end: float, payoff=(20.0, 22.0), keep=(), anchors=()) -> dict:
    return {"start": start, "end": end, "duration": end - start, "payoff_start": payoff[0], "payoff_end": payoff[1],
            "must_keep_ranges": [{"start": a, "end": b, "reason": "money"} for a, b in keep],
            "anchor_moments": list(anchors)}


class IntegrityTests(unittest.TestCase):
    def review(self, c, ev):
        return si.review(c, ev, source_duration=120.0, max_duration=MAX)

    def test_a_causally_complete_story_passes_untouched(self) -> None:
        ev = evidence(phrase(12.0, 10), phrase(20.0, 6), phrase(23.0, 4))
        result = self.review(clip(11.0, 26.0), ev)
        self.assertEqual(result.status, "pass", result.checks)
        self.assertEqual(result.final, (11.0, 26.0))

    def test_a_start_inside_a_running_sentence_is_completed_once(self) -> None:
        ev = evidence(phrase(10.0, 12), phrase(20.0, 6), phrase(23.0, 4))        # 10.0 .. 13.55 one phrase
        result = self.review(clip(12.1, 26.0, keep=[(19.8, 22.3)]), ev)
        self.assertEqual(result.status, "repaired")
        self.assertEqual(result.final[0], 10.0)
        self.assertEqual(result.clip["must_keep_ranges"][0], {"start": 19.8, "end": 22.3, "reason": "money"})
        self.assertTrue(result.recheck)                                             # re-checked, no second round

    def test_missing_setup_brings_in_the_adjacent_phrase_and_protects_it(self) -> None:
        ev = evidence(phrase(17.0, 6), phrase(20.0, 6), phrase(23.0, 4))            # setup phrase ends ~18.75
        result = self.review(clip(19.8, 26.0), ev)
        self.assertEqual(result.status, "repaired")
        self.assertAlmostEqual(result.final[0], 17.0)
        kinds = [r.get("kind") for r in result.clip["must_keep_ranges"]]
        self.assertIn("causal_setup", kinds)

    def test_a_silent_visual_setup_is_a_setup(self) -> None:
        ev = evidence(phrase(20.0, 6), phrase(23.0, 4), visual=[{"start": 16.0, "end": 19.0, "type": "action"}])
        result = self.review(clip(15.5, 26.0), ev)
        self.assertEqual(result.status, "pass", result.checks)

    def test_a_one_word_reaction_is_enough(self) -> None:
        ev = evidence(phrase(12.0, 10), phrase(20.0, 6), phrase(22.6, 1))
        self.assertEqual(self.review(clip(11.0, 23.5), ev).status, "pass")

    def test_an_unprovable_reaction_warns_instead_of_rejecting(self) -> None:
        ev = evidence(phrase(12.0, 10), phrase(20.0, 6), phrase(30.0, 4))           # next speech after a scene gap
        result = self.review(clip(11.0, 22.6), ev)
        self.assertEqual(result.status, "warn")
        self.assertEqual(result.final, (11.0, 22.6))

    def test_expansion_never_exceeds_the_analyzer_maximum(self) -> None:
        ev = evidence(phrase(12.0, 10), phrase(20.0, 6), phrase(62.8, 30, step=0.2))
        c = clip(11.0, 62.9, payoff=(20.0, 22.0))                                   # end cuts the last phrase
        result = self.review(c, ev)
        self.assertLessEqual(result.final[1] - result.final[0], MAX + 1e-6)
        self.assertEqual(result.final, (11.0, 62.9))
        self.assertEqual(result.status, "warn")

    def test_a_protected_range_crossing_a_boundary_is_included_not_cut(self) -> None:
        ev = evidence(phrase(12.0, 10), phrase(20.0, 6), phrase(23.0, 4))
        result = self.review(clip(11.0, 24.0, keep=[(23.0, 25.5)]), ev)
        self.assertEqual(result.status, "repaired")
        self.assertGreaterEqual(result.final[1], 25.5)

    def test_missing_payoff_reselects_only_with_an_alternate_and_never_a_pinned_clip(self) -> None:
        ev = evidence(phrase(12.0, 10), phrase(20.0, 6), phrase(23.0, 4), phrase(60.0, 10), phrase(64.0, 6))
        broken = clip(30.0, 50.0, payoff=(20.0, 22.0))                               # claims a payoff it lacks
        good = clip(59.0, 67.0, payoff=(64.0, 65.0))
        index, result, considered = si.review_selection([broken, good], 1, evidence=ev, source_duration=120.0,
                                                        max_duration=MAX, rank=[1, 2], user_pinned=False)
        self.assertEqual(index, 2)
        self.assertTrue(result.status.startswith("reselected"))
        self.assertEqual(considered[0]["clip_index"], 2)
        index, result, _ = si.review_selection([broken, good], 1, evidence=ev, source_duration=120.0,
                                               max_duration=MAX, rank=[1, 2], user_pinned=True)
        self.assertEqual((index, result.status), (1, "warn"))
        index, result, _ = si.review_selection([broken], 1, evidence=ev, source_duration=120.0,
                                               max_duration=MAX, rank=[1], user_pinned=False)
        self.assertEqual((index, result.status), (1, "warn"))                        # no alternate: kept, disclosed

    def test_boundaries_only_ever_expand(self) -> None:
        ev = evidence(phrase(12.0, 10), phrase(20.0, 6), phrase(23.0, 4))
        for duration in (0.0, 24.0, 120.0):                     # even a too-short "source" never shrinks it
            result = si.review(clip(10.0, 30.0), ev, source_duration=duration, max_duration=MAX)
            self.assertLessEqual(result.final[0], 10.0)
            self.assertGreaterEqual(result.final[1], 30.0)

    def test_no_duration_bucket_constants(self) -> None:
        names = [n for n in dir(si) if n.isupper()]
        self.assertFalse([n for n in names if "MIN_" in n and ("SETUP" in n or "REACTION" in n or "BEAT" in n)])


if __name__ == "__main__":
    unittest.main()
