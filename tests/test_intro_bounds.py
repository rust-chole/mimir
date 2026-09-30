"""Evidence-derived cold-open bounds (ai/editor/intro_bounds.py).

Synthetic evidence only: every expectation follows from the event shape, the
words and the shots, never from a fixed per-type duration.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from ai.editor import intro_bounds as ib
from ai.editor.intro_bounds import Envelope, EventEvidence, Word

FRAME = ib.ENVELOPE_FRAME_S


def envelope(duration: float, floor: float = -50.0, loud: list[tuple[float, float, float]] = ()) -> Envelope:
    """Frame levels: ``floor`` everywhere, ``level`` inside each (start, end, level)."""
    count = int(round(duration / FRAME))
    levels = [floor] * count
    for start, end, level in loud:
        for index in range(int(start / FRAME), min(count, int(end / FRAME))):
            levels[index] = level
    return Envelope(tuple(levels))


def spoken(text: str, start: float, gap: float = 0.08, word_s: float = 0.26) -> list[Word]:
    words, t = [], start
    for token in text.split():
        words.append(Word(token, round(t, 3), round(t + word_s, 3)))
        t += word_s + gap
    return words


class IntroBoundsTests(unittest.TestCase):
    def test_tiny_self_contained_event_gives_a_small_intro(self) -> None:
        env = envelope(20.0, loud=[(10.0, 10.35, -8.0)])
        bounds = ib.compute_intro_bounds(EventEvidence(10.02, 10.30), clip_duration=20.0, envelope=env)
        self.assertLessEqual(bounds.duration, ib.MIN_INTRO_S + 0.05)
        self.assertLessEqual(bounds.start, 10.0)
        self.assertGreaterEqual(bounds.end, 10.35)

    def test_reaction_that_lands_after_the_event_is_included(self) -> None:
        # Impact, then a sustained scream that only ends at 12.4 s.
        env = envelope(20.0, loud=[(10.0, 10.4, -8.0), (10.5, 12.4, -12.0)])
        tiny = ib.compute_intro_bounds(EventEvidence(10.02, 10.30), clip_duration=20.0,
                                       envelope=envelope(20.0, loud=[(10.0, 10.35, -8.0)]))
        bounds = ib.compute_intro_bounds(EventEvidence(10.02, 10.30), clip_duration=20.0, envelope=env)
        self.assertGreaterEqual(bounds.end, 12.4)
        self.assertGreater(bounds.duration, tiny.duration + 1.5)
        self.assertLessEqual(bounds.duration, ib.MAX_INTRO_S)

    def test_the_phrase_that_carries_the_event_is_never_cut(self) -> None:
        words = spoken("what the hell is that", 5.0)
        env = envelope(15.0, loud=[(5.0, 6.6, -14.0)])
        bounds = ib.compute_intro_bounds(EventEvidence(5.4, 5.7), clip_duration=15.0, words=words, envelope=env)
        self.assertLessEqual(bounds.start, words[0].start)
        self.assertGreaterEqual(bounds.end, words[-1].end)
        for word in words:
            self.assertFalse(word.start < bounds.start < word.end or word.start < bounds.end < word.end)

    def test_an_unrelated_edge_phrase_is_excluded_at_the_pause(self) -> None:
        setup = spoken("so anyway i was just telling chat about the new setup today", 2.0)
        event = spoken("oh no", setup[-1].end + 0.6)
        words = setup + event
        env = envelope(15.0, loud=[(2.0, setup[-1].end, -20.0), (event[0].start, event[-1].end + 0.2, -8.0)])
        bounds = ib.compute_intro_bounds(EventEvidence(event[0].start, event[-1].end), clip_duration=15.0,
                                         words=words, envelope=env)
        self.assertGreaterEqual(bounds.start, setup[-1].end)
        self.assertGreaterEqual(bounds.end, event[-1].end)

    def test_continuous_audio_does_not_stretch_to_the_search_limit(self) -> None:
        env = envelope(30.0, floor=-18.0, loud=[(12.0, 12.5, -6.0)])   # loud bed everywhere, one louder hit
        flat = envelope(30.0, floor=-18.0)
        noisy = ib.compute_intro_bounds(EventEvidence(12.0, 12.5), clip_duration=30.0, envelope=flat)
        self.assertIsNone(noisy.onset)
        self.assertLess(noisy.duration, ib.MAX_ONSET_SEARCH_S + ib.MAX_DECAY_SEARCH_S)
        hit = ib.compute_intro_bounds(EventEvidence(12.0, 12.5), clip_duration=30.0, envelope=env)
        self.assertLess(hit.duration, 2.0)

    def test_no_flash_frame_shot_at_the_start(self) -> None:
        env = envelope(20.0, loud=[(9.6, 11.0, -8.0)])
        bounds = ib.compute_intro_bounds(EventEvidence(10.0, 10.8), clip_duration=20.0, envelope=env, cuts=[9.8])
        self.assertAlmostEqual(bounds.start, 9.8, places=3)
        self.assertTrue(bounds.shot_snaps)

    def test_safety_ceiling_cuts_only_at_word_gaps(self) -> None:
        words = spoken(" ".join(f"w{i}" for i in range(40)), 1.0, gap=0.05)   # one 12 s breathless phrase
        env = envelope(20.0, loud=[(1.0, 13.0, -12.0)])
        bounds = ib.compute_intro_bounds(EventEvidence(6.0, 6.4), clip_duration=20.0, words=words, envelope=env)
        self.assertLessEqual(bounds.duration, ib.MAX_INTRO_S + 1e-6)
        self.assertLessEqual(bounds.start, 6.0)
        self.assertGreaterEqual(bounds.end, 6.4)
        for word in words:
            self.assertFalse(word.start < bounds.start < word.end or word.start < bounds.end < word.end)

    def test_editor_lead_in_is_honoured(self) -> None:
        words = spoken("are you serious right now", 7.0)
        env = envelope(20.0, loud=[(8.2, 8.8, -8.0)])
        bounds = ib.compute_intro_bounds(EventEvidence(8.3, 8.7, lead_in_start=words[0].start), clip_duration=20.0,
                                         words=words, envelope=env)
        self.assertLessEqual(bounds.start, words[0].start)

    def test_durations_follow_the_event_not_a_bucket(self) -> None:
        durations = set()
        for scream_end in (10.9, 11.6, 12.3, 13.1):
            env = envelope(20.0, loud=[(10.0, scream_end, -9.0)])
            durations.add(round(ib.compute_intro_bounds(EventEvidence(10.0, 10.3), clip_duration=20.0,
                                                        envelope=env).duration, 2))
        self.assertEqual(len(durations), 4)

    def test_phrases_split_on_pauses_and_sentence_ends(self) -> None:
        words = spoken("wait. what", 1.0, gap=0.15) + spoken("no way", 3.0)
        phrases = ib.speech_phrases(words)
        self.assertEqual([p.text for p in phrases], ["wait.", "what", "no way"])


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not available")
class IntroBoundsMediaTests(unittest.TestCase):
    def test_envelope_and_scene_cuts_from_real_media(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shots.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "lavfi", "-i", "color=c=black:size=160x90:rate=25:duration=2",
                 "-f", "lavfi", "-i", "color=c=white:size=160x90:rate=25:duration=2",
                 "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                 "-f", "lavfi", "-i", "sine=frequency=500:sample_rate=16000:duration=0.5",
                 "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v];[2:a]atrim=0:1.5[s];[s][3:a]concat=n=2:v=0:a=1,"
                                    "apad=whole_dur=4[a]",
                 "-map", "[v]", "-map", "[a]", "-t", "4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                 str(path)], check=True)
            env = ib.audio_envelope(path)
            self.assertGreater(len(env.db), 150)
            loud = [i * env.frame_s for i, v in enumerate(env.db) if v > -30.0]
            self.assertTrue(loud and 1.4 <= loud[0] <= 1.6)
            cuts = ib.scene_cuts(path, 0.0, 4.0)
            self.assertTrue(any(abs(c - 2.0) <= 0.08 for c in cuts), cuts)


if __name__ == "__main__":
    unittest.main()
