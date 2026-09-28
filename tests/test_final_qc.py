"""Final QC on rendered MP4s: the ruler must pass a correct short AND catch each failure class.

Every fixture is real media built with FFmpeg; the good candidate is composed
with the production intro filter graph (intro_renderer.build_filter_complex),
and each fault is injected into an otherwise identical render.
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))

try:
    import numpy  # noqa: F401

    HAVE_NUMPY = True
except ImportError:  # pragma: no cover
    HAVE_NUMPY = False

W, H, FPS = 320, 180, 30
INTRO = (8.0, 9.5)
WORDS = [("alpha", 5.0, 5.3), ("beta", 5.4, 5.7), ("gamma", 5.8, 6.1),
         ("delta", 8.2, 8.5), ("echo", 8.6, 8.9), ("fox", 9.0, 9.3)]


def run(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


def ass_time(seconds: float) -> str:
    cs = int(round(seconds * 100))
    return f"0:{cs // 6000:02d}:{(cs // 100) % 60:02d}.{cs % 100:02d}"


def caption_ass(path: Path) -> Path:
    """Presentation-style ASS: one positioned line per page, future words transparent."""
    header = (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {W}\nPlayResY: {H}\nScaledBorderAndShadow: yes\n\n"
              "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
              "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, "
              "Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
              "Style: Main,DejaVu Sans,18,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,"
              "0,0,0,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    events = []
    for page in (WORDS[:3], WORDS[3:]):
        end = page[-1][2] + 0.4
        for active in range(len(page)):
            a = page[active][1]
            b = page[active + 1][1] if active + 1 < len(page) else end
            parts = ["{\\an2\\pos(160,170)}"]
            for index, (word, _s, _e) in enumerate(page):
                spacer = " " if index else ""
                if index > active:
                    parts.append(("{\\1a&HFF&\\3a&HFF&}" if index == active + 1 else "") + spacer + word)
                else:
                    parts.append(spacer + word)
            events.append(f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Main,,0,0,0,,{''.join(parts)}")
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return path


def truth_doc(path: Path, words=WORDS) -> Path:
    path.write_text(json.dumps({"kind": "mimir_caption_truth", "version": 1, "signature": "x",
                                "words": [[i, w, s, e, "A", ""] for i, (w, s, e) in enumerate(words)]}),
                    encoding="utf-8")
    return path


def timeline_doc(main_duration: float, headline: str = "") -> dict:
    duration = INTRO[1] - INTRO[0]
    return {"version": 1, "kind": "mimir_final_timeline", "fps": FPS,
            "intro": {"source": "clean_paced_clip", "paced": list(INTRO), "duration": duration, "headline": headline,
                      "headline_final": [0.0, 1.2] if headline else None, "normal_captions": False,
                      "peak": {"peak_start": 8.3, "peak_end": 9.2}},
            "restart": {"transition": "hard_cut", "main_restart_paced": 0.0, "first_caption_paced": 5.0,
                        "protected_paced": [[4.8, 6.3]]},
            "main": {"duration": main_duration, "effective_duration": main_duration},
            "expected_final_duration": duration + main_duration}


def compose(intro_source: Path, main: Path, out: Path, *, main_start: float = 0.0, hook_ass: str = "",
            silent_intro: bool = False) -> Path:
    from ai.editor import intro_renderer

    graph, maps = intro_renderer.build_filter_complex(
        teaser_start=INTRO[0], teaser_end=INTRO[1], ass_path=hook_ass, width=W, height=H, fps=float(FPS),
        edited_has_audio=True, main_has_audio=True, transition_duration=0.0, main_start=main_start)
    if silent_intro:
        graph = graph.replace("[0:a]atrim=", "[0:a]volume=0,atrim=", 1)
    run("-i", str(intro_source), "-i", str(main), "-filter_complex", graph, *maps, "-r", str(FPS),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out))
    return out


@unittest.skipUnless(HAVE_FFMPEG and HAVE_NUMPY, "ffmpeg/numpy not available")
class FinalQcTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from ai.editor import final_qc

        cls.qc = final_qc
        cls._tmp = tempfile.TemporaryDirectory(prefix="mimir qc ş'")
        root = Path(cls._tmp.name)
        cls.root = root
        cls.clean = root / "clean.mp4"
        run("-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate={FPS}:duration=12",
            "-f", "lavfi", "-i", "anoisesrc=color=pink:sample_rate=48000:duration=12:amplitude=0.2",
            "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p",
            "-c:a", "aac", str(cls.clean))
        cls.ass = caption_ass(root / "captions.ass")
        cls.truth = truth_doc(root / "truth.json")
        cls.main = root / "main.mp4"
        from ai.editor.caption_renderer import escape_filter_path

        run("-i", str(cls.clean), "-vf", f"subtitles=filename='{escape_filter_path(cls.ass)}'",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p", "-c:a", "copy",
            str(cls.main))
        cls.main_duration = final_qc.video_facts(cls.main)["duration"]
        cls.good = compose(cls.clean, cls.main, root / "good.mp4")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def context(self, candidate: Path, **overrides):
        values = dict(candidate=candidate, timeline_doc=timeline_doc(self.main_duration), intro_source=self.clean,
                      main_render=self.main, burned_ass=self.ass, truth_path=self.truth)
        values.update(overrides)
        return self.qc.QcContext(**values)

    def statuses(self, ctx) -> dict[str, str]:
        return {row["check"]: row["status"] for row in self.qc.run_final_qc(ctx)}

    def test_correct_short_passes_every_check(self) -> None:
        rows = self.qc.run_final_qc(self.context(self.good))
        failed = [(r["check"], r["detail"]) for r in rows if r["status"] != "pass"]
        self.assertEqual(failed, [])
        self.assertGreaterEqual(len(rows), 15)

    def test_speech_captions_in_the_intro_are_caught(self) -> None:
        bad = compose(self.main, self.main, self.root / "captions_in_intro.mp4")   # intro cut from the captioned main
        status = self.statuses(self.context(bad))
        self.assertEqual(status["no_captions_in_intro"], "fail")

    def test_frozen_intro_is_caught(self) -> None:
        still = self.root / "still.mp4"
        run("-i", str(self.clean), "-vf", f"select='eq(n\\,{int(INTRO[0] * FPS)})',loop=loop=-1:size=1,"
            f"setpts=N/({FPS}*TB),trim=duration=12", "-af", "anull", "-t", "12", "-r", str(FPS), "-c:v", "libx264",
            "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(still))
        bad = compose(still, self.main, self.root / "frozen.mp4")
        status = self.statuses(self.context(bad))
        self.assertEqual(status["intro_moving"], "fail")
        self.assertEqual(status["intro_is_source_footage"], "fail")

    def test_offset_main_is_caught(self) -> None:
        bad = compose(self.clean, self.main, self.root / "offset.mp4", main_start=0.4)
        status = self.statuses(self.context(bad))
        self.assertEqual(status["main_matches_render"], "fail")
        self.assertEqual(status["main_av_sync"], "fail")
        self.assertEqual(status["composition_clock"], "fail")

    def test_silent_intro_is_caught(self) -> None:
        bad = compose(self.clean, self.main, self.root / "silent.mp4", silent_intro=True)
        self.assertEqual(self.statuses(self.context(bad))["intro_audio"], "fail")

    def test_mistimed_missing_and_extra_caption_words_are_caught(self) -> None:
        late = [list(w) for w in WORDS]
        late[1][1] += 0.1
        status = self.statuses(self.context(self.good, truth_path=truth_doc(self.root / "late.json",
                                                                            [tuple(w) for w in late])))
        self.assertEqual(status["caption_words_timing"], "fail")
        dropped = self.statuses(self.context(self.good, truth_path=truth_doc(self.root / "drop.json", WORDS[1:])))
        self.assertEqual(dropped["caption_words_timing"], "fail")         # "alpha" burned but not in truth
        extra = WORDS + [("golf", 10.0, 10.3)]
        missing = self.statuses(self.context(self.good, truth_path=truth_doc(self.root / "extra.json", extra)))
        self.assertEqual(missing["caption_words_timing"], "fail")         # spoken word never burned

    def test_words_cut_by_the_restart_are_caught(self) -> None:
        doc = timeline_doc(self.main_duration)
        doc["restart"]["main_restart_paced"] = 5.5
        rows = {r["check"]: r for r in self.qc.run_final_qc(self.context(self.good, timeline_doc=doc))}
        self.assertEqual(rows["caption_words_timing"]["status"], "fail")
        self.assertEqual(rows["protected_story_present"]["status"], "fail")

    def test_missing_headline_is_caught_and_no_headline_is_legitimate(self) -> None:
        expecting = self.context(self.good, timeline_doc=timeline_doc(self.main_duration, headline="HE DID WHAT"))
        self.assertEqual(self.statuses(expecting)["intro_headline"], "fail")
        self.assertEqual(self.statuses(self.context(self.good))["intro_headline"], "pass")

    def test_effect_covering_captions_is_caught(self) -> None:
        covered = self.root / "covered.mp4"
        intro = INTRO[1] - INTRO[0]
        run("-i", str(self.good), "-vf", f"drawbox=x=40:y=150:w=240:h=30:color=white@1.0:t=fill:"
            f"enable='between(t,{intro + 5.0},{intro + 6.2})'", "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "copy", str(covered))
        status = self.statuses(self.context(covered, effect_windows=[(intro + 5.0, intro + 6.2)]))
        self.assertEqual(status["effects_clear"], "fail")
        inside_intro = self.statuses(self.context(self.good, effect_windows=[(0.5, 1.0)]))
        self.assertEqual(inside_intro["effects_clear"], "fail")

    def test_peak_before_restart_is_caught(self) -> None:
        doc = timeline_doc(self.main_duration)
        doc["restart"]["main_restart_paced"] = 8.5
        self.assertEqual(self.statuses(self.context(self.good, timeline_doc=doc))["peak_recurs"], "fail")

    def test_burned_word_parser_handles_merged_pieces_and_labels(self) -> None:
        onsets = self.qc.burned_word_onsets(self.ass)
        self.assertEqual([t for _, t in onsets], [w for w, _, _ in WORDS])
        self.assertEqual([round(t, 2) for t, _ in onsets], [s for _, s, _ in WORDS])

    def test_unreadable_candidate_fails_closed(self) -> None:
        broken = self.root / "broken.mp4"
        broken.write_bytes(b"\x00" * 4096)
        rows = self.qc.run_final_qc(self.context(broken))
        self.assertTrue(rows and all(r["status"] == "fail" for r in rows))


if __name__ == "__main__":
    unittest.main()
