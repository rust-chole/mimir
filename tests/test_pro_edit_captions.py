"""Caption presentation layer (keyless, deterministic; libass test needs FFmpeg)."""
from __future__ import annotations

import copy
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pro_edit_fixtures as fx  # noqa: F401  (sys.path)
from ai.editor import caption_renderer, captions, intro_renderer
from ai.editor.pro_edit import caption_presentation as cp
from ai.editor.pro_edit import font_metrics as fm
from ai.editor.pro_edit.caption_guard import CaptionIntegrity, caption_safe_region
from ai.editor.pro_edit.config import ProEditConfig, load_config
from ai.editor.pro_edit.framing import Box, FramingRequest, SubjectBox, solve_framing
from ai.editor.pro_edit.intro_timeline import IntroTimeline, hook_text_band
from ai.editor.pro_edit.policy import _has_other_channels
from ai.editor.pro_edit.schema import (
    CameraMode,
    CaptionStyle,
    EditEvent,
    EditPlan,
    EditTarget,
    MotionPreset,
    ReasonCode,
    StoryRole,
)
from ai.editor.pro_edit.timebase import TimelineDomain

HAVE_FFMPEG = bool(shutil.which("ffmpeg"))
BUILTIN = fm.builtin_metrics()


def words(text: str, *, start: float = 0.3, gap: float = 0.06, sentence_gap: float = 0.5,
          secondary: range = range(0), label: str = "") -> list[dict]:
    rows, t = [], start
    for index, word in enumerate(text.split()):
        duration = 0.18 + 0.02 * len(word)
        second = index in secondary
        rows.append({"word": word, "edited_start": round(t, 3), "edited_end": round(t + duration, 3),
                     "speaker_raw": "B" if second else "A", "speaker_role": "secondary" if second else "main",
                     "speaker_label": label if second else "", "speaker_confidence": 0.9})
        t += duration + (sentence_gap if word.endswith((".", "!", "?")) else gap)
    return rows


def profile(rows: list[dict], duration: float | None = None) -> dict:
    ends = [float(r["edited_end"]) for r in rows if isinstance(r, dict) and isinstance(r.get("edited_end"),
                                                                                    (int, float))]
    end = max(ends, default=1.0)
    return {"status": "ok", "timing_basis": "exact_final_48k_audio",
            "clip_duration": duration if duration is not None else round(end + 1.0, 3), "words": rows}


def human_profile(rows: list[dict], names: dict[str, str], duration: float | None = None, **extra) -> dict:
    """Profile whose raw -> name mapping passed the human voice checkpoint.

    Captions V24 (current root) prints a name only when it is human-confirmed
    (``captions._trusted_human_display_map``); this mirrors real profiles
    (``speaker_names.source = manual_voice_calibrated_v28``)."""
    data = profile(rows, duration)
    data["display_labels"] = dict(names)
    data["speaker_names"] = {**names, "source": "manual_voice_calibrated_v28"}
    data.update(extra)
    return data


def timeline(duration: float = 60.0, events: list | None = None) -> dict:
    return {"edited": {"estimated_duration": duration}, "events": events or []}


def caption_event(event_id: str, start: float, end: float, style: str = "default", ids=()) -> EditEvent:
    return EditEvent(event_id, start, end, StoryRole.PAYOFF, CameraMode.PRESERVE, MotionPreset.STATIC_CLEAN,
                     EditTarget(), 0.6, 0.9, ReasonCode.PAYOFF_HIT, CaptionStyle(style), tuple(ids))


def plan(*events: EditEvent) -> EditPlan:
    return EditPlan(2, "pro_stream_v1", TimelineDomain.PACED_CLIP, tuple(events))


def build(rows: list[dict], *, width: int = 1080, height: int = 1920, the_plan: EditPlan | None = None,
          events: list | None = None, metrics: fm.FontMetrics = BUILTIN) -> cp.CaptionPresentation:
    return cp.build_presentation(profile=profile(rows), clip_timeline=timeline(events=events), plan=the_plan,
                                 width=width, height=height, metrics=metrics)


SENTENCE = ("I can't believe he actually did that, wait for it... Bu gerçekten inanılmaz bir an! "
            "Anlayamıyorum neden böyle yaptı. And the crowd went absolutely wild tonight")


class TruthTests(unittest.TestCase):
    def test_tokens_equal_the_words_the_baseline_renderer_displays(self) -> None:
        rows = words(SENTENCE)
        rows += [{"word": "  ", "edited_start": 1.0, "edited_end": 1.1},              # empty -> skipped
                 {"word": "late", "edited_start": 99.0, "edited_end": 99.5},           # clamped past end
                 {"word": "tiny", "edited_start": 2.0, "edited_end": 2.01},            # min event floor
                 {"word": "bad", "edited_start": "x", "edited_end": 1.0}, "junk"]
        data = profile(rows, duration=12.0)
        tokens = cp.presentation_tokens(data, 12.0)
        cp.verify_token_parity(tokens, data, 12.0)
        baseline = captions._profile_edited_words(data, 12.0)
        self.assertEqual([(t.text, t.start, t.end) for t in tokens],
                         [(w["word"], w["edited_start"], w["edited_end"]) for w in baseline])
        # ids are indices into the authoritative profile list
        self.assertTrue(all(data["words"][t.word_id]["word"].strip() == t.text for t in tokens))

    def test_presentation_never_mutates_or_retimes_truth(self) -> None:
        rows = words(SENTENCE, secondary=range(10, 16), label="TYLA")
        data = profile(rows)
        frozen = copy.deepcopy(data)
        presentation = cp.build_presentation(profile=data, clip_timeline=timeline(), plan=None, width=1080,
                                             height=1920, metrics=BUILTIN)
        self.assertEqual(data, frozen)
        by_id = {t.word_id: t for t in presentation.tokens}
        seen = [t.word_id for p in presentation.pages for t in p.tokens]
        self.assertEqual(sorted(seen), sorted(by_id))                   # every word exactly once
        for page in presentation.pages:
            self.assertEqual(page.start, page.tokens[0].start)            # never earlier than speech
            boundaries = [cp._cs(t.start) for t in page.tokens] + [cp._cs(page.end)]
            for active, token in enumerate(page.tokens):
                # a word is visible only in intervals starting at/after its own onset
                self.assertGreaterEqual(boundaries[active], cp._cs(token.start))
                self.assertEqual(token, by_id[token.word_id])

    def test_parity_mismatch_is_rejected(self) -> None:
        data = profile(words("one two three"))
        tokens = list(cp.presentation_tokens(data, 5.0))
        tokens[1] = cp.CaptionTokenRef(tokens[1].word_id, "TWO", tokens[1].start, tokens[1].end, "A", "main", "",
                                       "two")
        with self.assertRaises(cp.CaptionPresentationError):
            cp.verify_token_parity(tokens, data, 5.0)

    def test_no_profile_means_no_presentation(self) -> None:
        with self.assertRaises(cp.CaptionPresentationError):
            cp.build_presentation(profile=None, clip_timeline=timeline(), plan=None, width=1080, height=1920)
        with self.assertRaises(cp.CaptionPresentationError):
            cp.build_presentation(profile={"status": "ok", "timing_basis": "vod", "words": words("a b")},
                                  clip_timeline=timeline(), plan=None, width=1080, height=1920)


class PaginationTests(unittest.TestCase):
    def test_hard_breaks_are_never_crossed(self) -> None:
        # Captions V24 group rule (current root): raw A/B diarization is metadata, not a
        # layout command; only a visible (human-confirmed) name change is a speaker break.
        rows = words("alpha beta gamma. delta epsilon zeta eta", secondary=range(5, 7), label="B")
        rows[3]["edited_start"] += 0.5          # >= GROUP_BREAK_GAP before "delta"
        rows[3]["edited_end"] += 0.5
        confirmed = human_profile(rows, {"B": "TYLA"})
        for data in (profile(rows), confirmed):
            presentation = cp.build_presentation(profile=data, clip_timeline=timeline(), plan=None, width=1080,
                                                 height=1920, metrics=BUILTIN)
            for page in presentation.pages:
                texts = [t.text for t in page.tokens]
                self.assertFalse(any(t.endswith(".") for t in texts[:-1]), texts)   # sentence end closes a page
                self.assertEqual(len({t.speaker_label.casefold() for t in page.tokens}), 1)   # one visible identity
                gaps = [b.start - a.end for a, b in zip(page.tokens, page.tokens[1:])]
                self.assertTrue(all(g < captions.GROUP_BREAK_GAP for g in gaps))
        # Unconfirmed "B" is never printed; the confirmed name starts its own page.
        self.assertFalse(any(p.label for p in build(rows).pages))
        named = cp.build_presentation(profile=confirmed, clip_timeline=timeline(), plan=None, width=1080,
                                      height=1920, metrics=BUILTIN)
        tyla = [p for p in named.pages if p.label == "TYLA"]
        self.assertTrue(tyla and all(t.speaker_raw == "B" for p in tyla for t in p.tokens))

    def test_page_stays_visible_through_intra_page_gaps(self) -> None:
        rows = words("we are here now", gap=0.30)        # 0.18 < gap < 0.42: baseline flickers
        data = profile(rows)
        presentation = build(rows)
        self.assertEqual(len(presentation.pages), 1)
        page = presentation.pages[0]
        events = cp.page_events(page, presentation.geometry)
        covered = sorted((a, b) for a, b, *_ in events)
        self.assertEqual(covered[0][0], cp._cs(page.start))
        for (a1, b1), (a2, b2) in zip(covered, covered[1:]):
            self.assertEqual(b1, a2)                       # contiguous: no blank frames inside a page
        self.assertEqual(covered[-1][1], cp._cs(page.end))
        # The baseline ASS for the same words has holes (documented A-gap).
        with tempfile.TemporaryDirectory() as tmp:
            ass = captions.create_ass_for_clip({}, timeline(), Path(tmp) / "b.ass", data)
            times = [line.split(",")[1:3] for line in ass.read_text(encoding="utf-8-sig").splitlines()
                     if line.startswith("Dialogue:")]
        holes = sum(1 for (a, b), (c, d) in zip(times, times[1:]) if b != c)
        self.assertGreater(holes, 0)

    def test_no_orphan_pages_in_ordinary_speech(self) -> None:
        presentation = build(words(SENTENCE))
        self.assertEqual(presentation.metrics()["orphan_pages"], 0)
        self.assertLessEqual(presentation.metrics()["max_words_per_page"], cp.MAX_PAGE_WORDS)

    def test_reading_density_groups_fast_speech_into_fewer_flips(self) -> None:
        slow = build(words("one two three four five six seven eight", gap=0.25))
        fast = build([dict(r, edited_start=round(r["edited_start"] * 0.45, 3),
                           edited_end=round(r["edited_start"] * 0.45 + 0.12, 3))
                      for r in words("one two three four five six seven eight", gap=0.25)])
        self.assertLessEqual(len(fast.pages), len(slow.pages))
        self.assertGreaterEqual(min(p.end - p.start for p in fast.pages), 0.32)   # terminal readability floor

    def test_terminal_hold_capped_at_next_page_in_same_lane(self) -> None:
        rows = words("quick. next")
        rows[1]["edited_start"] = rows[0]["edited_end"] + 0.10
        rows[1]["edited_end"] = rows[1]["edited_start"] + 0.3
        presentation = build(rows)
        first, second = presentation.pages
        self.assertLessEqual(first.end, second.start)
        self.assertGreaterEqual(first.end, first.tokens[-1].end)


class LayoutTests(unittest.TestCase):
    def test_rendered_width_not_character_count_drives_breaks(self) -> None:
        geometry = cp.LayoutGeometry.for_output(1080, 1920)
        measure = cp._Measure(BUILTIN, geometry.font_px)
        wide, narrow = ["WWWWW", "MMMMM", "WWWWW"], ["iiiii", "lllll", "iiiii"]
        limit = geometry.max_line_px - 2 * geometry.outline_px
        self.assertEqual(len(cp.break_lines(narrow, [1, 1, 1], measure, limit, max_lines=2)[0]), 1)
        self.assertEqual(len(cp.break_lines(wide, [1, 1, 1], measure, limit, max_lines=2)[0]), 2)

    def test_lines_fit_and_avoid_single_word_lines(self) -> None:
        rows = words("Anlayamıyorum konuşmalarından sıkıldığımızı hissettiriyorsunuz")
        presentation = build(rows)
        geometry = presentation.geometry
        for page in presentation.pages:
            for line in page.lines:
                self.assertLessEqual(line.width_px + 2 * geometry.outline_px, geometry.max_line_px + 0.5)
        geometry = cp.LayoutGeometry.for_output(1080, 1920)
        measure = cp._Measure(BUILTIN, geometry.font_px)
        breaks, _ = cp.break_lines(["this", "is", "a", "fairly", "long", "caption"], [1] * 6, measure, 700,
                                   max_lines=2)
        self.assertNotIn(1, breaks)                       # no single-word top line
        self.assertNotEqual(breaks[0], 3)                 # never end a line on "a"

    def test_overlong_single_word_is_fit_scaled_not_overflowing(self) -> None:
        presentation = build(words("Muvaffakiyetsizleştiricileştiriveremeyebileceklerimizdenmişsinizcesine"))
        page = presentation.pages[0]
        self.assertLess(page.fit_scale, 1.0)
        self.assertIn("\\fs", presentation.ass_text)
        self.assertLessEqual(page.lines[0].width_px, presentation.geometry.width)

    def test_layout_profiles(self) -> None:
        cases = {(1080, 1920): "vertical", (1920, 1080): "landscape", (1080, 1080): "square",
                 (1080, 1350): "generic_preserve", (1440, 1080): "generic_preserve", (720, 1280): "vertical"}
        for (w, h), name in cases.items():
            geometry = cp.LayoutGeometry.for_output(w, h)
            self.assertEqual(geometry.profile.name.value, name, (w, h))
            self.assertLess(geometry.max_line_px, w)
            self.assertLess(geometry.main_bottom, h)
        vertical = cp.LayoutGeometry.for_output(1080, 1920)
        self.assertEqual(vertical.font_px, captions.FONT_SIZE)                          # MIMIR geometry kept
        self.assertEqual(vertical.main_bottom, captions.PLAY_RES_Y - captions.CAPTION_MARGIN_V)
        landscape = cp.LayoutGeometry.for_output(1920, 1080)
        self.assertGreater(landscape.font_px / 1080, 74 / 1920 * 1.3)   # not the tiny 1920-PlayRes scaling
        with self.assertRaises(cp.CaptionPresentationError):
            cp.select_layout_profile(0, 1080)
        ass = build(words("hello there"), width=1920, height=1080).ass_text
        self.assertIn("PlayResX: 1920", ass)
        self.assertIn("PlayResY: 1080", ass)

    def test_geometry_is_stable_while_a_page_is_on_screen(self) -> None:
        rows = words("keep this line perfectly still", gap=0.1)
        presentation = build(rows, the_plan=plan(caption_event("e1", 0.0, 5.0, "emphasis", [2])))
        for page in presentation.pages:
            events = cp.page_events(page, presentation.geometry)
            by_line: dict[str, set[str]] = {}
            for _a, _b, _layer, _style, text in events:
                pos = re.search(r"\\pos\([^)]*\)", text).group(0)
                scales = tuple(re.findall(r"\\fscx([\d.]+)", text))
                by_line.setdefault(pos, set()).add(scales)
                self.assertNotIn("\\r", text)                        # no reset (would drop page tags)
                self.assertNotRegex(text, r"\\t\([^)]*\\fsc")         # no width-changing animation
            for pos, variants in by_line.items():
                self.assertEqual(len(variants), 1, (pos, variants))   # same scale tags in every state

    def test_speaker_label_is_width_inclusive_and_lanes_do_not_collide(self) -> None:
        rows = words("main speaker talks here then guest replies back", secondary=range(5, 8), label="TYLA")
        # Captions V24: the secondary lane exists only inside MEASURED overlap, and a
        # label is printed only when human-confirmed -> the guest talks over the main speaker.
        for offset, index in enumerate(range(5, 8)):
            anchor = rows[2 + offset]
            rows[index]["edited_start"] = round(anchor["edited_start"] + 0.05, 3)
            rows[index]["edited_end"] = round(anchor["edited_end"] + 0.05, 3)
        data = human_profile(rows, {"B": "TYLA"}, primary_speaker="A", secondary_speaker="B")
        presentation = cp.build_presentation(profile=data, clip_timeline=timeline(), plan=None, width=1080,
                                             height=1920, metrics=BUILTIN)
        secondary = [p for p in presentation.pages if p.lane == "secondary"]
        self.assertTrue(secondary and all(p.label == "TYLA" for p in secondary))
        measure = cp._Measure(BUILTIN, presentation.geometry.font_px)
        line = secondary[0].lines[0]
        texts = [next(t.text for t in secondary[0].tokens if t.word_id == w) for w in line.word_ids]
        self.assertAlmostEqual(line.width_px, round(cp.line_width(texts, [1.0] * len(texts), measure, "TYLA"), 1))
        self.assertGreater(line.width_px, cp.line_width(texts, [1.0] * len(texts), measure))
        self.assertIn("TYLA:", presentation.ass_text)
        # Actual line boxes (+ outline) of the two lanes never overlap.
        geometry = presentation.geometry
        main_top = presentation.lane_bottoms["main"] - BUILTIN.line_height(geometry.font_px) - geometry.outline_px
        second_bottom = presentation.lane_bottoms["secondary"] + geometry.outline_px + geometry.shadow_px
        self.assertLess(second_bottom, main_top)
        region = caption_safe_region(self._write(presentation))
        self.assertEqual({b.style for b in region.bands}, {"MimirMain", "MimirSecondary"})

    def _write(self, presentation: cp.CaptionPresentation) -> Path:
        self._tmp = tempfile.TemporaryDirectory()
        return cp.write_presentation(presentation, Path(self._tmp.name) / "p.ass")

    def tearDown(self) -> None:
        if hasattr(self, "_tmp"):
            self._tmp.cleanup()


class GlyphMetricsTests(unittest.TestCase):
    def test_builtin_table_turkish_and_fallbacks(self) -> None:
        self.assertEqual(BUILTIN.advance_units("ş"), BUILTIN.advance_units("s"))
        self.assertEqual(BUILTIN.advance_units("İ"), BUILTIN.advance_units("I"))
        self.assertEqual(BUILTIN.advance_units("ğ"), BUILTIN.advance_units("g"))
        self.assertEqual(BUILTIN.advance_units("ı"), 278)
        self.assertEqual(BUILTIN.advance_units("\u0307"), 0.0)                      # combining mark
        self.assertGreaterEqual(BUILTIN.advance_units("漢"), 1000)                  # wide glyph: conservative
        self.assertGreater(BUILTIN.text_width("WWW", 74), BUILTIN.text_width("iii", 74) * 3)
        # libass sizing: advance = units * size / (winAscent + winDescent)
        self.assertAlmostEqual(BUILTIN.text_width("H", 74), 722 * 74 / (1000 * 2288 / 2048), places=6)

    def test_parsed_system_font_agrees_with_builtin_table(self) -> None:
        metrics = fm.load_metrics("Arial", True)
        if metrics.is_builtin:
            self.skipTest("no Arial-compatible font installed; builtin table in use")
        for code in range(0x20, 0x7F):
            ch = chr(code)
            self.assertAlmostEqual(metrics.advance_units(ch) * 1000 / metrics.units_per_em,
                                   BUILTIN.advance_units(ch), delta=1.5, msg=ch)
        for ch in "çğıİöşüÇĞÖŞÜ":
            self.assertGreater(metrics.advance_units(ch), 0)

    def test_unreadable_font_and_missing_resolver_fall_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            junk = Path(tmp) / "junk.ttf"
            junk.write_bytes(b"not a font at all" * 10)
            with self.assertRaises(fm.FontParseError):
                fm.parse_font(junk)
            fm.load_metrics.cache_clear()
            with mock.patch.object(fm, "_fc_match", return_value=(junk, 0)), \
                    mock.patch.object(fm, "_windows_font", return_value=None):
                self.assertTrue(fm.load_metrics("Arial", True).is_builtin)
            fm.load_metrics.cache_clear()
            with mock.patch.object(fm, "_fc_match", return_value=None), \
                    mock.patch.object(fm, "_windows_font", return_value=None):
                self.assertTrue(fm.load_metrics("Arial", True).is_builtin)
            fm.load_metrics.cache_clear()


class StyleAndEmphasisTests(unittest.TestCase):
    ROWS = words("this is the moment everything changed forever. nobody expected that result at all")

    def test_style_resolution_is_declarative_and_sparse(self) -> None:
        impact = caption_event("pay", 0.0, 9.0, "impact")
        presentation = build(self.ROWS, the_plan=plan(impact))
        impact_pages = [p for p in presentation.pages if p.style is CaptionStyle.IMPACT]
        self.assertTrue(impact_pages)
        self.assertLessEqual(len(impact_pages), cp.STYLE_DEFINITIONS[CaptionStyle.IMPACT].max_pages_per_directive)
        self.assertLessEqual(len(impact_pages), max(1, int(cp.MAX_IMPACT_PAGE_RATIO * len(presentation.pages))))
        geometry = presentation.geometry
        for page in impact_pages:
            self.assertAlmostEqual(page.font_px, round(geometry.font_px * 1.08 * page.fit_scale, 1))
        self.assertIn(f"\\fs{cp._num(impact_pages[0].font_px)}", presentation.ass_text)
        plain = build(self.ROWS)
        self.assertTrue(all(p.style is CaptionStyle.DEFAULT for p in plain.pages))
        self.assertNotIn("\\fs", plain.ass_text)

    def test_emphasis_is_validated_and_degrades_without_dropping_words(self) -> None:
        tokens = cp.presentation_tokens(profile(self.ROWS), 99.0)
        target = tokens[3].word_id                              # "moment"
        far = tokens[-1].word_id                                # outside the event window
        event = caption_event("e1", tokens[2].start, tokens[5].end, "emphasis", [target, far, 999])
        presentation = build(self.ROWS, the_plan=plan(event))
        reasons = {d["word_id"]: d["reason"] for d in presentation.emphasis.degraded}
        self.assertEqual(reasons[999], "unknown_word_id")
        self.assertEqual(reasons[far], "outside_event_window")
        self.assertIn(target, presentation.emphasis.applied)
        self.assertEqual(sorted(t.word_id for p in presentation.pages for t in p.tokens),
                         sorted(t.word_id for t in tokens))    # nothing dropped
        page = next(p for p in presentation.pages if target in p.emphasis_ids)
        self.assertEqual(page.style, CaptionStyle.EMPHASIS)
        self.assertIn("\\fscx108", presentation.ass_text)      # static, reserved emphasis scale

    def test_emphasis_budget_per_page_and_clip(self) -> None:
        tokens = cp.presentation_tokens(profile(self.ROWS), 99.0)
        greedy = caption_event("e1", 0.0, 99.0, "default", [t.word_id for t in tokens[:6]])
        presentation = build(self.ROWS, the_plan=plan(greedy))
        for page in presentation.pages:
            cap = 1 if len(page.tokens) <= 4 else 2
            self.assertLessEqual(len(page.emphasis_ids), cap)
        self.assertLessEqual(len(presentation.emphasis.applied), max(1, int(0.15 * len(tokens))))
        self.assertTrue(any(d["reason"] in ("page_emphasis_budget", "clip_emphasis_budget")
                            for d in presentation.emphasis.degraded))

    def test_terra_markers_match_in_paced_time_after_cuts(self) -> None:
        tokens = cp.presentation_tokens(profile(self.ROWS), 99.0)
        target = tokens[4]                                     # "everything"
        marker = {"type": "caption_emphasis", "word": "Everything", "source_time": target.start + 3.0,
                  "edited_time": target.start + 0.12}          # raw clip time is 3 s later (a cut before)
        presentation = build(self.ROWS, events=[marker])
        self.assertEqual(presentation.emphasis.terra_matched, 1)
        self.assertIn(target.word_id, presentation.emphasis.applied)
        # The baseline matcher compares RAW source_time with the final clock and misses it.
        baseline_word = {"normalized": target.normalized, "source_start": target.start}
        markers = captions.get_emphasis_markers({"events": [marker]})
        self.assertFalse(captions.word_is_emphasized(baseline_word, markers))

    def test_caption_channel_counts_as_a_channel_and_is_marked_executed(self) -> None:
        event = caption_event("pay", 0.0, 4.0, "impact")
        self.assertTrue(_has_other_channels(event))
        presentation = build(self.ROWS, the_plan=plan(event))
        rows = cp.mark_caption_channel([{"channel": "caption_style", "event_id": "pay", "executed": False},
                                        {"channel": "sfx", "event_id": "pay", "executed": False}], presentation)
        self.assertTrue(rows[0]["executed"])
        self.assertEqual(rows[0]["executor"], "caption_presentation")
        self.assertFalse(rows[1]["executed"])
        self.assertFalse(cp.mark_caption_channel([dict(rows[0])], None)[0]["executed"])

    def test_config_switch(self) -> None:
        self.assertTrue(ProEditConfig().captions)
        self.assertFalse(load_config(True, {"MIMIR_PRO_EDIT_CAPTIONS": "0"}).captions)
        self.assertTrue(load_config(True, {}).captions)
        self.assertIn("captions", ProEditConfig().signature_payload())


class AssTextTests(unittest.TestCase):
    def test_escaping_unicode_and_control_characters(self) -> None:
        rows = words("İstanbul'da {evil} back\\slash şöyle çığlık new\nline tab\tend ok\u2028sep")
        presentation = build(rows)
        dialogues = [line for line in presentation.ass_text.split("\n") if line.startswith("Dialogue:")]
        self.assertTrue(dialogues)
        joined = "\n".join(dialogues)
        self.assertIn("İstanbul'da", joined)
        self.assertIn("şöyle", joined)
        self.assertIn("\\{evil\\}", joined)
        self.assertNotIn("\t", joined)
        self.assertNotIn("\u2028", joined)
        self.assertNotIn("\r", presentation.ass_text)
        text_fields = [line.split(",", 9)[9] for line in dialogues]
        for text in text_fields:
            stripped = re.sub(r"(?<!\\)\{[^}]*\}", "", text)
            self.assertNotIn("{evil}", stripped)
        with tempfile.TemporaryDirectory() as tmp:
            path = cp.write_presentation(presentation, Path(tmp) / "t.ass")
            self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))       # UTF-8 BOM like the baseline

    def test_manifest_is_complete_and_secret_free(self) -> None:
        event = caption_event("pay", 0.0, 3.0, "impact")
        presentation = build(words(SENTENCE), the_plan=plan(event))
        manifest = presentation.manifest(caption_signature="abc")
        for key in ("version", "style_pack", "layout_profile", "play_res", "font", "truth", "emphasis",
                    "directives", "metrics", "pages", "ass_signature"):
            self.assertIn(key, manifest)
        self.assertEqual(manifest["truth"]["retimed_words"], 0)
        self.assertEqual(manifest["directives"][0]["event_id"], "pay")
        text = repr(manifest).lower()
        for secret in ("api_key", "authorization", "bearer", "sk-"):
            self.assertNotIn(secret, text)


class SafeRegionAndIntroTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_safe_region_follows_the_actual_positioned_events(self) -> None:
        presentation = build(words("first page words. second page here"))
        region = caption_safe_region(cp.write_presentation(presentation, self.dir / "p.ass"))
        self.assertEqual(region.source, "ass_events")
        first, second = presentation.pages[:2]
        top = region.top(first.start, first.start + 0.05)
        height = presentation.geometry.font_px * 1.22 + 2 * (presentation.geometry.outline_px
                                                           + presentation.geometry.shadow_px)
        expected = (first.lines[0].y - height + presentation.geometry.outline_px
                    + presentation.geometry.shadow_px) / 1920 - 0.012
        self.assertAlmostEqual(top, expected, places=3)
        gap_start, gap_end = first.end + 0.01, second.start - 0.01
        if gap_end > gap_start:
            self.assertIsNone(region.top(gap_start, gap_end))

    def test_safe_region_still_parses_the_baseline_ass(self) -> None:
        data = profile(words("baseline captions stay parsable"))
        ass = captions.create_ass_for_clip({}, timeline(), self.dir / "b.ass", data)
        region = caption_safe_region(ass)
        self.assertFalse(region.empty)
        band = region.bands[0]
        bottom = (captions.PLAY_RES_Y - captions.CAPTION_MARGIN_V) / captions.PLAY_RES_Y
        self.assertAlmostEqual(band.y1, bottom + 0.012, places=4)
        self.assertLess(band.y0, bottom - captions.FONT_SIZE * 1.08 / captions.PLAY_RES_Y)   # \fscy108 counted

    def test_top_aligned_and_margin_override_events(self) -> None:
        path = self.dir / "custom.ass"
        path.write_text(
            "[Script Info]\nPlayResX: 1000\nPlayResY: 1000\n\n[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
            "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: S,Arial,50,&H00FFFFFF,&H00FFFFFF,&H00000000,&HFF000000,-1,0,0,0,100,100,0,0,1,0,0,2,10,10,100,1\n"
            "\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Dialogue: 0,0:00:00.00,0:00:01.00,S,,0,0,0,,{\\an8}top\n"
            "Dialogue: 0,0:00:02.00,0:00:03.00,S,,0,0,300,,bottom override\n", encoding="utf-8")
        region = caption_safe_region(path)
        self.assertAlmostEqual(region.top(0.1, 0.2), 0.1 - 0.012, places=4)
        self.assertAlmostEqual(region.top(2.1, 2.2), (1000 - 300 - 61) / 1000 - 0.012, places=4)
        self.assertIsNone(region.top(1.2, 1.8))

    def test_intro_hook_band_uses_intro_renderer_layout(self) -> None:
        intro = IntroTimeline(teaser_start=3.2, teaser_end=4.6, transition=0.16, main_restart=0.0,
                              first_caption=0.3, main_duration=10.0, teaser_record_signature="x")
        record = {"intro_text": "wait for it", "intro": {"duration": 1.0}}
        band = hook_text_band(record, intro, width=1080, height=1920)
        line_1, line_2 = intro_renderer.split_balanced_two_lines("WAIT FOR IT")
        self.assertEqual(band.lines, tuple(x for x in (line_1, line_2) if x))
        self.assertLess(band.y0, intro_renderer.LINE_1_Y_RATIO)
        self.assertGreater(band.y1, intro_renderer.LINE_2_Y_RATIO if line_2 else intro_renderer.SINGLE_LINE_Y_RATIO)
        self.assertGreaterEqual(band.start, intro.teaser_start)
        self.assertLessEqual(band.end, intro.teaser_end)
        self.assertIsNone(hook_text_band({"intro_text": "  "}, intro, width=1080, height=1920))
        self.assertIsNone(hook_text_band(None, intro, width=1080, height=1920))

    def test_framing_moves_face_out_of_text_band_or_reports_it(self) -> None:
        face = SubjectBox(Box(0.45, 0.20, 0.55, 0.28), "face")

        def face_in_crop(framing):
            size = 1.0 / framing.zoom
            return (face.box.y0 - framing.origin[1]) / size, (face.box.y1 - framing.origin[1]) / size

        free = solve_framing(FramingRequest(desired_zoom=1.25, camera=CameraMode.SPEAKER_CLOSE, max_zoom=1.3,
                                            subjects=(face,)))
        top, bottom = face_in_crop(free)
        self.assertTrue(top < 0.50 and bottom > 0.30)               # would sit under the hook text
        framing = solve_framing(FramingRequest(desired_zoom=1.25, camera=CameraMode.SPEAKER_CLOSE, max_zoom=1.3,
                                               subjects=(face,), avoid_bands=((0.30, 0.50),)))
        self.assertTrue(framing.text_clear)
        self.assertEqual(framing.zoom, free.zoom)                   # soft rule: zoom untouched
        top, bottom = face_in_crop(framing)
        self.assertTrue(bottom <= 0.30 or top >= 0.50, (top, bottom))
        huge = SubjectBox(Box(0.3, 0.1, 0.7, 0.9), "face")
        blocked = solve_framing(FramingRequest(desired_zoom=1.1, camera=CameraMode.SPEAKER_MEDIUM, max_zoom=1.2,
                                               subjects=(huge,), avoid_bands=((0.3, 0.5),)))
        self.assertIn(blocked.text_clear, (False, None))

    def test_presentation_starts_where_the_baseline_intro_handoff_expects(self) -> None:
        data = profile(words(SENTENCE, start=1.7))
        baseline = captions.create_ass_for_clip({}, timeline(), self.dir / "base.ass", data)
        presentation = cp.build_presentation(profile=data, clip_timeline=timeline(), plan=None, width=1080,
                                             height=1920, metrics=BUILTIN)
        ours = cp.write_presentation(presentation, self.dir / "pres.ass")
        cp.check_intro_handoff(ours, baseline)
        restart_a = intro_renderer.calculate_main_restart_seconds(caption_path=baseline, main_duration=20.0)
        restart_b = intro_renderer.calculate_main_restart_seconds(caption_path=ours, main_duration=20.0)
        self.assertEqual(restart_a, restart_b)
        broken = self.dir / "early.ass"
        broken.write_text(ours.read_text(encoding="utf-8-sig").replace("Dialogue: 0,0:00:01.70",
                                                                       "Dialogue: 0,0:00:01.20", 1),
                          encoding="utf-8-sig")
        with self.assertRaises(cp.CaptionPresentationError):
            cp.check_intro_handoff(broken, baseline)

    def test_integrity_signature_unaffected_by_presentation(self) -> None:
        data = profile(words(SENTENCE))
        profile_path = self.dir / "profile.json"
        profile_path.write_text(__import__("json").dumps(data), encoding="utf-8")
        baseline = captions.create_ass_for_clip({}, timeline(), self.dir / "base.ass", data)
        before = CaptionIntegrity.capture(profile_path, baseline)
        presentation = cp.build_presentation(profile=data, clip_timeline=timeline(), plan=None, width=1080,
                                             height=1920, metrics=BUILTIN)
        cp.write_presentation(presentation, self.dir / "pres.ass")
        before.verify(profile_path, baseline, stage="presentation")


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
class LibassRenderTests(unittest.TestCase):
    @fx.needs_opencv
    def test_rendered_pages_stay_inside_the_frame_and_do_not_move(self) -> None:
        import cv2
        import numpy as np

        rows = words("Muhteşem bir gün geçirdik arkadaşlar gerçekten harika WWWWWWW MMMMMMM", gap=0.08)
        presentation = cp.build_presentation(profile=profile(rows), clip_timeline=timeline(), plan=None,
                                             width=540, height=960)          # real font metrics
        with tempfile.TemporaryDirectory() as tmp:
            ass = cp.write_presentation(presentation, Path(tmp) / "p.ass")
            page = presentation.pages[0]
            first_word_end = page.tokens[1].start if len(page.tokens) > 1 else page.end
            t1 = page.start + 0.02
            t2 = min(page.end, first_word_end + 0.2) - 0.02
            frames = []
            for i, t in enumerate((t1, t2)):
                out = Path(tmp) / f"f{i}.png"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                                f"color=black:s=540x960:d={page.end + 0.5}:r=50", "-vf",
                                f"subtitles=filename='{caption_renderer.escape_filter_path(ass)}'", "-ss",
                                f"{t:.3f}", "-frames:v", "1",
                                str(out)], check=True, capture_output=True)
                frames.append(cv2.imread(str(out), cv2.IMREAD_GRAYSCALE))
            for frame in frames:
                cols = np.where(frame.max(axis=0) > 40)[0]
                self.assertTrue(len(cols))
                self.assertGreater(cols.min(), 2)                 # no clipping at the frame edges
                self.assertLess(cols.max(), 540 - 3)
            # The first word's ink stays in place when the next word appears (no reflow).
            ink_1 = set(np.where((frames[0] > 40).any(axis=0))[0].tolist())
            ink_2 = set(np.where((frames[1] > 40).any(axis=0))[0].tolist())
            self.assertEqual(min(ink_1), min(ink_2))
            self.assertTrue(ink_1 <= ink_2)
            self.assertGreater(len(ink_2), len(ink_1))
            for page in presentation.pages:
                for line in page.lines:
                    self.assertLessEqual(line.width_px, presentation.geometry.max_line_px)


if __name__ == "__main__":
    unittest.main()


# ============================================================
# Caption presentation v2
# ============================================================

def libass_frame(ass: Path, width: int, height: int, t: float, out: Path):
    import cv2

    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=black:s={width}x{height}:d={t + 1:.2f}:r=50",
                    # Production escaping: a bare Windows drive colon would split the filter options.
                    "-vf", f"subtitles=filename='{caption_renderer.escape_filter_path(ass)}'", "-ss", f"{t:.3f}",
                    "-frames:v", "1", str(out)],
                   check=True, capture_output=True)
    return cv2.imread(str(out))


class StyleSystemTests(unittest.TestCase):
    def test_definitions_are_declarative_and_enforce_the_geometry_law(self) -> None:
        for style, definition in cp.STYLE_DEFINITIONS.items():
            self.assertEqual(definition.active_font_scale, 1.0, style)
            self.assertLessEqual(definition.animation_attack_ms, 150)
        with self.assertRaises(ValueError):
            cp.CaptionStyleDefinition(CaptionStyle.DEFAULT, active_font_scale=1.1)
        with self.assertRaises(ValueError):
            cp.CaptionStyleDefinition(CaptionStyle.DEFAULT, animation_release_ms=900)
        with self.assertRaises(ValueError):
            cp.CaptionStyleDefinition(CaptionStyle.DEFAULT, emphasis_color="#ff00ff")
        ranks = [cp.STYLE_DEFINITIONS[s].rank for s in (CaptionStyle.DEFAULT, CaptionStyle.EMPHASIS,
                                                         CaptionStyle.IMPACT)]
        self.assertEqual(ranks, sorted(ranks))

    def test_resolver_turns_intent_into_physical_values(self) -> None:
        geometry = cp.LayoutGeometry.for_output(1080, 1920)
        default = cp.resolve_caption_style(CaptionStyle.DEFAULT, "main", geometry)
        impact = cp.resolve_caption_style(CaptionStyle.IMPACT, "secondary", geometry)
        self.assertEqual(default.font_px, 74.0)
        self.assertEqual(default.active_outline_px, default.outline_px)          # colour-only highlight
        self.assertAlmostEqual(impact.font_px, round(74 * 1.08, 1))
        self.assertGreater(impact.emphasis_outline_px, impact.outline_px)
        self.assertEqual(impact.ass_style, "MimirSecondary")
        self.assertEqual(impact.active_colour, cp.PALETTES["secondary"].active)
        tertiary = cp.resolve_caption_style(CaptionStyle.DEFAULT, "tertiary", geometry)
        self.assertEqual(tertiary.active_colour, cp.PALETTES["main"].active)     # baseline palette parity

    def test_accent_is_outline_and_colour_only_and_bounded_by_the_word_interval(self) -> None:
        rows = words("this is the payoff moment right here", gap=0.05)
        tokens = cp.presentation_tokens(profile(rows), 99.0)
        event = caption_event("pay", 0.0, 9.0, "impact", [tokens[3].word_id])
        presentation = build(rows, the_plan=plan(event))
        page = next(p for p in presentation.pages if tokens[3].word_id in p.emphasis_ids)
        for a, b, _layer, _style, text in cp.page_events(page, presentation.geometry):
            for t0, t1 in re.findall(r"\\t\((\d+),(\d+),", text):
                self.assertLessEqual(int(t1), (b - a) * 10)                  # never past the next onset
            for block in re.findall(r"\\t\([^)]*\)", text):
                self.assertNotRegex(block, r"\\fsc|\\fs\d|\\pos|\\move")   # no geometry animation
        self.assertRegex(presentation.ass_text, r"\\t\(0,\d+,\\c&H[0-9A-F]{6}&\\bord[\d.]+\)")
        plain = build(rows)
        self.assertNotRegex(plain.ass_text, r"\\t\([^)]*\\bord")            # DEFAULT: colour only


class UncertaintyAndSpeakerTests(unittest.TestCase):
    def test_uncertainty_mask_is_shown_but_never_emphasized(self) -> None:
        rows = words("he said ??? right there okay")
        rows[4]["masked_unknown"] = True                                  # flagged truth word
        tokens = cp.presentation_tokens(profile(rows), 99.0)
        self.assertTrue(tokens[2].uncertain and tokens[4].uncertain and not tokens[1].uncertain)
        event = caption_event("e1", 0.0, 9.0, "emphasis", [tokens[2].word_id, tokens[4].word_id])
        marker = {"type": "caption_emphasis", "word": "okay", "edited_time": tokens[5].start}
        presentation = build(rows, the_plan=plan(event), events=[marker])
        reasons = {d["word_id"]: d["reason"] for d in presentation.emphasis.degraded}
        self.assertEqual(reasons[tokens[2].word_id], "uncertainty_mask")
        self.assertEqual(reasons[tokens[4].word_id], "uncertainty_mask")
        self.assertNotIn(tokens[2].word_id, presentation.emphasis.applied)
        self.assertIn("???", presentation.ass_text)                        # still displayed
        self.assertEqual(presentation.metrics()["uncertain_words"], 2)
        self.assertEqual(presentation.metrics()["invalid_emphasis_ignored"], 2)

    def test_unresolved_speaker_stays_unlabeled_in_the_main_lane(self) -> None:
        rows = words("nobody knows who said this line")
        for row in rows:
            row.update(speaker_raw="", speaker_role="main", speaker_label="")
        presentation = build(rows)
        self.assertTrue(all(p.lane == "main" and not p.label for p in presentation.pages))
        self.assertNotIn(":", "".join(re.sub(r"\{[^}]*\}", "", line.split(",", 9)[9])
                                      for line in presentation.ass_text.splitlines() if line.startswith("Dialogue")))

    def test_third_speaker_never_gets_a_third_lane(self) -> None:
        # Captions V24 (current root) keeps at most two defensible lanes: a third / unstable
        # cluster loses permission to create a caption lane and an unconfirmed name is never
        # printed. (The V16-era tertiary lane of Pro Edit V3.1 is therefore never used.)
        rows = words("main person keeps talking over everyone here")
        extra = [{"word": w, "edited_start": 0.5 + i * 0.25, "edited_end": 0.7 + i * 0.25, "speaker_raw": "C",
                  "speaker_role": "tertiary", "speaker_label": "SAM", "speaker_confidence": 0.8}
                 for i, w in enumerate("wait what".split())]
        presentation = build(rows + extra)
        self.assertEqual({p.lane for p in presentation.pages}, {"main"})
        self.assertFalse(any(p.label for p in presentation.pages))
        self.assertEqual(presentation.metrics()["same_lane_overlaps"], 0)
        dialogue = [line for line in presentation.ass_text.splitlines() if line.startswith("Dialogue:")]
        self.assertFalse(any(",MimirTertiary," in line or "SAM:" in line for line in dialogue))
        shown = {t.word_id for p in presentation.pages for t in p.tokens}
        self.assertEqual(shown, {t.word_id for t in presentation.tokens})          # every word still shown
        with tempfile.TemporaryDirectory() as tmp:
            region = caption_safe_region(cp.write_presentation(presentation, Path(tmp) / "t.ass"))
        self.assertEqual({b.style for b in region.bands}, {"MimirMain"})

    def test_other_lane_never_cuts_the_terminal_hold(self) -> None:
        rows = words("okay so", gap=0.05)
        reply = {"word": "yes", "edited_start": rows[-1]["edited_start"] + 0.05,
                 "edited_end": rows[-1]["edited_start"] + 0.3, "speaker_raw": "B", "speaker_role": "secondary",
                 "speaker_label": "", "speaker_confidence": 0.9}
        presentation = build(rows + [reply])
        main = next(p for p in presentation.pages if p.lane == "main")
        self.assertGreaterEqual(main.end - main.tokens[-1].start, captions.MIN_TERMINAL_DISPLAY_DURATION - 1e-6)


class TerminalReadabilityTests(unittest.TestCase):
    def test_page_breaks_avoid_terminal_flashes_when_a_soft_break_allows(self) -> None:
        texts = "one two three four five six seven eight nine ten".split()
        rows, t = [], 0.3
        for index, word in enumerate(texts):
            rapid = word == "four"                       # "five" starts 0.1 s after "four"
            rows.append({"word": word, "edited_start": round(t, 3), "edited_end": round(t + (0.08 if rapid else 0.25), 3),
                         "speaker_raw": "A", "speaker_role": "main", "speaker_label": ""})
            t += 0.1 if rapid else 0.3
        presentation = build(rows)
        ends_before_five = [p for p in presentation.pages if p.tokens[-1].text == "four"]
        self.assertEqual(ends_before_five, [])           # never cut right before the rapid onset
        self.assertEqual(presentation.metrics()["terminal_flash_pages"], 0)
        # A hard break (sentence end) with an immediate next sentence cannot be avoided: reported.
        hard = words("done. next thing", gap=0.02, sentence_gap=0.02)
        for row in hard:
            row["edited_end"] = round(row["edited_start"] + 0.06, 3)
        hard[1]["edited_start"] = round(hard[0]["edited_start"] + 0.08, 3)
        hard[1]["edited_end"] = round(hard[1]["edited_start"] + 0.2, 3)
        hard[2]["edited_start"] = round(hard[1]["edited_end"] + 0.02, 3)
        hard[2]["edited_end"] = round(hard[2]["edited_start"] + 0.2, 3)
        self.assertGreaterEqual(build(hard).metrics()["terminal_flash_pages"], 1)


class EscapingAndGlyphTests(unittest.TestCase):
    def test_backslash_can_never_form_an_ass_escape(self) -> None:
        text = cp.display_text("C:\\New")
        self.assertEqual(text, "C:\\\u2060New")
        self.assertNotIn("\\N", text)
        self.assertEqual(cp.display_text("a{b}c"), "a\\{b\\}c")
        self.assertEqual(cp.display_text("x\\{y"), "x\\\u2060\\{y")

    def test_symbols_quotes_numbers_survive(self) -> None:
        phrase = "it’s “quoted” well-known 100% #1 $5 & <tag> O'Brien Çağrı-İnce 3.5x"
        presentation = build(words(phrase))
        visible = "".join(re.sub(r"(?<!\\)\{[^}]*\}", "", line.split(",", 9)[9])
                          for line in presentation.ass_text.splitlines() if line.startswith("Dialogue"))
        for token in phrase.split():
            self.assertIn(token, visible)
        measure = cp._Measure(BUILTIN, 74)
        self.assertGreater(measure.word("100%"), measure.word("100"))

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
    @fx.needs_opencv
    def test_libass_renders_escaped_text_on_one_line(self) -> None:
        import numpy as np

        rows = words("C:\\New {x} done")
        presentation = cp.build_presentation(profile=profile(rows), clip_timeline=timeline(), plan=None,
                                             width=640, height=640)
        with tempfile.TemporaryDirectory() as tmp:
            ass = cp.write_presentation(presentation, Path(tmp) / "e.ass")
            page = presentation.pages[-1]
            frame = libass_frame(ass, 640, 640, page.end - 0.05, Path(tmp) / "e.png")
        ink_rows = np.where((frame.max(axis=2) > 60).any(axis=1))[0]
        line_height = BUILTIN.line_height(presentation.geometry.font_px)
        self.assertLess(ink_rows.max() - ink_rows.min(), 1.3 * line_height)      # one line, no \N break

    def test_font_substitution_is_reported(self) -> None:
        metrics = fm.load_metrics("Arial", True)
        report = metrics.to_dict()
        self.assertIn("resolved_family", report)
        if not metrics.is_builtin:
            self.assertTrue(report["resolved_family"])
            self.assertEqual(report["substituted"], report["resolved_family"].casefold() != "arial")


class DiagnosticsAndCacheTests(unittest.TestCase):
    def test_manifest_records_what_was_rendered(self) -> None:
        rows = words(SENTENCE, secondary=range(10, 16), label="TYLA")
        tokens = cp.presentation_tokens(profile(rows), 99.0)
        presentation = build(rows, the_plan=plan(caption_event("pay", 0.0, 3.0, "impact", [tokens[2].word_id])))
        manifest = presentation.manifest(caption_signature="sig")
        for key in ("layout_profile", "safe_region", "styles", "safe_width_px", "lane_bottoms"):
            self.assertIn(key, manifest)
        page = manifest["pages"][0]
        for key in ("page_id", "word_ids", "start", "end", "lines", "style", "emphasized_word_ids",
                    "speaker_role", "safe_region"):
            self.assertIn(key, page)
        metrics = manifest["metrics"]
        for key in ("avg_words_per_page", "wps_p50", "max_simultaneous_words", "speaker_pages",
                    "terminal_flash_pages", "invalid_emphasis_ignored", "fit_scaled_pages", "safe_band"):
            self.assertIn(key, metrics)
        # The manifest safe region agrees with the band caption_guard parses from the burned ASS.
        with tempfile.TemporaryDirectory() as tmp:
            region = caption_safe_region(cp.write_presentation(presentation, Path(tmp) / "m.ass"))
        guard_top = min(b.y0 for b in region.bands)
        guard_bottom = max(b.y1 for b in region.bands)
        self.assertLessEqual(guard_top, manifest["safe_region"]["band_y"][0] + 1e-6)
        self.assertGreaterEqual(guard_bottom, manifest["safe_region"]["band_y"][1] - 1e-6)

    def test_presentation_changes_never_rebill_the_planner(self) -> None:
        from ai.editor import pro_edit

        self.assertIn(cp, pro_edit.MODULES)
        self.assertNotIn(cp, pro_edit.PLAN_MODULES)
        self.assertNotIn(fm, pro_edit.PLAN_MODULES)
        config = ProEditConfig(captions=True)
        off = ProEditConfig(captions=False)
        self.assertEqual(config.plan_signature_payload(), off.plan_signature_payload())
        self.assertNotEqual(config.signature_payload(), off.signature_payload())   # fast-resume still notices


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
class LibassStabilityTests(unittest.TestCase):
    def fill_columns(self, frame, rows: slice):
        import numpy as np

        white = frame.min(axis=2) > 200
        return set(np.where(white[rows].any(axis=0))[0].tolist())

    @fx.needs_opencv
    def test_outline_accent_does_not_move_words(self) -> None:
        rows = words("this is the payoff moment", gap=0.25)
        tokens = cp.presentation_tokens(profile(rows), 99.0)
        presentation = cp.build_presentation(
            profile=profile(rows), clip_timeline=timeline(),
            plan=plan(caption_event("pay", 0.0, 9.0, "impact", [tokens[3].word_id])), width=720, height=1280)
        page = next(p for p in presentation.pages if tokens[3].word_id in p.emphasis_ids)
        line = page.lines[0]
        band = slice(int(line.y - page.font_px * 1.3), int(line.y + 10))
        with tempfile.TemporaryDirectory() as tmp:
            ass = cp.write_presentation(presentation, Path(tmp) / "a.ass")
            accent = libass_frame(ass, 720, 1280, tokens[3].start + 0.05, Path(tmp) / "1.png")    # mid attack
            settled = libass_frame(ass, 720, 1280, tokens[4].start + 0.25, Path(tmp) / "2.png")
        first_word = self.fill_columns(accent, band)
        # Spoken words keep exactly the same glyph fill columns while the accent plays.
        spoken = {c for c in first_word if c < min(self.fill_columns(settled, band) - first_word, default=10_000)}
        self.assertTrue(spoken)
        self.assertTrue(spoken <= self.fill_columns(settled, band))

    @fx.needs_opencv
    def test_two_line_page_line_one_is_stable_when_line_two_appears(self) -> None:
        rows = words("Anlayamıyorum konuşmalarından sıkıldığımızı", gap=0.2)
        presentation = cp.build_presentation(profile=profile(rows), clip_timeline=timeline(), plan=None,
                                             width=720, height=1280)
        page = next(p for p in presentation.pages if len(p.lines) == 2)
        top, bottom = page.lines
        top_band = slice(int(top.y - page.font_px * 1.2), int(top.y + 4))
        order = {t.word_id: t for t in page.tokens}
        before = order[bottom.word_ids[0]].start - 0.05
        after = order[bottom.word_ids[0]].start + 0.1
        with tempfile.TemporaryDirectory() as tmp:
            ass = cp.write_presentation(presentation, Path(tmp) / "b.ass")
            f1 = libass_frame(ass, 720, 1280, before, Path(tmp) / "1.png")
            f2 = libass_frame(ass, 720, 1280, after, Path(tmp) / "2.png")
        import numpy as np

        # Line 1 never reflows. The active word is highlight-coloured in f1 and inactive in f2,
        # so anti-aliased edge pixels differ in intensity with the fill colour (measured on
        # Windows / Arial Bold: one inter-glyph column 64 vs 46 at the 60 threshold). Compare
        # geometry colour-independently: identical strong-ink columns, identical extent, and
        # the best rigid horizontal alignment of the two masks is a zero shift.
        ink = lambda f, level: np.where((f.max(axis=2) > level)[top_band].any(axis=0))[0]

        # Strong ink per column, normalized by the frame's own fill brightness (the highlight
        # and inactive fills differ: 255 vs 242), with hysteresis: a column clearly inked in
        # one frame must be at least edge-inked in the other. A reflow of >= 1 px moves a
        # glyph stem fully off a column (1.0 -> ~0) and trips this; an anti-aliased edge whose
        # blend depends on the fill colour (measured on Linux / libass: 155 vs 165) does not.
        def normalized(frame):
            column = frame.max(axis=2)[top_band].max(axis=0).astype(float)
            return column / max(1.0, float(column.max()))

        def reflowed(a, b, high=0.70, low=0.55):
            return [c for c in range(len(a)) if (a[c] >= high and b[c] < low) or (b[c] >= high and a[c] < low)]

        n1, n2 = normalized(f1), normalized(f2)
        self.assertEqual(reflowed(n1, n2), [])
        self.assertTrue(reflowed(n1, np.roll(n2, 1)), "the ink check must detect a 1 px reflow")
        self.assertEqual((ink(f1, 60).min(), ink(f1, 60).max()), (ink(f2, 60).min(), ink(f2, 60).max()))
        m1 = (f1.max(axis=2) > 60)[top_band].astype(int)
        m2 = (f2.max(axis=2) > 60)[top_band].astype(int)
        scores = {shift: int((np.roll(m1, shift, axis=1) == m2).sum()) for shift in range(-3, 4)}
        self.assertEqual(max(scores, key=scores.get), 0, scores)
