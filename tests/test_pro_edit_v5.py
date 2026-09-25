"""Pro Edit V5: machine-verifiable checks (keyless; FFmpeg/libass where noted).

Written for the V7.1-based V5 build on Linux; in the current-root rebase this
suite is executed natively on Windows (FFmpeg/libass burns from stress paths
with spaces, Turkish characters, apostrophes and [ , ; ]; static AST hygiene
checks over ai/**). Pixel-evidence tests need the optional OpenCV/numpy and
are skipped (reported) without them.
"""
from __future__ import annotations

import ast
import dataclasses
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path, PurePosixPath, PureWindowsPath
from unittest import mock

import pro_edit_fixtures as fx
import test_pro_edit_captions as base
from ai.editor import caption_renderer, intro_renderer
from ai.editor.pro_edit import caption_background as bg
from ai.editor.pro_edit import caption_brand as brand_mod
from ai.editor.pro_edit import caption_layout as layout_mod
from ai.editor.pro_edit import caption_legibility as leg
from ai.editor.pro_edit import caption_platform as platform_mod
from ai.editor.pro_edit import caption_presentation as cp
from ai.editor.pro_edit import caption_primitives as prim
from ai.editor.pro_edit import editorial_energy as energy
from ai.editor.pro_edit import font_metrics as fm
from ai.editor.pro_edit.caption_action import activity_hotspot, derive_action_regions, detect_people_in_spans
from ai.editor.pro_edit.caption_background import RegionStats
from ai.editor.pro_edit.caption_guard import caption_safe_region
from ai.editor.pro_edit.caption_occupancy import VisualOccupancyMap
from ai.editor.pro_edit.caption_placement import PlacementEvidence, TimedBox
from ai.editor.pro_edit.caption_platform import NormBox
from ai.editor.pro_edit.config import ProEditConfig, load_config
from ai.editor.pro_edit.executor import subtitle_filter_for
from ai.editor.pro_edit.schema import CaptionStyle, EmphasisReason, MotionPreset, StoryRole
from test_pro_edit_captions_v4 import HAVE_FFMPEG, build, fx_span, text_bands

REPO = Path(__file__).resolve().parents[1]
PRO_EDIT = REPO / "ai" / "editor" / "pro_edit"
STRESS_DIR = "Test User/MIMIR Test/Türkçe Dosya çğıİöşü"


# ============================================================
# helpers
# ============================================================

def hud_frames(count: int, *, transient_at: tuple[int, ...] = (), shape=(360, 640)):
    """Static HUD text top-left, a moving block, transient text bottom-right in a few samples."""
    import cv2
    import numpy as np

    frames = []
    for index in range(count):
        image = np.zeros((shape[0], shape[1], 3), np.uint8)
        image[:] = np.linspace(40, 140, shape[1], dtype=np.uint8)[None, :, None]
        cv2.putText(image, "SCORE 1250  KILLS 7", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        x = 40 + (index * 23) % 400
        image[180:240, x:x + 60] = (30, 200, 60)
        if index in transient_at:
            cv2.putText(image, "new follower lol", (380, 330), cv2.FONT_HERSHEY_PLAIN, 1.3, (240, 240, 240), 1)
        frames.append((image, index * 0.5))
    return frames


def stats(lum: float, edges: float = 0.02, text: float = 0.0, *, spread: float = 0.05, samples: int = 4,
          confidence: float = 1.0) -> RegionStats:
    return RegionStats(samples, max(0.0, lum - spread), lum, min(1.0, lum + spread), edges, text, confidence)


class StubBackground:
    """region_stats by vertical position: fn(y_center) -> RegionStats."""

    fingerprint = "stub"

    def __init__(self, fn):
        self.fn = fn

    def region_stats(self, start, end, box):
        return self.fn((box[1] + box[3]) / 2.0)


class StubUi:
    """TEXT_LIKE persistence map: persistent cells in [y0, y1]."""

    def __init__(self, y0: float, y1: float, cols: int = 12, rows: int = 20):
        self.cols, self.rows = cols, rows
        self.values = [1.0 if y0 <= (r + 0.5) / rows <= y1 else 0.0 for r in range(rows) for _c in range(cols)]
        self.inner = bg.CaptionBackgroundMap(cols, rows, 2.0, (), 0, "ui")

    def window(self, start, end):
        return self.values

    def region_score(self, box, values):
        return self.inner.region_score(box, values)


def white_brand(**changes) -> brand_mod.BrandProfile:
    return dataclasses.replace(brand_mod.MIMIR_DEFAULT, profile_id="test_brand", **changes)


def dialogue_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("Dialogue")]


def render_frame(ass: Path, width: int, height: int, t: float, out: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=black:s={width}x{height}:d=4:r=25",
                    "-vf", f"subtitles=filename='{caption_renderer.escape_filter_path(ass)}'", "-ss", f"{t:.2f}",
                    "-frames:v", "1", str(out)], check=True, capture_output=True)


# ============================================================
# BACKGROUND ANALYSIS (TEXT_LIKE occupancy, luminance, panels)
# ============================================================

class BackgroundAnalysisTests(unittest.TestCase):
    @fx.needs_opencv
    def test_static_hud_text_is_persistent_occupancy_and_transient_text_is_not(self) -> None:
        found = bg.analyze_frames(hud_frames(12, transient_at=(3, 7)), cols=24, rows=14, sample_fps=2.0,
                                  fingerprint="hud")
        values = found.text_occupancy(0.0, 6.0)
        hud = found.region_score((0.0, 0.03, 0.55, 0.12), values)
        transient = found.region_score((0.58, 0.85, 1.0, 0.95), values)
        middle = found.region_score((0.0, 0.45, 1.0, 0.7), values)
        self.assertGreater(hud, 0.3)
        self.assertEqual(transient, 0.0)                       # 2 of 12 samples: never permanent UI
        self.assertEqual(middle, 0.0)                          # moving block is activity, not text
        boxes = bg.ui_boxes(found, 0.0, 6.0)
        self.assertTrue(any(b[0][1] < 0.15 and b[0][0] < 0.2 for b in boxes))
        self.assertIn("no OCR", found.to_dict()["kind"])

    @fx.needs_opencv
    def test_plain_shapes_noise_and_smooth_texture_are_not_text(self) -> None:
        import cv2
        import numpy as np

        rng = np.random.default_rng(3)
        frames = []
        for index in range(8):
            image = np.full((360, 640, 3), 90, np.uint8)
            cv2.rectangle(image, (50, 50), (250, 120), (250, 250, 250), -1)
            cv2.circle(image, (420, 200), 60, (20, 200, 30), -1)
            texture = cv2.resize(rng.integers(0, 255, (45, 80)).astype(np.float32), (640, 360),
                                 interpolation=cv2.INTER_CUBIC)
            image[250:, :] = np.clip(texture[250:, :, None], 0, 255).astype(np.uint8)
            frames.append((image, index * 0.5))
        found = bg.analyze_frames(frames, cols=24, rows=14, sample_fps=2.0, fingerprint="shapes")
        self.assertLess(max(found.text_occupancy(0.0, 4.0)), 1e-9)

    @fx.needs_opencv
    def test_luminance_statistics_segments_and_panels(self) -> None:
        import numpy as np

        frames = []
        for index in range(8):
            image = np.zeros((180, 320, 3), np.uint8)
            image[:, :160] = 255 if index < 4 else 0              # left half white, then a cut to black
            image[:, 159:161] = 128
            image[:, 238:242] = 255                               # persistent vertical line (panel divider)
            frames.append((image, index * 0.5))
        found = bg.analyze_frames(frames, cols=16, rows=9, sample_fps=2.0, fingerprint="lum")
        white = found.region_stats(0.0, 1.5, (0.05, 0.1, 0.4, 0.9))
        black = found.region_stats(2.0, 3.5, (0.05, 0.1, 0.4, 0.9))
        self.assertGreater(white.lum_p10, 0.9)
        self.assertLess(black.lum_p90, 0.05)
        self.assertEqual(found.cut_samples, 1)
        self.assertEqual({s.segment for s in found.samples}, {0, 1})
        panels_x, _ = found.panel_lines(0.0, 4.0)
        self.assertTrue(any(abs(x - 0.75) < 0.02 for x in panels_x), panels_x)

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
    @fx.needs_opencv
    def test_real_clip_sidecar_in_unicode_path_is_cached_by_fingerprint(self) -> None:
        from ai.editor.pro_edit.media import probe_media

        with tempfile.TemporaryDirectory(prefix="mimir v5 ") as tmp:
            folder = Path(tmp) / STRESS_DIR
            folder.mkdir(parents=True)
            clip = folder / "oyun çekimi 1.mp4"
            frames = hud_frames(6)
            process = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                        "-s", "640x360", "-r", "2", "-i", "-", "-c:v", "libx264", "-pix_fmt",
                                        "yuv420p", "-r", "30", str(clip)], stdin=subprocess.PIPE)
            for image, _t in frames:
                process.stdin.write(image.tobytes())
            process.stdin.close()
            self.assertEqual(process.wait(), 0)
            media = probe_media(clip)
            cache = folder / "arka plan ş.json"
            first, s1 = bg.load_or_analyze(cache, media, "clip-1")
            second, s2 = bg.load_or_analyze(cache, media, "clip-1")
            third, s3 = bg.load_or_analyze(cache, media, "clip-2")
            self.assertFalse(cache.with_name(cache.name + ".tmp").exists())
        self.assertEqual((s1, s2, s3), ("analyzed", "cached", "analyzed"))
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertNotEqual(first.fingerprint, third.fingerprint)
        self.assertGreater(first.region_score((0.0, 0.03, 0.5, 0.12), first.text_occupancy(0.0, 3.0)), 0.2)


# ============================================================
# LEGIBILITY
# ============================================================

WHITE_ON_BLACK = leg.TextAppearance(("&HFFFFFF&",), "&H00000000", 6.0, 74.0, 2.0, "&H000000&", 0x8C)


class LegibilityTests(unittest.TestCase):
    def test_ladder_is_monotone_in_luminance_and_complexity(self) -> None:
        thin = dataclasses.replace(WHITE_ON_BLACK, outline_px=2.0)
        for appearance in (WHITE_ON_BLACK, thin):
            previous = -1
            for lum in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
                verdict = leg.assess(0, stats(lum), appearance).verdict
                self.assertGreaterEqual(leg.rank(verdict), previous, (lum, verdict))
                previous = leg.rank(verdict)
            previous = -1
            for edges in (0.0, 0.08, 0.15, 0.22, 0.3, 0.4, 0.6):
                verdict = leg.assess(0, stats(0.7, edges), appearance).verdict
                self.assertGreaterEqual(leg.rank(verdict), previous, (edges, verdict))
                previous = leg.rank(verdict)
        self.assertEqual(leg.assess(0, stats(0.05), WHITE_ON_BLACK).verdict, leg.Legibility.CLEAR)
        # 2 px on 74 px cannot reach the needed ratio even at x1.5 -> the next rung (shadow).
        self.assertEqual(leg.assess(0, stats(0.9), thin).verdict, leg.Legibility.NEEDS_SHADOW)
        medium = dataclasses.replace(WHITE_ON_BLACK, outline_px=3.7)
        boosted = leg.assess(0, stats(0.9), medium)
        self.assertEqual(boosted.verdict, leg.Legibility.NEEDS_OUTLINE)
        self.assertTrue(1.0 < boosted.outline_boost <= leg.MAX_OUTLINE_BOOST)
        self.assertEqual(leg.assess(0, stats(0.7, 0.25), WHITE_ON_BLACK).verdict, leg.Legibility.NEEDS_SHADOW)
        self.assertEqual(leg.assess(0, stats(0.7, 0.5), WHITE_ON_BLACK).verdict, leg.Legibility.NEEDS_PLATE)

    def test_verdict_uses_the_actual_brand_colours(self) -> None:
        dark_text = leg.TextAppearance(("&H101010&",), "&H00FFFFFF", 2.0, 74.0, 2.0, "&HFFFFFF&", 0x40)
        white_text = dataclasses.replace(WHITE_ON_BLACK, outline_px=2.0)
        bright = stats(0.95)
        self.assertEqual(leg.assess(0, bright, dark_text).verdict, leg.Legibility.CLEAR)
        self.assertNotEqual(leg.assess(0, bright, white_text).verdict, leg.Legibility.CLEAR)
        dark = stats(0.01)
        self.assertEqual(leg.assess(0, dark, white_text).verdict, leg.Legibility.CLEAR)
        self.assertNotEqual(leg.assess(0, dark, dark_text).verdict, leg.Legibility.CLEAR)
        # An outline too close to the fill cannot separate the glyph: no outline-only verdict.
        grey_halo = leg.TextAppearance(("&HFFFFFF&",), "&H00DDDDDD", 6.0, 74.0, 2.0, "&H000000&", 0x40)
        plated = leg.assess(0, stats(0.9), grey_halo)
        self.assertEqual(plated.verdict, leg.Legibility.NEEDS_PLATE)
        # The plate is only as opaque as needed to reach the target contrast (never above the cap).
        opacity = 1 - plated.plate_alpha / 255
        self.assertTrue(0.75 <= opacity <= leg.MAX_PLATE_OPACITY + 0.01, opacity)
        self.assertIsNone(leg.plate_opacity(dataclasses.replace(grey_halo, plate_colour="&HFFFFFF&"), stats(0.9)))

    def test_uncertain_without_evidence_and_plate_fallback(self) -> None:
        self.assertEqual(leg.assess(0, None, WHITE_ON_BLACK).verdict, leg.Legibility.UNCERTAIN)
        self.assertEqual(leg.assess(0, stats(0.5, confidence=0.2), WHITE_ON_BLACK).verdict,
                         leg.Legibility.UNCERTAIN)
        no_plate = leg.assess(0, stats(0.7, 0.5), WHITE_ON_BLACK, allow_plate=False)
        self.assertEqual((no_plate.verdict, no_plate.plate), (leg.Legibility.NEEDS_SHADOW, False))
        self.assertEqual(no_plate.degraded_from, "LEGIBILITY_NEEDS_PLATE")

    def test_minimum_intervention_is_rendered_without_touching_text_or_timing(self) -> None:
        rows = base.words("this bright wall makes white captions hard to read today")
        plain = build(rows)
        busy = StubBackground(lambda y: stats(0.75, 0.25))                   # needs shadow
        treated = build(rows, evidence=PlacementEvidence(), background=busy)
        self.assertEqual([(t.word_id, t.text, t.start, t.end) for t in treated.tokens],
                         [(t.word_id, t.text, t.start, t.end) for t in plain.tokens])
        self.assertEqual([(p.start, p.end, [l.word_ids for l in p.lines]) for p in treated.pages],
                         [(p.start, p.end, [l.word_ids for l in p.lines]) for p in plain.pages])
        self.assertTrue(all(p.legibility == "LEGIBILITY_NEEDS_SHADOW" for p in treated.pages))
        lines = dialogue_lines(treated.ass_text)
        self.assertTrue(all("\\shad" in line and "\\bord" in line for line in lines))
        self.assertEqual({re.search(r"\\pos\([^)]*\)", l).group(0) for l in lines},
                         {re.search(r"\\pos\([^)]*\)", l).group(0) for l in dialogue_lines(plain.ass_text)})
        self.assertTrue(set(treated.manifest()["metrics"]["primitives"]) >= {"legibility_outline_boost",
                                                                              "legibility_shadow"})
        with tempfile.TemporaryDirectory() as tmp:
            region = caption_safe_region(cp.write_presentation(treated, Path(tmp) / "s.ass"))
        page = treated.pages[0]
        band = cp.page_safe_band(page, treated)
        self.assertGreaterEqual(max(b.y1 for b in region.bands), band[1] - 1e-4)   # guard counts \shad
        clear = build(rows, evidence=PlacementEvidence(), background=StubBackground(lambda y: stats(0.02)))
        self.assertEqual(ass_body(clear.ass_text), ass_body(plain.ass_text))       # nothing needed: no change

    def test_plates_are_budgeted_and_brand_policy_is_respected(self) -> None:
        rows = base.words(" ".join(f"word{i}" for i in range(40)), sentence_gap=0.9)
        worst = StubBackground(lambda y: stats(0.8, 0.55))
        built = build(rows, evidence=PlacementEvidence(), background=worst)
        plates = sum(1 for d in built.legibility if d.plate)
        self.assertGreaterEqual(plates, 1)
        self.assertLessEqual(plates, max(1, int(prim.LEGIBILITY_PLATE_MAX_RATIO * len(built.pages))))
        degraded = [d for d in built.legibility if d.degraded_from]
        self.assertEqual(len(degraded) + plates, len(built.pages))
        self.assertTrue(all(d.verdict is leg.Legibility.NEEDS_SHADOW for d in degraded))
        drawn = [l for l in dialogue_lines(built.ass_text) if "\\p1" in l]
        self.assertEqual(len({l.split(",")[1] for l in drawn}), plates)
        no_plate = build(rows, brand=white_brand(legibility="no_plate"), evidence=PlacementEvidence(),
                         background=worst)
        self.assertFalse(any(d.plate for d in no_plate.legibility))
        off = build(rows, brand=white_brand(legibility="off"), evidence=PlacementEvidence(), background=worst)
        self.assertEqual(off.legibility, ())
        with self.assertRaises(brand_mod.BrandProfileError):
            brand_mod.parse_brand({"profile_id": "x", "version": 1, "legibility": "always"})

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
    @fx.needs_opencv
    def test_legibility_pixels_stay_inside_the_guard_band(self) -> None:
        import cv2
        import numpy as np

        rows = base.words("shadow check page here")
        built = cp.build_presentation(profile=base.profile(rows), clip_timeline=base.timeline(), plan=None,
                                      width=540, height=960, evidence=PlacementEvidence(),
                                      background=StubBackground(lambda y: stats(0.8, 0.28)))
        self.assertEqual(built.pages[0].legibility, "LEGIBILITY_NEEDS_SHADOW")
        with tempfile.TemporaryDirectory() as tmp:
            ass = cp.write_presentation(built, Path(tmp) / "l.ass")
            region = caption_safe_region(ass)
            out = Path(tmp) / "l.png"
            render_frame(ass, 540, 960, built.pages[0].end - 0.05, out)
            frame = cv2.imread(str(out))
        lit = np.where((frame.max(axis=2) > 20).any(axis=1))[0]
        self.assertTrue(len(lit))
        self.assertGreaterEqual(lit.min() / 960, min(b.y0 for b in region.bands) - 1e-3)
        self.assertLessEqual(lit.max() / 960, max(b.y1 for b in region.bands) + 1e-3)


def ass_body(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(("Style:", "Dialogue:"))]


# ============================================================
# REASON -> PRIMITIVE GRAMMAR
# ============================================================

def reason_plan(tokens, pairs):
    event = base.caption_event("e", 0.0, 30.0, "emphasis", [tokens[i].word_id for i, _r in pairs])
    return base.plan(dataclasses.replace(event, emphasis_reasons=tuple((tokens[i].word_id, r) for i, r in pairs)))


class ReasonGrammarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = base.words("Kai scored 100 points but lost anyway tonight", gap=0.1)
        self.tokens = cp.presentation_tokens(base.profile(self.rows), 99.0)

    def test_library_has_at_most_twelve_primitives_and_default_brand_is_unchanged(self) -> None:
        self.assertEqual(len(prim.PRIMITIVE_LIBRARY), 12)
        self.assertLessEqual(len(prim.PRIMITIVE_LIBRARY), prim.MAX_LIBRARY_SIZE)
        self.assertTrue(prim.REASON_PRIMITIVES.isdisjoint(brand_mod.MIMIR_DEFAULT.allowed_primitives))
        expressive, notes = brand_mod.load_brand("mimir_expressive")
        self.assertEqual(notes, ())
        self.assertTrue(prim.REASON_PRIMITIVES <= expressive.allowed_primitives)
        self.assertEqual(expressive.lanes, brand_mod.MIMIR_DEFAULT.lanes)

    def test_name_number_contrast_map_to_distinct_static_treatments(self) -> None:
        expressive, _ = brand_mod.load_brand("mimir_expressive")
        planned = reason_plan(self.tokens, [(0, EmphasisReason.NAME), (2, EmphasisReason.NUMBER),
                                            (5, EmphasisReason.CONTRAST)])
        built = build(self.rows, brand=expressive, plan=planned, verified_names=["Kai Cenat"])
        effects = [built.settings.effects_for(p) for p in built.pages]
        underline = set().union(*(e.underline_word_ids for e in effects))
        badges = set().union(*(e.badge_word_ids for e in effects))
        contrast = set().union(*(e.contrast_word_ids for e in effects))
        emphasized = set(built.emphasis.applied)
        self.assertEqual(underline, {self.tokens[0].word_id} & emphasized)
        self.assertEqual(badges, {self.tokens[2].word_id} & emphasized)
        self.assertEqual(contrast, {self.tokens[5].word_id} & emphasized)
        self.assertTrue(underline or badges or contrast)
        text = built.ass_text
        if contrast:
            self.assertIn("\\3c", text)
        # Geometry law: every drawing is static (no \t, no \move) and text positions are the default's.
        default = build(self.rows, plan=planned, verified_names=["Kai Cenat"])
        pos = lambda t: sorted({m for l in dialogue_lines(t) if "\\p1" not in l
                                for m in re.findall(r"\\pos\([^)]*\)", l)})
        self.assertEqual(pos(text), pos(default.ass_text))
        for line in dialogue_lines(text):
            if "\\p1" in line:
                self.assertNotIn("\\t(", line)
                self.assertNotIn("\\move", line)

    def test_underline_stays_inside_the_text_band_and_badges_are_guarded(self) -> None:
        expressive, _ = brand_mod.load_brand("mimir_expressive")
        planned = reason_plan(self.tokens, [(0, EmphasisReason.NAME)])
        built = build(self.rows, brand=expressive, plan=planned)
        page = next(p for p in built.pages if self.tokens[0].word_id in p.emphasis_ids)
        drawings = cp._drawings(page, built.geometry, built.settings)
        self.assertEqual([d.kind for d in drawings], ["underline"])
        x, y, w, h = drawings[0].box
        line = page.lines[0]
        self.assertLessEqual(y + h, line.y + 1)                              # inside the line box
        self.assertGreaterEqual(y, line.y - built.metrics_font.line_height(page.font_px) * 1.2)
        long_rows = base.words("they scored 100 points. " + " ".join(f"then word{i}." for i in range(14)),
                               sentence_gap=0.9)
        long_tokens = cp.presentation_tokens(base.profile(long_rows), 99.0)
        badge_plan = reason_plan(long_tokens, [(2, EmphasisReason.NUMBER)])
        badged = build(long_rows, brand=expressive, plan=badge_plan)
        self.assertGreaterEqual(len(badged.pages), 8)                        # strong budget allows one
        badge_lines = [l for l in dialogue_lines(badged.ass_text) if "\\p1" in l]
        if badged.emphasis.applied:
            self.assertTrue(badge_lines)
            self.assertIn("\\3a&H00&", badge_lines[0])
            with tempfile.TemporaryDirectory() as tmp:
                region = caption_safe_region(cp.write_presentation(badged, Path(tmp) / "b.ass"))
            top = int(re.search(r"\\pos\((\d+),(\d+)\)", badge_lines[0]).group(2)) / 1920
            self.assertLessEqual(min(b.y0 for b in region.bands), top)

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
    @fx.needs_opencv
    def test_underline_pixels_render_inside_the_guard_band(self) -> None:
        import cv2
        import numpy as np

        expressive, _ = brand_mod.load_brand("mimir_expressive")
        planned = reason_plan(self.tokens, [(0, EmphasisReason.NAME)])
        built = cp.build_presentation(profile=base.profile(self.rows), clip_timeline=base.timeline(), plan=planned,
                                      width=540, height=960, brand=expressive)
        page = next(p for p in built.pages if self.tokens[0].word_id in p.emphasis_ids)
        with tempfile.TemporaryDirectory() as tmp:
            ass = cp.write_presentation(built, Path(tmp) / "u.ass")
            region = caption_safe_region(ass)
            out = Path(tmp) / "u.png"
            render_frame(ass, 540, 960, page.end - 0.05, out)
            frame = cv2.imread(str(out))
        lit = np.where((frame.max(axis=2) > 20).any(axis=1))[0]
        self.assertGreaterEqual(lit.min() / 960, min(b.y0 for b in region.bands) - 1e-3)
        self.assertLessEqual(lit.max() / 960, max(b.y1 for b in region.bands) + 1e-3)


# ============================================================
# PLACEMENT: UI occupancy, action regions, layout
# ============================================================

class PlacementV5Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = base.words("this is the moment everything changed forever")
        self.band = text_bands(build(self.rows))[0]

    def test_static_ui_under_the_caption_band_moves_captions(self) -> None:
        ui = StubUi(self.band[0] - 0.03, self.band[1] + 0.03)
        moved = build(self.rows, evidence=PlacementEvidence(ui=ui))
        self.assertNotIn("bottom", moved.placement.zones)
        self.assertIn("text_like", "text_like_occupancy")
        kept = build(self.rows, evidence=PlacementEvidence())
        self.assertEqual(set(kept.placement.zones), {"bottom"})

    def test_face_outranks_ui_and_action_outranks_ui(self) -> None:
        # UI at the bottom band, a (thin) face under every other zone: the face wins -> bottom stays.
        ui = StubUi(self.band[0] - 0.03, self.band[1] + 0.03)
        faces = tuple(TimedBox(0.0, 60.0, NormBox(0.05, c - 0.01, 0.95, c + 0.01), "face")
                      for c in (0.12, 0.40, 0.60))                           # centres of top/upper/lower zones
        built = build(self.rows, evidence=PlacementEvidence(boxes=faces, ui=ui))
        self.assertEqual(set(built.placement.zones), {"bottom"})
        action = TimedBox(0.0, 60.0, NormBox(0.1, self.band[0] - 0.02, 0.9, self.band[1] + 0.02), "action_region")
        moved = build(self.rows, evidence=PlacementEvidence(boxes=(action,)))
        self.assertNotIn("bottom", moved.placement.zones)
        weak = dataclasses.replace(action, weight=0.05)
        stays = build(self.rows, evidence=PlacementEvidence(boxes=(weak,)))
        self.assertEqual(set(stays.placement.zones), {"bottom"})

    def test_layout_multipliers_are_capped(self) -> None:
        evidence = PlacementEvidence(multipliers={"ui": 99.0, "activity": -3})
        self.assertEqual(evidence.multiplier("ui"), 1.5)
        self.assertEqual(evidence.multiplier("activity"), 0.0)


def face_track(subject_id: str, cx: float, cy: float, size: float, duration: float = 20.0, step: float = 0.25):
    from ai.editor.pro_edit.subjects import SubjectSample, SubjectTrack

    samples = tuple(SubjectSample(round(i * step, 3), cx, cy, size * 0.8, size, 0.9)
                    for i in range(int(duration / step)))
    return SubjectTrack(subject_id, "face", samples)


def activity_map(value: float, cols: int = 9, rows: int = 16) -> VisualOccupancyMap:
    cells = bytes([int(value * 255)]) * (cols * rows)
    return VisualOccupancyMap(cols, rows, 6.0, tuple(i / 2 for i in range(40)), (cells,) * 40, 0, "a")


class StubLayoutBackground:
    def __init__(self, ui_fraction: float = 0.0, panels=((), ())):
        self.samples = (1,)
        self.ui_fraction = ui_fraction
        self.panels = panels

    def text_occupancy(self, start, end):
        cells = 100
        hot = int(round(self.ui_fraction * cells))
        return [1.0] * hot + [0.0] * (cells - hot)

    def panel_lines(self, start, end):
        return self.panels


class LayoutClassifierTests(unittest.TestCase):
    def test_classes_from_existing_evidence(self) -> None:
        classify = layout_mod.classify_layout
        L = layout_mod.LayoutClass
        self.assertEqual(classify([face_track("a", 0.5, 0.4, 0.3)], 20.0).layout, L.TALKING_HEAD)
        self.assertEqual(classify([face_track("a", 0.3, 0.4, 0.2), face_track("b", 0.7, 0.4, 0.2)], 20.0).layout,
                         L.DUAL_TALKING_HEAD)
        cam = classify([face_track("c", 0.88, 0.85, 0.08)], 20.0, occupancy=activity_map(0.3),
                       background=StubLayoutBackground(0.03))
        self.assertEqual(cam.layout, L.GAMEPLAY_FACE_CAM)
        self.assertIsNotNone(cam.facecam_box)
        self.assertGreater(cam.facecam_box[0], 0.6)
        game = classify([], 20.0, occupancy=activity_map(0.3), background=StubLayoutBackground(0.04))
        self.assertEqual(game.layout, L.FULLSCREEN_GAMEPLAY)
        self.assertEqual(game.weights, {"ui": 1.25, "activity": 1.5})
        share = classify([], 20.0, occupancy=activity_map(0.01), background=StubLayoutBackground(0.2))
        self.assertEqual(share.layout, L.SCREEN_SHARE)
        panels = classify([], 20.0, occupancy=activity_map(0.08), background=StubLayoutBackground(0.0, ((0.5,), ())))
        self.assertEqual(panels.layout, L.MULTI_PANEL)

    def test_low_or_missing_evidence_is_unknown(self) -> None:
        L = layout_mod.LayoutClass
        self.assertEqual(layout_mod.classify_layout([], 20.0).layout, L.UNKNOWN)
        brief = face_track("a", 0.5, 0.4, 0.3, duration=4.0)                  # 20% of the clip
        self.assertEqual(layout_mod.classify_layout([brief], 20.0).layout, L.UNKNOWN)
        three = [face_track(n, x, 0.4, 0.2) for n, x in (("a", 0.2), ("b", 0.5), ("c", 0.8))]
        self.assertEqual(layout_mod.classify_layout(three, 20.0).layout, L.UNKNOWN)
        self.assertEqual(layout_mod.classify_layout([], 0.0).layout, L.UNKNOWN)


class ActionRegionTests(unittest.TestCase):
    def test_hotspot_in_payoff_span_only_and_global_motion_is_ignored(self) -> None:
        cols, rows = 9, 16
        cells = bytes(230 if (c in (6, 7) and r in (3, 4, 5)) else 10 for r in range(rows) for c in range(cols))
        occupancy = VisualOccupancyMap(cols, rows, 6.0, tuple(i / 6 for i in range(60)), (cells,) * 60, 0, "h")
        spans = [fx_span(StoryRole.PAYOFF, 2.0, 4.0), fx_span(StoryRole.SETUP, 5.0, 7.0)]
        actions = derive_action_regions(spans, occupancy)
        self.assertEqual([a.story_span_id for a in actions], ["s_payoff"])
        x0, y0, x1, y1 = actions[0].bbox
        self.assertTrue(0.6 < x0 < x1 <= 0.9 and 0.15 < y0 < y1 < 0.4)
        self.assertEqual(actions[0].evidence_source, "story_span+activity")
        uniform = activity_map(0.9)
        self.assertIsNone(activity_hotspot(uniform, 0.0, 5.0))
        self.assertIsNone(activity_hotspot(activity_map(0.1), 0.0, 5.0))

    @fx.needs_opencv
    def test_optional_detector_never_fails_the_short(self) -> None:
        spans = [fx_span(StoryRole.PAYOFF, 0.0, 1.0)]
        found, status = detect_people_in_spans("/nonexistent/clip.mp4", 640, 360, spans)
        self.assertEqual(found, [])
        self.assertIn("unavailable", status)
        import numpy as np

        from ai.editor.pro_edit.caption_action import HogPersonDetector

        detector = HogPersonDetector()
        self.assertEqual(detector.detect(np.zeros((360, 640, 3), np.uint8)), [])


# ============================================================
# GLOBAL EDITORIAL ENERGY
# ============================================================

class FakeFps:
    fps = 30.0


@dataclasses.dataclass
class FakeParams:
    preset: str


@dataclasses.dataclass
class FakeOp:
    source_event_id: str
    start_frame: int
    attack_frames: int
    params: FakeParams


@dataclasses.dataclass
class FakeResolved:
    ops: tuple
    fps: FakeFps = dataclasses.field(default_factory=FakeFps)


@dataclasses.dataclass
class FakeEvent:
    event_id: str
    role: StoryRole
    motion: MotionPreset


@dataclasses.dataclass
class FakePlan:
    events: tuple


class EnergyTests(unittest.TestCase):
    def impact_presentation(self):
        rows = base.words("this is the moment everything changed forever tonight", gap=0.05)
        tokens = cp.presentation_tokens(base.profile(rows), 99.0)
        planned = base.plan(base.caption_event("pay", 0.0, 9.0, "impact", [tokens[1].word_id]))
        return build(rows, plan=planned)

    def test_lowest_priority_degrades_first_and_payoff_camera_is_protected(self) -> None:
        presentation = self.impact_presentation()
        impact_page = next(p for p in presentation.pages if p.style is CaptionStyle.IMPACT)
        t = impact_page.start
        frame = int(round(t * 30))
        resolved = FakeResolved((FakeOp("cam_pay", frame, 6, FakeParams("punch_in")),
                                 FakeOp("cam_x", frame + 3, 6, FakeParams("punch_in_fast"))))
        plan = FakePlan((FakeEvent("cam_pay", StoryRole.PAYOFF, MotionPreset.PUNCH_IN),
                         FakeEvent("cam_x", StoryRole.ESCALATION, MotionPreset.PUNCH_IN_FAST)))
        recs = [{"channel": "sfx", "event_id": "s1", "frame": frame}, {"channel": "support_visual",
                                                                         "event_id": "m1", "frame": frame}]
        events = energy.collect_events(presentation=presentation, resolved=resolved, plan=plan,
                                       recommendations=recs)
        result = energy.coordinate(events, presentation=presentation, budget=4)
        channels = [d.channel for d in result.decisions]
        self.assertEqual(channels[:2], ["sfx", "support_visual"])
        self.assertLessEqual(result.max_load_after, 4)
        self.assertGreater(result.max_load_before, 4)
        self.assertNotIn("cam_pay", result.camera_motions)                   # payoff source action protected
        if "camera" in channels:
            self.assertLess(channels.index("caption") if "caption" in channels else -1, channels.index("camera"))
        self.assertEqual(result.dropped_recommendations, {("sfx", "s1"), ("support_visual", "m1")})
        again = energy.coordinate(energy.collect_events(presentation=presentation, resolved=resolved, plan=plan,
                                                        recommendations=recs), presentation=presentation, budget=4)
        self.assertEqual(again.to_dict(), result.to_dict())                     # deterministic

    def test_caption_degradation_keeps_text_timing_and_band(self) -> None:
        presentation = self.impact_presentation()
        page = next(p for p in presentation.pages if p.style is CaptionStyle.IMPACT)
        effects = presentation.settings.effects_for(page)
        calm = energy.caption_state_effects(effects, "calm")
        self.assertTrue(prim.REQUIRED_PRIMITIVES <= calm.primitives)
        self.assertNotIn(prim.PrimitiveId.SOFT_IMPACT_OUTLINE, calm.primitives)
        updated = cp.apply_effect_overrides(presentation, {page.page_id: calm}, {page.page_id: "calm"})
        self.assertEqual(updated.pages, presentation.pages)
        before = cp.page_safe_band(page, presentation)
        after = cp.page_safe_band(page, updated)
        self.assertGreaterEqual(before[0], after[0] - 1e-9)
        self.assertLessEqual(after[1], before[1] + 1e-9)
        self.assertNotEqual(updated.ass_text, presentation.ass_text)
        strip = lambda text: [re.sub(r"\{[^}]*\}", "", l) for l in dialogue_lines(text)]
        self.assertEqual(strip(updated.ass_text), strip(presentation.ass_text))   # same words, same events
        self.assertEqual(updated.manifest()["pages"][page.page_id]["energy"], "calm")

    def test_transition_is_never_degraded_and_unresolved_is_reported(self) -> None:
        events = [energy.EnergyEvent("transition:intro", "transition", 1.0, 1.5, 5, protected=True)]
        result = energy.coordinate(events, budget=4)
        self.assertEqual(result.decisions, [])
        self.assertEqual(len(result.unresolved), 1)

    def test_table_override_is_strict_and_camera_softening_maps_motions(self) -> None:
        table, problem = energy.parse_energy_table('{"sfx": 2}')
        self.assertEqual((table["sfx"], problem), (2, ""))
        for bad in ('{"meme": 1}', '{"sfx": 9}', '{"sfx": true}', "not json"):
            table, problem = energy.parse_energy_table(bad)
            self.assertEqual(table, dict(energy.ENERGY_TABLE))
            self.assertTrue(problem)
        self.assertEqual(energy.CAMERA_SOFTER[MotionPreset.PUNCH_IN_FAST], MotionPreset.SLOW_PUSH)
        self.assertTrue(all(m not in energy.CAMERA_SOFTER.values() for m in energy.CAMERA_SOFTER))


# ============================================================
# PLATFORM PROFILE v2
# ============================================================

def profile_set() -> dict:
    return {
        "schema_version": 2, "profile_set_id": "team_measured", "source": "team measurement 2026-09 (example)",
        "variants": [
            {"platform": "tiktok", "output_aspect": "9:16", "ui_variant": "organic_feed", "device_class": "phone",
             "profile": {"profile_id": "tt_feed", "version": 1, "minimum_edge_padding": 0.03,
                         "right_ui_column": [0.86, 0.4, 1.0, 0.85]},
             "description_footprints": {"short": {"box": [0.0, 0.84, 0.8, 0.92]},
                                        "long": {"box": [0.0, 0.74, 0.8, 0.92], "mode": "hard"}}},
            {"platform": "tiktok", "output_aspect": "9:16", "ui_variant": "ad", "device_class": "any",
             "profile": {"profile_id": "tt_ad", "version": 1, "bottom_ui_region": [0.0, 0.78, 1.0, 1.0]}},
            {"platform": "any", "output_aspect": "16:9", "ui_variant": "any", "device_class": "any",
             "profile": {"profile_id": "landscape", "version": 1}},
        ]}


class PlatformV2Tests(unittest.TestCase):
    def load(self, data, name="tiktok", **kw):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profil ş.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return platform_mod.load_platform_profile(name, str(path), **kw)

    def test_most_specific_variant_per_output_and_footprint(self) -> None:
        profile, notes = self.load(profile_set(), variant="organic_feed", device="phone", description="long")
        self.assertEqual(notes, ())
        vertical = profile.for_output(1080, 1920)
        ids = {r.region_id: r.mode for r in vertical.reserved_regions}
        self.assertEqual(ids, {"right_ui_column": "soft", "description_long": "hard"})
        self.assertFalse(vertical.verified)
        ad, _ = self.load(profile_set(), variant="ad")
        self.assertEqual([r.region_id for r in ad.for_output(1080, 1920).reserved_regions], ["bottom_ui_region"])
        self.assertEqual(profile.for_output(1920, 1080).reserved_regions, ())
        self.assertIn("landscape", profile.for_output(1920, 1080).profile_id)
        square = profile.for_output(1080, 1080)
        self.assertIn("no variant matches", square.notes[-1])

    def test_strict_validation_and_v1_compatibility(self) -> None:
        for mutate in (lambda d: d.update(extra=1), lambda d: d.update(schema_version=3),
                       lambda d: d["variants"][0].update(ui_variant="bad id!"),
                       lambda d: d["variants"][0].update(description_footprints={"huge": [0, 0, 1, 1]}),
                       lambda d: d["variants"][0]["profile"].update(minimum_edge_padding=float("nan")),
                       lambda d: d.update(variants=[])):
            data = profile_set()
            mutate(data)
            profile, notes = self.load(data)
            self.assertEqual(profile.profile_id, "generic", data)
            self.assertIn("rejected", notes[0])
        v1 = {"profile_id": "measured", "version": 1, "output_aspect": "9:16", "right_ui_column": [0.9, 0.3, 1, 0.9]}
        profile, notes = self.load(v1, name="custom")
        self.assertEqual((profile.profile_id, notes), ("measured", ()))

    def test_named_platforms_still_ship_no_invented_geometry(self) -> None:
        for name in platform_mod.NAMED_PLATFORMS:
            profile, notes = platform_mod.load_platform_profile(name)
            self.assertEqual(profile.profile_id, "generic_conservative")
            self.assertFalse(profile.verified)
        self.assertIn("BLOCKED_DATA", platform_mod.__doc__)


# ============================================================
# TEXT SHAPING vs libass
# ============================================================

class ShapingV5Tests(unittest.TestCase):
    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
    @fx.needs_opencv
    def test_width_prediction_bounds_libass_raster_for_unicode_classes(self) -> None:
        import cv2
        import numpy as np

        metrics = fm.load_metrics("Arial", True)
        if metrics.is_builtin:
            self.skipTest("no system font")
        from test_pro_edit_captions_v4 import base_ass_header

        cases = {"combining": "cafe\u0301 re\u0301sume\u0301", "precomposed": "café résumé",
                 "turkish": "İĞÜŞÖÇ ığüşöç", "ligature": "office affine", "cjk": "日本語の字幕",
                 "arabic": "مرحبا بالعالم", "hebrew": "שלום עולם", "emoji": "ok 👍 go 👍👍",
                 "zwj": "hi 👨\u200d👩\u200d👧 yo"}
        with tempfile.TemporaryDirectory() as tmp:
            for name, text in cases.items():
                ass = Path(tmp) / f"{name}.ass"
                ass.write_text(base_ass_header(1900, 300) + "Dialogue: 0,0:00:00.00,0:00:01.00,M,,0,0,0,,"
                               + "{\\an7\\pos(20,60)}" + text + "\n", encoding="utf-8")
                out = Path(tmp) / f"{name}.png"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=black:s=1900x300:d=1",
                                "-vf", f"subtitles=filename='{caption_renderer.escape_filter_path(ass)}'",
                                "-frames:v", "1", str(out)],
                               check=True, capture_output=True)
                cols = np.where((cv2.imread(str(out), 0) > 30).any(axis=0))[0]
                ink = cols.max() - cols.min() + 1 if len(cols) else 0
                self.assertLessEqual(ink, metrics.text_width(text, 100) + 2, name)

    def test_emoji_estimate_is_at_least_one_size(self) -> None:
        metrics = fm.builtin_metrics()
        px = metrics.text_width("👍", 100)
        self.assertGreaterEqual(px, 100 * fm.EMOJI_SIZE_RATIO - 1e-6)
        self.assertEqual(metrics.text_width("\u0301", 100), 0.0)             # combining mark: no advance


# ============================================================
# WINDOWS HYGIENE (static + path logic + Linux-executed stress paths)
# ============================================================

def _calls(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def _name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def v4_escape_reference(value: str) -> str:
    """The pre-V5 caption_renderer.escape_filter_path body, applied to an already resolved POSIX string."""
    if len(value) >= 2 and value[1] == ":":
        value = value[0] + "\\:" + value[2:]
    return value.replace("'", r"\'")


class WindowsPathLike:
    """Stand-in for pathlib.Path so the escaping functions see a resolved Windows path."""

    def __init__(self, value):
        self.value = PureWindowsPath(str(value))

    def resolve(self):
        return self

    def as_posix(self):
        return self.value.as_posix()

    def __str__(self):
        return str(self.value)


class WindowsHygieneTests(unittest.TestCase):
    def test_no_shell_true_and_every_text_subprocess_decodes_utf8(self) -> None:
        offenders = []
        for path in (REPO / "ai").rglob("*.py"):
            # utf-8-sig: current-root sources written by Windows tools may carry a BOM.
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for call in _calls(tree):
                keywords = {k.arg: k.value for k in call.keywords if k.arg}
                shell = keywords.get("shell")
                if isinstance(shell, ast.Constant) and shell.value is True:
                    offenders.append(f"{path.relative_to(REPO)}:{call.lineno} shell=True")
                if path.is_relative_to(PRO_EDIT) and _name(call) in ("run", "Popen", "check_output"):
                    text = keywords.get("text") or keywords.get("universal_newlines")
                    if isinstance(text, ast.Constant) and text.value is True and "encoding" not in keywords:
                        offenders.append(f"{path.relative_to(REPO)}:{call.lineno} text=True without encoding")
        self.assertEqual(offenders, [])

    def test_pro_edit_text_io_always_names_an_encoding(self) -> None:
        offenders = []
        for path in PRO_EDIT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for call in _calls(tree):
                name = _name(call)
                keywords = {k.arg for k in call.keywords}
                if name in ("read_text", "write_text") and "encoding" not in keywords:
                    offenders.append(f"{path.name}:{call.lineno} {name}")
                if name == "open" and "encoding" not in keywords:
                    mode = next((a.value for a in call.args if isinstance(a, ast.Constant)
                                 and isinstance(a.value, str)), "r")
                    if "b" not in mode:
                        offenders.append(f"{path.name}:{call.lineno} open({mode!r})")
        self.assertEqual(offenders, [])

    def test_filter_path_escaping_is_unchanged_for_windows_paths_and_fixed_for_apostrophes(self) -> None:
        windows_paths = [r"C:\Users\Test User\MIMIR Test\Türkçe Dosya\altyazı.ass",
                         r"D:\vod_output\clip_01 [final], v2;x\caps.ass", r"C:\a\b.ass",
                         r"E:\İĞÜŞÖÇ ığüşöç\çıktı klasörü\x.ass"]
        for module in (caption_renderer, intro_renderer):
            with mock.patch.object(module, "Path", WindowsPathLike):
                for value in windows_paths:
                    expected = v4_escape_reference(PureWindowsPath(value).as_posix())
                    self.assertEqual(module.escape_filter_path(value), expected, (module.__name__, value))
                apostrophe = module.escape_filter_path(r"C:\Users\O'Brien\caps.ass")
        self.assertEqual(apostrophe, "C\\:/Users/O'\\\\\\''Brien/caps.ass")

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
    def test_ffmpeg_burns_captions_from_stress_paths(self) -> None:
        """Path matrix (spaces, Turkish, apostrophes, [,;]) + a Unicode/space fontsdir; runs natively on Windows."""
        header = ("[Script Info]\nScriptType: v4.00+\nPlayResX: 320\nPlayResY: 240\n\n[V4+ Styles]\n"
                  "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
                  "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                  "Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: D,Arial,40,&H00FFFFFF,&H00FFFFFF,"
                  "&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1\n\n[Events]\n"
                  "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
                  "Dialogue: 0,0:00:00.00,0:00:01.00,D,,0,0,0,,ÇĞİÖŞÜ test\n")
        # A bold face that exists on this machine (Linux Liberation or Windows Arial Bold), so the
        # Unicode fontsdir branch is exercised on Windows too (drive colon + spaces + Turkish).
        candidates = (Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
                      Path(fm.os.environ.get("WINDIR") or r"C:\Windows") / "Fonts" / "arialbd.ttf")
        font = next((path for path in candidates if path.is_file()), candidates[0])
        with tempfile.TemporaryDirectory(prefix="mimir win ") as tmp:
            for rel, name in ((STRESS_DIR, "altyazı çğıİöşü.ass"), ("O'Brien's dir", "it's.ass"),
                              ("semi;colon,comma[x]", "a b.ass")):
                folder = Path(tmp) / rel
                folder.mkdir(parents=True)
                ass = folder / name
                ass.write_text(header, encoding="utf-8-sig")
                fonts_dir = None
                if font.is_file():
                    fonts_dir = folder / "yazı tipleri ş"
                    fonts_dir.mkdir()
                    shutil.copy(font, fonts_dir / "Marka Yazı.ttf")
                for label, graph in (("baseline", f"subtitles=filename='{caption_renderer.escape_filter_path(ass)}'"),
                                     ("pro_edit", subtitle_filter_for(ass, fonts_dir))):
                    out = folder / f"çıktı {label}.mp4"
                    run = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                                          "color=c=blue:s=320x240:d=1", "-vf", graph, "-frames:v", "1", str(out)],
                                         capture_output=True, encoding="utf-8", errors="replace")
                    self.assertEqual(run.returncode, 0, (rel, label, run.stderr[-300:]))
                    self.assertTrue(out.is_file())

    def test_windows_font_directory_logic_with_spaces_and_unicode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="Win Dir ş ") as tmp:
            fonts = Path(tmp) / "Fonts"
            fonts.mkdir()
            name = fm._WINDOWS_FILES.get(("arial", True))
            self.assertTrue(name)
            (fonts / name).write_bytes(b"\0")
            with mock.patch.object(fm.sys, "platform", "win32"), mock.patch.dict(fm.os.environ, {"WINDIR": tmp}):
                found = fm._windows_font("Arial", True)
        self.assertIsNotNone(found)
        self.assertEqual(found[0].name, name)


# ============================================================
# CONFIG / CACHE IDENTITY
# ============================================================

class ConfigV5Tests(unittest.TestCase):
    def test_defaults_invalid_values_and_plan_key_exclusion(self) -> None:
        default = load_config(True, {})
        self.assertEqual((default.caption_legibility, default.caption_ui, default.caption_layout,
                          default.caption_objects, default.energy, default.energy_window_ms, default.energy_budget),
                         (True, True, True, "off", True, 800, 4))
        bad = load_config(True, {"MIMIR_CAPTION_OBJECTS": "yolo", "MIMIR_PRO_EDIT_ENERGY_WINDOW_MS": "5",
                                 "MIMIR_PRO_EDIT_ENERGY_BUDGET": "x", "MIMIR_CAPTION_DESCRIPTION": "huge",
                                 "MIMIR_PRO_EDIT_ENERGY_TABLE": '{"meme": 3}',
                                 "MIMIR_CAPTION_PLATFORM_VARIANT": "bad id!"})
        self.assertEqual((bad.caption_objects, bad.energy_window_ms, bad.energy_budget, bad.caption_description,
                          bad.caption_platform_variant), ("off", 800, 4, "", ""))
        self.assertGreaterEqual(len(bad.problems), 5)
        changed = load_config(True, {"MIMIR_CAPTION_LEGIBILITY": "0", "MIMIR_CAPTION_UI_OCCUPANCY": "0",
                                     "MIMIR_CAPTION_LAYOUT": "0", "MIMIR_CAPTION_OBJECTS": "hog_person",
                                     "MIMIR_PRO_EDIT_ENERGY": "0", "MIMIR_PRO_EDIT_ENERGY_BUDGET": "6",
                                     "MIMIR_CAPTION_PLATFORM_VARIANT": "organic_feed"})
        self.assertEqual(changed.plan_signature_payload(), default.plan_signature_payload())
        self.assertNotEqual(changed.signature_payload(), default.signature_payload())

    def test_new_modules_never_enter_the_plan_cache_signature(self) -> None:
        from ai.editor import pro_edit

        for module in (pro_edit.caption_background, pro_edit.caption_legibility, pro_edit.caption_layout,
                       pro_edit.caption_action, pro_edit.editorial_energy):
            self.assertIn(module, pro_edit.MODULES)
            self.assertNotIn(module, pro_edit.PLAN_MODULES)
        self.assertEqual(pro_edit.PRO_EDIT_VERSION, 5)


# ============================================================
# STAGE INTEGRATION (real FFmpeg decode + analysis)
# ============================================================

@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available on PATH")
class StageV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import test_pro_edit_render as render
        from ai.editor import captions

        cls._tmp = tempfile.TemporaryDirectory(prefix="mimir v5 ş ")
        cls.root = Path(cls._tmp.name)
        cls.clip = cls.root / "clip_01_demo_edited.mp4"
        render.make_clip(cls.clip, "testsrc2")
        cls.timeline = cls.root / "timeline.json"
        cls.timeline.write_text(json.dumps(render.timeline_for("demo v5")), encoding="utf-8")
        cls.profile = cls.root / "speakers.json"
        cls.profile.write_text(json.dumps(render.profile()), encoding="utf-8")
        cls.ass = Path(captions.create_ass_for_clip({}, render.timeline_for("demo v5")["timelines"][0],
                                                    cls.root / "caps.ass", render.profile()))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def prepare(self, name: str, **config):
        from ai.editor.pro_edit.planner import StaticEditPlanner
        from ai.editor.pro_edit.stage import ProEditRequest, prepare_pro_edit
        from test_pro_edit_captions_v4 import FaceAtBottomProvider

        return prepare_pro_edit(ProEditRequest(
            config=ProEditConfig(enabled=True, planner="static", **config), timeline_path=self.timeline,
            clip_index=1, edited_clip_path=self.clip, caption_path=self.ass, output_path=self.root / f"{name}.mp4",
            artifact_dir=self.root / name, input_signature=f"sig-{name}", force=True,
            speaker_profile_path=self.profile, planner=StaticEditPlanner(), subject_provider=FaceAtBottomProvider()))

    @fx.needs_opencv
    def test_v5_evidence_is_computed_cached_and_reported(self) -> None:
        prep = self.prepare("v5")
        self.assertTrue(prep.ready, prep.reason)
        self.assertEqual(prep.captions["level"], "v5_placement")
        self.assertEqual(prep.captions["background"], "analyzed")
        self.assertTrue(prep.artifacts.caption_background.is_file())
        self.assertTrue(prep.artifacts.energy.is_file())
        self.assertIn(prep.energy["status"], ("within_budget", "applied"))
        manifest = json.loads(prep.artifacts.caption_presentation.read_text(encoding="utf-8"))
        self.assertIn("legibility", manifest)
        self.assertNotIn("not_analysed", manifest["legibility"]["counts"])
        self.assertIn(manifest["evidence"]["layout"]["layout"], [c.value for c in layout_mod.LayoutClass])
        self.assertIn("render_options", json.loads(prep.artifacts.resolved.read_text(encoding="utf-8")))

    def test_v5_switches_off_restore_v4_evidence(self) -> None:
        prep = self.prepare("v5off", caption_legibility=False, caption_ui=False, caption_layout=False, energy=False)
        self.assertTrue(prep.ready, prep.reason)
        self.assertEqual(prep.captions["background"], "disabled")
        self.assertEqual(prep.energy, {"status": "disabled"})
        manifest = json.loads(prep.artifacts.caption_presentation.read_text(encoding="utf-8"))
        self.assertEqual(manifest["legibility"]["counts"], {"not_analysed": manifest["metrics"]["pages"]})
        self.assertFalse(prep.artifacts.caption_background.exists())

    def test_background_failure_falls_back_without_failing(self) -> None:
        with mock.patch.object(bg, "load_or_analyze", side_effect=OSError("decoder broke")):
            prep = self.prepare("bgfail")
        self.assertTrue(prep.ready, prep.reason)
        self.assertEqual(prep.captions["background"], "failed")
        self.assertEqual(prep.captions["level"], "v5_placement")


if __name__ == "__main__":
    unittest.main()
