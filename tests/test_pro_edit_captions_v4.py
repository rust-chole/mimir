"""Caption presentation V4: machine-verifiable checks (keyless; FFmpeg/libass where noted)."""
from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pro_edit_fixtures as fx
import test_pro_edit_captions as base
from ai.editor import caption_renderer, captions
from ai.editor.pro_edit import caption_brand as brand_mod
from ai.editor.pro_edit import caption_placement as placement_mod
from ai.editor.pro_edit import caption_platform as platform_mod
from ai.editor.pro_edit import caption_presentation as cp
from ai.editor.pro_edit import caption_primitives as prim
from ai.editor.pro_edit import font_metrics as fm
from ai.editor.pro_edit.caption_guard import CaptionSafeRegion, caption_safe_region
from ai.editor.pro_edit.caption_occupancy import VisualOccupancyMap, analyze_frames, load_or_analyze
from ai.editor.pro_edit.caption_placement import PlacementEvidence, TimedBox
from ai.editor.pro_edit.caption_platform import NormBox
from ai.editor.pro_edit.config import ProEditConfig, load_config
from ai.editor.pro_edit.executor import subtitle_filter_for
from ai.editor.pro_edit.ffmpeg_filters import validate_filter_graph
from ai.editor.pro_edit.schema import CaptionStyle, EmphasisReason, StoryRole, planner_json_schema
from ai.editor.pro_edit.style import PRO_STREAM_V1
from ai.editor.pro_edit.validator import PlanStatus, validate_plan_payload

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
GOLDEN = Path(__file__).resolve().parent / "golden" / "caption_presentation_v3_1.json"
BUILTIN = base.BUILTIN
# A regular (non-bold) face that exists on the machine: Liberation on Linux, Arial on Windows.
REGULAR_FONT_CANDIDATES = (
    (Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"), "Liberation Sans"),
    (Path(os.environ.get("WINDIR") or r"C:\Windows") / "Fonts" / "arial.ttf", "Arial"),
)
REGULAR_FONT, REGULAR_FAMILY = next(((path, family) for path, family in REGULAR_FONT_CANDIDATES if path.is_file()),
                                    REGULAR_FONT_CANDIDATES[0])


def ass_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(("Style:", "Dialogue:", "PlayRes"))]


def face(y0: float, y1: float, x0: float = 0.3, x1: float = 0.7, start: float = 0.0, end: float = 60.0) -> TimedBox:
    return TimedBox(start, end, NormBox(x0, y0, x1, y1), "face")


def build(rows, **kwargs) -> cp.CaptionPresentation:
    options = dict(profile=base.profile(rows), clip_timeline=base.timeline(), plan=None, width=1080, height=1920,
                   metrics=BUILTIN)
    options.update(kwargs)
    return cp.build_presentation(**options)


def text_bands(presentation: cp.CaptionPresentation) -> list[tuple[float, float]]:
    geometry = presentation.geometry
    return [((line.y - BUILTIN.line_height(page.font_px)) / geometry.height, line.y / geometry.height)
            for page in presentation.pages for line in page.lines]


# Frozen V3.1 cases whose input carries raw secondary/tertiary roles and UNCONFIRMED labels.
# V3.1 displayed those (permanent A/B/C lanes); the current root's captions V24 policy
# does not (lanes only inside measured overlap, human-confirmed names only).
V24_SPEAKER_POLICY_CASES = frozenset({"vertical_dual_label", "vertical_tertiary"})


def _neutral_speakers(rows: list) -> list:
    return [dict(row, speaker_raw="", speaker_role="main", speaker_label="") if isinstance(row, dict) else row
            for row in rows]


class GoldenParityTests(unittest.TestCase):
    """mimir_default + generic platform + no evidence == frozen V3.1 output."""

    def test_default_path_reproduces_v3_1_exactly(self) -> None:
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(golden), 5)
        self.assertLessEqual(V24_SPEAKER_POLICY_CASES, set(golden))
        for name, case in golden.items():
            inp = case["input"]
            plan = base.plan(*[base.caption_event(*e) for e in inp["plan"]]) if inp["plan"] else None
            for evidence in (None, PlacementEvidence()):
                options = dict(clip_timeline=base.timeline(events=inp["events"]), plan=plan, width=inp["width"],
                               height=inp["height"], metrics=BUILTIN, evidence=evidence,
                               brand=brand_mod.MIMIR_DEFAULT, platform=platform_mod.GENERIC)
                built = cp.build_presentation(profile=base.profile(inp["rows"]), **options)
                if name in V24_SPEAKER_POLICY_CASES:
                    # Unconfirmed raw speaker metadata has NO effect on the presentation:
                    # byte-identical to the same words with the speaker fields cleared.
                    neutral = cp.build_presentation(profile=base.profile(_neutral_speakers(inp["rows"])), **options)
                    self.assertEqual(built.ass_text, neutral.ass_text, f"{name} evidence={evidence}")
                    self.assertEqual({p.lane for p in built.pages}, {"main"}, name)
                    self.assertFalse(any(p.label for p in built.pages), name)
                    continue
                self.assertEqual(ass_lines(built.ass_text), case["ass_lines"], f"{name} evidence={evidence}")

    def test_default_brand_is_the_v3_1_palette(self) -> None:
        self.assertEqual(dict(cp.palettes_for(brand_mod.MIMIR_DEFAULT)), dict(cp.PALETTES))
        self.assertEqual(brand_mod.MIMIR_DEFAULT.allowed_primitives, prim.V31_PRIMITIVES)
        self.assertEqual(brand_mod.MIMIR_DEFAULT.font_family, captions.FONT_NAME)


class PlatformProfileTests(unittest.TestCase):
    def test_builtin_and_named_profiles_never_claim_verified_geometry(self) -> None:
        generic, notes = platform_mod.load_platform_profile("generic")
        self.assertEqual((generic.profile_id, notes), ("generic", ()))
        self.assertEqual(generic.reserved_regions, ())
        for name in platform_mod.NAMED_PLATFORMS:
            profile, notes = platform_mod.load_platform_profile(name)
            self.assertEqual(profile.profile_id, "generic_conservative")
            self.assertFalse(profile.verified)
            self.assertIn("no verified", notes[0])
        self.assertIn("NOT platform measurements", platform_mod.GENERIC_CONSERVATIVE.source)
        _profile, notes = platform_mod.load_platform_profile("mystery")
        self.assertIn("unknown", notes[0])

    def test_custom_json_profile_is_validated_strictly(self) -> None:
        good = {"profile_id": "measured_2026", "version": 1, "output_aspect": "9:16", "source": "team measurement",
                "minimum_edge_padding": 0.03, "right_ui_column": [0.87, 0.35, 1.0, 0.9], "shorthand_mode": "hard",
                "reserved_regions": [{"id": "bottom", "box": [0, 0.9, 1, 1], "mode": "soft", "weight": 2.5}],
                "preferred_caption_regions": [[0.05, 0.5, 0.85, 0.85]]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"
            path.write_text(json.dumps(good), encoding="utf-8")
            profile, notes = platform_mod.load_platform_profile("custom", str(path))
            self.assertEqual((profile.profile_id, notes), ("measured_2026", ()))
            self.assertEqual({r.region_id: r.mode for r in profile.reserved_regions},
                             {"bottom": "soft", "right_ui_column": "hard"})
            for bad in ({**good, "extra": 1}, {**good, "minimum_edge_padding": float("nan")},
                        {**good, "right_ui_column": [0.9, 0.2, 0.8, 0.3]}, {**good, "output_aspect": "wide"},
                        {**good, "reserved_regions": [{"box": [0, 0, 2, 1]}]}, {**good, "version": 0}):
                path.write_text(json.dumps(bad), encoding="utf-8")
                profile, notes = platform_mod.load_platform_profile("custom", str(path))
                self.assertEqual(profile.profile_id, "generic", bad)
                self.assertIn("rejected", notes[0])
            profile, notes = platform_mod.load_platform_profile("custom", None)
            self.assertEqual(profile.profile_id, "generic")
        conservative = platform_mod.GENERIC_CONSERVATIVE
        self.assertTrue(conservative.for_output(1080, 1920).reserved_regions)
        self.assertEqual(conservative.for_output(1920, 1080).reserved_regions, ())    # aspect-specific only


class PlacementTests(unittest.TestCase):
    ROWS = base.words("this is the moment everything changed forever and nobody expected that result")

    def test_no_evidence_keeps_every_page_at_the_v3_1_position(self) -> None:
        plain = build(self.ROWS)
        solved = build(self.ROWS, evidence=PlacementEvidence())
        self.assertEqual(ass_lines(plain.ass_text), ass_lines(solved.ass_text))
        self.assertEqual(solved.placement.zones, {"bottom": solved.placement.clusters})

    def test_face_under_the_caption_band_moves_captions_per_page(self) -> None:
        default_band = text_bands(build(self.ROWS))[0]
        evidence = PlacementEvidence(boxes=(face(default_band[0] - 0.03, default_band[1] + 0.03),))
        moved = build(self.ROWS, evidence=evidence)
        self.assertNotIn("bottom", moved.placement.zones)
        for top, bottom in text_bands(moved):
            self.assertTrue(bottom <= default_band[0] - 0.03 or top >= default_band[1] + 0.03)
        self.assertEqual(moved.placement.switches, 0)                 # one stable position
        region = caption_safe_region(self._write(moved))
        guard_top = min(b.y0 for b in region.bands)
        self.assertAlmostEqual(guard_top, min(t for t, _ in text_bands(moved)), delta=0.03)   # guard sees the move

    def test_brief_face_blips_do_not_make_captions_bounce(self) -> None:
        band = text_bands(build(self.ROWS))[0]
        blips = tuple(face(band[0], band[1], start=t, end=t + 0.1) for t in (0.8, 2.6))
        stable = build(self.ROWS, evidence=PlacementEvidence(boxes=blips))
        self.assertLessEqual(stable.placement.switches, 1)

    def test_simultaneous_lanes_move_together_and_never_overlap(self) -> None:
        rows = base.words("main speaker keeps talking here while", gap=0.05)
        rows += [{"word": w, "edited_start": 0.5 + i * 0.3, "edited_end": 0.75 + i * 0.3, "speaker_raw": "B",
                  "speaker_role": "secondary", "speaker_label": "SAM", "speaker_confidence": 0.9}
                 for i, w in enumerate("no way".split())]
        band = text_bands(build(rows))[0]
        moved = build(rows, evidence=PlacementEvidence(boxes=(face(band[0] - 0.12, band[1] + 0.02),)))
        zones = {page.lane: page.placement_zone for page in moved.pages if page.start < 1.5}
        self.assertEqual(len(set(zones.values())), 1)                 # the stack moves as one
        boxes = sorted(text_bands(moved))
        for (a0, a1), (b0, b1) in zip(boxes, boxes[1:]):
            if a0 != b0:
                self.assertLessEqual(a1, b0 + 1e-6)

    def test_story_region_outranks_a_face(self) -> None:
        band = text_bands(build(self.ROWS))[0]
        story = TimedBox(0.0, 60.0, NormBox(0.1, band[0] - 0.02, 0.9, band[1] + 0.02), "story_region")
        faces = tuple(face(y0, y0 + 0.08) for y0 in (0.30, 0.52))
        moved = build(self.ROWS, evidence=PlacementEvidence(boxes=(story, *faces)))
        for top, bottom in text_bands(moved):
            self.assertTrue(bottom <= band[0] or top >= band[1])

    def test_hard_platform_region_is_never_covered_and_bottom_policy_is_respected(self) -> None:
        band = text_bands(build(self.ROWS))[0]
        hard = platform_mod.PlatformSafeZoneProfile(
            "t", 1, reserved_regions=(platform_mod.ReservedRegion("ui", NormBox(0.0, band[0] - 0.01, 1.0, 1.0),
                                                                  "hard"),))
        moved = build(self.ROWS, platform=hard, evidence=PlacementEvidence(platform=hard))
        self.assertTrue(all(bottom <= band[0] - 0.01 for _t, bottom in text_bands(moved)))
        fixed = dataclasses.replace(brand_mod.MIMIR_DEFAULT, caption_anchor_policy="bottom", profile_id="fixed")
        kept = build(self.ROWS, brand=fixed, evidence=PlacementEvidence(boxes=(face(band[0], band[1]),)))
        self.assertEqual(set(kept.placement.zones), {"bottom"})

    def test_no_valid_zone_keeps_bottom_and_reports_it(self) -> None:
        everywhere = platform_mod.PlatformSafeZoneProfile(
            "wall", 1, reserved_regions=(platform_mod.ReservedRegion("all", NormBox(0, 0, 1, 1), "hard"),))
        kept = build(self.ROWS, platform=everywhere, evidence=PlacementEvidence(platform=everywhere))
        self.assertEqual(set(kept.placement.zones), {"bottom"})
        self.assertTrue(kept.placement.fallbacks)

    def test_upper_captions_become_avoid_bands_for_the_camera(self) -> None:
        from ai.editor.pro_edit.presets import story_constraints
        from ai.editor.pro_edit.camera import OutputProfile, base_window

        ws = fx.Workspace()
        try:
            context = fx.make_context(ws)
            top_ass = ws.root / "top.ass"
            presentation = build(base.words("words placed at the top"), width=1920, height=1080,
                                 evidence=PlacementEvidence(boxes=(face(0.8, 0.98),)))
            top_ass.write_text(presentation.ass_text, encoding="utf-8-sig")
            context = dataclasses.replace(context, caption_region=caption_safe_region(top_ass))
            page = presentation.pages[0]
            constraints = story_constraints(context, context.spans, page.start, page.end,
                                            base_window(1920, 1080, OutputProfile.PRESERVE), PRO_STREAM_V1,
                                            captions=True)
            self.assertIsNone(constraints.caption_top)
            self.assertTrue(constraints.avoid_bands and constraints.avoid_bands[0][1] < 0.5)
        finally:
            ws.cleanup()

    def _write(self, presentation) -> Path:
        self._tmp = tempfile.TemporaryDirectory()
        return cp.write_presentation(presentation, Path(self._tmp.name) / "p.ass")

    def tearDown(self) -> None:
        if hasattr(self, "_tmp"):
            self._tmp.cleanup()


def moving_square_frames(count: int, *, row: int, size: int = 24, shape=(160, 90), cut_at: int | None = None):
    import numpy as np

    frames = []
    for index in range(count):
        image = np.full((shape[0], shape[1], 3), 60, np.uint8)
        x = (index * 7) % (shape[1] - size)
        image[row:row + size, x:x + size] = 230
        if cut_at is not None and index >= cut_at:
            image = 255 - image
        frames.append((image, index / 6.0))
    return frames


class ActivityMapTests(unittest.TestCase):
    @fx.needs_opencv
    def test_motion_is_localized_and_named_honestly(self) -> None:
        occupancy = analyze_frames(moving_square_frames(24, row=110), cols=9, rows=16, sample_fps=6.0,
                                   fingerprint="t")
        values = occupancy.window(0.0, 4.0)
        low = occupancy.region_score((0.0, 0.66, 1.0, 0.95), values)
        high = occupancy.region_score((0.0, 0.05, 1.0, 0.4), values)
        self.assertGreater(low, 5 * max(high, 0.01))
        self.assertIn("not object detection", occupancy.to_dict()["kind"])
        again = VisualOccupancyMap.from_dict(json.loads(json.dumps(occupancy.to_dict())))
        self.assertEqual(again.window(0.0, 4.0), values)

    @fx.needs_opencv
    def test_scene_cuts_are_skipped_and_uniform_change_is_not_activity(self) -> None:
        occupancy = analyze_frames(moving_square_frames(12, row=20, cut_at=6), cols=9, rows=16, sample_fps=6.0,
                                   fingerprint="t")
        self.assertEqual(occupancy.cut_samples, 1)
        self.assertLess(occupancy.confidence, 1.0)

    def test_activity_under_the_band_moves_captions_and_flags_busy_when_unavoidable(self) -> None:
        rows = base.words("this is the moment everything changed forever")
        band = text_bands(build(rows))[0]
        cols, rows_n = 9, 16
        hot = bytes(255 if (band[0] - 0.05) <= (r + 0.5) / rows_n <= (band[1] + 0.05) else 0
                    for r in range(rows_n) for _c in range(cols))
        occupancy = VisualOccupancyMap(cols, rows_n, 6.0, tuple(i / 6.0 for i in range(60)), (hot,) * 60, 0, "x")
        moved = build(rows, evidence=PlacementEvidence(occupancy=occupancy))
        self.assertNotIn("bottom", moved.placement.zones)
        busy_all = VisualOccupancyMap(cols, rows_n, 6.0, tuple(i / 6.0 for i in range(60)),
                                      (bytes([255]) * (cols * rows_n),) * 60, 0, "y")
        busy = build(rows, evidence=PlacementEvidence(occupancy=busy_all))
        self.assertTrue(all(d.busy_background for d in busy.placement.decisions))

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
    @fx.needs_opencv
    def test_real_clip_analysis_is_cached_by_fingerprint(self) -> None:
        from ai.editor.pro_edit.media import probe_media

        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "moving box.mp4"
            # overlay re-evaluates x per frame (drawbox in this FFmpeg evaluates once: a static box).
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                            "color=c=0x303030:s=320x568:r=30:d=3", "-f", "lavfi", "-i", "color=c=white:s=40x40:r=30:d=3",
                            "-filter_complex", "[0][1]overlay=x='mod(t*120,280)':y=420:shortest=1",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True, capture_output=True)
            media = probe_media(clip)
            cache = Path(tmp) / "occ.json"
            first, status1 = load_or_analyze(cache, media, "clip-1")
            second, status2 = load_or_analyze(cache, media, "clip-1")
            third, status3 = load_or_analyze(cache, media, "clip-2")
        self.assertEqual((status1, status2, status3), ("analyzed", "cached", "analyzed"))
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertNotEqual(first.fingerprint, third.fingerprint)
        values = first.window(0.0, 3.0)
        self.assertGreater(first.region_score((0, 0.72, 1, 0.82), values),
                           3 * max(first.region_score((0, 0.1, 1, 0.5), values), 0.01))


class BrandProfileTests(unittest.TestCase):
    def good(self) -> dict:
        return {"profile_id": "team_brand", "version": 1, "primary_text_color": "#EEEEEE",
                "active_text_color": "#22DD88", "emphasis_text_color": "#FF3366", "outline_color": "#101010",
                "outline_scale": 1.2, "safe_width_ratio": 0.8, "allowed_primitives": [
                    "static", "active_color_ease", "active_outline_ease", "static_emphasis_scale", "page_alpha_in"]}

    def load(self, data) -> tuple[brand_mod.BrandProfile, tuple[str, ...]]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "b.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return brand_mod.load_brand(str(path))

    def test_valid_custom_brand_changes_only_presentation(self) -> None:
        brand, notes = self.load(self.good())
        self.assertEqual((brand.profile_id, notes), ("team_brand", ()))
        self.assertEqual(brand.lanes["main"].active, "&H88DD22&")
        rows = base.words(base.SENTENCE)
        default = build(rows)
        custom = build(rows, brand=brand)
        self.assertIn("&H88DD22&", custom.ass_text)
        self.assertIn("&H00101010", custom.ass_text)
        self.assertLess(custom.geometry.max_line_px, default.geometry.max_line_px)
        self.assertEqual([t.word_id for p in custom.pages for t in p.tokens],
                         [t.word_id for p in default.pages for t in p.tokens])        # same truth, same order

    def test_invalid_brands_fall_back_to_mimir_default(self) -> None:
        good = self.good()
        for bad in ({**good, "active_text_color": "gold"}, {**good, "outline_scale": -1},
                    {**good, "outline_scale": float("inf")}, {**good, "safe_width_ratio": 1.4},
                    {**good, "allowed_primitives": ["static", "sparkle_storm"]},
                    {**good, "allowed_primitives": ["static"]}, {**good, "default_style": "impact"},
                    {**good, "optional_font_path": "/nope/font.ttf"}, {**good, "raw_ass": "{\\blur9}"},
                    {**good, "profile_id": "mimir_default"}):
            brand, notes = self.load(bad)
            self.assertEqual(brand.profile_id, "mimir_default", bad)
            self.assertIn("rejected", notes[0])

    @unittest.skipUnless(REGULAR_FONT.is_file(), "no regular test font (Liberation Sans / Windows Arial) installed")
    def test_custom_font_path_is_measured_and_reported(self) -> None:
        brand, notes = self.load({**self.good(), "optional_font_path": str(REGULAR_FONT), "font_family": "Brand"})
        self.assertEqual(notes, ())
        from ai.editor.pro_edit.stage import _brand_font

        kwargs, note = _brand_font(brand, base.profile(base.words("Merhaba dünya")))
        self.assertEqual(kwargs["font_family"], REGULAR_FAMILY)
        self.assertFalse(kwargs["bold"])                                   # regular face: no synthetic bold
        # resolve(): Windows reports the canonical on-disk case (C:\Windows vs %WINDIR% = C:\WINDOWS).
        self.assertEqual(kwargs["fonts_dir"], str(REGULAR_FONT.resolve().parent))
        presentation = build(base.words("Merhaba dünya"), brand=brand, metrics=kwargs["metrics"],
                             font_family=kwargs["font_family"], bold=kwargs["bold"])
        self.assertIn(f"Style: MimirMain,{REGULAR_FAMILY},", presentation.ass_text)
        self.assertRegex(presentation.ass_text,
                         rf"Style: MimirMain,{re.escape(REGULAR_FAMILY)},[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,0,")
        _kwargs, missing = _brand_font(brand, base.profile(base.words("漢字 テスト")))
        self.assertIn("lacks glyphs", missing)
        graph = subtitle_filter_for("/tmp/a b/c.ass", kwargs["fonts_dir"])
        validate_filter_graph(graph)
        self.assertIn(":fontsdir='", graph)


class PrimitiveTests(unittest.TestCase):
    def test_library_is_small_and_complete(self) -> None:
        # V4 capped the library at 8; the V5 spec raises the cap to 12 (5 additions).
        self.assertLessEqual(len(prim.PRIMITIVE_LIBRARY), prim.MAX_LIBRARY_SIZE)
        self.assertLessEqual(prim.MAX_LIBRARY_SIZE, 12)
        for primitive in prim.PRIMITIVE_LIBRARY.values():
            record = primitive.to_dict()
            for key in ("id", "version", "geometry", "timing", "safe_bounds", "styles", "fallback", "validation"):
                self.assertIn(key, record)
        self.assertTrue(prim.REQUIRED_PRIMITIVES <= prim.DEFAULT_PRIMITIVES)

    def test_no_geometry_animation_in_any_primitive(self) -> None:
        brand = dataclasses.replace(brand_mod.MIMIR_DEFAULT, profile_id="all",
                                    allowed_primitives=frozenset(prim.PrimitiveId))
        rows = base.words("Kai scored 100 points in the final round tonight", gap=0.1)
        tokens = cp.presentation_tokens(base.profile(rows), 99.0)
        event = base.caption_event("pay", 0.0, 9.0, "impact", [tokens[0].word_id, tokens[2].word_id])
        built = build(rows, brand=brand, plan=base.plan(event), verified_names=["Kai Cenat"])
        for line in built.ass_text.splitlines():
            if not line.startswith("Dialogue"):
                continue
            for block in re.findall(r"\\t\([^)]*\)", line):
                self.assertNotRegex(block, r"\\fsc|\\fs\d|\\pos|\\move|\\frz|\\fr[xy]")
            self.assertNotIn("\\move", line)

    def test_page_fade_is_bounded_and_only_on_the_page_start(self) -> None:
        brand = dataclasses.replace(brand_mod.MIMIR_DEFAULT, profile_id="fade",
                                    allowed_primitives=prim.V31_PRIMITIVES | {prim.PrimitiveId.PAGE_ALPHA_IN})
        rows = base.words("long enough page here. ok", gap=0.12, sentence_gap=0.9)
        built = build(rows, brand=brand)
        fades = [(line, int(m)) for line in built.ass_text.splitlines() for m in re.findall(r"\\fad\((\d+),0\)", line)]
        self.assertTrue(fades)
        for line, ms in fades:
            start = line.split(",")[1]
            self.assertLessEqual(ms, prim.PAGE_FADE_MAX_MS)
            self.assertIn(start, {cp._ass_time_cs(cp._cs(p.start)) for p in built.pages})
        short_page = next(p for p in built.pages if p.end - p.start < prim.PAGE_FADE_MIN_PAGE_S)
        self.assertFalse(any(cp._ass_time_cs(cp._cs(short_page.start)) == l.split(",")[1] for l, _ in fades))

    def test_backplate_is_budgeted_layered_below_text_and_visible_to_caption_guard(self) -> None:
        brand = dataclasses.replace(brand_mod.MIMIR_DEFAULT, profile_id="plates",
                                    allowed_primitives=prim.V31_PRIMITIVES | {prim.PrimitiveId.STATIC_ACCENT_BACKPLATE})
        rows = base.words("this is the moment everything changed forever and nobody expected that result at all")
        busy = VisualOccupancyMap(9, 16, 6.0, tuple(i / 2 for i in range(40)), (bytes([255]) * 144,) * 40, 0, "b")
        built = build(rows, brand=brand, evidence=PlacementEvidence(occupancy=busy))
        plates = [l for l in built.ass_text.splitlines() if "\\p1" in l]
        texts = [l for l in built.ass_text.splitlines() if l.startswith("Dialogue") and "\\p1" not in l]
        self.assertTrue(plates)
        self.assertLess(max(int(l.split(",")[0].split()[1]) for l in plates),
                        min(int(l.split(",")[0].split()[1]) for l in texts))
        self.assertTrue(built.primitive_log.degraded)                  # budget stopped some plates
        self.assertLess(len({l.split(",")[1] for l in plates}), len(built.pages))
        with tempfile.TemporaryDirectory() as tmp:
            region = caption_safe_region(cp.write_presentation(built, Path(tmp) / "plates.ass"))
        plate_top = min(int(re.search(r"\\pos\((\d+),(\d+)\)", l).group(2)) for l in plates) / 1920
        self.assertLessEqual(min(b.y0 for b in region.bands), plate_top)
        manifest = built.manifest()
        self.assertLessEqual(manifest["metrics"]["safe_band"][0], plate_top + 1e-4)     # 4-decimal rounding

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
    @fx.needs_opencv
    def test_backplate_pixels_lie_inside_the_guard_band(self) -> None:
        import cv2
        import numpy as np

        brand = dataclasses.replace(brand_mod.MIMIR_DEFAULT, profile_id="plates",
                                    allowed_primitives=prim.V31_PRIMITIVES | {prim.PrimitiveId.STATIC_ACCENT_BACKPLATE},
                                    backplate_color="&HFFFFFF&", backplate_alpha=0)
        rows = base.words("plate check page here")
        busy = VisualOccupancyMap(9, 16, 6.0, tuple(i / 2 for i in range(20)), (bytes([255]) * 144,) * 20, 0, "b")
        built = cp.build_presentation(profile=base.profile(rows), clip_timeline=base.timeline(), plan=None,
                                      width=540, height=960, brand=brand, evidence=PlacementEvidence(occupancy=busy))
        with tempfile.TemporaryDirectory() as tmp:
            ass = cp.write_presentation(built, Path(tmp) / "p.ass")
            region = caption_safe_region(ass)
            out = Path(tmp) / "f.png"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=black:s=540x960:d=3:r=25",
                            "-vf", f"subtitles=filename='{caption_renderer.escape_filter_path(ass)}'", "-ss",
                            f"{built.pages[0].start + 0.1:.2f}",
                            "-frames:v", "1", str(out)], check=True, capture_output=True)
            frame = cv2.imread(str(out))
        rows_lit = np.where((frame.max(axis=2) > 40).any(axis=1))[0]
        self.assertTrue(len(rows_lit))
        self.assertGreaterEqual(rows_lit.min() / 960, min(b.y0 for b in region.bands) - 1e-3)
        self.assertLessEqual(rows_lit.max() / 960, max(b.y1 for b in region.bands) + 1e-3)


class EmphasisReasonTests(unittest.TestCase):
    def test_planner_contract_is_backward_compatible(self) -> None:
        ws = fx.Workspace()
        try:
            context = fx.make_context(ws)
            word = next(w for w in context.words if 13.0 <= w.start <= 14.0)
            old = fx.plan(fx.event("e1", 13.0, 14.5, caption_style="impact", emphasis_word_ids=[word.id]))
            report_old = validate_plan_payload(old, context, PRO_STREAM_V1)
            self.assertEqual(report_old.status, PlanStatus.VALID)
            self.assertNotIn("emphasis_reasons", report_old.plan.events[0].to_dict())      # plan hash unchanged
            new = fx.plan(fx.event("e1", 13.0, 14.5, caption_style="impact", emphasis_word_ids=[word.id],
                                   emphasis_reasons=[{"word_id": word.id, "reason": "surprise"},
                                                     {"word_id": 99999, "reason": "name"},
                                                     {"word_id": word.id, "reason": "invented"}]))
            report_new = validate_plan_payload(new, context, PRO_STREAM_V1)
            self.assertEqual(report_new.plan.events[0].emphasis_reasons, ((word.id, EmphasisReason.SURPRISE),))
            self.assertIn("emphasis_reason_dropped", [i.code for i in report_new.issues])
        finally:
            ws.cleanup()
        schema = planner_json_schema("pro_stream_v1")
        event = schema["properties"]["events"]["items"]
        self.assertIn("emphasis_reasons", event["required"])                                 # strict-mode safe
        self.assertFalse(event["properties"]["emphasis_reasons"]["items"]["additionalProperties"])

    def test_local_derivation_is_conservative(self) -> None:
        rows = base.words("Kai scored 100 points and then everyone started screaming wildly")
        tokens = cp.presentation_tokens(base.profile(rows), 99.0)
        spans = [fx_span(StoryRole.PAYOFF, tokens[5].start, tokens[6].end),
                 fx_span(StoryRole.REACTION, tokens[8].start, tokens[9].end)]
        names = cp.name_tokens(["Kai Cenat"])
        self.assertEqual(cp.derive_reason(tokens[0], names, spans), "name")
        self.assertEqual(cp.derive_reason(tokens[2], names, spans), "number")
        self.assertEqual(cp.derive_reason(tokens[5], names, spans), "payoff")
        self.assertEqual(cp.derive_reason(tokens[8], names, spans), "reaction")
        self.assertEqual(cp.derive_reason(tokens[4], names, spans), "generic")
        self.assertTrue(set(cp.LOCAL_REASONS).isdisjoint({"contrast", "surprise"}))
        planned = base.plan(dataclasses.replace(base.caption_event("e", 0.0, 9.0, "emphasis", [tokens[5].word_id]),
                                                emphasis_reasons=((tokens[5].word_id, EmphasisReason.CONTRAST),)))
        manifest = build(rows, plan=planned, spans=spans, verified_names=["Kai Cenat"]).manifest()["emphasis_reasons"]
        self.assertEqual(manifest[str(tokens[5].word_id)], {"reason": "contrast", "source": "planner"})
        derived = base.plan(base.caption_event("e", 0.0, 9.0, "emphasis", [tokens[0].word_id]))
        manifest = build(rows, plan=derived, spans=spans, verified_names=["Kai Cenat"]).manifest()["emphasis_reasons"]
        self.assertEqual(manifest[str(tokens[0].word_id)], {"reason": "name", "source": "derived"})


def fx_span(role: StoryRole, start: float, end: float):
    from ai.editor.pro_edit.story import Protection, StorySpan
    from ai.editor.pro_edit.timebase import TimelineDomain

    return StorySpan(f"s_{role.value}", role, start, end, TimelineDomain.PACED_CLIP, Protection.CAMERA_EDIT_ALLOWED)


class PlanCacheIdentityTests(unittest.TestCase):
    def test_key_depends_only_on_semantic_planning_inputs(self) -> None:
        from ai.editor.pro_edit.stage import planner_cache_key

        ws = fx.Workspace()
        try:
            context = fx.make_context(ws)
            config = ProEditConfig(enabled=True, planner="model")
            key = planner_cache_key(context, config)
            for presentation_only in (dict(captions=False), dict(caption_brand="/x.json"),
                                      dict(caption_platform="generic_conservative"), dict(caption_activity=False),
                                      dict(caption_shaper="auto")):
                self.assertEqual(planner_cache_key(context, dataclasses.replace(config, **presentation_only)), key)
            self.assertNotEqual(planner_cache_key(context, dataclasses.replace(config, model="other-model")), key)
            other_words = fx.make_context(ws, words=[("hello", 1.0, 1.4, "A"), ("there", 1.5, 1.9, "A")])
            self.assertNotEqual(planner_cache_key(other_words, config), key)
            with mock.patch.object(cp, "CAPTION_PRESENTATION_VERSION", 99):
                self.assertEqual(planner_cache_key(context, config), key)
        finally:
            ws.cleanup()
        env = load_config(True, {"MIMIR_CAPTION_BRAND_PROFILE": "/x.json", "MIMIR_CAPTION_SHAPER": "auto"})
        self.assertEqual(env.plan_signature_payload(), ProEditConfig(enabled=True).plan_signature_payload())
        self.assertNotEqual(env.signature_payload(), ProEditConfig(enabled=True).signature_payload())
        self.assertEqual(load_config(True, {"MIMIR_CAPTION_SHAPER": "magic"}).caption_shaper, "naive")


class ShapingAndKerningTests(unittest.TestCase):
    def test_kerning_is_parsed_but_never_narrows_layout(self) -> None:
        metrics = fm.load_metrics("Arial", True)
        if metrics.is_builtin or not metrics.kern_pairs:
            self.skipTest("no system font with a classic kern table")
        for text in ("AVAVAV", "To To", "WAWA"):
            self.assertLess(metrics.kerned_width(text, 100), metrics.text_width(text, 100))
        self.assertEqual(metrics.kerned_width("HHHH", 100), metrics.text_width("HHHH", 100))
        built = build(base.words("AVATAR WAVES To Tokyo"), metrics=metrics)
        measure = cp._Measure(metrics, built.geometry.font_px)
        line = built.pages[0].lines[0]
        texts = [next(t.text for t in built.pages[0].tokens if t.word_id == w) for w in line.word_ids]
        self.assertAlmostEqual(line.width_px, round(cp.line_width(texts, [1.0] * len(texts), measure), 1))
        self.assertGreater(built.manifest()["shaping"]["max_line_tightening_ratio"], 0.0)

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
    @fx.needs_opencv
    def test_libass_raster_matches_unkerned_not_kerned_widths(self) -> None:
        import cv2
        import numpy as np

        metrics = fm.load_metrics("Arial", True)
        if metrics.is_builtin or not metrics.kern_pairs:
            self.skipTest("no system font with a classic kern table")
        head = base_ass_header(1600, 400)
        text = "AVAVAVAVAV"
        with tempfile.TemporaryDirectory() as tmp:
            ass = Path(tmp) / "k.ass"
            ass.write_text(head + f"Dialogue: 0,0:00:00.00,0:00:01.00,M,,0,0,0,,{{\\an7\\pos(100,100)}}{text}\n",
                           encoding="utf-8")
            out = Path(tmp) / "k.png"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=black:s=1600x400:d=1",
                            "-vf", f"subtitles=filename='{caption_renderer.escape_filter_path(ass)}'",
                            "-frames:v", "1", str(out)],
                           check=True, capture_output=True)
            cols = np.where((cv2.imread(str(out), 0) > 60).any(axis=0))[0]
        ink = cols.max() - cols.min() + 1
        unkerned, kerned = metrics.text_width(text, 100), metrics.kerned_width(text, 100)
        self.assertLessEqual(ink, unkerned + 2)                 # layout width is an upper bound
        self.assertGreater(ink, kerned + 0.5 * (unkerned - kerned))   # libass here does not kern

    def test_harfbuzz_envelope_never_narrows_and_is_optional(self) -> None:
        metrics = fm.load_metrics("Arial", True)
        shaper, note = fm.select_shaper(metrics, "naive")
        self.assertEqual(note, "naive")
        with mock.patch.dict("sys.modules", {"uharfbuzz": None}):
            fallback, note = fm.select_shaper(metrics, "harfbuzz")
        self.assertTrue(note.startswith("naive"))
        self.assertIsInstance(fallback, fm.NaiveShaper)
        try:
            import uharfbuzz  # noqa: F401
        except ImportError:
            self.skipTest("uharfbuzz not installed (optional)")
        if metrics.is_builtin:
            self.skipTest("no font file to shape")
        envelope, note = fm.select_shaper(metrics, "auto")
        self.assertEqual(note, "harfbuzz_envelope")
        for text in ("Merhaba dünya", "İstanbul’da ŞİMDİ", "AVAV", "مرحبا بالعالم", "office affluent"):
            self.assertGreaterEqual(envelope.width(text, 74), metrics.text_width(text, 74) - 1e-9)
        self.assertAlmostEqual(envelope.width("Merhaba dünya", 74), metrics.text_width("Merhaba dünya", 74))


def base_ass_header(width: int, height: int) -> str:
    return (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {width}\nPlayResY: {height}\nWrapStyle: 2\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
            "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
            "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: M,Arial,100,&H00FFFFFF,&H00FFFFFF,&H00000000,&HFF000000,-1,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")


class FaceAtBottomProvider:
    name = "test_faces"

    def load(self, duration_s: float):
        from ai.editor.pro_edit.subjects import SubjectSample, SubjectTrack

        samples = tuple(SubjectSample(round(t * 0.2, 2), 0.5, 0.86, 0.2, 0.2, 0.95) for t in range(int(duration_s * 5)))
        return (SubjectTrack("face_1", "face", samples),)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available on PATH")
class StageIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import test_pro_edit_render as render

        cls.render = render
        cls._tmp = tempfile.TemporaryDirectory(prefix="mimir v4 ")
        cls.root = Path(cls._tmp.name)
        cls.clip = cls.root / "clip_01_demo_edited.mp4"
        render.make_clip(cls.clip, "testsrc2")
        cls.timeline = cls.root / "timeline.json"
        cls.timeline.write_text(json.dumps(render.timeline_for("demo v4")), encoding="utf-8")
        cls.profile = cls.root / "speakers.json"
        cls.profile.write_text(json.dumps(render.profile()), encoding="utf-8")
        cls.ass = Path(captions.create_ass_for_clip({}, render.timeline_for("demo v4")["timelines"][0],
                                                    cls.root / "caps.ass", render.profile()))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def prepare(self, name: str, **config):
        from ai.editor.pro_edit.planner import StaticEditPlanner
        from ai.editor.pro_edit.stage import ProEditRequest, prepare_pro_edit

        return prepare_pro_edit(ProEditRequest(
            config=ProEditConfig(enabled=True, planner="static", **config), timeline_path=self.timeline,
            clip_index=1, edited_clip_path=self.clip, caption_path=self.ass, output_path=self.root / f"{name}.mp4",
            artifact_dir=self.root / name, input_signature=f"sig-{name}", force=True,
            speaker_profile_path=self.profile, planner=StaticEditPlanner(), subject_provider=FaceAtBottomProvider()))

    @fx.needs_opencv
    def test_face_evidence_moves_captions_and_the_camera_region_follows(self) -> None:
        prep = self.prepare("faces")
        self.assertTrue(prep.ready, prep.reason)
        self.assertEqual(prep.captions["level"], "v5_placement")          # V5 rung above v4_placement
        self.assertNotIn("bottom", prep.captions["placement_zones"])
        self.assertEqual(prep.captions["activity"], "analyzed")
        manifest = json.loads(prep.artifacts.caption_presentation.read_text(encoding="utf-8"))
        self.assertIn("faces", manifest["placement"]["evidence"])
        region = caption_safe_region(prep.presentation_ass)
        self.assertLess(max(b.y1 for b in region.bands), 0.5)                 # captions now at the top
        self.assertTrue(prep.artifacts.caption_occupancy.is_file())
        again = self.prepare("faces")
        self.assertEqual(again.captions["activity"], "analyzed")               # force=True re-analyzes

    @fx.needs_opencv
    def test_fallback_ladder(self) -> None:
        with mock.patch.object(cp, "solve_placement", side_effect=RuntimeError("boom")):
            prep = self.prepare("ladder")
        self.assertEqual(prep.captions["level"], "v31_placement")
        self.assertEqual(prep.captions["fallbacks"], 2)                    # v5_placement + v4_placement
        with mock.patch("ai.editor.pro_edit.stage.load_or_analyze", side_effect=OSError("no frames")):
            prep = self.prepare("noactivity")
        self.assertEqual((prep.captions["level"], prep.captions["activity"]), ("v5_placement", "failed"))
        with mock.patch("ai.editor.pro_edit.stage.build_presentation", side_effect=RuntimeError("all broken")):
            prep = self.prepare("baseline")
        self.assertEqual(prep.captions["status"], "baseline_ass")
        self.assertFalse(prep.ready)                                          # static plan + no presentation


class ReviewRegressionTests(unittest.TestCase):
    """Defects found in the V4 self-review."""

    def test_tertiary_lane_inherits_main_roles_unless_set_explicitly(self) -> None:
        only_active = brand_mod.parse_brand({"profile_id": "a", "version": 1, "active_text_color": "#112233"})
        self.assertEqual(only_active.lanes["tertiary"].active, only_active.lanes["main"].active)
        mixed = brand_mod.parse_brand({"profile_id": "b", "version": 1, "primary_text_color": "#EEEEEE",
                                       "speaker_tertiary_active_color": "#445566"})
        self.assertEqual(mixed.lanes["tertiary"].active, "&H665544&")          # explicit value kept
        self.assertEqual(mixed.lanes["tertiary"].text, mixed.lanes["main"].text)
        self.assertEqual(mixed.lanes["secondary"], brand_mod.MIMIR_DEFAULT.lanes["secondary"])
        with self.assertRaises(brand_mod.BrandProfileError):
            brand_mod.parse_brand({"profile_id": "c", "version": 1, "primary_text_color": "#EEEEEE",
                                   "speaker_primary_color": "#000000"})

    def test_kern_parser_ignores_minimum_tables_and_uses_exact_format0_size(self) -> None:
        import struct

        def subtable(coverage: int, pairs: list[tuple[int, int, int]], wrap: bool = False) -> bytes:
            body = struct.pack(">HHHH", len(pairs), 0, 0, 0) + b"".join(struct.pack(">HHh", *p) for p in pairs)
            length = 0 if wrap else 6 + len(body)
            return struct.pack(">HHH", 0, length, coverage) + body

        table = struct.pack(">HH", 0, 3) + subtable(0x0001, [(1, 2, -50)], wrap=True) \
            + subtable(0x0003, [(1, 2, -999)]) + subtable(0x0001, [(3, 4, -20)])
        found = fm._kern_pairs(table, {b"kern": (0, len(table))})
        self.assertEqual(found, {(1, 2): -50, (3, 4): -20})
