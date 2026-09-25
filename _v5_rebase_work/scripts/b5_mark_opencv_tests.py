"""Batch B5 helper: mark the tests that measure OpenCV/numpy pixel evidence.

The list is exactly the set that failed ONLY with ModuleNotFoundError (cv2/numpy)
or with evidence status 'failed' in the production .venv (no OpenCV installed).
Each name must exist exactly once; the decorator is inserted directly above the
``def`` line (stacks with existing skip decorators).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1] / "final_root" / "tests"

MARK = {
    "test_pro_edit_captions.py": [
        "test_libass_renders_escaped_text_on_one_line",
        "test_rendered_pages_stay_inside_the_frame_and_do_not_move",
        "test_outline_accent_does_not_move_words",
        "test_two_line_page_line_one_is_stable_when_line_two_appears",
    ],
    "test_pro_edit_captions_v4.py": [
        "test_motion_is_localized_and_named_honestly",
        "test_real_clip_analysis_is_cached_by_fingerprint",
        "test_scene_cuts_are_skipped_and_uniform_change_is_not_activity",
        "test_backplate_pixels_lie_inside_the_guard_band",
        "test_libass_raster_matches_unkerned_not_kerned_widths",
        "test_face_evidence_moves_captions_and_the_camera_region_follows",
        "test_fallback_ladder",
    ],
    "test_pro_edit_protection.py": [
        "test_identity_survives_missed_detections_and_is_sparse",
        "test_scene_cut_forces_redetection",
        "test_two_subjects_do_not_swap",
    ],
    "test_pro_edit_v5.py": [
        "test_optional_detector_never_fails_the_short",
        "test_luminance_statistics_segments_and_panels",
        "test_plain_shapes_noise_and_smooth_texture_are_not_text",
        "test_real_clip_sidecar_in_unicode_path_is_cached_by_fingerprint",
        "test_static_hud_text_is_persistent_occupancy_and_transient_text_is_not",
        "test_legibility_pixels_stay_inside_the_guard_band",
        "test_underline_pixels_render_inside_the_guard_band",
        "test_width_prediction_bounds_libass_raster_for_unicode_classes",
        "test_v5_evidence_is_computed_cached_and_reported",
    ],
}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    total = 0
    for name, tests in MARK.items():
        path = TESTS / name
        text = path.read_text(encoding="utf-8")
        if name == "test_pro_edit_v5.py" and "import pro_edit_fixtures as fx" not in text:
            anchor = "import test_pro_edit_captions as base\n"
            if text.count(anchor) != 1:
                raise SystemExit("v5 import anchor missing")
            text = text.replace(anchor, "import pro_edit_fixtures as fx\n" + anchor)
        for test in tests:
            pattern = re.compile(rf"^(?P<indent>[ \t]+)def {test}\(", re.M)
            found = list(pattern.finditer(text))
            if len(found) != 1:
                raise SystemExit(f"{name}: {test} found {len(found)} times")
            match = found[0]
            decorator = f"{match.group('indent')}@fx.needs_opencv\n"
            if text[:match.start()].endswith(decorator):
                continue
            text = text[:match.start()] + decorator + text[match.start():]
            total += 1
        compile(text, str(path), "exec", dont_inherit=True)
        path.write_bytes(text.encode("utf-8"))   # keep LF (write_text would emit CRLF on Windows)
    print(f"marked {total} tests with @fx.needs_opencv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
