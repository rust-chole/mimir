import numpy as np

from mimir.media.frames import SampledFrame
from mimir.media.motion import motion_series, shot_boundaries
from mimir.vision.regions import classify_layout


def scene(color, t):
    image = np.zeros((90, 160, 3), np.uint8)
    image[:] = color
    image[30:60, 40:120] = (color[2], color[0], color[1])
    return SampledFrame(t, image)


def test_flash_burst_is_not_a_cut_but_a_persistent_change_is():
    green, white, blue = (30, 140, 40), (250, 250, 250), (160, 60, 20)
    flash = [scene(green if not 2.0 <= i / 10 < 2.6 else (white if i % 2 else (40, 120, 250)), i / 10)
             for i in range(50)]
    assert shot_boundaries(motion_series(flash)) == []
    cut = [scene(green if i < 20 else blue, i / 10) for i in range(50)]
    cuts = shot_boundaries(motion_series(cut))
    assert len(cuts) == 1 and abs(cuts[0] - 1.95) < 0.06


def face(box, std, coverage=0.95):
    return {"id": "f", "coverage": coverage, "median_box": box, "position_std": std, "t0": 0.0, "t1": 30.0}


def test_pixel_stable_corner_face_over_moving_content_is_a_facecam():
    corner = [0.86, 0.72, 0.92, 0.82]
    layout = classify_layout([face(corner, 0.001)], 0.09, 2.5, (0.0, 30.0))
    assert layout["class"] == "facecam_gameplay"
    # a real (drifting) person in a corner of a calm scene is not a facecam
    assert classify_layout([face(corner, 0.012)], 0.09, 2.5, (0.0, 30.0))["class"] != "facecam_gameplay"
    # and a stable corner face over a static picture is not gameplay
    assert classify_layout([face(corner, 0.001)], 0.09, 0.6, (0.0, 30.0))["class"] != "facecam_gameplay"
