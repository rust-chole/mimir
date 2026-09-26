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


def test_yunet_is_the_production_detector_and_fallback_is_explicit(tmp_path, monkeypatch):
    from mimir.vision import faces

    monkeypatch.delenv("MIMIR_FACE_MODEL", raising=False)
    detector, reason = faces.load_detector()
    assert detector.name == "yunet" and reason == ""
    assert faces.detector_fingerprint().startswith("yunet:")
    monkeypatch.setattr(faces, "MODEL_DIR", tmp_path)
    detector, reason = faces.load_detector()
    assert detector.name == "haar" and "missing" in reason
    (tmp_path / faces.YUNET_FILE).write_bytes(b"not a model")
    detector, reason = faces.load_detector()
    assert detector.name == "haar" and "SHA-256" in reason


def test_small_corner_facecam_over_gameplay_is_detected():
    import cv2
    import skimage.data

    from mimir.vision.faces import load_detector
    from tests.synth import scene

    face = cv2.cvtColor(skimage.data.astronaut(), cv2.COLOR_RGB2BGR)[20:300, 110:340]
    cam = cv2.resize(face, (89, 108), interpolation=cv2.INTER_AREA)      # face ~40 px tall in 1280x720
    frame = cv2.resize(scene.gameplay_frame(3.0, 99.0), (1280, 720), interpolation=cv2.INTER_AREA)
    frame[720 - 120:720 - 12, 1280 - 101:1280 - 12] = cam
    detector, _ = load_detector()
    found = detector.detect(frame)
    assert len(found) == 1
    assert found[0].cx > 0.9 and found[0].cy > 0.85
    assert load_detector()[0].detect(cv2.resize(scene.gameplay_frame(5.0, 99.0), (1280, 720))) == []
