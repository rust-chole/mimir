"""Local face detectors behind one interface.

* YuNet (cv2.FaceDetectorYN): accurate, fast, 5 landmarks, real scores. Used
  when the ONNX model is available (MIMIR_PRO_EDIT_FACE_MODEL or
  ai/editor/pro_edit/models/face_detection_yunet_2023mar.onnx; MIT license,
  opencv_zoo). Not auto-downloaded.
* Haar cascades bundled with opencv-python(-headless): no extra file. Frontal
  + profile (both directions), confidence from cascade level weights.

Both return normalized boxes in the sampled frame; detectors never decide
camera behavior.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from ai.editor.pro_edit.vision.cv_runtime import OpenCVUnavailable, load_opencv

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
MIN_FACE_FRACTION = 0.035  # of frame height


@dataclass(frozen=True)
class Detection:
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float
    kind: str = "face"
    detector: str = ""

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


class FaceDetector(Protocol):
    name: str

    def detect(self, image: Any) -> list[Detection]:
        ...


def _iou(a: Detection, b: Detection) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(detections: Sequence[Detection], threshold: float = 0.35) -> list[Detection]:
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: -d.confidence):
        if all(_iou(det, k) < threshold for k in kept):
            kept.append(det)
    return kept


class YuNetFaceDetector:
    name = "yunet"

    def __init__(self, model_path: str | Path, score_threshold: float = 0.6, nms_threshold: float = 0.3) -> None:
        cv2 = load_opencv()

        self._cv2 = cv2
        self._model = str(model_path)
        self._score = score_threshold
        self._nms = nms_threshold
        self._size: tuple[int, int] | None = None
        self._net = cv2.FaceDetectorYN.create(self._model, "", (320, 320), score_threshold, nms_threshold, 50)

    def detect(self, image: Any) -> list[Detection]:
        h, w = image.shape[:2]
        if self._size != (w, h):
            self._net.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self._net.detect(image)
        result: list[Detection] = []
        for row in faces if faces is not None else []:
            x, y, bw, bh = (float(v) for v in row[:4])
            score = float(row[-1])
            if bh / h < MIN_FACE_FRACTION:
                continue
            result.append(Detection(max(0.0, x / w), max(0.0, y / h), min(1.0, (x + bw) / w),
                                    min(1.0, (y + bh) / h), min(1.0, score), "face", self.name))
        return result


class HaarFaceDetector:
    name = "haar"

    def __init__(self) -> None:
        cv2 = load_opencv()

        self._cv2 = cv2
        base = Path(cv2.data.haarcascades)
        self._frontal = cv2.CascadeClassifier(str(base / "haarcascade_frontalface_alt2.xml"))
        self._profile = cv2.CascadeClassifier(str(base / "haarcascade_profileface.xml"))
        if self._frontal.empty():
            raise RuntimeError("OpenCV frontal face cascade missing")

    def _run(self, cascade: Any, gray: Any, min_size: int, weight: float, flip: bool) -> list[Detection]:
        cv2 = self._cv2
        h, w = gray.shape[:2]
        source = cv2.flip(gray, 1) if flip else gray
        try:
            boxes, _levels, weights = cascade.detectMultiScale3(
                source, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size), outputRejectLevels=True)
        except cv2.error:
            return []
        result = []
        for (x, y, bw, bh), level_weight in zip(boxes if len(boxes) else [], weights if len(weights) else []):
            if flip:
                x = w - x - bw
            confidence = max(0.3, min(0.95, 0.35 + 0.08 * float(level_weight))) * weight
            result.append(Detection(x / w, y / h, (x + bw) / w, (y + bh) / h, confidence, "face", self.name))
        return result

    def detect(self, image: Any) -> list[Detection]:
        cv2 = self._cv2
        gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        min_size = max(12, int(gray.shape[0] * MIN_FACE_FRACTION))
        found = self._run(self._frontal, gray, min_size, 1.0, False)
        if not found and not self._profile.empty():
            found += self._run(self._profile, gray, min_size, 0.85, False)
            found += self._run(self._profile, gray, min_size, 0.85, True)
        return non_max_suppression(found)


def default_model_path() -> Path | None:
    configured = os.getenv("MIMIR_PRO_EDIT_FACE_MODEL", "").strip()
    for candidate in ([Path(configured)] if configured else []) + [MODEL_DIR / YUNET_FILENAME]:
        if candidate.is_file():
            return candidate
    return None


def load_detector() -> tuple[FaceDetector | None, str]:
    """Best available local detector, or (None, reason) - never raises."""
    try:
        load_opencv()
    except OpenCVUnavailable as error:  # exact cause + remedy (not installed / blocked / broken)
        return None, error.reason
    model = default_model_path()
    if model is not None:
        try:
            return YuNetFaceDetector(model), f"yunet:{model.name}"
        except Exception as error:  # corrupt model / old OpenCV -> fall back to Haar, reported
            reason = f"yunet unavailable ({type(error).__name__}); "
        else:
            reason = ""
    else:
        reason = "yunet model file not present; "
    try:
        return HaarFaceDetector(), reason + "haar cascades"
    except Exception as error:
        return None, reason + f"haar unavailable ({type(error).__name__}: {error})"
