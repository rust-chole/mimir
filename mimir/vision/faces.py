"""Local face detectors behind one interface.

YuNet (bundled ONNX model, verified by SHA-256) is the production detector: on the
ground-truth benchmark it found 95% of faces (small facecams, several people, turned,
tilted, edge-cut and partly covered faces) against 44% for the OpenCV Haar cascades,
with no false positives and ~17x less time per frame. Haar remains only as a recorded
fallback when the model file is missing or cannot be loaded.

Detectors only report normalized boxes; they never decide camera behavior.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "models"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
MIN_FACE_FRACTION = 0.035


@dataclass(frozen=True)
class Detection:
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float
    detector: str = ""

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


class FaceDetector(Protocol):
    name: str

    def detect(self, image: np.ndarray) -> list[Detection]:
        ...


def iou(a: Detection, b: Detection) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def nms(detections: Sequence[Detection], threshold: float = 0.35) -> list[Detection]:
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: -d.confidence):
        if all(iou(det, k) < threshold for k in kept):
            kept.append(det)
    return kept


class YuNetDetector:
    name = "yunet"

    def __init__(self, model: Path, score: float = 0.6) -> None:
        self._net = cv2.FaceDetectorYN.create(str(model), "", (320, 320), score, 0.3, 50)
        self._size: tuple[int, int] | None = None

    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        if self._size != (w, h):
            self._net.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self._net.detect(image)
        rows = []
        for row in faces if faces is not None else []:
            x, y, bw, bh = (float(v) for v in row[:4])
            if bh / h < MIN_FACE_FRACTION:
                continue
            rows.append(Detection(max(0.0, x / w), max(0.0, y / h), min(1.0, (x + bw) / w), min(1.0, (y + bh) / h),
                                  min(1.0, float(row[-1])), self.name))
        return rows


class HaarDetector:
    name = "haar"

    def __init__(self) -> None:
        base = Path(cv2.data.haarcascades)
        self._frontal = cv2.CascadeClassifier(str(base / "haarcascade_frontalface_alt2.xml"))
        self._profile = cv2.CascadeClassifier(str(base / "haarcascade_profileface.xml"))
        if self._frontal.empty():
            raise RuntimeError("OpenCV frontal face cascade missing")

    def _run(self, cascade, gray: np.ndarray, min_size: int, weight: float, flip: bool) -> list[Detection]:
        h, w = gray.shape[:2]
        source = cv2.flip(gray, 1) if flip else gray
        try:
            boxes, _levels, weights = cascade.detectMultiScale3(source, scaleFactor=1.1, minNeighbors=5,
                                                                 minSize=(min_size, min_size), outputRejectLevels=True)
        except cv2.error:
            return []
        rows = []
        for (x, y, bw, bh), level in zip(boxes if len(boxes) else [], weights if len(weights) else []):
            if flip:
                x = w - x - bw
            confidence = max(0.3, min(0.95, 0.35 + 0.08 * float(level))) * weight
            rows.append(Detection(x / w, y / h, (x + bw) / w, (y + bh) / h, confidence, self.name))
        return rows

    def detect(self, image: np.ndarray) -> list[Detection]:
        gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        min_size = max(12, int(gray.shape[0] * MIN_FACE_FRACTION))
        found = self._run(self._frontal, gray, min_size, 1.0, False)
        if not found and not self._profile.empty():
            found += self._run(self._profile, gray, min_size, 0.85, False)
            found += self._run(self._profile, gray, min_size, 0.85, True)
        return nms(found)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_detector() -> tuple[FaceDetector, str]:
    """(detector, fallback reason). The reason is empty when YuNet is in use; callers record it."""
    configured = os.getenv("MIMIR_FACE_MODEL", "").strip()
    if configured:
        path = Path(configured)
        if not path.is_file():
            return HaarDetector(), f"MIMIR_FACE_MODEL {configured} does not exist"
    else:
        path = MODEL_DIR / YUNET_FILE
        if not path.is_file():
            return HaarDetector(), f"bundled face model {path} is missing"
        if _sha256(path) != YUNET_SHA256:
            return HaarDetector(), f"bundled face model {path} failed its SHA-256 check"
    try:
        return YuNetDetector(path), ""
    except cv2.error as error:
        return HaarDetector(), f"YuNet could not be loaded ({str(error).strip()[:120]})"


def detector_fingerprint() -> str:
    """Identity of the detector ``load_detector`` will use (part of the vision cache key)."""
    configured = os.getenv("MIMIR_FACE_MODEL", "").strip()
    path = Path(configured) if configured else MODEL_DIR / YUNET_FILE
    if path.is_file():
        return f"yunet:{_sha256(path)[:16]}"
    return f"haar:opencv-{cv2.__version__}"
