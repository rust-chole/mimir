"""Pixel evidence: structure-only comparison robust to encoding noise and grading."""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

ANALYSIS_WIDTH = 270


def structure(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    small = cv2.resize(image, (ANALYSIS_WIDTH, int(round(h * ANALYSIS_WIDTH / w))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.GaussianBlur(gray, (0, 0), 1.0) - cv2.GaussianBlur(gray, (0, 0), 4.0)


def row_mask(height: int, bands: Sequence[tuple[float, float]]) -> np.ndarray:
    mask = np.ones(height, dtype=bool)
    for a, b in bands:
        mask[int(a * height):int(np.ceil(b * height))] = False
    return mask


def ncc(a: np.ndarray, b: np.ndarray, mask_bands: Sequence[tuple[float, float]] = ()) -> float:
    sa, sb = structure(a), structure(b)
    rows = row_mask(sa.shape[0], mask_bands)
    x, y = sa[rows].ravel(), sb[rows].ravel()
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denominator < 1e-6:
        return 0.0
    return float((x * y).sum() / denominator)


def texture(image: np.ndarray) -> float:
    return float(structure(image).std())


def band_ink(rendered: np.ndarray, predicted: np.ndarray, band: tuple[float, float], threshold: float = 40.0) -> float:
    """Fraction of band pixels whose luma differs strongly from the caption-free prediction (burned text)."""
    h = rendered.shape[0]
    a, b = int(band[0] * h), int(np.ceil(band[1] * h))
    r = cv2.cvtColor(rendered[a:b], cv2.COLOR_BGR2GRAY).astype(np.int16)
    p = cv2.cvtColor(predicted[a:b], cv2.COLOR_BGR2GRAY).astype(np.int16)
    return float((np.abs(r - p) > threshold).mean())
