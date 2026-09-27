"""Per-frame compositor: render-plan state -> output frame (OpenCV, sub-pixel exact).

Window states map a source-normalized window onto the whole output with one
affine resample; area outside the source is filled with a blurred, darkened
cover-scaled copy of the same frame. Stack states render the facecam panel on
top and the gameplay window below. Captions are NOT drawn here (libass burns
them after this layer, so captions are never zoomed or cropped).
"""
from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

BG_DARKEN = 0.55
BG_SCALE = 0.25
SEAM_PX = 6
FLASH_PEAK = 0.65


class Compositor:
    def __init__(self, src_w: int, src_h: int, out_w: int, out_h: int, stack_split: float) -> None:
        self.src_w, self.src_h, self.out_w, self.out_h = src_w, src_h, out_w, out_h
        self.split_px = int(round(out_h * stack_split))
        self.out_aspect = out_w / out_h

    # ---------------------------------------------------------------- window

    def _window_matrix(self, cx: float, cy: float, h: float, out_w: int, out_h: int) -> tuple[np.ndarray, float,
                                                                                                tuple[float, float]]:
        aspect = out_w / out_h
        win_h = h * self.src_h
        win_w = win_h * aspect
        x0 = cx * self.src_w - win_w / 2
        y0 = cy * self.src_h - win_h / 2
        scale = out_h / win_h
        matrix = np.array([[scale, 0.0, -scale * x0], [0.0, scale, -scale * y0]], dtype=np.float64)
        return matrix, scale, (x0, y0)

    def _warp(self, image: np.ndarray, matrix: np.ndarray, scale: float, size: tuple[int, int]) -> np.ndarray:
        if scale < 0.75:  # downscale with an area filter first to avoid aliasing
            factor = min(1.0, scale * 1.25)
            small = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            m = matrix.copy()
            m[:, :2] /= factor
            return cv2.warpAffine(small, m, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return cv2.warpAffine(image, matrix, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def _background(self, image: np.ndarray, cx: float, cy: float, size: tuple[int, int]) -> np.ndarray:
        """Blurred, darkened cover-scaled copy of the frame (blurred at low resolution: cheap)."""
        out_w, out_h = size
        small = cv2.resize(image, None, fx=BG_SCALE, fy=BG_SCALE, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigmaX=6.0)
        sh, sw = small.shape[:2]
        cover = max(out_w / sw, out_h / sh) * 1.08
        x0 = min(max(cx * sw - out_w / cover / 2, 0.0), max(0.0, sw - out_w / cover))
        y0 = min(max(cy * sh - out_h / cover / 2, 0.0), max(0.0, sh - out_h / cover))
        matrix = np.array([[cover, 0.0, -cover * x0], [0.0, cover, -cover * y0]])
        bg = cv2.warpAffine(small, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return cv2.convertScaleAbs(bg, alpha=BG_DARKEN)

    def window(self, image: np.ndarray, cx: float, cy: float, h: float, out_w: int, out_h: int) -> np.ndarray:
        matrix, scale, (x0, y0) = self._window_matrix(cx, cy, h, out_w, out_h)
        fg = self._warp(image, matrix, scale, (out_w, out_h))
        rx0, ry0 = -scale * x0, -scale * y0
        rx1, ry1 = scale * (self.src_w - x0), scale * (self.src_h - y0)
        if rx0 <= 0.5 and ry0 <= 0.5 and rx1 >= out_w - 0.5 and ry1 >= out_h - 0.5:
            return fg
        out = self._background(image, cx, cy, (out_w, out_h))
        ix0, iy0 = max(0, int(np.ceil(rx0))), max(0, int(np.ceil(ry0)))
        ix1, iy1 = min(out_w, int(np.floor(rx1))), min(out_h, int(np.floor(ry1)))
        if ix1 > ix0 and iy1 > iy0:
            out[iy0:iy1, ix0:ix1] = fg[iy0:iy1, ix0:ix1]
        return out

    # ----------------------------------------------------------------- stack

    def stack(self, image: np.ndarray, top: Sequence[float], bottom: Sequence[float]) -> np.ndarray:
        out = np.empty((self.out_h, self.out_w, 3), dtype=np.uint8)
        x0, y0, x1, y1 = top
        box_w, box_h = (x1 - x0) * self.src_w, (y1 - y0) * self.src_h
        scale_x, scale_y = self.out_w / box_w, self.split_px / box_h
        matrix = np.array([[scale_x, 0.0, -scale_x * x0 * self.src_w], [0.0, scale_y, -scale_y * y0 * self.src_h]])
        out[:self.split_px] = self._warp(image, matrix, min(scale_x, scale_y), (self.out_w, self.split_px))
        cx, cy, h = bottom
        out[self.split_px:] = self.window(image, cx, cy, h, self.out_w, self.out_h - self.split_px)
        half = SEAM_PX // 2
        out[max(0, self.split_px - half):self.split_px + half] = (18, 18, 18)
        return out

    # ----------------------------------------------------------------- frame

    def frame(self, image: np.ndarray, layout: int, window: Sequence[float], top: Sequence[float] | None
              ) -> np.ndarray:
        if layout == 1 and top is not None:
            return self.stack(image, top, window)
        cx, cy, h = window
        return self.window(image, cx, cy, h, self.out_w, self.out_h)


def flash_alpha(frame: int, center: int, frames: int) -> float:
    half = max(1, frames // 2 + 1)
    distance = abs(frame - center)
    return FLASH_PEAK * max(0.0, 1.0 - distance / half) if distance < half else 0.0


def apply_flash(image: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return image
    return cv2.addWeighted(image, 1.0 - alpha, np.full_like(image, 255), alpha, 0.0)


def top_boxes(plan: dict[str, Any]) -> dict[int, list[float]]:
    boxes: dict[int, list[float]] = {}
    for run in plan.get("stack_tops", []):
        for index in range(run["start"], run["end"]):
            boxes[index] = run["box"]
    return boxes
