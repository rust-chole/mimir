"""Scene drawing for synthetic VODs: real face imagery, mouth animation, gameplay, UI, IRL action."""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

W, H = 1920, 1080


@lru_cache(maxsize=1)
def _astronaut() -> np.ndarray:
    import skimage.data

    return cv2.cvtColor(skimage.data.astronaut(), cv2.COLOR_RGB2BGR)


# head-and-shoulders crop of the astronaut image and the mouth inside that crop
CROP = (110, 20, 340, 300)          # x0, y0, x1, y1 in the 512x512 image
MOUTH = (100, 120, 130, 132)        # x0, y0, x1, y1 inside the crop


@lru_cache(maxsize=16)
def person_image(variant: int, height: int) -> np.ndarray:
    x0, y0, x1, y1 = CROP
    image = _astronaut()[y0:y1, x0:x1].copy()
    if variant % 2 == 1:
        image = cv2.flip(image, 1)
    if variant >= 1:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + 25 * variant) % 180
        image = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    scale = height / image.shape[0]
    return cv2.resize(image, (int(round(image.shape[1] * scale)), height), interpolation=cv2.INTER_AREA)


@dataclass
class PersonPlacement:
    variant: int
    cx: float           # normalized head-crop center x
    top: float          # normalized top y
    height: float       # normalized crop height
    phase: float = 0.0
    sway: float = 1.0


def draw_person(canvas: np.ndarray, placement: PersonPlacement, t: float, openness: float,
                dx: float = 0.0) -> tuple[int, int, int, int]:
    h_px = int(placement.height * canvas.shape[0])
    image = person_image(placement.variant, h_px).copy()
    scale = h_px / (CROP[3] - CROP[1])
    if openness > 0:
        mx0, my0, mx1, my1 = (int(v * scale) for v in MOUTH)
        if placement.variant % 2 == 1:
            mx0, mx1 = image.shape[1] - mx1, image.shape[1] - mx0
        cx, cy = (mx0 + mx1) // 2, (my0 + my1) // 2
        axes = (max(2, int((mx1 - mx0) * 0.42)), max(1, int((my1 - my0) * 0.55 * openness)))
        cv2.ellipse(image, (cx, cy), axes, 0, 0, 360, (25, 15, 70), -1)
    sway_x = placement.sway * (0.010 * math.sin(2 * math.pi * 0.11 * t + placement.phase)
                               + 0.004 * math.sin(2 * math.pi * 0.37 * t))
    sway_y = placement.sway * 0.006 * math.sin(2 * math.pi * 0.23 * t + placement.phase)
    x = int((placement.cx + sway_x + dx) * canvas.shape[1] - image.shape[1] / 2)
    y = int((placement.top + sway_y) * canvas.shape[0])
    return paste(canvas, image, x, y)


def paste(canvas: np.ndarray, image: np.ndarray, x: int, y: int) -> tuple[int, int, int, int]:
    h, w = image.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = image[y0 - y:y1 - y, x0 - x:x1 - x]
    return x0, y0, x1, y1


def openness_at(words: list[dict], speaker: str, t: float, phase: float) -> float:
    for word in words:
        if word["speaker"] == speaker and word["start"] <= t <= word["end"]:
            return 0.35 + 0.65 * abs(math.sin(2 * math.pi * 5.5 * t + phase))
    return 0.0


@lru_cache(maxsize=4)
def room_background(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        canvas[y] = (40 + 30 * y / H, 48 + 20 * y / H, 70 + 25 * y / H)
    for _ in range(38):  # shelves / posters: structure for pixel proofs
        x, y = int(rng.integers(0, W - 120)), int(rng.integers(0, int(H * 0.55)))
        w, h = int(rng.integers(40, 180)), int(rng.integers(30, 140))
        color = tuple(int(c) for c in rng.integers(30, 200, 3))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (20, 20, 20), 2)
    cv2.rectangle(canvas, (0, int(H * 0.78)), (W, H), (35, 55, 60), -1)
    noise = rng.integers(-12, 12, (H, W, 1)).astype(np.int16)
    return np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def gameplay_frame(t: float, explosion_t: float) -> np.ndarray:
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = (30, 60, 25)
    offset = int(t * 220) % 120
    for x in range(-offset, W, 120):
        cv2.line(canvas, (x, 0), (x + 200, H), (50, 90, 45), 3)
    for y in range(0, H, 90):
        cv2.line(canvas, (0, y), (W, y), (45, 80, 40), 2)
    for k in range(9):  # enemies
        ex = int((0.5 + 0.42 * math.sin(0.7 * t + k * 1.3)) * W)
        ey = int((0.45 + 0.35 * math.cos(0.9 * t + k * 0.8)) * H)
        cv2.circle(canvas, (ex, ey), 26, (40, 40, 200), -1)
        cv2.circle(canvas, (ex, ey), 26, (10, 10, 10), 3)
    px = int((0.45 + 0.1 * math.sin(1.4 * t)) * W)
    cv2.rectangle(canvas, (px - 30, int(0.72 * H)), (px + 30, int(0.72 * H) + 70), (220, 200, 40), -1)
    cv2.putText(canvas, f"SCORE {int(t * 37):05d}", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
    cv2.putText(canvas, "HP |||||||||", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 255, 80), 3)
    dt = t - explosion_t
    if 0 <= dt < 1.4:
        radius = int(60 + 900 * dt)
        overlay = canvas.copy()
        cv2.circle(overlay, (int(0.55 * W), int(0.45 * H)), radius, (60, 200, 255), -1)
        cv2.circle(overlay, (int(0.55 * W), int(0.45 * H)), int(radius * 0.6), (255, 255, 255), -1)
        alpha = max(0.0, 1.0 - dt / 1.4)
        canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)
    return canvas


def ui_frame(t: float, error_t: float, lines: list[str]) -> np.ndarray:
    canvas = np.full((H, W, 3), 245, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (W, 70), (60, 60, 60), -1)
    cv2.putText(canvas, "project_manager.exe - Dashboard", (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (240, 240, 240), 2)
    cv2.rectangle(canvas, (0, 70), (360, H), (225, 228, 235), -1)
    for k, item in enumerate(["Overview", "Reports", "Invoices", "Settings", "Users", "Billing", "Export"]):
        cv2.putText(canvas, item, (40, 150 + 70 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2)
    scroll = int(t * 6) % 40
    for k, line in enumerate(lines):
        y = 140 + 52 * k - scroll
        if 90 < y < H - 20:
            cv2.putText(canvas, line, (420, y), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (30, 30, 30), 2)
    cx = int(900 + 300 * math.sin(0.6 * t))
    cy = int(500 + 180 * math.cos(0.45 * t))
    cv2.fillPoly(canvas, [np.array([[cx, cy], [cx + 18, cy + 44], [cx + 28, cy + 26]])], (0, 0, 0))
    if t >= error_t:
        cv2.rectangle(canvas, (560, 330), (1360, 720), (40, 40, 210), -1)
        cv2.rectangle(canvas, (560, 330), (1360, 720), (10, 10, 60), 6)
        cv2.putText(canvas, "FATAL ERROR", (700, 470), cv2.FONT_HERSHEY_DUPLEX, 2.4, (255, 255, 255), 5)
        cv2.putText(canvas, "All data deleted", (720, 580), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return canvas


def boxes_frame(canvas: np.ndarray, t: float, fall_t: float) -> None:
    """A tower of boxes on the right that topples and scatters at ``fall_t``."""
    base_x, base_y = int(0.72 * W), int(0.78 * H)
    colors = [(40, 120, 200), (60, 160, 230), (30, 90, 170), (80, 180, 240), (50, 140, 210)]
    for k in range(5):
        w, h = 170, 110
        x = base_x - w // 2
        y = base_y - (k + 1) * h
        dt = t - fall_t
        if dt > 0:
            fall = min(1.0, dt / 0.7)
            x += int((k + 1) * 140 * fall * (1 if k % 2 else -0.6))
            y = int(y + (base_y - h - y) * fall * fall)
            angle = 70 * fall * (1 if k % 2 else -1)
            box = cv2.boxPoints(((x + w / 2, y + h / 2), (w, h), angle)).astype(np.int32)
            cv2.fillPoly(canvas, [box], colors[k])
            cv2.polylines(canvas, [box], True, (20, 20, 20), 3)
        else:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), colors[k], -1)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (20, 20, 20), 3)
            cv2.line(canvas, (x, y + h // 2), (x + w, y + h // 2), (20, 60, 100), 2)
