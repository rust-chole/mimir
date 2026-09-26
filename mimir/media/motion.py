"""Motion and shot evidence from sampled frames (numpy/OpenCV, deterministic)."""
from __future__ import annotations

import statistics
from dataclasses import dataclass, replace
from typing import Sequence

import cv2
import numpy as np

from mimir.media.frames import SampledFrame

GRID_COLS = 12
GRID_ROWS = 8
SETTLE_WINDOW = (0.3, 1.5)   # seconds after a jump in which the old picture must NOT come back
SETTLED_RATIO = 0.6          # fraction of the cut threshold the settled distance must keep


@dataclass(frozen=True)
class MotionSample:
    t: float
    energy: float                   # mean abs luma difference to the previous sample (0..255)
    grid: np.ndarray                # (GRID_ROWS, GRID_COLS) mean abs difference per cell
    hist_distance: float            # HSV histogram distance to the previous sample (0..1)
    settled_distance: float = 1.0   # min distance between pictures before the jump and 0.3-1.5 s after it
                                    # (a cut persists; a flash/explosion returns). 1.0 = no later evidence


def _gray_small(image: np.ndarray, width: int = 160) -> np.ndarray:
    h, w = image.shape[:2]
    scale = width / float(w)
    small = cv2.resize(image, (width, max(2, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)


def color_hist(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
    return hist


def hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def settled_distances(times: Sequence[float], hists: Sequence[np.ndarray], jumps: Sequence[float]) -> list[float]:
    """For every sample with a histogram jump: the smallest distance between any picture shortly
    BEFORE the jump and any picture after it settled (SETTLE_WINDOW on both sides). A cut separates
    them all; a flash, explosion or strobe returns to the old picture and scores low."""
    out: list[float] = []
    count = len(times)
    lo, hi = SETTLE_WINDOW
    for index in range(count):
        if index == 0 or jumps[index] < 0.2:
            out.append(jumps[index])
            continue
        before = {index - 1}
        j = index - 2
        while j >= 0 and times[j] >= times[index] - hi:
            if times[j] <= times[index] - lo:
                before.add(j)
            j -= 1
        after = []
        for j in range(index, count):
            if times[j] > times[index] + hi:
                break
            if times[j] >= times[index] + lo:
                after.append(j)
        out.append(min(hist_distance(hists[b], hists[a]) for b in before for a in after) if after else 1.0)
    return out


def motion_series(frames: Sequence[SampledFrame]) -> list[MotionSample]:
    rows: list[MotionSample] = []
    hists: list[np.ndarray] = []
    previous_gray = None
    for frame in frames:
        gray = _gray_small(frame.image)
        hist = color_hist(frame.image)
        if previous_gray is None:
            grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
            rows.append(MotionSample(frame.t, 0.0, grid, 0.0))
        else:
            diff = cv2.absdiff(gray, previous_gray).astype(np.float32)
            grid = cv2.resize(diff, (GRID_COLS, GRID_ROWS), interpolation=cv2.INTER_AREA)
            rows.append(MotionSample(frame.t, float(diff.mean()), grid, hist_distance(hists[-1], hist)))
        previous_gray = gray
        hists.append(hist)
    settled = settled_distances([r.t for r in rows], hists, [r.hist_distance for r in rows])
    return [replace(row, settled_distance=value) for row, value in zip(rows, settled)]


def dedupe_cuts(cuts: Sequence[float], min_gap: float = 0.25) -> list[float]:
    """One cut per boundary when several detectors report it a sample apart."""
    out: list[float] = []
    for cut in sorted(cuts):
        if not out or cut - out[-1] >= min_gap:
            out.append(cut)
    return out


def shot_boundaries(samples: Sequence[MotionSample], *, min_gap: float = 0.6) -> list[float]:
    """Hard cuts: a histogram jump that clearly exceeds the local distribution AND persists
    (flashes, explosions and strobes return to the old picture and are not cuts)."""
    if len(samples) < 3:
        return []
    distances = [s.hist_distance for s in samples[1:]]
    median = statistics.median(distances)
    mad = statistics.median(abs(d - median) for d in distances) or 1e-3
    cuts: list[float] = []
    for index in range(1, len(samples)):
        sample = samples[index]
        threshold = max(0.35, median + 8.0 * mad)
        if (sample.hist_distance >= threshold and sample.energy >= 12.0
                and sample.settled_distance >= SETTLED_RATIO * threshold):
            t = (samples[index - 1].t + sample.t) / 2.0
            if not cuts or t - cuts[-1] >= min_gap:
                cuts.append(round(t, 3))
    return cuts


def visual_peaks(samples: Sequence[MotionSample], cuts: Sequence[float], *, max_peaks: int = 8,
                 min_separation: float = 0.9) -> list[dict[str, float]]:
    """Motion bursts (not caused by an edit cut) scored against the local baseline."""
    if len(samples) < 8:
        return []
    energies = np.array([s.energy for s in samples], dtype=np.float64)
    baseline = float(np.median(energies))
    mad = float(np.median(np.abs(energies - baseline))) or 0.5
    scored: list[tuple[float, int]] = []
    for index, sample in enumerate(samples):
        if any(abs(sample.t - cut) < 0.25 for cut in cuts):
            continue
        z = (sample.energy - baseline) / (1.4826 * mad)
        if z > 3.0:
            scored.append((z, index))
    scored.sort(reverse=True)
    peaks: list[dict[str, float]] = []
    for z, index in scored:
        t = samples[index].t
        if any(abs(t - p["t"]) < min_separation for p in peaks):
            continue
        peaks.append({"t": round(t, 3), "z": round(float(z), 2), "energy": round(samples[index].energy, 3),
                      "score": round(float(min(1.0, z / 12.0)), 3)})
        if len(peaks) >= max_peaks:
            break
    return sorted(peaks, key=lambda p: p["t"])
