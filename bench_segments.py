"""Diagnostic colour segmentation for the bench clips: orange gloves, one bluish-grey object.

These are clip-specific HSV thresholds plus nearest-centroid identity tracking.
They are diagnostic inputs that let real frames flow through the replay before a
trained detector exists; they are not the reference's functional surface points
and carry no validated accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ORANGE_HSV = ((8, 120, 120), (30, 255, 255))
OBJECT_HSV = ((90, 35, 30), (135, 255, 255))  # bluish-grey band: caught the metal can in smoke_03; also matches blue objects and the dark glove cuffs
GLOVE_MIN_AREA = 1500
OBJECT_MIN_AREA = 300
GLOVE_GATE_PX = 150.0
OBJECT_GATE_PX = 60.0
OBJECT_CLEARANCE_FROM_GLOVE_PX = 100.0


@dataclass(frozen=True)
class Segment:
    area: int
    centroid: tuple[float, float]  # column, row
    mask: np.ndarray


def _cv2():
    import cv2  # imported lazily so the pure helpers stay importable without OpenCV
    return cv2


def hsv_of(rgb: np.ndarray) -> np.ndarray:
    return _cv2().cvtColor(np.ascontiguousarray(rgb), _cv2().COLOR_RGB2HSV)


def segments(hsv: np.ndarray, bounds, min_area: int, open_kernel: int = 3) -> list[Segment]:
    cv2 = _cv2()
    mask = cv2.inRange(hsv, np.array(bounds[0], np.uint8), np.array(bounds[1], np.uint8))
    if open_kernel:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_kernel, open_kernel), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    out = []
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            out.append(Segment(area, (float(centroids[label][0]), float(centroids[label][1])), labels == label))
    return sorted(out, key=lambda s: -s.area)


def distance(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def nearest_valid_pixel(depth: np.ndarray, mask: np.ndarray, centre, invalid=0) -> tuple[int, int] | None:
    """Integer (column,row) inside the mask with valid depth, nearest to the centroid; None if the mask has none."""
    rows, cols = np.nonzero(mask & (depth != invalid))
    if rows.size == 0:
        return None
    k = int(np.argmin((cols - centre[0]) ** 2 + (rows - centre[1]) ** 2))
    return int(cols[k]), int(rows[k])


def assign_hands(gloves: list[Segment], top_hand: str) -> dict[str, Segment]:
    """First-frame identity: the higher glove in the image is `top_hand`; needs exactly two gloves."""
    if top_hand not in ("left", "right") or len(gloves) < 2:
        return {}
    top, bottom = sorted(gloves[:2], key=lambda s: s.centroid[1])
    other = "right" if top_hand == "left" else "left"
    return {top_hand: top, other: bottom}


class IdentityTracker:
    """Nearest-centroid persistence with a gate; a lost identity stays lost until re-acquired inside the gate."""

    def __init__(self, gate_px: float):
        self.gate = gate_px
        self.last: dict[str, tuple[float, float]] = {}

    def seed(self, assignments: dict[str, Segment]) -> None:
        self.last = {identity: seg.centroid for identity, seg in assignments.items()}

    def match(self, candidates: list[Segment]) -> dict[str, Segment | None]:
        remaining = list(candidates)
        result: dict[str, Segment | None] = {}
        for identity, previous in sorted(self.last.items()):
            best = min(remaining, key=lambda s: distance(s.centroid, previous), default=None)
            if best is not None and distance(best.centroid, previous) <= self.gate:
                result[identity] = best
                remaining.remove(best)
                self.last[identity] = best.centroid
            else:
                result[identity] = None
        return result


def pick_initial_object(candidates: list[Segment], glove_centroids, clearance_px: float = OBJECT_CLEARANCE_FROM_GLOVE_PX) -> Segment | None:
    """Largest object-band segment that is not adjacent to a glove (the glove cuffs fall in the same band)."""
    for seg in candidates:
        if all(distance(seg.centroid, g) >= clearance_px for g in glove_centroids):
            return seg
    return None
