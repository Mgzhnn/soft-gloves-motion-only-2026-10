"""Small current-region comparator; descriptors never become masks or physical forces."""
from itertools import product
import numpy as np


class RGBHistogram:
    scheme = "masked_rgb_4x4x4_l1/v1"

    def describe(self, rgb, mask):
        if rgb.ndim != 3 or rgb.shape[2] != 3 or mask.shape != rgb.shape[:2] or mask.dtype != bool:
            raise ValueError("RGB image and boolean visible-region mask required")
        if rgb.dtype != np.uint8:
            raise ValueError("RGB uint8 0..255 required")
        pixels = rgb[mask]
        if not len(pixels):
            return None
        bins = pixels.astype(np.int64) // 64
        indices = bins[:, 0] * 16 + bins[:, 1] * 4 + bins[:, 2]
        hist = np.bincount(indices, minlength=64).astype(float)
        return hist / hist.sum()


class DINOv3Boundary:
    scheme = "dinov3_vitb16_masked_patch_l2_PROPOSED"

    def describe(self, rgb, mask):
        raise NotImplementedError("Interface only: gated weights, preprocessing and runtime not admitted")


class MissingObjectProvider:
    def observe(self, rgb, stamp):
        return ()  # No fabricated object observations from an image.


def global_assignment(costs, unmatched_cost=1.0):
    """Exact tiny assignment with no-match; <=4 objects is a fixture bound, not a lab limit."""
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 2 or max(costs.shape, default=0) > 4 or np.isnan(costs).any():
        raise ValueError("finite-or-infinity cost matrix with at most four rows/columns required")
    if not np.isfinite(unmatched_cost) or unmatched_cost < 0 or (costs < 0).any():
        raise ValueError("nonnegative costs required")
    choices = (None,) + tuple(range(costs.shape[1]))  # equal cost favours explicit no-match
    best = None
    best_cost = float("inf")
    for assignment in product(choices, repeat=costs.shape[0]):
        selected = [j for j in assignment if j is not None]
        if len(set(selected)) != len(selected):
            continue
        cost = sum(unmatched_cost if j is None else costs[i, j] for i, j in enumerate(assignment))
        if cost < best_cost:
            best, best_cost = assignment, cost
    return best, best_cost
