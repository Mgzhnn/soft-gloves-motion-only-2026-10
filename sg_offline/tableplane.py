"""Table-plane geometry from registered depth: back-projection, seeded RANSAC plane, height above the table.

The reference removed the table with RANSAC and clustered the remaining points. This module provides both
halves of proposed (5 October 2026): the plane-only comparator (``clusters`` on the above-table mask)
and the metric geometry a segmenter-based provider needs (3-D points inside a mask, height above the plane,
centroid and extent). Everything is deterministic: RANSAC samples come from a generator seeded by the caller
(use the frame index), so a replay reproduces bit for bit. No learned component lives here.

Conventions: points are in the colour-camera optical frame of the registered depth (x right, y down, z forward,
metres). The plane is stored as a unit normal ``n`` and offset ``d`` with signed height ``h(p) = n.p + d``,
oriented so that the camera origin has positive height; objects standing on the table therefore have positive
height and points below the table (noise, holes in the surface) negative height.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees

import numpy as np

from .rgbd import CameraCalibration


def backproject(depth, calibration: CameraCalibration, invalid=0):
    """(H, W, 3) metric points and an (H, W) validity mask from a registered depth image."""
    depth = np.asarray(depth)
    if depth.shape != (calibration.height, calibration.width):
        raise ValueError("depth resolution differs from the calibration")
    z = depth.astype(np.float64) * calibration.depth_scale_m
    valid = (depth != invalid) & np.isfinite(z) & (z > 0)
    u = np.arange(calibration.width, dtype=np.float64)[None, :]
    v = np.arange(calibration.height, dtype=np.float64)[:, None]
    x = (u - calibration.cx) * z / calibration.fx
    y = (v - calibration.cy) * z / calibration.fy
    return np.dstack([x, y, z]), valid


@dataclass(frozen=True)
class Plane:
    normal: tuple[float, float, float]
    offset: float
    inliers: int
    samples: int
    inlier_fraction: float
    rmse_m: float
    tilt_deg: float  # angle between the plane normal and the optical axis (0 = camera looks straight down at the table)
    seed: int

    def height(self, points):
        """Signed height above the plane, positive toward the camera, for points of shape (..., 3)."""
        return np.asarray(points) @ np.asarray(self.normal) + self.offset


def _least_squares_plane(points):
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1]
    return normal / np.linalg.norm(normal), -float(normal @ centroid) / np.linalg.norm(normal)


def fit_table_plane(points, valid, *, seed: int, iterations=200, inlier_m=0.01, max_samples=4000,
                    min_inlier_fraction=0.3, region=None) -> Plane | None:
    """Seeded RANSAC plane on a subsample of valid points (optionally within ``region``), refined by least squares.

    Returns None when fewer than three points are available or the best plane explains less than
    ``min_inlier_fraction`` of the sampled points. The settings are batch settings; record them with the run.
    """
    if type(seed) is not int:
        raise ValueError("an integer seed is required so the fit is reproducible")
    if iterations < 1 or inlier_m <= 0 or max_samples < 3 or not 0 < min_inlier_fraction <= 1:
        raise ValueError("invalid RANSAC settings")
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    if region is not None:
        mask = mask & np.asarray(region, dtype=bool).reshape(-1)
    candidates = pts[mask]
    if len(candidates) < 3:
        return None
    rng = np.random.default_rng(seed)
    if len(candidates) > max_samples:
        candidates = candidates[rng.choice(len(candidates), max_samples, replace=False)]
    n = len(candidates)
    best_count, best_inliers = 0, None
    triples = rng.integers(0, n, size=(iterations, 3))
    for a, b, c in triples:
        if a == b or b == c or a == c:
            continue
        p0, p1, p2 = candidates[a], candidates[b], candidates[c]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        offset = -float(normal @ p0)
        inliers = np.abs(candidates @ normal + offset) < inlier_m
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers
    if best_inliers is None or best_count < 3 or best_count / n < min_inlier_fraction:
        return None
    normal, offset = _least_squares_plane(candidates[best_inliers])
    if offset < 0:  # orient so the camera origin (height = offset) is on the positive side
        normal, offset = -normal, -offset
    residuals = candidates[best_inliers] @ normal + offset
    tilt = degrees(acos(min(1., abs(float(normal[2])))))
    return Plane(tuple(float(v) for v in normal), float(offset), best_count, n, best_count / n,
                 float(np.sqrt(np.mean(residuals ** 2))), float(tilt), seed)


def table_support(points, valid, plane: Plane, min_height_m=0.015, max_height_m=0.6):
    """Valid points standing above the table within a height band: the candidates for objects and hands."""
    if not 0 <= min_height_m < max_height_m:
        raise ValueError("height band must satisfy 0 <= min < max")
    h = plane.height(points)
    return np.asarray(valid, dtype=bool) & (h > min_height_m) & (h < max_height_m)


def table_surface(points, valid, plane: Plane, tolerance_m=0.015):
    """Valid points lying on the table itself (within ``tolerance_m`` of the plane)."""
    if tolerance_m <= 0:
        raise ValueError("tolerance must be positive")
    return np.asarray(valid, dtype=bool) & (np.abs(plane.height(points)) <= tolerance_m)


def _box_overlap_fraction(component_mask, box):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    h, w = component_mask.shape
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        return 0.
    inside = int(component_mask[y0:y1, x0:x1].sum())
    return inside / max(int(component_mask.sum()), 1)


def clusters(support, *, min_area=200, exclude_boxes=(), box_overlap=0.5, drop_bottom_border=True, drop_top_border=True,
             border_px=2, surface=None, ring_px=7, min_ring_surface_fraction=0.2, points=None, max_range_m=None):
    """Connected components of the above-table mask, with arm, hand and far-field removal (the plane-only comparator).

    Being above the table plane is not the same as standing on the table, so components are excluded, with the
    reason recorded, when they: touch the bottom image border (the operator's arms entering the view); touch the top
    border (far field beyond the table edge); lie more than ``box_overlap`` inside an excluded box (tracked hands);
    have a median range beyond ``max_range_m`` (``points`` required); or are not surrounded by table surface, i.e.
    fewer than ``min_ring_surface_fraction`` of the pixels in a ``ring_px`` ring around the component are on the
    plane (``surface`` from ``table_surface``). All thresholds are batch settings to record with the run.
    Returns a list of dicts sorted by area, largest first; ``excluded`` is "" for kept components.
    """
    import cv2
    mask = np.asarray(support, dtype=np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    h = mask.shape[0]
    kernel = np.ones((2 * ring_px + 1, 2 * ring_px + 1), np.uint8) if surface is not None else None
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = labels == i
        top = int(stats[i, cv2.CC_STAT_TOP]) <= border_px
        bottom = int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]) >= h - border_px
        reason = ""
        ring_fraction = None
        median_range = None
        if points is not None:
            median_range = float(np.median(np.asarray(points)[component][:, 2]))
        if drop_bottom_border and bottom:
            reason = "touches_bottom_border_arm"
        elif drop_top_border and top:
            reason = "touches_top_border_far_field"
        elif any(_box_overlap_fraction(component, b) > box_overlap for b in exclude_boxes):
            reason = "inside_hand_box"
        elif max_range_m is not None and median_range is not None and median_range > max_range_m:
            reason = "beyond_max_range"
        if surface is not None:
            ring = cv2.dilate(component.astype(np.uint8), kernel).astype(bool) & ~component
            ring_fraction = float(np.asarray(surface, dtype=bool)[ring].mean()) if ring.any() else 0.
            if not reason and ring_fraction < min_ring_surface_fraction:
                reason = "not_on_table_surface"
        out.append(dict(mask=component, area=area, centroid_px=(float(centroids[i][0]), float(centroids[i][1])),
                        touches_bottom=bottom, touches_top=top, ring_surface_fraction=ring_fraction,
                        median_range_m=median_range, excluded=reason))
    return sorted(out, key=lambda c: -c["area"])


def region_geometry(points, valid, plane: Plane, mask, min_points=20, min_height_m=0.015):
    """Metric summary of the valid above-table points inside ``mask``: centroid, height, extent, a representative pixel.

    Only points higher than ``min_height_m`` above the plane count (the same band as ``table_support``), so
    table pixels inside a loose mask do not pull the centroid down. Returns None when fewer than ``min_points``
    such points lie inside the mask. The pixel is the qualifying in-mask pixel nearest to the mask centroid.
    """
    if min_height_m < 0:
        raise ValueError("minimum height must be nonnegative")
    region = np.asarray(mask, dtype=bool) & np.asarray(valid, dtype=bool)
    h = plane.height(points)
    region &= h > min_height_m
    rows, cols = np.nonzero(region)
    if rows.size < min_points:
        return None
    xyz = np.asarray(points)[rows, cols]
    heights = h[rows, cols]
    centroid = xyz.mean(axis=0)
    n = np.asarray(plane.normal)
    # extent along the plane: project onto the plane and take the bounding size of the footprint
    footprint = xyz - np.outer(xyz @ n + plane.offset, n)
    spread = footprint - footprint.mean(axis=0)
    _, s, _ = np.linalg.svd(spread, full_matrices=False)
    cy, cx = rows.mean(), cols.mean()
    k = int(np.argmin((cols - cx) ** 2 + (rows - cy) ** 2))
    return dict(centroid_xyz=tuple(float(v) for v in centroid), n_points=int(rows.size),
                height_m=float(np.percentile(heights, 95)), footprint_extent_m=float(2 * s[0] / np.sqrt(max(rows.size, 1))),
                pixel=(int(cols[k]), int(rows[k])))
