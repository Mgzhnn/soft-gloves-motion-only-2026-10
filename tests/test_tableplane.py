"""Table-plane geometry: synthetic depth scenes, participant N=0."""
import numpy as np
import pytest

from sg_offline.rgbd import CameraCalibration
from sg_offline.tableplane import backproject, clusters, fit_table_plane, region_geometry, table_support, table_surface

CAL = CameraCalibration("test", 160, 120, 150., 150., 80., 60., .001, "cam")


def synthetic_scene(tilt_deg=35., box_height=.10, arm=True):
    """A table seen from a head camera tilted down by `tilt_deg`, a box standing on it, and an arm entering from the bottom.
    Returns (depth uint16 in mm, box pixel mask, arm pixel mask, true plane normal and offset)."""
    t = np.deg2rad(tilt_deg)
    # Camera axes: x right, y down, z forward; the camera looks down by `tilt_deg`. World "up" in camera
    # coordinates is then n = (0, -cos t, -sin t); the table lies 0.8 m below the camera along n, so its
    # points satisfy n.p + d = 0 with d = 0.8, and the camera origin has signed height d > 0 (above the table).
    n = np.array([0., -np.cos(t), -np.sin(t)])
    d = 0.8
    u = np.arange(CAL.width)[None, :]; v = np.arange(CAL.height)[:, None]
    dirs = np.dstack([(u - CAL.cx) / CAL.fx * np.ones_like(v, dtype=float), (v - CAL.cy) / CAL.fy * np.ones_like(u, dtype=float), np.ones((CAL.height, CAL.width))])
    denom = dirs @ n  # negative where the ray goes down toward the table

    def hit(offset):  # ray p = s*dir meets the plane n.p + offset = 0 -> s = -offset / (n.dir); z = s because dir_z = 1
        return np.where(denom < -1e-6, -offset / denom, np.nan)

    box = np.zeros((CAL.height, CAL.width), bool); box[50:80, 60:90] = True
    arm = np.zeros_like(box); arm[85:120, 100:125] = True
    z = hit(d)
    z = np.where(box, hit(d - box_height), z)   # the box top is a parallel plane `box_height` above the table
    z = np.where(arm, hit(d - .05), z)          # the arm is a slab 5 cm above the table, reaching the bottom edge
    depth = np.where(np.isfinite(z) & (z > 0), np.round(z * 1000), 0).astype(np.uint16)
    return depth, box, arm, n, d


def test_backproject_matches_the_pinhole_model_and_flags_invalid_depth():
    depth = np.full((CAL.height, CAL.width), 1000, np.uint16); depth[0, 0] = 0
    pts, valid = backproject(depth, CAL)
    assert pts.shape == (CAL.height, CAL.width, 3) and not valid[0, 0] and valid[60, 80]
    assert pts[60, 80] == pytest.approx((0., 0., 1.)) and pts[60, 95] == pytest.approx(((95 - 80) / 150., 0., 1.))
    with pytest.raises(ValueError):
        backproject(depth[:-1], CAL)


def test_ransac_recovers_the_table_plane_deterministically_and_orients_it_toward_the_camera():
    depth, box, arm, n_true, d_true = synthetic_scene()
    pts, valid = backproject(depth, CAL)
    plane = fit_table_plane(pts, valid, seed=7)
    assert plane is not None and plane.inlier_fraction > .6 and plane.rmse_m < .002
    assert abs(float(np.asarray(plane.normal) @ n_true)) > .999 and plane.offset == pytest.approx(d_true, abs=.01)
    assert plane.offset > 0 and plane.height(np.zeros(3)) > 0           # the camera is on the positive side
    assert 50 < plane.tilt_deg < 60   # normal vs optical axis: 90 - 35 degrees (0 would be looking straight down)
    again = fit_table_plane(pts, valid, seed=7)
    assert again == plane                                                # same seed, same result
    assert fit_table_plane(pts, valid, seed=8).inlier_fraction > .6
    assert fit_table_plane(pts, np.zeros_like(valid), seed=1) is None   # nothing valid
    with pytest.raises(ValueError):
        fit_table_plane(pts, valid, seed="7")


def test_table_support_and_clusters_find_the_box_and_drop_the_arm():
    depth, box, arm, _, _ = synthetic_scene()
    pts, valid = backproject(depth, CAL)
    plane = fit_table_plane(pts, valid, seed=3)
    support = table_support(pts, valid, plane, min_height_m=.015)
    assert support[box].mean() > .9 and support[~(box | arm)].mean() < .02
    surface = table_surface(pts, valid, plane)
    assert surface[~(box | arm)].mean() > .9 and not surface[box].any()
    found = clusters(support, min_area=50, surface=surface, points=pts, max_range_m=2.)
    kinds = {c["excluded"]: c for c in found}
    assert "" in kinds and kinds[""]["area"] == pytest.approx(box.sum(), rel=.15)
    assert kinds[""]["ring_surface_fraction"] > .8 and kinds[""]["median_range_m"] < 2.
    assert "touches_bottom_border_arm" in kinds                          # the arm reaches the bottom edge and is dropped
    hand_box = (55, 45, 95, 85)                                          # a tracked hand box over the object: excluded for the comparator
    assert clusters(support, min_area=50, exclude_boxes=[hand_box])[0]["excluded"] in ("inside_hand_box", "touches_bottom_border_arm")
    # far field: an above-plane blob touching the top border, and a floating blob not surrounded by table
    far = support.copy(); far[0:20, 10:60] = True
    assert "touches_top_border_far_field" in {c["excluded"] for c in clusters(far, min_area=50, surface=surface)}
    floating = support.copy(); floating[30:45, 120:150] = True; no_table = surface.copy(); no_table[22:53, 112:158] = False
    assert "not_on_table_surface" in {c["excluded"] for c in clusters(floating, min_area=50, surface=no_table)}
    assert all(c["excluded"] != "beyond_max_range" for c in found) and any(c["excluded"] == "beyond_max_range" for c in clusters(support, min_area=50, points=pts, max_range_m=.5))
    with pytest.raises(ValueError):
        table_support(pts, valid, plane, min_height_m=.5, max_height_m=.1)
    with pytest.raises(ValueError):
        table_surface(pts, valid, plane, tolerance_m=0.)


def test_region_geometry_reports_height_centroid_extent_and_a_valid_pixel():
    depth, box, arm, _, _ = synthetic_scene(box_height=.10)
    pts, valid = backproject(depth, CAL)
    plane = fit_table_plane(pts, valid, seed=3)
    geo = region_geometry(pts, valid, plane, box)
    assert geo is not None and geo["n_points"] == box.sum()
    assert .07 < geo["height_m"] < .13 and geo["footprint_extent_m"] > .05
    px = geo["pixel"]; assert box[px[1], px[0]] and valid[px[1], px[0]]
    assert region_geometry(pts, valid, plane, np.zeros_like(box)) is None
    table_only = ~(box | arm); table_only[:40] = False
    assert region_geometry(pts, valid, plane, table_only) is None       # points ON the table have no positive height
