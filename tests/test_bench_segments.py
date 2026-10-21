"""Pure helpers of the diagnostic bench segmentation; OpenCV-dependent parts are exercised by the runner."""
import numpy as np
import pytest

import bench_segments as seg


def _segment(cx, cy, area=2000):
    mask = np.zeros((480, 640), dtype=bool)
    mask[int(cy) - 10:int(cy) + 10, int(cx) - 10:int(cx) + 10] = True
    return seg.Segment(area, (float(cx), float(cy)), mask)


def test_assign_hands_uses_image_height_and_operator_side():
    top, bottom = _segment(360, 112), _segment(342, 383)
    assert seg.assign_hands([bottom, top], "right") == {"right": top, "left": bottom}
    assert seg.assign_hands([bottom, top], "left") == {"left": top, "right": bottom}
    assert seg.assign_hands([top], "left") == {} and seg.assign_hands([top, bottom], "up") == {}


def test_tracker_keeps_identity_inside_gate_and_reports_loss_outside():
    tracker = seg.IdentityTracker(gate_px=150.0)
    tracker.seed({"right": _segment(360, 112), "left": _segment(342, 383)})
    moved = tracker.match([_segment(342, 384), _segment(410, 229)])
    assert moved["right"].centroid == (410.0, 229.0) and moved["left"].centroid == (342.0, 384.0)
    lost = tracker.match([_segment(342, 384)])
    assert lost["left"] is not None and lost["right"] is None
    assert tracker.match([_segment(700, 700)]) == {"left": None, "right": None}


def test_nearest_valid_pixel_skips_invalid_depth_and_stays_inside_mask():
    depth = np.zeros((480, 640), dtype=np.uint16)
    mask = np.zeros((480, 640), dtype=bool)
    mask[200:220, 300:320] = True
    assert seg.nearest_valid_pixel(depth, mask, (310, 210)) is None
    depth[205, 318] = 900
    depth[100, 100] = 900  # valid but outside the mask
    assert seg.nearest_valid_pixel(depth, mask, (310, 210)) == (318, 205)


def test_initial_object_rejects_segments_adjacent_to_gloves():
    cuff = _segment(431, 81, area=825)
    can = _segment(354, 255, area=1535)
    gloves = [(360.0, 113.0), (342.0, 383.0)]
    assert seg.pick_initial_object([can, cuff], gloves) is can
    assert seg.pick_initial_object([cuff], gloves) is None
