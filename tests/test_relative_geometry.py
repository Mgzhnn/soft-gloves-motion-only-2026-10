"""Pose-free same-time distance rate: frame-invariant, causal, and feeding the approach-rate feature."""
import numpy as np
import pytest

from sg_offline.contracts import Packet, Point, Source, Stamp
from sg_offline.inference import hypotheses
from sg_offline.replay import Replay
from sg_offline.state import CausalKinematics, RelativeGeometry

CLOCK = "cam_clock"


def _stamp(name, t):
    return Stamp(f"{name}@{t}", t, t, Source.RECORDED, CLOCK)


def _point(identity, xyz, t, *, frame="cam_optical", fixed=False, cal="cal-1", definition="fp"):
    return Point(identity, None if xyz is None else tuple(float(v) for v in xyz), _stamp(identity, t), frame=frame,
                 frame_is_fixed=fixed, calibration_id=cal, definition=definition)


def _rot(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _packet(t, left, right, obj, **kw):
    return Packet(f"pk@{t}", t, (left, right), (obj,) if obj is not None else (), clock=CLOCK,
                  session_id="s", participant_id="p", **kw)


def test_distance_rate_survives_camera_rotation_and_translation_without_a_fixed_frame():
    # World truth: right hand approaches the object at 1 m/s along x; the camera rotates 25 deg and shifts between frames.
    hand_w = [np.array([0.5, 0, 1.0]), np.array([0.4, 0, 1.0])]
    obj_w = np.array([0.0, 0, 1.0])
    left_w = np.array([-0.3, 0.2, 1.0])
    cams = [(np.eye(3), np.zeros(3)), (_rot(25), np.array([0.05, -0.02, 0.03]))]
    rel, kin = RelativeGeometry(), CausalKinematics()
    outs = []
    for k, t in enumerate((10.0, 10.1)):
        R, tr = cams[k]
        c = lambda w: R @ w + tr  # noqa: E731
        pk = _packet(t, _point("left", c(left_w), t), _point("right", c(hand_w[k]), t), _point("cup", c(obj_w), t))
        motions = kin.update(pk)
        outs.append((rel.update(pk), motions, pk))
    relatives, motions, pk = outs[-1]
    r = relatives[("right", "cup")]
    assert r.distance_m == pytest.approx(0.4) and r.rate_mps == pytest.approx(-1.0, abs=1e-9)
    assert r.reason == "past_secant_distance_rate_frame_invariant" and r.source == "same_time_distance_rate"
    assert motions["right"].velocity_mps is None and motions["right"].velocity_reason == "moving_or_unverified_reference_frame"
    bundle = hypotheses(pk, motions, (), kinetics=False, relatives=relatives)
    right_cup = [h for h in bundle.hypotheses if h.object_ids[1] == "cup"]
    assert right_cup
    row = bundle.features[bundle.hypotheses.index(right_cup[0])]
    assert row[15 + 1] == pytest.approx(-1.0 / .2) and row[15 + 8] == 1.0  # approach-rate column and availability flag


def test_fixed_frame_velocities_and_pose_free_rate_agree_for_linear_motion():
    rel, kin = RelativeGeometry(), CausalKinematics()
    for t, x in ((0.0, 0.6), (0.1, 0.5)):
        pk = _packet(t, _point("left", (-0.3, 0.2, 1), t, fixed=True), _point("right", (x, 0, 1), t, fixed=True),
                     _point("cup", (0, 0, 1), t, fixed=True))
        motions, relatives = kin.update(pk), rel.update(pk)
    delta = np.array(motions["right"].point.xyz) - np.array(motions["cup"].point.xyz)
    radial = float(np.dot(delta / np.linalg.norm(delta), np.array(motions["right"].velocity_mps) - np.array(motions["cup"].velocity_mps)))
    assert radial == pytest.approx(relatives[("right", "cup")].rate_mps, abs=1e-9)


@pytest.mark.parametrize("case", ["missing_object", "asynchronous", "calibration_changed", "definition_changed", "gap_exceeded", "nonadvancing"])
def test_each_guard_withholds_with_its_own_reason(case):
    rel = RelativeGeometry(max_gap_s=0.25)
    first = _packet(0.0, _point("left", (-0.3, 0, 1), 0.0), _point("right", (0.5, 0, 1), 0.0), _point("cup", (0, 0, 1), 0.0))
    rel.update(first)
    t = 0.1
    right = _point("right", (0.4, 0, 1), t)
    cup = _point("cup", (0, 0, 1), t)
    if case == "missing_object":
        cup = Point("cup", None, _stamp("cup", t), frame="cam_optical", reason="lost"); expected = "point_unavailable"
    elif case == "asynchronous":
        cup = _point("cup", (0, 0, 1), t - 0.02); expected = "asynchronous_points"
    elif case == "calibration_changed":
        right = _point("right", (0.4, 0, 1), t, cal="cal-2"); cup = _point("cup", (0, 0, 1), t, cal="cal-2"); expected = "calibration_changed"
    elif case == "definition_changed":
        right = _point("right", (0.4, 0, 1), t, definition="centroid"); expected = "point_definition_changed"
    elif case == "gap_exceeded":
        t = 0.5; right = _point("right", (0.4, 0, 1), t); cup = _point("cup", (0, 0, 1), t); expected = "gap_exceeded"
    else:
        t = 0.0; right = _point("right", (0.4, 0, 1), t); cup = _point("cup", (0, 0, 1), t); expected = "nonadvancing_acquisition"
    out = rel.update(_packet(t, _point("left", (-0.3, 0, 1), t), right, cup))[("right", "cup")]
    assert out.rate_mps is None and out.reason == expected
    if case in ("missing_object", "asynchronous"):
        assert out.distance_m is None


def test_replay_logs_relative_geometry_and_no_longer_blames_missing_velocity_when_rate_exists():
    packets = []
    for k, t in enumerate((0.0, 0.1, 0.2, 0.3)):
        x = 0.5 - 0.12 * k
        packets.append(_packet(t, _point("left", (-0.3, 0.2, 1), t), _point("right", (x, 0, 1), t), _point("cup", (0, 0, 1), t)))
    rows = Replay().run(packets)
    assert rows[0]["implementation_version"] == "motion-only-build/7"
    logged = rows[-1]["relative_geometry"]["right|cup"]
    assert logged["source"] == "same_time_distance_rate" and logged["rate_mps"] == pytest.approx(-1.2, abs=1e-9)
    right_out = rows[-1]["outputs"][1]
    assert right_out["reason"] != "past_relative_approach_rate_unavailable_or_ambiguous"
    assert all(m["velocity_mps"] is None for r in rows for m in r["kinematics"].values())  # still no directional velocity
