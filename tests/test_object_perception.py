"""Object-observation contract and providers. Synthetic frames, participant N=0."""
from dataclasses import replace

import numpy as np
import pytest

from sg_offline.contracts import Stamp, Source
from sg_offline.perception import ObjectObservation, packet_from_detections
from test_rgbd import frame


def observation_stamp(f):
    return replace(f.rgb_stamp, sample_id=f.frame_id + ":perception", available_s=f.available_s)


def test_point_from_xyz_is_stamped_like_a_lifted_pixel_and_guards_time_and_geometry():
    f = frame()
    obs = observation_stamp(f)
    lifted = f.point_at("cup", (3, 2), observation=obs, definition="d", decision_s=f.available_s)
    direct = f.point_from_xyz("cup", (0.1, -0.05, 0.9), observation=obs, definition="d", decision_s=f.available_s)
    assert direct.xyz == (0.1, -0.05, 0.9) and direct.frame == lifted.frame and direct.calibration_id == lifted.calibration_id
    assert direct.stamp.acquired_s == lifted.stamp.acquired_s and direct.stamp.clock == lifted.stamp.clock and direct.stamp.source == lifted.stamp.source
    missing = f.point_from_xyz("cup", None, observation=obs, definition="d", decision_s=f.available_s, reason="mask_absent")
    assert missing.xyz is None and missing.reason == "mask_absent"
    with pytest.raises(ValueError, match="reason"):
        f.point_from_xyz("cup", None, observation=obs, definition="d", decision_s=f.available_s)
    with pytest.raises(ValueError, match="front of the camera"):
        f.point_from_xyz("cup", (0., 0., -1.), observation=obs, definition="d", decision_s=f.available_s)
    with pytest.raises(ValueError, match="not available"):
        f.point_from_xyz("cup", (0., 0., 1.), observation=obs, definition="d", decision_s=f.available_s - 1.)
    with pytest.raises(ValueError, match="clock"):
        f.point_from_xyz("cup", (0., 0., 1.), observation=replace(obs, clock="other"), definition="d", decision_s=f.available_s)


def test_object_observations_enter_the_packet_with_their_metric_point_and_details_stay_outside():
    f = frame()
    details = {}
    cup = ObjectObservation("cup", (0.1, 0.0, 0.8), pixel=(3, 2), details=dict(height_m=.09, n_points=120, source="plane_cluster"))
    lost = ObjectObservation("jar", None, reason="mask_absent", details=dict(source="sam2"))
    packet, diag = packet_from_detections(f, {"left": None, "right": None}, {"cup": cup, "jar": lost},
                                          participant_id="p", session_id="s", available_s=f.available_s, object_details=details)
    by_id = {o.identity: o for o in packet.objects}
    assert by_id["cup"].xyz == (0.1, 0.0, 0.8) and by_id["cup"].definition.endswith(":cup") and by_id["cup"].reason == ""
    assert by_id["jar"].xyz is None and by_id["jar"].reason == "mask_absent"
    assert details["cup"]["height_m"] == .09 and details["cup"]["pixel"] == (3, 2) and details["jar"]["reason"] == "mask_absent"
    assert all(k not in by_id["cup"].__dict__ for k in ("height_m", "details"))          # measurements never enter the point
    # legacy forms still work side by side
    packet2, _ = packet_from_detections(f, {"left": None, "right": None}, {"cup": (3, 2), "marker": (None, "colour_absent")},
                                        participant_id="p", session_id="s", available_s=f.available_s)
    assert {o.identity: o.reason for o in packet2.objects} == {"cup": "", "marker": "colour_absent"}
    with pytest.raises(ValueError, match="identity"):
        packet_from_detections(f, {"left": None, "right": None}, {"cup": replace(cup, identity="other")},
                               participant_id="p", session_id="s", available_s=f.available_s)


def test_object_observation_contract_refuses_hand_names_and_silent_absence():
    with pytest.raises(ValueError):
        ObjectObservation("left", (0., 0., 1.))
    with pytest.raises(ValueError):
        ObjectObservation("cup", None)
    with pytest.raises(ValueError):
        ObjectObservation("cup", (0., float("nan"), 1.))
    assert ObjectObservation("cup", None, reason="plane_unavailable").reason == "plane_unavailable"
