from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from sg_offline.contracts import Packet, Point, Source, Stamp
from sg_offline.fixtures import scenarios
from sg_offline.inference import LinearJoint
from sg_offline.recordings import RECORDING_SCHEMA, RGBDRecording
from sg_offline.replay import Replay, canonical
from sg_offline.rgbd import CameraCalibration, CameraPose, RGBDFrame, same_time_distance
from sg_offline.state import CausalKinematics


def calibration(**changes):
    values = dict(calibration_id="synthetic-calibration", width=7, height=5,
                  fx=100., fy=100., cx=3., cy=2., depth_scale_m=.001, optical_frame="eye_optical")
    return CameraCalibration(**(values | changes))


def stamp(time=1., source=Source.SYNTHETIC):
    return Stamp(f"sample-{time}", time, time, source)


def frame():
    return RGBDFrame("frame", np.zeros((5, 7, 3), dtype=np.uint8),
                     np.full((5, 7), 1000., dtype=float), stamp(), stamp(),
                     calibration(), True, 0.)


def sample(image, pixel=(4, 2), observation=None, decision_s=1.):
    return image.point_at("left", pixel, observation=observation or stamp(),
                          definition="synthetic_point", decision_s=decision_s)


def pose(time=1., angle=0., translation=(0., 0., 0.), source=Source.SYNTHETIC):
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    rotation = ((cosine, -sine, 0.), (sine, cosine, 0.), (0., 0., 1.))
    return CameraPose(f"pose-{time}", "eye_optical", "fixed_session_world", rotation,
                      translation, stamp(time, source), True)


@pytest.mark.parametrize("changes", [dict(fx=0.), dict(fy=-1.), dict(depth_scale_m=0.),
    dict(cx=float("nan")), dict(width=True), dict(height=0), dict(projection="distorted"),
    dict(depth_convention="radial_range"), dict(calibration_id="")])
def test_invalid_calibration_is_rejected(changes):
    with pytest.raises(ValueError):
        calibration(**changes)


def test_projection_uses_metric_depth_and_preserves_provenance():
    point = sample(frame())
    assert point.xyz == pytest.approx((.01, 0., 1.))
    assert point.frame == "eye_optical" and not point.frame_is_fixed
    assert point.calibration_id == "synthetic-calibration"
    assert point.sigma_m is None


@pytest.mark.parametrize("depth", [0., -1., float("nan"), float("inf"), 65535.])
def test_bad_depth_is_missing_not_a_point_at_zero(depth):
    image = frame()
    values = image.depth.copy()
    values[2, 4] = depth
    image = replace(image, depth=values, invalid_depth_value=65535.)
    result = sample(image)
    assert result.xyz is None and result.reason == "invalid_depth"


@pytest.mark.parametrize("pixel", [None, (-1, 2), (7, 2), (4, 5)])
def test_unavailable_pixel_remains_missing(pixel):
    assert sample(frame(), pixel).xyz is None


def test_noninteger_pixel_requires_explicit_sampling_policy():
    with pytest.raises(ValueError):
        sample(frame(), (.5, 2))


def test_frame_registration_resolution_and_mutation():
    image = frame()
    with pytest.raises(ValueError):
        replace(image, registered_to_rgb=False)
    with pytest.raises(ValueError):
        replace(image, depth=np.ones((3, 3)))
    with pytest.raises(ValueError):
        replace(image, rgb=np.zeros((5, 7, 3)))
    with pytest.raises(ValueError):
        image.depth[0, 0] = 5
    array = np.full((5, 7), 1000.)
    copied = replace(image, depth=array)
    array[:] = 0
    assert sample(copied).xyz is not None


def test_pair_clock_skew_and_availability_are_enforced():
    image = frame()
    with pytest.raises(ValueError):
        replace(image, depth_stamp=replace(stamp(), clock="other"))
    with pytest.raises(ValueError):
        replace(image, depth_stamp=stamp(1.1))
    delayed = replace(image, depth_stamp=replace(stamp(), available_s=1.2))
    with pytest.raises(ValueError):
        sample(delayed)
    assert sample(delayed, decision_s=1.2).stamp.available_s == 1.2
    with pytest.raises(ValueError):
        sample(image, observation=stamp(1.1), decision_s=1.1)
    with pytest.raises(ValueError):
        sample(image, observation=replace(stamp(), available_s=1.1))


@pytest.mark.parametrize("rotation", [((2., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
    ((-1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), ((float("nan"), 0., 0.), (0., 1., 0.), (0., 0., 1.))])
def test_pose_rejects_scale_reflection_and_nan(rotation):
    with pytest.raises(ValueError):
        replace(pose(), rotation=rotation)


def test_pose_is_timestamp_matched_and_causal():
    point = sample(frame())
    delayed = replace(pose(), stamp=replace(stamp(), available_s=1.2))
    with pytest.raises(ValueError):
        delayed.transform(point, decision_s=1.)
    transformed = delayed.transform(point, decision_s=1.2)
    assert transformed.stamp.available_s == 1.2
    assert transformed.frame_is_fixed and transformed.transform_id == delayed.pose_id
    with pytest.raises(ValueError):
        pose(.9).transform(point, decision_s=1.)
    with pytest.raises(ValueError):
        replace(pose(), source_frame="wrong").transform(point, decision_s=1.)
    with pytest.raises(ValueError):
        replace(pose(), stamp=replace(stamp(), clock="other")).transform(point, decision_s=1.)
    with pytest.raises(ValueError):
        replace(pose(), target_is_fixed=False)


def test_pose_uncertainty_not_silently_treated_as_exact_or_synthetic():
    point = replace(sample(frame()), sigma_m=.001)
    transformed = pose(source=Source.RECORDED).transform(point, decision_s=1.)
    assert transformed.stamp.source == Source.RECORDED
    assert transformed.sigma_m is None
    assert pose().transform(sample(frame(), None), decision_s=1.).xyz is None


def camera_packet(index, camera_pose, world_shift=0.):
    world = scenarios()[0]
    time = camera_pose.stamp.acquired_s
    def observed(point):
        position = np.array(point.xyz)
        if point.identity == "left":
            position[0] += world_shift
        optical = np.asarray(camera_pose.rotation).T @ (position - camera_pose.translation_m)
        return replace(point, xyz=tuple(float(value) for value in optical),
                       stamp=Stamp(f"{index}:{point.identity}", time, time, Source.SYNTHETIC),
                       frame="eye_optical", frame_is_fixed=False, calibration_id="synthetic-calibration")
    return Packet(f"camera-{index}", time, tuple(observed(point) for point in world.hands),
                  tuple(observed(point) for point in world.objects))


def compensated(packet, camera_pose):
    return replace(packet, hands=tuple(camera_pose.transform(point, decision_s=packet.decision_s) for point in packet.hands),
                   objects=tuple(camera_pose.transform(point, decision_s=packet.decision_s) for point in packet.objects))


@pytest.mark.parametrize("angle,translation", [(0., (.1, -.2, .05)),
    (np.pi / 2, (0., 0., 0.)), (np.pi / 3, (.1, -.2, .05))])
def test_head_motion_does_not_become_task_motion(angle, translation):
    initial_pose = pose(.5)
    changed_pose = pose(.6, angle, translation)
    initial = camera_packet(0, initial_pose)
    changed = camera_packet(1, changed_pose)
    assert same_time_distance(initial.hands[0], initial.objects[0]) == pytest.approx(
        same_time_distance(changed.hands[0], changed.objects[0]))
    raw = CausalKinematics()
    raw.update(initial)
    motions = raw.update(changed)
    assert all(motion.velocity_mps is None for motion in motions.values())
    assert motions["left"].velocity_reason == "moving_or_unverified_reference_frame"
    fixed = CausalKinematics()
    fixed.update(compensated(initial, initial_pose))
    for motion in fixed.update(compensated(changed, changed_pose)).values():
        assert motion.velocity_mps == pytest.approx((0., 0., 0.), abs=1e-12)


def test_compensation_preserves_actual_hand_motion_and_missing_pose_withholds_velocity_only():
    initial_pose = pose(.5)
    changed_pose = pose(.6, np.pi / 2, (.1, -.2, 0.))
    initial = camera_packet(0, initial_pose)
    changed = camera_packet(1, changed_pose, world_shift=.1)
    kinematics = CausalKinematics()
    kinematics.update(compensated(initial, initial_pose))
    motions = kinematics.update(compensated(changed, changed_pose))
    assert motions["left"].velocity_mps == pytest.approx((1., 0., 0.))
    assert motions["container"].velocity_mps == pytest.approx((0., 0., 0.))
    model = LinearJoint()
    model.W[7, -1] = 20.
    # Without a pose the grasp used to be withheld for missing relative velocity. Now the
    # frame-invariant same-time distance rate supplies the approach rate, so the proposal is admitted
    # while directional velocity stays withheld and the provenance names the pose-free route.
    raw_rows = Replay(model=model).run([initial, changed])
    assert all(m["velocity_mps"] is None for m in raw_rows[1]["kinematics"].values())
    assert raw_rows[1]["relative_geometry"]["left|container"]["source"] == "same_time_distance_rate"
    assert raw_rows[1]["outputs"][0]["status"] == "proposed_simulation_only"
    rows = Replay(model=model).run([compensated(initial, initial_pose), compensated(changed, changed_pose)])
    assert rows[1]["outputs"][0]["status"] == "proposed_simulation_only"
    assert rows[1]["relative_geometry"]["left|container"]["rate_mps"] == pytest.approx(
        raw_rows[1]["relative_geometry"]["left|container"]["rate_mps"])
    assert canonical(rows[:1]) == canonical(Replay(model=model).run([compensated(initial, initial_pose)]))


def test_fixed_name_alone_does_not_authorize_velocity_and_calibration_change_resets():
    packets = [replace(packet, hands=tuple(replace(point, frame_is_fixed=False) for point in packet.hands))
               for packet in scenarios()[:2]]
    kinematics = CausalKinematics()
    kinematics.update(packets[0])
    assert kinematics.update(packets[1])["left"].velocity_mps is None
    kinematics = CausalKinematics()
    kinematics.update(scenarios()[0])
    later = scenarios()[1]
    later = replace(later, hands=(replace(later.hands[0], calibration_id="changed"), later.hands[1]))
    assert kinematics.update(later)["left"].velocity_reason == "calibration_changed"
    with pytest.raises(ValueError):
        replace(later.hands[0], frame_is_fixed="true")


def test_distance_rejects_mixed_frames_and_asynchronous_points():
    point = sample(frame())
    with pytest.raises(ValueError):
        same_time_distance(point, replace(point, frame="other"))
    with pytest.raises(ValueError):
        same_time_distance(point, replace(point, stamp=stamp(2.)))
    assert same_time_distance(point, replace(point, xyz=None, reason="missing")) is None


def write_recording(tmp_path):
    image = frame()
    np.save(tmp_path / "rgb.npy", image.rgb)
    np.save(tmp_path / "depth.npy", image.depth)
    records = [dict(frame_id=f"frame-{time}", rgb_path="rgb.npy", depth_path="depth.npy",
                    rgb_stamp=asdict(stamp(time)), depth_stamp=asdict(stamp(time)),
                    registered_to_rgb=True, max_sync_skew_s=0.) for time in (1., 2.)]
    manifest = dict(schema=RECORDING_SCHEMA, participant_id="synthetic-person", session_id="synthetic-session",
                    camera_model="synthetic_pinhole", calibration=asdict(calibration()), frames=records)
    path = tmp_path / "recording.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_recording_loads_deterministically_and_retains_identity(tmp_path):
    path, _ = write_recording(tmp_path)
    recording = RGBDRecording(path)
    assert recording.session_id == "synthetic-session"
    assert [image.frame_id for image in recording] == ["frame-1.0", "frame-2.0"]
    for first, second in zip(recording, recording):
        np.testing.assert_array_equal(first.depth, second.depth)


@pytest.mark.parametrize("mutation", ["schema", "extra_label", "duplicate", "reversed", "path_escape", "reused_sample", "clock_change"])
def test_recording_rejects_schema_time_and_provenance_errors(tmp_path, mutation):
    path, manifest = write_recording(tmp_path)
    if mutation == "schema":
        manifest["schema"] = "old"
    elif mutation == "extra_label":
        manifest["reference_request"] = "G"
    elif mutation == "duplicate":
        manifest["frames"][1]["frame_id"] = manifest["frames"][0]["frame_id"]
    elif mutation == "reversed":
        manifest["frames"].reverse()
    elif mutation == "path_escape":
        manifest["frames"][0]["rgb_path"] = "../outside.npy"
    elif mutation == "reused_sample":
        manifest["frames"][1]["rgb_stamp"]["sample_id"] = manifest["frames"][0]["rgb_stamp"]["sample_id"]
    elif mutation == "clock_change":
        manifest["frames"][1]["rgb_stamp"]["clock"] = "different"
        manifest["frames"][1]["depth_stamp"]["clock"] = "different"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        list(RGBDRecording(path))


def test_recording_refuses_pickled_arrays(tmp_path):
    path, _ = write_recording(tmp_path)
    np.save(tmp_path / "depth.npy", np.array([{"unexpected": "object"}], dtype=object))
    with pytest.raises(ValueError):
        list(RGBDRecording(path))
