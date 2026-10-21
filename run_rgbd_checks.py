"""Synthetic RGB-D recording replay with known camera poses and supplied pixels."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from sg_offline.contracts import Packet, Source, Stamp
from sg_offline.evaluation import summarize
from sg_offline.recordings import RECORDING_SCHEMA, RGBDRecording
from sg_offline.replay import IMPLEMENTATION_VERSION, Replay, canonical, trace_hash, write_trace
from sg_offline.rgbd import CameraCalibration, CameraPose, same_time_distance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "experiments/results/rgbd")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration = CameraCalibration("synthetic-only", 81, 81, 100., 100., 40., 40., .001, "eye_optical")
    records, pixel_records, poses = [], [], []
    for index, left_position in enumerate((-.2, -.1, -.05)):
        time = .5 + index * .1
        stamp = Stamp(f"frame-{index}", time, time, Source.SYNTHETIC)
        angle = index * np.pi / 2
        cosine, sine = float(np.cos(angle)), float(np.sin(angle))
        rotation = ((cosine, -sine, 0.), (sine, cosine, 0.), (0., 0., 1.))
        pose = CameraPose(f"pose-{index}", "eye_optical", "synthetic_fixed_world", rotation,
                          (.01 * index, 0., 0.), stamp, True)
        poses.append(pose)
        rgb = np.zeros((81, 81, 3), dtype=np.uint8)
        pixels = {}
        positions = {"left": (left_position, 0., 1.), "right": (.3, 0., 1.), "container": (0., 0., 1.)}
        colors = {"left": (255, 0, 0), "right": (0, 255, 0), "container": (0, 0, 255)}
        for identity, position in positions.items():
            optical = np.asarray(rotation).T @ (np.asarray(position) - pose.translation_m)
            column = int(round(calibration.fx * optical[0] / optical[2] + calibration.cx))
            row = int(round(calibration.fy * optical[1] / optical[2] + calibration.cy))
            pixels[identity] = (column, row)
            rgb[row, column] = colors[identity]
        pixel_records.append(pixels)
        np.save(args.output_dir / f"rgb_{index}.npy", rgb)
        np.save(args.output_dir / f"depth_{index}.npy", np.full((81, 81), 1000, dtype=np.uint16))
        records.append(dict(frame_id=f"frame-{index}", rgb_path=f"rgb_{index}.npy",
                            depth_path=f"depth_{index}.npy", rgb_stamp=asdict(stamp), depth_stamp=asdict(stamp),
                            registered_to_rgb=True, max_sync_skew_s=0.))
    manifest_path = args.output_dir / "recording.json"
    manifest_path.write_text(json.dumps(dict(schema=RECORDING_SCHEMA, participant_id="synthetic-person",
        session_id="synthetic-rgbd", camera_model="synthetic_pinhole_not_D435_measurement",
        calibration=asdict(calibration), frames=records), indent=2) + "\n")
    (args.output_dir / "provided_pixels_and_poses.json").write_text(json.dumps(
        dict(source="known_synthetic_geometry_not_detector_or_pose_estimator", pixels=pixel_records,
             poses=[asdict(pose) for pose in poses]), indent=2) + "\n")
    optical_packets, fixed_packets = [], []
    for image, pixels, pose in zip(RGBDRecording(manifest_path), pixel_records, poses):
        points = {identity: image.point_at(identity, pixel, observation=image.rgb_stamp,
                  definition="synthetic_surface_correspondence", decision_s=image.available_s)
                  for identity, pixel in pixels.items()}
        optical = Packet(image.frame_id, image.available_s, (points["left"], points["right"]),
                          (points["container"],), session_id="synthetic-rgbd")
        fixed = replace(optical,
                        hands=tuple(pose.transform(point, decision_s=image.available_s) for point in optical.hands),
                        objects=tuple(pose.transform(point, decision_s=image.available_s) for point in optical.objects))
        assert np.isclose(same_time_distance(optical.hands[0], optical.objects[0]),
                          same_time_distance(fixed.hands[0], fixed.objects[0]), atol=1e-12)
        optical_packets.append(optical)
        fixed_packets.append(fixed)
    raw_rows = Replay().run(optical_packets)
    fixed_rows = Replay().run(fixed_packets)
    assert all(motion["velocity_mps"] is None for row in raw_rows for motion in row["kinematics"].values())
    assert canonical(fixed_rows) == canonical(Replay().run(fixed_packets))
    assert canonical(fixed_rows[:2]) == canonical(Replay().run(fixed_packets[:2]))
    np.testing.assert_allclose(fixed_rows[1]["kinematics"]["left"]["velocity_mps"], (1., 0., 0.), atol=1e-12)
    write_trace(args.output_dir / "optical_without_pose.jsonl", raw_rows)
    write_trace(args.output_dir / "fixed_with_known_pose.jsonl", fixed_rows)
    summary = dict(implementation=IMPLEMENTATION_VERSION, source="synthetic_only", participant_N=0,
                   population="not_applicable_synthetic", frames=len(fixed_rows),
                   real_camera_adapter=False, detector_executed=False, pose_estimator_executed=False,
                   registered_depth_fixture_loaded=True, unknown_frame_velocity_withheld=True,
                   relative_distance_preserved=True, true_hand_motion_preserved=True,
                   deterministic_repeat=True, causal_prefix=True,
                   trace_hash=trace_hash(fixed_rows), **summarize(fixed_rows))
    (args.output_dir / "geometry_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
