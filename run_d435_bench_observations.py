"""Feed diagnostic glove/object observations from a converted D435 bench clip through the replay.

Colour-segmented glove centroids stand in for hand functional points and the
centroid of one colour-band object segment for the object; both are labelled diagnostic. Two replays run: optical
frame only (directional velocity withheld) and, if the operator asserts the bench
camera did not move, an identity pose into an assumed fixed frame. Outputs: two
traces, an overlay video and a summary with the assumptions spelled out.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

import bench_segments as seg
from sg_offline.contracts import Packet, Source, Stamp
from sg_offline.evaluation import summarize
from sg_offline.recordings import RGBDRecording
from sg_offline.replay import IMPLEMENTATION_VERSION, Replay, trace_hash, write_trace
from sg_offline.rgbd import CameraPose, same_time_distance

HAND_DEFINITION = "diagnostic_glove_colour_segment_centroid_not_functional_point"
OBJECT_DEFINITION = "diagnostic_object_colour_segment_centroid"
FIXED_FRAME = "bench_fixed_assumed_static_camera"


def observe(recording: RGBDRecording, top_hand: str):
    """Yield (frame, points, segments) per frame; identities persist by nearest centroid."""
    hands = seg.IdentityTracker(seg.GLOVE_GATE_PX)
    obj = seg.IdentityTracker(seg.OBJECT_GATE_PX)
    for frame in recording:
        hsv = seg.hsv_of(frame.rgb)
        gloves = seg.segments(hsv, seg.ORANGE_HSV, seg.GLOVE_MIN_AREA)
        candidates = seg.segments(hsv, seg.OBJECT_HSV, seg.OBJECT_MIN_AREA)
        if not hands.last:
            hands.seed(seg.assign_hands(gloves, top_hand))
            matched = {k: v for k, v in hands.match(gloves).items()} if hands.last else {"left": None, "right": None}
        else:
            matched = hands.match(gloves)
        if not obj.last:
            initial = seg.pick_initial_object(candidates, [s.centroid for s in gloves[:2]]) if len(gloves) >= 2 else None
            if initial is not None:
                obj.seed({"object": initial})
        obj_match = obj.match(candidates).get("object") if obj.last else None
        points = {}
        for identity, definition, segment, reason in (
                ("left", HAND_DEFINITION, matched.get("left"), "glove_segment_not_found"),
                ("right", HAND_DEFINITION, matched.get("right"), "glove_segment_not_found"),
                ("object", OBJECT_DEFINITION, obj_match, "object_segment_not_found_or_occluded")):
            pixel = None if segment is None else seg.nearest_valid_pixel(frame.depth, segment.mask, segment.centroid)
            point = frame.point_at(identity, pixel, observation=frame.rgb_stamp, definition=definition,
                                   decision_s=frame.available_s)
            if pixel is None:
                point = replace(point, reason=reason if segment is None else "no_valid_depth_in_segment")
            points[identity] = point
        yield frame, points, {"left": matched.get("left"), "right": matched.get("right"), "object": obj_match}


def static_pose(frame) -> CameraPose:
    stamp = Stamp(f"assumed-static-pose:{frame.frame_id}", frame.depth_stamp.acquired_s, frame.depth_stamp.available_s,
                  Source.SYNTHETIC, frame.depth_stamp.clock)
    return CameraPose(f"bench-static:{frame.frame_id}", frame.calibration.optical_frame, FIXED_FRAME,
                      np.eye(3), (0., 0., 0.), stamp, True)


def draw_overlay(frame, points, segments, row, distance_m, out_path):
    import cv2
    bgr = cv2.cvtColor(np.array(frame.rgb), cv2.COLOR_RGB2BGR)
    colours = {"left": (0, 200, 255), "right": (255, 120, 0), "object": (255, 60, 200)}
    for identity, segment in segments.items():
        if segment is None:
            continue
        contours, _ = cv2.findContours(segment.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, contours, -1, colours[identity], 1)
    y = 18
    for identity, point in points.items():
        if point.xyz is not None:
            col = int(round(frame.calibration.fx * point.xyz[0] / point.xyz[2] + frame.calibration.cx))
            rowi = int(round(frame.calibration.fy * point.xyz[1] / point.xyz[2] + frame.calibration.cy))
            cv2.circle(bgr, (col, rowi), 5, colours[identity], -1)
            label = f"{identity}: z={point.xyz[2]:.3f} m"
        else:
            label = f"{identity}: missing ({point.reason})"
        cv2.putText(bgr, label, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colours[identity], 1); y += 16
    for side, d in distance_m.items():
        cv2.putText(bgr, f"{side}-object distance: {'n/a' if d is None else f'{d:.3f} m'}", (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1); y += 16
    for output in row["outputs"]:
        cv2.putText(bgr, f"{output['side']}: {output['status']} [{output['reason']}]", (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 200), 1); y += 15
    cv2.putText(bgr, "diagnostic colour segments; simulation-only outputs", (6, frame.calibration.height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.imwrite(str(out_path), bgr)
    return bgr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--top-hand", choices=("left", "right"), required=True,
                    help="which of the operator's hands is higher in the image at the start (operator-supplied)")
    ap.add_argument("--assume-static-camera", action="store_true",
                    help="operator asserts the bench camera did not move; enables the fixed-frame replay")
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    recording = RGBDRecording(a.manifest)
    clock = None
    optical, fixed, meta = [], [], []
    frames_cache = []
    for frame, points, segments in observe(recording, a.top_hand):
        clock = frame.rgb_stamp.clock
        packet = Packet(frame.frame_id, frame.available_s, (points["left"], points["right"]), (points["object"],),
                        clock=clock, session_id=recording.session_id, participant_id=recording.participant_id)
        optical.append(packet)
        if a.assume_static_camera:
            pose = static_pose(frame)
            fixed.append(replace(packet, hands=tuple(pose.transform(p, decision_s=packet.decision_s) for p in packet.hands),
                                 objects=tuple(pose.transform(p, decision_s=packet.decision_s) for p in packet.objects)))
        distances = {side: same_time_distance(points[side], points["object"]) for side in ("left", "right")}
        meta.append(dict(frame_id=frame.frame_id, found={k: v is not None for k, v in segments.items()},
                         reasons={k: p.reason for k, p in points.items()}, distance_m=distances))
        frames_cache.append((frame, points, segments, distances))

    rows_optical = Replay().run(optical)
    write_trace(a.out_dir / "trace_optical_frame.jsonl", rows_optical)
    rows_fixed = Replay().run(fixed) if fixed else None
    if rows_fixed:
        write_trace(a.out_dir / "trace_assumed_static_camera.jsonl", rows_fixed)
    rows_for_overlay = rows_fixed or rows_optical

    frame_dir = a.out_dir / "overlay"; frame_dir.mkdir(exist_ok=True)
    writer = None
    for (frame, points, segments, distances), row in zip(frames_cache, rows_for_overlay):
        img = draw_overlay(frame, points, segments, row, distances, frame_dir / f"{Path(frame.frame_id).name}.png")
        if a.video:
            import cv2
            if writer is None:
                writer = cv2.VideoWriter(str(a.out_dir / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (img.shape[1], img.shape[0]))
            writer.write(img)
    if writer is not None:
        writer.release()

    found = {k: sum(m["found"][k] for m in meta) for k in ("left", "right", "object")}
    dists = [m["distance_m"][s] for m in meta for s in ("left", "right") if m["distance_m"][s] is not None]
    velocities = None
    if rows_fixed:
        velocities = sum(1 for r in rows_fixed for m in r["kinematics"].values() if m["velocity_mps"] is not None)
    summary = dict(
        implementation=IMPLEMENTATION_VERSION, source="recorded_bench_clip_diagnostic_segments",
        manifest=str(a.manifest), frames=len(meta), participant_N=1, population="operator_not_a_participant",
        observation_provider="bench_segments (HSV colour thresholds + nearest-centroid tracking); object = the metal can in smoke_03, caught by a bluish-grey band",
        detector_trained=False, functional_points_are_reference_definition=False,
        hand_identity_assumption=f"top glove at start = operator's {a.top_hand} hand (operator-supplied)",
        camera_pose="assumed_static_identity_not_estimated" if a.assume_static_camera else "none_velocity_withheld",
        segments_found=found,
        hand_object_distance_m={"n": len(dists), "min": round(min(dists), 3) if dists else None, "median": round(float(np.median(dists)), 3) if dists else None},
        velocity_ticks_with_value=velocities,
        optical=dict(trace_hash=trace_hash(rows_optical), **summarize(rows_optical)),
        assumed_static=dict(trace_hash=trace_hash(rows_fixed), **summarize(rows_fixed)) if rows_fixed else None,
        per_frame=meta,
        scope="Software integration on one operator bench clip. No reference labels: every request reference is unknown, "
              "so no accuracy, unwanted/missed proposal rate or scientific claim follows. Outputs are simulation-only.")
    (a.out_dir / "bench_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_frame"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
