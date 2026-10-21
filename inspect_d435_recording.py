"""Inspect a native RealSense recording (.db3 / .bag) before any adapter work.

Reads the file with pyrealsense2 playback and reports what was actually
recorded: device fields, sensors, stream profiles, intrinsics, depth scale,
depth->color extrinsics, per-stream frame counts, timestamp domains, frame
intervals, gaps and counter resets, and the invalid-depth fraction on sampled
frames. Read-only: writes no manifest and makes no claim beyond file contents.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pyrealsense2 as rs


def _profile_dict(p: rs.stream_profile) -> dict:
    d = {
        "stream": str(p.stream_type()).replace("stream.", ""),
        "index": p.stream_index(),
        "format": str(p.format()).replace("format.", ""),
        "fps": p.fps(),
        "unique_id": p.unique_id(),
    }
    if p.is_video_stream_profile():
        v = p.as_video_stream_profile()
        try:
            i = v.get_intrinsics()
            d["intrinsics"] = {
                "width": i.width, "height": i.height,
                "fx": i.fx, "fy": i.fy, "ppx": i.ppx, "ppy": i.ppy,
                "model": str(i.model).replace("distortion.", ""),
                "coeffs": list(i.coeffs),
            }
        except RuntimeError as e:  # intrinsics can be absent in a file
            d["intrinsics_error"] = str(e)
        else:
            d["width"], d["height"] = v.width(), v.height()
    return d


def _device_info(dev: rs.device) -> dict:
    out = {}
    for name in dir(rs.camera_info):
        if name.startswith("_"):
            continue
        key = getattr(rs.camera_info, name)
        if isinstance(key, rs.camera_info) and dev.supports(key):
            out[name] = dev.get_info(key)
    return out


def inspect(path: Path, sample_every: int) -> dict:
    ctx = rs.context()
    dev = ctx.load_device(str(path))
    playback = dev.as_playback()
    playback.set_real_time(False)
    report: dict = {
        "file": str(path),
        "bytes": path.stat().st_size,
        "duration_s": playback.get_duration().total_seconds(),
        "device_info": _device_info(dev),
        "sensors": [],
    }
    depth_scale = None
    profiles_by_stream: dict[str, rs.stream_profile] = {}
    for s in dev.query_sensors():
        entry = {"name": s.get_info(rs.camera_info.name), "profiles": []}
        for p in s.get_stream_profiles():
            pd = _profile_dict(p)
            entry["profiles"].append(pd)
            profiles_by_stream.setdefault(pd["stream"], p)
        if s.is_depth_sensor():
            try:
                depth_scale = s.as_depth_sensor().get_depth_scale()
            except RuntimeError as e:
                entry["depth_scale_error"] = str(e)
        report["sensors"].append(entry)
    report["depth_scale_m"] = depth_scale
    if {"depth", "color"} <= profiles_by_stream.keys():
        try:
            ex = profiles_by_stream["depth"].get_extrinsics_to(profiles_by_stream["color"])
            report["extrinsics_depth_to_color"] = {"rotation": list(ex.rotation), "translation_m": list(ex.translation)}
        except RuntimeError as e:  # profiles declared but no extrinsics stored in the file
            report["extrinsics_depth_to_color"] = None
            report["extrinsics_error"] = str(e)
    else:
        report["extrinsics_depth_to_color"] = None
        report["missing_streams"] = sorted({"depth", "color"} - profiles_by_stream.keys())

    # Frame pass through the sensor API (as rs-convert does): the pipeline's
    # syncer would hold frames of a stream whose partner never delivered.
    import threading
    import time
    lock = threading.Lock()
    per_stream: dict[str, dict] = {}
    n_frames = 0

    def on_frame(f: rs.frame) -> None:
        nonlocal n_frames
        p = f.get_profile()
        key = str(p.stream_type()).replace("stream.", "")
        with lock:
            n_frames += 1
            st = per_stream.setdefault(key, {"count": 0, "numbers": [], "stamps_ms": [], "domains": set(),
                                             "zero_depth_fraction_samples": []})
            st["count"] += 1
            st["numbers"].append(f.get_frame_number())
            st["stamps_ms"].append(f.get_timestamp())
            st["domains"].add(str(f.get_frame_timestamp_domain()).replace("timestamp_domain.", ""))
            if key == "depth" and st["count"] % sample_every == 1:
                a = np.asanyarray(f.get_data())
                st["zero_depth_fraction_samples"].append(float((a == 0).mean()))

    sensors = dev.query_sensors()
    for s in sensors:
        s.open(s.get_stream_profiles())
        s.start(on_frame)
    try:
        last_pos, stalled_since = -1, time.monotonic()
        while True:
            time.sleep(0.05)
            pos = playback.get_position()
            if pos != last_pos:
                last_pos, stalled_since = pos, time.monotonic()
            elif time.monotonic() - stalled_since > 1.5:
                break
            if playback.current_status() == rs.playback_status.stopped and pos == last_pos:
                break
    finally:
        for s in sensors:
            try:
                s.stop()
                s.close()
            except RuntimeError:
                pass
    n_sets = n_frames

    report["frames_delivered"] = n_sets
    streams_out = {}
    for key, st in per_stream.items():
        stamps = st["stamps_ms"]
        nums = st["numbers"]
        intervals = [b - a for a, b in zip(stamps, stamps[1:])]
        med = statistics.median(intervals) if intervals else None
        gaps = [(i + 1, round(iv, 1)) for i, iv in enumerate(intervals) if med and iv > 1.5 * med]
        resets = [i + 1 for i, (a, b) in enumerate(zip(nums, nums[1:])) if b <= a]
        streams_out[key] = {
            "count": st["count"],
            "timestamp_domains": sorted(st["domains"]),
            "first_stamp_ms": stamps[0] if stamps else None,
            "last_stamp_ms": stamps[-1] if stamps else None,
            "span_s": (stamps[-1] - stamps[0]) / 1000.0 if len(stamps) > 1 else None,
            "median_interval_ms": round(med, 3) if med else None,
            "estimated_fps": round(1000.0 / med, 2) if med else None,
            "gaps_over_1p5x_median": gaps[:20],
            "gap_count": len(gaps),
            "frame_number_resets_at": resets[:20],
            "zero_depth_fraction_samples": [round(z, 4) for z in st["zero_depth_fraction_samples"]] or None,
        }
    report["streams"] = streams_out
    report["scope"] = ("File-content inspection only. Not a calibration, synchronization or device validation; "
                       "USB mode, cable and host contention are not recoverable from the file.")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recording", type=Path, help=".db3 or .bag written by librealsense")
    ap.add_argument("--json", type=Path, help="write the report here as JSON")
    ap.add_argument("--sample-every", type=int, default=25, help="sample invalid-depth fraction every N depth frames")
    a = ap.parse_args(argv)
    if not a.recording.exists():
        print(f"missing file: {a.recording}", file=sys.stderr)
        return 2
    rep = inspect(a.recording, a.sample_every)
    text = json.dumps(rep, indent=1, default=str)
    if a.json:
        a.json.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
