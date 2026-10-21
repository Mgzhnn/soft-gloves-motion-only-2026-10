"""Convert a native D435 recording (.db3/.bag) into an `sg-rgbd-recording/1` manifest.

Registration is done with librealsense's calibrated `align` block, so every
emitted depth image lives in the colour camera's rectified pinhole frame at the
colour resolution. Pairs whose RGB/depth acquisition skew exceeds the declared
tolerance are dropped and listed, not silently kept. Provenance that the manifest
schema does not admit (device fields, profiles, extrinsics, skew statistics,
drops) goes to a sidecar `adapter_provenance.json` next to the manifest.

This establishes the file contract only: it is not a calibration check, a
synchronization validation or a statement about the scene content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from inspect_d435_recording import _device_info, _profile_dict
from sg_offline.contracts import Source, Stamp
from sg_offline.recordings import RECORDING_SCHEMA, RGBDRecording
from sg_offline.rgbd import CameraCalibration

CLOCK = "d435_global_time_epoch_s"  # librealsense global-time domain, milliseconds since epoch, converted to seconds


def stamp_from_frame(stream: str, frame_number: int, acquired_ms: float, arrival_ms: float | None) -> Stamp:
    """Acquisition from the frame timestamp; availability from host arrival when the file has it."""
    acquired_s = acquired_ms / 1000.0
    available_s = acquired_s if arrival_ms is None else max(acquired_s, arrival_ms / 1000.0)
    return Stamp(f"{stream}#{frame_number}@{acquired_ms:.3f}", acquired_s, available_s, Source.RECORDED, CLOCK)


class OrderGuard:
    """Refuse frames that would break the reader's causal ordering rules instead of writing them."""

    def __init__(self):
        self.last_rgb = self.last_depth = self.last_available = -1.0
        self.samples: set[tuple[str, str]] = set()

    def reason(self, rgb: Stamp, depth: Stamp) -> str:
        if rgb.acquired_s <= self.last_rgb or depth.acquired_s <= self.last_depth:
            return "non_increasing_acquisition"
        if max(rgb.available_s, depth.available_s) < self.last_available:
            return "availability_regressed"
        if ("rgb", rgb.sample_id) in self.samples or ("depth", depth.sample_id) in self.samples:
            return "reused_sample_id"
        return ""

    def admit(self, rgb: Stamp, depth: Stamp) -> None:
        self.last_rgb, self.last_depth = rgb.acquired_s, depth.acquired_s
        self.last_available = max(rgb.available_s, depth.available_s)
        self.samples |= {("rgb", rgb.sample_id), ("depth", depth.sample_id)}


def rectified_calibration(intrinsics, depth_scale_m: float, calibration_id: str, optical_frame: str) -> CameraCalibration:
    if any(c != 0.0 for c in intrinsics.coeffs):
        raise ValueError("colour intrinsics carry distortion; adapter only emits rectified pinhole images")
    return CameraCalibration(calibration_id, int(intrinsics.width), int(intrinsics.height),
                             float(intrinsics.fx), float(intrinsics.fy), float(intrinsics.ppx), float(intrinsics.ppy),
                             float(depth_scale_m), optical_frame)


# 300 s at 30 fps: covers the single 270-second recording script without silent truncation.
# The byte and free-space bounds below still stop extraction early and record the reason.
DEFAULT_MAX_FRAMES = 9000


def timestamp_domain_ok(domain) -> bool:
    """Only librealsense's global-time domain is on the host clock that the arrival metadata uses.
    Hardware-clock or system-time recordings would mix clocks under the global-time label, so refuse them."""
    return str(domain).rsplit(".", 1)[-1] == "global_time"


def _metadata_ms(frame, key):
    return float(frame.get_frame_metadata(key)) if frame.supports_frame_metadata(key) else None


EXTRACTION_FILE_PATTERNS = ("rgb_*.npy", "depth_*.npy", "recording.json", "adapter_provenance.json")


def _clear_previous_extraction(out_dir: Path) -> list[str]:
    """Remove the files an earlier extraction wrote, so an interrupted ``--overwrite`` run cannot leave
    frames of two recordings side by side under one manifest (review C6, 2 Oct 2026). Only the adapter's
    own file patterns are touched; anything else in the directory is left alone."""
    removed = []
    for pattern in EXTRACTION_FILE_PATTERNS:
        for path in sorted(out_dir.glob(pattern)):
            if path.is_file():
                path.unlink(); removed.append(path.name)
    return removed


def convert(source: Path, out_dir: Path, participant_id: str, session_id: str, *, max_sync_skew_s: float,
            optical_frame: str, overwrite: bool, max_frames: int = DEFAULT_MAX_FRAMES, max_output_gib: float = 12., min_free_gib: float = 10.) -> dict:
    import pyrealsense2 as rs  # native dependency kept out of sg_offline

    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_dir} is not empty; pass --overwrite to replace it")
    out_dir.mkdir(parents=True, exist_ok=True)
    replaced = _clear_previous_extraction(out_dir) if overwrite else []
    if max_frames < 1 or max_output_gib <= 0 or min_free_gib < 0:
        raise ValueError("invalid extraction/storage bounds")
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    written_bytes = 0
    stop_reason = "end_of_recording"

    ctx = rs.context()
    dev = ctx.load_device(str(source))
    dev.as_playback().set_real_time(False)
    info = _device_info(dev)
    declared = {}
    depth_scale = None
    for sensor in dev.query_sensors():
        for p in sensor.get_stream_profiles():
            pd = _profile_dict(p)
            declared.setdefault(pd["stream"], []).append(pd)
        if sensor.is_depth_sensor():
            depth_scale = float(sensor.as_depth_sensor().get_depth_scale())
    if not {"depth", "color"} <= declared.keys() or depth_scale is None:
        raise ValueError("registration needs both depth and colour streams with a depth scale in the file")
    if declared["color"][0]["format"] != "rgb8":
        raise ValueError(f"unsupported colour format {declared['color'][0]['format']}; expected rgb8")
    del dev

    pipe = rs.pipeline(ctx)
    cfg = rs.config()
    cfg.enable_device_from_file(str(source), repeat_playback=False)
    profile = pipe.start(cfg)
    profile.get_device().as_playback().set_real_time(False)
    align = rs.align(rs.stream.color)
    arrival_key = rs.frame_metadata_value.time_of_arrival

    calibration = None
    records, skews_ms, dropped = [], [], []
    guard = OrderGuard()
    framesets = 0
    arrival_supported = {"depth": None, "color": None}
    extrinsics = None
    try:
        while True:
            ok, fs = pipe.try_wait_for_frames(3000)
            if not ok:
                break
            framesets += 1
            depth, color = fs.get_depth_frame(), fs.get_color_frame()
            if not depth or not color:
                dropped.append({"frameset": framesets, "reason": "incomplete_frameset"})
                continue
            if str(depth.get_frame_timestamp_domain()) != str(color.get_frame_timestamp_domain()):
                raise ValueError("depth and colour timestamps are in different domains; clocks are not mapped")
            if not timestamp_domain_ok(color.get_frame_timestamp_domain()):
                raise ValueError(f"timestamp domain {color.get_frame_timestamp_domain()} is not global_time; "
                                 "acquisition and host-arrival times would be on different clocks")
            if extrinsics is None:
                ex = depth.get_profile().get_extrinsics_to(color.get_profile())
                extrinsics = {"rotation": list(ex.rotation), "translation_m": list(ex.translation)}
            for key, frame in (("depth", depth), ("color", color)):
                if arrival_supported[key] is None:
                    arrival_supported[key] = bool(frame.supports_frame_metadata(arrival_key))
            rgb_stamp = stamp_from_frame("color", color.get_frame_number(), color.get_timestamp(), _metadata_ms(color, arrival_key))
            depth_stamp = stamp_from_frame("depth", depth.get_frame_number(), depth.get_timestamp(), _metadata_ms(depth, arrival_key))
            skew_ms = (rgb_stamp.acquired_s - depth_stamp.acquired_s) * 1000.0
            if abs(skew_ms) > max_sync_skew_s * 1000.0:
                dropped.append({"frameset": framesets, "reason": "sync_skew_exceeds_tolerance", "skew_ms": round(skew_ms, 3)})
                continue
            reason = guard.reason(rgb_stamp, depth_stamp)
            if reason:
                dropped.append({"frameset": framesets, "reason": reason})
                continue
            aligned = align.process(fs)
            adepth = aligned.get_depth_frame()
            if calibration is None:
                intr = adepth.get_profile().as_video_stream_profile().get_intrinsics()
                serial = info.get("serial_number", "unknown")
                calibration = rectified_calibration(
                    intr, depth_scale, f"d435-{serial}-fw{info.get('firmware_version', '?')}-color{intr.width}x{intr.height}-align", optical_frame)
            rgb = np.asanyarray(color.get_data())
            dep = np.asanyarray(adepth.get_data())
            if rgb.shape != (calibration.height, calibration.width, 3) or dep.shape != (calibration.height, calibration.width):
                raise ValueError("frame resolution changed inside the recording")
            expected_bytes = rgb.size + dep.size*2 + 1024
            if len(records) >= max_frames:
                stop_reason = "max_frames"; break
            if written_bytes + expected_bytes > max_output_gib*1024**3:
                stop_reason = "max_output_gib"; break
            if shutil.disk_usage(out_dir).free - expected_bytes < min_free_gib*1024**3:
                stop_reason = "minimum_free_space"; break
            index = len(records)
            rgb_path, depth_path = f"rgb_{index:04d}.npy", f"depth_{index:04d}.npy"
            np.save(out_dir / rgb_path, np.ascontiguousarray(rgb, dtype=np.uint8))
            np.save(out_dir / depth_path, np.ascontiguousarray(dep, dtype=np.uint16))
            written_bytes += (out_dir/rgb_path).stat().st_size + (out_dir/depth_path).stat().st_size
            records.append(dict(frame_id=f"{session_id}/frame-{index:04d}", rgb_path=rgb_path, depth_path=depth_path,
                                rgb_stamp=asdict(rgb_stamp), depth_stamp=asdict(depth_stamp),
                                registered_to_rgb=True, max_sync_skew_s=max_sync_skew_s))
            guard.admit(rgb_stamp, depth_stamp)
            skews_ms.append(skew_ms)
    finally:
        pipe.stop()
    if not records:
        reasons = sorted({d["reason"] for d in dropped})
        raise ValueError(f"no admissible RGB-D pairs in {framesets} framesets ({', '.join(reasons) or 'no framesets'}); nothing written")

    manifest = dict(schema=RECORDING_SCHEMA, participant_id=participant_id, session_id=session_id,
                    camera_model=f"{info.get('name', 'RealSense')} sn {info.get('serial_number', '?')} fw {info.get('firmware_version', '?')}",
                    calibration=asdict(calibration), frames=records)
    manifest_path = out_dir / "recording.json"
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")

    abs_skews = [abs(s) for s in skews_ms]
    provenance = {
        "adapter": "d435_recording_adapter.py",
        "stop_reason": stop_reason, "partial_extraction": stop_reason != "end_of_recording",
        "replaced_previous_extraction_files": len(replaced),
        "written_array_bytes": written_bytes,
        "bounds": dict(max_frames=max_frames,max_output_gib=max_output_gib,min_free_gib=min_free_gib),
        "librealsense": rs.__version__,
        "source_file": str(source), "source_sha256": digest, "source_bytes": source.stat().st_size,
        "device_info": info,
        "declared_profiles": declared,
        "extrinsics_depth_to_color_raw": extrinsics,
        "registration": "pyrealsense2.align(stream.color): depth reprojected into the colour camera at colour resolution; "
                        "calibration therefore equals the colour intrinsics",
        "timestamp_domain": "global_time", "clock": CLOCK,
        "availability_from_time_of_arrival_metadata": arrival_supported,
        "max_sync_skew_s": max_sync_skew_s,
        "framesets_read": framesets, "frames_written": len(records), "dropped": dropped,
        "skew_ms_written_pairs": {"median": round(statistics.median(skews_ms), 4) if skews_ms else None,
                                  "max_abs": round(max(abs_skews), 4) if abs_skews else None},
        "scope": "File contract only. Not a calibration, synchronization or scene validation; USB mode/cable/host "
                 "contention are recorded from the file header, not verified.",
    }
    (out_dir / "adapter_provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")

    # Read the result back through the reader so the emitted contract is exercised, not assumed.
    loaded = sum(1 for _ in RGBDRecording(manifest_path))
    if loaded != len(records):
        raise RuntimeError(f"reader loaded {loaded} frames but {len(records)} were written")
    return {"manifest": str(manifest_path), "frames_written": len(records), "framesets_read": framesets,
            "dropped": len(dropped), "skew_ms": provenance["skew_ms_written_pairs"], "reader_reloaded": loaded,
            "stop_reason": stop_reason, "partial_extraction": stop_reason != "end_of_recording"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recording", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--participant", required=True, help="pseudonymous participant identifier")
    ap.add_argument("--session", required=True, help="pseudonymous session identifier")
    ap.add_argument("--max-sync-skew-s", type=float, default=0.005, help="declared RGB/depth tolerance (configuration, not a validated bound)")
    ap.add_argument("--optical-frame", default="d435_color_optical")
    ap.add_argument("--max-frames",type=int,default=DEFAULT_MAX_FRAMES)
    ap.add_argument("--max-output-gib",type=float,default=12.)
    ap.add_argument("--min-free-gib",type=float,default=10.)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if not a.recording.exists():
        print(f"missing file: {a.recording}", file=sys.stderr)
        return 2
    summary = convert(a.recording, a.out_dir, a.participant, a.session, max_sync_skew_s=a.max_sync_skew_s,
                      optical_frame=a.optical_frame, overwrite=a.overwrite, max_frames=a.max_frames,
                      max_output_gib=a.max_output_gib, min_free_gib=a.min_free_gib)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
