"""Convert an image-sequence D435 export into an `sg-rgbd-recording/1` manifest.

Folder contract (as recorded on 2026-09-29 with the lab's capture script; tool name to be confirmed):
    <stem>_camera_depth/<i>.png          uint16 depth, one file per frame (D435 z16 in depth units)
    <stem>_camera_depth_filled/<i>.png   hole-filled copy; NOT used (interpolated values are not measurements)
    <stem>_camera_rgb.avi                colour video, one frame per depth image, lossy (FMP4 in the 29 Sep files)
    <stem>_camera_color_timestamp.txt    realsense_time <tab> frame_index   (epoch seconds)
    <stem>_camera_depth_timestamp.txt    realsense_time <tab> frame_index
    <stem>_camera.txt                    host_time <tab> frame_index        (arrival on the recording host)

The export carries no calibration, so this adapter DECLARES what `d435_recording_adapter.py` could read from a
native file, and writes every declaration to `adapter_provenance.json` next to the manifest:
  * the colour-camera intrinsics and their source (`--fx --fy --cx --cy --calibration-source`),
  * that the depth images are registered to the colour image (`--registered-to-rgb`, required; for the
    29 Sep files this was checked visually: depth edges trace the colour objects),
  * the depth scale (`--depth-scale`, D435 default 0.001 m per unit),
  * the clock: RealSense timestamps are taken as the global-time domain in epoch seconds, host arrival gives
    availability (`--clock`),
  * that avi frame i, depth image i and timestamp row i are the same instant (the tables share frame_index).
Depth values equal to the sensor saturation code 65535 are stored as invalid (0) and counted. Pairs whose
colour/depth timestamps differ by more than the declared tolerance are dropped and listed, as in the native
adapter. Colour is lossy video: the extracted RGB is what the codec kept, which the provenance states.
This establishes the file contract only; it is not a calibration, synchronization or scene validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sg_offline.contracts import Source, Stamp
from sg_offline.recordings import RECORDING_SCHEMA, RGBDRecording
from sg_offline.rgbd import CameraCalibration

DEFAULT_CLOCK = "d435_global_time_epoch_s"  # the native adapter's clock name; same domain assumed here
SATURATION = 65535
EXTRACTION_FILE_PATTERNS = ("rgb_*.npy", "depth_*.npy", "recording.json", "adapter_provenance.json")


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


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_table(path: Path, value_name: str) -> np.ndarray:
    """A two-column text table `<value_name> <tab> frame_index`; returns values indexed by frame_index 0..N-1."""
    lines = path.read_text().strip().splitlines()
    header = lines[0].split()
    if header != [value_name, "frame_index"]:
        raise ValueError(f"{path.name}: expected header '{value_name} frame_index', found {header}")
    rows = [l.split() for l in lines[1:]]
    index = np.array([int(r[1]) for r in rows]); values = np.array([float(r[0]) for r in rows])
    if not np.array_equal(index, np.arange(len(index))):
        raise ValueError(f"{path.name}: frame_index must run 0..N-1 without gaps")
    invalid = np.flatnonzero(~np.isfinite(values) | (values < 0))
    if len(invalid):
        raise ValueError(f"{path.name}: finite nonnegative timestamps required at frame_index {int(invalid[0])}")
    return values


def discover(folder: Path) -> dict:
    camera = sorted(folder.glob("*_camera.txt"))
    if len(camera) != 1:
        raise ValueError(f"{folder}: expected exactly one *_camera.txt, found {len(camera)}")
    stem = camera[0].name[:-len("_camera.txt")]
    files = dict(stem=stem, host=camera[0], color_ts=folder / f"{stem}_camera_color_timestamp.txt",
                 depth_ts=folder / f"{stem}_camera_depth_timestamp.txt", avi=folder / f"{stem}_camera_rgb.avi",
                 depth_dir=folder / f"{stem}_camera_depth", filled_dir=folder / f"{stem}_camera_depth_filled")
    for key in ("color_ts", "depth_ts", "avi", "depth_dir"):
        if not files[key].exists():
            raise ValueError(f"{folder}: missing {files[key].name}")
    return files


def _clear_previous_extraction(out_dir: Path) -> list[str]:
    removed = []
    for pattern in EXTRACTION_FILE_PATTERNS:
        for path in sorted(out_dir.glob(pattern)):
            if path.is_file():
                path.unlink(); removed.append(path.name)
    return removed


def convert(folder: Path, out_dir: Path, participant_id: str, session_id: str, *, calibration: CameraCalibration,
            calibration_source: str, registered_to_rgb: bool, max_sync_skew_s: float = .005, clock: str = DEFAULT_CLOCK,
            overwrite: bool = False, max_frames: int | None = None, max_output_gib: float = 12., min_free_gib: float = 10.,
            recording_tool: str = "unknown") -> dict:
    import cv2  # kept out of sg_offline

    if registered_to_rgb is not True:
        raise ValueError("this export has no registration record; pass --registered-to-rgb only after checking that depth edges trace the colour objects")
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_dir} is not empty; pass --overwrite to replace it")
    out_dir.mkdir(parents=True, exist_ok=True)
    replaced = _clear_previous_extraction(out_dir) if overwrite else []
    files = discover(folder)
    color_ts = read_table(files["color_ts"], "realsense_time")
    depth_ts = read_table(files["depth_ts"], "realsense_time")
    host_ts = read_table(files["host"], "host_time")
    depth_files = {int(p.stem): p for p in files["depth_dir"].glob("*.png") if p.stem.isdigit()}
    n = len(color_ts)
    if not (len(depth_ts) == len(host_ts) == n) or set(depth_files) != set(range(n)):
        raise ValueError(f"table lengths {len(color_ts)}/{len(depth_ts)}/{len(host_ts)} and {len(depth_files)} depth images disagree")
    cap = cv2.VideoCapture(str(files["avi"]))
    if not cap.isOpened():
        raise ValueError(f"cannot open {files['avi'].name}")
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)); codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    records, skews_ms, dropped = [], [], []
    guard = OrderGuard(); written_bytes = 0; stop_reason = "end_of_recording"; saturated_pixels = 0; frames_with_saturation = 0
    try:
        for i in range(n):
            ok, bgr = cap.read()
            if not ok:
                stop_reason = "video_ended_before_tables"; dropped.append({"frame": i, "reason": "no_video_frame"}); break
            rgb_stamp = Stamp(f"color#{i}@{color_ts[i]:.6f}", float(color_ts[i]), float(max(color_ts[i], host_ts[i])), Source.RECORDED, clock)
            depth_stamp = Stamp(f"depth#{i}@{depth_ts[i]:.6f}", float(depth_ts[i]), float(max(depth_ts[i], host_ts[i])), Source.RECORDED, clock)
            skew_ms = (rgb_stamp.acquired_s - depth_stamp.acquired_s) * 1000.
            if abs(skew_ms) > max_sync_skew_s * 1000.:
                dropped.append({"frame": i, "reason": "sync_skew_exceeds_tolerance", "skew_ms": round(skew_ms, 3)}); continue
            reason = guard.reason(rgb_stamp, depth_stamp)
            if reason:
                dropped.append({"frame": i, "reason": reason}); continue
            depth = cv2.imread(str(depth_files[i]), cv2.IMREAD_UNCHANGED)
            if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
                raise ValueError(f"{depth_files[i].name}: expected a 2-D uint16 PNG")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape != (calibration.height, calibration.width, 3) or depth.shape != (calibration.height, calibration.width):
                raise ValueError(f"frame {i}: resolution {rgb.shape[:2]} / {depth.shape} differs from the declared calibration {calibration.height}x{calibration.width}")
            saturated = int((depth == SATURATION).sum())
            if saturated:
                depth = depth.copy(); depth[depth == SATURATION] = 0; saturated_pixels += saturated; frames_with_saturation += 1
            if max_frames is not None and len(records) >= max_frames:
                stop_reason = "max_frames"; break
            expected_bytes = rgb.size + depth.size * 2 + 1024
            if written_bytes + expected_bytes > max_output_gib * 1024 ** 3:
                stop_reason = "max_output_gib"; break
            if shutil.disk_usage(out_dir).free - expected_bytes < min_free_gib * 1024 ** 3:
                stop_reason = "minimum_free_space"; break
            index = len(records)
            rgb_path, depth_path = f"rgb_{index:04d}.npy", f"depth_{index:04d}.npy"
            np.save(out_dir / rgb_path, np.ascontiguousarray(rgb, dtype=np.uint8))
            np.save(out_dir / depth_path, np.ascontiguousarray(depth, dtype=np.uint16))
            written_bytes += (out_dir / rgb_path).stat().st_size + (out_dir / depth_path).stat().st_size
            records.append(dict(frame_id=f"{session_id}/frame-{index:04d}", source_frame_index=i, rgb_path=rgb_path, depth_path=depth_path,
                                rgb_stamp=asdict(rgb_stamp), depth_stamp=asdict(depth_stamp),
                                registered_to_rgb=True, max_sync_skew_s=max_sync_skew_s))
            guard.admit(rgb_stamp, depth_stamp); skews_ms.append(skew_ms)
    finally:
        cap.release()
    if not records:
        raise ValueError(f"no admissible RGB-D pairs ({sorted({d['reason'] for d in dropped})}); nothing written")
    # `source_frame_index` is provenance; the reader's frame schema does not admit extra keys, so it goes to the sidecar.
    source_index = {r["frame_id"]: r.pop("source_frame_index") for r in records}
    manifest = dict(schema=RECORDING_SCHEMA, participant_id=participant_id, session_id=session_id,
                    camera_model=f"RealSense D435 image-sequence export ({files['stem']}); serial not in export",
                    calibration=asdict(calibration), frames=records)
    (out_dir / "recording.json").write_text(json.dumps(manifest, indent=1) + "\n")
    abs_skews = [abs(s) for s in skews_ms]
    gaps = np.diff([r["rgb_stamp"]["acquired_s"] for r in records]) if len(records) > 1 else np.array([])
    provenance = {
        "adapter": "image_sequence_adapter.py", "recording_tool": recording_tool,
        "source_folder": str(folder.resolve()), "source_stem": files["stem"],
        "source_sha256": {k: sha256_file(files[k]) for k in ("color_ts", "depth_ts", "host", "avi")},
        "depth_png_sha256_of_concatenated_hashes": hashlib.sha256("".join(sha256_file(depth_files[i]) for i in range(n)).encode()).hexdigest(),
        "filled_depth_folder_ignored": files["filled_dir"].exists(),
        "video": {"codec": codec, "reported_frames": reported_frames, "note": "lossy video; the extracted RGB is what the codec kept"},
        "declared": {
            "calibration_source": calibration_source,
            "registered_to_rgb": "declared by the operator after a visual check (depth edges trace the colour objects); the export has no registration record",
            "depth_scale_m": calibration.depth_scale_m,
            "clock": clock, "clock_note": "RealSense timestamps taken as global-time epoch seconds; availability = host arrival time (camera.txt)",
            "frame_correspondence": "avi frame i == depth/<i>.png == timestamp row i (tables share frame_index)",
            "saturation_code_65535_stored_as_invalid_0": True,
        },
        "stop_reason": stop_reason, "partial_extraction": stop_reason != "end_of_recording",
        "replaced_previous_extraction_files": len(replaced), "written_array_bytes": written_bytes,
        "bounds": dict(max_frames=max_frames, max_output_gib=max_output_gib, min_free_gib=min_free_gib),
        "max_sync_skew_s": max_sync_skew_s, "frames_in_tables": n, "frames_written": len(records), "dropped": dropped,
        "saturated_pixels_zeroed": saturated_pixels, "frames_with_saturation": frames_with_saturation,
        "skew_ms_written_pairs": {"median": round(float(np.median(skews_ms)), 4) if skews_ms else None,
                                  "max_abs": round(float(max(abs_skews)), 4) if abs_skews else None},
        "frame_interval_s": {"median": round(float(np.median(gaps)), 5) if gaps.size else None, "max": round(float(gaps.max()), 4) if gaps.size else None},
        "source_frame_index_by_frame_id": source_index,
        "scope": "File contract only. Not a calibration, synchronization or scene validation; the intrinsics and registration are declarations to be confirmed with the recording script.",
    }
    (out_dir / "adapter_provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    loaded = sum(1 for _ in RGBDRecording(out_dir / "recording.json"))
    if loaded != len(records):
        raise RuntimeError(f"reader loaded {loaded} frames but {len(records)} were written")
    return {"manifest": str(out_dir / "recording.json"), "frames_written": len(records), "frames_in_tables": n,
            "dropped": len(dropped), "skew_ms": provenance["skew_ms_written_pairs"], "frame_interval_s": provenance["frame_interval_s"],
            "saturated_pixels_zeroed": saturated_pixels, "reader_reloaded": loaded, "stop_reason": stop_reason,
            "partial_extraction": stop_reason != "end_of_recording"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", type=Path); ap.add_argument("out_dir", type=Path)
    ap.add_argument("--participant", required=True); ap.add_argument("--session", required=True)
    ap.add_argument("--fx", type=float, required=True); ap.add_argument("--fy", type=float, required=True)
    ap.add_argument("--cx", type=float, required=True); ap.add_argument("--cy", type=float, required=True)
    ap.add_argument("--width", type=int, default=640); ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--depth-scale", type=float, default=.001, help="metres per depth unit (D435 default 0.001)")
    ap.add_argument("--calibration-id", required=True); ap.add_argument("--calibration-source", required=True,
                    help="where the intrinsics come from, e.g. 'device record d435_smoke_03.inspect.json colour 640x480; same camera assumed'")
    ap.add_argument("--optical-frame", default="d435_color_optical")
    ap.add_argument("--registered-to-rgb", action="store_true", help="declare that depth is registered to colour (required)")
    ap.add_argument("--recording-tool", default="unknown")
    ap.add_argument("--clock", default=DEFAULT_CLOCK)
    ap.add_argument("--max-sync-skew-s", type=float, default=.005)
    ap.add_argument("--max-frames", type=int); ap.add_argument("--max-output-gib", type=float, default=12.); ap.add_argument("--min-free-gib", type=float, default=10.)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if not a.folder.is_dir():
        print(f"missing folder: {a.folder}", file=sys.stderr); return 2
    calibration = CameraCalibration(a.calibration_id, a.width, a.height, a.fx, a.fy, a.cx, a.cy, a.depth_scale, a.optical_frame)
    summary = convert(a.folder, a.out_dir, a.participant, a.session, calibration=calibration, calibration_source=a.calibration_source,
                      registered_to_rgb=a.registered_to_rgb, max_sync_skew_s=a.max_sync_skew_s, clock=a.clock, overwrite=a.overwrite,
                      max_frames=a.max_frames, max_output_gib=a.max_output_gib, min_free_gib=a.min_free_gib, recording_tool=a.recording_tool)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
