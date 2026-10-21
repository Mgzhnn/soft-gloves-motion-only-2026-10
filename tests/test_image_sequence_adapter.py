"""Image-sequence export adapter and configurable colour targets (2 October 2026). Synthetic files, N=0."""
import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from sg_offline.perception import ColourObjects, DEFAULT_COLOUR_TARGETS
from sg_offline.recordings import RGBDRecording
from sg_offline.rgbd import CameraCalibration
import image_sequence_adapter as adapter


def make_export(folder: Path, n=4, width=64, height=48, saturate_frame=0, skew_frame=None):
    """A tiny export in the 29 Sep folder layout: depth PNGs, a colour avi, three timestamp tables."""
    stem = "20260929_000000"
    depth_dir = folder / f"{stem}_camera_depth"; depth_dir.mkdir(parents=True)
    (folder / f"{stem}_camera_depth_filled").mkdir()
    writer = cv2.VideoWriter(str(folder / f"{stem}_camera_rgb.avi"), cv2.VideoWriter_fourcc(*"mp4v"), 30., (width, height))
    if not writer.isOpened():
        pytest.skip("OpenCV build cannot write mp4v video")
    t0 = 1790644350.0
    color, depth_t, host = [], [], []
    for i in range(n):
        frame = np.full((height, width, 3), 40, np.uint8); frame[10:30, 10:30] = (200, 30, 30)  # a blue patch (BGR)
        writer.write(frame)
        depth = np.full((height, width), 600 + i, np.uint16); depth[0, 0] = 0
        if i == saturate_frame:
            depth[5:8, 5:8] = 65535
        cv2.imwrite(str(depth_dir / f"{i}.png"), depth)
        t = t0 + i / 30.
        color.append(t); depth_t.append(t + (0.2 if i == skew_frame else 0.001)); host.append(t + 0.003)
    writer.release()
    for name, values in (("camera_color_timestamp", color), ("camera_depth_timestamp", depth_t)):
        (folder / f"{stem}_{name}.txt").write_text("realsense_time\tframe_index\n" + "".join(f"{v:.6f}\t{i}\n" for i, v in enumerate(values)))
    (folder / f"{stem}_camera.txt").write_text("host_time\tframe_index\n" + "".join(f"{v:.6f}\t{i}\n" for i, v in enumerate(host)))
    return stem


def calibration(width=64, height=48):
    return CameraCalibration("test-cal", width, height, 60., 60., width / 2, height / 2, .001, "d435_color_optical")


def test_adapter_writes_a_loadable_manifest_and_declares_what_it_cannot_measure(tmp_path):
    make_export(tmp_path / "trial", skew_frame=2)
    out = tmp_path / "out"
    summary = adapter.convert(tmp_path / "trial", out, "p01", "s01", calibration=calibration(), calibration_source="test",
                              registered_to_rgb=True, recording_tool="synthetic")
    assert summary["frames_in_tables"] == 4 and summary["frames_written"] == 3 and summary["dropped"] == 1  # the skewed pair is dropped, not kept
    recording = RGBDRecording(out / "recording.json")
    frames = list(recording)
    assert len(frames) == 3 and frames[0].depth.dtype == np.uint16 and frames[0].rgb.shape == (48, 64, 3)
    assert frames[0].rgb_stamp.clock == adapter.DEFAULT_CLOCK and frames[0].rgb_stamp.available_s > frames[0].rgb_stamp.acquired_s  # host arrival
    assert int(frames[0].depth[5, 5]) == 0 and summary["saturated_pixels_zeroed"] == 9                                  # 65535 -> invalid
    assert int(frames[1].depth[20, 20]) == 601
    prov = json.loads((out / "adapter_provenance.json").read_text())
    assert prov["declared"]["calibration_source"] == "test" and prov["filled_depth_folder_ignored"] is True
    assert prov["dropped"][0]["reason"] == "sync_skew_exceeds_tolerance" and prov["recording_tool"] == "synthetic"
    assert prov["source_frame_index_by_frame_id"]["s01/frame-0002"] == 3  # frame 2 of the export was dropped, so written frame 2 is export frame 3
    assert "video" in prov and prov["video"]["reported_frames"] == 4
    with pytest.raises(ValueError, match="registered"):
        adapter.convert(tmp_path / "trial", tmp_path / "out2", "p01", "s01", calibration=calibration(), calibration_source="test", registered_to_rgb=False)


def test_adapter_refuses_mismatched_tables_and_resolution(tmp_path):
    stem = make_export(tmp_path / "trial")
    (tmp_path / "trial" / f"{stem}_camera.txt").write_text("host_time\tframe_index\n1.0\t0\n")
    with pytest.raises(ValueError, match="disagree"):
        adapter.convert(tmp_path / "trial", tmp_path / "out", "p", "s", calibration=calibration(), calibration_source="t", registered_to_rgb=True)
    make_export(tmp_path / "trial2")
    with pytest.raises(ValueError, match="resolution"):
        adapter.convert(tmp_path / "trial2", tmp_path / "out3", "p", "s", calibration=calibration(width=32), calibration_source="t", registered_to_rgb=True)


def test_colour_targets_are_configurable_and_recorded(tmp_path):
    rgb = np.zeros((40, 60, 3), np.uint8); rgb[5:20, 5:20] = (30, 30, 200)          # a blue patch (RGB)
    depth = np.full((40, 60), 700, np.uint16)
    default = ColourObjects(min_area=10).locate(rgb, depth)
    assert set(default) == set(DEFAULT_COLOUR_TARGETS) and all(v[0] is None for v in default.values())  # no green, no red
    targets = {"blue_cup": [[[95, 80, 60], [130, 255, 255]]]}
    path = tmp_path / "objects.json"; path.write_text(json.dumps({"targets": targets}))
    custom = ColourObjects.from_json(path, min_area=10)
    found = custom.locate(rgb, depth)
    assert set(found) == {"blue_cup"} and found["blue_cup"][0] is not None and 5 <= found["blue_cup"][0][0] < 20
    assert custom.targets == {"blue_cup": [((95, 80, 60), (130, 255, 255))]}
    with pytest.raises(ValueError):
        ColourObjects(targets={"left": [((0, 0, 0), (1, 1, 1))]})   # a hand name cannot be an object identity
    with pytest.raises(ValueError):
        ColourObjects(targets={"cup": []})
