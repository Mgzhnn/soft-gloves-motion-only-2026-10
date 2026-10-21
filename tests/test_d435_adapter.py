"""Adapter contract checks: pure helpers always; the native sample only when it is present locally."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Without pyrealsense2 (the .venv-gpu environment) importing the adapter raised at collection and pytest then ran no test
# at all unless this file was passed to --ignore; the module is now skipped there (5 October 2026).
pytest.importorskip("pyrealsense2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import d435_recording_adapter as adapter  # noqa: E402
from sg_offline.contracts import Source  # noqa: E402
from sg_offline.recordings import RGBDRecording  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[5] / "data" / "d435_smoke_03.db3"  # repository root / data


def test_stamp_uses_arrival_for_availability_and_survives_counter_resets():
    early = adapter.stamp_from_frame("depth", 0, 1000.0, 1062.0)
    reset = adapter.stamp_from_frame("depth", 0, 1300.0, None)
    assert early.acquired_s == 1.0 and early.available_s == 1.062 and early.source is Source.RECORDED
    assert reset.available_s == reset.acquired_s == 1.3
    assert early.sample_id != reset.sample_id and early.clock == reset.clock == adapter.CLOCK


def test_arrival_before_acquisition_does_not_make_availability_precede_acquisition():
    s = adapter.stamp_from_frame("color", 7, 2000.0, 1990.0)
    assert s.available_s == s.acquired_s


def test_order_guard_names_each_violation():
    guard = adapter.OrderGuard()
    rgb = adapter.stamp_from_frame("color", 1, 1000.0, 1050.0)
    depth = adapter.stamp_from_frame("depth", 1, 1000.1, 1050.0)
    assert guard.reason(rgb, depth) == ""
    guard.admit(rgb, depth)
    assert guard.reason(rgb, depth) == "non_increasing_acquisition"
    late_arrival_rgb = adapter.stamp_from_frame("color", 2, 1066.0, 1300.0)  # host received this pair late
    late_arrival_depth = adapter.stamp_from_frame("depth", 2, 1066.1, 1300.0)
    assert guard.reason(late_arrival_rgb, late_arrival_depth) == ""
    guard.admit(late_arrival_rgb, late_arrival_depth)
    regressed = adapter.stamp_from_frame("color", 3, 1132.0, 1150.0)  # acquired later but available before the previous pair
    regressed_depth = adapter.stamp_from_frame("depth", 3, 1132.1, 1150.0)
    assert guard.reason(regressed, regressed_depth) == "availability_regressed"
    reused = adapter.OrderGuard()
    reused.admit(rgb, depth)
    fresh_depth = adapter.stamp_from_frame("depth", 9, 1200.0, 1250.0)
    assert reused.reason(adapter.stamp_from_frame("color", 1, 1000.0, 1050.0), fresh_depth) == "non_increasing_acquisition"


def test_only_rectified_intrinsics_become_a_calibration():
    good = SimpleNamespace(width=640, height=480, fx=614.4, fy=614.4, ppx=320.8, ppy=238.3, coeffs=[0.0] * 5)
    cal = adapter.rectified_calibration(good, 0.001, "cal", "d435_color_optical")
    assert (cal.width, cal.height, cal.depth_scale_m, cal.projection) == (640, 480, 0.001, "rectified_pinhole")
    distorted = SimpleNamespace(**{**good.__dict__, "coeffs": [0.1, 0.0, 0.0, 0.0, 0.0]})
    with pytest.raises(ValueError):
        adapter.rectified_calibration(distorted, 0.001, "cal", "d435_color_optical")


@pytest.mark.skipif(not SAMPLE.exists(), reason="local D435 sample not present; native conversion needs it")
def test_native_sample_converts_and_reloads_through_reader(tmp_path):
    pytest.importorskip("pyrealsense2")
    summary = adapter.convert(SAMPLE, tmp_path, "dev-operator", "test-session", max_sync_skew_s=0.005,
                              optical_frame="d435_color_optical", overwrite=True)
    assert summary["frames_written"] == summary["reader_reloaded"] > 0
    assert summary["skew_ms"]["max_abs"] <= 5.0
    assert summary["stop_reason"] == "end_of_recording" and summary["partial_extraction"] is False  # printed, not hidden
    manifest = json.loads((tmp_path / "recording.json").read_text())
    assert set(manifest) == {"schema", "participant_id", "session_id", "camera_model", "calibration", "frames"}
    frames = list(RGBDRecording(tmp_path / "recording.json"))
    assert frames[0].rgb.shape == (480, 640, 3) and frames[0].depth.shape == (480, 640)
    provenance = json.loads((tmp_path / "adapter_provenance.json").read_text())
    assert provenance["source_sha256"] and provenance["registration"].startswith("pyrealsense2.align")
    assert all(d["reason"] in {"sync_skew_exceeds_tolerance", "incomplete_frameset", "non_increasing_acquisition",
                               "availability_regressed", "reused_sample_id"} for d in provenance["dropped"])
