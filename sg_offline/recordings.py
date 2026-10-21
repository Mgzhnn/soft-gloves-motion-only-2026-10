"""Versioned NumPy recording reader; native camera adapters remain separate."""
import json
from pathlib import Path

import numpy as np

from .inputs import stamp
from .rgbd import CameraCalibration, RGBDFrame

RECORDING_SCHEMA = "sg-rgbd-recording/1"


class RGBDRecording:
    def __init__(self, manifest_path):
        self.path = Path(manifest_path).resolve()
        data = json.loads(self.path.read_text())
        required = {"schema", "participant_id", "session_id", "camera_model", "calibration", "frames"}
        if set(data) != required or data["schema"] != RECORDING_SCHEMA:
            raise ValueError("unknown recording schema, missing fields or unsupported metadata")
        for key in ("participant_id", "session_id", "camera_model"):
            if not isinstance(data[key], str) or not data[key]:
                raise ValueError("recording identity and explicit camera model/unknown required")
        if not isinstance(data["frames"], list) or not data["frames"]:
            raise ValueError("recording needs a nonempty frame list")
        self.participant_id = data["participant_id"]
        self.session_id = data["session_id"]
        self.camera_model = data["camera_model"]
        self.calibration = CameraCalibration(**data["calibration"])
        self.records = data["frames"]

    def _array(self, relative_path):
        relative = Path(relative_path)
        path = (self.path.parent / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(self.path.parent) or path.suffix != ".npy":
            raise ValueError("array path must identify a local recording .npy file")
        return np.load(path, allow_pickle=False)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        data = dict(self.records[index])
        rgb = self._array(data.pop("rgb_path")); depth = self._array(data.pop("depth_path"))
        data["rgb_stamp"] = stamp(data["rgb_stamp"])
        data["depth_stamp"] = stamp(data["depth_stamp"])
        return RGBDFrame(rgb=rgb, depth=depth, calibration=self.calibration, **data)

    def __iter__(self):
        seen_ids = set()
        seen_samples = set()
        last_rgb = last_depth = last_available = -1.
        clock = None
        for record in self.records:
            data = dict(record)
            rgb_path = data.pop("rgb_path")
            depth_path = data.pop("depth_path")
            data["rgb_stamp"] = stamp(data["rgb_stamp"])
            data["depth_stamp"] = stamp(data["depth_stamp"])
            frame = RGBDFrame(rgb=self._array(rgb_path), depth=self._array(depth_path),
                              calibration=self.calibration, **data)
            if frame.frame_id in seen_ids:
                raise ValueError("duplicate frame identity")
            seen_ids.add(frame.frame_id)
            if clock is not None and frame.rgb_stamp.clock != clock:
                raise ValueError("recording clock changed without mapping")
            clock = frame.rgb_stamp.clock
            if (frame.rgb_stamp.acquired_s <= last_rgb or frame.depth_stamp.acquired_s <= last_depth
                    or frame.available_s < last_available):
                raise ValueError("recording must preserve causal acquisition/availability order")
            for stream, value in (("rgb", frame.rgb_stamp), ("depth", frame.depth_stamp)):
                key = (stream, value.sample_id)
                if key in seen_samples:
                    raise ValueError("recording reused a stream sample identity")
                seen_samples.add(key)
            last_rgb = frame.rgb_stamp.acquired_s
            last_depth = frame.depth_stamp.acquired_s
            last_available = frame.available_s
            yield frame
