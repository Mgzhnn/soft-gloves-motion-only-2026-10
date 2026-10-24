"""Calibrated, rectified RGB-D geometry and externally supplied camera poses."""
from dataclasses import dataclass, replace
from math import isfinite

import numpy as np

from .contracts import Point, Source, Stamp


def combined_source(*sources):
    if Source.MEASURED in sources:
        return Source.MEASURED
    if Source.RECORDED in sources:
        return Source.RECORDED
    return Source.SYNTHETIC


@dataclass(frozen=True)
class CameraCalibration:
    calibration_id: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale_m: float
    optical_frame: str
    projection: str = "rectified_pinhole"
    depth_convention: str = "optical_z"

    def __post_init__(self):
        if not self.calibration_id or not self.optical_frame:
            raise ValueError("calibration identity and optical frame required")
        if any(type(size) is not int or size <= 0 for size in (self.width, self.height)):
            raise ValueError("positive integer image dimensions required")
        if not all(isfinite(value) for value in (self.fx, self.fy, self.cx, self.cy, self.depth_scale_m)):
            raise ValueError("finite camera parameters required")
        if self.fx <= 0 or self.fy <= 0 or self.depth_scale_m <= 0:
            raise ValueError("positive focal lengths and depth scale required")
        if self.projection != "rectified_pinhole" or self.depth_convention != "optical_z":
            raise ValueError("only rectified pinhole pixels with optical-z depth are supported")


@dataclass(frozen=True)
class RGBDFrame:
    frame_id: str
    rgb: np.ndarray
    depth: np.ndarray
    rgb_stamp: Stamp
    depth_stamp: Stamp
    calibration: CameraCalibration
    registered_to_rgb: bool
    max_sync_skew_s: float
    invalid_depth_value: float = 0.

    def __post_init__(self):
        if not self.frame_id or self.registered_to_rgb is not True:
            raise ValueError("frame identity and explicit RGB registration required")
        if not isfinite(self.max_sync_skew_s) or self.max_sync_skew_s < 0:
            raise ValueError("finite nonnegative synchronization tolerance required")
        if self.rgb_stamp.clock != self.depth_stamp.clock:
            raise ValueError("RGB/depth clocks must already be mapped")
        if abs(self.rgb_stamp.acquired_s - self.depth_stamp.acquired_s) > self.max_sync_skew_s:
            raise ValueError("RGB/depth acquisition skew exceeds declared tolerance")
        expected = (self.calibration.height, self.calibration.width)
        if self.rgb.shape != (*expected, 3) or self.rgb.dtype != np.uint8:
            raise ValueError("RGB must be uint8 HxWx3 matching calibrated resolution")
        if self.depth.shape != expected or self.depth.dtype.kind not in "uif":
            raise ValueError("depth must be a numeric HxW array matching calibrated resolution")
        for field in ("rgb", "depth"):
            values = np.array(getattr(self, field), copy=True)
            values.setflags(write=False)
            object.__setattr__(self, field, values)

    @property
    def available_s(self):
        return max(self.rgb_stamp.available_s, self.depth_stamp.available_s)

    def _observation_stamp(self, identity, observation, decision_s):
        if not isfinite(decision_s) or decision_s < self.available_s:
            raise ValueError("frame is not available at decision time")
        if observation.clock != self.rgb_stamp.clock or observation.acquired_s != self.rgb_stamp.acquired_s:
            raise ValueError("pixel observation must refer to this RGB acquisition and clock")
        if observation.available_s < self.rgb_stamp.available_s or observation.available_s > decision_s:
            raise ValueError("pixel observation availability is inconsistent")
        return Stamp(
            f"{self.frame_id}:{identity}:{observation.sample_id}",
            self.depth_stamp.acquired_s, max(self.available_s, observation.available_s),
            combined_source(self.rgb_stamp.source, self.depth_stamp.source, observation.source),
            self.depth_stamp.clock,
        )

    def point_from_xyz(self, identity, xyz, *, observation, definition, decision_s, reason="", sigma_m=None):
        """A metric point already computed from THIS frame's registered depth (for example a region centroid
        from ``tableplane.region_geometry``), stamped exactly as ``point_at`` stamps a lifted pixel: the depth
        acquisition time, availability no earlier than the observation, this frame's optical frame and
        calibration. An unavailable point needs a reason. Used by the object providers."""
        stamp = self._observation_stamp(identity, observation, decision_s)
        if xyz is None:
            if not reason:
                raise ValueError("missing point requires reason")
            return Point(identity, None, stamp, frame=self.calibration.optical_frame, reason=reason,
                         definition=definition, calibration_id=self.calibration.calibration_id)
        coordinates = tuple(float(v) for v in xyz)
        if len(coordinates) != 3 or not all(isfinite(v) for v in coordinates) or coordinates[2] <= 0:
            raise ValueError("object point must be a finite xyz in front of the camera")
        return Point(identity, coordinates, stamp, frame=self.calibration.optical_frame, reason="",
                     definition=definition, calibration_id=self.calibration.calibration_id, sigma_m=sigma_m)

    def point_at(self, identity, pixel, *, observation, definition, decision_s):
        stamp = self._observation_stamp(identity, observation, decision_s)
        coordinates = None
        reason = "pixel_unavailable"
        if pixel is not None:
            if len(pixel) != 2 or any(type(value) is not int for value in pixel):
                raise ValueError("point sampling requires integer column,row or None")
            column, row = pixel
            if not (0 <= column < self.calibration.width and 0 <= row < self.calibration.height):
                reason = "pixel_outside_image"
            else:
                raw_depth = float(self.depth[row, column])
                depth_m = raw_depth * self.calibration.depth_scale_m
                if raw_depth == self.invalid_depth_value or not isfinite(depth_m) or depth_m <= 0:
                    reason = "invalid_depth"
                else:
                    coordinates = ((column - self.calibration.cx) * depth_m / self.calibration.fx,
                                   (row - self.calibration.cy) * depth_m / self.calibration.fy, depth_m)
                    reason = ""
        return Point(identity, coordinates, stamp, frame=self.calibration.optical_frame,
                     reason=reason, definition=definition, calibration_id=self.calibration.calibration_id)


@dataclass(frozen=True)
class CameraPose:
    pose_id: str
    source_frame: str
    target_frame: str
    rotation: tuple
    translation_m: tuple
    stamp: Stamp
    target_is_fixed: bool

    def __post_init__(self):
        if not self.pose_id or not self.source_frame or not self.target_frame or self.source_frame == self.target_frame:
            raise ValueError("pose requires identity and distinct named source/target frames")
        if self.target_is_fixed is not True:
            raise ValueError("pose target must be an explicitly fixed reference frame")
        rotation = np.asarray(self.rotation, dtype=float)
        translation = np.asarray(self.translation_m, dtype=float)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("pose requires a 3x3 rotation and metric xyz translation")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("pose must be finite")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-7, rtol=0):
            raise ValueError("rotation must be proper and orthonormal; no reflection or scale")
        object.__setattr__(self, "rotation", tuple(tuple(float(value) for value in row) for row in rotation))
        object.__setattr__(self, "translation_m", tuple(float(value) for value in translation))

    def transform(self, point, *, decision_s):
        if not isfinite(decision_s) or max(self.stamp.available_s, point.stamp.available_s) > decision_s:
            raise ValueError("point or pose is not yet available")
        if point.frame != self.source_frame or point.stamp.clock != self.stamp.clock:
            raise ValueError("pose/point frame or clock mismatch")
        if point.stamp.acquired_s != self.stamp.acquired_s:
            raise ValueError("pose must match the point acquisition; no stale pose reuse")
        coordinates = None if point.xyz is None else tuple(
            float(value) for value in np.asarray(self.rotation) @ point.xyz + self.translation_m)
        stamp = replace(point.stamp,
                        sample_id=f"{point.stamp.sample_id}|pose:{self.stamp.sample_id}",
                        available_s=max(self.stamp.available_s, point.stamp.available_s),
                        source=combined_source(point.stamp.source, self.stamp.source))
        return replace(point, xyz=coordinates, stamp=stamp, frame=self.target_frame,
                       frame_is_fixed=True, transform_id=self.pose_id, sigma_m=None)


def same_time_distance(first, second):
    if first.frame != second.frame or first.stamp.clock != second.stamp.clock:
        raise ValueError("relative distance requires a shared frame and clock")
    if first.stamp.acquired_s != second.stamp.acquired_s:
        raise ValueError("relative distance requires simultaneous point acquisitions")
    if first.xyz is None or second.xyz is None:
        return None
    return float(np.linalg.norm(np.asarray(first.xyz) - second.xyz))
