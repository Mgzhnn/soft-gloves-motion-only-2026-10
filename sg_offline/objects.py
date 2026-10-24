"""Object providers for the object-perception comparison.

* ``PlaneClusterObjects`` (B, the matched simple comparator): RANSAC table plane, clusters above it with arm,
  hand and far-field removal (``tableplane.clusters``), identities kept by nearest 3-D centroid.
* ``MaskObjects`` (C + B): per-frame object masks produced offline by ``segment_objects.py`` (SAM 2), each
  localised in 3-D from the registered depth above the table plane.

Both return ``{identity: ObjectObservation}`` for every identity they know, available or not, so the packet
always lists the same objects and every arm of the comparison sees identical observations. Nothing here
holds a position over from an earlier frame: an occluded or merged object is unavailable with a reason,
because the geometry contract admits only same-time observations.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .appearance import global_assignment
from .contracts import SIDES
from .perception import ObjectObservation
from .rgbd import CameraCalibration
from .tableplane import backproject, clusters, fit_table_plane, region_geometry, table_support, table_surface


MASK_SCHEMA = "sg-object-masks/1"  # written by segment_objects.py and checked by MaskObjects (one source for both)


@dataclass(frozen=True)
class PlaneSettings:
    """Batch settings of the table-plane geometry; record them with every run, never tune them per clip."""
    inlier_m: float = 0.01
    iterations: int = 200
    max_samples: int = 4000
    min_inlier_fraction: float = 0.3
    min_height_m: float = 0.015
    max_height_m: float = 0.6
    surface_tolerance_m: float = 0.015
    min_area_px: int = 300
    ring_px: int = 7
    min_ring_surface_fraction: float = 0.2
    max_range_m: float = 1.2
    hand_box_overlap: float = 0.5
    min_points: int = 20

    def __post_init__(self):
        if not 0 <= self.min_height_m < self.max_height_m or self.inlier_m <= 0 or self.min_area_px < 1 or self.max_range_m <= 0:
            raise ValueError("invalid plane settings")


class TablePlaneObserver:
    """Fits the table plane of one registered depth frame deterministically (seed = source frame index)."""

    def __init__(self, calibration: CameraCalibration, settings: PlaneSettings = PlaneSettings()):
        self.calibration = calibration
        self.settings = settings

    def observe(self, depth, seed: int, invalid=0):
        """``invalid`` is the frame's declared invalid-depth marker (``RGBDFrame.invalid_depth_value``)."""
        s = self.settings
        points, valid = backproject(depth, self.calibration, invalid)
        plane = fit_table_plane(points, valid, seed=seed, iterations=s.iterations, inlier_m=s.inlier_m,
                                max_samples=s.max_samples, min_inlier_fraction=s.min_inlier_fraction)
        if plane is None:
            return dict(points=points, valid=valid, plane=None, support=None, surface=None)
        return dict(points=points, valid=valid, plane=plane,
                    support=table_support(points, valid, plane, s.min_height_m, s.max_height_m),
                    surface=table_surface(points, valid, plane, s.surface_tolerance_m))


def _plane_summary(plane):
    return None if plane is None else dict(inlier_fraction=round(plane.inlier_fraction, 3), tilt_deg=round(plane.tilt_deg, 1),
                                           camera_height_m=round(plane.offset, 3), rmse_m=round(plane.rmse_m, 4))


class PlaneClusterObjects:
    """Plane-only comparator with generic identities that persist only while an object stays in view.

    Kept clusters are matched to existing tracks by 3-D centroid distance (``gate_m``); unmatched clusters open a
    new identity while fewer than ``max_objects`` exist; a track that has been missing for ``reacquire_after``
    frames may adopt an unmatched cluster beyond the gate (an object that was moved while out of view).
    Identities are generic (``object_1`` …); the provider cannot know what the objects are.

    Known limits (measured; behaviour unchanged): a track's last position
    is kept in camera coordinates however old it is, so after a gap an identity can adopt a DIFFERENT physical
    object that appears within the gate, without the ``reacquired`` flag (trial_2: ``object_2`` is the cream
    cylinder in 185 and the blue one in 15 of its 212 available frames; 160 of the 173 two-object frames are the
    two cylinders). Only the four largest kept clusters reach the matcher. Read per-identity counts of this
    comparator as "an object-like cluster", not as one object.
    """

    def __init__(self, calibration: CameraCalibration, settings: PlaneSettings = PlaneSettings(), *, max_objects=2,
                 gate_m=0.15, reacquire_after=30, name_prefix="object"):
        if not 1 <= max_objects <= 4 or gate_m <= 0 or reacquire_after < 1:
            raise ValueError("invalid tracker settings (at most four objects)")
        self.observer = TablePlaneObserver(calibration, settings)
        self.settings = settings
        self.max_objects, self.gate_m, self.reacquire_after, self.prefix = max_objects, gate_m, reacquire_after, name_prefix
        self.tracks: dict[str, dict] = {}  # identity -> {"xyz": last centroid, "missing": frames since seen}
        self.name = "plane_clusters"

    def _candidates(self, scene, hand_boxes):
        s = self.settings
        found = clusters(scene["support"], min_area=s.min_area_px, exclude_boxes=hand_boxes, box_overlap=s.hand_box_overlap,
                         surface=scene["surface"], ring_px=s.ring_px, min_ring_surface_fraction=s.min_ring_surface_fraction,
                         points=scene["points"], max_range_m=s.max_range_m)
        out = []
        for c in found:
            if c["excluded"]:
                continue
            geo = region_geometry(scene["points"], scene["valid"], scene["plane"], c["mask"], s.min_points, s.min_height_m)
            if geo is not None:
                out.append((c, geo))
        return out[:4], found  # global_assignment handles at most four candidates

    def observe(self, depth, hand_boxes=(), *, seed: int, invalid=0):
        scene = self.observer.observe(depth, seed, invalid)
        observations = {}
        if scene["plane"] is None:
            for identity in self.tracks:
                self.tracks[identity]["missing"] += 1
                observations[identity] = ObjectObservation(identity, None, reason="plane_unavailable", details=dict(source=self.name, plane=None))
            return observations
        candidates, found = self._candidates(scene, hand_boxes)
        excluded = {}
        for c in found:
            if c["excluded"]:
                excluded[c["excluded"]] = excluded.get(c["excluded"], 0) + 1
        identities = list(self.tracks)
        assignment = [None] * len(identities)
        if identities and candidates:
            costs = np.array([[float(np.linalg.norm(np.asarray(self.tracks[i]["xyz"]) - geo["centroid_xyz"])) for _, geo in candidates]
                              for i in identities])
            assignment, _ = global_assignment(costs, unmatched_cost=self.gate_m)
            assignment = list(assignment)  # global_assignment returns a tuple; later steps append and reassign
        used = {j for j in assignment if j is not None}
        # re-acquisition: long-missing tracks adopt leftover clusters beyond the gate
        for k, identity in enumerate(identities):
            if assignment[k] is None and self.tracks[identity]["missing"] >= self.reacquire_after:
                leftovers = [j for j in range(len(candidates)) if j not in used]
                if leftovers:
                    j = min(leftovers, key=lambda j: float(np.linalg.norm(np.asarray(self.tracks[identity]["xyz"]) - candidates[j][1]["centroid_xyz"])))
                    assignment[k] = j; used.add(j); self.tracks[identity]["reacquired"] = True
        # new identities for leftover clusters while capacity remains
        for j in range(len(candidates)):
            if j not in used and len(self.tracks) < self.max_objects:
                identity = f"{self.prefix}_{len(self.tracks) + 1}"
                self.tracks[identity] = dict(xyz=candidates[j][1]["centroid_xyz"], missing=0, reacquired=False)
                identities.append(identity); assignment.append(j); used.add(j)
        ignored = len(candidates) - len(used)
        for k, identity in enumerate(identities):
            j = assignment[k]
            track = self.tracks[identity]
            if j is None:
                track["missing"] += 1
                observations[identity] = ObjectObservation(identity, None, reason="cluster_not_found_occluded_or_merged",
                                                           details=dict(source=self.name, missing_frames=track["missing"], plane=_plane_summary(scene["plane"])))
                continue
            c, geo = candidates[j]
            reacquired = track.pop("reacquired", False)
            track.update(xyz=geo["centroid_xyz"], missing=0)
            observations[identity] = ObjectObservation(identity, geo["centroid_xyz"], pixel=geo["pixel"],
                                                       details=dict(source=self.name, height_m=round(geo["height_m"], 4), footprint_extent_m=round(geo["footprint_extent_m"], 4),
                                                                    n_points=geo["n_points"], area_px=c["area"], ring_surface_fraction=c["ring_surface_fraction"],
                                                                    reacquired=reacquired, extra_clusters_ignored=ignored, excluded_clusters=excluded,
                                                                    plane=_plane_summary(scene["plane"])))
        return observations


class MaskObjects:
    """Segmenter masks plus table-plane geometry (C + B).

    ``masks_dir`` is the output of ``segment_objects.py``: ``objects.json`` (schema ``sg-object-masks/1``) names the
    label values and ``masks/<source_index:05d>.png`` holds one uint8 label image per source frame (0 = background).
    The reader checks the schema and the names; ``run_perception.build_object_provider`` also checks that the set
    was segmented from the recording being processed. A mask file missing inside the range the set declares as
    written is an error; outside it (frames before the prompt frame) the objects are ``mask_file_absent``.
    """

    def __init__(self, calibration: CameraCalibration, masks_dir, settings: PlaneSettings = PlaneSettings()):
        self.masks_dir = Path(masks_dir)
        meta = json.loads((self.masks_dir / "objects.json").read_text())
        if meta.get("schema") != MASK_SCHEMA:
            raise ValueError(f"objects.json: mask-set schema {meta.get('schema')!r} is not {MASK_SCHEMA}")
        self.labels = {int(k): str(v) for k, v in meta["objects"].items()}
        if not self.labels or any(k < 1 for k in self.labels):
            raise ValueError("objects.json must map positive label values to object names")
        names = list(self.labels.values())
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("objects.json must give every label a distinct object name")
        if any(name in SIDES for name in names):
            raise ValueError("an object must not carry a hand name")
        first, written = meta.get("first_source_index"), meta.get("frames_written")
        # frames the set declares as written (every frame from the prompt frame on); None for a set without the record
        self.written_range = (first, first + written) if type(first) is int and type(written) is int else None
        self.pattern = meta.get("mask_files", "masks/{source_index:05d}.png")
        self.observer = TablePlaneObserver(calibration, settings)
        self.settings = settings
        self.provenance = meta
        self.name = "sam2_masks_plus_plane"

    def mask_path(self, source_index: int) -> Path:
        return self.masks_dir / self.pattern.format(source_index=source_index)

    def observe(self, depth, source_index: int, *, seed: int | None = None, invalid=0):
        import cv2
        seed = source_index if seed is None else seed
        path = self.mask_path(source_index)
        if not path.exists():
            if self.written_range is not None and self.written_range[0] <= source_index < self.written_range[1]:
                # Inside the declared range a missing file is lost data (a failed write, a deleted file), not an
                # absent object, so it is not reported as "mask_file_absent" like any other frame.
                raise ValueError(f"{path.name} is missing although objects.json declares frames {self.written_range[0]}..{self.written_range[1] - 1} "
                                 "as written; the mask set is incomplete")
            return {name: ObjectObservation(name, None, reason="mask_file_absent", details=dict(source=self.name)) for name in self.labels.values()}
        label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if label is None or label.shape != depth.shape:
            raise ValueError(f"{path.name}: label image missing or not at the depth resolution")
        if label.dtype != np.uint8 or not set(map(int, np.unique(label))) <= {0, *self.labels}:
            raise ValueError(f"{path.name}: expected uint8 labels declared by objects.json (or background 0)")
        scene = self.observer.observe(depth, seed, invalid)
        observations = {}
        for value, name in self.labels.items():
            mask = label == value
            area = int(mask.sum())
            if area == 0:
                observations[name] = ObjectObservation(name, None, reason="mask_absent", details=dict(source=self.name, plane=_plane_summary(scene["plane"])))
                continue
            if scene["plane"] is None:
                observations[name] = ObjectObservation(name, None, reason="plane_unavailable", details=dict(source=self.name, area_px=area, plane=None))
                continue
            geo = region_geometry(scene["points"], scene["valid"], scene["plane"], mask, self.settings.min_points, self.settings.min_height_m)
            if geo is None:
                observations[name] = ObjectObservation(name, None, reason="mask_without_table_height",
                                                       details=dict(source=self.name, area_px=area, plane=_plane_summary(scene["plane"])))
                continue
            observations[name] = ObjectObservation(name, geo["centroid_xyz"], pixel=geo["pixel"],
                                                   details=dict(source=self.name, height_m=round(geo["height_m"], 4), footprint_extent_m=round(geo["footprint_extent_m"], 4),
                                                                n_points=geo["n_points"], area_px=area, plane=_plane_summary(scene["plane"])))
        return observations


def settings_record(settings: PlaneSettings) -> dict:
    return asdict(settings)
