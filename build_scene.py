"""Build the scene records (table plane + hand landmarks) for a STORED perception run, without a GPU.

A perception run before 16 October 2026 wrote packets, detections and diagnostics but no scene file. This tool refits the
table plane from the recording's raw depth for every processed frame (same seeded RANSAC and settings as the run's object
provider, seed = frame index) and lifts the three functional points per hand from the stored diagnostics (``point_xyz``),
writing ``scene.jsonl`` + ``scene_provenance.json`` into a NEW directory. Nothing in the stored run is touched. New runs of
``run_perception.py`` write the scene file themselves.
usage: build_scene.py RUN_DIR MANIFEST OUT_DIR [--plane-settings JSON]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from sg_offline.inputs import packets_from_jsonl
from sg_offline.objects import PlaneSettings, settings_record
from sg_offline.recordings import RGBDRecording
from sg_offline.scene import EXTRA_LANDMARK_NAMES, HandLandmarks, LANDMARK_NAMES, Scene, plane_from_fit, write_scenes
from sg_offline.tableplane import backproject, fit_table_plane


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def frame_index_of(packet_id):
    return int(packet_id.rsplit("-", 1)[1])


def build_scenes(run_dir, manifest, settings: PlaneSettings = PlaneSettings(), diagnostics=None):
    """Yield one Scene per stored packet. ``diagnostics``: packet_id -> diagnostics row (landmarks), optional."""
    recording = RGBDRecording(manifest)
    calibration = recording.calibration
    for packet in packets_from_jsonl(Path(run_dir) / "packets.jsonl"):
        if (packet.participant_id, packet.session_id) != (recording.participant_id, recording.session_id):
            raise ValueError(f"{packet.packet_id}: the packets and the recording name different participants or sessions")
        index = frame_index_of(packet.packet_id)
        frame = recording[index]
        stamp = packet.hands[0].stamp
        if stamp.acquired_s != frame.depth_stamp.acquired_s or not stamp.sample_id.startswith(frame.frame_id + ":"):
            raise ValueError(f"{packet.packet_id}: the packet was not made from frame {index} of this recording")
        points, valid = backproject(frame.depth, calibration, frame.invalid_depth_value)
        fit = fit_table_plane(points, valid, seed=index, iterations=settings.iterations, inlier_m=settings.inlier_m,
                              max_samples=settings.max_samples, min_inlier_fraction=settings.min_inlier_fraction)
        acquired = packet.hands[0].stamp.acquired_s
        plane = plane_from_fit(fit, calibration.optical_frame, calibration.calibration_id, acquired)
        landmarks = []
        diag = (diagnostics or {}).get(packet.packet_id)
        if diag is not None:
            for hand in packet.hands:
                xyz = diag["hands"][hand.identity].get("point_xyz") or {}; extra = diag["hands"][hand.identity].get("extra_xyz") or {}
                if any(xyz.get(n) is not None for n in LANDMARK_NAMES) or any(extra.get(n) is not None for n in EXTRA_LANDMARK_NAMES):
                    landmarks.append(HandLandmarks(hand.identity, {n: (None if xyz.get(n) is None else tuple(xyz[n])) for n in LANDMARK_NAMES},
                                                   calibration.optical_frame, calibration.calibration_id, acquired,
                                                   extra={n: (None if extra.get(n) is None else tuple(extra[n])) for n in EXTRA_LANDMARK_NAMES if n in extra}))
        yield Scene(packet.packet_id, plane, tuple(landmarks), source="build_scene.py: plane refit from stored depth (seed = frame index); landmarks from stored perception diagnostics")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path); ap.add_argument("manifest", type=Path); ap.add_argument("out_dir", type=Path)
    ap.add_argument("--plane-settings", type=Path)
    a = ap.parse_args(argv)
    if a.out_dir.exists() and any(a.out_dir.iterdir()):
        ap.error("output directory must be empty")
    run = json.loads((a.run_dir / "provenance.json").read_text()) if (a.run_dir / "provenance.json").exists() else {}
    if run.get("manifest_sha256") and run["manifest_sha256"] != sha256_file(a.manifest):
        ap.error("the manifest is not the one this run was made from (manifest_sha256 differs from the run's provenance.json)")
    if a.plane_settings:
        settings = PlaneSettings(**json.loads(a.plane_settings.read_text()))
    else:  # the run's own plane settings when it used the plane provider, else the defaults
        recorded = (run.get("settings", {}).get("object_provider") or {}).get("plane_settings")
        settings = PlaneSettings(**recorded) if recorded else PlaneSettings()
    diag_path = a.run_dir / "perception_diagnostics.jsonl"
    diagnostics = {}
    if diag_path.exists():
        for line in diag_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line); diagnostics[row["frame_id"]] = row
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    scenes = list(build_scenes(a.run_dir, a.manifest, settings, diagnostics))
    n = write_scenes(a.out_dir / "scene.jsonl", scenes)
    planes = sum(1 for s in scenes if s.plane is not None and s.plane.available)
    complete = {side: sum(1 for s in scenes if s.landmarks_for(side) is not None and s.landmarks_for(side).complete) for side in ("left", "right")}
    provenance = dict(script="build_scene.py", started_utc=started, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                      run_dir=str(a.run_dir.resolve()), packets_sha256=sha256_file(a.run_dir / "packets.jsonl"),
                      diagnostics_sha256=sha256_file(diag_path) if diag_path.exists() else None,
                      manifest=str(a.manifest.resolve()), manifest_sha256=sha256_file(a.manifest), plane_settings=settings_record(settings),
                      scenes=n, planes_available=planes, landmarks_complete=complete,
                      scope="scene records rebuilt offline from stored depth and diagnostics; no perception model ran; CPU")
    (a.out_dir / "scene_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(dict(scenes=n, planes_available=planes, landmarks_complete=complete)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
