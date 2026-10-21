"""Bounded RGB-D -> hand/object observations -> packets -> simulation trace (no labels).

Object providers, one per run, recorded in the summary and provenance:
  colour  the 23 September colour-region stand-in (``--objects targets.json``; default green cup / red marker);
  plane   RANSAC table plane + clusters above it (the matched simple comparator; identities object_1, object_2 ...);
  masks   SAM 2 masks precomputed by ``segment_objects.py`` (``--masks DIR``) + table-plane geometry.
All arms of a comparison must use the same provider; the choice is a batch setting.

Timing policy (2 October 2026). ``serial`` (the 23 Sep behaviour) simulates one worker whose frames queue
while it is busy: at 30 fps with ~40 ms per frame the queue grows without bound and the supervisor withholds
almost every tick as stale. ``drop_while_busy`` simulates one worker that skips frames arriving while it is busy
(latest-frame processing). ``fixed_latency`` processes every selected frame and stamps availability at the frame
time plus a declared constant (``--fixed-latency-s``): the processed-frame set is then deterministic, which a
comparison of providers or models on identical frames needs. ``--decimate N`` processes every Nth frame from
``--start``. Choose one policy per batch and record it; never per clip to improve a result.

Every run writes ``provenance.json`` (command, settings, checkpoint and code hashes, versions, Git state) and
``frame_times.csv`` (frame index -> session-relative seconds for annotators) next to its outputs, and appends
one row to ``the run folder`` when that registry exists.
"""
import argparse
import csv
from dataclasses import asdict
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import numpy as np
from sg_offline.ledger import FUNCTIONAL_POINTS, frame_time_rows
from sg_offline.objects import MaskObjects, PlaneClusterObjects, PlaneSettings, settings_record
from sg_offline.contracts import SIDES
from sg_offline.perception import AnatomicalTracker, ColourObjects, HandDetection, packet_from_detections
from sg_offline.recordings import RGBDRecording
from sg_offline.replay import Config, Replay, write_trace
from sg_offline.scene import EXTRA_LANDMARK_NAMES, HandLandmarks, LANDMARK_NAMES, Scene, plane_from_fit, write_scenes
EXTRA_FROM_MANO = {"thumb_tip": 4, "index_tip": 8}  # MANO joint order (0 wrist; thumb 1-4; index 5-8; tip = last of each chain)
from sg_offline.objects import TablePlaneObserver
from sg_offline.evaluation import summarize

QUEUE_POLICIES = ("serial", "drop_while_busy", "fixed_latency")
OBJECT_PROVIDERS = ("colour", "plane", "masks")
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST_CSV = Path(__file__).resolve().parents[2] / "manifest.csv"


def schedule(frame_available_s, previous_done_s, elapsed_s, policy="serial", fixed_latency_s=0.1):
    """Availability time of this frame's perception output, or None when the policy skips the frame."""
    if policy not in QUEUE_POLICIES:
        raise ValueError("unknown queue policy")
    if policy == "fixed_latency":
        if fixed_latency_s < 0:
            raise ValueError("fixed latency must be nonnegative")
        return frame_available_s + fixed_latency_s
    if policy == "drop_while_busy":
        if frame_available_s < previous_done_s:
            return None
        return frame_available_s + elapsed_s
    return max(frame_available_s, previous_done_s) + elapsed_s


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def implementation_hashes(root=None):
    """sha256 of every implementation source file, so a run names the exact code that produced it."""
    root = Path(root or Path(__file__).resolve().parent)
    files = sorted(list(root.glob("*.py")) + list((root / "sg_offline").glob("*.py")))
    return {str(p.relative_to(root)): sha256_file(p) for p in files}


def write_frame_times(path, records):
    rows = frame_time_rows(records)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["index"])
        writer.writeheader(); writer.writerows(rows)
    return len(rows)


def git_state(repo_root=REPO_ROOT):
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True).stdout.strip()
        return dict(commit=head, dirty_tracked_or_untracked=bool(dirty), note="the implementation itself is Git-ignored; see implementation_sha256")
    except Exception as exc:  # repository state is context, not a requirement
        return dict(commit=None, error=f"{type(exc).__name__}: {exc}")


def append_manifest_row(path, row):
    """Append one run to a CSV registry whose header defines the columns; False when the registry is absent."""
    path = Path(path)
    if not path.exists():
        return False
    with open(path, newline="") as f:
        header = next(csv.reader(f), [])
    required = {"run_id", "output_path", "status"} & row.keys()
    if not required <= set(header):
        raise ValueError("registry header is missing supplied run identity/status columns")
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=header, extrasaction="ignore").writerow({k: row.get(k, "") for k in header})
    return True


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_object_provider(kind, calibration, *, objects_file=None, masks_dir=None, plane_settings=None, recording=None):
    """One provider per run. Returns (provider, record) where record goes to the summary and provenance.

    With ``recording`` (the RGBDRecording being processed) a mask set is accepted only if it was segmented from
    that recording: same participant and session in ``objects.json`` and, when the segmentation provenance is
    present, the same manifest hash. Before 5 October 2026 another session's masks were consumed without a word."""
    if kind not in OBJECT_PROVIDERS:
        raise ValueError("unknown object provider")
    settings = PlaneSettings(**json.loads(Path(plane_settings).read_text())) if plane_settings else PlaneSettings()
    if kind == "colour":
        provider = ColourObjects.from_json(objects_file) if objects_file else ColourObjects()
        return provider, dict(kind="colour", objects_file=str(objects_file) if objects_file else None, min_area=provider.min_area,
                              targets={k: [list(map(list, r)) for r in v] for k, v in provider.targets.items()},
                              caveat="each object is the single colour region of its declared range, not a recognised object; clips without the configured colours yield no objects")
    if kind == "plane":
        provider = PlaneClusterObjects(calibration, settings)
        return provider, dict(kind="plane", plane_settings=settings_record(settings), max_objects=provider.max_objects, gate_m=provider.gate_m,
                              reacquire_after_frames=provider.reacquire_after,
                              caveat="objects are clusters above the RANSAC table plane with arms, hands and far field removed; a held object merges with the hand and is unavailable; "
                                     "identities are generic and may change object after a gap")
    if masks_dir is None:
        raise ValueError("--masks DIR is required for the masks provider")
    provider = MaskObjects(calibration, masks_dir, settings)
    seg = Path(masks_dir) / "segmentation_provenance.json"
    segmentation = json.loads(seg.read_text()) if seg.exists() else {}
    if recording is not None:
        missing = [k for k in ("participant_id", "session_id") if not provider.provenance.get(k)]
        if not segmentation.get("manifest_sha256"):
            missing.append("manifest_sha256")
        if provider.written_range is None or provider.written_range[0] < 0 or provider.written_range[1] <= provider.written_range[0]:
            missing.append("valid written frame range")
        if missing:
            raise ValueError(f"the mask set {masks_dir} lacks required recording provenance: {', '.join(missing)}")
        differs = [k for k in ("participant_id", "session_id") if provider.provenance.get(k) is not None and provider.provenance[k] != getattr(recording, k)]
        if segmentation.get("manifest_sha256") is not None and segmentation["manifest_sha256"] != sha256_file(recording.path):
            differs.append("manifest_sha256")
        if differs:
            raise ValueError(f"the mask set {masks_dir} was made from another recording ({', '.join(differs)} differ)")
    return provider, dict(kind="masks", masks_dir=str(Path(masks_dir).resolve()), objects=list(provider.labels.values()), plane_settings=settings_record(settings),
                          mask_schema=provider.provenance.get("schema"), masks_session_id=provider.provenance.get("session_id"),
                          segmentation_provenance_sha256=sha256_file(seg) if seg.exists() else None,
                          segmentation_seconds_per_frame_median=segmentation.get("seconds_per_frame_median"),
                          caveat="objects are precomputed SAM 2 masks localised above the RANSAC table plane; a mask is a segment, not a verified object identity; "
                                 "the segmenter's own time per frame is recorded here and is NOT part of the per-frame processing time of this run")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", type=Path); ap.add_argument("output", type=Path)
    provider = ap.add_mutually_exclusive_group(required=True)
    provider.add_argument("--wilor", type=Path)
    provider.add_argument("--sparse-pose", type=Path)
    ap.add_argument("--detector", type=Path, help="optional adapted WiLoR detector checkpoint")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--confidence", type=float, default=.3)
    ap.add_argument("--decimate", type=int, default=1, help="process every Nth frame from --start (1 = every frame)")
    ap.add_argument("--queue-policy", choices=QUEUE_POLICIES, default="serial")
    ap.add_argument("--fixed-latency-s", type=float, default=.1, help="declared constant latency for the fixed_latency policy")
    ap.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV, help="run registry to append to (skipped when absent)")
    ap.add_argument("--object-provider", choices=OBJECT_PROVIDERS, default="colour")
    ap.add_argument("--objects", type=Path, help="colour provider: JSON of colour targets {identity: [[[H,S,V],[H,S,V]], ...]}")
    ap.add_argument("--masks", type=Path, help="masks provider: output directory of segment_objects.py")
    ap.add_argument("--plane-settings", type=Path, help="plane/masks providers: JSON with PlaneSettings fields (defaults otherwise)")
    ap.add_argument("--rate-window", type=int, default=1, help="replay Config.rate_window: 1 = one-frame secant (default); K > 1 = causal least-squares slope over up to K past pairs (batch setting, 7 Oct 2026)")
    a = ap.parse_args(argv)
    if a.start < 0 or a.max_frames < 1 or not 0 < a.confidence < 1 or a.decimate < 1 or a.fixed_latency_s < 0 or a.rate_window < 1:
        ap.error("invalid bounds")
    if a.output.exists() and any(a.output.iterdir()):
        ap.error("output directory must be empty")
    # An option the chosen provider does not read used to be accepted and ignored (a run "with masks" on the colour default).
    for option, value, users in (("--masks", a.masks, ("masks",)), ("--objects", a.objects, ("colour",)), ("--plane-settings", a.plane_settings, ("plane", "masks"))):
        if value is not None and a.object_provider not in users:
            ap.error(f"{option} is not used by --object-provider {a.object_provider}")
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    command = " ".join(sys.argv if argv is None else ["run_perception.py", *map(str, argv)])
    import torch, cv2
    from sg_offline.local_models import load_yolo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = {}
    if a.wilor:
        from zero_shot_hands import WilorRunner, FUNCTIONAL_FROM_MANO
        # Local pre-existing legacy checkpoint; scoped loader compatibility, restored afterwards.
        from functools import partial
        original_load = torch.load
        torch.load = partial(original_load, weights_only=False)
        try:
            runner = WilorRunner(a.wilor.resolve(), device)
        finally:
            torch.load = original_load
        for name in ("pretrained_models/detector.pt", "pretrained_models/wilor_final.ckpt", "mano_data/MANO_RIGHT.pkl"):
            p = a.wilor.resolve() / name
            checkpoints[str(p)] = sha256_file(p) if p.exists() else "missing"
        if a.detector:
            if not a.detector.is_file():
                ap.error("detector must be a local checkpoint")
            runner.detector = load_yolo(a.detector).to(device)
            checkpoints[str(a.detector.resolve())] = sha256_file(a.detector)
    else:
        if not a.sparse_pose.is_file():
            ap.error("sparse pose requires an existing local checkpoint")
        runner = load_yolo(a.sparse_pose).to(device)
        if tuple(getattr(runner.model.model[-1], "kpt_shape", ())) != (3, 3):
            raise ValueError("sparse comparator checkpoint must predict exactly three functional points with visibility")
        checkpoints[str(a.sparse_pose.resolve())] = sha256_file(a.sparse_pose)
    recording = RGBDRecording(a.manifest)
    if a.start >= len(recording):
        ap.error("start frame is outside the recording")
    # The tracker's confidence floor is the run's --confidence (it used to stay at its own 0.3 whatever was asked).
    identity = AnatomicalTracker(min_confidence=a.confidence)
    objects, object_record = build_object_provider(a.object_provider, recording.calibration, objects_file=a.objects, masks_dir=a.masks,
                                                   plane_settings=a.plane_settings, recording=recording)
    # What the adapter recorded about a bounded extraction travels with the run: a recording cut by a frame or byte
    # limit is not the whole session, and nothing downstream could tell.
    sidecar = a.manifest.resolve().parent / "adapter_provenance.json"
    extraction = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    packets = []; detections = []; diagnostics = []; scenes = []; previous_done = -1.
    # The scene record (16 Oct 2026): the table plane of every processed frame (the object providers' own fit for plane/masks,
    # a separate fit with the same settings for colour) and the lifted functional points per hand, written beside the packets.
    plane_observer = TablePlaneObserver(recording.calibration, PlaneSettings(**json.loads(a.plane_settings.read_text())) if a.plane_settings else PlaneSettings())
    decimated = dropped_busy = 0; truncated = False; object_seconds = []
    a.output.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(recording):
        if index < a.start:
            continue
        if (index - a.start) % a.decimate:
            decimated += 1; continue
        if len(packets) >= a.max_frames:
            # Only a frame that WOULD have been selected is a sign of truncation; before 5 October 2026 any remaining
            # frame set the flag, also one that decimation skips anyway.
            truncated = True; break
        if a.queue_policy == "drop_while_busy" and frame.available_s < previous_done:
            dropped_busy += 1; continue  # the worker was still busy when this frame arrived
        if device.type == "cuda": torch.cuda.synchronize()
        start = time.perf_counter(); bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        found = []
        if a.wilor:
            boxes, sides, scores = runner.detect(bgr, a.confidence)
            estimates = runner.estimate(bgr, boxes, sides) if boxes else []
            for box, side, score, est in zip(boxes, sides, scores, estimates):
                found.append(HandDetection(tuple(box), "right" if side else "left", score,
                    {p: tuple(est["joints2d"][j]) for p,j in FUNCTIONAL_FROM_MANO.items()}, "wilor",
                    "MANO_MCP_and_thumb_CMC_proxy_not_validated_trapezium",
                    extra_points={p: tuple(est["joints2d"][j]) for p, j in EXTRA_FROM_MANO.items()}))
        else:
            result = runner(bgr, conf=a.confidence, verbose=False)[0]
            # Ultralytics 8.1 can return keypoints shape (1,0,9) when boxes are empty.
            # No detection is a normal missing observation, not a malformed pose checkpoint.
            if len(result.boxes) and (result.keypoints is None or result.keypoints.data.shape[1] != 3):
                raise ValueError("sparse model returned malformed keypoints")
            keypoints = result.keypoints.data.cpu().numpy() if len(result.boxes) else []
            for box, kp in zip(result.boxes.data.cpu().numpy(), keypoints):
                found.append(HandDetection(tuple(float(v) for v in box[:4]), "right" if int(box[5]) else "left", float(box[4]),
                    {p: tuple(float(v) for v in uv[:2]) if len(uv) < 3 or uv[2] >= a.confidence else None
                     for p,uv in zip(FUNCTIONAL_POINTS, kp)}, "yolo_three_point", "labelled_three_functional_points"))
        assigned = identity.assign(found, frame.rgb_stamp.acquired_s)
        hand_boxes = [tuple(d.box) for d in assigned.values() if d is not None]
        t_obj = time.perf_counter()
        if a.object_provider == "colour":
            observed = objects.locate(frame.rgb, frame.depth, frame.invalid_depth_value)  # (pixel, reason) per object
        elif a.object_provider == "plane":
            observed = objects.observe(frame.depth, hand_boxes, seed=index, invalid=frame.invalid_depth_value)
        else:
            observed = objects.observe(frame.depth, index, seed=index, invalid=frame.invalid_depth_value)
        object_seconds.append(time.perf_counter() - t_obj)
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        # Offline timing simulation under the chosen queue policy. This is not measured online latency.
        available = schedule(frame.available_s, previous_done, elapsed, a.queue_policy, a.fixed_latency_s)
        details = {}
        packet, diag = packet_from_detections(frame, assigned, observed, participant_id=recording.participant_id,
                      session_id=recording.session_id, available_s=available, object_details=details)
        previous_done = available; packets.append(packet)
        fit = plane_observer.observe(frame.depth, index, frame.invalid_depth_value)["plane"]
        landmarks = []
        for side in SIDES:
            xyz = diag[side].get("point_xyz") or {}; extra = diag[side].get("extra_xyz") or {}
            if any(xyz.get(n) is not None for n in LANDMARK_NAMES) or any(extra.get(n) is not None for n in EXTRA_LANDMARK_NAMES):
                landmarks.append(HandLandmarks(side, {n: (None if xyz.get(n) is None else tuple(xyz[n])) for n in LANDMARK_NAMES},
                                               recording.calibration.optical_frame, recording.calibration.calibration_id, frame.depth_stamp.acquired_s,
                                               extra={n: (None if extra.get(n) is None else tuple(extra[n])) for n in EXTRA_LANDMARK_NAMES if n in extra}))
        scenes.append(Scene(packet.packet_id, plane_from_fit(fit, recording.calibration.optical_frame, recording.calibration.calibration_id, frame.depth_stamp.acquired_s),
                            tuple(landmarks), source="run_perception.py: seeded RANSAC table plane (seed = frame index) and lifted functional points"))
        detections.append(dict(participant_id=recording.participant_id, session_id=recording.session_id,
            frame_id=frame.frame_id, frame_index=index, detections=[asdict(d) for d in found],
            # What the runtime actually used per anatomical side (None = withheld/missing), so the
            # evaluator can score identity handling, not only the raw detector.
            assigned={side: (asdict(d) if d is not None else None) for side, d in assigned.items()},
            processing_s=elapsed, acquired_s=frame.rgb_stamp.acquired_s, available_s=available))
        diagnostics.append(dict(frame_id=frame.frame_id, frame_index=index, hands=diag, objects={p.identity:p.reason for p in packet.objects},
                                object_details=details, object_xyz={p.identity: (list(p.xyz) if p.xyz is not None else None) for p in packet.objects}))
    # Perception outputs are saved BEFORE the replay: a replay error (for example two selected frames with one
    # availability stamp under fixed_latency, which the recording contract admits) used to lose the whole pass.
    for name, values in (("packets", [asdict(p) for p in packets]), ("detections", detections), ("perception_diagnostics", diagnostics)):
        (a.output/(name+".jsonl")).write_text("".join(json.dumps(v)+"\n" for v in values))
    frame_rows = write_frame_times(a.output/"frame_times.csv", recording.records)
    scene_rows = write_scenes(a.output/"scene.jsonl", scenes)
    replay = Replay(Config(rate_window=a.rate_window))
    rows = replay.run(packets)
    write_trace(a.output/"trace.jsonl", rows)
    manifest_sha = hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    object_identities = sorted({p.identity for pk in packets for p in pk.objects})
    availability_note = {"serial": "measured processing durations plus serial queue simulation; not online timing validation",
                         "drop_while_busy": "measured processing durations; frames arriving while busy were skipped (latest-frame policy); not online timing validation",
                         "fixed_latency": f"declared constant latency of {a.fixed_latency_s} s after frame availability; deterministic frame set; not online timing validation"}[a.queue_policy]
    summary = dict(frames=len(packets), frames_in_recording=len(recording), start=a.start, max_frames=a.max_frames,
        truncated_by_max_frames=truncated, decimate=a.decimate, frames_skipped_by_decimation=decimated,
        queue_policy=a.queue_policy, fixed_latency_s=a.fixed_latency_s if a.queue_policy == "fixed_latency" else None, frames_dropped_while_busy=dropped_busy,
        manifest=str(a.manifest.resolve()), manifest_sha256=manifest_sha,
        recording_partial_extraction=extraction.get("partial_extraction"), recording_extraction_stop_reason=extraction.get("stop_reason"),
        available_hand_points={s:sum(p.hands[i].xyz is not None for p in packets) for i,s in enumerate(("left","right"))},
        available_object_points={o: sum(1 for pk in packets for p in pk.objects if p.identity == o and p.xyz is not None) for o in object_identities},
        object_provider=object_record, object_processing_s_median=float(np.median(object_seconds)) if object_seconds else None,
        availability=availability_note,
        functional_point_caveat="WiLoR thumb CMC is only a trapezium proxy; predicted visibility is unverified",
        object_caveat=object_record["caveat"],
        inference_scope="offline grasp proposals only under feature layout v1; the scene file (scene.jsonl) carries the table plane and functional points for the per-action replay (layout v2)",
        scene_records=scene_rows, scene_planes_available=sum(1 for s in scenes if s.plane is not None and s.plane.available),
        evaluation=summarize(rows))
    (a.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    finished_utc = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    versions = dict(python=platform.python_version(), torch=getattr(torch, "__version__", None), ultralytics=_version("ultralytics"),
                    numpy=np.__version__, opencv=getattr(cv2, "__version__", None))
    hardware = torch.cuda.get_device_name(0) if device.type == "cuda" else f"cpu ({platform.processor() or platform.machine()})"
    provenance = dict(started_utc=started_utc, finished_utc=finished_utc, host=socket.gethostname(), command=command,
        settings=dict(start=a.start, max_frames=a.max_frames, confidence=a.confidence, decimate=a.decimate, queue_policy=a.queue_policy,
                      fixed_latency_s=a.fixed_latency_s if a.queue_policy == "fixed_latency" else None,
                      provider="wilor" if a.wilor else "sparse_pose", detector=str(a.detector) if a.detector else None,
                      tracker=dict(min_confidence=identity.min_confidence, max_gap_s=identity.max_gap_s),
                      object_provider=object_record, replay_configuration=asdict(replay.config)),
        device=str(device), hardware=hardware, versions=versions, checkpoints_sha256=checkpoints,
        implementation_sha256=implementation_hashes(), manifest=str(a.manifest.resolve()), manifest_sha256=manifest_sha,
        participant_id=recording.participant_id, session_id=recording.session_id, frame_times_rows=frame_rows, git=git_state(),
        scope="run provenance for audit; it does not validate the recording, the labels or the perception accuracy")
    (a.output/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    registered = append_manifest_row(a.manifest_csv, dict(
        run_id=f"{a.output.resolve().name}@{started_utc}", date_utc=started_utc, git_commit=provenance["git"].get("commit") or "",
        dirty_worktree=provenance["git"].get("dirty_tracked_or_untracked", ""), command=command, config_path=str(a.manifest.resolve()),
        config_sha256=manifest_sha, seed="", dataset_id=f"{recording.participant_id}/{recording.session_id}", dataset_version=manifest_sha[:12],
        split_manifest_sha256="", environment_snapshot=json.dumps(versions, sort_keys=True), hardware=hardware,
        output_path=str(a.output.resolve()), status="completed",
        notes=f"frames={len(packets)} truncated={truncated} policy={a.queue_policy} decimate={a.decimate} dropped_busy={dropped_busy} objects={a.object_provider} rate_window={a.rate_window}"))
    summary["provenance"] = str((a.output/"provenance.json").resolve()); summary["registered_in_manifest_csv"] = registered
    print(json.dumps(summary, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
