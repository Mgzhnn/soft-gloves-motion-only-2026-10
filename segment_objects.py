"""Segment the objects of one recording with SAM 2 (video predictor), offline and causal, in chunks.

Proposed (5 October 2026). Input: an `sg-rgbd-recording/1` manifest and a prompts file; output: one
uint8 label image per source frame (0 = background, k = object k) under ``masks/``, ``objects.json`` naming
the labels, and ``segmentation_provenance.json`` (checkpoint sha256, sam2 commit, config, prompts, chunking,
timing, per-object mask-area summary, frames where an object was lost).

Causality: prompts are given on one frame and propagated FORWARD only; a frame's mask depends on earlier
frames and the prompt, never on later ones (checked on 5 October 2026: a 60-frame run and a 30-frame prefix give
identical masks on the shared frames). Memory: frames are processed in chunks; each chunk starts on the last
frame of the previous one and is prompted with that frame's predicted masks (mask hand-over), so a 2,000-frame
clip never needs more than one chunk of frames on the GPU.

prompts.json: {"frame_index": 0, "objects": {"cream_cylinder": {"points": [[u, v]], "labels": [1]}, ...}}
Point coordinates are pixels of the source frame; label 1 = foreground, 0 = background.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from sg_offline.objects import MASK_SCHEMA
from sg_offline.recordings import RGBDRecording

DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def chunk_ranges(n_frames: int, start: int, chunk: int):
    """[(first, last_exclusive, is_first)] covering start..n_frames-1; later chunks begin on the previous last frame."""
    if chunk < 2 or not 0 <= start < n_frames:
        raise ValueError("chunk must be at least 2 frames and start inside the recording")
    out, first, is_first = [], start, True
    while first < n_frames:
        last = min(first + chunk, n_frames)
        out.append((first, last, is_first))
        if last >= n_frames:
            break
        first, is_first = last - 1, False  # overlap one frame for the mask hand-over
    return out


def _write_image(path, image, params=None) -> None:
    """``cv2.imwrite`` reports a failed write (full disk, permissions) only by returning False; before 5 October 2026
    that was ignored and an incomplete mask set was declared complete."""
    import cv2
    ok = cv2.imwrite(str(path), image, params) if params else cv2.imwrite(str(path), image)
    if not ok:
        raise OSError(f"could not write {path}")


def _clear_previous_segmentation(out_dir: Path) -> int:
    """Remove what an earlier run of this driver wrote (mask images, objects.json, the provenance, temporary frames);
    any other file in the folder is left alone. Returns the number of files removed."""
    removed = 0
    masks = out_dir / "masks"
    for path in (sorted(masks.glob("*.png")) if masks.is_dir() else []):
        path.unlink(); removed += 1
    for name in ("objects.json", "segmentation_provenance.json"):
        if (out_dir / name).is_file():
            (out_dir / name).unlink(); removed += 1
    shutil.rmtree(out_dir / "_frames_tmp", ignore_errors=True)
    return removed


class Sam2Segmenter:
    """Thin wrapper over the SAM 2 video predictor; tests inject a fake with the same two methods."""

    def __init__(self, checkpoint: Path, config: str = DEFAULT_CONFIG, device="cuda", offload=True):
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        self.torch = torch
        self.predictor = build_sam2_video_predictor(config, str(checkpoint), device=device)
        self.device = device
        self.offload = offload  # keep the chunk's frames and memory bank in CPU RAM so chunks can be long

    def segment_chunk(self, frames_dir: Path, prompts: dict | None, handover: dict | None):
        """Yield (local_frame_index, {obj_id: bool mask}) for every frame of the JPEG directory, in order."""
        torch = self.torch
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            state = self.predictor.init_state(video_path=str(frames_dir), offload_video_to_cpu=self.offload, offload_state_to_cpu=self.offload)
            if prompts is not None:
                for obj_id, p in prompts.items():
                    self.predictor.add_new_points_or_box(state, frame_idx=0, obj_id=obj_id,
                                                         points=np.asarray(p["points"], dtype=np.float32), labels=np.asarray(p["labels"], dtype=np.int32))
            if handover is not None:
                for obj_id, mask in handover.items():
                    self.predictor.add_new_mask(state, frame_idx=0, obj_id=obj_id, mask=torch.from_numpy(mask))
            for frame_idx, obj_ids, logits in self.predictor.propagate_in_video(state):
                yield frame_idx, {int(o): (logits[j, 0] > 0).cpu().numpy() for j, o in enumerate(obj_ids)}
            self.predictor.reset_state(state)


def run_segmentation(recording: RGBDRecording, out_dir: Path, prompts: dict, segmenter, *, chunk=1000, jpeg_quality=95,
                     overwrite=False, handover_lookback=15, provenance_extra=None) -> dict:
    """Segment every frame from the prompt frame on, in chunks that start on the previous chunk's last frame.

    Hand-over: an object is passed to the next chunk as a mask prompt on that shared frame if it was seen there, or
    within the last ``handover_lookback`` frames (0.5 s at 30 fps: it is then still near where it was). An object
    unseen for longer stays lost until the end of the recording: the segmenter re-acquires objects that leave and
    re-enter the view only within a chunk, so long chunks (default 1000 frames, with frames offloaded to CPU RAM)
    keep that limitation rare. Both the lookback and the chunk length are recorded in the provenance.
    """
    import cv2
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_dir} is not empty; pass --overwrite to replace it")
    if handover_lookback < 0:
        raise ValueError("handover_lookback must be nonnegative")
    # Overwrite means replace: masks of an earlier run must not survive under the new label map (frames before a
    # later prompt frame would otherwise show objects this run never segmented; found 5 October 2026).
    replaced = _clear_previous_segmentation(out_dir) if overwrite and out_dir.exists() else 0
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")  # before any work, not when the record is assembled
    masks_dir = out_dir / "masks"; masks_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_frames_tmp"
    names = list(prompts["objects"])
    if not names or len(names) > 254:
        raise ValueError("between 1 and 254 objects are supported")
    ids = {name: k + 1 for k, name in enumerate(names)}  # label value k = object k
    start = int(prompts.get("frame_index", 0))
    n = len(recording)
    ranges = chunk_ranges(n, start, chunk)
    handover = None
    areas = {name: [] for name in names}; lost_at = {name: None for name in names}; written = 0; t0 = time.perf_counter(); per_frame_s = []
    last_seen = {ids[name]: None for name in names}  # obj id -> (source index, mask) of the last non-empty mask
    handed_over = []; chunks_without_objects = 0
    for first, last, is_first in ranges:
        if not is_first and not handover:
            # Nothing can be handed over: every object has been unseen for longer than the lookback. The SAM 2 video
            # predictor refuses to propagate without a prompt ("No input points or masks are provided for any object"),
            # which used to abort the run here and leave masks without objects.json (found 5 October 2026). The objects
            # stay lost, as the limitation below states, so the remaining frames get empty label images and the run
            # completes with its records. The hand-over frame itself was written by the previous chunk.
            for source in range(first + 1, last):
                for name in names:
                    areas[name].append(0)
                _write_image(masks_dir / f"{source:05d}.png", np.zeros((recording.calibration.height, recording.calibration.width), np.uint8)); written += 1
            chunks_without_objects += 1
            handed_over.append(dict(boundary_source_index=last - 1, objects=[], lost=sorted(last_seen)))
            continue
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        for local, source in enumerate(range(first, last)):
            rgb = recording[source].rgb
            _write_image(tmp / f"{local:05d}.jpg", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        point_prompts = {ids[name]: prompts["objects"][name] for name in names} if is_first else None
        tc = time.perf_counter(); returned = 0
        for local, masks in segmenter.segment_chunk(tmp, point_prompts, None if is_first else handover):
            if not isinstance(local, (int, np.integer)) or local != returned or local >= last - first:
                raise RuntimeError(f"segmenter frame index {local!r}; expected consecutive local index {returned}")
            source = first + local; returned += 1
            if not is_first and local == 0:
                continue  # the hand-over frame was already written by the previous chunk
            label = np.zeros((recording.calibration.height, recording.calibration.width), np.uint8)
            for name in names:
                m = masks.get(ids[name])
                if m is not None and m.shape == label.shape and m.any():
                    label[m & (label == 0)] = ids[name]
            # Areas, "seen" and the hand-over mask come from the label image as written: masks listed later cannot
            # overwrite earlier ones, so an overlapped object's raw mask said more than the file holds (5 Oct 2026).
            for name in names:
                visible = label == ids[name]
                area = int(visible.sum())
                areas[name].append(area)
                if area:
                    last_seen[ids[name]] = (source, visible)
                elif lost_at[name] is None:
                    lost_at[name] = source
            _write_image(masks_dir / f"{source:05d}.png", label); written += 1
        if returned != last - first:
            raise RuntimeError(f"the segmenter returned {returned} frames for a chunk of {last - first} (source frames {first}..{last - 1})")
        per_frame_s.append((time.perf_counter() - tc) / max(last - first, 1))
        boundary = last - 1
        handover = {k: v[1] for k, v in last_seen.items() if v is not None and boundary - v[0] <= handover_lookback}
        handed_over.append(dict(boundary_source_index=boundary, objects=sorted(handover), lost=[k for k, v in last_seen.items() if k not in handover]))
    shutil.rmtree(tmp, ignore_errors=True)
    meta = dict(schema=MASK_SCHEMA, objects={str(v): k for k, v in ids.items()}, mask_files="masks/{source_index:05d}.png",
                recording=str(recording.path), participant_id=recording.participant_id, session_id=recording.session_id,
                first_source_index=start, frames_written=written)
    (out_dir / "objects.json").write_text(json.dumps(meta, indent=1) + "\n")
    summary = {name: dict(frames=len(a), frames_with_mask=int(sum(1 for v in a if v > 0)), area_px_median=int(np.median([v for v in a if v > 0])) if any(a) else None,
                          first_lost_source_index=lost_at[name]) for name, a in areas.items()}
    provenance = dict(script="segment_objects.py", started_utc=started_utc,
                      finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                      manifest=str(recording.path), manifest_sha256=sha256_file(recording.path), prompts=prompts, chunk=chunk, chunk_ranges=ranges,
                      handover_lookback_frames=handover_lookback, handovers=handed_over[:-1],
                      chunks_written_empty_because_no_object_could_be_handed_over=chunks_without_objects,
                      replaced_previous_segmentation_files=replaced,
                      jpeg_quality=jpeg_quality, frames_total=n, frames_written=written, seconds_total=round(time.perf_counter() - t0, 1),
                      seconds_per_frame_median=round(float(np.median(per_frame_s)), 4) if per_frame_s else None, objects=summary,
                      causality="prompts on one frame, forward propagation only; chunks start on the previous chunk's last frame, prompted with masks seen there or within the lookback",
                      limitation="an object unseen for longer than the lookback at a chunk boundary stays lost for the rest of the recording; within a chunk the segmenter may re-acquire it",
                      scope="object masks from a zero-shot segmenter; not a measurement of object identity or position accuracy", **(provenance_extra or {}))
    (out_dir / "segmentation_provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    return provenance


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", type=Path); ap.add_argument("out_dir", type=Path)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True); ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--chunk", type=int, default=1000); ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--handover-lookback", type=int, default=15, help="frames: hand an object over if seen within this many frames of the boundary")
    ap.add_argument("--no-offload", action="store_true", help="keep frames on the GPU (needs much more GPU memory)")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if not a.checkpoint.is_file():
        print(f"missing checkpoint: {a.checkpoint}", file=sys.stderr); return 2
    prompts = json.loads(a.prompts.read_text())
    recording = RGBDRecording(a.manifest)
    segmenter = Sam2Segmenter(a.checkpoint, a.config, a.device, offload=not a.no_offload)
    extra = dict(checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=sha256_file(a.checkpoint), config=a.config, device=a.device, offload=not a.no_offload,
                 versions=dict(python=platform.python_version(), torch=importlib.metadata.version("torch"),
                               sam2=next((f"{d.version} {json.loads(d.read_text('direct_url.json')).get('vcs_info', {}).get('commit_id', '')}"
                                          for d in importlib.metadata.distributions() if d.metadata['Name'].lower() in ('sam-2', 'sam2') and d.read_text('direct_url.json')), "unknown")))
    provenance = run_segmentation(recording, a.out_dir, prompts, segmenter, chunk=a.chunk, jpeg_quality=a.jpeg_quality, overwrite=a.overwrite,
                                  handover_lookback=a.handover_lookback, provenance_extra=extra)
    print(json.dumps({k: provenance[k] for k in ("frames_total", "frames_written", "seconds_total", "seconds_per_frame_median", "objects")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
