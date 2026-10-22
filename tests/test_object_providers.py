"""Object providers and the chunked segmentation driver. Synthetic data, N=0."""
import json
from dataclasses import asdict

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from sg_offline.objects import MaskObjects, PlaneClusterObjects, PlaneSettings, TablePlaneObserver
from sg_offline.recordings import RECORDING_SCHEMA, RGBDRecording
from sg_offline.rgbd import CameraCalibration
import segment_objects
from test_tableplane import CAL, synthetic_scene

SETTINGS = PlaneSettings(min_area_px=50, max_range_m=5.)


def two_box_scene(shift_px=0):
    """Table + two boxes; the second box can be shifted to simulate motion between frames."""
    depth, box, arm, n, d = synthetic_scene(box_height=.10)
    pts_dummy = None
    second = np.zeros_like(box); second[30:55, 110 + shift_px:135 + shift_px] = True
    # a second box top: same construction as synthetic_scene's hit() for a 7 cm box
    t = np.deg2rad(35.); nn = np.array([0., -np.cos(t), -np.sin(t)])
    u = np.arange(CAL.width)[None, :]; v = np.arange(CAL.height)[:, None]
    dirs = np.dstack([(u - CAL.cx) / CAL.fx * np.ones_like(v, dtype=float), (v - CAL.cy) / CAL.fy * np.ones_like(u, dtype=float), np.ones((CAL.height, CAL.width))])
    denom = dirs @ nn
    z2 = np.where(denom < -1e-6, -(d - .07) / denom, np.nan)
    depth = depth.copy(); depth[second] = np.round(z2[second] * 1000).astype(np.uint16)
    return depth, box, second, arm


def test_plane_cluster_provider_tracks_two_objects_and_reports_their_geometry():
    provider = PlaneClusterObjects(CAL, SETTINGS, max_objects=2, gate_m=.15)
    depth, box, second, arm = two_box_scene()
    obs = provider.observe(depth, hand_boxes=(), seed=0)
    assert set(obs) == {"object_1", "object_2"} and all(o.xyz is not None for o in obs.values())
    heights = sorted(round(o.details["height_m"], 2) for o in obs.values())
    assert heights == pytest.approx([.07, .10], abs=.015)
    assert all(o.details["source"] == "plane_clusters" and o.details["plane"]["inlier_fraction"] > .5 for o in obs.values())
    # the arm cluster is excluded, never an object
    assert all("touches_bottom_border_arm" in o.details["excluded_clusters"] for o in obs.values())
    first = {k: o.xyz for k, o in obs.items()}
    # next frame: the second box moved slightly; identities persist by nearest centroid
    obs2 = provider.observe(two_box_scene(shift_px=3)[0], hand_boxes=(), seed=1)
    assert set(obs2) == set(obs)
    moved = [k for k in obs2 if np.linalg.norm(np.asarray(obs2[k].xyz) - first[k]) > 1e-3]
    assert len(moved) == 1 and all(np.linalg.norm(np.asarray(obs2[k].xyz) - first[k]) < .05 for k in obs2)


def test_plane_cluster_provider_withholds_occluded_objects_and_reacquires_after_a_move():
    provider = PlaneClusterObjects(CAL, SETTINGS, max_objects=2, gate_m=.05, reacquire_after=2)
    depth, box, second, arm = two_box_scene()
    provider.observe(depth, hand_boxes=(), seed=0)
    # a tracked hand box covering the first box: that object is occluded/merged -> unavailable with a reason, identity kept
    hand = (55, 45, 95, 85)
    obs = provider.observe(depth, hand_boxes=(hand,), seed=1)
    occluded = [k for k, o in obs.items() if o.xyz is None]
    assert len(occluded) == 1 and obs[occluded[0]].reason == "cluster_not_found_occluded_or_merged" and obs[occluded[0]].details["missing_frames"] == 1
    provider.observe(depth, hand_boxes=(hand,), seed=2)   # missing twice -> eligible for re-acquisition
    # the object reappears far from its old place (moved while occluded): beyond the gate, but re-acquired, not duplicated
    depth_moved, *_ = two_box_scene(shift_px=0)
    moved_box = np.zeros_like(box); moved_box[50:80, 10:40] = True
    depth_moved = depth_moved.copy(); depth_moved[box] = depth[box] * 0 + np.round(depth[box].mean()).astype(np.uint16) + 70  # flatten the old place toward the table
    depth_moved[moved_box] = depth[box].mean()  # approximately the same height at a new place
    obs3 = provider.observe(depth_moved.astype(np.uint16), hand_boxes=(), seed=3)
    assert set(obs3) == set(obs) and sum(o.xyz is not None for o in obs3.values()) >= 1
    assert any(o.details.get("reacquired") for o in obs3.values() if o.xyz is not None) or all(o.xyz is not None for o in obs3.values())


def test_plane_cluster_provider_reports_plane_failure_for_every_known_object():
    provider = PlaneClusterObjects(CAL, SETTINGS)
    provider.observe(two_box_scene()[0], seed=0)
    obs = provider.observe(np.zeros((CAL.height, CAL.width), np.uint16), seed=1)  # no valid depth at all
    assert obs and all(o.xyz is None and o.reason == "plane_unavailable" for o in obs.values())
    with pytest.raises(ValueError):
        PlaneClusterObjects(CAL, SETTINGS, max_objects=5)


def write_masks(tmp_path, label_images, names):
    masks = tmp_path / "seg" / "masks"; masks.mkdir(parents=True)
    for i, label in label_images.items():
        cv2.imwrite(str(masks / f"{i:05d}.png"), label)
    (tmp_path / "seg" / "objects.json").write_text(json.dumps(dict(schema="sg-object-masks/1", objects={str(k + 1): n for k, n in enumerate(names)},
                                                                   mask_files="masks/{source_index:05d}.png")))
    return tmp_path / "seg"


def test_mask_provider_localises_masks_above_the_table_and_names_every_absence(tmp_path):
    depth, box, second, arm = two_box_scene()
    label = np.zeros(depth.shape, np.uint8); label[box] = 1; label[second] = 2
    loose = label.copy(); loose[20:48, 0:30] = 3                     # a mask lying on bare table: no height
    seg = write_masks(tmp_path, {0: label, 1: loose}, ["cup", "jar", "ghost"])
    provider = MaskObjects(CAL, seg, SETTINGS)
    obs = provider.observe(depth, 0)
    assert obs["cup"].xyz is not None and round(obs["cup"].details["height_m"], 2) == pytest.approx(.10, abs=.015)
    assert obs["jar"].xyz is not None and round(obs["jar"].details["height_m"], 2) == pytest.approx(.07, abs=.015)
    assert obs["ghost"].xyz is None and obs["ghost"].reason == "mask_absent"
    obs1 = provider.observe(depth, 1)
    assert obs1["ghost"].xyz is None and obs1["ghost"].reason == "mask_without_table_height"
    obs9 = provider.observe(depth, 9)
    assert all(o.reason == "mask_file_absent" for o in obs9.values())
    assert obs["cup"].details["source"] == "sam2_masks_plus_plane" and obs["cup"].details["area_px"] == box.sum()
    with pytest.raises(ValueError, match="resolution"):
        provider.observe(depth[:-1], 0)


class FakeSegmenter:
    """Stands in for SAM 2: every frame's mask is the prompt point's 10x10 square shifted by the local frame index;
    an object whose prompt is missing stays empty; ``vanish`` makes an object's mask empty on given source frames.
    Records what it was prompted with."""

    def __init__(self, shape, vanish=None):
        self.shape = shape; self.calls = []; self.vanish = vanish or {}; self.offset = 0

    def segment_chunk(self, frames_dir, prompts, handover):
        n = len(sorted(frames_dir.glob("*.jpg")))
        self.calls.append(dict(frames=n, prompts=None if prompts is None else sorted(prompts), handover=None if handover is None else sorted(handover)))
        seeds = {}
        if prompts:
            for oid, p in prompts.items():
                seeds[oid] = tuple(int(v) for v in p["points"][0])
        if handover:
            for oid, m in handover.items():
                ys, xs = np.nonzero(m); seeds[oid] = (int(xs.mean()), int(ys.mean()))
        for local in range(n):
            out = {}
            for oid, (u, v) in seeds.items():
                m = np.zeros(self.shape, bool)
                if (self.offset + local) not in self.vanish.get(oid, ()):
                    u2 = min(u + local, self.shape[1] - 6); m[v:v + 10, u2:u2 + 10] = True
                out[oid] = m
            yield local, out
        self.offset += n - 1  # the next chunk starts on this chunk's last frame


def make_recording(tmp_path, n=7, width=64, height=48):
    cal = CameraCalibration("c", width, height, 60., 60., width / 2, height / 2, .001, "cam")
    frames = []
    for i in range(n):
        rgb = np.full((height, width, 3), 90, np.uint8); depth = np.full((height, width), 800, np.uint16)
        np.save(tmp_path / f"rgb{i}.npy", rgb); np.save(tmp_path / f"d{i}.npy", depth)
        stamp = dict(sample_id=f"s{i}", acquired_s=10. + i / 30, available_s=10.003 + i / 30, source="recorded", clock="c")
        frames.append(dict(frame_id=f"s/f{i}", rgb_path=f"rgb{i}.npy", depth_path=f"d{i}.npy", rgb_stamp=stamp, depth_stamp=dict(stamp, sample_id=f"d{i}"),
                           registered_to_rgb=True, max_sync_skew_s=.005))
    (tmp_path / "recording.json").write_text(json.dumps(dict(schema=RECORDING_SCHEMA, participant_id="p", session_id="s", camera_model="synthetic",
                                                                calibration=asdict(cal), frames=frames)))
    return RGBDRecording(tmp_path / "recording.json")


def test_chunk_ranges_overlap_one_frame_for_the_mask_handover():
    assert segment_objects.chunk_ranges(7, 0, 3) == [(0, 3, True), (2, 5, False), (4, 7, False)]
    assert segment_objects.chunk_ranges(5, 2, 10) == [(2, 5, True)]
    with pytest.raises(ValueError):
        segment_objects.chunk_ranges(5, 0, 1)


def test_chunked_segmentation_writes_every_frame_once_and_hands_masks_over(tmp_path):
    rec = make_recording(tmp_path, n=7)
    fake = FakeSegmenter((48, 64))
    prompts = dict(frame_index=0, objects={"cup": dict(points=[[10, 10]], labels=[1]), "jar": dict(points=[[30, 20]], labels=[1])})
    prov = segment_objects.run_segmentation(rec, tmp_path / "seg", prompts, fake, chunk=3)
    files = sorted((tmp_path / "seg" / "masks").glob("*.png"))
    assert [f.stem for f in files] == [f"{i:05d}" for i in range(7)] and prov["frames_written"] == 7
    assert fake.calls[0]["prompts"] == [1, 2] and fake.calls[0]["handover"] is None
    assert fake.calls[1]["prompts"] is None and fake.calls[1]["handover"] == [1, 2]     # later chunks are prompted with masks
    assert [c["frames"] for c in fake.calls] == [3, 3, 3]
    label6 = cv2.imread(str(files[6]), cv2.IMREAD_UNCHANGED)
    assert set(np.unique(label6)) == {0, 1, 2} and (label6 == 1).sum() == 100
    meta = json.loads((tmp_path / "seg" / "objects.json").read_text())
    assert meta["objects"] == {"1": "cup", "2": "jar"} and prov["objects"]["cup"]["frames_with_mask"] == 7
    assert not (tmp_path / "seg" / "_frames_tmp").exists()
    provider = MaskObjects(rec.calibration, tmp_path / "seg")
    assert set(provider.labels.values()) == {"cup", "jar"}
    assert prov["handover_lookback_frames"] == 15 and [h["boundary_source_index"] for h in prov["handovers"]] == [2, 4]
    with pytest.raises(FileExistsError):
        segment_objects.run_segmentation(rec, tmp_path / "seg", prompts, fake, chunk=3)


def test_handover_passes_recently_seen_objects_and_drops_long_lost_ones(tmp_path):
    rec = make_recording(tmp_path, n=9)
    prompts = dict(frame_index=0, objects={"cup": dict(points=[[10, 10]], labels=[1]), "jar": dict(points=[[30, 20]], labels=[1])})
    # the jar vanishes from source frame 3 onward; chunks of 5 -> boundary at frame 4: last seen at 2
    fake = FakeSegmenter((48, 64), vanish={2: set(range(3, 9))})
    prov = segment_objects.run_segmentation(rec, tmp_path / "a", prompts, fake, chunk=5, handover_lookback=1)
    assert prov["handovers"][0] == dict(boundary_source_index=4, objects=[1], lost=[2])     # 2 frames stale > lookback 1: lost
    assert fake.calls[1]["handover"] == [1] and prov["objects"]["jar"]["first_lost_source_index"] == 3
    fake2 = FakeSegmenter((48, 64), vanish={2: set(range(3, 9))})
    prov2 = segment_objects.run_segmentation(rec, tmp_path / "b", prompts, fake2, chunk=5, handover_lookback=2)
    assert prov2["handovers"][0]["objects"] == [1, 2]                                         # within the lookback: handed over
    with pytest.raises(ValueError):
        segment_objects.run_segmentation(rec, tmp_path / "c", prompts, FakeSegmenter((48, 64)), chunk=5, handover_lookback=-1)


class StrictFakeSegmenter(FakeSegmenter):
    """Like the real SAM 2 video predictor (checked on CPU with sam2.1_hiera_small): propagating a
    chunk that received neither a point prompt nor a hand-over mask raises instead of returning empty masks."""

    def segment_chunk(self, frames_dir, prompts, handover):
        if not prompts and not handover:
            raise RuntimeError("No input points or masks are provided for any object; please add inputs first.")
        yield from super().segment_chunk(frames_dir, prompts, handover)


def test_segmentation_completes_when_no_object_can_be_handed_to_the_next_chunk(tmp_path):
    """With every object out of view for longer than the lookback at
    a chunk boundary, the driver called the segmenter with an empty hand-over; the real predictor then raised and the
    run stopped with masks on disk but no objects.json or provenance. The objects stay lost (the documented limit),
    so the remaining frames are written as empty masks and the run is recorded."""
    rec = make_recording(tmp_path, n=12)
    prompts = dict(frame_index=0, objects={"cup": dict(points=[[10, 10]], labels=[1]), "jar": dict(points=[[30, 20]], labels=[1])})
    # both objects leave the view from source frame 2 on; chunks of 5 -> boundaries at frames 4 and 8; lookback 1
    fake = StrictFakeSegmenter((48, 64), vanish={1: set(range(2, 12)), 2: set(range(2, 12))})
    prov = segment_objects.run_segmentation(rec, tmp_path / "seg", prompts, fake, chunk=5, handover_lookback=1)
    files = sorted((tmp_path / "seg" / "masks").glob("*.png"))
    assert [f.stem for f in files] == [f"{i:05d}" for i in range(12)] and prov["frames_written"] == 12   # every frame once
    assert len(fake.calls) == 1                                                  # the segmenter is not asked to propagate nothing
    assert all(not cv2.imread(str(f), cv2.IMREAD_UNCHANGED).any() for f in files[2:])
    assert prov["chunks_written_empty_because_no_object_could_be_handed_over"] == 2
    assert [h["objects"] for h in prov["handovers"]] == [[], []] and prov["handovers"][0]["lost"] == [1, 2]
    assert prov["objects"]["cup"] == dict(frames=12, frames_with_mask=2, area_px_median=100, first_lost_source_index=2)
    assert not (tmp_path / "seg" / "_frames_tmp").exists()
    # the result is a complete mask set the provider can read: absent objects, with the reason
    provider = MaskObjects(rec.calibration, tmp_path / "seg")
    obs = provider.observe(np.full((48, 64), 800, np.uint16), 9)
    assert set(obs) == {"cup", "jar"} and all(o.xyz is None and o.reason == "mask_absent" for o in obs.values())
    # a partial loss is unchanged: the surviving object is still handed over and segmented
    fake2 = StrictFakeSegmenter((48, 64), vanish={2: set(range(2, 12))})
    prov2 = segment_objects.run_segmentation(rec, tmp_path / "seg2", prompts, fake2, chunk=5, handover_lookback=1)
    assert len(fake2.calls) == 3 and prov2["chunks_written_empty_because_no_object_could_be_handed_over"] == 0
    assert prov2["objects"]["cup"]["frames_with_mask"] == 12 and prov2["objects"]["jar"]["frames_with_mask"] == 2


def test_overwrite_removes_the_previous_mask_set_before_segmenting(tmp_path):
    """``overwrite=True`` used to skip only the not-empty refusal, so masks of an
    earlier run stayed on disk and were read through the NEW label map (frames before a later prompt frame showed
    objects the new run never segmented, possibly under swapped names). The driver now clears its own files first,
    as both recording adapters do; other files in the folder are left alone."""
    rec = make_recording(tmp_path, n=9)
    first = dict(frame_index=0, objects={"cup": dict(points=[[10, 10]], labels=[1]), "jar": dict(points=[[30, 20]], labels=[1])})
    segment_objects.run_segmentation(rec, tmp_path / "seg", first, FakeSegmenter((48, 64)), chunk=5)
    (tmp_path / "seg" / "notes.txt").write_text("not written by the driver")
    # re-segment in place from a later prompt frame, with the object order swapped
    second = dict(frame_index=4, objects={"jar": dict(points=[[30, 20]], labels=[1]), "cup": dict(points=[[10, 10]], labels=[1])})
    prov = segment_objects.run_segmentation(rec, tmp_path / "seg", second, FakeSegmenter((48, 64)), chunk=5, overwrite=True)
    files = sorted(f.stem for f in (tmp_path / "seg" / "masks").glob("*.png"))
    assert files == [f"{i:05d}" for i in range(4, 9)] and prov["frames_written"] == 5      # before: 9 files, 4 of them stale
    assert prov["replaced_previous_segmentation_files"] == 11                               # 9 masks + objects.json + provenance
    assert (tmp_path / "seg" / "notes.txt").exists()
    provider = MaskObjects(rec.calibration, tmp_path / "seg")
    assert provider.labels == {1: "jar", 2: "cup"}
    before_prompt = provider.observe(np.full((48, 64), 800, np.uint16), 2)
    assert all(o.reason == "mask_file_absent" for o in before_prompt.values())             # before: the first run's masks under the new names
    # without overwrite the refusal is unchanged
    with pytest.raises(FileExistsError):
        segment_objects.run_segmentation(rec, tmp_path / "seg", second, FakeSegmenter((48, 64)), chunk=5)


def test_the_mask_reader_refuses_a_mask_set_that_is_not_this_recordings(tmp_path):
    """`objects.json` records the schema, the session and the manifest the masks were made from, but the reader used
    only the label map: another session's masks (same resolution) were consumed as this session's objects, and an
    unknown schema string passed. Both are refused now; `build_object_provider` is given the recording."""
    from run_perception import build_object_provider
    rec = make_recording(tmp_path, n=6)
    prompts = dict(frame_index=0, objects={"cup": dict(points=[[10, 10]], labels=[1])})
    segment_objects.run_segmentation(rec, tmp_path / "seg", prompts, FakeSegmenter((48, 64)), chunk=5)
    provider, record = build_object_provider("masks", rec.calibration, masks_dir=tmp_path / "seg", recording=rec)
    assert record["kind"] == "masks" and record["mask_schema"] == "sg-object-masks/1" and record["masks_session_id"] == "s"
    meta_path = tmp_path / "seg" / "objects.json"; meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps(dict(meta, session_id="another-session")))
    with pytest.raises(ValueError, match="another recording"):
        build_object_provider("masks", rec.calibration, masks_dir=tmp_path / "seg", recording=rec)     # before: accepted
    meta_path.write_text(json.dumps(dict(meta, schema="sg-object-masks/999")))
    with pytest.raises(ValueError, match="schema"):
        MaskObjects(rec.calibration, tmp_path / "seg")                                                   # before: accepted
    meta_path.write_text(json.dumps(meta))
    prov_path = tmp_path / "seg" / "segmentation_provenance.json"; prov = json.loads(prov_path.read_text())
    prov_path.write_text(json.dumps(dict(prov, manifest_sha256="0" * 64)))
    with pytest.raises(ValueError, match="another recording"):
        build_object_provider("masks", rec.calibration, masks_dir=tmp_path / "seg", recording=rec)


def test_mask_names_must_be_distinct_object_names(tmp_path):
    """Two labels with one name silently merged into the later one, and a hand name was accepted until the first frame."""
    seg = write_masks(tmp_path, {}, ["cup", "cup"])
    with pytest.raises(ValueError, match="distinct"):
        MaskObjects(CAL, seg)
    (seg / "objects.json").write_text(json.dumps(dict(schema="sg-object-masks/1", objects={"1": "left"})))
    with pytest.raises(ValueError, match="hand"):
        MaskObjects(CAL, seg)


def test_a_mask_file_missing_inside_the_written_range_is_an_error_not_an_absent_object(tmp_path):
    """The reader reported any missing mask file as "object absent". Inside the range the mask set declares as written
    that is lost data (a failed write, a deleted file), so it raises; outside the range it stays `mask_file_absent`."""
    rec = make_recording(tmp_path, n=8)
    prompts = dict(frame_index=2, objects={"cup": dict(points=[[10, 10]], labels=[1])})
    segment_objects.run_segmentation(rec, tmp_path / "seg", prompts, FakeSegmenter((48, 64)), chunk=5)
    provider = MaskObjects(rec.calibration, tmp_path / "seg")
    depth = np.full((48, 64), 800, np.uint16)
    assert provider.observe(depth, 1)["cup"].reason == "mask_file_absent"            # before the prompt frame: by design
    (tmp_path / "seg" / "masks" / "00004.png").unlink()
    with pytest.raises(ValueError, match="missing"):
        provider.observe(depth, 4)                                                    # before: mask_file_absent


def test_segmentation_stops_when_a_frame_or_mask_cannot_be_written(tmp_path, monkeypatch):
    """`cv2.imwrite` returns False on a failed write (full disk, permissions) and the driver ignored it: the run ended
    normally, declared every frame written, and the provider then reported the objects absent in every frame."""
    rec = make_recording(tmp_path, n=6)
    prompts = dict(frame_index=0, objects={"cup": dict(points=[[10, 10]], labels=[1])})
    real = cv2.imwrite
    monkeypatch.setattr(cv2, "imwrite", lambda path, *a, **k: False if str(path).endswith("00003.png") else real(path, *a, **k))
    with pytest.raises(OSError, match="could not write"):
        segment_objects.run_segmentation(rec, tmp_path / "seg", prompts, FakeSegmenter((48, 64)), chunk=5)
    assert not (tmp_path / "seg" / "objects.json").exists()                           # an unfinished set is not readable as a mask set


def test_segmentation_provenance_times_and_areas_describe_what_was_written(tmp_path, monkeypatch):
    """`started_utc` was taken when the provenance was assembled, i.e. it was the END time (4 of 4 stored mask sets).
    And an object's area was counted from its raw mask although the label image, filled first-listed-object first,
    may not contain it."""
    import datetime as real_datetime
    events = []

    class Clock(real_datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            events.append("clock")
            return real_datetime.datetime(2026, 10, 5, 12, 0, len([e for e in events if e == "clock"]), tzinfo=tz)

    class Overlapping(FakeSegmenter):
        def segment_chunk(self, frames_dir, prompts, handover):
            events.append("segment")
            for local, masks in super().segment_chunk(frames_dir, prompts, handover):
                yield local, masks
    monkeypatch.setattr(segment_objects.datetime, "datetime", Clock)
    rec = make_recording(tmp_path, n=4)
    same_place = dict(frame_index=0, objects={"cup": dict(points=[[10, 10]], labels=[1]), "jar": dict(points=[[10, 10]], labels=[1])})
    prov = segment_objects.run_segmentation(rec, tmp_path / "seg", same_place, Overlapping((48, 64)), chunk=5)
    assert events.index("clock") < events.index("segment") and prov["started_utc"] < prov["finished_utc"]
    # the two masks coincide: the label image holds the first object only, and the record now says so
    assert prov["objects"]["cup"]["frames_with_mask"] == 4 and prov["objects"]["jar"]["frames_with_mask"] == 0


def test_the_plane_and_mask_providers_honour_the_frames_invalid_depth_marker(tmp_path):
    """A recording may declare a non-zero invalid-depth marker (rgbd_contract.md). The hand points and the colour
    provider honoured it; the plane and mask providers assumed 0 and took marked pixels for points 65 m away."""
    depth, box, second, arm = two_box_scene()
    holes = np.zeros(depth.shape, bool); holes[40:80, 10:50] = True                  # a block of missing depth on the table
    zero_marked = depth.copy(); zero_marked[holes] = 0
    marked = depth.copy(); marked[holes] = 65535
    label = np.zeros(depth.shape, np.uint8); label[box] = 1
    seg = write_masks(tmp_path, {0: label}, ["cup"])
    reference = MaskObjects(CAL, seg, SETTINGS).observe(zero_marked, 0)["cup"]
    honoured = MaskObjects(CAL, seg, SETTINGS).observe(marked, 0, invalid=65535)["cup"]
    assert honoured.xyz == reference.xyz and honoured.details["plane"] == reference.details["plane"]
    ignored = MaskObjects(CAL, seg, SETTINGS).observe(marked, 0)["cup"]              # the old call: marker not passed on
    assert ignored.details["plane"] != reference.details["plane"]
    a = PlaneClusterObjects(CAL, SETTINGS).observe(zero_marked, (), seed=0)
    b = PlaneClusterObjects(CAL, SETTINGS).observe(marked, (), seed=0, invalid=65535)
    assert {k: v.xyz for k, v in a.items()} == {k: v.xyz for k, v in b.items()}


def test_GUARD_the_mask_provider_reports_a_missing_table_plane_for_every_object(tmp_path):
    """This branch ran in no test; without it a frame in which the camera
    leaves the table would stop a perception run with an AttributeError."""
    depth, box, second, arm = two_box_scene()
    label = np.zeros(depth.shape, np.uint8); label[box] = 1
    seg = write_masks(tmp_path, {0: label}, ["cup", "jar"])
    obs = MaskObjects(CAL, seg, SETTINGS).observe(np.zeros(depth.shape, np.uint16), 0)
    assert obs["cup"].reason == "plane_unavailable" and obs["cup"].details["area_px"] == box.sum() and obs["jar"].reason == "mask_absent"
