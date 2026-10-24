"""End-to-end checks of the recording-to-scoring path. Synthetic, participant N=0.

One tiny synthetic `sg-rgbd-recording/1` clip (table + box, a fake right hand sliding toward and onto the box) goes
through `run_perception.main` (fake hand model, CPU, masks and plane providers, the three timing policies with a fake
processing clock), then through `label_frames` (Session.set_label / --frame-times), `evaluate_perception.main`,
`import_requests.main`, `ledger.to_opportunities(origin = session_origin_s)` and `evaluation.evaluate_opportunities` on
the trace read back from `trace.jsonl`. Before this file no test executed `run_perception.main`,
`evaluate_perception.main` or `import_requests.main`, and every test hand-built the files those commands write.
The first four tests pass on the earlier code as well: they pin the seams (cadence, origin, file formats). The `test_*`
tests below them fail on the earlier code.
Every run uses a registry path that does not exist and a stubbed Git query, so nothing in the repository is touched.
"""
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
try:                         # torch may be absent: main() only needs a device object and torch.load
    import torch
except ImportError:
    torch = None


class _Device:
    def __init__(self, kind): self.type = kind
    def __str__(self): return self.type


TORCH_STUB = SimpleNamespace(device=_Device, load=lambda *a, **k: None, __version__="stub (no torch in this environment)",
                             cuda=SimpleNamespace(is_available=lambda: False, synchronize=lambda: None))

import run_perception  # noqa: E402
import evaluate_perception  # noqa: E402
import import_requests  # noqa: E402
import label_frames  # noqa: E402
import zero_shot_hands  # noqa: E402
from sg_offline.evaluation import evaluate_opportunities, timeline_accounting  # noqa: E402
from sg_offline.ledger import read_opportunity_ledger, session_origin_s, to_opportunities  # noqa: E402
from sg_offline.recordings import RECORDING_SCHEMA, RGBDRecording  # noqa: E402
from sg_offline.stats import per_participant_success, unwanted_exposure  # noqa: E402

sys.path.insert(0, str(Path(run_perception.__file__).parent / "tests"))
from test_tableplane import CAL, synthetic_scene  # noqa: E402

N = 24                       # frames at 30 fps
T0 = 1790645000.0            # epoch-scale clock, as in the reference recordings
CLOCK = "d435_global_time_epoch_s"
HOST_DELAY = 0.003
FIRST_DELAY = 0.150          # the first frame of the real recording arrived 0.153 s after acquisition
STEP = 0.025                 # fake processing clock: every perf_counter() call advances by 25 ms -> 75 ms per frame


def acquired(i):
    """As in the real recording: the first frame is followed by a 0.2 s start-up gap, then 30 fps."""
    return T0 + (0. if i == 0 else .2 + (i - 1) / 30.)


DEPTH_EXTRA = 0.001          # the depth image arrives 1 ms after the colour image: frame availability = the later one


def available(i):
    """Availability of the FRAME (max of the two streams) = the session origin for i == 0."""
    return acquired(i) + (FIRST_DELAY if i == 0 else HOST_DELAY) + DEPTH_EXTRA


def hand_box(index):
    c = 6 + 3 * index        # slides right; its three points reach the box (columns 60..90) at frame 16
    return [float(c), 58., float(c + 16), 74.]


def hand_points(box):
    return {"thumb_mcp": (box[0] + 8., box[1] + 4.), "index_mcp": (box[0] + 8., box[1] + 8.), "trapezium": (box[0] + 8., box[1] + 12.)}


class FakeWilor:
    """Stands in for WilorRunner: one 'right' hand whose box is given by the frame index stored in pixel (0, 0)."""

    def __init__(self, repo, device):
        self.detector = None

    def detect(self, bgr, conf):
        return [hand_box(int(bgr[0, 0, 2]))], [1], [0.9]   # red channel of pixel (0, 0) carries the frame index

    def estimate(self, bgr, boxes, sides):
        out = []
        for box in boxes:
            pts = hand_points(box)
            joints = [list(pts["index_mcp"])] * 21
            joints[2] = list(pts["thumb_mcp"]); joints[5] = list(pts["index_mcp"]); joints[1] = list(pts["trapezium"])
            out.append(dict(right=1, joints2d=joints))
        return out


def make_recording(folder: Path):
    depth, box, arm, _, _ = synthetic_scene(box_height=.10)
    frames = []
    for i in range(N):
        rgb = np.full((CAL.height, CAL.width, 3), 90, np.uint8); rgb[0, 0, 0] = i
        np.save(folder / f"rgb_{i:04d}.npy", rgb); np.save(folder / f"depth_{i:04d}.npy", depth)
        t = acquired(i)
        frames.append(dict(frame_id=f"s/frame-{i:04d}", rgb_path=f"rgb_{i:04d}.npy", depth_path=f"depth_{i:04d}.npy",
                           rgb_stamp=dict(sample_id=f"color#{i}", acquired_s=t, available_s=available(i) - DEPTH_EXTRA, source="recorded", clock=CLOCK),
                           depth_stamp=dict(sample_id=f"depth#{i}", acquired_s=t + .001, available_s=available(i), source="recorded", clock=CLOCK),
                           registered_to_rgb=True, max_sync_skew_s=.005))
    manifest = folder / "recording.json"
    manifest.write_text(json.dumps(dict(schema=RECORDING_SCHEMA, participant_id="operator", session_id="s", camera_model="synthetic",
                                        calibration=asdict(CAL), frames=frames)))
    seg = folder / "seg"; (seg / "masks").mkdir(parents=True)
    label = np.zeros(depth.shape, np.uint8); label[box] = 1
    for i in range(N):
        cv2.imwrite(str(seg / "masks" / f"{i:05d}.png"), label)
    (seg / "objects.json").write_text(json.dumps(dict(schema="sg-object-masks/1", objects={"1": "cup"}, mask_files="masks/{source_index:05d}.png",
                                                      participant_id="operator", session_id="s", first_source_index=0, frames_written=N)))
    (seg / "segmentation_provenance.json").write_text(json.dumps({"manifest_sha256": run_perception.sha256_file(manifest)}))
    return manifest, seg


@pytest.fixture()
def clip(tmp_path):
    (tmp_path / "rec").mkdir()
    return make_recording(tmp_path / "rec")


def run(tmp_path, monkeypatch, name, *args, wilor=FakeWilor, manifest=None):
    """run_perception.main with the fake hand model, a deterministic processing clock, no Git query and a registry path
    that does not exist."""
    ticks = iter(range(10 ** 9))
    monkeypatch.setattr(zero_shot_hands, "WilorRunner", wilor)
    monkeypatch.setattr(run_perception, "time", SimpleNamespace(perf_counter=lambda: next(ticks) * STEP))
    monkeypatch.setattr(run_perception, "git_state", lambda *a, **k: dict(commit=None, note="not queried in tests"))
    if torch is None:
        monkeypatch.setitem(sys.modules, "torch", TORCH_STUB)
    out = tmp_path / name
    argv = [str(manifest or tmp_path / "rec" / "recording.json"), str(out), "--wilor", str(tmp_path / "no_wilor_repo"),
            "--manifest-csv", str(tmp_path / "absent_registry.csv"), *args]
    assert run_perception.main(argv) == 0
    return out


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


FIXED = ("--start", "0", "--max-frames", str(N), "--decimate", "2", "--queue-policy", "fixed_latency", "--fixed-latency-s", "0.1")


def test_c_main_path_with_masks_and_plane_providers(tmp_path, monkeypatch, clip, capsys):
    manifest, seg = clip
    origin, clock = session_origin_s(manifest)
    masks = run(tmp_path, monkeypatch, "run_masks", *FIXED, "--object-provider", "masks", "--masks", str(seg))
    summary = json.loads((masks / "summary.json").read_text())
    det = read_jsonl(masks / "detections.jsonl")
    # decimation and the deterministic frame set
    assert [d["frame_index"] for d in det] == list(range(0, N, 2)) and summary["frames"] == N // 2
    assert summary["frames_skipped_by_decimation"] == N // 2 and summary["frames_dropped_while_busy"] == 0 and summary["truncated_by_max_frames"] is False
    # fixed latency: every decision tick = that frame's availability + 0.1 s, on the recording's clock; frame ids join the tables
    times = {int(r["index"]): r for r in csv.DictReader(open(masks / "frame_times.csv"))}
    assert len(times) == N
    rows = read_jsonl(masks / "trace.jsonl")
    assert len(rows) == N // 2 and rows[0]["inputs"]["clock"] == clock
    for d, r in zip(det, rows):
        assert r["decision_s"] == pytest.approx(float(times[d["frame_index"]]["available_s"]) + .1, abs=1e-6)
        assert d["frame_id"] == times[d["frame_index"]]["frame_id"] == r["packet_id"]
    # the masks provider localises the cup in every processed frame; the fake hand is assigned as 'right' with a 3-D point
    assert summary["available_object_points"] == {"cup": N // 2} and summary["available_hand_points"] == {"left": 0, "right": N // 2}
    assert all(d["assigned"]["right"] is not None and d["assigned"]["left"] is None for d in det)
    assert summary["object_provider"]["kind"] == "masks" and json.loads((masks / "provenance.json").read_text())["settings"]["decimate"] == 2
    # plane provider on the same frames: one generic object at the same place (within 1 cm)
    (tmp_path / "plane_settings.json").write_text(json.dumps(dict(min_area_px=50, max_range_m=5.)))
    plane = run(tmp_path, monkeypatch, "run_plane", *FIXED, "--object-provider", "plane", "--plane-settings", str(tmp_path / "plane_settings.json"))
    dm = {d["frame_index"]: d for d in read_jsonl(masks / "perception_diagnostics.jsonl")}
    dp = {d["frame_index"]: d for d in read_jsonl(plane / "perception_diagnostics.jsonl")}
    assert set(dm) == set(dp)
    for k in dm:
        a, b = dm[k]["object_xyz"]["cup"], dp[k]["object_xyz"].get("object_1")
        assert a is not None and b is not None and np.linalg.norm(np.asarray(a) - b) < .01
    assert not (tmp_path / "absent_registry.csv").exists()


def test_c_start_decimate_truncation_and_the_two_queue_policies(tmp_path, monkeypatch, clip):
    manifest, seg = clip
    provider = ("--object-provider", "masks", "--masks", str(seg))
    # --start with --decimate: the decimation phase follows --start, and --max-frames truncation is flagged
    out = run(tmp_path, monkeypatch, "run_start1", "--start", "1", "--max-frames", "5", "--decimate", "2", "--queue-policy", "fixed_latency", *provider)
    s1 = json.loads((out / "summary.json").read_text())
    assert [d["frame_index"] for d in read_jsonl(out / "detections.jsonl")] == [1, 3, 5, 7, 9] and s1["truncated_by_max_frames"] is True
    avail = available
    # serial: 75 ms of work per 33 ms frame -> every frame is processed and queues behind the previous one
    serial = run(tmp_path, monkeypatch, "run_serial", "--max-frames", str(N), "--queue-policy", "serial", *provider)
    ds = read_jsonl(serial / "detections.jsonl")
    assert [d["frame_index"] for d in ds] == list(range(N))
    # (at epoch scale one double step is 2.4e-7 s, so 24 chained additions may drift by more than a microsecond)
    assert [d["available_s"] for d in ds] == pytest.approx([avail(0) + 3 * STEP * (k + 1) for k in range(N)], abs=5e-6)
    # drop_while_busy: frames that arrive while the worker is busy are skipped; a processed frame is stamped arrival + 75 ms
    drop = run(tmp_path, monkeypatch, "run_drop", "--max-frames", str(N), "--queue-policy", "drop_while_busy", *provider)
    dd = read_jsonl(drop / "detections.jsonl"); sd = json.loads((drop / "summary.json").read_text())
    assert [d["frame_index"] for d in dd] == [0, 2, 5, 8, 11, 14, 17, 20, 23] and sd["frames_dropped_while_busy"] == 15
    assert [d["available_s"] for d in dd] == pytest.approx([avail(d["frame_index"]) + 3 * STEP for d in dd], abs=1e-6)
    assert [r["decision_s"] for r in read_jsonl(drop / "trace.jsonl")] == pytest.approx([d["available_s"] for d in dd], abs=1e-9)


def test_a_labels_against_decimated_detections(tmp_path, monkeypatch, clip):
    manifest, seg = clip
    out = run(tmp_path, monkeypatch, "run", *FIXED, "--object-provider", "masks", "--masks", str(seg))
    # the operator labels every 3rd frame with the labelling tool's own writer (no GUI)
    session = label_frames.Session(RGBDRecording(manifest), tmp_path / "labels.jsonl", "ann", "unmarked", "bare-hands")
    labelled = label_frames.frame_indices(N, 3, None)
    for i in labelled:
        frame_id = session.recording.records[i]["frame_id"]
        box = hand_box(i)
        session.set_label(frame_id, "right", "hand_box", "visible", box=tuple(box))
        session.set_label(frame_id, "left", "hand_box", "out_of_frame")
        session.set_label(frame_id, "right", "thumb_mcp", "visible", *hand_points(box)["thumb_mcp"])
    evaluate_perception.main([str(out / "detections.jsonl"), str(tmp_path / "labels.jsonl"), str(tmp_path / "evaluation.json")])
    result = json.loads((tmp_path / "evaluation.json").read_text())
    even = [i for i in labelled if i % 2 == 0]
    assert result["counts"]["labelled_frames"] == len(even)                                       # only frames the run processed can be scored
    assert result["counts"]["reference_frames_without_predictions"] == len(labelled) - len(even)  # the odd labelled frames, made visible
    assert result["counts"]["unlabelled_prediction_frames"] == N // 2 - len(even)
    assert result["counts"]["matched_hands"] == len(even) and result["counts"]["handedness_correct"] == len(even)
    tracker = result["tracker"]["counts"]
    assert tracker["correct_identity"] == len(even) and tracker["correctly_empty_out_of_frame"] == len(even)
    assert result["tracker"]["point_error_px"]["thumb_mcp"] == dict(n=len(even), median=0.)


def test_b_and_d_requests_scored_on_a_replay_trace_with_the_session_origin(tmp_path, monkeypatch, clip):
    manifest, seg = clip
    out = run(tmp_path, monkeypatch, "run", *FIXED, "--object-provider", "masks", "--masks", str(seg))
    origin, clock = session_origin_s(manifest)
    # (d) the two frame-time tables agree with each other and with the ledger origin
    assert label_frames.main([str(manifest), str(tmp_path / "unused.jsonl"), "--annotator", "ann", "--pass", "unmarked", "--glove-condition", "bare",
                              "--frame-times", str(tmp_path / "times.csv")]) == 0
    assert (tmp_path / "times.csv").read_text() == (out / "frame_times.csv").read_text()
    times = list(csv.DictReader(open(out / "frame_times.csv")))
    assert float(times[0]["available_s"]) == origin and float(times[0]["session_relative_available_s"]) == 0.
    assert float(times[0]["session_relative_acquired_s"]) == pytest.approx(-(FIRST_DELAY + DEPTH_EXTRA), abs=1e-6)
    rows = read_jsonl(out / "trace.jsonl")
    proposals = [(r["decision_s"] - origin, r["packet_id"]) for r in rows if r["outputs"][1]["status"] == "proposed_simulation_only"]
    # the fake hand reaches the cup at frame 16 and keeps approaching until frame 20: three consecutive proposal ticks
    assert [p[1] for p in proposals] == ["s/frame-0016", "s/frame-0018", "s/frame-0020"]
    assert all(r["outputs"][0]["status"] == "withheld" for r in rows)       # the left hand is never observed
    first_rel = proposals[0][0]
    frame_of_first = next(t for t in times if t["frame_id"] == proposals[0][1])
    # a decision made from frame k is stamped at that frame's availability + the declared latency, in session-relative seconds
    assert first_rel == pytest.approx(float(frame_of_first["session_relative_available_s"]) + .1, abs=1e-6)
    assert first_rel == pytest.approx(available(16) - available(0) + .1, abs=1e-6)
    duration = round(float(times[-1]["session_relative_available_s"]) + .2, 2)
    onset = round(float(frame_of_first["session_relative_acquired_s"]) - .05, 3)      # the annotator's onset, in acquisition time
    split = round(onset + .2, 3)                                                       # the LEFT hand's state changes here: the request crosses rows
    with open(tmp_path / "timeline.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["start_s", "end_s", "left", "right", "note"])
        w.writerow(["0", onset, "NQ", "NQ", ""]); w.writerow([onset, split, "NQ", "G", ""])
        w.writerow([split, duration, "EXCLUDED", "G", "left out of view"])
    import_requests.main([str(tmp_path / "timeline.csv"), str(tmp_path / "requests.json"), "--participant", "operator", "--session", "s",
                          "--annotator", "ann", "--clock", clock, "--duration-s", str(duration)])
    ledger = read_opportunity_ledger(tmp_path / "requests.json")
    inside, outside = to_opportunities(ledger, origin)
    assert ledger["clock"]["name"] == rows[0]["inputs"]["clock"] and not outside
    outcomes = evaluate_opportunities(rows, inside)
    right = [o["hands"][1] for o in outcomes]
    assert [h["result"] for h in right] == ["correct_no_new_request", "compatible_first", "continuation_of_request"]
    assert right[1]["first_proposal"][0] - origin == pytest.approx(first_rel, abs=1e-6) and right[1]["early_by_s"] == 0.
    assert right[1]["proposal_count"] == 3 and right[1]["repeated_proposals"] == 2 and right[1]["proposal_runs"] == 1
    stats = per_participant_success(outcomes)["operator"]
    assert stats["evaluable_request_hands"] == 1 and stats["compatible_first"] == 1 and stats["request_continuation_rows"] == 1
    assert stats["unknown_reference_hands_by_reason"] == {"excluded": 1}
    exposure = unwanted_exposure(outcomes, inside)["operator"]
    assert exposure["exposure_hand_s"] == pytest.approx(onset * 2 + (split - onset), abs=1e-6)   # both hands at rest, then the left hand alone
    assert exposure["unwanted_proposals"] == 0 and exposure["proposals_per_hand_minute"] == 0.
    counts = timeline_accounting(rows, inside)
    assert counts["all_hand_ticks"] == 2 * len(rows) and counts.get("unlabelled_hand_ticks", 0) == 0
    # an annotated onset 0.3 s later (after the run began) turns the same proposals into an early first command
    with open(tmp_path / "timeline_late.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["start_s", "end_s", "left", "right", "note"])
        w.writerow(["0", round(first_rel + .1, 3), "NQ", "NQ", ""]); w.writerow([round(first_rel + .1, 3), duration, "NQ", "G", ""])
    import_requests.main([str(tmp_path / "timeline_late.csv"), str(tmp_path / "requests_late.json"), "--participant", "operator", "--session", "s",
                          "--annotator", "ann", "--clock", clock, "--duration-s", str(duration)])
    late_ops, _ = to_opportunities(read_opportunity_ledger(tmp_path / "requests_late.json"), origin)
    late_right = [o["hands"][1] for o in evaluate_opportunities(rows, late_ops)]
    assert late_right[0]["result"] == "unwanted" and late_right[1]["result"] == "wrong_timing_first"
    assert late_right[1]["early_by_s"] == pytest.approx(.1, abs=1e-3) and late_right[1]["run_active_at_window_start"] is True


# ---- regressions in the perception run -------------------------------------------------------------------------------

def test_the_run_confidence_is_the_trackers_confidence_floor(tmp_path, monkeypatch, clip):
    """`--confidence` reached only the detector; the identity tracker kept its own floor of 0.3, so a hand detected at 0.2
    in a run made with `--confidence 0.1` was logged as a raw box and never assigned."""
    class WeakHand(FakeWilor):
        def detect(self, bgr, conf):
            return [hand_box(int(bgr[0, 0, 2]))], [1], [0.2]
    manifest, seg = clip
    out = run(tmp_path, monkeypatch, "run_weak", *FIXED, "--confidence", "0.1", "--object-provider", "masks", "--masks", str(seg), wilor=WeakHand)
    det = read_jsonl(out / "detections.jsonl")
    assert all(len(d["detections"]) == 1 for d in det)
    assert sum(d["assigned"]["right"] is not None for d in det) == N // 2                     # before: 0 of 12
    assert json.loads((out / "provenance.json").read_text())["settings"]["tracker"] == dict(min_confidence=.1, max_gap_s=.25)


def test_truncation_is_flagged_only_when_a_selected_frame_was_left_out(tmp_path, monkeypatch, clip):
    """With `--decimate 2 --max-frames 12` on 24 frames every selected frame (0, 2, ..., 22) is processed; frame 23 is
    skipped by decimation anyway, yet `truncated_by_max_frames` was true because a frame followed the last processed one."""
    manifest, seg = clip
    out = run(tmp_path, monkeypatch, "run_exact", "--start", "0", "--max-frames", str(N // 2), "--decimate", "2", "--queue-policy", "fixed_latency",
              "--object-provider", "masks", "--masks", str(seg))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["frames"] == N // 2 and summary["truncated_by_max_frames"] is False        # before: True
    assert summary["frames"] + summary["frames_skipped_by_decimation"] == N                   # every frame accounted for
    cut = run(tmp_path, monkeypatch, "run_cut", "--start", "0", "--max-frames", "5", "--decimate", "2", "--queue-policy", "fixed_latency",
              "--object-provider", "masks", "--masks", str(seg))
    assert json.loads((cut / "summary.json").read_text())["truncated_by_max_frames"] is True


def test_an_option_the_provider_does_not_read_is_refused(tmp_path, monkeypatch, clip, capsys):
    """`--masks DIR` with the default colour provider was accepted and ignored: a run "with masks" used colour targets."""
    manifest, seg = clip
    for extra in (("--masks", str(seg)), ("--object-provider", "plane", "--objects", str(tmp_path / "targets.json")),
                  ("--object-provider", "colour", "--plane-settings", str(tmp_path / "plane.json"))):
        with pytest.raises(SystemExit):
            run(tmp_path, monkeypatch, "run_refused", *FIXED, *extra)
        assert not (tmp_path / "run_refused").exists()
    capsys.readouterr()


def test_a_mask_set_of_another_recording_stops_the_run_before_any_frame(tmp_path, monkeypatch, clip):
    manifest, seg = clip
    meta = json.loads((seg / "objects.json").read_text())
    (seg / "objects.json").write_text(json.dumps(dict(meta, session_id="another-session")))
    with pytest.raises(ValueError, match="another recording"):
        run(tmp_path, monkeypatch, "run_wrong_masks", *FIXED, "--object-provider", "masks", "--masks", str(seg))     # before: completed
    assert not (tmp_path / "run_wrong_masks").exists()


def test_perception_outputs_survive_a_replay_error_and_the_summary_carries_the_extraction_record(tmp_path, monkeypatch, clip):
    """The recording contract admits two frames with one availability stamp; `fixed_latency` then gives two packets one
    decision time, which the replay refuses. The run wrote nothing until after the replay, so the whole perception
    pass was lost. Also: a recording cut by an extraction limit was indistinguishable downstream."""
    manifest, seg = clip
    data = json.loads(manifest.read_text())
    for key in ("rgb_stamp", "depth_stamp"):                                   # a host stall: frames 2, 3 and 4 arrive in the same instant
        data["frames"][2][key]["available_s"] = data["frames"][4][key]["available_s"]
        data["frames"][3][key]["available_s"] = data["frames"][4][key]["available_s"]
    (tmp_path / "rec" / "tied.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="strict ordered unique decision packets"):
        run(tmp_path, monkeypatch, "run_tied", *FIXED, "--object-provider", "plane", manifest=tmp_path / "rec" / "tied.json")
    saved = tmp_path / "run_tied"
    assert len(read_jsonl(saved / "packets.jsonl")) == N // 2 and len(read_jsonl(saved / "detections.jsonl")) == N // 2   # before: no file
    assert (saved / "perception_diagnostics.jsonl").exists() and (saved / "frame_times.csv").exists()
    assert not (saved / "trace.jsonl").exists() and not (saved / "provenance.json").exists()     # an unfinished run stays recognisable
    # the adapter's record of a bounded extraction reaches the run summary
    (tmp_path / "rec" / "adapter_provenance.json").write_text(json.dumps(dict(partial_extraction=True, stop_reason="max_output_gib")))
    out = run(tmp_path, monkeypatch, "run_partial", *FIXED, "--object-provider", "masks", "--masks", str(seg))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["recording_partial_extraction"] is True and summary["recording_extraction_stop_reason"] == "max_output_gib"
    record = summary["object_provider"]
    assert record["mask_schema"] == "sg-object-masks/1" and record["masks_session_id"] == "s" and "segmentation_seconds_per_frame_median" in record
