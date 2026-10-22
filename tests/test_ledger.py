"""Ledger loaders and the four hand-checks of the ledger join."""
import copy
import json

import numpy as np
import pytest

from sg_offline.contracts import ACTIONS
from sg_offline.evaluation import evaluate_opportunities
from sg_offline.fixtures import scenarios
from sg_offline.ledger import (LandmarkLabel, landmark_coverage, ledger_summary, read_landmark_ledger,
                               read_opportunity_ledger, session_origin_s, to_opportunities, write_landmark_ledger)
from sg_offline.replay import Replay
import label_frames


def _ledger(entries, clock_kind="session_relative"):
    return {"schema": "sg-opportunity-ledger/1", "participant_id": "synthetic-person",
            "session_id": "synthetic-session", "clock": {"kind": clock_kind, "name": "session"},
            "opportunities": entries}


def _entry(oid, kind, start, end, left, right, **extra):
    base = dict(opportunity_id=oid, kind=kind, start_s=start, end_s=end,
                acceptable={"left": left, "right": right}, provenance="test", annotator="A1", prefix_only=True)
    base.update(extra)
    return base


def test_landmark_ledger_roundtrip_and_vocabulary(tmp_path):
    labels = [LandmarkLabel("s/f-0", "p", "s", "right", "thumb_mcp", "visible", 10., 20., "A1", "unmarked"),
              LandmarkLabel("s/f-0", "p", "s", "right", "index_mcp", "occluded", None, None, "A1", "unmarked")]
    path = tmp_path / "l.jsonl"
    write_landmark_ledger(path, labels)
    back = read_landmark_ledger(path)
    assert set(l.key() for l in back) == set(l.key() for l in labels)
    with pytest.raises(ValueError):
        LandmarkLabel("s/f-0", "p", "s", "right", "thumb_mcp", "occluded", 1., 2., "A1", "unmarked")
    with pytest.raises(ValueError):
        LandmarkLabel("s/f-0", "p", "s", "right", "thumb_mcp", "visible", None, None, "A1", "unmarked")
    with pytest.raises(ValueError):
        LandmarkLabel("s/f-0", "p", "s", "middle", "thumb_mcp", "visible", 1., 2., "A1", "unmarked")
    cov = landmark_coverage(back, image_size=(640, 480))
    assert cov["by_side_point_state"]["right"]["thumb_mcp"]["visible"] == 1
    assert cov["visible_pixels_outside_image"] == 0


def test_landmark_ledger_last_row_wins_and_rejects_bad_rows(tmp_path):
    path = tmp_path / "l.jsonl"
    a = LandmarkLabel("s/f-1", "p", "s", "left", "trapezium", "visible", 5., 5., "A1", "marker_on").to_row()
    b = dict(a, u=7., v=9.)
    path.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n", encoding="utf-8")
    (only,) = read_landmark_ledger(path)
    assert (only.u, only.v) == (7., 9.)
    path.write_text(json.dumps(dict(a, schema="other")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_landmark_ledger(path)


def test_render_labels_and_frame_indices():
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    labels = [LandmarkLabel("f", "p", "s", "right", "thumb_mcp", "visible", 10., 20., "A1", "unmarked"),
              LandmarkLabel("f", "p", "s", "left", "index_mcp", "occluded", None, None, "A1", "unmarked")]
    out = label_frames.render_labels(rgb, labels, "f", "hdr")
    assert out.shape == rgb.shape and out.any()
    assert label_frames.frame_indices(10, 4, None) == [0, 4, 8]
    assert label_frames.frame_indices(10, 1, [3, 3, 1]) == [1, 3]
    with pytest.raises(ValueError):
        label_frames.frame_indices(10, 1, [10])


def test_session_origin_from_manifest(tmp_path):
    manifest = {"schema": "sg-rgbd-recording/1", "participant_id": "p", "session_id": "s", "camera_model": "x",
                "calibration": {}, "frames": [{"rgb_stamp": {"available_s": 100.5, "clock": "c"},
                                               "depth_stamp": {"available_s": 100.4, "clock": "c"}}]}
    path = tmp_path / "recording.json"
    path.write_text(json.dumps(manifest))
    assert session_origin_s(path) == (100.5, "c")


def test_opportunity_ledger_validation(tmp_path):
    ok = _ledger([_entry("op1", "request", 0.5, 1.3, ["AR"], ["NQ"], entry_state={"left": "hand_open", "right": "unknown"},
                         onset_uncertainty_s=0.1),
                  _entry("nq1", "no_request_interval", 2., 3., ["NQ"], ["NQ"]),
                  _entry("oos1", "out_of_scope_request", 3., 4., ["HP"], None)])
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ok))
    ledger = read_opportunity_ledger(path)
    assert ledger_summary(ledger)["counts"] == {"request": 1, "no_request_interval": 1, "out_of_scope_request": 1}
    assert ledger_summary(ledger)["no_request_exposure_s"] == 1.
    for bad in (
        [_entry("x", "no_request_interval", 0, 1, ["G"], ["NQ"])],            # NQ interval referencing a request
        [_entry("x", "request", 0, 1, ["G", "NQ"], ["NQ"])],                    # mixed NQ and request
        [_entry("x", "request", 1, 0, ["G"], ["NQ"])],                          # reversed interval
        [_entry("x", "request", 0, 1, ["G"], ["NQ"], prefix_only="yes")],       # non-boolean prefix flag
        [_entry("x", "request", 0, 1, ["G"], ["NQ"]), _entry("x", "request", 2, 3, ["G"], ["NQ"])],  # duplicate id
        [_entry("x", "out_of_scope_request", 0, 1, ["NQ"], ["NQ"])],            # out-of-scope without an action
        [_entry("x", "request", 0, 1, ["G"], ["NQ"], entry_state={"left": "bent", "right": "unknown"})],
    ):
        path.write_text(json.dumps(_ledger(bad)))
        with pytest.raises(ValueError):
            read_opportunity_ledger(path)
    broken = dict(ok); broken["schema"] = "other"
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError):
        read_opportunity_ledger(path)


def test_scope_split_never_relabels_out_of_scope_as_no_request(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(_ledger([
        _entry("g", "request", 0.5, 1.3, ["G"], ["NQ"]),
        _entry("ar", "request", 2., 3., ["AR"], ["NQ"]),
        _entry("nq", "no_request_interval", 3., 4., ["NQ"], ["NQ"])])))
    ledger = read_opportunity_ledger(path)
    in_scope, out = to_opportunities(ledger, origin_s=0., admitted_actions=("G",))
    assert [o.opportunity_id for o in in_scope] == ["g", "ar", "nq"]
    assert in_scope[1].acceptable == (None, ("NQ",))
    assert [o.opportunity_id for o in out] == ["ar"]
    assert out[0].acceptable == (("AR",), ("NQ",))
    assert "prefix_only=True" in in_scope[0].provenance and "kind=request" in in_scope[0].provenance
    all_in, none_out = to_opportunities(ledger, origin_s=0., admitted_actions=ACTIONS)
    assert len(all_in) == 3 and not none_out
    with pytest.raises(ValueError):
        to_opportunities(ledger, origin_s=None)


def test_clock_shift_lands_on_trace_seconds(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(_ledger([_entry("g", "request", 0.5, 1.3, ["G"], ["NQ"],
                                               hand_windows_s={"left": [0.6, 1.2], "right": [0.5, 1.3]})])))
    (op,), _ = to_opportunities(read_opportunity_ledger(path), origin_s=1000.)
    assert (op.start_s, op.end_s) == (1000.5, 1001.3)
    assert op.hand_windows_s == ((1000.6, 1001.2), (1000.5, 1001.3))
    path.write_text(json.dumps(_ledger([_entry("g", "request", 5., 6., ["G"], ["NQ"])], clock_kind="trace")))
    (op,), _ = to_opportunities(read_opportunity_ledger(path), origin_s=None)
    assert (op.start_s, op.end_s) == (5., 6.)


# ---- the four ledger-join hand-checks (primary endpoints and denominator contract)

def _rows():
    return Replay().run(scenarios())


def test_handcheck_all_empty_output_is_missed_not_success():
    (op,), _ = to_opportunities(read_opportunity_ledger_from(_ledger([_entry("g", "request", 0.5, 1.3, ["G"], ["NQ"])])), 0.)
    result = evaluate_opportunities([], [op])[0]
    assert result["hands"][0]["result"] == "missed_including_withheld"
    assert result["hands"][1]["result"] == "correct_no_new_request"


def test_handcheck_missing_opportunity_row_is_not_evaluated():
    rows = _rows()
    (op,), _ = to_opportunities(read_opportunity_ledger_from(_ledger([_entry("g", "request", 0.5, 1.3, ["G"], ["NQ"])])), 0.)
    assert len(evaluate_opportunities(rows, [op])) == 1
    assert evaluate_opportunities(rows, []) == []


def test_handcheck_wrong_first_correct_later_stays_wrong():
    rows = _rows()
    first = next(r for r in rows if r["outputs"][0]["status"] == "proposed_simulation_only")
    (op,), _ = to_opportunities(read_opportunity_ledger_from(_ledger([_entry("g", "request", 0.5, 1.3, ["AR"], ["NQ"])])), 0., admitted_actions=ACTIONS)
    result = evaluate_opportunities(rows, [op])[0]
    assert result["hands"][0]["result"] == "wrong_first"
    assert result["hands"][0]["first_proposal"][1] == first["outputs"][0]["raw_outcome"]
    later = copy.deepcopy(rows)
    for r in later:
        if r["decision_s"] > first["decision_s"] and r["outputs"][0]["status"] == "proposed_simulation_only":
            r["outputs"][0]["raw_outcome"] = "AR"
    assert evaluate_opportunities(later, [op])[0]["hands"][0]["result"] == "wrong_first"


def test_handcheck_missing_reference_is_unknown_not_no_request():
    rows = _rows()
    (op,), _ = to_opportunities(read_opportunity_ledger_from(_ledger([_entry("u", "request", 0.5, 1.3, None, ["NQ"])])), 0.)
    result = evaluate_opportunities(rows, [op])[0]
    assert result["hands"][0]["result"] == "unknown_reference"
    assert result["fully_referenceable"] is False and result["joint_compatible"] is None


def read_opportunity_ledger_from(data):
    import tempfile, os
    fd, name = tempfile.mkstemp(suffix=".json"); os.close(fd)
    with open(name, "w", encoding="utf-8") as f:
        json.dump(data, f)
    try:
        return read_opportunity_ledger(name)
    finally:
        os.unlink(name)
