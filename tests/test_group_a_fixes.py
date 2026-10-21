"""Group A fixes (2026-09-23). Each test reproduces a reviewed failure; all fixtures are synthetic (N=0).

25 of the 29 items here failed on the pre-fix snapshot (the run folder
2026-09-23_before_group_a) and pass on the fixed code; four pass there by design (one checks unchanged behaviour,
three guard behaviour that was already correct), as project notes records (correction 5 October 2026: this line
said "every test"). Tests needing torch or pyrealsense2 skip without them.
"""
from dataclasses import asdict, replace
import json
import numpy as np
import pytest
from sg_offline.contracts import Source
from sg_offline.evaluation import Opportunity, evaluate_opportunities
from sg_offline.fixtures import scenarios
from sg_offline.ledger import (FUNCTIONAL_POINTS, LandmarkLabel, ledger_summary, read_opportunity_ledger,
                               to_opportunities, write_landmark_ledger)
from sg_offline.stats import per_participant_success


def trace(times, proposals=(), withheld=(), withheld_nq=()):
    """Minimal replay rows: right hand proposes G at `proposals`, a G is withheld at `withheld`,
    a low-confidence NQ is withheld at `withheld_nq`, else NQ."""
    rows = []
    for k, t in enumerate(times):
        at = lambda xs: any(abs(t-x) < 1e-9 for x in xs)
        right = (dict(side="right", raw_outcome="G", status="proposed_simulation_only", reason="software_rule_only") if at(proposals) else
                 dict(side="right", raw_outcome="G", status="withheld", reason="insufficient_uncalibrated_score") if at(withheld) else
                 dict(side="right", raw_outcome="NQ", status="withheld", reason="insufficient_uncalibrated_score") if at(withheld_nq) else
                 dict(side="right", raw_outcome="NQ", status="no_new_request", reason="software_rule_only"))
        rows.append(dict(packet_id=f"k{k}", decision_s=t, inputs=dict(participant_id="p", session_id="s"),
                         outputs=[dict(side="left", raw_outcome="NQ", status="no_new_request", reason="software_rule_only"), right]))
    return rows


TIMES = [k/10 for k in range(10, 30)]


def split_request(request_ids=(("r",), ("r",))):
    """One right-hand grasp request 1.0-3.0 s, split into two rows because the LEFT hand's state changes."""
    return [Opportunity("row0", "p", "s", 1.0, 2.0, (("NQ",), ("G",)), "fixture", request_ids=(None, request_ids[0][0])),
            Opportunity("row1", "p", "s", 2.0, 3.0, (None, ("G",)), "fixture", request_ids=(None, request_ids[1][0]),
                        reference_reasons=("excluded", "known"))]


# ---- one request that crosses rows is scored once ------------------------------------------------

def test_one_request_across_rows_is_scored_once_not_per_row():
    ops = split_request()
    timely = per_participant_success(evaluate_opportunities(trace(TIMES, proposals=[1.5]), ops))["p"]
    always = per_participant_success(evaluate_opportunities(trace(TIMES, proposals=TIMES), ops))["p"]
    # Before the fix: one timely proposal scored 1/2 and proposing on every tick scored 2/2.
    assert timely["evaluable_request_hands"] == always["evaluable_request_hands"] == 1
    assert timely["success_fraction"] == always["success_fraction"] == 1.0
    assert timely["request_continuation_rows"] == 1
    right = evaluate_opportunities(trace(TIMES, proposals=TIMES), ops)[0]["hands"][1]
    assert right["grouped_opportunities"] == ["row0", "row1"] and right["repeated_proposals"] == len(TIMES) - 1


def test_late_first_proposal_in_the_second_row_still_counts_once():
    ops = split_request()
    result = evaluate_opportunities(trace(TIMES, proposals=[2.4]), ops)
    assert [h["result"] for o in result for h in o["hands"][1:]] == ["compatible_first", "continuation_of_request"]
    assert result[0]["hands"][1]["first_proposal"] == (2.4, "G")
    assert result[0]["joint_compatible"] is None  # a per-row joint verdict would mix a two-row hand result


def test_grouped_request_with_a_window_gap_is_refused():
    ops = [Opportunity("a", "p", "s", 1., 2., (("NQ",), ("G",)), "f", hand_windows_s=((1., 2.), (1., 1.8)), request_ids=(None, "r")),
           Opportunity("b", "p", "s", 2., 3., (("NQ",), ("G",)), "f", request_ids=(None, "r"))]
    with pytest.raises(ValueError, match="contiguous"):
        evaluate_opportunities(trace(TIMES), ops)


def test_request_id_only_on_a_known_request_hand():
    with pytest.raises(ValueError, match="known request"):
        Opportunity("a", "p", "s", 1., 2., (("NQ",), ("G",)), "f", request_ids=("x", None))


def test_ungrouped_opportunities_score_exactly_as_before():
    op = Opportunity("a", "p", "s", 1., 2., (("NQ",), ("G",)), "f", hand_windows_s=((1., 2.), (1.3, 2.)))
    hand = evaluate_opportunities(trace(TIMES, proposals=[1.1, 1.5]), [op])[0]["hands"][1]
    assert hand["result"] == "wrong_timing_first" and hand["first_proposal"] == (1.1, "G") and hand["proposal_count"] == 2


# ---- the importer: automatic request ids, typo columns, rest rows, distinct UNKNOWN/EXCLUDED ------

def rows_csv(*rows):
    keys = ["start_s", "end_s", "left", "right", "note"]
    return [dict(zip(keys, r)) for r in rows]


def import_and_convert(rows, tmp_path, duration):
    from import_requests import convert
    data = convert(rows, "p", "s", "a", "clock", duration)
    path = tmp_path / "requests.json"; path.write_text(json.dumps(data))
    ledger = read_opportunity_ledger(path)
    return data, ledger, to_opportunities(ledger, 0.)


def test_importer_carries_one_request_across_a_row_split(tmp_path):
    data, ledger, (inside, _) = import_and_convert(rows_csv(
        ("0", "1", "NQ", "NQ", ""), ("1", "2", "NQ", "G", ""), ("2", "3", "EXCLUDED", "G", "left holding jar"),
        ("3", "4", "NQ", "NQ", "")), tmp_path, 4.)
    ids = [e.get("request_ids") for e in data["opportunities"]]
    assert ids[1]["right"] == ids[2]["right"] and ids[1]["right"].startswith("auto:right:")
    rows = trace([k/10 for k in range(0, 40)], proposals=[1.5])
    for r in rows:
        r["inputs"] = dict(participant_id="p", session_id="s")
    stats = per_participant_success(evaluate_opportunities(rows, inside))["p"]
    assert stats["evaluable_request_hands"] == 1 and stats["compatible_first"] == 1


def test_importer_explicit_ids_split_back_to_back_requests_and_refuse_reuse():
    from import_requests import convert
    base = dict(left="NQ", note="")
    rows = [dict(base, start_s="0", end_s="1", right="G", right_request_id="a"),
            dict(base, start_s="1", end_s="2", right="G", right_request_id="b")]
    data = convert(rows, "p", "s", "a", "clock", 2.)
    assert [e["request_ids"]["right"] for e in data["opportunities"]] == ["a", "b"]
    rows.append(dict(base, start_s="2", end_s="3", right="G", right_request_id="a"))
    with pytest.raises(ValueError, match="reused"):
        convert(rows, "p", "s", "a", "clock", 3.)
    with pytest.raises(ValueError, match="without a known request"):
        convert([dict(base, start_s="0", end_s="1", right="NQ", right_request_id="a")], "p", "s", "a", "clock", 1.)


def test_importer_rejects_a_mistyped_column():
    from import_requests import convert
    rows = [dict(start_s="0", end_s="1", left="NQ", right="G", right_start="0.4")]  # typo of right_start_s
    with pytest.raises(ValueError, match="unknown timeline columns"):
        convert(rows, "p", "s", "a", "clock", 1.)


def test_rest_rows_are_no_request_intervals_and_exposure_is_counted(tmp_path):
    data, ledger, _ = import_and_convert(rows_csv(("0", "2", "NQ", "NQ", ""), ("2", "3", "NQ", "G", "")), tmp_path, 3.)
    assert [e["kind"] for e in data["opportunities"]] == ["no_request_interval", "request"]
    assert ledger_summary(ledger)["no_request_exposure_s"] == 2.


def test_unknown_excluded_and_unsupported_hands_are_counted_separately(tmp_path):
    _, _, (inside, outside) = import_and_convert(rows_csv(
        ("0", "1", "EXCLUDED", "G", "left holding"), ("1", "2", "UNKNOWN", "G", ""), ("2", "3", "R", "G", "")), tmp_path, 3.)
    assert [o.reference_reasons for o in inside] == [("excluded", "known"), ("unknown", "known"), ("unsupported_action", "known")]
    rows = trace([k/10 for k in range(0, 30)])
    stats = per_participant_success(evaluate_opportunities(rows, inside))["p"]
    assert stats["unknown_reference_hands_by_reason"] == {"excluded": 1, "unknown": 1, "unsupported_action": 1}
    assert len(outside) == 1 and outside[0].reference_reasons == ("known", "known")


def test_misses_are_split_into_withheld_request_never_requested_and_no_ticks():
    ops = [Opportunity("a", "p", "s", 1., 1.5, (("NQ",), ("G",)), "f"),   # the model asked for G; supervisor withheld
           Opportunity("b", "p", "s", 1.5, 2., (("NQ",), ("G",)), "f"),   # only low-confidence NQ, also withheld
           Opportunity("c", "p", "s", 5., 6., (("NQ",), ("G",)), "f")]    # no replayed tick at all
    rows = trace(TIMES, withheld=[1.1, 1.2], withheld_nq=[1.6, 1.7])
    stats = per_participant_success(evaluate_opportunities(rows, ops))["p"]
    assert stats["missed_with_withheld_request_ticks"] == 1
    assert stats["missed_model_never_requested"] == 1  # a withheld NQ is not a withheld request
    assert stats["missed_no_decision_ticks"] == 1
    assert stats["missed_window_output_ticks"]["withheld_request:insufficient_uncalibrated_score"] == 2
    assert stats["missed_window_output_ticks"]["withheld_no_request:insufficient_uncalibrated_score"] == 2


def test_alternatives_written_in_another_order_are_the_same_request(tmp_path):
    from import_requests import convert
    data = convert(rows_csv(("0", "1", "NQ", "G|FP", ""), ("1", "2", "EXCLUDED", "FP|G", "left holding")), "p", "s", "a", "clock", 2.)
    path = tmp_path / "r.json"; path.write_text(json.dumps(data))
    inside, _ = to_opportunities(read_opportunity_ledger(path), 0., admitted_actions=("G", "FP"))
    stats = per_participant_success(evaluate_opportunities(trace([k/10 for k in range(20)], proposals=[.5]), inside))["p"]
    assert stats["evaluable_request_hands"] == 1 and stats["compatible_first"] == 1


def test_importer_and_evaluator_share_one_boundary_tolerance(tmp_path):
    _, _, (inside, _) = import_and_convert(rows_csv(("0", "1.0000005", "NQ", "G", ""), ("1.0", "2", "EXCLUDED", "G", "left holding")), tmp_path, 2.)
    stats = per_participant_success(evaluate_opportunities(trace([k/10 for k in range(20)], proposals=[.5]), inside))["p"]
    assert stats["evaluable_request_hands"] == 1


def test_out_of_scope_copies_keep_ids_only_for_the_masked_hand(tmp_path):
    _, _, (inside, outside) = import_and_convert(rows_csv(
        ("0", "1", "NQ", "G", ""), ("1", "2", "R", "G", ""), ("2", "3", "R", "G", "")), tmp_path, 3.)
    assert len({o.request_ids[1] for o in inside}) == 1  # the right grasp is one request in the scored list
    assert all(o.request_ids[1] is None and o.request_ids[0] is not None for o in outside)


# ---- perception evaluation scores the runtime left/right outputs ---------------------------------

def perception_labels():
    return [LandmarkLabel("frame", "p", "s", "left", "hand_box", "visible", None, None, "a", "unmarked", box=(0, 0, 20, 20)),
            LandmarkLabel("frame", "p", "s", "right", "hand_box", "visible", None, None, "a", "unmarked", box=(50, 0, 70, 20)),
            LandmarkLabel("frame", "p", "s", "left", "thumb_mcp", "visible", 5, 5, "a", "unmarked")]


def detection(box, side):
    return dict(box=box, side=side, points={"thumb_mcp": (5, 5)})


def test_perception_evaluation_scores_what_the_tracker_emitted():
    from evaluate_perception import evaluate
    raw = [detection((0, 0, 20, 20), "left"), detection((50, 0, 70, 20), "right")]
    withheld = dict(participant_id="p", session_id="s", frame_id="frame", detections=raw, assigned=dict(left=None, right=None))
    result = evaluate([withheld], perception_labels())
    assert result["counts"]["matched_hands"] == 2  # the raw detector found both hands ...
    assert result["tracker"]["counts"]["withheld_or_missed"] == 2  # ... but the runtime used neither
    swapped = dict(withheld, assigned=dict(left=detection((50, 0, 70, 20), "left"), right=None))
    assert evaluate([swapped], perception_labels())["tracker"]["counts"]["on_other_reference_hand"] == 1
    bystander = dict(withheld, assigned=dict(left=detection((200, 200, 210, 210), "left"), right=None))
    assert evaluate([bystander], perception_labels())["tracker"]["counts"]["not_on_any_reference_hand"] == 1
    correct = dict(withheld, assigned=dict(left=raw[0], right=raw[1]))
    tracked = evaluate([correct], perception_labels())["tracker"]
    assert tracked["counts"]["correct_identity"] == 2 and tracked["point_error_px"]["thumb_mcp"]["median"] == 0


def test_a_swap_is_counted_when_the_named_hand_is_out_of_frame():
    from evaluate_perception import evaluate
    labels = [LandmarkLabel("frame", "p", "s", "left", "hand_box", "out_of_frame", None, None, "a", "unmarked"),
              LandmarkLabel("frame", "p", "s", "right", "hand_box", "visible", None, None, "a", "unmarked", box=(50, 0, 70, 20))]
    row = dict(participant_id="p", session_id="s", frame_id="frame", detections=[],
               assigned=dict(left=detection((50, 0, 70, 20), "left"), right=None))  # the right glove emitted as "left"
    counts = evaluate([row], labels)["tracker"]["counts"]
    # since 5 October 2026 the swap onto an absent side has its own counter, so each group sums to its denominator
    assert counts["out_of_frame_side_emitted_on_other_reference_hand"] == 1 and "emitted_while_reference_out_of_frame" not in counts
    assert "on_other_reference_hand" not in counts


def test_perception_evaluation_says_when_assignments_were_not_logged():
    from evaluate_perception import evaluate
    row = dict(participant_id="p", session_id="s", frame_id="frame", detections=[])
    assert "not logged" in evaluate([row], perception_labels())["tracker"]["status"]
    unlabelled = dict(row, frame_id="other-frame", assigned=dict(left=None, right=None))
    assert "no labelled frame" in evaluate([unlabelled], perception_labels())["tracker"]["status"]


# ---- export: occluded hand boxes are skipped, and nothing is written when a label is invalid ------

def export_sessions(tmp_path, extra_label=None):
    from test_rgbd import frame
    from sg_offline.recordings import RECORDING_SCHEMA
    f = frame(); np.save(tmp_path/"rgb.npy", f.rgb); np.save(tmp_path/"depth.npy", f.depth)
    entries = []
    for split, person in [("train", "p1"), ("val", "p2")]:
        ids = [f"{split}-0", f"{split}-1"]
        manifest = dict(schema=RECORDING_SCHEMA, participant_id=person, session_id=split, camera_model="synthetic",
                        calibration=asdict(f.calibration),
                        frames=[dict(frame_id=i, rgb_path="rgb.npy", depth_path="depth.npy", rgb_stamp=asdict(replace(f.rgb_stamp, sample_id=i+":rgb", acquired_s=k, available_s=k)),
                                     depth_stamp=asdict(replace(f.depth_stamp, sample_id=i+":depth", acquired_s=k, available_s=k)),
                                     registered_to_rgb=True, max_sync_skew_s=0) for k, i in enumerate(ids)])
        (tmp_path/(split+".json")).write_text(json.dumps(manifest))
        labels = []
        for k, i in enumerate(ids):
            left_state = "occluded" if (split == "train" and k == 1) else "visible"
            labels += [LandmarkLabel(i, person, split, "left", "hand_box", left_state, None, None, "a", "unmarked",
                                     box=(0, 0, 6, 4) if left_state == "visible" else None),
                       LandmarkLabel(i, person, split, "right", "hand_box", "out_of_frame", None, None, "a", "unmarked")]
            labels += [LandmarkLabel(i, person, split, "left", p, "visible", 2+j, 2, "a", "unmarked") for j, p in enumerate(FUNCTIONAL_POINTS)]
        if extra_label and split == "val":
            labels = [extra_label(l) for l in labels]
        write_landmark_ledger(tmp_path/(split+".jsonl"), labels)
        entries.append(dict(split=split, manifest=str(tmp_path/(split+".json")), ledger=str(tmp_path/(split+".jsonl"))))
    return entries


def test_export_skips_occluded_hand_boxes_instead_of_aborting(tmp_path):
    from prepare_hand_training import export
    report = export(dict(sessions=export_sessions(tmp_path)), tmp_path/"out", "pose")
    assert report["frames"] == {"train": 1, "val": 2, "test": 0}
    assert report["skipped_frames_hand_box_occluded_or_ambiguous"]["train"] == 1
    assert "flip_idx" not in json.loads((tmp_path/"out"/"dataset.yaml").read_text())  # lets Ultralytics keep flips off


def test_export_writes_nothing_when_any_label_is_invalid(tmp_path):
    from prepare_hand_training import export
    outside = lambda l: replace(l, u=999.) if l.point == "thumb_mcp" else l  # a point outside the image, in the second split
    with pytest.raises(ValueError, match="outside image"):
        export(dict(sessions=export_sessions(tmp_path, outside)), tmp_path/"out", "pose")
    assert not (tmp_path/"out").exists()  # before the fix, the first split's images were already written


# ---- extraction covers the 270 s script ----------------------------------------------------------

def test_default_extraction_covers_the_270_second_script_at_30_fps():
    adapter = pytest.importorskip("d435_recording_adapter")
    import inspect
    assert adapter.DEFAULT_MAX_FRAMES >= 270 * 30
    assert inspect.signature(adapter.convert).parameters["max_frames"].default == adapter.DEFAULT_MAX_FRAMES


# ---- comparison: M-C arm, burden next to success, outcomes and cadence in the report --------------

def synthetic_sessions():
    sessions = []
    for person in ("p1", "p2"):
        packets = [replace(p, participant_id=person, session_id="s") for p in scenarios()]
        ops = [Opportunity("a", person, "s", .5, .9, (("G",), ("NQ",)), "synthetic_fixture"),
               Opportunity("b", person, "s", .9, 1.4, (("NQ",), ("NQ",)), "synthetic_fixture")]
        sessions.append(dict(packets=packets, opportunities=ops))
    return sessions


def test_comparison_runs_mc_rules_and_reports_burden_outcomes_and_cadence():
    from sg_offline.experiment import compare, factory
    assert factory("rules_mc").coordination is True and factory("rules").coordination is False
    result = compare(synthetic_sessions(), arms=("rules", "rules_mc", "flat", "kt_set"), steps=5, thresholds=(.6,), synthetic_check=True)
    mc = [f for f in result["folds"] if f["arm"] == "rules_mc"]
    assert len(mc) == 2 and all(f["steps"] == 0 and f["loss_before"] is None for f in mc)
    # 2026-09-23 review completion: the representation contrast is now one of the-named contrasts.
    contrast = [c for c in result["paired_contrasts"] if c["role"] == "secondary_representation"][0]
    assert "burden_difference_proposals_per_hand_minute" in contrast
    assert contrast["status"] == "descriptive_fixed_threshold"  # fixed threshold: not tuned
    metrics = result["folds"][0]["reporting_metrics"]
    assert metrics["outcomes"] and all("first_proposal" in h for o in metrics["outcomes"] for h in o["hands"])
    assert metrics["cadence"]["s"]["median_tick_interval_s"] == pytest.approx(.1)


def test_temporal_arms_have_runnable_matched_pairs():
    pytest.importorskip("torch")
    from sg_offline.experiment import ARMS, factory
    for kind in ("gru", "transformer", "tcn"):
        assert {kind, kind+"_pooled", kind+"_flat"} <= set(ARMS)
        assert [factory(kind+s).representation for s in ("", "_pooled", "_flat")] == ["set", "pooled", "flat"]


# ---- temporal models: stable loss and window guards ----------------------------------------------

def recorded_packets(gap_after=None):
    out = []
    for k, packet in enumerate(scenarios()):
        shift = 1.0 if gap_after is not None and k > gap_after else 0.
        def restamp(pt):
            return replace(pt, stamp=replace(pt.stamp, source=Source.RECORDED, acquired_s=pt.stamp.acquired_s+shift,
                                             available_s=pt.stamp.available_s+shift))
        out.append(replace(packet, decision_s=packet.decision_s+shift, hands=tuple(restamp(p) for p in packet.hands),
                           objects=tuple(restamp(p) for p in packet.objects), telemetry=(), participant_id="p1", session_id="s1"))
    return out


def test_loss_keeps_a_gradient_for_confidently_wrong_examples():
    torch = pytest.importorskip("torch")
    from sg_offline.sequence import SequenceReadout
    from sg_offline.training import bundles_for
    model = SequenceReadout("gru", window=3, hidden=4)
    windows = [tuple(b for _, b in bundles_for(scenarios()))[:3]]
    f, w, n = model._tensors(windows)
    y = torch.zeros(1, 49); y[0, 8] = 1.
    old = -torch.log((model._forward(f, w, n)*y).sum(dim=1).clamp_min(1e-12)).mean()
    new = -model._log_likelihood(f, w, n, y).mean()
    assert float(new.detach()) == pytest.approx(float(old.detach()), rel=1e-5)  # identical where representable
    with torch.no_grad():
        model.head.bias[8] = -80.  # the only acceptable outcome becomes ~1e-35 likely
    loss = -model._log_likelihood(f, w, n, y).mean()
    loss.backward()
    assert float(loss.detach()) > 30 and float(model.head.bias.grad.abs().sum()) > 0  # the old clamp gave exactly zero here


def test_training_windows_reset_at_the_replay_gap_by_default():
    from sg_offline.sequence import build_windows
    from sg_offline.training import bundles_for
    windows = [w for _, w in build_windows(bundles_for(recorded_packets(gap_after=3)), 5)]
    assert [len(w) for w in windows[3:6]] == [4, 1, 2]  # history cleared at the 1.1 s gap, as Replay.run does


def test_recorded_training_refuses_windows_built_for_another_length():
    pytest.importorskip("torch")
    from sg_offline.sequence import SequenceReadout, sequence_examples
    packets = recorded_packets()
    ops = [Opportunity("g", "p1", "s1", .5, .9, (("G",), ("NQ",)), "fixture"),
           Opportunity("n", "p1", "s1", .9, 1.4, (("NQ",), ("NQ",)), "fixture")]
    windows, _ = sequence_examples(packets, ops, window=4)
    assert all(w.window == 4 and w.max_gap_s == .25 for w in windows)
    with pytest.raises(ValueError, match="different window length"):
        SequenceReadout("gru", window=15, hidden=8).fit_recorded(windows, split_id="x", steps=1)
    model = SequenceReadout("gru", window=4, hidden=8).fit_recorded(windows, split_id="x", steps=2)
    assert "reset_gap_s=0.25" in model.fit_provenance


# ---- temporal tests that bite: padding and newest-tick truncation (review finding 21) ------------

@pytest.mark.parametrize("kind", ("gru", "transformer", "tcn"))
def test_padding_to_a_longer_window_does_not_change_the_prediction(kind):
    pytest.importorskip("torch")
    from sg_offline.sequence import SequenceReadout
    from sg_offline.training import bundles_for
    ticks = tuple(b for _, b in bundles_for(scenarios()))[:4]
    short = SequenceReadout(kind, window=5, hidden=8, seed=2)
    long = SequenceReadout(kind, window=30, hidden=8, seed=2)
    if kind == "transformer":  # same weights; only the positional table is longer
        long.pos.data[:, :5] = short.pos.data
    for a, b in zip([short.embed, short.core, short.head], [long.embed, long.core, long.head]):
        if kind != "tcn" or a is not short.core:
            b.load_state_dict(a.state_dict())
    if kind == "tcn":
        pytest.skip("the dilated stack depth depends on the window, so weights cannot be shared")
    assert np.allclose(short.predict_window(ticks), long.predict_window(ticks), atol=1e-6)


def test_an_over_long_window_keeps_the_newest_ticks():
    pytest.importorskip("torch")
    from sg_offline.sequence import SequenceReadout
    from sg_offline.training import bundles_for
    ticks = tuple(b for _, b in bundles_for(scenarios()))
    model = SequenceReadout("gru", window=3, hidden=8, seed=2)
    assert np.allclose(model.predict_window(ticks), model.predict_window(ticks[-3:]), atol=1e-7)
    assert not np.allclose(model.predict_window(ticks), model.predict_window(ticks[:3]), atol=1e-4)
