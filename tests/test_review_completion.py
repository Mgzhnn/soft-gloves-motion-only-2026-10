"""Review-completion fixes (2026-09-23): reproductions of the remaining review findings. Synthetic (N=0).

Each test marked regression failed on the pre-fix snapshot (an earlier snapshot);
tests marked GUARD protect behaviour that was already correct; a RULE test pins a deliberate behaviour
choice (it differs from the snapshot by design, not because the snapshot was wrong). No test sets or validates a
perception threshold: those need labelled recordings and are listed in the issue ledger.
"""
from dataclasses import asdict, replace
import json
import numpy as np
import pytest
from sg_offline.fixtures import make_packet, scenarios
from sg_offline.ledger import FUNCTIONAL_POINTS, LandmarkLabel, write_landmark_ledger, read_landmark_ledger
from sg_offline.perception import AnatomicalTracker, ColourObjects, HandDetection, packet_from_detections
from sg_offline.state import CausalKinematics, RelativeGeometry, DisabledKinetics
from sg_offline.inference import hypotheses, flat_features, FlatJoint, HypothesisPoolMLP, Bundle


def det(box, side, conf=.8):
    centre = ((box[0]+box[2])/2, (box[1]+box[3])/2)
    return HandDetection(box, side, conf, {p: centre for p in FUNCTIONAL_POINTS}, "test")


# ---- identity: single-hand flips and nested duplicates -------------------------------------------

def test_a_lone_hand_cannot_flip_to_the_other_side():
    tracker = AnatomicalTracker()
    assert tracker.assign([det((100, 100, 160, 160), "right")], 1.0)["right"] is not None
    # The same glove, same place, now classed "left" while the right hand is not seen this frame.
    out = tracker.assign([det((102, 101, 161, 162), "left")], 1.033)
    assert out["left"] is None and out["right"] is None  # before: emitted as the LEFT hand


def test_RULE_a_class_conflict_at_one_place_costs_one_frame_not_the_rest_of_the_track():
    # Real pattern from the bench clip: after a detection gap a lone box is taken by its class ("left"),
    # then the true right hand is detected at the same place. Withhold that frame only, then trust the
    # consistent detections again (the earlier rule kept blocking the right hand until the track expired).
    tracker = AnatomicalTracker()
    tracker.assign([det((275, 293, 455, 455), "left", .3)], 1.0)
    assert tracker.assign([det((294, 179, 456, 302), "right", .7)], 1.067)["right"] is None  # conflicting evidence
    assert tracker.assign([det((294, 180, 456, 303), "right", .7)], 1.133)["right"] is not None  # next frame decides


def test_GUARD_a_real_other_hand_elsewhere_or_beside_a_confirmed_hand_is_accepted():
    tracker = AnatomicalTracker()
    tracker.assign([det((100, 100, 160, 160), "right")], 1.0)
    assert tracker.assign([det((300, 100, 360, 160), "left")], 1.033)["left"] is not None  # away from the right
    tracker = AnatomicalTracker()
    tracker.assign([det((100, 100, 160, 160), "right")], 1.0)
    out = tracker.assign([det((101, 100, 161, 160), "right"), det((150, 100, 210, 160), "left")], 1.033)
    assert out["right"] is not None and out["left"] is not None  # right confirmed; touching left hand kept


def test_a_nested_box_of_the_other_class_does_not_make_one_glove_two_hands():
    big, small = det((100, 100, 200, 200), "right", .9), det((120, 120, 170, 170), "left", .6)  # IoU 0.25
    out = AnatomicalTracker().assign([big, small], 1.0)
    assert out["left"] is None and out["right"] is None  # before: both sides assigned from one glove


def test_GUARD_two_separate_hands_are_still_both_assigned():
    out = AnatomicalTracker().assign([det((100, 100, 160, 160), "right"), det((300, 100, 360, 160), "left")], 1.0)
    assert out["left"] is not None and out["right"] is not None


# ---- colour objects: depth fallback and distinct reasons (ranges stay unvalidated settings) ------

def colour_scene():
    rgb = np.zeros((60, 80, 3), dtype=np.uint8); depth = np.full((60, 80), 800, dtype=np.uint16)
    rgb[10:40, 10:40] = (40, 200, 70)          # one green region
    depth[20:30, 20:30] = 0                     # no depth at its centre (e.g. the cup opening)
    return rgb, depth


def test_object_point_falls_back_to_a_valid_depth_pixel_inside_the_region():
    rgb, depth = colour_scene()
    old_pixel = ColourObjects(min_area=10).pixels(rgb)["green_cup"]
    assert depth[old_pixel[1], old_pixel[0]] == 0  # the defect: the chosen pixel has no depth
    pixel, reason = ColourObjects(min_area=10).locate(rgb, depth, invalid=0)["green_cup"]
    assert pixel is not None and depth[pixel[1], pixel[0]] != 0 and 10 <= pixel[0] < 40 and 10 <= pixel[1] < 40
    assert reason == ""


def test_absent_and_ambiguous_colours_have_different_reasons():
    rgb, depth = colour_scene()
    rgb[45:55, 60:75] = (40, 200, 70)  # a second green region
    found = ColourObjects(min_area=10).locate(rgb, depth, invalid=0)
    assert found["green_cup"] == (None, "colour_ambiguous_multiple_regions")
    assert found["red_marker"] == (None, "colour_absent")


def test_the_packet_keeps_the_object_reason():
    from test_rgbd import frame
    f = frame()
    packet, _ = packet_from_detections(f, {"left": None, "right": None}, {"green_cup": (None, "colour_absent")},
                                       participant_id="p", session_id="s", available_s=f.available_s)
    assert packet.objects[0].xyz is None and packet.objects[0].reason == "colour_absent"


# ---- depth: record what a threshold would need, set none -----------------------------------------

def test_perception_diagnostics_record_point_depths_and_spread():
    from test_rgbd import frame
    f = frame()
    hand = HandDetection((0, 0, 6, 4), "left", .9, {"thumb_mcp": (1, 1), "index_mcp": (3, 2), "trapezium": (5, 3)}, "test")
    _, diag = packet_from_detections(f, {"left": hand, "right": None}, {}, participant_id="p", session_id="s",
                                     available_s=f.available_s)
    assert set(diag["left"]["point_xyz"]) == set(FUNCTIONAL_POINTS)
    assert diag["left"]["depth_spread_m"] == pytest.approx(0.) and diag["right"]["depth_spread_m"] is None


# ---- flat comparator: stable, information-matched, comparable capacity ---------------------------

def two_object_bundle(left_x, left_rate=None):
    objs = (("cup", (0., 0., 1.)), ("jar", (.2, 0., 1.)))
    first = make_packet(0, (left_x - (left_rate or 0.)*.1, 0., 1.), (.6, 0., 1.), objects=objs)
    now = make_packet(1, (left_x, 0., 1.), (.6, 0., 1.), objects=objs)
    kin, rel = CausalKinematics(), RelativeGeometry()
    for p in (first, now):
        motions, relatives = kin.update(p), rel.update(p)
    return hypotheses(now, motions, DisabledKinetics().estimate(now), False, .25, "weak_speed", relatives)


def test_flat_input_is_continuous_when_the_hand_passes_between_two_objects():
    # Continuity means the change shrinks with the move; a slot-sorted input jumps by 2.0 however small
    # the move (2026-09-24: criterion made scale-aware because the complete encoding has product terms).
    change = lambda half: float(np.abs(flat_features(two_object_bundle(.1 - half)) - flat_features(two_object_bundle(.1 + half))).max())
    assert change(.00001) < .01                                   # a 0.02 mm move (before: 2.0)
    assert change(.0001) == pytest.approx(change(.001) / 10, rel=.3)  # ten times smaller move, ten times smaller change


def test_GUARD_flat_input_ignores_names_and_hypothesis_order_but_keeps_information():
    bundle = two_object_bundle(.05, left_rate=-.2)
    renamed = replace(bundle, hypotheses=tuple(replace(h, object_ids=tuple(None if s is None else "z"+s for s in h.object_ids))
                                               for h in bundle.hypotheses))
    order = np.random.default_rng(0).permutation(len(bundle.hypotheses))
    shuffled = Bundle(tuple(bundle.hypotheses[i] for i in order), bundle.features[order], bundle.weights[order],
                      bundle.source_kinds, bundle.feature_mode)
    assert np.allclose(flat_features(bundle), flat_features(renamed)) and np.allclose(flat_features(bundle), flat_features(shuffled))
    faster = two_object_bundle(.05, left_rate=-.4)  # the same geometry, a different approach rate
    assert not np.allclose(flat_features(bundle), flat_features(faster))


def test_flat_capacity_and_initialisation_match_kt_set():
    from sg_offline.experiment import parameter_count
    flat, kt_set = FlatJoint(), HypothesisPoolMLP()
    assert abs(parameter_count(flat) - parameter_count(kt_set)) / parameter_count(kt_set) < .05  # before: 18,705 vs 2,051
    assert np.std(flat.B) == pytest.approx(np.std(kt_set.B), rel=.3)  # same initialisation rule (before: 0.21 vs 0.03)


def test_the_temporal_flat_variant_uses_the_same_stable_input():
    pytest.importorskip("torch")
    from sg_offline.sequence import SequenceReadout
    from sg_offline.inference import FLAT_DIM
    assert SequenceReadout("gru", representation="flat").embed[0].in_features == FLAT_DIM  # before: 19 x 42 = 798


# ---- comparable training ----------------------------------------------------------------------

def recorded_windows(n_copies=2):
    from sg_offline.contracts import Source
    from sg_offline.evaluation import Opportunity
    from sg_offline.sequence import sequence_examples
    out = []
    for c in range(n_copies):
        packets = []
        for k, p in enumerate(scenarios()):
            restamp = lambda pt: replace(pt, stamp=replace(pt.stamp, source=Source.RECORDED, sample_id=f"{c}:{pt.stamp.sample_id}"))
            packets.append(replace(p, packet_id=f"{c}-{p.packet_id}", hands=tuple(restamp(q) for q in p.hands),
                                   objects=tuple(restamp(q) for q in p.objects), telemetry=(), participant_id="p1", session_id=f"s{c}"))
        ops = [Opportunity("g", "p1", f"s{c}", .5, .9, (("G",), ("NQ",)), "f"), Opportunity("n", "p1", f"s{c}", .9, 1.4, (("NQ",), ("NQ",)), "f")]
        out.extend(sequence_examples(packets, ops, window=4)[0])
    return out


def test_one_sequence_training_step_is_one_pass_over_all_windows():
    pytest.importorskip("torch")
    from sg_offline.sequence import SequenceReadout
    windows = recorded_windows(n_copies=12)  # more windows than one minibatch of 64
    model = SequenceReadout("gru", window=4, hidden=8)
    seen = []
    original = model._log_likelihood
    def counting(f, w, n, y):
        seen.append(int(f.shape[0])); return original(f, w, n, y)
    model._log_likelihood = counting
    model._fit([w.bundles for w in windows], [w.label for w in windows], steps=1)
    training_batches = seen[1:-1]  # first and last calls are the loss checkpoints on all windows
    assert sum(training_batches) == len(windows) > 64  # before: one minibatch of 64 per step


def test_the_seed_changes_minibatch_order_not_only_initialisation():
    pytest.importorskip("torch")
    from sg_offline.sequence import SequenceReadout
    windows = recorded_windows(n_copies=12)
    a, b = SequenceReadout("gru", window=4, hidden=8, seed=1), SequenceReadout("gru", window=4, hidden=8, seed=2)
    for ma, mb in zip((a.embed, a.core, a.head), (b.embed, b.core, b.head)):
        mb.load_state_dict(ma.state_dict())  # identical starting weights
    for m in (a, b):
        m._fit([w.bundles for w in windows], [w.label for w in windows], steps=1)
    assert not np.allclose(a.predict_window(windows[0].bundles), b.predict_window(windows[0].bundles))


def people_sessions(people=("p1", "p2", "p3"), flip=None):
    from sg_offline.evaluation import Opportunity
    sessions = []
    for k, person in enumerate(people):
        packets = [replace(p, participant_id=person, session_id="s",
                           hands=tuple(replace(h, xyz=None if h.xyz is None else (h.xyz[0]+.01*k, h.xyz[1], h.xyz[2])) for h in p.hands))
                   for p in scenarios()]
        g, n = (("NQ",), ("G",)) if flip == person else (("G",), ("NQ",)), (("NQ",), ("NQ",))
        sessions.append(dict(packets=packets, opportunities=[Opportunity("a", person, "s", .5, .9, g, "f"),
                                                              Opportunity("b", person, "s", .9, 1.4, n, "f")]))
    return sessions


def test_development_early_stopping_keeps_the_best_development_loss():
    from sg_offline.experiment import compare
    # p3 is the development person for the p1 and p2 folds; its labels contradict the training person's,
    # so development loss rises with training and the best pass comes early.
    result = compare(people_sessions(flip="p3"), arms=("kt_set",), steps=12, thresholds=(.5,), selection="development",
                     max_unwanted_proposals_per_hand_minute=1e9, synthetic_check=True)
    checked = 0
    for fold in result["folds"]:
        stop = fold["early_stopping"]
        assert stop["criterion"] == "development log loss" and 1 <= stop["best_pass"] <= 12
        assert stop["development_loss_at_best"] == pytest.approx(min(stop["development_loss_curve"]))
        # the model really ends at the best pass, not the last one
        assert stop["development_loss_after_restore"] == pytest.approx(stop["development_loss_at_best"])
        if stop["best_pass"] < 12 and stop["development_loss_curve"][-1] > stop["development_loss_at_best"] + 1e-9:
            checked += 1
    assert checked >= 1


def test_GUARD_a_held_out_persons_labels_never_reach_their_own_fold():
    from sg_offline.experiment import compare
    runs = [compare(people_sessions(("p1", "p2"), flip=f), arms=("kt_set",), steps=6, thresholds=(.6,), synthetic_check=True)
            for f in (None, "p2")]
    curve = lambda r, held: [f["fitting_diagnostics"]["loss_curve"] for f in r["folds"] if f["held_out_person"] == held][0]
    assert curve(runs[0], "p2") == curve(runs[1], "p2")   # p2's labels changed; p2's own fold is unaffected
    assert curve(runs[0], "p1") != curve(runs[1], "p1")   # the fold trained on p2 does change


# ---- the primary contrast of accepted, and budget-infeasible reporting points --------------

def fold(arm, held, success, burden, selection="development", feasible=True, dev_success=None, seed=0, threshold=.6):
    curve = [] if selection == "fixed" else [dict(threshold=threshold, feasible=feasible, success=dev_success, metrics={})]
    return dict(arm=arm, held_out_person=held, seed=seed, threshold_selection=selection, reporting_threshold=threshold,
                development_budget_feasible=feasible, development_curve=curve,
                reporting_metrics=dict(success={held: dict(success_fraction=success)},
                                       unwanted={held: dict(proposals_per_hand_minute=burden)}))


def test_the_primary_contrast_is_kt_r_against_the_strongest_simple_learner():
    # 2 Oct 2026: the simple controls are the fitted M-I / M-C models (mi_learned / mc_learned); the hand-authored
    # rules moved to a descriptive floor. The selection rule (development success) is unchanged.
    from sg_offline.experiment import contrasts_from_folds
    folds = [fold("kt_r", "p1", .8, 2.), fold("mi_learned", "p1", .5, 1., dev_success=.4), fold("mc_learned", "p1", .6, 1., dev_success=.7),
             fold("kt_set", "p1", .7, 2.), fold("flat", "p1", .65, 2.)]
    contrasts = contrasts_from_folds(folds, (0,))
    primary = contrasts[0]
    assert primary["role"] == "primary" and "primary contrast" in primary["label"]
    assert primary["control_by_person"] == {"p1": "mc_learned"}  # the development-selected stronger simple learner
    assert primary["per_person"]["p1"] == pytest.approx(.8 - .6) and primary["n_people"] == 1
    assert [c["role"] for c in contrasts] == ["primary", "mechanism_mean_collapse", "mechanism_hard_collapse", "matched_control",
                                              "matched_information", "secondary_representation", "descriptive_floor"]


def test_budget_infeasible_points_are_excluded_and_listed_not_averaged():
    from sg_offline.experiment import contrasts_from_folds
    folds = [fold("kt_r", "p1", .9, 9., feasible=False), fold("mi_learned", "p1", .5, 1., dev_success=.4), fold("mc_learned", "p1", .4, 1., dev_success=.3),
             fold("kt_r", "p2", .8, 1.), fold("mi_learned", "p2", .5, 1., dev_success=.5), fold("mc_learned", "p2", .5, 1., dev_success=.2)]
    primary = contrasts_from_folds(folds, (0,))[0]
    assert primary["excluded_people"] == {"p1": "kt_r reporting point missed the development budget"}
    assert primary["n_people"] == 1 and primary["per_person"] == {"p2": pytest.approx(.3)}


def test_fixed_threshold_contrasts_are_marked_descriptive_and_cannot_pick_a_strongest_rule():
    from sg_offline.experiment import contrasts_from_folds
    folds = [fold(a, "p1", s, 1., selection="fixed") for a, s in (("kt_r", .8), ("mi_learned", .5), ("mc_learned", .6), ("kt_set", .7), ("flat", .6))]
    contrasts = {c["role"]: c for c in contrasts_from_folds(folds, (0,))}
    assert contrasts["primary"]["status"] == "not_selectable_without_development_split" and contrasts["primary"]["n_people"] == 0
    assert contrasts["matched_control"]["status"] == "descriptive_fixed_threshold" and contrasts["matched_control"]["n_people"] == 1


def test_compare_reports_the_primary_contrast():
    from sg_offline.experiment import compare
    result = compare(people_sessions(), arms=("rules", "rules_mc", "mi_learned", "mc_learned", "kt_set", "kt_r", "flat"), steps=4, thresholds=(.5, .6),
                     selection="development", max_unwanted_proposals_per_hand_minute=1e9, synthetic_check=True)
    assert "primary contrast" in result["primary_contrast"] and result["paired_contrasts"][0]["role"] == "primary"


# ---- temporal models: buffer on overflow, multi-layer causality ----------------------------------

def test_an_overflowing_tick_does_not_poison_the_history():
    pytest.importorskip("torch")
    from sg_offline.sequence import SequenceReadout
    from sg_offline.training import bundles_for
    ticks = [b for _, b in bundles_for(scenarios())]
    big = two_object_bundle(.05)  # 19 hypotheses
    model = SequenceReadout("gru", window=4, hidden=8, max_hypotheses=18)
    with pytest.raises(ValueError, match="capacity"):
        model.predict(big)
    fresh = SequenceReadout("gru", window=4, hidden=8, max_hypotheses=18)
    assert np.allclose(model.predict(ticks[0]), fresh.predict(ticks[0]))  # before: the refused tick stayed in the buffer


def test_GUARD_two_layer_transformer_stays_causal_and_stateful_equals_stateless():
    pytest.importorskip("torch")
    from sg_offline.sequence import SequenceReadout
    from sg_offline.training import bundles_for
    ticks = [b for _, b in bundles_for(scenarios())]
    model = SequenceReadout("transformer", window=5, hidden=8, layers=2)
    prefix = model.predict_window(tuple(ticks[:4]))
    assert np.allclose(prefix, model.predict_window(tuple(ticks[:4])))
    model.reset()
    for b in ticks[:4]:
        stateful = model.predict(b)
    assert np.allclose(stateful, prefix, atol=1e-6)


# ---- extraction, script and bookkeeping ----------------------------------------------------------

def test_only_the_global_time_domain_is_accepted_for_extraction():
    adapter = pytest.importorskip("d435_recording_adapter")
    assert adapter.timestamp_domain_ok("timestamp_domain.global_time")
    assert not adapter.timestamp_domain_ok("timestamp_domain.hardware_clock")
    assert not adapter.timestamp_domain_ok("timestamp_domain.system_time")


def test_the_script_names_the_marker_on_the_jar_body_and_logs_the_start_time(monkeypatch, capsys):
    import timed_recording
    from pathlib import Path
    script = json.loads((Path(timed_recording.__file__).with_name("recording_script.json")).read_text())
    assert "body" in " ".join(b["instruction"] for b in script["blocks"])
    clock = {"t": 0.}
    monkeypatch.setattr(timed_recording.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(timed_recording.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + 1.))
    monkeypatch.setattr("sys.argv", ["timed_recording.py", "--run"])
    timed_recording.main()
    assert "started at" in capsys.readouterr().out  # a wall-clock line to align prompts with the camera file


def test_hand_window_accounting_does_not_count_already_unknown_hands():
    from sg_offline.evaluation import Opportunity
    from sg_offline.training import label_ticks
    op = Opportunity("r", "synthetic-person", "synthetic-session", .5, 1.4, (None, ("G",)), "f", hand_windows_s=((.5, .7), (.9, 1.4)))
    _, counts = label_ticks(scenarios(), [op])
    assert counts["outside_hand_window"] == 4  # only the right hand's 4 ticks before 0.9 s (before: 11, the unknown left counted too)


def test_a_failed_label_write_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    import sg_offline.ledger as ledger
    box = LandmarkLabel("f", "p", "s", "left", "hand_box", "visible", None, None, "a", "unmarked", box=(0, 0, 6, 4))
    path = tmp_path / "a.jsonl"; write_landmark_ledger(path, [box])
    before = path.read_bytes()
    def boom(*a, **k): raise OSError("disk full")
    monkeypatch.setattr(ledger.os, "replace", boom)
    with pytest.raises(OSError):
        write_landmark_ledger(path, [box, replace(box, frame_id="g")])
    assert path.read_bytes() == before and read_landmark_ledger(path) == (box,)


def test_export_provenance_records_glove_condition_and_train_val_person_overlap(tmp_path):
    from test_group_a_fixes import export_sessions
    from prepare_hand_training import export
    report = export(dict(sessions=export_sessions(tmp_path)), tmp_path / "out", "pose")
    assert report["glove_conditions"] == {"train": ["unspecified"], "val": ["unspecified"], "test": []}
    assert report["train_val_people_overlap"] == []


def test_a_declared_checkpoint_hash_is_verified_when_the_file_is_given(tmp_path):
    from import_requests import convert
    from run_comparison import load_sessions
    packets = tmp_path / "packets.jsonl"
    packets.write_text("".join(json.dumps(asdict(p)) + "\n" for p in scenarios()))
    ledger = convert([dict(start_s="0", end_s="2", left="NQ", right="NQ")], "synthetic-person", "synthetic-session", "a", "session", 2.)
    (tmp_path / "ledger.json").write_text(json.dumps(ledger))
    checkpoint = tmp_path / "model.pt"; checkpoint.write_bytes(b"weights")
    import hashlib
    good = hashlib.sha256(b"weights").hexdigest()
    from test_review_implementation import write_manifest
    entry = dict(packets="packets.jsonl", ledger="ledger.json", manifest=write_manifest(tmp_path / "recording.json", "synthetic-person"),
                 perception_provenance=dict(fit="fixed_external", model_id="m", checkpoint_sha256="0"*64, checkpoint_path="model.pt"))
    with pytest.raises(ValueError, match="checkpoint"):
        load_sessions([entry], tmp_path)
    entry["perception_provenance"]["checkpoint_sha256"] = good
    _, sources = load_sessions([entry], tmp_path)
    assert sources[0]["perception"]["checkpoint_hash_verified"] is True


def test_a_box_below_the_confidence_floor_cannot_veto_a_confident_hand():
    """Architecture review, 5 October 2026. The cross-class duplicate veto was computed before the confidence floor, so
    a weak box nested in a confident hand of the other class removed that hand although the weak box itself could
    never be assigned. With `--confidence` lowered to find more hands this removed confident ones."""
    tracker = AnatomicalTracker(min_confidence=.3)
    result = tracker.assign([det((10, 10, 60, 60), "right", .9), det((20, 20, 40, 40), "left", .15)], 0.)
    assert result["right"] is not None and result["left"] is None                 # before: both None
    # two boxes that both count are still an ambiguous pair, at any floor
    result = AnatomicalTracker(min_confidence=.1).assign([det((10, 10, 60, 60), "right", .9), det((20, 20, 40, 40), "left", .15)], 0.)
    assert result == dict(left=None, right=None)
    # the floor is the tracker's setting: a lone hand at 0.2 is assigned when the run asks for 0.1
    assert AnatomicalTracker(min_confidence=.1).assign([det((10, 10, 60, 60), "left", .2)], 0.)["left"] is not None
    assert AnatomicalTracker(min_confidence=.3).assign([det((10, 10, 60, 60), "left", .2)], 0.)["left"] is None


def test_the_colour_provider_uses_the_same_depth_validity_as_the_point_lift():
    """`_nearest_valid_pixel` accepted any pixel that was not the invalid marker; `RGBDFrame.point_at` also rejects
    non-finite and non-positive depth. On a float depth image the colour provider could pick a NaN pixel as the
    object's "valid" pixel and the object was then lost although a valid pixel was next to it."""
    from sg_offline.perception import _nearest_valid_pixel
    depth = np.full((5, 5), 800., np.float32); depth[2, 2] = np.nan; depth[2, 1] = -5.
    mask = np.zeros((5, 5), bool); mask[2, 1:4] = True
    assert _nearest_valid_pixel(depth, mask, (2., 2.)) == (3, 2)                  # before: (2, 2), the NaN pixel
    assert _nearest_valid_pixel(np.full((5, 5), 800, np.uint16), mask, (2., 2.)) == (2, 2)
