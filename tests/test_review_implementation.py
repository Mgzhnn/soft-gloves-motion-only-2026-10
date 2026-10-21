"""Implementation of the 28 September 2026 independent strict review (2 October 2026). Synthetic fixtures, N=0.

regression tests fail on an earlier snapshot (or could not run there because the
feature did not exist); GUARD tests hold on both; RULE tests pin a deliberate behaviour choice. Nothing here
measures accuracy on gloved hands or validates a threshold.
"""
from dataclasses import replace
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from sg_offline.contracts import Packet, Point, Source, Stamp
from sg_offline.evaluation import Opportunity, evaluate_opportunities, proposal_runs
from sg_offline.fixtures import scenarios
from sg_offline.inference import Bundle, FactorizedJoint, HardCollapseLinear, LinearJoint, SimpleRules
from sg_offline.replay import Config, Replay
from sg_offline.state import RelativeGeometry
from sg_offline.stats import paired_interval, per_participant_success, t_975, unwanted_exposure
from sg_offline.training import bundles_for


# ---- fixtures -----------------------------------------------------------------------------------

def trace(times, right_proposals=(), person="p", session="s"):
    """Rows at `times`; the right hand proposes G at the listed times, otherwise NQ; the left hand never proposes."""
    rows = []
    for k, t in enumerate(times):
        prop = any(abs(t - x) < 1e-9 for x in right_proposals)
        right = dict(side="right", raw_outcome="G" if prop else "NQ",
                     status="proposed_simulation_only" if prop else "no_new_request", reason="")
        rows.append(dict(packet_id=f"k{k}", decision_s=t, inputs=dict(participant_id=person, session_id=session),
                         outputs=[dict(side="left", raw_outcome="NQ", status="no_new_request", reason=""), right]))
    return rows


TIMES = [round(k / 10, 6) for k in range(0, 100)]  # 0.0 .. 9.9 s at 10 Hz
REST = Opportunity("rest", "p", "s", 0., 5., (("NQ",), ("NQ",)), "f")
REQUEST = Opportunity("req", "p", "s", 5., 8., (("NQ",), ("G",)), "f")


def ticks_from(start, stop=10.):
    return [t for t in TIMES if start - 1e-9 <= t < stop]


def motion_bundles():
    return [b for _, b in bundles_for(scenarios())]


# ---- the early first command (review B3 / COD-4 / EVA-1) ------------------------------------------

def test_a_proposal_run_that_began_before_the_request_is_early_not_success():
    rows = trace(TIMES, right_proposals=ticks_from(2.0))  # proposing G continuously from 2.0 s; the request starts at 5.0 s
    outcomes = {o["opportunity_id"]: o for o in evaluate_opportunities(rows, [REST, REQUEST])}
    right = outcomes["req"]["hands"][1]
    assert right["result"] == "wrong_timing_first"          # before: compatible_first, success 1.0
    assert right["first_proposal"] == (5.0, "G") and right["proposal_run_start_s"] == pytest.approx(2.0)
    assert right["early_by_s"] == pytest.approx(3.0) and right["run_active_at_window_start"] is True
    rest = outcomes["rest"]["hands"][1]
    assert rest["result"] == "unwanted" and rest["proposal_count"] == 30 and rest["proposal_onsets"] == 1
    stats = per_participant_success(list(outcomes.values()))["p"]
    assert stats["wrong_timing_first"] == 1 and stats["wrong_timing_first_early"] == 1 and stats["success_fraction"] == 0.


def test_RULE_a_declared_early_limit_admits_a_slightly_early_run_and_records_how_early():
    rows = trace(TIMES, right_proposals=ticks_from(4.8))  # 0.2 s before the annotated onset
    strict = evaluate_opportunities(rows, [REST, REQUEST])[1]["hands"][1]
    assert strict["result"] == "wrong_timing_first" and strict["early_by_s"] == pytest.approx(0.2)
    tolerant = evaluate_opportunities(rows, [REST, REQUEST], early_limit_s=0.5)[1]["hands"][1]
    assert tolerant["result"] == "compatible_first" and tolerant["early_by_s"] == pytest.approx(0.2)
    assert tolerant["early_limit_s"] == 0.5
    with pytest.raises(ValueError):
        evaluate_opportunities(rows, [REST, REQUEST], early_limit_s=-1.)


def test_GUARD_a_burst_that_ended_before_the_window_does_not_fail_the_request_but_is_counted():
    rows = trace(TIMES, right_proposals=ticks_from(2.0, 2.6) + ticks_from(5.3))
    outcomes = evaluate_opportunities(rows, [REST, REQUEST])
    right = outcomes[1]["hands"][1]
    assert right["result"] == "compatible_first" and right["first_proposal"] == (5.3, "G") and right["early_by_s"] == 0.
    assert right["proposals_since_rearm_before_window"] == 6  # the 2.0-2.5 s burst, available for a stricter re-arm rule
    rest = outcomes[0]["hands"][1]
    assert rest["proposal_count"] == 6 and rest["proposal_onsets"] == 1 and rest["first_proposal"] == (2.0, "G")


def test_GUARD_a_run_continuing_from_a_previous_request_is_not_a_new_onset_and_rearm_is_the_previous_request_end():
    first = Opportunity("req1", "p", "s", 0., 2., (("NQ",), ("G",)), "f")
    rest = Opportunity("rest", "p", "s", 2., 5., (("NQ",), ("NQ",)), "f")
    second = Opportunity("req2", "p", "s", 5., 8., (("NQ",), ("G",)), "f")
    rows = trace(TIMES, right_proposals=ticks_from(0.5, 3.0) + ticks_from(6.0))
    outcomes = {o["opportunity_id"]: o for o in evaluate_opportunities(rows, [first, rest, second])}
    assert outcomes["req1"]["hands"][1]["result"] == "compatible_first"
    hand = outcomes["rest"]["hands"][1]
    assert hand["result"] == "unwanted" and hand["run_active_at_window_start"] is True and hand["proposal_onsets"] == 0
    later = outcomes["req2"]["hands"][1]
    assert later["result"] == "compatible_first" and later["rearm_s"] == pytest.approx(2.0)
    assert later["proposals_since_rearm_before_window"] == 10  # 2.0-2.9 s, after the first request ended


def test_onsets_distinguish_one_sustained_burst_from_seven_separate_ones():
    one = trace(TIMES, right_proposals=ticks_from(1.0, 2.4))                        # 14 consecutive ticks
    many = trace(TIMES, right_proposals=[round(1.0 + .4*k + .1*j, 6) for k in range(7) for j in range(2)])  # 7 x 2 ticks
    a = unwanted_exposure(evaluate_opportunities(one, [REST]), [REST])["p"]
    b = unwanted_exposure(evaluate_opportunities(many, [REST]), [REST])["p"]
    assert a["unwanted_proposals"] == b["unwanted_proposals"] == 14          # the tick metric cannot tell them apart
    assert (a["unwanted_onsets"], b["unwanted_onsets"]) == (1, 7)             # the onset metric can
    # exposure is per hand: two hands x 5 s = 10 hand-seconds of no-request time
    assert a["exposure_hand_s"] == pytest.approx(10.)
    assert a["onsets_per_hand_minute"] == pytest.approx(1 * 60 / 10.) and b["onsets_per_hand_minute"] == pytest.approx(7 * 60 / 10.)
    assert a["time_to_first_unwanted_s"] == [None, pytest.approx(1.0)] and a["intervals_with_unwanted"] == 1  # left censored, right at 1.0 s
    quiet = unwanted_exposure(evaluate_opportunities(trace(TIMES), [REST]), [REST])["p"]
    assert quiet["time_to_first_unwanted_s"] == [None, None] and quiet["unwanted_onsets"] == 0  # censored, not invented
    assert proposal_runs([(0., True, "G"), (.1, True, "G"), (.2, True, "R"), (.3, False, "NQ"), (.4, True, "R")]) == [
        dict(start_s=0., end_s=.1, outcome="G", ticks=2), dict(start_s=.2, end_s=.2, outcome="R", ticks=1),
        dict(start_s=.4, end_s=.4, outcome="R", ticks=1)]


# ---- learned simple controls and collapse twins (review B1 / B2 / COD-1 / ARC-1 / ARC-2) ----------

def test_RULE_the_hand_authored_rules_decide_identically_at_every_grid_threshold():
    packets = scenarios(); disagreements = 0; scores = set()
    for t in (.3, .4, .5, .6, .7, .8, .9):
        a = Replay(Config(proposal_score_threshold=t), SimpleRules()).run(packets)
        b = Replay(Config(proposal_score_threshold=t), SimpleRules(coordination=True)).run(packets)
        for ra, rb in zip(a, b):
            for oa, ob in zip(ra["outputs"], rb["outputs"]):
                scores.add(round(oa["grasp_score"], 3)); disagreements += (oa["status"], oa["raw_outcome"]) != (ob["status"], ob["raw_outcome"])
    assert disagreements == 0 and scores <= {.011, .862}  # why the rules are a floor, not the simple controls


def test_learned_mi_ignores_the_other_hand_while_learned_mc_uses_it():
    bundle = motion_bundles()[3]
    rng = np.random.default_rng(1)
    mi, mc = FactorizedJoint(), FactorizedJoint(cross_hand=True, coupling=True)
    for m in (mi, mc):
        for p in m._params():
            p[...] = rng.normal(0, .3, p.shape)
    features = bundle.features.copy(); features[:, 15] += 2.; features[:, 16] -= 1.  # the right hand's distance and approach rate
    other = Bundle(bundle.hypotheses, features, bundle.weights, bundle.source_kinds, bundle.feature_mode)
    left_marginal = lambda m, b: m.predict(b).reshape(7, 7).sum(axis=1)
    assert np.allclose(left_marginal(mi, bundle), left_marginal(mi, other))          # M-I: the left hand is independent of the right
    assert not np.allclose(left_marginal(mc, bundle), left_marginal(mc, other))      # M-C: explicit coordination
    assert (mi.n_parameters(), mc.n_parameters()) == (224, 637)
    assert mi.name == "learned_independent_hands" and mc.name == "learned_coordinated_hands"
    assert abs(mi.predict(bundle).sum() - 1) < 1e-9 and abs(mc.predict(bundle).sum() - 1) < 1e-9
    # M-I's joint is exactly the product of its two per-hand softmaxes.
    u, logits = mi._logits(bundle)
    from sg_offline.inference import softmax
    assert np.allclose(mi.predict(bundle), np.outer(softmax(mi.W_left @ u[0]), softmax(mi.W_right @ u[1])).ravel())


@pytest.mark.parametrize("variant", ["mi", "mc"])
@pytest.mark.parametrize("label", [7, tuple(i * 7 + 1 for i in range(7))])
def test_factorized_gradients_match_finite_differences(variant, label):
    bundle = motion_bundles()[2]
    model = FactorizedJoint() if variant == "mi" else FactorizedJoint(cross_hand=True, coupling=True)
    rng = np.random.default_rng(4)
    for p in model._params():
        p[...] = rng.normal(0, .05, p.shape)
    _, grads = model.loss_gradient(bundle, label, .003)
    eps = 1e-6
    for p, g in zip(model._params(), grads):
        for index in [(0, 0), (3, p.shape[1] - 1), (6, 2)]:
            old = p[index]
            p[index] = old + eps; hi = model.loss_gradient(bundle, label, .003)[0]
            p[index] = old - eps; lo = model.loss_gradient(bundle, label, .003)[0]
            p[index] = old
            assert g[index] == pytest.approx((hi - lo) / (2 * eps), abs=1e-6)


@pytest.mark.parametrize("make", [FactorizedJoint, lambda: FactorizedJoint(cross_hand=True, coupling=True), HardCollapseLinear])
def test_new_arms_fit_synthetically_and_record_provenance(make):
    bs = motion_bundles()[:3]; labels = [0, 7, 7]; model = make()
    before = np.mean([-np.log(model.predict(b)[y]) for b, y in zip(bs, labels)])
    model.fit_synthetic(bs, labels, steps=60)
    after = np.mean([-np.log(model.predict(b)[y]) for b, y in zip(bs, labels)])
    assert after < before and model.fit_provenance.startswith("synthetic_only")
    assert model.training_feature_mode == "motion_only"
    with pytest.raises(ValueError, match="feature mode differs"):
        model.predict(replace(bs[0], feature_mode="motion_and_kinetics"))


def test_collapse_twins_share_kt_r_size_and_hard_collapse_is_order_invariant():
    from sg_offline.experiment import factory, parameter_count
    assert parameter_count(factory("kt_r")) == parameter_count(factory("kt_r_mean")) == parameter_count(factory("kt_r_hard")) == 2058
    bundle = motion_bundles()[7]  # 19 hypotheses
    rng = np.random.default_rng(2)
    W = rng.normal(0, .1, (49, 42))
    mean, hard, kt_r = LinearJoint(False), HardCollapseLinear(), LinearJoint()
    for m in (mean, hard, kt_r):
        m.W[:] = W
    order = rng.permutation(len(bundle.weights))
    permuted = Bundle(tuple(bundle.hypotheses[i] for i in order), bundle.features[order], bundle.weights[order], bundle.source_kinds, bundle.feature_mode)
    assert np.allclose(hard.predict(bundle), hard.predict(permuted))                  # order of hypotheses cannot matter
    assert np.allclose(mean.predict(bundle), mean.predict(permuted))
    via_factory = factory("kt_r_mean"); via_factory.W[:] = W
    assert np.allclose(mean.predict(bundle), via_factory.predict(bundle))              # the arm IS LinearJoint(mixture=False)
    assert not np.allclose(hard.predict(bundle), mean.predict(bundle))                # the three twins read different inputs ...
    assert not np.allclose(kt_r.predict(bundle), mean.predict(bundle))
    from sg_offline.inference import softmax
    top = np.isclose(bundle.weights, bundle.weights.max())
    expected = softmax(bundle.features[top].mean(axis=0) @ W.T)                        # ... and hard collapse reads the top hypothesis only
    assert np.allclose(hard.predict(bundle), expected) and top.sum() >= 1
    assert hard.name == "hard_collapse_linear" and factory("kt_r_mean").name == "direct_linear"


def test_GUARD_experiment_arms_include_the_new_controls_and_the_old_specification_is_kept():
    from sg_offline.experiment import ARMS, CONTRAST_SPECS, SIMPLE_ARMS, RULE_ARMS, factory
    assert {"mi_learned", "mc_learned", "kt_r_mean", "kt_r_hard", "rules", "rules_mc", "flat", "kt_set", "kt_r"} <= set(ARMS)
    assert SIMPLE_ARMS == ("mi_learned", "mc_learned") and RULE_ARMS == ("rules", "rules_mc")
    assert factory("mi_learned").cross_hand is False and factory("mc_learned").coupling is True
    assert [e["role"] for e in CONTRAST_SPECS["primary_contrast"]][:3] == ["primary", "mechanism_mean_collapse", "mechanism_hard_collapse"]
    assert CONTRAST_SPECS["rules_contrast"][0]["control"] == "strongest_rule"  # the 23 Sep primary, reproducible on request


# ---- silent operating points and the development selection (review B1 / COD-1) --------------------

def people_sessions(people=("p1", "p2", "p3")):
    sessions = []
    for k, person in enumerate(people):
        packets = [replace(p, participant_id=person, session_id="s",
                           hands=tuple(replace(h, xyz=None if h.xyz is None else (h.xyz[0] + .01 * k, h.xyz[1], h.xyz[2])) for h in p.hands))
                   for p in scenarios()]
        # a one-tick request (known reference, first tick withheld -> success 0) and a long no-request interval
        # in which the rules do propose: every grid threshold <= 0.8 then violates a zero budget, 0.9 is silent.
        ops = [Opportunity("a", person, "s", .5, .6, (("G",), ("NQ",)), "f"), Opportunity("b", person, "s", .6, 1.4, (("NQ",), ("NQ",)), "f")]
        sessions.append(dict(packets=packets, opportunities=ops))
    return sessions


def test_a_silent_development_point_is_never_selected_as_feasible():
    from sg_offline.experiment import compare
    result = compare(people_sessions(), arms=("rules",), selection="development", max_unwanted_proposals_per_hand_minute=0.,
                     thresholds=(.3, .5, .7, .9), synthetic_check=True)
    for fold in result["folds"]:
        curve = {p["threshold"]: p for p in fold["development_curve"]}
        assert curve[.9]["within_budget"] is True and curve[.9]["proposes"] is False and curve[.9]["feasible"] is False
        assert all(not curve[t]["within_budget"] for t in (.3, .5, .7))
        assert fold["development_budget_feasible"] is False  # before: 0.9 was "feasible" and became the control
        assert fold["development_selection_reason"].startswith("silent")
    assert result["configuration"]["silent_points"]


def test_exclusion_reasons_name_silent_controls():
    from sg_offline.experiment import _exclusion_reason
    budget = dict(arm="kt_r", development_selection_reason="no operating point met the development unwanted-proposal budget")
    silent = dict(arm="kt_set", development_selection_reason="silent: only operating points that never proposed on development data met the budget")
    legacy = dict(arm="flat")
    assert _exclusion_reason(budget) == "kt_r reporting point missed the development budget"
    assert _exclusion_reason(legacy) == "flat reporting point missed the development budget"
    assert _exclusion_reason(silent).startswith("kt_set reporting point was silent")


def test_compare_runs_the_learned_controls_and_collapse_arms_with_mechanism_contrasts():
    from sg_offline.experiment import compare
    sessions = []
    for person in ("p1", "p2"):
        packets = [replace(p, participant_id=person, session_id="s") for p in scenarios()]
        ops = [Opportunity("a", person, "s", .5, .9, (("G",), ("NQ",)), "f"), Opportunity("b", person, "s", .9, 1.4, (("NQ",), ("NQ",)), "f")]
        sessions.append(dict(packets=packets, opportunities=ops))
    result = compare(sessions, arms=("mi_learned", "mc_learned", "kt_r", "kt_r_mean", "kt_r_hard"), steps=20, thresholds=(.6,), synthetic_check=True)
    assert all(f["fitting_improved"] for f in result["folds"])
    assert {f["parameter_count"] for f in result["folds"] if f["arm"] == "mi_learned"} == {224}
    assert {f["parameter_count"] for f in result["folds"] if f["arm"] in ("kt_r", "kt_r_mean", "kt_r_hard")} == {2058}
    roles = {c["role"]: c for c in result["paired_contrasts"]}
    assert roles["mechanism_mean_collapse"]["n_people"] == 2 and roles["mechanism_hard_collapse"]["n_people"] == 2
    assert roles["mechanism_mean_collapse"]["interval_95_among_included_people"]["mean"] is None  # 2 people: no interval, with a reason
    assert "burden_difference_onsets_per_hand_minute" in roles["mechanism_mean_collapse"]
    assert result["schema"] == "sg-offline-comparison/3" and result["configuration"]["early_limit_s"] == 0.


# ---- participant-level interval (review EVA-2 / B12) ----------------------------------------------

def test_paired_interval_appears_only_from_five_people():
    few = paired_interval([.1, .2])
    assert few["mean"] is None and few["n_people"] == 2 and "fewer than 5" in few["reason"]
    six = paired_interval([.1, .2, .3, .4, .5, .6])
    sd = np.std([.1, .2, .3, .4, .5, .6], ddof=1)
    assert six["mean"] == pytest.approx(.35) and six["sd_of_differences"] == pytest.approx(sd)
    assert six["lower"] == pytest.approx(.35 - 2.571 * sd / np.sqrt(6), abs=1e-9) and six["upper"] == pytest.approx(.35 + 2.571 * sd / np.sqrt(6), abs=1e-9)
    assert t_975(1) == 12.706 and t_975(35) == 2.042 and t_975(200) == 1.960
    with pytest.raises(ValueError):
        t_975(0)


# ---- least-squares distance rate (review B10 / ARC-3) --------------------------------------------

def _pk(t, hand_x, obj_x=0., noise=0.):
    s = lambda name: Stamp(f"{name}@{t}", t, t, Source.RECORDED, "cam")
    pt = lambda name, x: Point(name, (x, 0., 1.), s(name), frame="cam_optical", calibration_id="c", definition="fp")
    return Packet(f"pk@{t}", t, (pt("left", -.4), pt("right", hand_x + noise)), (pt("cup", obj_x),), clock="cam", session_id="s", participant_id="p")


def test_least_squares_rate_window_is_exact_on_linear_motion_and_damps_one_frame_noise():
    dt = 1 / 15; truth = -.3  # m/s, the right hand approaching the cup
    noisy = [_pk(k * dt, .6 + truth * k * dt, noise=(.002 if k % 2 else -.002)) for k in range(12)]
    clean = [_pk(k * dt, .6 + truth * k * dt) for k in range(12)]
    errors = {}
    for window in (1, 4):
        rel = RelativeGeometry(window=window)
        rates = [rel.update(p)[("right", "cup")].rate_mps for p in noisy]
        errors[window] = np.mean([abs(r - truth) for r in rates[5:] if r is not None])
        tracker = RelativeGeometry(window=window)
        exact = [tracker.update(p)[("right", "cup")] for p in clean]
        assert exact[0].rate_mps is None and all(abs(e.rate_mps - truth) < 1e-9 for e in exact[1:])
        assert "least_squares" in exact[-1].reason if window > 1 else exact[-1].reason == "past_secant_distance_rate_frame_invariant"
    assert errors[1] > .04 and errors[4] < errors[1] / 2  # one-frame secant: +-0.004 m / 0.067 s = +-0.06 m/s of noise
    tracker = RelativeGeometry()
    secant = [tracker.update(p)[("right", "cup")] for p in noisy]
    assert secant[2].rate_mps == pytest.approx((secant[2].distance_m - secant[1].distance_m) / dt)  # window 1 is the old secant
    with pytest.raises(ValueError):
        RelativeGeometry(window=0)
    with pytest.raises(ValueError):
        Config(rate_window=0)


def test_the_rate_window_is_a_replay_setting_recorded_in_the_trace_and_reset_by_gaps():
    dt = 1 / 15
    packets = [_pk(k * dt, .6 - .3 * k * dt) for k in range(6)] + [_pk(2. + k * dt, .3 - .3 * k * dt) for k in range(3)]
    rows = Replay(Config(rate_window=3)).run(packets)
    assert rows[0]["configuration"]["rate_window"] == 3 and rows[0]["implementation_version"] == "motion-only-build/7"
    reasons = [r["relative_geometry"]["right|cup"]["reason"] for r in rows]
    assert reasons[1] == "past_secant_distance_rate_frame_invariant" and reasons[3].startswith("past_least_squares_distance_rate_frame_invariant_4")
    assert reasons[6] == "gap_exceeded" and reasons[7] == "past_secant_distance_rate_frame_invariant"  # the chain restarts after the 2 s gap
    default = Replay().run(packets)
    assert all(r["relative_geometry"]["right|cup"]["reason"] != reasons[3] for r in default[3:6])


# ---- run provenance, timing policy, frame times (review A1 / A6 / A8 / C6) ------------------------

def test_queue_policies_frame_times_and_provenance_helpers():
    import run_perception as rp
    assert rp.schedule(10.0, 9.0, .04) == pytest.approx(10.04)                       # serial, idle worker
    assert rp.schedule(10.0, 10.5, .04) == pytest.approx(10.54)                       # serial, queued behind the previous frame
    assert rp.schedule(10.0, 10.5, .04, "drop_while_busy") is None                    # latest-frame policy skips it
    assert rp.schedule(10.6, 10.5, .04, "drop_while_busy") == pytest.approx(10.64)
    assert rp.schedule(10.0, 99.0, .04, "fixed_latency", fixed_latency_s=.1) == pytest.approx(10.1)  # deterministic: ignores the worker
    with pytest.raises(ValueError):
        rp.schedule(0., 0., 0., "fifo")
    with pytest.raises(ValueError):
        rp.schedule(0., 0., 0., "fixed_latency", fixed_latency_s=-1.)
    from sg_offline.rgbd import CameraCalibration
    cal = CameraCalibration("c", 64, 48, 60., 60., 32., 24., .001, "cam")
    provider, record = rp.build_object_provider("plane", cal)
    assert record["kind"] == "plane" and record["plane_settings"]["min_height_m"] == .015 and provider.name == "plane_clusters"
    with pytest.raises(ValueError, match="masks"):
        rp.build_object_provider("masks", cal)
    records = [dict(frame_id=f"s/f-{k}", rgb_stamp=dict(acquired_s=100. + k / 15, available_s=100.06 + k / 15, clock="c"),
                    depth_stamp=dict(acquired_s=100. + k / 15, available_s=100.05 + k / 15, clock="c")) for k in range(3)]
    rows = rp.frame_time_rows(records)
    assert [r["index"] for r in rows] == [0, 1, 2]
    assert rows[0]["session_relative_acquired_s"] == pytest.approx(-.06) and rows[0]["session_relative_available_s"] == 0.
    assert rows[2]["session_relative_available_s"] == pytest.approx(2 / 15)
    hashes = rp.implementation_hashes()
    assert "sg_offline/inference.py" in hashes and "run_perception.py" in hashes and all(len(h) == 64 for h in hashes.values())


def test_run_registry_rows_follow_the_registry_header(tmp_path):
    import run_perception as rp
    registry = tmp_path / "manifest.csv"
    assert rp.append_manifest_row(registry, dict(run_id="x")) is False  # absent registry: nothing written
    registry.write_text("run_id,date_utc,command,notes\n")
    assert rp.append_manifest_row(registry, dict(run_id="r1", command="cmd", unknown="ignored", notes="n")) is True
    rows = list(csv.DictReader(registry.open()))
    assert rows == [dict(run_id="r1", date_utc="", command="cmd", notes="n")]


def test_label_tool_shows_session_relative_time_and_exports_frame_times(tmp_path):
    from test_group_a_fixes import export_sessions
    import label_frames
    from sg_offline.recordings import RGBDRecording
    entries = export_sessions(tmp_path)
    session = label_frames.Session(RGBDRecording(entries[0]["manifest"]), tmp_path / "l.jsonl", "a", "unmarked")
    assert session.time_of(1) == pytest.approx(1.0) and session.origin_s == 0.
    out = tmp_path / "times.csv"
    rc = label_frames.main([entries[0]["manifest"], str(tmp_path / "m.jsonl"), "--annotator", "a", "--pass", "unmarked",
                            "--glove-condition", "x", "--frame-times", str(out)])
    rows = list(csv.DictReader(out.open()))
    assert rc == 0 and len(rows) == 2 and float(rows[1]["session_relative_acquired_s"]) == pytest.approx(1.0)


def test_adapter_overwrite_clears_only_its_own_previous_files(tmp_path):
    adapter = pytest.importorskip("d435_recording_adapter")
    for name in ("rgb_0000.npy", "depth_0000.npy", "recording.json", "adapter_provenance.json", "notes.txt"):
        (tmp_path / name).write_text("x")
    removed = adapter._clear_previous_extraction(tmp_path)
    assert sorted(removed) == ["adapter_provenance.json", "depth_0000.npy", "recording.json", "rgb_0000.npy"]
    assert [p.name for p in tmp_path.iterdir()] == ["notes.txt"]


def test_timed_recording_loads_the_proposed_script_and_keeps_the_accepted_default(capsys):
    import timed_recording
    here = Path(timed_recording.__file__).parent
    v2 = timed_recording.load_script(here / "recording_script_v2_proposed.json")
    text = " ".join(b["instruction"] for b in v2["blocks"]).lower()
    assert v2["duration_s"] == 300 and "release" in text and "both hands" in text and "10 cm" in text
    assert "proposed" in v2["status"] and "awaiting_decision" in v2["status"]
    timed_recording.main(["--script", str(here / "recording_script_v2_proposed.json")])
    assert "recording_script_v2_proposed.json" in capsys.readouterr().out
    default = timed_recording.load_script(here / "recording_script.json")
    assert default["duration_s"] == 270  # the accepted script is unchanged


# ---- architecture and code review, 5 October 2026 ----------

def _recorded(packet, person):
    """A fixture packet relabelled as a recorded observation of ``person`` (the recorded trainer refuses synthetic ones)."""
    def mark(p):
        return replace(p, stamp=replace(p.stamp, source=Source.RECORDED))
    return replace(packet, participant_id=person, hands=tuple(mark(h) for h in packet.hands),
                   objects=tuple(mark(o) for o in packet.objects), telemetry=tuple(mark(t) for t in packet.telemetry))


def _comparison_inputs(tmp_path, people=("p1", "p2", "p3")):
    from dataclasses import asdict
    from import_requests import convert
    entries = []
    for person in people:
        (tmp_path / f"{person}.packets.jsonl").write_text("".join(json.dumps(asdict(_recorded(p, person))) + "\n" for p in scenarios()))
        timeline = [dict(start_s="0", end_s="0.9", left="G", right="NQ"), dict(start_s="0.9", end_s="2", left="NQ", right="NQ")]
        (tmp_path / f"{person}.ledger.json").write_text(json.dumps(convert(timeline, person, "synthetic-session", "a", "session", 2.)))
        entries.append(dict(packets=f"{person}.packets.jsonl", ledger=f"{person}.ledger.json", origin_s=0.,
                            perception_provenance=dict(fit="fixed_external", model_id="fixture", checkpoint_sha256="0" * 64)))
    return entries


def test_the_comparison_cli_accepts_the_documented_replay_config(tmp_path):
    """The `rate_window` setting goes in the comparison's replay `config`; the CLI passed that JSON object on as a
    dict where ``compare`` needs a ``Config`` and stopped with an AttributeError, so no arm could be run with a
    chosen rate window through the CLI (ledger R39)."""
    import run_comparison
    entries = _comparison_inputs(tmp_path)
    base = dict(arms=["kt_set", "kt_r"], seeds=[0], steps=2, thresholds=[0.6], selection="fixed")
    (tmp_path / "config.json").write_text(json.dumps(dict(sessions=entries, config=dict(rate_window=2), **base)))
    run_comparison.main([str(tmp_path / "config.json"), str(tmp_path / "report.json")])
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["schema"] == "sg-offline-comparison/3" and report["configuration"]["rate_window"] == 2
    assert report["configuration"]["replay_configuration"]["rate_window"] == 2 and report["configuration"]["replay_configuration"]["max_age_s"] == .25
    assert report["configuration"]["fixed_threshold"] == .6 and report["sources"][0]["origin_s"] == 0.
    # an unknown replay setting is refused, not ignored
    (tmp_path / "bad.json").write_text(json.dumps(dict(sessions=entries, config=dict(rate_windw=2), **base)))
    with pytest.raises(TypeError):
        run_comparison.main([str(tmp_path / "bad.json"), str(tmp_path / "bad_report.json")])
    # without the key the default replay configuration is unchanged
    (tmp_path / "plain.json").write_text(json.dumps(dict(sessions=entries, **base)))
    run_comparison.main([str(tmp_path / "plain.json"), str(tmp_path / "plain_report.json")])
    assert json.loads((tmp_path / "plain_report.json").read_text())["configuration"]["rate_window"] == 1


def test_folds_mode_applies_the_declared_minimum_people_to_the_combined_contrasts(tmp_path):
    """In folds mode the per-fold reports used the declared ``min_people_for_interval`` while the combined contrasts,
    the only place where several people meet, silently used the default 5 (ledger R39)."""
    import run_comparison
    entries = _comparison_inputs(tmp_path)
    folds = [dict(held_out_person=p, sessions=entries) for p in ("p1", "p2", "p3")]
    base = dict(arms=["kt_set", "kt_r"], seeds=[0], steps=2, thresholds=[0.6], selection="fixed")
    for name, declared in (("low", 2), ("high", 4)):
        (tmp_path / f"{name}.json").write_text(json.dumps(dict(folds=folds, min_people_for_interval=declared, **base)))
        run_comparison.main([str(tmp_path / f"{name}.json"), str(tmp_path / f"{name}_report.json")])
        report = json.loads((tmp_path / f"{name}_report.json").read_text())
        assert {r["configuration"]["min_people_for_interval"] for r in report["reports"]} == {declared}
        top = [c for c in report["paired_contrasts"] if c["role"] == "matched_control"][0]
        assert top["n_people"] == 3
        interval = top["interval_95_among_included_people"]
        if declared <= 3:
            assert interval["mean"] is not None and interval["n_people"] == 3     # before: "fewer than 5 people"
        else:
            assert interval["mean"] is None and "fewer than 4 people" in interval["reason"]


def test_scoring_refuses_a_ledger_that_joins_no_decision_tick():
    """A ledger imported with another session id, or converted with a wrong origin, joined no decision tick; every
    request then counted as missed and every no-request interval as clean, i.e. a burden of 0.0 per hand-minute
    without any error. A session whose labels meet none of its ticks is now refused."""
    from sg_offline.experiment import score
    packets = scenarios()                       # decision ticks at 0.5 .. 1.3 s
    person, session = packets[0].participant_id, packets[0].session_id

    def ops(session_id, shift=0.):
        return [Opportunity("a", person, session_id, .5 + shift, .9 + shift, (("G",), ("NQ",)), "f"),
                Opportunity("b", person, session_id, .9 + shift, 1.4 + shift, (("NQ",), ("NQ",)), "f")]
    good = score(SimpleRules(), [dict(packets=packets, opportunities=ops(session))], Config())
    assert good["unwanted"][person]["unwanted_proposals"] == 2 and good["success"][person]["compatible_first"] == 1
    with pytest.raises(ValueError, match="identity"):
        score(SimpleRules(), [dict(packets=packets, opportunities=ops("another-session"))], Config())
    with pytest.raises(ValueError, match="no decision tick"):
        score(SimpleRules(), [dict(packets=packets, opportunities=ops(session, 1000.))], Config())   # before: burden 0.0
    # a session without any reference interval is still scored (nothing to join)
    assert score(SimpleRules(), [dict(packets=packets, opportunities=[])], Config())["success"] == {}


def test_no_request_time_without_any_decision_tick_is_reported_beside_the_burden_rate():
    """Exposure comes from the ledger alone, so labelled no-request time that the trace never covered (a truncated
    or late-started run) silently lowers the burden rate. The rate keeps its definition; the uncovered intervals
    and their hand-seconds are now reported next to it, like `missed_no_decision_ticks` on the request side."""
    rows = trace(ticks_from(0., 5.), right_proposals=[2.0])          # the trace stops at 4.9 s
    ops = [Opportunity("a", "p", "s", 0., 5., (("NQ",), ("NQ",)), "f"), Opportunity("b", "p", "s", 5., 10., (("NQ",), ("NQ",)), "f")]
    burden = unwanted_exposure(evaluate_opportunities(rows, ops), ops)["p"]
    assert burden["exposure_hand_s"] == 20. and burden["no_request_hand_intervals"] == 4 and burden["unwanted_proposals"] == 1
    assert burden["no_request_hand_intervals_without_decision_ticks"] == 2
    assert burden["exposure_hand_s_without_decision_ticks"] == 10.
    full = unwanted_exposure(evaluate_opportunities(trace(TIMES, right_proposals=[2.0]), ops), ops)["p"]
    assert full["no_request_hand_intervals_without_decision_ticks"] == 0 and full["exposure_hand_s_without_decision_ticks"] == 0.


def test_an_opportunity_built_from_lists_is_scored_like_one_built_from_tuples():
    """`acceptable == ("NQ",)` is compared as a tuple in the evaluator, the statistics and the ledger summary. An
    Opportunity rebuilt from JSON carried lists, so a no-request hand was scored as a missed request and its
    exposure disappeared. The record now normalises its collections."""
    rows = trace(TIMES, right_proposals=[2.0])
    from_json = Opportunity("rest", "p", "s", 0., 5., [["NQ"], ["NQ"]], "f", hand_windows_s=[[0., 5.], [0., 5.]])
    assert from_json.acceptable == (("NQ",), ("NQ",)) and from_json.hand_windows_s == ((0., 5.), (0., 5.))
    hands = evaluate_opportunities(rows, [from_json])[0]["hands"]
    assert [h["result"] for h in hands] == ["correct_no_new_request", "unwanted"]       # before: both missed_including_withheld
    assert unwanted_exposure(evaluate_opportunities(rows, [from_json]), [from_json])["p"]["exposure_hand_s"] == 10.
    request = Opportunity("req", "p", "s", 5., 8., [None, ["G"]], "f", request_ids=[None, "r1"], reference_reasons=["unknown", "known"])
    assert request.acceptable == (None, ("G",)) and request.request_ids == (None, "r1") and request.reference_reasons == ("unknown", "known")
    with pytest.raises(ValueError):
        Opportunity("bad", "p", "s", 0., 5., ("NQ", "NQ"), "f")       # a bare string is not a set of outcomes


def test_the_early_tolerance_is_the_same_for_a_hand_window_as_for_split_rows():
    """PROTOCOLS allows two ways to label the same situation: split the row at the hand's onset, or keep one row and
    give the hand a window. With a declared early tolerance the split form admitted a run that began 0.2 s early
    while the window form still failed it, because the tolerance was applied only to a run reaching back from the
    row's first tick. At the default (0, strict) both forms fail, as before."""
    rows = trace(TIMES, right_proposals=ticks_from(4.8, 8.))                       # the run begins 0.2 s before the onset at 5.0 s
    split = [Opportunity("rest", "p", "s", 0., 5., (("NQ",), ("NQ",)), "f"), Opportunity("req", "p", "s", 5., 8., (("NQ",), ("G",)), "f")]
    window = [Opportunity("req", "p", "s", 0., 8., (("NQ",), ("G",)), "f", hand_windows_s=((0., 8.), (5., 8.)))]
    for limit, expected in ((0., "wrong_timing_first"), (.5, "compatible_first")):
        by_rows = evaluate_opportunities(rows, split, early_limit_s=limit)[1]["hands"][1]
        by_window = evaluate_opportunities(rows, window, early_limit_s=limit)[0]["hands"][1]
        assert by_rows["result"] == by_window["result"] == expected
        assert by_rows["early_by_s"] == pytest.approx(.2) and by_window["early_by_s"] == pytest.approx(.2)
    # a first proposal far before the window still fails under the same tolerance, and is counted as early
    far = evaluate_opportunities(trace(TIMES, right_proposals=[1.0] + ticks_from(4.9, 8.)), window, early_limit_s=.5)
    assert far[0]["hands"][1]["result"] == "wrong_timing_first" and far[0]["hands"][1]["early_by_s"] == pytest.approx(4.)
    assert per_participant_success(far)["p"]["wrong_timing_first_early"] == 1
    # a first proposal after the window is late, as before
    late = Opportunity("req", "p", "s", 0., 8., (("NQ",), ("G",)), "f", hand_windows_s=((0., 8.), (5., 6.)))
    out = evaluate_opportunities(trace(TIMES, right_proposals=[7.0]), [late], early_limit_s=.5)
    assert out[0]["hands"][1]["result"] == "wrong_timing_first" and per_participant_success(out)["p"]["wrong_timing_first_late"] == 1


def test_the_interval_minimum_is_validated_before_any_fitting():
    """`min_people=1` reached a division by zero (n - 1) at the very end of a comparison, after all fitting."""
    from sg_offline.experiment import compare
    with pytest.raises(ValueError, match="at least 2"):
        paired_interval([.1], min_people=1)
    with pytest.raises(ValueError, match="min_people_for_interval"):
        compare(people_sessions(), arms=("rules",), min_people_for_interval=1, synthetic_check=True)


def test_the_importer_reads_a_timeline_saved_with_a_byte_order_mark(tmp_path):
    """A spreadsheet's "CSV UTF-8" export starts with a byte-order mark; the first header then read as
    '\\ufeffstart_s' and the importer answered "missing required columns: ['start_s']"."""
    import import_requests
    body = "start_s,end_s,left,right,note\n0.00,2.00,NQ,NQ,\n2.00,4.00,G,NQ,\n"
    (tmp_path / "bom.csv").write_bytes(b"\xef\xbb\xbf" + body.replace("\n", "\r\n").encode())
    import_requests.main([str(tmp_path / "bom.csv"), str(tmp_path / "bom.json"), "--participant", "p", "--session", "s",
                          "--annotator", "a", "--clock", "session", "--duration-s", "4"])
    ledger = json.loads((tmp_path / "bom.json").read_text())
    assert [(e["start_s"], e["end_s"]) for e in ledger["opportunities"]] == [(0., 2.), (2., 4.)]


def test_rows_that_meet_within_the_tolerance_are_written_exactly_contiguous():
    """The importer accepts a row that starts within 1e-6 s of the previous end, but the scorer compares exactly:
    rows 0-2.0000005 and 2.0-4 were imported and then refused as "overlapping no-request exposure"."""
    from import_requests import convert
    from sg_offline.ledger import _entry
    rows = [dict(start_s="0", end_s="2.0000005", left="NQ", right="NQ"), dict(start_s="2.0", end_s="4", left="NQ", right="G")]
    entries = convert(rows, "p", "s", "a", "session", 4.)["opportunities"]
    assert entries[1]["start_s"] == entries[0]["end_s"] == 2.0000005 and entries[1]["hand_windows_s"]["right"][0] == 2.0000005
    ops = [Opportunity(e.opportunity_id, "p", "s", e.start_s, e.end_s, e.acceptable, e.provenance, e.hand_windows_s) for e in map(_entry, entries)]
    burden = unwanted_exposure(evaluate_opportunities(trace(TIMES), ops), ops)["p"]       # before: ValueError (overlap)
    assert burden["exposure_hand_s"] == pytest.approx(6.)
    with pytest.raises(ValueError, match="partition"):                                    # a real gap or overlap is still refused
        convert([dict(start_s="0", end_s="2", left="NQ", right="NQ"), dict(start_s="2.01", end_s="4", left="NQ", right="NQ")], "p", "s", "a", "session", 4.)


def test_a_hand_window_is_refused_for_a_hand_without_a_request():
    """A window column on a no-request hand was accepted and produced a negative time to the first unwanted proposal."""
    from import_requests import convert
    row = dict(start_s="0", end_s="4", left="NQ", right="G", left_start_s="1", left_end_s="3")
    with pytest.raises(ValueError, match="hand window"):
        convert([row], "p", "s", "a", "session", 4.)
    ok = convert([dict(start_s="0", end_s="4", left="NQ", right="G", right_start_s="1", right_end_s="3")], "p", "s", "a", "session", 4.)
    assert ok["opportunities"][0]["hand_windows_s"] == dict(left=[0., 4.], right=[1., 3.])


def _box_labels(frame="frame", left=None, right=(50, 0, 70, 20)):
    from sg_offline.ledger import LandmarkLabel
    out = []
    for side, box in (("left", left), ("right", right)):
        out.append(LandmarkLabel(frame, "p", "s", side, "hand_box", "visible" if box else "out_of_frame", None, None, "a", "unmarked", box=box))
    return out


def _assigned(box, side):
    return dict(box=list(box), side=side, confidence=.9, points={}, model="m", definition="d")


def test_identity_errors_on_an_absent_side_are_counted_under_their_own_denominator():
    """`on_other_reference_hand` was incremented both for a visible reference side and for a side whose reference is
    out of frame, so neither group of counters summed to its denominator (1 visible side, 2 visible-side counts)."""
    from evaluate_perception import evaluate
    row = dict(participant_id="p", session_id="s", frame_id="frame", detections=[],
               assigned=dict(left=_assigned((50, 0, 70, 20), "left"), right=None))      # the right hand emitted as "left"; no right output
    counts = evaluate([row], _box_labels())["tracker"]["counts"]
    visible = sum(counts.get(k, 0) for k in ("correct_identity", "on_other_reference_hand", "not_on_any_reference_hand", "withheld_or_missed"))
    absent = sum(counts.get(k, 0) for k in ("correctly_empty_out_of_frame", "out_of_frame_side_emitted_on_other_reference_hand", "emitted_while_reference_out_of_frame"))
    assert counts["reference_sides"] == 1 and visible == 1 and counts["withheld_or_missed"] == 1
    assert counts["reference_out_of_frame_sides"] == 1 and absent == 1 and counts["out_of_frame_side_emitted_on_other_reference_hand"] == 1


def test_a_perception_evaluation_names_its_inputs_and_never_replaces_an_existing_report(tmp_path):
    """The step-3 perception report carried no input paths or hashes and silently overwrote an earlier report."""
    import hashlib
    import evaluate_perception
    from sg_offline.ledger import write_landmark_ledger
    row = dict(participant_id="p", session_id="s", frame_id="frame", detections=[], assigned=dict(left=None, right=_assigned((50, 0, 70, 20), "right")))
    (tmp_path / "detections.jsonl").write_text(json.dumps(row) + "\n")
    write_landmark_ledger(tmp_path / "labels.jsonl", _box_labels())
    evaluate_perception.main([str(tmp_path / "detections.jsonl"), str(tmp_path / "labels.jsonl"), str(tmp_path / "evaluation.json")])
    report = json.loads((tmp_path / "evaluation.json").read_text())
    assert report["tracker"]["counts"]["correct_identity"] == 1
    assert report["inputs"]["detections_sha256"] == hashlib.sha256((tmp_path / "detections.jsonl").read_bytes()).hexdigest()
    assert report["inputs"]["labels_sha256"] == hashlib.sha256((tmp_path / "labels.jsonl").read_bytes()).hexdigest()
    assert report["inputs"]["match_min_iou"] == evaluate_perception.MATCH_MIN_IOU and report["inputs"]["prediction_rows"] == 1
    with pytest.raises(SystemExit):
        evaluate_perception.main([str(tmp_path / "detections.jsonl"), str(tmp_path / "labels.jsonl"), str(tmp_path / "evaluation.json")])


def test_the_leak_check_needs_lists_of_people_and_sources_record_the_origin(tmp_path):
    """`set("p1")` is {"p", "1"}: with `training_people` given as a string the outer test person passed the leak
    check. The comparison report also did not record the `origin_s` each ledger was converted with."""
    from run_comparison import load_sessions
    entries = _comparison_inputs(tmp_path, people=("p1",))
    adapted = dict(fit="adapted", model_id="m", checkpoint_sha256="0" * 64, training_people="p1", development_people=[])
    with pytest.raises(ValueError, match="lists"):
        load_sessions([dict(entries[0], perception_provenance=adapted)], tmp_path, held="p1")     # before: accepted
    with pytest.raises(ValueError, match="leaked"):
        load_sessions([dict(entries[0], perception_provenance=dict(adapted, training_people=["p1"]))], tmp_path, held="p1")
    _, sources = load_sessions([dict(entries[0], perception_provenance=dict(adapted, training_people=["p2"]))], tmp_path, held="p1")
    assert sources[0]["origin_s"] == 0.


def test_a_fixed_threshold_contrast_names_arms_that_never_proposed():
    """The silent-point rule of 2 October lives in development selection only. With a fixed threshold (the default) an
    arm that never proposes on the test person still entered every direct contrast, unflagged: "KT-R minus flat =
    +1.0" for a flat arm that issued nothing. Nothing is excluded on test outcomes; the contrast now says so."""
    from sg_offline.experiment import contrasts_from_folds

    def fold(arm, person, proposals, success):
        hands = [dict(result="compatible_first" if proposals else "missed_including_withheld", proposal_count=proposals)]
        return dict(arm=arm, held_out_person=person, seed=0, threshold_selection="fixed", reporting_threshold=.6,
                    development_budget_feasible=True, development_curve=[], development_selection_reason="fixed reporting threshold; no development selection",
                    reporting_metrics=dict(success={person: dict(success_fraction=success)},
                                           unwanted={person: dict(proposals_per_hand_minute=0., onsets_per_hand_minute=0.)},
                                           outcomes=[dict(participant_id=person, hands=hands)]))
    folds = [f for person in ("p1", "p2") for f in (fold("kt_r", person, 3, 1.), fold("flat", person, 0, 0.), fold("kt_set", person, 2, 1.))]
    by_role = {c["role"]: c for c in contrasts_from_folds(folds, (0,), "primary_contrast")}
    info = by_role["matched_information"]                                     # KT-R minus flat
    assert info["per_person"] == dict(p1=1., p2=1.) and info["status"] == "descriptive_fixed_threshold"
    assert info["silent_arms_on_test"] == dict(p1=["flat"], p2=["flat"])
    assert info["reporting_points"]["p1"]["flat"]["proposes_on_test"] is False and info["reporting_points"]["p1"]["kt_r"]["proposes_on_test"] is True
    assert by_role["matched_control"]["silent_arms_on_test"] == {}            # KT-R minus KT-SET: both proposed
    legacy = [dict(f, reporting_metrics={k: v for k, v in f["reporting_metrics"].items() if k != "outcomes"}) for f in folds]
    unknown = {c["role"]: c for c in contrasts_from_folds(legacy, (0,), "primary_contrast")}["matched_information"]
    assert unknown["reporting_points"]["p1"]["flat"]["proposes_on_test"] is None and unknown["silent_arms_on_test"] == {}


def test_driver_summaries_carry_the_same_build_identifier_as_their_traces(tmp_path, capsys, monkeypatch):
    """`run_motion_checks.py` and `run_rgbd_checks.py` stamped "motion-only-build/2" into summaries whose own trace
    rows said "motion-only-build/5" (the stored summary of 2 October has this mismatch)."""
    import run_motion_checks, run_rgbd_checks
    from sg_offline import replay
    for driver, summary_name, trace_name in ((run_motion_checks, "motion_only_summary.json", "synthetic_motion_trace.jsonl"),
                                             (run_rgbd_checks, "geometry_summary.json", "fixed_with_known_pose.jsonl")):
        out = tmp_path / driver.__name__
        monkeypatch.setattr("sys.argv", [driver.__name__ + ".py", "--output-dir", str(out)])   # never the repository's results folder
        driver.main()
        summary = json.loads((out / summary_name).read_text())
        row = json.loads((out / trace_name).read_text().splitlines()[0])
        assert summary["implementation"] == row["implementation_version"] == "motion-only-build/7"
    assert replay.IMPLEMENTATION_VERSION == "motion-only-build/7"          # the single source the drivers import
    capsys.readouterr()


def _scene(t, left, right, objects, fixed=False):
    """One recorded-source packet in a named camera frame (moving unless ``fixed``)."""
    def pt(identity, xyz):
        return Point(identity, None if xyz is None else tuple(float(v) for v in xyz), Stamp(f"{identity}@{t!r}", t, t, Source.RECORDED, "cam"),
                     frame="cam_optical", frame_is_fixed=fixed, calibration_id="c", definition="fp", reason="" if xyz is not None else "lost")
    return Packet(f"pk@{t!r}", t, (pt("left", left), pt("right", right)), tuple(pt(k, v) for k, v in objects),
                  clock="cam", session_id="s", participant_id="p")


def test_the_rule_target_at_an_exact_distance_tie_does_not_depend_on_hypothesis_order():
    """`hand_target_features` picks the object with the largest summed hypothesis weight and breaks ties by name. At an
    exact distance tie the two sums differed in the last bit, so the name was never consulted and the target, with it
    the rule's score and the supervisor's grasp gate, followed the order of summation."""
    from sg_offline.inference import hand_target_features, hypotheses
    from sg_offline.state import CausalKinematics, DisabledKinetics
    objects = (("cup", (0., 0., 1.)), ("jar", (.2, 0., 1.)))
    rng = np.random.default_rng(0); changed = scenes = 0
    for _ in range(60):
        right_x, speed = float(rng.uniform(-.6, .9)), float(rng.uniform(.05, .3))
        kin, rel = CausalKinematics(), RelativeGeometry()
        for p in (_scene(0.0, (.1 + speed * .1, 0., 1.), (right_x, 0., 1.), objects), _scene(0.1, (.1, 0., 1.), (right_x, 0., 1.), objects)):
            motions, relatives = kin.update(p), rel.update(p)
        b = hypotheses(p, motions, DisabledKinetics().estimate(p), False, .25, "weak_speed", relatives)
        base = hand_target_features(b, 0); scenes += 1
        assert base[1] < 0                       # the name decides: "cup", which the left hand is approaching
        for _ in range(5):
            order = rng.permutation(len(b.weights))
            shuffled = Bundle(tuple(b.hypotheses[i] for i in order), b.features[order], b.weights[order], b.source_kinds, b.feature_mode)
            changed += not np.array_equal(hand_target_features(shuffled, 0), base)
    assert scenes == 60 and changed == 0         # before: the target flipped in a quarter of the shuffles


def test_the_pair_gate_does_not_withhold_a_hand_for_a_partner_that_cannot_act():
    """The experimental pair gate (off by default) looked at the raw choice of BOTH hands before any availability
    check: with the left hand out of view and a model whose raw choice for it was G, the visible right hand's
    grasp was withheld as an "ambiguous pair"."""
    from sg_offline.fixtures import make_packet

    class BothGrasp:
        name = "fixture_both_grasp"

        def predict(self, bundle):
            p = np.full((7, 7), 1e-6); p[1, 1] = 1.
            return (p / p.sum()).ravel()
    packets = [make_packet(0, None, (.5, 0., 1.)), make_packet(1, None, (.4, 0., 1.))]
    for gate in (False, True):
        left, right = Replay(Config(task_gate=gate), model=BothGrasp()).run(packets)[1]["outputs"]
        assert (left["status"], left["reason"]) == ("withheld", "required_hand_observation_unavailable")
        assert right["status"] == "proposed_simulation_only"            # before, with the gate on: withheld (pair gate)
    # the gate still withholds a real ambiguous pair: both hands visible, both G, on different objects
    both = [make_packet(0, (-.5, 0., 1.), (.62, 0., 1.), objects=(("container", (0., 0., 1.)), ("part", (.12, 0., 1.)))),
            make_packet(1, (-.4, 0., 1.), (.52, 0., 1.), objects=(("container", (0., 0., 1.)), ("part", (.12, 0., 1.))))]
    outputs = Replay(Config(task_gate=True), model=BothGrasp()).run(both)[1]["outputs"]
    assert [o["reason"] for o in outputs] == ["experimental_pair_gate_ambiguous_association"] * 2


def test_conditional_action_scores_are_absent_when_no_request_is_certain():
    """With a no-request marginal of 1 the trace's `conditional_action_scores` divided by a 1e-300 floor: all zeros, or
    values of 1e283 for a marginal within rounding of 1. A conditional distribution does not exist there."""
    class SureNoRequest:
        name = "fixture_sure_no_request"

        def __init__(self, leak):
            self.leak = leak

        def predict(self, bundle):
            p = np.full((7, 7), self.leak); p[0, 0] = 0.; p[0, 0] = 1. - p.sum()
            return p.ravel()
    for leak in (0., 1e-18):
        out = Replay(Config(), model=SureNoRequest(leak)).run(scenarios()[:1])[0]["outputs"][0]
        assert out["raw_outcome"] == "NQ" and out["conditional_action_scores"] is None
    out = Replay(Config(), model=SureNoRequest(1e-3)).run(scenarios()[:1])[0]["outputs"][0]
    assert sum(out["conditional_action_scores"]) == pytest.approx(1.)


def test_the_rate_window_also_reaches_the_feature_for_fixed_frame_points():
    """With points in a fixed frame the approach-rate FEATURE was always the one-frame velocity projection, whatever
    `rate_window` said, while the trace recorded the window and logged the least-squares rate beside it. A window
    study on fixed-frame data (the fixtures, a posed camera) therefore showed no effect. Window 1 is bit-identical."""
    dt, truth = 1 / 15, -.03

    def clip(fixed):
        return [_scene(round(k * dt, 9), (-.4, 0., 1.), (.17 + truth * k * dt + (.002 if k % 2 else -.002), 0., 1.), (("cup", (0., 0., 1.)),), fixed)
                for k in range(16)]

    def feature(packets, window):
        return np.array([[row[15 + 1] * .2 for row, h in zip(b.features, b.hypotheses) if h.object_ids[1] == "cup"][0]
                         for _, b in bundles_for(packets, Config(rate_window=window))])
    for fixed in (False, True):
        error = {k: float(np.mean(np.abs(feature(clip(fixed), k)[5:] - truth))) for k in (1, 4)}
        assert error[1] > .04 and error[4] < .5 * error[1]              # before, fixed frame: both 0.06
    logged = [r["relative_geometry"]["right|cup"]["rate_mps"] for r in Replay(Config(rate_window=4)).run(clip(True))]
    assert feature(clip(True), 4)[6:] == pytest.approx(logged[6:])      # the feature is the logged least-squares rate


# ---- pins added after the test audit of 5 October 2026 ---------------------------------------------------------------
# The audit changed one line at a time in a scratch copy (101 mutants) and found that 64 went unnoticed. The tests below
# state the VALUES and the neighbouring layouts of the numbers that step 3 reports; they pass on the pre-review code too.

def test_GUARD_the_early_rule_in_the_layouts_the_first_tests_left_open():
    # (i) a request that crosses rows (shared request id): the walk-back still sees the session before the first row
    rows = trace(TIMES, right_proposals=ticks_from(2.0, 8.))
    first = Opportunity("req-a", "p", "s", 5., 6., (("NQ",), ("G",)), "f", request_ids=(None, "r1"))
    second = Opportunity("req-b", "p", "s", 6., 8., (("G",), ("G",)), "f", request_ids=(None, "r1"))
    out = evaluate_opportunities(rows, [REST, first, second])
    assert out[1]["hands"][1]["result"] == "wrong_timing_first" and out[1]["hands"][1]["early_by_s"] == pytest.approx(3.)
    assert out[2]["hands"][1]["result"] == "continuation_of_request"
    # (ii) a burst that ended before the window and a new proposal inside it: on time, not active at the window start
    hand = evaluate_opportunities(trace(TIMES, right_proposals=ticks_from(4.0, 5.0) + ticks_from(5.3, 8.)), [REST, REQUEST])[1]["hands"][1]
    assert hand["result"] == "compatible_first" and hand["early_by_s"] == 0. and hand["run_active_at_window_start"] is False
    assert hand["first_proposal"][0] == pytest.approx(5.3) and hand["proposals_since_rearm_before_window"] == 10
    # (iii) a positive tolerance does not switch the rule off, and only the adjacent run is measured
    assert evaluate_opportunities(rows, [REST, REQUEST], early_limit_s=.5)[1]["hands"][1]["result"] == "wrong_timing_first"
    bursts = trace(TIMES, right_proposals=ticks_from(2.0, 2.6) + ticks_from(4.8, 8.))
    hand = evaluate_opportunities(bursts, [REST, REQUEST], early_limit_s=.5)[1]["hands"][1]
    assert hand["result"] == "compatible_first" and hand["early_by_s"] == pytest.approx(.2)
    # (iv) early-by is measured from the window start, not from the first tick inside the window
    late_start = Opportunity("req", "p", "s", 5.03, 8., (("NQ",), ("G",)), "f")
    rest = Opportunity("rest", "p", "s", 0., 5.03, (("NQ",), ("NQ",)), "f")
    hand = evaluate_opportunities(trace(TIMES, right_proposals=ticks_from(4.8, 8.)), [rest, late_start], early_limit_s=.25)[1]["hands"][1]
    assert hand["early_by_s"] == pytest.approx(.23) and hand["result"] == "compatible_first"


def test_GUARD_burden_values_first_unwanted_times_and_timeline_categories():
    # 14 unwanted ticks of the right hand in a 5 s rest row of two hands = 10 hand-seconds
    rest = unwanted_exposure(evaluate_opportunities(trace(TIMES, right_proposals=ticks_from(1.0, 2.4)), [REST]), [REST])["p"]
    assert rest["unwanted_proposals"] == 14 and rest["exposure_hand_s"] == 10.
    assert rest["proposals_per_hand_minute"] == pytest.approx(14 * 60 / 10.) and rest["onsets_per_hand_minute"] == pytest.approx(1 * 60 / 10.)
    # the time to the first unwanted proposal is counted from the interval's own start
    later = Opportunity("later", "p", "s", 3., 8., (("NQ",), ("NQ",)), "f")
    burden = unwanted_exposure(evaluate_opportunities(trace(TIMES, right_proposals=[4.0]), [later]), [later])["p"]
    assert burden["time_to_first_unwanted_s"] == [None, pytest.approx(1.)] and burden["intervals_with_unwanted"] == 1
    # every category of the timeline accounting, ticks and proposals
    from sg_offline.evaluation import timeline_accounting
    ops = [Opportunity("rest", "p", "s", 0., 2., (("NQ",), ("NQ",)), "f"),
           Opportunity("req", "p", "s", 2., 6., (("NQ",), ("G",)), "f", hand_windows_s=((2., 6.), (3., 5.))),
           Opportunity("unk", "p", "s", 6., 8., (None, ("NQ",)), "f")]
    counts = timeline_accounting(trace(TIMES, right_proposals=[1.0, 2.5, 4.0, 5.5, 7.0, 9.0]), ops)
    assert counts == dict(all_hand_ticks=200, no_request_hand_ticks=20 + 40 + 20 + 20, no_request_proposals=1 + 1,
                          request_hand_ticks=20, request_proposals=1, outside_request_window_hand_ticks=20, outside_request_window_proposals=2,
                          unknown_or_excluded_hand_ticks=20, unlabelled_hand_ticks=40, unlabelled_proposals=1)


def test_GUARD_the_perception_evaluators_thresholds_and_incomplete_frames():
    from evaluate_perception import evaluate
    labels = _box_labels(left=(0, 0, 20, 20), right=(50, 0, 70, 20))

    def tracker_counts(left_box):
        row = dict(participant_id="p", session_id="s", frame_id="frame", detections=[], assigned=dict(left=_assigned(left_box, "left"), right=None))
        return evaluate([row], labels)["tracker"]["counts"]
    assert tracker_counts((0, 0, 20, 20))["correct_identity"] == 1
    assert tracker_counts((6, 0, 26, 20))["correct_identity"] == 1                    # IoU 0.54 with the left reference box
    assert tracker_counts((14, 0, 34, 20))["not_on_any_reference_hand"] == 1          # IoU 0.18: below the 0.3 match threshold
    assert tracker_counts((34, 0, 54, 20))["not_on_any_reference_hand"] == 1          # IoU 0.11 with the right box, 0 with the left
    assert tracker_counts((44, 0, 64, 20))["on_other_reference_hand"] == 1            # on the right hand under the left name
    # touching hands: a "left" output that overlaps both reference boxes belongs to the one it overlaps more
    touching = _box_labels(left=(0, 0, 20, 20), right=(10, 0, 30, 20))
    row = dict(participant_id="p", session_id="s", frame_id="frame", detections=[], assigned=dict(left=_assigned((9, 0, 29, 20), "left"), right=None))
    assert evaluate([row], touching)["tracker"]["counts"]["on_other_reference_hand"] == 1     # IoU 0.38 with left, 0.90 with right
    # a hand box marked occluded removes the frame from every count and is reported
    from sg_offline.ledger import LandmarkLabel
    occluded = [LandmarkLabel("frame", "p", "s", "left", "hand_box", "occluded", None, None, "a", "unmarked"), labels[1]]
    row = dict(participant_id="p", session_id="s", frame_id="frame", detections=[], assigned=dict(left=None, right=_assigned((50, 0, 70, 20), "right")))
    result = evaluate([row], occluded)
    assert result["counts"] == dict(incomplete_box_reference_frames=1, reference_frames_without_predictions=0) and "counts" not in result["tracker"]
    # a labelled frame the run never processed is counted, never scored
    extra = _box_labels(frame="frame-not-processed")
    assert evaluate([row], labels + extra)["counts"]["reference_frames_without_predictions"] == 1


def test_GUARD_importer_guarantees_named_in_the_protocol():
    from import_requests import convert

    def rows(*triples):
        return [dict(start_s=str(a), end_s=str(b), left=l, right=r, note=n) for a, b, l, r, n in triples]
    with pytest.raises(ValueError, match="partition"):                                # an overlap
        convert(rows((0, 1, "NQ", "NQ", ""), (.5, 2, "NQ", "NQ", "")), "p", "s", "a", "session", 2.)
    with pytest.raises(ValueError, match="reason in note"):                           # an exclusion without a note
        convert(rows((0, 2, "EXCLUDED", "NQ", "")), "p", "s", "a", "session", 2.)
    with pytest.raises(ValueError, match="declared recording duration"):
        convert(rows((0, 2, "NQ", "NQ", "")), "p", "s", "a", "session", 3.)
    # back-to-back requests with different labels, and a request after a window that ended inside its row, are two requests
    two = convert(rows((0, 1, "NQ", "G", ""), (1, 2, "NQ", "R", "")), "p", "s", "a", "session", 2.)["opportunities"]
    assert [e["request_ids"]["right"] for e in two] == ["auto:right:0001", "auto:right:0002"]
    early_end = [dict(start_s="0", end_s="1", left="NQ", right="G", right_end_s="0.6"), dict(start_s="1", end_s="2", left="NQ", right="G")]
    assert [e["request_ids"]["right"] for e in convert(early_end, "p", "s", "a", "session", 2.)["opportunities"]] == ["auto:right:0001", "auto:right:0002"]
    with pytest.raises(ValueError, match="must keep its label"):                      # one explicit id cannot change its label
        convert([dict(start_s="0", end_s="1", left="NQ", right="G", right_request_id="x"), dict(start_s="1", end_s="2", left="NQ", right="R", right_request_id="x")],
                "p", "s", "a", "session", 2.)
    with pytest.raises(ValueError, match="changes its acceptable actions"):           # the evaluator's own check
        evaluate_opportunities(trace(TIMES), [Opportunity("a", "p", "s", 0., 1., (("NQ",), ("G",)), "f", request_ids=(None, "x")),
                                              Opportunity("b", "p", "s", 1., 2., (("NQ",), ("FP",)), "f", request_ids=(None, "x"))])


def test_GUARD_a_control_that_misses_the_budget_excludes_the_person_and_the_early_limit_reaches_the_score():
    from sg_offline.experiment import contrasts_from_folds, score

    def fold(arm, person, success, feasible, reason="development-selected"):
        return dict(arm=arm, held_out_person=person, seed=0, threshold_selection="development", reporting_threshold=.6,
                    development_budget_feasible=feasible, development_curve=[dict(threshold=.6, success=success)], development_selection_reason=reason,
                    reporting_metrics=dict(success={person: dict(success_fraction=success)}, unwanted={person: dict(proposals_per_hand_minute=0., onsets_per_hand_minute=0.)}))
    silent = "silent: only operating points that never proposed on development data met the budget"
    folds = [fold("kt_r", "p1", 1., True), fold("kt_set", "p1", 0., False, silent), fold("kt_r", "p2", 1., True), fold("kt_set", "p2", 1., True)]
    matched = {c["role"]: c for c in contrasts_from_folds(folds, (0,), "primary_contrast")}["matched_control"]
    assert matched["per_person"] == dict(p2=0.) and matched["excluded_people"]["p1"].startswith("kt_set reporting point was silent")
    assert matched["n_people"] == 1 and matched["status"] == "development_selected_some_people_excluded"
    # the declared early tolerance is the one the scoring uses
    packets = scenarios(); person, session = packets[0].participant_id, packets[0].session_id
    proposing = [r["decision_s"] for r in Replay().run(packets) if r["outputs"][0]["status"] == "proposed_simulation_only"]
    onset = proposing[0] + .05                                                        # the annotated onset lies 0.05 s after the run began
    ops = [Opportunity("rest", person, session, .4, onset, (("NQ",), ("NQ",)), "f"), Opportunity("req", person, session, onset, 1.4, (("G",), ("NQ",)), "f")]
    strict = score(SimpleRules(), [dict(packets=packets, opportunities=ops)], Config())["success"][person]
    tolerant = score(SimpleRules(), [dict(packets=packets, opportunities=ops)], Config(), early_limit_s=.2)["success"][person]
    assert strict["wrong_timing_first_early"] == 1 and tolerant["compatible_first"] == 1


def test_GUARD_training_features_use_the_same_rate_window_as_the_replay():
    packets = [_scene(round(k / 15, 9), (-.4, 0., 1.), (.3 - .02 * k + (.004 if k % 2 else 0.), 0., 1.), (("cup", (0., 0., 1.)),)) for k in range(8)]
    config = Config(rate_window=3)
    logged = [r["relative_geometry"]["right|cup"]["rate_mps"] for r in Replay(config).run(packets)]
    feature = [[row[15 + 1] * .2 for row, h in zip(b.features, b.hypotheses) if h.object_ids[1] == "cup"][0] for _, b in bundles_for(packets, config)]
    assert feature[3:] == pytest.approx(logged[3:]) and logged[5] != pytest.approx(Replay(Config()).run(packets)[5]["relative_geometry"]["right|cup"]["rate_mps"])


def test_GUARD_the_hard_collapse_twin_averages_a_real_tie_and_ignores_hypothesis_order():
    from sg_offline.inference import hypotheses
    from sg_offline.state import CausalKinematics, DisabledKinetics
    p = _scene(0., (-.02, 0., 1.), (.02, 0., 1.), (("cup", (0., 0., 1.)),))          # both hands 2 cm from one object, no velocities
    b = hypotheses(p, CausalKinematics().update(p), DisabledKinetics().estimate(p), False, .25, "weak_speed", RelativeGeometry().update(p))
    top = np.isclose(b.weights, b.weights.max())
    assert top.sum() == 2                                                             # the two "shared" role hypotheses tie
    hard = HardCollapseLinear(); hard.W = np.random.default_rng(1).normal(0, .5, hard.W.shape)
    order = np.arange(len(b.weights))[::-1]
    reversed_bundle = Bundle(tuple(b.hypotheses[i] for i in order), b.features[order], b.weights[order], b.source_kinds, b.feature_mode)
    assert hard.predict(b) == pytest.approx(hard.predict(reversed_bundle), abs=1e-12)
    expected = b.features[top].mean(axis=0)
    assert hard._inputs(b)[0][0] == pytest.approx(expected) and not np.allclose(expected, b.features[int(np.argmax(b.weights))])
