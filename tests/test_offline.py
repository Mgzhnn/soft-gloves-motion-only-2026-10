from dataclasses import replace, asdict
import json
import numpy as np
import pytest
from sg_offline.contracts import Stamp, Source, Point, Packet, LoadEstimate
from sg_offline.state import CausalKinematics, UnavailableKinetics, CalibratedTension, Calibration
from sg_offline.appearance import RGBHistogram, DINOv3Boundary, global_assignment
from sg_offline.inference import (hypotheses, LinearJoint, MLPJoint, HypothesisPoolMLP,
                                  SimpleRules, UnresolvedBoundary, Bundle, Hypothesis)
from sg_offline.fixtures import scenarios, make_packet
from sg_offline.replay import Replay, Config, trace_hash, canonical
from sg_offline.inputs import packets_from_jsonl
from sg_offline.evaluation import Reference, summarize, Opportunity, evaluate_opportunities


def calibration(side="left", **kwargs):
    return Calibration("synthetic-"+side, "fixture_actuator_wire_A", 10., 0., .2, (0., 2.), side, **kwargs)


def bundles(packets=None, kinetic=True):
    packets = packets or scenarios()
    kin = CausalKinematics(); provider = CalibratedTension({s: calibration(s) for s in ("left", "right")})
    return [hypotheses(p, kin.update(p), provider.estimate(p), kinetic) for p in packets]


@pytest.mark.parametrize("units", ["px", "mm", "normalized", "N"])
def test_metric_contract_rejects_other_units(units):
    with pytest.raises(ValueError):
        replace(scenarios()[0].hands[0], units=units)


def test_frames_and_clocks_cannot_mix():
    p = scenarios()[0]
    with pytest.raises(ValueError):
        replace(p, hands=(replace(p.hands[0], frame="world"), p.hands[1]))
    with pytest.raises(ValueError):
        replace(p, clock="unmapped_camera_clock")


@pytest.mark.parametrize("acquired,available", [(1., .5), (float("nan"), 2.), (0., float("inf")), (-1., 0.)])
def test_timestamp_domain(acquired, available):
    with pytest.raises(ValueError):
        Stamp("x", acquired, available, Source.SYNTHETIC)


def test_future_data_and_commands_rejected():
    p = scenarios()[0]
    with pytest.raises(ValueError):
        replace(p, decision_s=.1)
    with pytest.raises(ValueError):
        replace(p, previous_commands=("G", "NQ"), previous_command_times_s=(p.decision_s, None))
    with pytest.raises(ValueError):
        replace(p, previous_commands=("G", "NQ"))


def test_no_hand_object_namespace_collision():
    p = scenarios()[0]
    with pytest.raises(ValueError):
        replace(p, objects=(replace(p.objects[0], identity="left"),))


def test_missing_point_has_reason_and_is_not_zero():
    with pytest.raises(ValueError):
        replace(scenarios()[0].hands[0], xyz=None)
    assert scenarios()[4].hands[0].xyz is None


def test_missing_kinetics_and_zero_tension_are_different():
    p = make_packet(0, (0., 0., 1.), (.2, 0., 1.), raw=(0., None))
    unknown = UnavailableKinetics().estimate(p)
    calibrated = CalibratedTension({"left": calibration()}).estimate(p)
    assert unknown[0].value_N is None
    assert calibrated[0].value_N == 0.
    assert calibrated[1].value_N is None


def test_calibration_is_hand_bound_and_real_admission_denied():
    with pytest.raises(ValueError):
        calibration(synthetic=False)
    with pytest.raises(ValueError):
        CalibratedTension({"right": calibration("left")})
    p = make_packet(0, (0., 0., 1.), (.2, 0., 1.), raw=(.6, .6))
    p = replace(p, telemetry=tuple(replace(t, stamp=replace(t.stamp, source=Source.RECORDED)) for t in p.telemetry))
    assert CalibratedTension({"left": calibration()}).estimate(p)[0].value_N is None


def test_site_range_and_staleness_do_not_clip_to_force():
    p = make_packet(0, (0., 0., 1.), (.2, 0., 1.), raw=(3., 1.))
    provider = CalibratedTension({"left": calibration()})
    assert provider.estimate(p)[0].reason == "outside_calibration_range"
    p = replace(p, telemetry=(replace(p.telemetry[0], raw_value=1., sensor_site="elsewhere"),))
    assert provider.estimate(p)[0].reason == "site_or_unit_mismatch"
    p = replace(p, telemetry=(replace(p.telemetry[0], sensor_site="fixture_actuator_wire_A",
                                    stamp=replace(p.telemetry[0].stamp, acquired_s=.1)),))
    assert provider.estimate(p)[0].reason == "stale_channel"


def test_load_quantity_cannot_be_renamed():
    p = make_packet(0, (0., 0., 1.), (.2, 0., 1.), raw=(1., None))
    load = CalibratedTension({"left": calibration()}).estimate(p)[0]
    with pytest.raises(ValueError):
        replace(load, quantity="fingertip_force")
    with pytest.raises(ValueError):
        replace(load, frame="arbitrary")


def test_causal_secant_and_gap():
    p0, p1 = scenarios()[:2]
    kin = CausalKinematics(); assert kin.update(p0)["left"].velocity_mps is None
    assert kin.update(p1)["left"].velocity_mps == pytest.approx((1., 0., 0.))
    assert kin.update(scenarios()[8])["left"].velocity_mps is None


def test_point_definition_change_does_not_make_velocity():
    p0, p1 = scenarios()[:2]
    kin = CausalKinematics(); kin.update(p0)
    p1 = replace(p1, hands=(replace(p1.hands[0], definition="different_landmark"), p1.hands[1]))
    assert kin.update(p1)["left"].velocity_reason == "point_definition_changed"


def test_delayed_observation_does_not_overwrite_newer_history():
    kin = CausalKinematics()
    for p in scenarios()[:4]:
        kin.update(p)
    prior = kin.previous["left"]
    kin.update(scenarios()[5])
    assert kin.previous["left"] == prior


def test_sample_identity_immutable_but_unchanged_reuse_logged():
    p0, p1 = scenarios()[:2]
    conflict = replace(p1, hands=(replace(p1.hands[0], stamp=replace(p1.hands[0].stamp,
                                                                                sample_id=p0.hands[0].stamp.sample_id)), p1.hands[1]))
    with pytest.raises(ValueError):
        Replay().run([p0, conflict])
    reused = replace(p1, hands=(p0.hands[0], p1.hands[1]))
    rows = Replay().run([p0, reused])
    assert p0.hands[0].stamp.sample_id in rows[1]["reused_sample_ids"]


def test_session_and_packet_uniqueness():
    p0, p1 = scenarios()[:2]
    with pytest.raises(ValueError):
        Replay().run([p0, replace(p1, session_id="another")])
    with pytest.raises(ValueError):
        Replay().run([p0, replace(p1, packet_id=p0.packet_id)])


def test_full_replay_determinism_and_causal_prefix():
    p = scenarios()
    full = Replay().run(p)
    assert trace_hash(full) == trace_hash(Replay().run(p))
    assert canonical(full[:4]) == canonical(Replay().run(p[:4]))
    altered_future = p[:4] + [replace(q, objects=()) for q in p[4:]]
    assert canonical(full[:4]) == canonical(Replay().run(altered_future)[:4])


def test_both_hands_asynchronous_and_missingness_withheld():
    rows = Replay().run(scenarios())
    assert rows[1]["outputs"][0]["status"] == "proposed_simulation_only"
    assert rows[1]["outputs"][1]["raw_outcome"] == "NQ"
    assert rows[4]["outputs"][0]["status"] == "withheld"
    assert rows[4]["outputs"][1]["status"] == "proposed_simulation_only"
    assert rows[5]["outputs"][0]["reason"] == "stale_hand_observation"
    assert all(x["physical_command"] is None for r in rows for x in r["outputs"])


def test_stale_object_cannot_admit_biased_model():
    p0, p1 = scenarios()[:2]
    p1 = replace(p1, objects=(replace(p1.objects[0], stamp=replace(p1.objects[0].stamp, acquired_s=.1)),))
    model = LinearJoint(); model.W[7, -1] = 20  # deliberate G/NQ adversarial bias
    rows = Replay(model=model).run([p0, p1])
    assert rows[1]["outputs"][0]["reason"] == "fresh_target_geometry_unavailable"


def test_first_frame_missing_velocity_and_unimplemented_actions_withheld():
    p = scenarios()
    model = LinearJoint(); model.W[7, -1] = 20
    assert Replay(model=model).run(p[:1])[0]["outputs"][0]["status"] == "withheld"
    model.W[:] = 0; model.W[3*7, -1] = 20  # AR/NQ
    assert Replay(model=model).run(p[:2])[1]["outputs"][0]["reason"] == "action_specific_geometry_not_implemented"


def test_ambiguous_associations_normalized_and_null_retained():
    b = bundles()[-2]
    assert b.features.shape[1] == 42
    assert np.isclose(b.weights.sum(), 1.)
    assert any(None in h.object_ids for h in b.hypotheses)
    assert len({h.object_ids for h in b.hypotheses}) > 1
    assert {"shared", "unrelated", "symmetric", "unresolved"}.issubset({h.relation for h in b.hypotheses})


def test_candidate_truncation_is_exposed_and_unknown_score_retained():
    objects = tuple((f"o{i}", (i*.02, 0., 1.)) for i in range(4))
    p = make_packet(0, (0., 0., 1.), (.1, 0., 1.), objects=objects)
    b = bundles([p])[0]
    assert all(h.candidate_truncated for h in b.hypotheses)
    assert sum(h.weight for h in b.hypotheses if h.object_ids[0] is None) > .25/2.25


def test_role_prior_can_change_with_both_hands_moving():
    p0 = make_packet(0, (-.04, 0., 1.), (.04, 0., 1.))
    p1 = make_packet(1, (-.03, 0., 1.), (.02, 0., 1.))
    p2 = make_packet(2, (-.01, 0., 1.), (.01, 0., 1.))
    bs = bundles([p0, p1, p2])
    def support_difference(b):
        return sum(h.weight*(1 if h.roles == ("support", "manipulate") else -1 if h.roles == ("manipulate", "support") else 0) for h in b.hypotheses)
    assert support_difference(bs[1]) > 0
    assert support_difference(bs[2]) < 0


def test_mixture_gradient_against_numerical_difference():
    b = bundles()[2]; model = LinearJoint()
    rng = np.random.default_rng(9); model.W[:] = rng.normal(0, .02, model.W.shape)
    _, gradient = model.loss_gradient(b, 7, .003)
    for i, j in [(7, 0), (0, 41), (14, 3), (33, 35)]:
        old = model.W[i, j]; eps = 1e-6
        model.W[i, j] = old+eps; hi = model.loss_gradient(b, 7, .003)[0]
        model.W[i, j] = old-eps; lo = model.loss_gradient(b, 7, .003)[0]
        model.W[i, j] = old
        assert gradient[i, j] == pytest.approx((hi-lo)/(2*eps), abs=1e-7)


@pytest.mark.parametrize("factory", [LinearJoint, lambda: LinearJoint(False), MLPJoint, HypothesisPoolMLP])
def test_task_trainable_readouts_synthetic_loss_decreases(factory):
    bs = bundles()[:3]; labels = [0, 7, 7]; model = factory()
    initial = np.mean([-np.log(model.predict(b)[y]) for b, y in zip(bs, labels)])
    model.fit_synthetic(bs, labels, steps=60)
    after = np.mean([-np.log(model.predict(b)[y]) for b, y in zip(bs, labels)])
    assert after < initial
    assert model.fit_provenance.startswith("synthetic_only")
    assert all(np.isclose(model.predict(b).sum(), 1) for b in bs)


def test_nonlinear_control_capacity_and_hypothesis_information():
    model = HypothesisPoolMLP()
    assert sum(a.size for a in [model.A, model.a, model.B, model.b]) == 2051
    assert LinearJoint().W.size == 2058
    b = bundles()[1]
    order = np.arange(len(b.weights))[::-1]
    permuted = Bundle(tuple(b.hypotheses[i] for i in order), b.features[order], b.weights[order], b.source_kinds)
    assert model.predict(b) == pytest.approx(model.predict(permuted))


def test_no_request_high_load_does_not_force_request():
    b = bundles()[-1]
    p = SimpleRules(coordination=True, kinetics=True).predict(b).reshape(7,7)
    assert np.unravel_index(p.argmax(), p.shape) == (0, 0)


def test_reference_unknown_and_withholding_kept_in_denominator():
    rows = Replay().run(scenarios())
    r = Reference(rows[4]["packet_id"], "op1", "synthetic-person", "synthetic-session", ("G", None), "synthetic")
    counts = summarize(rows, [r])["counts"]
    assert counts["all_hand_decision_ticks"] == 18
    assert counts["known_reference_ticks"] == 1
    assert counts["unknown_reference_ticks"] == 17
    assert counts["missed_or_withheld_request_ticks"] == 1


def test_json_roundtrip_and_reference_fields_not_runtime_inputs(tmp_path):
    path = tmp_path/"packets.jsonl"
    path.write_text("\n".join(json.dumps(asdict(p)) for p in scenarios()))
    assert list(packets_from_jsonl(path)) == scenarios()
    data = asdict(scenarios()[0]); data["reference_label"] = "G"
    path.write_text(json.dumps(data))
    with pytest.raises(TypeError):
        list(packets_from_jsonl(path))


def test_histogram_mask_and_empty_region():
    rgb = np.zeros((2,2,3), dtype=np.uint8); rgb[0,0] = [255,0,0]
    mask = np.zeros((2,2), bool); mask[0,0] = True
    hist = RGBHistogram().describe(rgb, mask)
    assert hist[48] == 1 and hist.sum() == 1
    assert RGBHistogram().describe(rgb, np.zeros_like(mask)) is None
    with pytest.raises(ValueError):
        RGBHistogram().describe(rgb, mask.astype(np.uint8))


def test_global_assignment_is_not_greedy_and_ties_abstain():
    assert global_assignment([[.1,.2],[.11,10]])[0] == (1,0)
    assert global_assignment([[1.]], unmatched_cost=1.)[0] == (None,)
    assert global_assignment([[float("inf")]])[0] == (None,)


def test_stubs_are_not_silent_models_and_config_cannot_enable_hardware():
    with pytest.raises(NotImplementedError):
        DINOv3Boundary().describe(None, None)
    with pytest.raises(NotImplementedError):
        UnresolvedBoundary().predict(None)
    with pytest.raises(ValueError):
        Config(simulation_only=False)


def test_event_endpoint_keeps_first_wrong_proposal():
    rows = Replay().run(scenarios())
    op = Opportunity("op", "synthetic-person", "synthetic-session", .5, 1.3, (("AR",), ("NQ",)), "synthetic_only")
    result = evaluate_opportunities(rows, [op])[0]
    assert result["hands"][0]["result"] == "wrong_first"
    assert result["hands"][0]["first_proposal"] == (.6, "G")
    assert result["joint_compatible"] is False


def test_opportunity_without_any_tracking_is_missed_not_removed():
    op = Opportunity("op", "synthetic-person", "synthetic-session", 3., 4., (("G",), ("NQ",)), "synthetic_only")
    result = evaluate_opportunities([], [op])[0]
    assert result["hands"][0]["result"] == "missed_including_withheld"
    assert result["joint_compatible"] is False


def test_unknown_event_reference_does_not_become_no_request():
    op = Opportunity("op", "synthetic-person", "synthetic-session", 0., 4., (None, ("NQ",)), "synthetic_only")
    result = evaluate_opportunities([], [op])[0]
    assert result["hands"][0]["result"] == "unknown_reference"
    assert result["joint_compatible"] is None


def test_role_modes_independent_of_task_gate():
    p = scenarios()[3]
    kin = CausalKinematics()
    for packet in scenarios()[:3]: kin.update(packet)
    motion = kin.update(p); loads = UnavailableKinetics().estimate(p)
    neutral = hypotheses(p, motion, loads, role_mode="neutral")
    left = sum(h.weight for h in neutral.hypotheses if h.roles == ("support", "manipulate"))
    right = sum(h.weight for h in neutral.hypotheses if h.roles == ("manipulate", "support"))
    assert left == pytest.approx(right)
    assert Config(role_mode="neutral", task_gate=True).task_gate


def test_equal_mean_features_do_not_force_equal_set_control_outputs():
    model = HypothesisPoolMLP(seed=5)
    x = np.zeros((2,42)); x[:, -1] = 1; x[:,0] = [.1, 8.]
    h = (Hypothesis((None,None), "unresolved", ("other","other"), .5),)*2
    mixture = Bundle(h, x, np.array([.5,.5]), ("synthetic",))
    collapsed = Bundle(h, np.tile(x.mean(axis=0), (2,1)), np.array([.5,.5]), ("synthetic",))
    assert not np.allclose(model.predict(mixture), model.predict(collapsed), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("factory", [LinearJoint, MLPJoint, HypothesisPoolMLP])
def test_synthetic_trainer_enforces_source_provenance(factory):
    b = replace(bundles()[0], source_kinds=("recorded",))
    with pytest.raises(ValueError):
        factory().fit_synthetic([b], [0], steps=1)


def test_immutable_feature_bundle_and_point_inputs():
    b = bundles()[0]
    with pytest.raises(ValueError): b.features[0,0] = 2
    with pytest.raises(ValueError): replace(scenarios()[0].hands[0], xyz=[0., 0., 1.])


def test_asynchronous_reference_keeps_too_early_first_action():
    rows = Replay().run(scenarios())
    op = Opportunity("op", "synthetic-person", "synthetic-session", .5, 1.3,
                     (("G",), ("NQ",)), "synthetic_only", hand_windows_s=((.7, 1.2), (.5, 1.3)))
    outcome = evaluate_opportunities(rows, [op])[0]
    assert outcome["hands"][0]["result"] == "wrong_timing_first"
    assert outcome["hands"][0]["first_proposal"] == (.6, "G")


def test_joint_acceptable_sets_are_not_cartesian_and_participant_matched():
    import copy
    row = copy.deepcopy(Replay().run(scenarios())[1])
    row["outputs"][1].update(raw_outcome="G", status="proposed_simulation_only")
    op = Opportunity("op", "synthetic-person", "synthetic-session", .5, 1.,
                     (("G","R"), ("G","R")), "synthetic_only",
                     acceptable_joint_pairs=(("G","R"),("R","G")))
    assert evaluate_opportunities([row], [op])[0]["joint_compatible"] is False
    other = replace(op, participant_id="another-person")
    assert evaluate_opportunities([row], [other])[0]["hands"][0]["result"] == "missed_including_withheld"


def test_event_evaluator_rejects_reordered_or_duplicate_traces():
    rows = Replay().run(scenarios())
    with pytest.raises(ValueError): evaluate_opportunities(rows[::-1], [])
    with pytest.raises(ValueError): evaluate_opportunities([rows[0], rows[0]], [])


def test_tick_reference_never_crosses_participant_or_session():
    rows = Replay(Config()).run(scenarios())
    r = Reference(rows[4]["packet_id"], "op1", "another-person", "synthetic-session", ("G", None), "synthetic")
    result = summarize(iter(rows), [r])
    assert result["counts"]["unknown_reference_ticks"] == 18
    assert result["unmatched_reference_records"] == 1
    r = replace(r, participant_id="synthetic-person", session_id="another-session")
    assert summarize(rows, [r])["unmatched_reference_records"] == 1
    with pytest.raises(ValueError): summarize(rows[::-1])


def test_event_generator_has_same_first_issued_result():
    rows = Replay(Config()).run(scenarios())
    op = Opportunity("op", "synthetic-person", "synthetic-session", .5, 1.3, (("AR",), ("NQ",)), "synthetic_only")
    assert evaluate_opportunities(iter(rows), [op]) == evaluate_opportunities(rows, [op])


def test_v0_target_revision_rejects_unmigrated_flexion_contract():
    packet = scenarios()[0]
    assert packet.schema == "sg-offline/3"
    with pytest.raises(ValueError):
        replace(packet, schema="sg-offline/2")
    with pytest.raises(ValueError):
        replace(packet, schema="sg-offline/1")
    estimate = CalibratedTension({s: calibration(s) for s in ("left", "right")}).estimate(packet)[0]
    assert estimate.quantity == "specified_transmission_wire_axial_tension"
    with pytest.raises(ValueError):
        replace(estimate, quantity="proximal_flexion_tendon_axial_tension")
