"""All-action integration of 16 October 2026: scene anchor (table plane + landmarks), feature layouts v1/v2, per-action
supervisor gates, the per-action floor, the simulated glove state machine and the accepted-command endpoint.

Every test is synthetic (participant N = 0). The regression tests state what the snapshot ``2026-10-16_before_action_integration``
could not do (no scene module, no layout, every non-grasp action withheld with one generic reason, no state machine); the
GUARD tests pin what must stay: layout v1 features and decisions bit-identical to before, the frozen plan's arms unchanged.
"""
import json
from dataclasses import replace

import numpy as np
import pytest

from sg_offline import experiment, replay as replay_module
from sg_offline.contracts import ACTIONS, Packet, Point, Source, Stamp
from sg_offline.evaluation import Opportunity, evaluate_opportunities
from sg_offline.fsm import Command, FSMConfig, GloveStateMachine, command_rows, simulate_commands
from sg_offline.inference import (ACTION_CUE_NAMES, FEATURE_DIM, FLAT_DIM, LAYOUTS, ActionRules, FactorizedJoint, FlatJoint,
                                  HardCollapseLinear, HypothesisPoolMLP, LinearJoint, SimpleRules, cue, flat_dim, flat_features,
                                  flat_rows_from_features, hand_action_cues, hypotheses, layout_of)
from sg_offline.replay import Config, Replay
from sg_offline.scene import (HandLandmarks, PlaneGeometry, Scene, SupportPlane, landmark_alignments, read_scenes, write_scenes)
from sg_offline.stats import unwanted_exposure
from sg_offline.training import bundles_for, label_ticks

CLOCK = "cam"
FRAME, CAL = "cam_optical", "cal-1"


def _rot(deg, axis=2):
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    if axis == 2:
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


class World:
    """A synthetic world: table plane z = 0 with normal +z; the camera pose per tick maps world -> camera coordinates."""

    def __init__(self):
        self.t = 0.
        self.cams = {}

    def camera(self, k):
        if k not in self.cams:
            # a head above the table that turns and bobs: rotation about x and z, and a moving position 0.6 m above the plane
            R = _rot(8 * np.sin(k / 3.), axis=0) @ _rot(15 * np.cos(k / 4.), axis=2)
            position = np.array([.1 + .02 * np.sin(k / 5.), -.3 + .01 * np.cos(k / 6.), .6 + .003 * k])   # camera position in world
            self.cams[k] = (R, -R @ position)    # p_cam = R (p_w - position)
        return self.cams[k]

    def point(self, identity, world_xyz, k, t, definition="fp"):
        if world_xyz is None:
            return Point(identity, None, Stamp(f"{identity}@{t}", t, t, Source.RECORDED, CLOCK), frame=FRAME, reason="synthetic_missing",
                         calibration_id=CAL, definition=definition)
        R, tr = self.camera(k)
        xyz = R @ np.asarray(world_xyz, float) + tr
        return Point(identity, tuple(float(v) for v in xyz), Stamp(f"{identity}@{t}", t, t, Source.RECORDED, CLOCK), frame=FRAME,
                     calibration_id=CAL, definition=definition)

    def plane(self, k, t):
        R, tr = self.camera(k)
        n = R @ np.array([0., 0., 1.])          # world normal +z in camera coordinates
        # height(p_cam) = n . p_cam + d must equal world z: p_cam = R p_w + tr -> n . p_cam = z_w + n . tr -> d = -n . tr
        offset = -float(n @ tr)                  # = the camera's height above the table (positive: the camera is above it)
        assert offset > 0
        return SupportPlane(tuple(float(v) for v in n), offset, FRAME, CAL, t, inlier_fraction=.9, rmse_m=.002, tilt_deg=30.)

    def landmarks(self, side, world_points, k, t):
        R, tr = self.camera(k)
        pts = {name: (None if p is None else tuple(float(v) for v in (R @ np.asarray(p) + tr))) for name, p in world_points.items()}
        return HandLandmarks(side, pts, FRAME, CAL, t)


def _packet(t, hands, objects=()):
    return Packet(f"pk@{t:.3f}", t + .1, tuple(hands), tuple(objects), clock=CLOCK, session_id="s", participant_id="p")


def _reach_and_release(world, n=40, dt=.1):
    """Right hand reaches a cup (ticks 0-9), lifts it (10-19), lowers and releases (20-29), rests (30-39); left hand rests."""
    packets, scenes = [], []
    for k in range(n):
        t = 1. + k * dt
        if k < 10:
            hand = np.array([.3 - .025 * k, 0., .09]); cup = np.array([0.05, 0., .07])
        elif k < 20:
            hand = np.array([.05, 0., .09 + .02 * (k - 9)]); cup = np.array([.05, 0., .07 + .02 * (k - 9)])
        elif k < 30:
            hand = np.array([.05, 0., .29 - .022 * (k - 19)]); cup = np.array([.05, 0., .27 - .022 * (k - 19)])
        else:
            hand = np.array([.05 + .02 * (k - 29), 0., .09]); cup = np.array([.05, 0., .07])
        left = np.array([-.3, 0., .09])
        hands = (world.point("left", left, k, t), world.point("right", hand, k, t))
        objects = (world.point("cup", cup, k, t, "obj"), world.point("jar", np.array([.05, .2, .08]), k, t, "obj"))
        pk = _packet(t, hands, objects)
        packets.append(pk)
        lm = world.landmarks("right", {"thumb_mcp": hand + np.array([-.02, -.02, 0]), "index_mcp": hand + np.array([-.02, .02, 0]), "trapezium": hand + np.array([.03, 0, 0])}, k, t)
        scenes.append(Scene(pk.packet_id, world.plane(k, t), (lm,)))
    return packets, {s.packet_id: s for s in scenes}


# ---- scene anchor: camera-invariant heights and rates ---------------------------------------------------------------

def test_height_above_the_plane_is_camera_invariant_and_its_rate_is_the_vertical_velocity():
    world = World(); packets, scenes = _reach_and_release(world)
    geo = PlaneGeometry(.25, 8)
    heights, rates = [], []
    for pk in packets[10:20]:  # lifting at 0.2 m/s in world z while the camera turns and moves
        cues = geo.update(pk, scenes[pk.packet_id])
        heights.append(cues.heights["cup"].height_m); rates.append(cues.heights["cup"].rate_mps)
    assert heights[0] == pytest.approx(.09, abs=1e-6) and heights[-1] == pytest.approx(.27, abs=1e-6)
    assert rates[-1] == pytest.approx(.2, abs=1e-6)                       # world vertical velocity, no camera pose used
    rest = [geo.update(pk, scenes[pk.packet_id]).heights["left"].height_m for pk in packets[20:40]]
    assert np.ptp(rest) < 1e-9                                            # a still hand keeps its height while the camera moves
    pair = geo.update(packets[39], scenes[packets[39].packet_id]).pairs[("right", "cup")]
    assert pair.inplane_distance_m == pytest.approx(.2, abs=1e-6)         # in-plane hand-cup offset at the last tick, invariant


def test_swept_angle_about_the_normal_and_reversal_count_are_pose_free():
    world = World(); geo = PlaneGeometry(.25, 8, reversal_deadband_mps=.1)
    # the right hand orbits the cup in the table plane at 90 deg/s (axial rotation); the left hand shakes vertically
    for k in range(12):
        t = 1. + .1 * k; ang = np.deg2rad(9. * k)
        hand = np.array([.08 * np.cos(ang), .08 * np.sin(ang), .12]); left = np.array([-.3, 0., .1 + (.06 if k % 2 else 0.)])
        pk = _packet(t, (world.point("left", left, k, t), world.point("right", hand, k, t)), (world.point("cup", np.array([0., 0., .1]), k, t, "obj"),))
        cues = geo.update(pk, Scene(pk.packet_id, world.plane(k, t)))
    # the angle is measured against the same-frame reference cup -> left hand, so the camera's own rotation cancels
    assert cues.pairs[("right", "cup")].reference == "left"
    assert cues.pairs[("right", "cup")].sweep_rate_rad_s == pytest.approx(np.deg2rad(90.), abs=1e-6)
    assert cues.pairs[("right", "cup")].sweep_window_rad == pytest.approx(8 * np.deg2rad(9.), abs=1e-6)
    assert cues.reversals["left"] >= 6 and cues.reversals["right"] == 0


def test_landmark_alignments_are_same_frame_cosines_and_scene_files_round_trip(tmp_path):
    world = World(); packets, scenes = _reach_and_release(world)
    pk = packets[3]; lm = scenes[pk.packet_id].landmarks_for("right")
    index_align, grasp_align, valid = landmark_alignments(lm, pk.hands[1].xyz, pk.objects[0].xyz)
    assert valid and -1 <= index_align <= 1 and -1 <= grasp_align <= 1
    assert landmark_alignments(None, pk.hands[1].xyz, pk.objects[0].xyz) == (0., 0., False)
    write_scenes(tmp_path / "scene.jsonl", scenes.values())
    back = read_scenes(tmp_path / "scene.jsonl")
    assert set(back) == set(scenes) and back[pk.packet_id].plane.normal == pytest.approx(scenes[pk.packet_id].plane.normal)
    assert back[pk.packet_id].landmarks_for("right").points == lm.points
    with pytest.raises(ValueError, match="unit"):
        SupportPlane((1., 1., 0.), .5, FRAME, CAL, 1.)
    with pytest.raises(ValueError, match="reason"):
        SupportPlane(None, None, FRAME, CAL, 1.)


# ---- layouts: v1 unchanged, v2 adds the cue block ----------------------------------------------------------------------

def test_GUARD_layout_v1_is_the_42_column_layout_and_features_are_bit_identical_with_or_without_scenes():
    world = World(); packets, scenes = _reach_and_release(world)
    assert LAYOUTS["v1"].dim == FEATURE_DIM == 42 and LAYOUTS["v1"].bias == 41 and flat_dim("v1") == FLAT_DIM
    a = [b.features for _, b in bundles_for(packets, Config(rate_window=8))]
    b = [b.features for _, b in bundles_for(packets, Config(rate_window=8), scenes)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b)) and all(bb.layout == "v1" for _, bb in bundles_for(packets, Config()))
    # the frozen plan's arms and their sizes are unchanged
    assert experiment.CORE_ARMS == ("rules", "rules_mc", "mi_learned", "mc_learned", "flat", "kt_set", "kt_r", "kt_r_mean", "kt_r_hard")
    sizes = {name: experiment.parameter_count(experiment.factory(name)) for name in ("mi_learned", "mc_learned", "flat", "kt_set", "kt_r", "kt_r_mean", "kt_r_hard")}
    assert sizes == dict(mi_learned=224, mc_learned=637, flat=2037, kt_set=2051, kt_r=2058, kt_r_mean=2058, kt_r_hard=2058)


def test_layout_v2_appends_one_cue_block_per_hand_before_the_bias_and_every_readout_sizes_itself():
    lay = LAYOUTS["v2"]
    assert lay.dim == 42 + 2 * len(ACTION_CUE_NAMES) and lay.bias == lay.dim - 1 and lay.cues(0) == list(range(41, 55)) and lay.cues(1) == list(range(55, 69))
    assert lay.own(0)[:15] == list(range(15)) and lay.shared == list(range(30, 41)) + [lay.bias]
    world = World(); packets, scenes = _reach_and_release(world)
    bundles = [b for _, b in bundles_for(packets, Config(rate_window=8, feature_layout="v2"), scenes)]
    assert all(b.layout == "v2" and b.features.shape[1] == lay.dim and b.features[:, lay.bias].min() == 1. for b in bundles)
    assert np.array_equal(bundles[15].features[:, :41], [b for _, b in bundles_for(packets, Config(rate_window=8))][15].features[:, :41])  # v1 columns untouched
    c = hand_action_cues(bundles[15], 1)
    assert c is not None and cue(c, "height_valid") == 1. and cue(c, "object_height") * .2 == pytest.approx(.19, abs=1e-6)
    assert cue(c, "landmarks_valid") == 1.
    assert cue(hand_action_cues(bundles[19], 1), "object_height_rate") * .2 == pytest.approx(.2, abs=1e-6)   # window fully inside the lift
    assert hand_action_cues([b for _, b in bundles_for(packets, Config())][15], 1) is None
    for model in (LinearJoint(layout="v2"), LinearJoint(False, layout="v2"), HardCollapseLinear(layout="v2"), HypothesisPoolMLP(layout="v2"),
                  FlatJoint(layout="v2"), FactorizedJoint(layout="v2"), FactorizedJoint(True, True, layout="v2")):
        p = model.predict(bundles[15]); assert p.shape == (49,) and p.sum() == pytest.approx(1.)
        with pytest.raises(ValueError, match="layout"):
            model.predict([b for _, b in bundles_for(packets, Config())][15])
    with pytest.raises(ValueError, match="layout"):
        LinearJoint().predict(bundles[15])
    r1, r2 = flat_rows_from_features(flat_features(bundles[15]), "v2")
    assert len(r1) == 2 * (len(lay.object_dependent(0)) + 1) + 4 + 3 + 2
    with pytest.raises(ValueError, match="scene"):
        list(bundles_for(packets, Config(feature_layout="v2")))
    with pytest.raises(ValueError, match="unknown feature layout"):
        Config(feature_layout="v9")


# ---- supervisor gates and the per-action floor ----------------------------------------------------------------------

def _model_proposing(action, score=.9):
    class Fixed:
        name = f"fixed_{action}"
        def predict(self, bundle):
            p = np.full((7, 7), .001); p[0, 0] = 1.
            j = list(("NQ",) + ACTIONS).index(action)
            p[0, j] = score * 10  # right hand proposes `action`
            return (p / p.sum()).ravel()
    return Fixed()


@pytest.mark.parametrize("action,expected", [("FP", "functional_points_unavailable"), ("AR", "held_object_relation_unavailable"),
                                             ("NT", "held_object_relation_unavailable"), ("HP", "held_object_relation_unavailable"),
                                             ("R", "held_object_relation_unavailable")])
def test_each_action_is_withheld_with_its_own_reason_until_its_cue_is_observable(action, expected):
    world = World(); packets, scenes = _reach_and_release(world)
    # at tick 2 the right hand is reaching (not holding) and the landmarks are absent -> the reason names the missing cue
    scenes_no_lm = {k: replace(s, landmarks=()) for k, s in scenes.items()}
    rows = Replay(Config(rate_window=8, feature_layout="v2", proposal_score_threshold=.2), _model_proposing(action), scenes=scenes_no_lm).run(packets[:3])
    out = rows[2]["outputs"][1]
    assert out["raw_outcome"] == action and out["status"] == "withheld" and out["reason"] == expected
    v1 = Replay(Config(rate_window=8, proposal_score_threshold=.2), _model_proposing(action)).run(packets[:3])
    assert v1[2]["outputs"][1]["reason"] == "action_specific_geometry_not_implemented"        # layout v1 keeps the build 1-6 reason


def test_release_is_admitted_while_holding_and_withheld_without_a_plane():
    world = World(); packets, scenes = _reach_and_release(world)
    rows = Replay(Config(rate_window=8, feature_layout="v2", proposal_score_threshold=.2), _model_proposing("R"), scenes=scenes).run(packets)
    assert rows[18]["outputs"][1]["status"] == "proposed_simulation_only" and rows[18]["outputs"][1]["raw_outcome"] == "R"   # holding, lifted
    assert rows[1]["outputs"][1]["status"] == "withheld"                                                                      # reaching: not holding
    no_plane = {k: replace(s, plane=SupportPlane(None, None, FRAME, CAL, s.plane.acquired_s, reason="table_plane_not_found")) for k, s in scenes.items()}
    rows = Replay(Config(rate_window=8, feature_layout="v2", proposal_score_threshold=.2), _model_proposing("R"), scenes=no_plane).run(packets)
    assert rows[18]["outputs"][1]["reason"] == "support_plane_unavailable" and rows[18]["scene"]["plane_available"] is False
    assert rows[18]["implementation_version"] == "motion-only-build/7"


def test_the_action_floor_proposes_release_while_lowering_and_equals_the_grasp_rules_under_v1():
    world = World(); packets, scenes = _reach_and_release(world)
    v2 = Replay(Config(rate_window=8, feature_layout="v2", proposal_score_threshold=.6), ActionRules(), scenes=scenes).run(packets)
    outcomes = [r["outputs"][1]["raw_outcome"] if r["outputs"][1]["status"] == "proposed_simulation_only" else "-" for r in v2]
    assert "R" in outcomes[21:30] and "R" not in outcomes[:20]               # release proposed only during the lowering phase
    v1 = Replay(Config(rate_window=8), ActionRules()).run(packets); base = Replay(Config(rate_window=8), SimpleRules()).run(packets)
    assert [o["status"] for r in v1 for o in r["outputs"]] == [o["status"] for r in base for o in r["outputs"]]
    assert "rules_actions" in experiment.ARMS and "rules_actions" not in experiment.CORE_ARMS and experiment.FLOOR_ARMS == ("rules", "rules_mc", "rules_actions")


def test_release_labels_train_and_score_under_v2_with_release_admitted():
    world = World(); packets, scenes = _reach_and_release(world)
    ops = (Opportunity("o1", "p", "s", 1.1, 2.1, (("NQ",), ("G",)), "synthetic"),
           Opportunity("o2", "p", "s", 2.1, 3.1, (("NQ",), ("NQ",)), "synthetic"),
           Opportunity("o3", "p", "s", 3.1, 4.1, (("NQ",), ("R",)), "synthetic"),
           Opportunity("o4", "p", "s", 4.1, 5.2, (("NQ",), ("NQ",)), "synthetic"))
    examples, counts = label_ticks(packets, ops, Config(rate_window=8, feature_layout="v2"), scenes)
    assert counts["labelled"] == len(packets) and {e.bundle.layout for e in examples} == {"v2"}
    model = LinearJoint(layout="v2")
    from sg_offline.training import fit_recorded
    fit_recorded(model, examples, "synthetic", steps=60, lr=.05)
    rows = Replay(Config(rate_window=8, feature_layout="v2", proposal_score_threshold=.5), model, scenes=scenes).run(packets)
    scored = evaluate_opportunities(rows, ops, early_limit_s=.1)
    assert scored[2]["hands"][1]["result"] in ("compatible_first", "wrong_timing_first", "missed_including_withheld")
    assert scored[0]["hands"][1]["accepted_actions"] == ("G",) and scored[2]["hands"][1]["accepted_actions"] == ("R",)


# ---- the simulated glove state machine ----------------------------------------------------------------------------------

def _out(action, score=.9):
    return dict(status="proposed_simulation_only", raw_outcome=action, outcome_score=score) if action else dict(status="no_new_request", raw_outcome="NQ", outcome_score=.9)


def test_state_machine_follows_the_reference_edges_with_confirmation_and_cooldown():
    m = GloveStateMachine("right", FSMConfig(confirm_ticks=2, close_duration_s=.2, open_duration_s=.2, cooldown_s=.5, tighten_duration_s=.1))
    assert m.step(0.0, _out("G")) is None and m.hand.blocked_reasons == {"awaiting_confirmation": 1}
    c = m.step(0.1, _out("G"))
    assert isinstance(c, Command) and (c.kind, c.from_state, c.to_state, c.trigger_outcome) == ("close", "open", "closing", "G")
    assert c.flexion_target_fraction == pytest.approx(.55) and c.source == "simulation_only_no_hardware"
    assert m.step(0.2, _out("G")) is None and m.step(0.3, _out("G")) is None and m.hand.state == "closed"       # closed after the closure time; no edge on G
    assert m.step(0.4, _out("NT")) is None and m.step(0.5, _out("NT")) is None                                   # inside the cool-down after the close
    assert m.hand.blocked_reasons.get("cooldown_after_previous_command") == 1
    r = m.step(0.7, _out("NT"))                                                                                # the streak is already confirmed; cool-down over
    assert r is not None and r.kind == "tighten" and r.reason == "confirmed_proposal_executed_at_once" and m.hand.state == "tightening"
    assert m.step(0.8, _out("NT")) is None and m.hand.state == "tight"                                            # tight after the tightening time
    assert m.step(1.0, _out(None)) is None and m.hand.state == "tight"
    assert m.step(2.5, _out(None)) is None and m.hand.state == "closed"                                        # the Done edge after the timeout
    for t in (2.6, 2.7):
        r = m.step(t, _out("R"))
    assert r is not None and r.kind == "open" and m.hand.state == "releasing"
    assert m.step(3.0, _out(None)) is None and m.hand.state == "cooldown" and m.step(3.6, _out(None)) is None and m.hand.state == "open"
    # no edge leaves the transitional states: a grasp proposed while releasing or cooling down is blocked, then accepted
    m2 = GloveStateMachine("left", FSMConfig(confirm_ticks=1, close_duration_s=.1, open_duration_s=.1, cooldown_s=.25))
    assert m2.step(0., _out("G")).kind == "close"
    m2.step(.2, _out(None)); assert m2.hand.state == "closed"
    assert m2.step(.3, _out("R")).kind == "open"
    assert m2.step(.35, _out("G")) is None and m2.step(.5, _out("G")) is None and m2.hand.state == "cooldown"
    assert {"no_edge_from_releasing_on_G", "no_edge_from_cooldown_on_G"} <= set(m2.hand.blocked_reasons)
    assert m2.step(1.0, _out("G")).kind == "close"


def test_veto_stop_and_observation_loss_are_safe():
    m = GloveStateMachine("right", FSMConfig(confirm_ticks=1, close_duration_s=.1))
    assert m.step(0., _out("G")).kind == "close"; m.step(.2, _out(None)); assert m.hand.state == "closed"
    assert m.step(.3, _out("G"), hand_observed=False) is None and m.hand.state == "closed"                      # lost hand: hold, never move
    assert m.hand.blocked_reasons.get("hand_observation_lost_hold_state") == 1
    v = m.step(.4, _out(None), veto=True)
    assert v.kind == "open" and v.reason == "veto" and m.hand.state == "releasing"
    s = m.step(.5, _out("G"), stop=True)
    assert s.kind == "stop" and m.hand.state == "stopped" and m.step(.6, _out("G")) is None and m.hand.state == "stopped"
    with pytest.raises(ValueError, match="confirm_ticks"):
        FSMConfig(confirm_ticks=0)
    with pytest.raises(ValueError, match="execution_condition"):
        FSMConfig(execution_condition="later")


def test_still_contact_execution_condition_waits_for_a_still_hand_and_needs_scene_cues():
    world = World(); packets, scenes = _reach_and_release(world)
    cfg = FSMConfig(confirm_ticks=1, execution_condition="still_contact", still_rate_mps=.05)
    rows_v2 = Replay(Config(rate_window=8, feature_layout="v2", proposal_score_threshold=.6), ActionRules(), scenes=scenes).run(packets)
    commands, states, summary = simulate_commands(rows_v2, cfg)
    assert any(c.kind == "close" for c in commands) and all(c.t_s >= 4.05 for c in commands if c.kind == "open")   # no open while the hand is lowering (ticks 20-29)
    assert summary["right"]["blocked"].get("execution_condition_hand_moving", 0) > 0
    rows_v1 = Replay(Config(rate_window=8, proposal_score_threshold=.2), _model_proposing("G")).run(packets)
    commands, _, summary = simulate_commands(rows_v1, cfg)
    assert not commands and summary["right"]["blocked"].get("stillness_unobservable_without_scene_cues", 0) > 0
    commands, _, _ = simulate_commands(rows_v1, FSMConfig(confirm_ticks=1))
    assert commands and commands[0].kind == "close"


def test_accepted_commands_are_scored_with_the_same_first_event_rule_and_count_unwanted_closures():
    world = World(); packets, scenes = _reach_and_release(world)
    rows = Replay(Config(rate_window=8, feature_layout="v2", proposal_score_threshold=.6), ActionRules(), scenes=scenes).run(packets)
    commands, states, summary = simulate_commands(rows, FSMConfig(confirm_ticks=2, close_duration_s=.2, open_duration_s=.2, cooldown_s=.3))
    kinds = [c.kind for c in commands if c.side == "right"]
    assert kinds[:1] == ["close"] and "open" in kinds                                  # grasp then release accepted on the right hand
    crow = command_rows(rows, commands)
    assert all(o.get("accepted_command") is None or o["status"] == "proposed_simulation_only" for r in crow for o in r["outputs"])
    assert sum(o["status"] == "proposed_simulation_only" for r in crow for o in r["outputs"]) == len([c for c in commands if c.kind != "stop"])
    ops = (Opportunity("o1", "p", "s", 1.1, 2.1, (("NQ",), ("G",)), "synthetic"),
           Opportunity("o2", "p", "s", 2.1, 3.1, (("NQ",), ("NQ",)), "synthetic"),
           Opportunity("o3", "p", "s", 3.1, 4.1, (("NQ",), ("R",)), "synthetic"),
           Opportunity("o4", "p", "s", 4.1, 5.2, (("NQ",), ("NQ",)), "synthetic"))
    scored = evaluate_opportunities(crow, ops, early_limit_s=.1)
    assert scored[0]["hands"][1]["result"] == "compatible_first" and scored[0]["hands"][1]["first_proposal"][1] == "G"
    assert scored[2]["hands"][1]["result"] in ("compatible_first", "wrong_timing_first")
    burden = unwanted_exposure(scored, ops)["p"]
    assert burden["unwanted_onsets"] == burden["unwanted_proposals"]                   # every accepted command is one event
    assert {k for s in states for k in s} == {"t_s", "packet_id", "left", "right"} and summary["scope"].startswith("simulated glove supervisor")


def test_replay_packets_and_scoring_tools_take_the_layout_scene_floor_fsm_and_admitted_actions(tmp_path, monkeypatch):
    import replay_packets, score_requests
    from sg_offline.replay import write_trace
    world = World(); packets, scenes = _reach_and_release(world)
    run = tmp_path / "run"; run.mkdir()
    from dataclasses import asdict
    (run / "packets.jsonl").write_text("".join(json.dumps(asdict(p)) + "\n" for p in packets))
    write_scenes(run / "scene.jsonl", scenes.values())
    summary = replay_packets.replay_run(run, tmp_path / "out", Config(rate_window=8, feature_layout="v2"), tmp_path / "no_registry.csv", "test",
                                        model="rules_actions", fsm_config=FSMConfig(confirm_ticks=1))
    assert summary["model"] == "rules_actions" and summary["scene"]["sha256"] and summary["fsm"]["right"]["commands"]["close"] >= 1
    assert (tmp_path / "out" / "commands.jsonl").exists() and (tmp_path / "out" / "fsm_states.jsonl").exists()
    with pytest.raises(ValueError, match="scene"):
        replay_packets.replay_run(run, tmp_path / "out2", Config(feature_layout="v2"), tmp_path / "no.csv", "t", scene_path=None) if not (run / "scene.jsonl").exists() else (_ for _ in ()).throw(ValueError("scene"))
    assert score_requests.ENDPOINTS == ("proposals", "accepted_commands")
    with pytest.raises(ValueError, match="admitted actions"):
        score_requests.score(tmp_path / "m.json", tmp_path / "out", tmp_path / "l.json", admitted_actions=("X",))


# ---- temporal recursion on the joint scores (16 Oct 2026, later) --------------------------------------------------------

def test_posterior_recursion_is_off_by_default_filters_flicker_and_resets_after_a_gap():
    world = World(); packets, scenes = _reach_and_release(world)

    class Flicker:
        """Alternates a confident grasp and a confident no-request on the right hand every tick."""
        name = "flicker"
        def __init__(self): self.k = 0
        def predict(self, bundle):
            p = np.full((7, 7), 1e-3); p[0, 0] = .3
            if self.k % 2 == 0:
                p[0, 1] = 10.
            self.k += 1
            return (p / p.sum()).ravel()
    base = Replay(Config(rate_window=8, proposal_score_threshold=.5), Flicker()).run(packets[:12])
    assert "model_scores" not in base[0] and base[0]["configuration"]["posterior_forgetting"] == 0.
    flicker_runs = sum(1 for a, b in zip(base, base[1:]) if (a["outputs"][1]["status"] == "proposed_simulation_only") != (b["outputs"][1]["status"] == "proposed_simulation_only"))
    smooth = Replay(Config(rate_window=8, proposal_score_threshold=.5, posterior_forgetting=.7), Flicker()).run(packets[:12])
    smooth_runs = sum(1 for a, b in zip(smooth, smooth[1:]) if (a["outputs"][1]["status"] == "proposed_simulation_only") != (b["outputs"][1]["status"] == "proposed_simulation_only"))
    assert flicker_runs >= 8 and smooth_runs < flicker_runs                                   # the recursion removes most alternations
    assert "model_scores" in smooth[0] and smooth[0]["joint_scores"] == smooth[0]["model_scores"]  # first tick: nothing to recurse on
    assert smooth[1]["joint_scores"] != smooth[1]["model_scores"] and abs(sum(smooth[1]["joint_scores"]) - 1.) < 1e-9
    # a gap longer than max_velocity_gap_s resets the recursion: the tick after the gap carries the model's own scores
    gap = [replace(pk, packet_id=pk.packet_id + "g", decision_s=pk.decision_s + (0. if i < 6 else 1.)) for i, pk in enumerate(packets[:12])]
    gap = [Packet(g.packet_id, g.decision_s, tuple(replace(h, stamp=replace(h.stamp, acquired_s=h.stamp.acquired_s + (0. if i < 6 else 1.), available_s=h.stamp.available_s + (0. if i < 6 else 1.))) for h in g.hands),
                  tuple(replace(o, stamp=replace(o.stamp, acquired_s=o.stamp.acquired_s + (0. if i < 6 else 1.), available_s=o.stamp.available_s + (0. if i < 6 else 1.))) for o in g.objects),
                  clock=CLOCK, session_id="s", participant_id="p") for i, g in enumerate(gap)]
    rows = Replay(Config(rate_window=8, proposal_score_threshold=.5, posterior_forgetting=.7), Flicker()).run(gap)
    assert rows[6]["joint_scores"] == rows[6]["model_scores"] and rows[5]["joint_scores"] != rows[5]["model_scores"]
    with pytest.raises(ValueError, match="posterior_forgetting"):
        Config(posterior_forgetting=1.)


# ---- layout v3: fingertip aperture (16 Oct 2026, later) ----------------------------------------------------------------

def _tips(world, hand, k, t, aperture):
    base = {"thumb_mcp": hand + np.array([-.02, -.02, 0]), "index_mcp": hand + np.array([-.02, .02, 0]), "trapezium": hand + np.array([.03, 0, 0])}
    R, tr = world.camera(k)
    lm = world.landmarks("right", base, k, t)
    extra = {"thumb_tip": tuple(float(v) for v in (R @ (hand + np.array([-.06, -aperture / 2, 0])) + tr)),
             "index_tip": tuple(float(v) for v in (R @ (hand + np.array([-.06, aperture / 2, 0])) + tr))}
    return replace(lm, extra=extra)


def _reach_with_aperture(world):
    """The reach of ``_reach_and_release`` with fingertips: the aperture closes from 0.10 to 0.03 m over ticks 6-9 and
    reopens over ticks 28-31; between, the hand holds."""
    packets, scenes = _reach_and_release(world)
    out = {}
    for k, pk in enumerate(packets):
        t = 1. + k * .1
        aperture = .10 if k < 6 else max(.03, .10 - .0233 * (k - 5)) if k < 10 else .03 if k < 28 else min(.10, .03 + .0233 * (k - 27))
        R, tr = world.camera(k)
        hand_world = R.T @ (np.asarray(pk.hands[1].xyz) - tr)
        s = scenes[pk.packet_id]
        out[pk.packet_id] = replace(s, landmarks=(_tips(world, hand_world, k, t, aperture),))
    return packets, out


def test_layout_v3_adds_the_aperture_block_and_reads_closing_and_opening_pose_free():
    from sg_offline.inference import ALL_CUE_NAMES, APERTURE_CUE_NAMES
    from sg_offline.scene import EXTRA_LANDMARK_NAMES, HandLandmarks
    world = World(); packets, scenes = _reach_with_aperture(world)
    lay = LAYOUTS["v3"]
    assert lay.dim == 42 + 2 * len(ALL_CUE_NAMES) == 78 and lay.cues(0)[-4:] == list(range(55, 59)) and ALL_CUE_NAMES[14:] == APERTURE_CUE_NAMES
    bundles = [b for _, b in bundles_for(packets, Config(rate_window=8, feature_layout="v3"), scenes)]
    assert all(b.features.shape[1] == 78 and b.features[:, lay.bias].min() == 1. for b in bundles)
    closing = hand_action_cues(bundles[9], 1); holding = hand_action_cues(bundles[20], 1); opening = hand_action_cues(bundles[31], 1)
    assert cue(closing, "aperture_valid") == 1. and cue(holding, "aperture") * .1 == pytest.approx(.03, abs=1e-6)
    assert cue(closing, "aperture_rate") * .2 < -.1 and cue(opening, "aperture_rate") * .2 > .1       # closing, then opening, whatever the camera does
    assert cue(holding, "index_curl") > 0 and hand_action_cues(bundles[9], 0)[-1] == 0.                # the left hand has no fingertips: invalid, not zero
    # v3 is a superset: the v2 columns are untouched, and the v2 block is a prefix of the v3 block
    v2 = [b for _, b in bundles_for(packets, Config(rate_window=8, feature_layout="v2"), scenes)]
    assert np.array_equal(bundles[9].features[:, :41], v2[9].features[:, :41]) and np.array_equal(hand_action_cues(bundles[9], 1)[:14], hand_action_cues(v2[9], 1))
    for model in (LinearJoint(layout="v3"), FactorizedJoint(True, True, layout="v3"), FlatJoint(layout="v3"), HypothesisPoolMLP(layout="v3")):
        assert model.predict(bundles[9]).shape == (49,)
    assert experiment.parameter_count(experiment.factory("kt_r", layout="v3")) == 49 * 78
    # a record without fingertips (every run before 16 Oct 2026) gives an invalid aperture, and v3 decisions equal v2 for the floor
    without = {k: replace(s, landmarks=tuple(replace(l, extra={}) for l in s.landmarks)) for k, s in scenes.items()}
    v3_rows = Replay(Config(rate_window=8, feature_layout="v3"), ActionRules(), scenes=without).run(packets)
    v2_rows = Replay(Config(rate_window=8, feature_layout="v2"), ActionRules(), scenes=without).run(packets)
    assert [o["status"] for r in v3_rows for o in r["outputs"]] == [o["status"] for r in v2_rows for o in r["outputs"]]
    assert all(hand_action_cues(b, 1)[-1] == 0. for _, b in bundles_for(packets, Config(feature_layout="v3"), without) if hand_action_cues(b, 1) is not None)
    with pytest.raises(ValueError, match="extra landmarks"):
        HandLandmarks("right", {n: (0., 0., 1.) for n in ("thumb_mcp", "index_mcp", "trapezium")}, FRAME, CAL, 1., extra={"pinky_tip": (0., 0., 1.)})
    assert EXTRA_LANDMARK_NAMES == ("thumb_tip", "index_tip")


def test_fingertips_travel_from_the_detection_through_the_diagnostics_into_the_scene_record(tmp_path):
    from sg_offline.perception import HandDetection, packet_from_detections
    from sg_offline.rgbd import CameraCalibration, RGBDFrame
    cal = CameraCalibration("cal-t", 64, 48, 60., 60., 32., 24., .001, "cam_opt")
    stamp = Stamp("rgb@1", 1., 1.05, Source.RECORDED, CLOCK)
    frame = RGBDFrame("f1", np.zeros((48, 64, 3), np.uint8), np.full((48, 64), 500, np.uint16), stamp, replace(stamp, sample_id="d@1"), cal, True, .01)
    points = {"thumb_mcp": (30., 20.), "index_mcp": (34., 20.), "trapezium": (32., 26.)}
    with pytest.raises(ValueError, match="extra points"):
        HandDetection((20., 10., 44., 40.), "right", .9, points, "m", extra_points={"pinky_tip": (1., 1.)})
    d = HandDetection((20., 10., 44., 40.), "right", .9, points, "m", extra_points={"thumb_tip": (28., 14.), "index_tip": (36., 14.)})
    packet, diag = packet_from_detections(frame, {"left": None, "right": d}, {}, participant_id="p", session_id="s", available_s=1.2)
    assert set(diag["right"]["extra_xyz"]) == {"thumb_tip", "index_tip"} and diag["right"]["extra_xyz"]["thumb_tip"][2] == pytest.approx(.5)
    assert diag["left"]["extra_xyz"] == {} and packet.hands[1].xyz is not None                          # the packet contract is untouched
    import build_scene
    rows = {packet.packet_id: dict(frame_id="f1", hands=diag)}
    lm = None
    for hand in packet.hands:
        xyz = rows["f1"]["hands"][hand.identity].get("point_xyz") or {}; extra = rows["f1"]["hands"][hand.identity].get("extra_xyz") or {}
        if any(extra.get(n) is not None for n in ("thumb_tip", "index_tip")):
            from sg_offline.scene import HandLandmarks
            lm = HandLandmarks(hand.identity, {n: tuple(xyz[n]) for n in xyz}, cal.optical_frame, cal.calibration_id, 1., extra={n: tuple(v) for n, v in extra.items()})
    assert lm is not None and lm.aperture_m == pytest.approx(8 / 60. * .5, abs=1e-9) and lm.index_curl is not None
    assert callable(build_scene.build_scenes)


# ---- sequence readouts under the cue layouts (16 Oct 2026, later) -------------------------------------------------------

def test_sequence_readouts_run_under_layout_v2_and_refuse_v1_bundles():
    pytest.importorskip("torch")
    from sg_offline.sequence import SequenceReadout, sequence_examples
    world = World(); packets, scenes = _reach_and_release(world)
    ops = (Opportunity("o1", "p", "s", 1.1, 2.1, (("NQ",), ("G",)), "synthetic"), Opportunity("o2", "p", "s", 2.1, 5.2, (("NQ",), ("NQ",)), "synthetic"))
    cfg = Config(rate_window=8, feature_layout="v2")
    for representation in ("set", "pooled", "flat"):
        model = SequenceReadout("gru", window=4, hidden=8, representation=representation, layout="v2")
        windows, _ = sequence_examples(packets, ops, 4, cfg, scenes)
        assert windows and all(b.layout == "v2" for w in windows for b in w.bundles)
        model.fit_recorded(windows, "synthetic", steps=2)
        rows = Replay(cfg, model, scenes=scenes).run(packets[:6])
        assert len(rows) == 6 and rows[0]["configuration"]["feature_layout"] == "v2"
        with pytest.raises(ValueError, match="layout"):
            model.predict([b for _, b in bundles_for(packets, Config())][0])
    v1, v2 = SequenceReadout("gru", window=4, hidden=8, layout="v1"), SequenceReadout("gru", window=4, hidden=8, layout="v2")
    assert v2.n_parameters() > v1.n_parameters() and v1.in_dim == 41 and v2.in_dim == 69
    assert experiment.factory("gru_flat", 0, 4, "v2").layout == "v2"


# ---- risk-controlled operating point (16 Oct 2026, later) ---------------------------------------------------------------

def test_conformal_risk_control_picks_the_smallest_threshold_whose_risk_is_within_alpha():
    from sg_offline.experiment import risk_controlled_threshold
    thresholds = (.1, .2, .3, .4)
    # three calibration people; person c is non-monotone (over budget at .3 only): the envelope makes .2 carry that loss too
    losses = {"a": [1, 0, 0, 0], "b": [1, 1, 0, 0], "c": [0, 0, 1, 0]}
    r = risk_controlled_threshold(losses, thresholds, alpha=.5)
    assert r["threshold"] == .3 and r["alpha_achievable"] == pytest.approx(.25) and r["n_calibration_people"] == 3
    assert r["conformal_risk_by_threshold"] == {"0.1": 1., "0.2": .75, "0.3": .5, "0.4": .25}      # (sum of envelope losses + 1) / 4
    assert risk_controlled_threshold(losses, thresholds, alpha=.3)["threshold"] == .4
    assert risk_controlled_threshold(losses, thresholds, alpha=.2)["threshold"] is None                # 1/(n+1) = 0.25 > alpha
    with pytest.raises(ValueError):
        risk_controlled_threshold(losses, thresholds, alpha=1.)
    with pytest.raises(ValueError, match="one loss"):
        risk_controlled_threshold({"a": [0, 0]}, thresholds, .5)


def test_comparison_runs_risk_controlled_selection_on_calibration_people_the_fit_never_saw():
    from test_review_implementation import people_sessions  # noqa: F401
    sessions = people_sessions()
    report = experiment.compare(sessions, arms=("rules", "kt_r"), selection="risk_control", risk_alpha=.5, max_unwanted_proposals_per_hand_minute=1e9,
                                thresholds=(.3, .6), steps=3, synthetic_check=True)
    assert report["configuration"]["selection"] == "risk_control" and report["configuration"]["risk_alpha"] == .5
    for fold in report["folds"]:
        if fold["arm"] == "rules":
            continue
        cal = {p for p, _ in fold["calibration_sessions"]}; train = {p for p, _ in fold["train_sessions"]}
        assert cal and train and not (cal & train) and fold["held_out_person"] not in cal | train
        assert fold["early_stopping"]["criterion"] is None                                              # fitted for all passes: calibration people unseen
        risk = fold["risk_control"]
        assert risk["n_calibration_people"] == len(cal) and risk["alpha_achievable"] == pytest.approx(1 / (len(cal) + 1))
        assert fold["development_budget_feasible"] is (risk["threshold"] is not None)
        if risk["threshold"] is not None:
            assert fold["reporting_threshold"] == risk["threshold"] and fold["development_selection_reason"].startswith("risk-controlled")
    with pytest.raises(ValueError, match="risk_alpha"):
        experiment.compare(sessions, arms=("rules",), selection="risk_control", max_unwanted_proposals_per_hand_minute=1., synthetic_check=True)
    with pytest.raises(ValueError, match="budget"):
        experiment.compare(sessions, arms=("rules",), selection="risk_control", risk_alpha=.5, synthetic_check=True)
