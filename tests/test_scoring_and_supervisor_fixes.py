"""Scoring, threshold selection, scene cues and supervisor timing: one regression per defect. Synthetic, participant N=0."""
import numpy as np

from sg_offline.evaluation import Opportunity, evaluate_opportunities
from sg_offline.experiment import THRESHOLD_GRID, risk_controlled_threshold
from sg_offline.fsm import FSMConfig, GloveStateMachine, command_rows, simulate_commands
from sg_offline.scene import PlaneGeometry, Scene
from sg_offline.stats import unwanted_exposure
from test_action_integration import World, _packet

IDLE = dict(status="no_new_request", raw_outcome="NQ")


def _row(t, proposing):
    out = dict(status="proposed_simulation_only", raw_outcome="G", outcome_score=.9) if proposing else dict(IDLE)
    return dict(packet_id=f"p{t}", decision_s=t, outputs=[out, dict(IDLE)],
                inputs=dict(participant_id="P1", session_id="S1", hands=[dict(xyz=[0, 0, 0]), dict(xyz=[0, 0, 0])]))


def _ops():
    return [Opportunity("nq", "P1", "S1", 0., 1., (("NQ",), ("NQ",)), "test"),
            Opportunity("req", "P1", "S1", 1., 2., (("G",), ("NQ",)), "test")]


def _by_id(outcomes):
    return {o["opportunity_id"]: o["hands"][0] for o in outcomes}


def test_an_early_run_within_the_tolerance_is_credited_once_not_also_charged_as_unwanted():
    rows = [_row(t, t >= .95) for t in (.1, .5, .9, .95, 1., 1.05, 1.5)]
    outcomes = evaluate_opportunities(rows, _ops(), early_limit_s=.1)
    hands = _by_id(outcomes)
    assert hands["req"]["result"] == "compatible_first" and abs(hands["req"]["early_by_s"] - .05) < 1e-9
    assert hands["nq"]["result"] == "correct_no_new_request" and hands["nq"]["tolerated_early_ticks"] == 1
    assert unwanted_exposure(outcomes, _ops())["P1"]["unwanted_onsets"] == 0
    # beyond the tolerance the request fails on timing and the no-request row keeps the onset
    outcomes = evaluate_opportunities(rows, _ops(), early_limit_s=.02)
    hands = _by_id(outcomes)
    assert hands["req"]["result"] == "wrong_timing_first" and hands["nq"]["result"] == "unwanted"
    # with the default strict tolerance nothing changes
    assert _by_id(evaluate_opportunities(rows, _ops()))["nq"]["tolerated_early_ticks"] == 0


def test_an_early_run_that_stops_before_the_window_stays_unwanted():
    rows = [_row(t, t == .95) for t in (.1, .5, .9, .95, 1., 1.05, 1.5)]
    hands = _by_id(evaluate_opportunities(rows, _ops(), early_limit_s=.1))
    assert hands["nq"]["result"] == "unwanted" and hands["req"]["result"] == "missed_including_withheld"


def test_accepted_commands_get_the_same_early_tolerance_as_proposals():
    ts = [round(.05 * k, 2) for k in range(1, 40)]
    rows = [_row(t, t >= .9) for t in ts]                      # proposing from 0.90 s, the request opens at 1.0 s
    commands, _, _ = simulate_commands(rows, FSMConfig(confirm_ticks=2))
    assert [(c.t_s, c.kind) for c in commands if c.side == "left"] == [(.95, "close")]
    for scored in (rows, command_rows(rows, commands)):
        outcomes = evaluate_opportunities(scored, _ops(), early_limit_s=.1)
        hands = _by_id(outcomes)
        assert hands["req"]["result"] == "compatible_first" and hands["nq"]["result"] == "correct_no_new_request"
        assert unwanted_exposure(outcomes, _ops())["P1"]["unwanted_onsets"] == 0
    # strict tolerance: the command 0.05 s early is early, as a proposal run would be
    hands = _by_id(evaluate_opportunities(command_rows(rows, commands), _ops()))
    assert hands["req"]["result"] == "missed_including_withheld" and hands["nq"]["result"] == "unwanted"


def test_a_silent_top_threshold_does_not_make_every_threshold_infeasible():
    k = len(THRESHOLD_GRID)
    losses = {f"P{i}": [0] * k for i in range(20)}             # every person within budget everywhere
    record = risk_controlled_threshold(losses, THRESHOLD_GRID, alpha=.2, silent=[k - 1])
    assert record["threshold"] == THRESHOLD_GRID[0] and record["silent_thresholds"] == [THRESHOLD_GRID[-1]]
    # a silent point is never chosen, even when it is the only one within budget
    losses = {f"P{i}": [1] * (k - 1) + [0] for i in range(20)}
    assert risk_controlled_threshold(losses, THRESHOLD_GRID, alpha=.2, silent=[k - 1])["threshold"] is None


def test_shake_reversals_do_not_survive_a_gap_in_the_height_chain():
    world = World()
    geo = PlaneGeometry(.25, 8, reversal_deadband_mps=.1)

    def tick(k, t, z):
        packet = _packet(t, (world.point("left", np.array([-.3, 0., z]), k, t), world.point("right", np.array([.3, 0., .1]), k, t)),
                         (world.point("cup", np.array([0., 0., .1]), k, t, "obj"),))
        return geo.update(packet, Scene(packet.packet_id, world.plane(k, t)))

    for k in range(10):                                        # shaking at 0.6 m/s
        cues = tick(k, 1. + .1 * k, .1 + (.06 if k % 2 else 0.))
    assert cues.reversals["left"] >= 6
    assert tick(10, 4., .1).reversals["left"] == 0             # 3 s later, still: the chain restarted
    assert tick(11, 4.1, .1).reversals["left"] == 0


def test_the_cooldown_starts_when_the_opening_ends_and_is_not_skipped():
    m = GloveStateMachine("left", FSMConfig(confirm_ticks=1))  # open 0.8 s, cool-down 1.0 s
    m.step(0., dict(status="proposed_simulation_only", raw_outcome="G"))
    m.step(2., IDLE, veto=True)
    assert m.hand.state == "releasing"
    m.step(3.1, IDLE)                                          # opening ended at 2.8 s
    assert m.hand.state == "cooldown" and abs(m.hand.since_s - 2.8) < 1e-9
    m.step(3.7, IDLE)
    assert m.hand.state == "cooldown"
    m.step(3.8, IDLE)
    assert m.hand.state == "open"


def test_the_object_size_adjustment_spans_the_documented_band_and_reaches_the_supervisor():
    targets = {}
    for size in (.01, .03, .08, .13, .3):
        m = GloveStateMachine("left", FSMConfig(confirm_ticks=1))
        targets[size] = m.step(1., dict(status="proposed_simulation_only", raw_outcome="G"), object_size_m=size).flexion_target_fraction
    assert abs(targets[.03] - .65) < 1e-9 and abs(targets[.08] - .55) < 1e-9 and abs(targets[.13] - .45) < 1e-9
    assert targets[.01] == targets[.03] and targets[.3] == targets[.13]
    rows = [dict(_row(t, True), object_size_m=dict(left=.03, right=None)) for t in (1., 1.1)]
    commands, _, _ = simulate_commands(rows, FSMConfig(confirm_ticks=1))
    assert commands[0].object_size_m == .03 and abs(commands[0].flexion_target_fraction - .65) < 1e-9


def test_the_comparison_reads_the_time_origin_from_the_recording_manifest(tmp_path):
    import pytest
    from run_comparison import load_sessions
    from test_review_implementation import _comparison_inputs, write_manifest
    entry = _comparison_inputs(tmp_path, people=("p1",))[0]
    write_manifest(tmp_path / entry["manifest"], "p1", origin=.3)
    sessions, sources = load_sessions([entry], tmp_path)
    assert sources[0]["origin_s"] == .3 and sources[0]["manifest_sha256"]
    assert sessions[0]["opportunities"][0].start_s == .3        # the ledger's 0 s is the recording's first frame
    with pytest.raises(ValueError, match="differs from the recording manifest"):
        load_sessions([dict(entry, origin_s=0.)], tmp_path)   # a typed origin that disagrees is refused
    with pytest.raises(ValueError, match="manifest"):
        load_sessions([{k: v for k, v in entry.items() if k != "manifest"}], tmp_path)
    write_manifest(tmp_path / entry["manifest"], "p2")
    with pytest.raises(ValueError, match="same participant"):
        load_sessions([entry], tmp_path)
