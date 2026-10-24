"""Coverage pins for declared behaviours that no test protected (13 of 28 single-setting
mutants survived both suites). Tests only; no behaviour changed. Synthetic, participant N = 0.

1. best_action_threshold proposes when the best action's score reaches the threshold (>=, not >).
2. best_action_score is the score of the most probable non-NQ action, not the grasp score.
3. marginal_argmax withholds exactly as build 5 did (score < threshold); the expected rows were checked against
   build 5 on the same cases.
4. The patience stop is honoured by every fitting loop (LinearJoint, MLPJoint, HypothesisPoolMLP, FactorizedJoint and
   the torch SequenceReadout), not only the one test_protocol_settings exercises.
5. fit_diagnostic.py defaults: 2000 passes, patience None, onset burden, learning rate 0.01, THRESHOLDS equal to
   experiment.THRESHOLD_GRID, and a specification's decision_rule reaches the replay configuration.
"""
import inspect
import json

import numpy as np
import pytest

import fit_diagnostic
import sg_offline.experiment as experiment
from sg_offline.experiment import ARMS, RULE_ARMS, SEQUENCE_ARMS, THRESHOLD_GRID, _examples, compare, factory, fit_learned_arm
from sg_offline.fixtures import scenarios
from sg_offline.replay import DECISION_RULES, Config, Replay
from test_review_implementation import people_sessions  # noqa: F401


class RightHandJoint:
    """Left hand certainly NQ; the right hand's marginal is exactly ``right`` (one non-zero row of the joint, so each
    column sum is the given number plus zeros and a score can equal a threshold bit for bit)."""
    name = "right_hand_stub"

    def __init__(self, right):
        r = np.zeros(7); r[:len(right)] = right
        self.joint = np.outer(np.eye(7)[0], r).ravel()

    def predict(self, bundle):
        return self.joint


def _right(right, threshold, rule):
    rows = Replay(Config(proposal_score_threshold=threshold, decision_rule=rule), RightHandJoint(right)).run(scenarios())
    return [r["outputs"][1] for r in rows]


def _above(t):
    return float(np.nextafter(t, 1.))


def _grasp_argmax_at(t):
    """G scores exactly t and is the first maximum: the other six outcomes share 1 - t, each below t (t > 1/7)."""
    return [(1. - t) / 6., t] + [(1. - t) / 6.] * 5


# On all nine scenario ticks the right hand is observed and fresh, so no observation reason masks a score decision;
# tick 0 has no approach rate yet, so a G there is withheld for that reason whatever the score.


@pytest.mark.parametrize("t", THRESHOLD_GRID)
def test_best_action_threshold_proposes_when_the_score_reaches_the_threshold(t):
    at = _right([1. - t, t], t, "best_action_threshold")
    assert all(o["best_action_score"] == t and o["raw_outcome"] == "G" for o in at)        # score == threshold: G is the decision
    assert [o["status"] for o in at[1:]] == ["proposed_simulation_only"] * 8
    above = _right([1. - t, t], _above(t), "best_action_threshold")
    assert all(o["raw_outcome"] == "NQ" and o["status"] == "no_new_request" for o in above)  # below the operating point: no new request


def test_best_action_score_is_the_most_probable_action_not_the_grasp_score():
    right = [.5, .1, .3, .1]                                                                # NQ, G, FP, AR: FP is the best action
    for rule in DECISION_RULES:
        outs = _right(right, .2, rule)
        assert all(o["best_action_score"] == .3 and o["grasp_score"] == .1 for o in outs)
    assert all(o["raw_outcome"] == "FP" for o in _right(right, .3, "best_action_threshold"))   # the operating point reads FP's 0.3, not G's 0.1
    assert all(o["raw_outcome"] == "NQ" for o in _right(right, _above(.3), "best_action_threshold"))
    assert all(o["best_action_score"] == .4 for o in _right([.2, .4, .3, .1], .2, "best_action_threshold"))   # G best: the two scores agree


# marginal_argmax (builds 1-5): the most probable outcome, NQ included, withheld when its own score is below the
# threshold. (right-hand marginal, threshold) -> (raw outcome, status and reason on ticks 1-8); tick 0 differs only
# where G passes the score rule (no approach rate yet). Every row was produced identically by build 5.
BUILD5 = [
    ([.4, .6], .6, "G", ("proposed_simulation_only", "software_rule_only")),               # score == threshold: not withheld
    ([.4, .6], _above(.6), "G", ("withheld", "insufficient_uncalibrated_score")),
    ([.6, .4], .6, "NQ", ("no_new_request", "software_rule_only")),
    ([.55, .45], .6, "NQ", ("withheld", "insufficient_uncalibrated_score")),                # a below-threshold NQ is withheld, not "no new request"
    ([.3, .2, .5], .5, "FP", ("withheld", "action_specific_geometry_not_implemented")),
    ([.3, .2, .5], .6, "FP", ("withheld", "insufficient_uncalibrated_score")),
    ([.5, .5], .5, "NQ", ("no_new_request", "software_rule_only")),                         # a tie goes to the first outcome
]


@pytest.mark.parametrize("right, threshold, outcome, expected", BUILD5)
def test_marginal_argmax_withholds_exactly_as_build_5(right, threshold, outcome, expected):
    outs = _right(right, threshold, "marginal_argmax")
    assert all(o["raw_outcome"] == outcome and o["outcome_score"] == max(right) for o in outs)
    assert all((o["status"], o["reason"]) == expected for o in outs[1:])
    first = (outs[0]["status"], outs[0]["reason"])
    assert first == (("withheld", "past_relative_approach_rate_unavailable_or_ambiguous") if expected[0] == "proposed_simulation_only" else expected)


@pytest.mark.parametrize("t", [t for t in THRESHOLD_GRID if t > 1 / 7])
def test_marginal_argmax_boundary_at_every_grid_threshold(t):
    right = _grasp_argmax_at(t)
    at = _right(right, t, "marginal_argmax")
    assert all(o["raw_outcome"] == "G" and o["outcome_score"] == t for o in at)
    assert [o["status"] for o in at[1:]] == ["proposed_simulation_only"] * 8                  # build 5: `score < threshold` withholds; equality does not
    above = _right(right, _above(t), "marginal_argmax")
    assert all((o["raw_outcome"], o["status"], o["reason"]) == ("G", "withheld", "insufficient_uncalibrated_score") for o in above)


LEARNED = [a for a in ARMS if a not in experiment.FLOOR_ARMS]   # every fitted arm (the per-action floor has no fitting loop)


def _arm_examples(arm):
    if arm in SEQUENCE_ARMS:
        pytest.importorskip("torch")
    model = factory(arm, 0, 15)
    examples, _ = _examples(model, people_sessions()[:1], 15, Config())
    data = [e.bundles for e in examples] if hasattr(model, "predict_window") else [e.bundle for e in examples]
    return model, examples, data


@pytest.mark.parametrize("arm", LEARNED)
def test_every_fitting_loop_stops_when_the_patience_rule_asks(arm, monkeypatch):
    model, examples, data = _arm_examples(arm)
    calls = []
    model._fit(data, [e.label for e in examples], 20, .01, .001, callback=lambda p: calls.append(p) or len(calls) >= 3)
    assert calls == [1, 2, 3]                                                               # the loop stops at the pass that asked
    # the same arm through the shared fold function: a development loss that never improves after pass 1
    model, examples, _ = _arm_examples(arm)
    monkeypatch.setattr(experiment, "mean_log_loss", lambda model, examples: 1.)
    _, _, stopping = fit_learned_arm(model, examples, examples[:4], 20, .01, .001, "t", synthetic_check=True, patience=2)
    assert (stopping["stopped_by"], stopping["best_pass"], stopping["passes_run"]) == ("patience", 1, 3)
    assert len(stopping["development_loss_curve"]) == 3 and stopping["max_passes"] == 20


def test_GUARD_fit_diagnostic_function_defaults_are_the_frozen_protocol():
    diag = inspect.signature(fit_diagnostic.diagnose).parameters
    assert diag["steps"].default == 2000 and diag["patience"].default is None
    assert diag["burden_unit"].default == "onsets" and diag["lr"].default == .01
    assert fit_diagnostic.THRESHOLDS == THRESHOLD_GRID and diag["thresholds"].default == THRESHOLD_GRID
    shared = inspect.signature(compare).parameters                                          # the diagnostic is the comparison fold for one person
    for key in ("seeds", "steps", "lr", "regularization", "thresholds", "burden_unit", "window", "patience"):
        assert diag[key].default == shared[key].default, key


def _run_spec_until_fitting(tmp_path, monkeypatch, **declared):
    """Run the specification route of fit_diagnostic.py up to the fitting call and return the arguments it receives."""
    spec = dict(scope="synthetic default check", rate_window=8, arms=["rules"], max_unwanted_proposals_per_hand_minute=10,
                training=[dict(run="run_1", manifest="m1.json", ledger="l1.json")],
                development=[dict(run="run_2", manifest="m2.json", ledger="l2.json")], **declared)
    path = tmp_path / "spec.json"; path.write_text(json.dumps(spec))
    seen = dict(load_configs=[])

    class Reached(Exception):
        pass

    def fake_load(entry, base, role, config):
        seen["load_configs"].append(config)
        return (dict(packets=(), opportunities=[]),
                dict(role=role, session_id=entry["run"], perception_checkpoints_sha256={"detector": "x"},
                     source_settings={}, object_provider={"kind": "masks"}))

    def fake_diagnose(*args, **kwargs):
        seen["arguments"] = inspect.signature(original).bind(*args, **kwargs).arguments
        raise Reached

    original = fit_diagnostic.diagnose
    monkeypatch.setattr(fit_diagnostic, "load_session", fake_load)
    monkeypatch.setattr(fit_diagnostic, "diagnose", fake_diagnose)
    out = tmp_path / "out"; registry = tmp_path / "absent_registry.csv"
    with pytest.raises(Reached):
        fit_diagnostic.run_diagnostic(path, out, registry)
    assert not out.exists() and not registry.exists()
    seen["defaults"] = inspect.signature(original).parameters
    return seen


def test_GUARD_a_specification_that_omits_a_setting_runs_with_the_frozen_default(tmp_path, monkeypatch):
    seen = _run_spec_until_fitting(tmp_path, monkeypatch)
    a = seen["arguments"]
    assert a["steps"] == 2000 and a["patience"] is None and a["burden_unit"] == "onsets" and a["lr"] == .01
    assert a["thresholds"] == THRESHOLD_GRID and a["config"].decision_rule == "best_action_threshold" and a["config"].rate_window == 8
    defaults = seen["defaults"]
    # every omitted setting equals the function default; early_limit_s is left out on purpose: the frozen tolerance
    # must be named in each specification, whatever the code default
    for key in ("seeds", "steps", "lr", "regularization", "thresholds", "burden_unit", "window", "patience"):
        assert a[key] == defaults[key].default, key


def test_GUARD_a_declared_specification_setting_reaches_fitting_and_replay(tmp_path, monkeypatch):
    seen = _run_spec_until_fitting(tmp_path, monkeypatch, decision_rule="marginal_argmax", steps=40, patience=7, lr=.02,
                                   burden_unit="ticks", thresholds=[.3, .6])
    a = seen["arguments"]
    assert a["config"].decision_rule == "marginal_argmax" and all(c.decision_rule == "marginal_argmax" for c in seen["load_configs"])
    assert (a["steps"], a["patience"], a["lr"], a["burden_unit"], a["thresholds"]) == (40, 7, .02, "ticks", (.3, .6))
