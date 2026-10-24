"""The training-protocol settings: the supervisor's best-action operating-point rule, the declared
threshold grid, and patience-based stopping with a pass cap. Synthetic, participant N=0."""
import inspect

import numpy as np
import pytest

import sg_offline.replay as replay
from sg_offline.experiment import THRESHOLD_GRID, compare, fit_learned_arm
from sg_offline.fixtures import scenarios
from sg_offline.inference import LinearJoint, SimpleRules
from sg_offline.replay import Config, Replay
from sg_offline.training import Example
from test_review_implementation import motion_bundles, people_sessions  # noqa: F401


class FixedJoint:
    """A model whose right hand always carries P(G) = p_g (the rest on NQ) and whose left hand is NQ."""
    name = "fixed_joint_stub"

    def __init__(self, p_g):
        self.joint = np.outer([1., 0, 0, 0, 0, 0, 0], [1. - p_g, p_g, 0, 0, 0, 0, 0]).ravel()

    def predict(self, bundle):
        return self.joint


def _statuses(model, **config):
    rows = Replay(Config(**config), model).run(scenarios())
    return rows, [[o["status"] for o in r["outputs"]] for r in rows]


def test_best_action_rule_makes_the_threshold_an_operating_point_below_one_half():
    model = FixedJoint(.3)
    rows, new = _statuses(model, proposal_score_threshold=.2)                       # default rule: best_action_threshold
    _, old = _statuses(model, proposal_score_threshold=.2, decision_rule="marginal_argmax")
    proposed_new = sum(s[1] == "proposed_simulation_only" for s in new); proposed_old = sum(s[1] == "proposed_simulation_only" for s in old)
    assert proposed_new > 0 and proposed_old == 0                                       # NQ (0.7) is the argmax: the old rule can never propose
    assert all(s[1] in ("no_new_request", "withheld") for s in old)
    first = next(r for r in rows if r["outputs"][1]["status"] == "proposed_simulation_only")["outputs"][1]
    assert first["raw_outcome"] == "G" and first["argmax_outcome"] == "NQ" and first["best_action_score"] == pytest.approx(.3)
    assert first["decision_rule"] == "per_hand_best_action_threshold_then_common_supervisor"
    assert rows[0]["implementation_version"] == "motion-only-build/7" and rows[0]["configuration"]["decision_rule"] == "best_action_threshold"
    # above the operating point the hand is simply no new request, never a withheld request
    rows_hi, hi = _statuses(model, proposal_score_threshold=.4)
    assert not any(s[1] == "proposed_simulation_only" for s in hi)
    assert all(o["reason"] != "insufficient_uncalibrated_score" for r in rows_hi for o in r["outputs"])
    with pytest.raises(ValueError, match="decision rule"):
        Config(decision_rule="argmax")


def test_RULE_the_hand_authored_rules_decide_identically_under_both_decision_rules_inside_the_grid():
    for threshold in (.02, .3, .6, .8):
        _, new = _statuses(SimpleRules(), proposal_score_threshold=threshold)
        _, old = _statuses(SimpleRules(), proposal_score_threshold=threshold, decision_rule="marginal_argmax")
        assert new == old and any(s[1] == "proposed_simulation_only" for s in new)
    _, silent = _statuses(SimpleRules(), proposal_score_threshold=.9)              # the constant 0.862 is below 0.9 under either rule
    assert not any("proposed_simulation_only" in s for s in silent)


def _examples(labels):
    bundles = motion_bundles()
    return [Example(b, lab, "p", "s", f"id{i}", float(i), "op", "synthetic") for i, (b, lab) in enumerate(zip(bundles, labels))]


def test_patience_stops_after_the_declared_passes_without_a_new_best_and_restores_the_best_pass():
    train = _examples([0, 7] * 6)                                                    # NQ|NQ and G|NQ
    constant = _examples([tuple(range(49))] * 4)                                     # every pair accepted: development loss 0 (up to rounding) at every pass
    before, after, stopping = fit_learned_arm(LinearJoint(), train, constant, steps=30, lr=.02, regularization=.001, split_id="t", synthetic_check=True, patience=5)
    assert stopping["stopped_by"] == "patience" and stopping["patience"] == 5 and stopping["max_passes"] == 30
    assert stopping["passes_run"] == stopping["best_pass"] + 5 < 30                   # exactly `patience` passes after the last (rounding-level) improvement
    assert len(stopping["development_loss_curve"]) == stopping["passes_run"] and stopping["development_loss_at_best"] == pytest.approx(0., abs=1e-9)
    assert stopping["development_loss_after_restore"] == pytest.approx(stopping["development_loss_at_best"], abs=1e-12) and after < before
    _, _, full = fit_learned_arm(LinearJoint(), train, constant, steps=30, lr=.02, regularization=.001, split_id="t", synthetic_check=True, patience=None)
    assert full["stopped_by"] == "max_passes" and full["passes_run"] == 30
    with pytest.raises(ValueError, match="patience"):
        fit_learned_arm(LinearJoint(), train, constant, steps=3, lr=.02, regularization=.001, split_id="t", synthetic_check=True, patience=0)


def test_GUARD_comparison_defaults_and_records_carry_the_declared_grid_cap_patience_and_rule():
    sig = inspect.signature(compare).parameters
    assert sig["steps"].default == 2000 and sig["patience"].default is None and sig["thresholds"].default == THRESHOLD_GRID   # frozen: all passes, best development pass
    assert THRESHOLD_GRID == (.02, .05, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9) and replay.IMPLEMENTATION_VERSION == "motion-only-build/7"
    result = compare(people_sessions(), arms=("kt_r",), steps=8, patience=3, selection="development", max_unwanted_proposals_per_hand_minute=1e9,
                     thresholds=(.3, .6), synthetic_check=True)
    assert result["configuration"]["patience"] == 3 and result["configuration"]["decision_rule"] == "best_action_threshold"
    for fold in result["folds"]:
        es = fold["early_stopping"]
        assert es["stopped_by"] in ("patience", "max_passes") and es["passes_run"] <= 8 and es["max_passes"] == 8 and es["patience"] == 3
    with pytest.raises(ValueError, match="patience"):
        compare(people_sessions(), arms=("rules",), patience=-1, synthetic_check=True)
