"""Flat-comparator information fix and contrast reporting. Synthetic, participant N=0.

Regression tests failed on the earlier code; GUARD tests hold on both.
Counterexample: at an exact tie between two objects, averaging the two
candidate orders turned approach rates (-0.1, +0.1) and (-0.2, +0.2) m/s into the same flat input.
"""
from dataclasses import replace
import numpy as np
import pytest
from sg_offline.fixtures import make_packet
from sg_offline.state import CausalKinematics, RelativeGeometry, DisabledKinetics
from sg_offline.inference import hypotheses, flat_features


def head_camera(packet):
    """As on the head-mounted camera: points in the moving optical frame, so no velocity is admitted and
    only same-time distances and their rates carry motion."""
    move = lambda pt: replace(pt, frame="camera_optical", frame_is_fixed=False)
    return replace(packet, hands=tuple(move(h) for h in packet.hands), objects=tuple(move(o) for o in packet.objects))


def tie_bundle(speed, midpoint=.1):
    """Left hand exactly between a cup (x=0) and a jar (x=0.2), moving toward the cup at `speed` m/s."""
    objs = (("cup", (0., 0., 1.)), ("jar", (.2, 0., 1.)))
    before = head_camera(make_packet(0, (midpoint + speed*.1, 0., 1.), (.6, 0., 1.), objects=objs))   # 0.1 s earlier
    now = head_camera(make_packet(1, (midpoint, 0., 1.), (.6, 0., 1.), objects=objs))
    kin, rel = CausalKinematics(), RelativeGeometry()
    for p in (before, now):
        motions, relatives = kin.update(p), rel.update(p)
    return hypotheses(now, motions, DisabledKinetics().estimate(now), False, .25, "weak_speed", relatives)


def test_the_reviewers_tie_counterexample_gives_different_flat_inputs():
    slow, fast = tie_bundle(.1), tie_bundle(.2)
    rates = lambda b: sorted({round(float(r[1]) * .2, 6) for r, h in zip(b.features, b.hypotheses) if h.object_ids[0]})
    assert rates(slow) == [-.1, .1] and rates(fast) == [-.2, .2]  # the set model sees different candidates
    assert float(np.abs(flat_features(slow) - flat_features(fast)).max()) > .1  # before: identical


def test_both_objects_can_be_recovered_from_the_flat_input_even_at_a_tie():
    from sg_offline.inference import flat_object_rows, flat_rows_from_features
    for bundle in (tie_bundle(.1), tie_bundle(.2), tie_bundle(.15, midpoint=.07)):
        truth = flat_object_rows(bundle)
        recovered = flat_rows_from_features(flat_features(bundle))
        pairs = [(recovered[0], recovered[1]), (recovered[1], recovered[0])]
        assert any(np.allclose(a, truth[0], atol=1e-9) and np.allclose(b, truth[1], atol=1e-9) for a, b in pairs)


def test_GUARD_the_fixed_flat_input_stays_continuous_and_name_blind():
    change = lambda half: float(np.abs(flat_features(tie_bundle(.1, midpoint=.1 - half)) - flat_features(tie_bundle(.1, midpoint=.1 + half))).max())
    assert change(.00001) < .01 and change(.0001) == pytest.approx(change(.001) / 10, rel=.3)
    a = tie_bundle(.1, midpoint=.099)
    renamed = replace(a, hypotheses=tuple(replace(h, object_ids=tuple(None if s is None else "x" + s for s in h.object_ids))
                                          for h in a.hypotheses))
    assert np.array_equal(flat_features(a), flat_features(renamed))


def test_GUARD_the_flat_capacity_stays_matched_to_kt_set():
    from sg_offline.experiment import parameter_count
    from sg_offline.inference import FlatJoint, HypothesisPoolMLP
    assert abs(parameter_count(FlatJoint()) - parameter_count(HypothesisPoolMLP())) / parameter_count(HypothesisPoolMLP()) < .05


# ---- contrasts: an average over the remaining people is shown with its feasibility counts --------

def fold(arm, held, success, feasible=True, dev_success=.5, selection="development"):
    curve = [] if selection == "fixed" else [dict(threshold=.6, feasible=feasible, success=dev_success, metrics={})]
    return dict(arm=arm, held_out_person=held, seed=0, threshold_selection=selection, reporting_threshold=.6,
                development_budget_feasible=feasible, development_curve=curve,
                reporting_metrics=dict(success={held: dict(success_fraction=success)},
                                       unwanted={held: dict(proposals_per_hand_minute=1.)}))


def test_a_conditional_average_carries_its_feasibility_counts():
    from sg_offline.experiment import contrasts_from_folds
    # the simple controls are the fitted M-I / M-C models; the selection and exclusion rules are unchanged.
    folds = [fold("kt_r", "p1", .9, feasible=False), fold("mi_learned", "p1", .5), fold("mc_learned", "p1", .4, dev_success=.3),
             fold("kt_r", "p2", .8), fold("mi_learned", "p2", .5), fold("mc_learned", "p2", .5, dev_success=.2),
             fold("kt_r", "p3", .7), fold("mi_learned", "p3", .4), fold("mc_learned", "p3", .6, dev_success=.9)]
    primary = contrasts_from_folds(folds, (0,))[0]
    assert "mean_difference" not in primary  # no unqualified average
    assert primary["mean_difference_among_included_people"] == pytest.approx(((.8-.5) + (.7-.6)) / 2)
    assert primary["people_with_candidate"] == 3 and primary["n_people"] == 2
    assert primary["feasibility"]["kt_r"] == {"feasible": ["p2", "p3"], "budget_infeasible": ["p1"], "fixed_threshold": []}
    assert "not an estimate for all" in primary["population_note"]
