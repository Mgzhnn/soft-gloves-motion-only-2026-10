"""The plan defaults accepted by owner on 16 October 2026: the hard-collapse control's exact-tie rule and
the onset-based development budget. Synthetic, participant N=0."""
import numpy as np
import pytest

import sg_offline.inference as inference
from sg_offline.inference import Bundle, HardCollapseLinear
from test_review_implementation import motion_bundles, people_sessions  # noqa: F401


def _two_hypothesis_bundle(w0):
    from dataclasses import replace
    base = motion_bundles()[7]
    features = base.features[:2].copy(); features[1, 0] = features[0, 0] + 1.0   # two clearly different rows (distance column; kinetic columns stay zero)
    hyps = (replace(base.hypotheses[0], weight=w0), replace(base.hypotheses[1], weight=1. - w0))   # provenance weights must match
    return Bundle(hyps, features, np.array([w0, 1. - w0]), base.source_kinds, base.feature_mode)


def test_hard_collapse_reads_only_the_maximum_unless_the_weights_tie_within_1e_9():
    hard = HardCollapseLinear()
    near = _two_hypothesis_bundle(0.500001)                                   # the review's example (R76): not a tie
    x, _ = hard._inputs(near)
    assert np.allclose(x[0], near.features[0])                                # before the change: the two rows were averaged
    exact = _two_hypothesis_bundle(0.5)
    x, _ = hard._inputs(exact)
    assert np.allclose(x[0], exact.features.mean(axis=0))                     # an exact tie is still averaged
    within = _two_hypothesis_bundle(0.5 + 4e-10)
    x, _ = hard._inputs(within)
    assert np.allclose(x[0], within.features.mean(axis=0)) and getattr(inference, "HARD_TIE_TOLERANCE", None) == 1e-9


def test_development_budget_counts_onsets_by_default_and_ticks_only_when_asked():
    from sg_offline.experiment import compare
    sessions = people_sessions()
    wide = compare(sessions, arms=("rules",), selection="development", max_unwanted_proposals_per_hand_minute=1e9,
                   thresholds=(.3, .7), synthetic_check=True, burden_unit="ticks")
    fold = wide["folds"][0]
    point = next(p for p in fold["development_curve"] if p["threshold"] == .3)
    rates = {person: (u["onsets_per_hand_minute"], u["proposals_per_hand_minute"]) for person, u in point["metrics"]["unwanted"].items()}
    assert rates and all(o <= t for o, t in rates.values()) and any(o < t for o, t in rates.values())   # onsets never exceed ticks
    budget = max(o for o, t in rates.values()) + 0.5 * min(t - o for o, t in rates.values() if t > o)   # between the two readings
    by_unit = {unit: compare(sessions, arms=("rules",), selection="development", max_unwanted_proposals_per_hand_minute=budget,
                             thresholds=(.3,), synthetic_check=True, burden_unit=unit) for unit in ("onsets", "ticks")}
    on, ti = by_unit["onsets"]["folds"][0], by_unit["ticks"]["folds"][0]
    assert on["development_curve"][0]["within_budget"] is True and ti["development_curve"][0]["within_budget"] is False
    assert on["development_budget_unit"] == "onsets" and ti["development_budget_unit"] == "ticks"
    assert by_unit["onsets"]["configuration"]["burden_unit"] == "onsets"
    default = compare(sessions, arms=("rules",), selection="development", max_unwanted_proposals_per_hand_minute=budget, thresholds=(.3,), synthetic_check=True)
    assert default["configuration"]["burden_unit"] == "onsets" and default["folds"][0]["development_curve"][0]["within_budget"] is True
    with pytest.raises(ValueError, match="burden_unit"):
        compare(sessions, arms=("rules",), selection="fixed", synthetic_check=True, burden_unit="events")
