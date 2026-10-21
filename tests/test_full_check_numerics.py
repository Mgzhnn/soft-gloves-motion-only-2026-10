"""Finite-logit training regressions; synthetic offline software, participant N=0."""
import numpy as np
import pytest

from sg_offline.inference import (FactorizedJoint, FlatJoint, HardCollapseLinear,
                                  HypothesisPoolMLP, LinearJoint, MLPJoint)
from sg_offline.training import bundles_for, mean_log_loss, Example
from test_full_check_streams import packet


@pytest.mark.parametrize("factory", [LinearJoint, lambda: LinearJoint(False), HardCollapseLinear,
                                     FactorizedJoint, lambda: FactorizedJoint(True, True),
                                     MLPJoint, FlatJoint, HypothesisPoolMLP])
def test_finite_extreme_logits_keep_a_corrective_target_gradient(factory):
    bundle = next(bundles_for([packet(1., .1)]))[1]
    model = factory()
    if hasattr(model, "W"):
        target = model.W
        index = (7, -1)
    elif isinstance(model, FactorizedJoint):
        target = model.W_left
        index = (1, -1)
    else:
        target = model.b
        index = 7
    target[index] = -800.
    model._fit([bundle], [7], steps=1, lr=.01, regularization=0.)
    assert target[index] > -800., "a finite confidently wrong target must receive a corrective update"
    assert model.fit_diagnostics["loss_curve"][0]["unregularized_log_loss"] > 800.


def test_extreme_partial_labels_keep_the_conditional_target_gradient():
    bundle = next(bundles_for([packet(1., .1)]))[1]
    model = LinearJoint()
    model.W[7:14, -1] = -800.
    loss, gradient = model.loss_gradient(bundle, tuple(range(7, 14)))
    assert loss > 800.
    assert gradient[7:14, -1].sum() == pytest.approx(-1.)
    assert gradient[:, -1].sum() == pytest.approx(0., abs=1e-12)
