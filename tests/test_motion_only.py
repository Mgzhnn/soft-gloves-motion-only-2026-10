from dataclasses import replace

import numpy as np
import pytest

from sg_offline.contracts import Source
from sg_offline.fixtures import scenarios
from sg_offline.inference import (
    HypothesisPoolMLP, KINETIC_FEATURE_COLUMNS, LinearJoint, MLPJoint,
    SimpleRules, hypotheses,
)
from sg_offline.replay import Config, Replay, canonical
from sg_offline.state import Calibration, CalibratedTension, CausalKinematics


class ForbiddenProvider:
    def estimate(self, packet):
        raise AssertionError("disabled kinetics provider was called")


def varied_packets():
    return [replace(packet, telemetry=tuple(
        replace(channel, raw_value=1.9, sensor_site="different-site", units="A",
                stamp=replace(channel.stamp, source=Source.RECORDED,
                              acquired_s=0., sample_id="reused-telemetry"))
        for channel in packet.telemetry)) for packet in scenarios()]


def make_bundles(packets, kinetics=False):
    motion = CausalKinematics()
    provider = CalibratedTension({
        side: Calibration("test-" + side, "fixture_actuator_wire_A", 10., 0., .2, (0., 2.), side)
        for side in ("left", "right")
    })
    return [hypotheses(packet, motion.update(packet),
                       provider.estimate(packet) if kinetics else (), kinetics=kinetics)
            for packet in packets]


MODEL_FACTORIES = [LinearJoint, lambda: LinearJoint(False), MLPJoint, HypothesisPoolMLP]
ALL_FACTORIES = MODEL_FACTORIES + [SimpleRules, lambda: SimpleRules(True), lambda: SimpleRules(True, True)]


def test_disabled_kinetics_provider_is_never_called():
    rows = Replay(kinetics=ForbiddenProvider()).run(scenarios())
    assert all(row["kinetics_status"] == "disabled_by_configuration" for row in rows)
    assert all(load["value_N"] is None for row in rows for load in row["kinetics"])
    assert all(load["reason"] == "kinetics_deferred_by_motion_only_configuration"
               for row in rows for load in row["kinetics"])
    with pytest.raises(AssertionError, match="disabled kinetics provider was called"):
        Replay(Config(allow_kinetics_features=True), kinetics=ForbiddenProvider()).run(scenarios())


def test_telemetry_does_not_change_features_hypotheses_or_training_source():
    baseline = make_bundles(scenarios())
    changed = make_bundles(varied_packets())
    missing = make_bundles([replace(packet, telemetry=()) for packet in scenarios()])
    for original, altered, absent in zip(baseline, changed, missing):
        for candidate in (altered, absent):
            assert original.hypotheses == candidate.hypotheses
            np.testing.assert_array_equal(original.features, candidate.features)
            np.testing.assert_array_equal(original.weights, candidate.weights)
            assert candidate.source_kinds == ("synthetic",)
            assert candidate.feature_mode == "motion_only"
            assert not candidate.features[:, KINETIC_FEATURE_COLUMNS].any()


@pytest.mark.parametrize("factory", ALL_FACTORIES)
@pytest.mark.parametrize("task_gate", [False, True])
def test_telemetry_changes_do_not_change_paired_decisions(factory, task_gate):
    model = factory()
    if hasattr(model, "fit_synthetic"):
        model.fit_synthetic(make_bundles(scenarios())[:3], [0, 7, 7], steps=15)
    engine = Replay(Config(task_gate=task_gate), model=model, kinetics=ForbiddenProvider())
    baseline = engine.run(scenarios())
    altered = engine.run(varied_packets())
    absent = engine.run([replace(packet, telemetry=()) for packet in scenarios()])
    for original, changed, missing in zip(baseline, altered, absent):
        for candidate in (changed, missing):
            for key in ("kinematics", "kinetics", "hypotheses", "joint_scores", "outputs", "reused_sample_ids"):
                assert canonical(original[key]) == canonical(candidate[key])
            assert candidate["feature_mode"] == "motion_only"
        assert changed["inputs"]["telemetry"]
        assert changed["ignored_telemetry_sample_ids"] == ["reused-telemetry", "reused-telemetry"]
        assert missing["ignored_telemetry_sample_ids"] == []


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
def test_synthetic_training_is_identical_despite_ignored_recorded_telemetry(factory):
    original = factory()
    altered = factory()
    original.fit_synthetic(make_bundles(scenarios())[:3], [0, 7, 7], steps=15)
    altered.fit_synthetic(make_bundles(varied_packets())[:3], [0, 7, 7], steps=15)
    for bundle in make_bundles(scenarios()):
        np.testing.assert_array_equal(original.predict(bundle), altered.predict(bundle))
    assert altered.training_feature_mode == "motion_only"


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
@pytest.mark.parametrize("training_kinetics", [False, True])
def test_model_cannot_silently_change_feature_mode(factory, training_kinetics):
    model = factory()
    training = make_bundles(scenarios(), kinetics=training_kinetics)[:3]
    incompatible = make_bundles(scenarios(), kinetics=not training_kinetics)[:3]
    model.fit_synthetic(training, [0, 7, 7], steps=2)
    with pytest.raises(ValueError, match="feature mode differs"):
        model.predict(incompatible[0])
    with pytest.raises(ValueError, match="feature mode differs"):
        model.fit_synthetic(incompatible, [0, 7, 7], steps=2)


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
def test_mixed_feature_modes_cannot_enter_training(factory):
    motion = make_bundles(scenarios())[0]
    kinetic = make_bundles(scenarios(), kinetics=True)[0]
    with pytest.raises(ValueError, match="one explicit feature mode"):
        factory().fit_synthetic([motion, kinetic], [0, 0], steps=1)


@pytest.mark.parametrize("column", KINETIC_FEATURE_COLUMNS)
def test_motion_only_bundle_rejects_injected_load_value_or_validity(column):
    bundle = make_bundles(scenarios())[0]
    features = bundle.features.copy()
    features[0, column] = 1.
    with pytest.raises(ValueError, match="cannot contain kinetic"):
        replace(bundle, features=features)


def test_ignored_telemetry_does_not_hide_real_visual_training_provenance():
    packet = scenarios()[0]
    hands = tuple(replace(point, stamp=replace(point.stamp, source=Source.RECORDED))
                  for point in packet.hands)
    bundle = make_bundles([replace(packet, hands=hands)])[0]
    assert "recorded" in bundle.source_kinds
    with pytest.raises(ValueError, match="rejects recorded/measured"):
        LinearJoint().fit_synthetic([bundle], [0], steps=1)


def test_enabled_kinetics_still_exposes_load_and_provenance():
    original = make_bundles(scenarios(), kinetics=True)[2]
    altered = make_bundles(varied_packets(), kinetics=True)[2]
    assert original.features[:, KINETIC_FEATURE_COLUMNS].any()
    assert not altered.features[:, KINETIC_FEATURE_COLUMNS].any()
    assert "recorded" in altered.source_kinds


def test_motion_trace_records_training_mode_and_remains_causal():
    model = LinearJoint()
    model.fit_synthetic(make_bundles(scenarios())[:3], [0, 7, 7], steps=2)
    engine = Replay(model=model)
    full = engine.run(scenarios())
    assert canonical(full[:4]) == canonical(engine.run(scenarios()[:4]))
    assert all(row["model_training_feature_mode"] == "motion_only" for row in full)
    assert all(output["physical_command"] is None for row in full for output in row["outputs"])
