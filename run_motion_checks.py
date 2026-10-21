"""Reproduce motion-only synthetic replay and matching training checks."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from sg_offline.evaluation import summarize
from sg_offline.fixtures import scenarios
from sg_offline.inference import FactorizedJoint, HardCollapseLinear, HypothesisPoolMLP, LinearJoint, MLPJoint, SimpleRules, hypotheses
from sg_offline.inputs import packets_from_jsonl
from sg_offline.replay import IMPLEMENTATION_VERSION, Config, Replay, canonical, trace_hash, write_trace
from sg_offline.state import CausalKinematics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "experiments/results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = Config.read(Path(__file__).with_name("config.json"))
    if config.allow_kinetics_features:
        raise ValueError("this runner requires motion-only configuration")
    packets = [replace(packet, telemetry=()) for packet in scenarios()]
    observation_path = args.output_dir / "synthetic_motion_observations.jsonl"
    observation_path.write_text("".join(canonical(asdict(packet)) + "\n" for packet in packets))
    decoded = list(packets_from_jsonl(observation_path))
    rows = Replay(config).run(decoded)
    assert trace_hash(rows) == trace_hash(Replay(config).run(decoded))
    assert canonical(rows[:4]) == canonical(Replay(config).run(decoded[:4]))
    write_trace(args.output_dir / "synthetic_motion_trace.jsonl", rows)
    kinematics = CausalKinematics()
    bundles = [hypotheses(packet, kinematics.update(packet), (), kinetics=False)
               for packet in decoded]
    models = [SimpleRules(), SimpleRules(True), FactorizedJoint(), FactorizedJoint(cross_hand=True, coupling=True),
              LinearJoint(False), HardCollapseLinear(), LinearJoint(), MLPJoint(), HypothesisPoolMLP()]
    comparisons = []
    for model in models:
        if hasattr(model, "fit_synthetic"):
            model.fit_synthetic(bundles[:3], [0, 7, 7], steps=30)
        for task_gate in (False, True):
            trace = Replay(replace(config, task_gate=task_gate), model=model).run(decoded)
            arm = f"{model.name}_gate{int(task_gate)}"
            write_trace(args.output_dir / f"{arm}.jsonl", trace)
            comparisons.append({"arm": arm, "feature_mode": "motion_only",
                                "training_feature_mode": getattr(model, "training_feature_mode", None),
                                "trace_hash": trace_hash(trace), "generalization_measured": False})
    result = {"implementation": IMPLEMENTATION_VERSION, "source": "synthetic_only",
              "camera_assumption": "fixed_shared_frame_fixture_not_eye_camera_validation",
              "participant_N": 0, "population": "not_applicable_synthetic",
              "feature_mode": "motion_only", "kinetics_provider_called": False,
              "deterministic_repeat": True, "causal_prefix": True,
              "trace_hash": trace_hash(rows), "comparisons": comparisons, **summarize(rows)}
    (args.output_dir / "motion_only_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
