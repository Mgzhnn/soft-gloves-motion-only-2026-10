"""Deterministic paired proposals with a simulation-only sink. No hardware API."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from .contracts import SCHEMA, SIDES, OUTCOMES, Packet
from .state import CausalKinematics, RelativeGeometry, UnavailableKinetics, DisabledKinetics
from .inference import hypotheses, SimpleRules, hand_target_features, hand_action_cues, layout_of, cue
from .scene import PlaneGeometry

# The one place the build identifier lives: driver scripts with their own copies let them go stale and stamped
# them into summaries beside traces of a later build.
IMPLEMENTATION_VERSION = "motion-only-build/7"  # build 7: feature layouts, scene anchor, per-action gates
DECISION_RULES = ("best_action_threshold", "marginal_argmax")
# Per-action observability gates of the common supervisor under layout v2. Each action is proposed only when
# the cue it needs is observable at this tick; the reason names
# what was missing. Under layout v1 every non-grasp action keeps the build 1-6 reason, so stored traces reproduce.
ACTION_GATE_REASONS = dict(FP="functional_points_unavailable", AR="held_object_relation_unavailable",
                           NT="support_plane_unavailable", HP="second_object_or_support_plane_unavailable",
                           R="held_object_relation_unavailable")
HOLD_INPLANE_M, HOLD_OBJECT_HEIGHT_M = .07, .03   # "holding": the candidate object within reach in the plane and lifted


@dataclass(frozen=True)
class Config:
    schema: str = SCHEMA
    max_age_s: float = .25
    max_velocity_gap_s: float = .25
    proposal_score_threshold: float = .6
    task_gate: bool = False
    allow_kinetics_features: bool = False
    simulation_only: bool = True
    role_mode: str = "weak_speed"
    grasp_distance_limit_m: float | None = None
    # Past pairs in the frame-invariant distance-rate estimate: 1 = one-frame secant (original behaviour),
    # K > 1 = causal least-squares slope over up to K past pairs. One value per batch, chosen on development data.
    rate_window: int = 1
    # How the common supervisor turns a hand's marginal scores into a decision.
    # "best_action_threshold": the most probable non-NQ action is proposed when its score reaches the threshold,
    # otherwise the hand is no new request, so the threshold is a true operating point selectable on development
    # data under the burden budget. "marginal_argmax" is the rule of builds 1-5: the most probable outcome (NQ
    # included) and a withheld tick when its score is below the threshold; name it to reproduce older traces.
    decision_rule: str = "best_action_threshold"
    # Feature layout: "v1" = the 42 columns of every earlier record (default, bit-identical); "v2" adds the
    # per-action cue block from the scene anchor (table plane + landmarks). With "v2" the replay needs scene records.
    feature_layout: str = "v1"
    # Causal window of the plane-anchored rates (height rate, swept angle), in past pairs, like rate_window; and the
    # vertical-speed dead band under which a sign change does not count as a shake reversal (m/s).
    plane_window: int = 8
    reversal_deadband_mps: float = .1
    # Temporal recursion on the readout's joint scores (borrowed from the reference recursive filter,
    # prior = last posterior): p_t = normalise(p_model,t * p_{t-1}^lambda), reset to the model's own scores after a gap
    # longer than max_velocity_gap_s or at a session start. 0 (default) = no recursion, bit-identical to every earlier
    # record. Applied to EVERY arm identically before the supervisor, so it cannot confound the contrast; a declared
    # value is a plan setting, never tuned on test people.
    posterior_forgetting: float = 0.

    def __post_init__(self):
        if self.decision_rule not in DECISION_RULES:
            raise ValueError("unknown supervisor decision rule")
        layout_of(self.feature_layout)
        if type(self.plane_window) is not int or self.plane_window < 1:
            raise ValueError("plane_window must be a positive integer")
        if not 0 <= self.reversal_deadband_mps < float("inf"):
            raise ValueError("reversal_deadband_mps must be a finite nonnegative speed")
        if not 0 <= self.posterior_forgetting < 1:
            raise ValueError("posterior_forgetting must be in [0, 1): 0 = no recursion, values near 1 remember almost everything")
        if self.schema != SCHEMA or self.simulation_only is not True:
            raise ValueError("only this schema and simulation-only execution are supported")
        if not 0 < self.max_age_s < float("inf") or not 0 < self.max_velocity_gap_s < float("inf"):
            raise ValueError("positive finite age/gap bounds required")
        if type(self.rate_window) is not int or self.rate_window < 1:
            raise ValueError("rate_window must be a positive integer")
        if not 0 < self.proposal_score_threshold <= 1:
            raise ValueError("invalid demonstration score threshold")
        if type(self.task_gate) is not bool or type(self.allow_kinetics_features) is not bool:
            raise ValueError("flags must be boolean")
        if self.grasp_distance_limit_m is not None and not 0 < self.grasp_distance_limit_m < float("inf"):
            raise ValueError("grasp distance limit must be positive finite or None")
        if self.role_mode not in ("weak_speed", "neutral", "permuted"):
            raise ValueError("invalid independently selected role mode")

    @classmethod
    def read(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    @property
    def feature_mode(self):
        return "motion_and_kinetics" if self.allow_kinetics_features else "motion_only"


class SimulationSupervisor:
    def __init__(self, config):
        self.config = config

    def action_gate(self, outcome, bundle, hand, target):
        """"" when the action's cue is observable for this hand at this tick; otherwise the withholding reason."""
        if layout_of(bundle.layout).cues_per_hand == 0:
            return "action_specific_geometry_not_implemented"  # builds 1-6: no cue block exists under layout v1
        cues = hand_action_cues(bundle, hand)
        if cues is None:
            return "fresh_target_geometry_unavailable"  # no candidate object for this hand (admit() catches this earlier; kept explicit)
        if cue(cues, "height_valid") < .5:
            return "support_plane_unavailable"
        holding = (cue(cues, "inplane_distance") * .1 < HOLD_INPLANE_M and cue(cues, "object_height") * .2 > HOLD_OBJECT_HEIGHT_M
                   and target is not None and target[6] > .5)
        if outcome == "FP":
            return "" if cue(cues, "landmarks_valid") > .5 else ACTION_GATE_REASONS["FP"]
        if outcome == "NT":
            return "" if holding else ACTION_GATE_REASONS["AR"]
        if outcome == "HP":
            return "" if holding and cue(cues, "over_valid") > .5 else ("second_object_unavailable" if holding else ACTION_GATE_REASONS["AR"])
        return "" if holding else ACTION_GATE_REASONS[outcome]  # AR, R

    def admit(self, packet, probabilities, bundle):
        p = np.asarray(probabilities).reshape(7, 7)
        if not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1.):
            raise ValueError("joint model must return normalized finite nonnegative scores")
        marginals = (p.sum(axis=1), p.sum(axis=0))
        argmax = tuple(int(np.argmax(m)) for m in marginals)
        threshold = self.config.proposal_score_threshold
        if self.config.decision_rule == "marginal_argmax":
            # builds 1-5: the most probable outcome, withheld when its own score is below the threshold
            choice = argmax
            below = tuple(marginals[i][choice[i]] < threshold for i in range(2))
        else:
            # best_action_threshold: the threshold is the operating point on the best action's
            # score; below it the hand is no new request, never a withheld request
            best = tuple(1 + int(np.argmax(m[1:])) for m in marginals)
            choice = tuple(best[i] if marginals[i][best[i]] >= threshold else 0 for i in range(2))
            below = (False, False)
        shared = sum(h.weight for h in bundle.hypotheses
                     if h.object_ids[0] is not None and h.object_ids[0] == h.object_ids[1])
        # A pair exists only when both hands are observed and fresh: a hand that cannot act is not a partner
        # (otherwise its raw choice alone could withhold the other hand's grasp under the pair gate).
        usable = all(point.xyz is not None and packet.decision_s - point.stamp.acquired_s <= self.config.max_age_s for point in packet.hands)
        paired_grasp = usable and tuple(OUTCOMES[i] for i in choice) == ("G", "G")
        commands = []
        for i, (side, point) in enumerate(zip(SIDES, packet.hands)):
            outcome = OUTCOMES[choice[i]]
            target = hand_target_features(bundle, i)
            grasp_gate = (point.xyz is not None and packet.decision_s - point.stamp.acquired_s <= self.config.max_age_s
                          and target is not None and target[8] > .5)
            if self.config.grasp_distance_limit_m is not None:
                grasp_gate = grasp_gate and target[0]*.1 < self.config.grasp_distance_limit_m
            reason = ""
            if point.xyz is None:
                reason = "required_hand_observation_unavailable"
            elif packet.decision_s - point.stamp.acquired_s > self.config.max_age_s:
                reason = "stale_hand_observation"
            elif below[i]:
                reason = "insufficient_uncalibrated_score"
            elif outcome != "NQ" and not any(h.object_ids[i] is not None for h in bundle.hypotheses):
                reason = "fresh_target_geometry_unavailable"
            elif outcome == "G" and (target is None or target[8] <= .5):
                reason = "past_relative_approach_rate_unavailable_or_ambiguous"
            elif outcome == "G" and not grasp_gate:
                reason = "configured_grasp_distance_gate"
            elif outcome in ("FP", "AR", "NT", "HP", "R"):
                reason = self.action_gate(outcome, bundle, i, target)
            elif self.config.task_gate and paired_grasp and shared < .7:
                reason = "experimental_pair_gate_ambiguous_association"
            status = "withheld" if reason else "no_new_request" if outcome == "NQ" else "proposed_simulation_only"
            commands.append(dict(side=side, raw_outcome=outcome, argmax_outcome=OUTCOMES[argmax[i]],
                                 marginal_scores=marginals[i].tolist(),
                                 grasp_score=float(marginals[i][1]),
                                 best_action_score=float(marginals[i][1:].max()),
                                 grasp_gate_eligible=bool(grasp_gate),
                                 raw_grasp_above_threshold=bool(marginals[i][1] >= threshold),
                                 decision_rule=f"per_hand_{self.config.decision_rule}_then_common_supervisor",
                                 request_score=float(1-marginals[i][0]),
                                 outcome_score=float(marginals[i][choice[i]]),
                                 # None when a request is (numerically) excluded: no conditional distribution exists
                                 conditional_action_scores=((marginals[i][1:]/(1-marginals[i][0])).tolist()
                                                            if 1-marginals[i][0] > 1e-12 else None),
                                 status=status, reason=reason or "software_rule_only",
                                 physical_command=None))
        return commands


class Replay:
    def __init__(self, config=Config(), model=None, kinetics=None, scenes=None):
        """``scenes``: ``packet_id -> scene.Scene`` (from ``scene.read_scenes``); required with layout v2, ignored under v1
        except that a v1 replay given scenes still records nothing from them (features are bit-identical)."""
        self.config = config
        self.model = model or SimpleRules()
        self.kinetics = (kinetics or UnavailableKinetics()) if config.allow_kinetics_features else DisabledKinetics()
        self.scenes = scenes
        if layout_of(config.feature_layout).cues_per_hand and scenes is None:
            raise ValueError("feature layout v2 needs scene records (scene.jsonl): pass scenes=read_scenes(path)")

    def run(self, packets):
        trained_gap = getattr(self.model, "training_reset_gap_s", None)
        if trained_gap is not None and trained_gap != self.config.max_velocity_gap_s:
            raise ValueError("replay history-reset gap differs from recorded sequence training")
        if hasattr(self.model, "reset"):
            self.model.reset()
        kin = CausalKinematics(self.config.max_velocity_gap_s)
        rel = RelativeGeometry(self.config.max_velocity_gap_s, self.config.rate_window)
        anchored = layout_of(self.config.feature_layout).cues_per_hand > 0
        plane = PlaneGeometry(self.config.max_velocity_gap_s, self.config.plane_window, reversal_deadband_mps=self.config.reversal_deadband_mps) if anchored else None
        supervisor = SimulationSupervisor(self.config)
        last = -1.; packet_ids = set(); rows = []; samples = {}; session = None
        posterior = None  # the recursed joint scores of the previous tick (posterior_forgetting > 0)
        for packet in packets:
            if not isinstance(packet, Packet) or packet.decision_s <= last or packet.packet_id in packet_ids:
                raise ValueError("strict ordered unique decision packets required")
            if session is not None and (packet.participant_id, packet.session_id, packet.clock) != session:
                raise ValueError("new session or clock requires a separate replay/history reset")
            session = (packet.participant_id, packet.session_id, packet.clock)
            reused = []
            active_records = packet.hands + packet.objects
            if self.config.allow_kinetics_features:
                active_records += packet.telemetry
            for record in active_records:
                entity = getattr(record, "identity", getattr(record, "side", ""))
                key = (type(record).__name__, entity, record.stamp.sample_id)
                if key in samples:
                    if samples[key] != record:
                        raise ValueError("sample identity reused for a changed observation")
                    reused.append(record.stamp.sample_id)
                samples[key] = record
            if last >= 0 and packet.decision_s - last > self.config.max_velocity_gap_s:
                posterior = None  # a gap resets the recursion as it resets every causal history
                if hasattr(self.model, "reset"):
                    self.model.reset()
            last = packet.decision_s; packet_ids.add(packet.packet_id)
            motions = kin.update(packet)
            relatives = rel.update(packet)
            loads = self.kinetics.estimate(packet)
            cues = None
            if anchored:
                scene = self.scenes.get(packet.packet_id)
                cues = plane.update(packet, scene)
            bundle = hypotheses(packet, motions, loads, self.config.allow_kinetics_features,
                                self.config.max_age_s, self.config.role_mode, relatives, cues, self.config.feature_layout)
            p = self.model.predict(bundle)
            model_scores = None
            if self.config.posterior_forgetting > 0:
                model_scores = np.asarray(p, dtype=float).copy()
                if posterior is not None:
                    q = model_scores * np.power(posterior, self.config.posterior_forgetting)
                    p = q / q.sum()
                posterior = np.asarray(p, dtype=float)
            commands = supervisor.admit(packet, p, bundle)
            scene_record = None
            if anchored:
                scene_record = dict(plane_available=cues.plane_available, reason=cues.reason,
                                    heights={k: (None if v.height_m is None else round(v.height_m, 4)) for k, v in cues.heights.items()},
                                    height_rates={k: (None if v.rate_mps is None else round(v.rate_mps, 4)) for k, v in cues.heights.items()},
                                    landmarks_complete={s: (l is not None and l.complete) for s, l in cues.landmarks.items()})
            rows.append(dict(schema=SCHEMA, packet_id=packet.packet_id, decision_s=packet.decision_s,
                             implementation_version=IMPLEMENTATION_VERSION,
                             command_history_policy="external_past_only_no_oracle_feedback", configuration=asdict(self.config),
                             model=self.model.name, model_fit=getattr(self.model, "fit_provenance", "hand_authored_unvalidated"),
                             simulation_only=True, inputs=asdict(packet),
                             feature_mode=self.config.feature_mode,
                             model_training_feature_mode=getattr(self.model, "training_feature_mode", None),
                             kinetics_status="enabled_conditional_provider" if self.config.allow_kinetics_features else "disabled_by_configuration",
                             ignored_telemetry_sample_ids=([record.stamp.sample_id for record in packet.telemetry]
                                                           if not self.config.allow_kinetics_features else []),
                             reused_sample_ids=reused,
                             kinematics={k: asdict(v) for k, v in motions.items()},
                             relative_geometry={f"{k[0]}|{k[1]}": asdict(v) for k, v in relatives.items()},
                             kinetics=[asdict(l) for l in loads],
                             hypotheses=[asdict(h) for h in bundle.hypotheses],
                             joint_scores=np.asarray(p).tolist(), outputs=commands,
                             **({"scene": scene_record} if anchored else {}),
                             **({"model_scores": model_scores.tolist()} if model_scores is not None else {})))
        return rows


def canonical(rows):
    return json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)


def trace_hash(rows):
    return hashlib.sha256(canonical(rows).encode()).hexdigest()


def write_trace(path, rows):
    Path(path).write_text("".join(canonical(row)+"\n" for row in rows))
