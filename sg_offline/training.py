"""Training the small readouts on RECORDED data with ledger labels, under person-level splits.

This is the recorded-data training path. It is deliberately separate from ``fit_synthetic``:

* labels come only from an opportunity ledger (``ledger.py``), never from packets;
* labels are per hand and use half-open intervals plus declared hand windows; unknown
  hands are marginalised, alternatives are summed, and unknown is never changed to NQ;
* every example carries participant/session identity, so splits are by person and all of a
  person's material stays on one side;
* the fitted model records its provenance (source kinds, participants, example count, split id).

Nothing here claims the labels are correct or the fit generalises; those are results of an evaluation on real data.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .contracts import OUTCOMES, Packet
from .evaluation import Opportunity
from .inference import FEATURE_MODES, Bundle, hypotheses, validate_model_mode, label_indices, layout_of
from .replay import Config
from .scene import PlaneGeometry
from .state import CausalKinematics, DisabledKinetics, RelativeGeometry, UnavailableKinetics

LABEL_RULE = "tick_inside_interval_per_hand_window_marginal_likelihood_v2"


@dataclass(frozen=True)
class Example:
    bundle: Bundle
    label: int | tuple[int, ...]
    participant_id: str
    session_id: str
    packet_id: str
    decision_s: float
    opportunity_id: str
    provenance: str


def pair_index(left: str, right: str) -> int:
    return OUTCOMES.index(left) * 7 + OUTCOMES.index(right)


def bundles_for(packets, config: Config = Config(), scenes=None):
    """Replay the causal feature pipeline (no model, no supervisor) and yield (packet, bundle).
    ``scenes`` (``packet_id -> Scene``) is required under feature layout v2, exactly as for ``Replay``."""
    kin = CausalKinematics(config.max_velocity_gap_s)
    rel = RelativeGeometry(config.max_velocity_gap_s, config.rate_window)
    kinetics = UnavailableKinetics() if config.allow_kinetics_features else DisabledKinetics()
    anchored = layout_of(config.feature_layout).cues_per_hand > 0
    if anchored and scenes is None:
        raise ValueError("feature layout v2 needs scene records (scene.jsonl)")
    plane = PlaneGeometry(config.max_velocity_gap_s, config.plane_window, reversal_deadband_mps=config.reversal_deadband_mps) if anchored else None
    last = -1.; session = None; packet_ids = set(); samples = {}
    for packet in packets:
        if not isinstance(packet, Packet) or packet.decision_s <= last or packet.packet_id in packet_ids:
            raise ValueError("strict ordered unique decision packets required")
        if session is not None and (packet.participant_id, packet.session_id, packet.clock) != session:
            raise ValueError("one session and clock per bundle stream; reset between sessions")
        active_records = packet.hands + packet.objects
        if config.allow_kinetics_features:
            active_records += packet.telemetry
        for record in active_records:
            entity = getattr(record, "identity", getattr(record, "side", ""))
            key = (type(record).__name__, entity, record.stamp.sample_id)
            if key in samples and samples[key] != record:
                raise ValueError("sample identity reused for a changed observation")
            samples[key] = record
        packet_ids.add(packet.packet_id)
        session = (packet.participant_id, packet.session_id, packet.clock); last = packet.decision_s
        motions = kin.update(packet)
        relatives = rel.update(packet)
        loads = kinetics.estimate(packet)
        cues = plane.update(packet, scenes.get(packet.packet_id)) if anchored else None
        yield packet, hypotheses(packet, motions, loads, config.allow_kinetics_features,
                                 config.max_age_s, config.role_mode, relatives, cues, config.feature_layout)


def label_ticks(packets, opportunities, config: Config = Config(), scenes=None):
    """Join bundles with in-scope ledger intervals; return (examples, accounting)."""
    ops = [o for o in opportunities if isinstance(o, Opportunity)]
    keys = [(o.participant_id, o.session_id, o.opportunity_id) for o in ops]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate opportunity")
    counts = dict(ticks=0, labelled=0, outside_any_interval=0, unknown_reference=0,
                  ambiguous_alternatives=0, overlapping_intervals=0, partial_reference=0,
                  outside_hand_window=0)
    examples = []
    for packet, bundle in bundles_for(packets, config, scenes):
        counts["ticks"] += 1
        hits = [o for o in ops if o.participant_id == packet.participant_id and o.session_id == packet.session_id
                and o.start_s <= packet.decision_s < o.end_s]
        if not hits:
            counts["outside_any_interval"] += 1; continue
        if len(hits) > 1:
            counts["overlapping_intervals"] += 1; continue
        op = hits[0]
        allowed = list(op.acceptable)
        if op.hand_windows_s:
            for i, (start, end) in enumerate(op.hand_windows_s):
                if allowed[i] is not None and allowed[i] != ("NQ",) and not start <= packet.decision_s < end:
                    allowed[i] = None
                    counts["outside_hand_window"] += 1
        if all(a is None for a in allowed):
            counts["unknown_reference"] += 1; continue
        counts["partial_reference"] += int(any(a is None for a in allowed))
        counts["ambiguous_alternatives"] += int(any(a is not None and len(a) > 1 for a in allowed))
        valid = tuple(pair_index(l, r) for l in (allowed[0] or OUTCOMES)
                      for r in (allowed[1] or OUTCOMES)
                      if op.acceptable_joint_pairs is None or (l, r) in op.acceptable_joint_pairs
                      or any(a is None for a in allowed))
        label = valid[0] if len(valid) == 1 else valid
        examples.append(Example(bundle, label, packet.participant_id, packet.session_id, packet.packet_id,
                                packet.decision_s, op.opportunity_id, f"{LABEL_RULE}|{op.provenance}"))
        counts["labelled"] += 1
    return tuple(examples), counts


def validate_recorded_training(examples, allow_measured=True):
    if not examples:
        raise ValueError("no labelled recorded examples")
    allowed = {"recorded", "measured"} if allow_measured else {"recorded"}
    for e in examples:
        if not isinstance(e, Example):
            raise ValueError("labelled examples required")
        label_indices(e.label)
        if not set(e.bundle.source_kinds) <= allowed:
            raise ValueError("recorded trainer rejects synthetic (or disallowed) source bundles")
    modes = {e.bundle.feature_mode for e in examples}
    if len(modes) != 1 or not modes <= set(FEATURE_MODES):
        raise ValueError("training requires one explicit feature mode")


def fit_recorded(model, examples, split_id: str, steps=200, lr=.02, regularization=.001, callback=None):
    """Fit a readout on recorded examples; refuses synthetic bundles and mixed modes.
    ``callback(pass_number)`` runs after every full-batch pass (used for development early stopping)."""
    validate_recorded_training(examples)
    if not hasattr(model, "_fit"):
        raise ValueError("model has no trainable parameters")
    for e in examples:
        validate_model_mode(model, e.bundle)
    bundles = [e.bundle for e in examples]; labels = [e.label for e in examples]
    model._fit(bundles, labels, steps, lr, regularization, callback=callback)
    people = sorted({e.participant_id for e in examples})
    model.fit_provenance = (f"recorded_ledger_labels|participants={','.join(people)}|n={len(examples)}"
                            f"|split={split_id}|rule={LABEL_RULE}")
    return model


def participants(examples):
    return sorted({e.participant_id for e in examples})


def leave_one_person_out(examples):
    """Outer splits: each person held out once; all of that person's sessions are in the test side."""
    for held in participants(examples):
        train = tuple(e for e in examples if e.participant_id != held)
        test = tuple(e for e in examples if e.participant_id == held)
        yield held, train, test


def development_split(train_examples, dev_participant=None):
    """Inner split by person for model/threshold selection; never uses the held-out person."""
    people = participants(train_examples)
    if len(people) < 2:
        raise ValueError("development split needs at least two training participants")
    dev = dev_participant or people[-1]
    if dev not in people:
        raise ValueError("development participant must be a training participant")
    return (tuple(e for e in train_examples if e.participant_id != dev),
            tuple(e for e in train_examples if e.participant_id == dev))


def mean_log_loss(model, examples):
    losses = []
    if hasattr(model, "reset"):
        model.reset()
    for e in examples:
        if hasattr(model, "log_likelihood"):
            loss = -float(model.log_likelihood(e.bundle, e.label))
            if not isfinite(loss):
                raise ValueError("nonfinite log likelihood")
            losses.append(loss)
            continue
        if hasattr(model, "predict_window"):
            if not hasattr(e, "bundles"):
                raise ValueError("sequence loss requires complete causal Windows, including unlabelled history")
            prediction = model.predict_window(e.bundles)
        else:
            prediction = model.predict(e.bundle)
        p = float(np.asarray(prediction)[label_indices(e.label)].sum())
        if not isfinite(p):
            raise ValueError("nonfinite prediction")
        losses.append(-np.log(max(p, 1e-300)))
    return float(np.mean(losses)) if losses else float("nan")


def session_development_split(examples, dev_session=None):
    """Reserve whole sessions when the outer training fold has only one participant."""
    keys = sorted({(e.participant_id, e.session_id) for e in examples})
    if len(keys) < 2:
        raise ValueError("need a separate whole development session or independently fixed settings")
    dev = dev_session or keys[-1]
    if dev not in keys:
        raise ValueError("unknown development session")
    return (tuple(e for e in examples if (e.participant_id, e.session_id) != dev),
            tuple(e for e in examples if (e.participant_id, e.session_id) == dev))
