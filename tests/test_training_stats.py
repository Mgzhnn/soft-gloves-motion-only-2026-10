"""Recorded-data training path, person-level splits and participant-level accounting."""
from dataclasses import replace

import numpy as np
import pytest

from sg_offline.contracts import Source
from sg_offline.evaluation import Opportunity, evaluate_opportunities
from sg_offline.fixtures import scenarios
from sg_offline.inference import HypothesisPoolMLP, LinearJoint, MLPJoint, SimpleRules
from sg_offline.replay import Replay
from sg_offline.stats import paired_contrast, per_participant_success, unwanted_exposure
from sg_offline.training import (development_split, fit_recorded, label_ticks, leave_one_person_out,
                                 mean_log_loss, pair_index)


def recorded_packets(person="p1", session="s1", offset=0.):
    """Fixture scenarios re-stamped as RECORDED observations of one person/session."""
    out = []
    for packet in scenarios():
        def restamp(pt):
            return replace(pt, stamp=replace(pt.stamp, source=Source.RECORDED, sample_id=f"{person}:{pt.stamp.sample_id}",
                                             acquired_s=pt.stamp.acquired_s + offset, available_s=pt.stamp.available_s + offset))
        out.append(replace(packet, packet_id=f"{person}-{packet.packet_id}", decision_s=packet.decision_s + offset,
                           hands=tuple(restamp(p) for p in packet.hands), objects=tuple(restamp(p) for p in packet.objects),
                           telemetry=(), participant_id=person, session_id=session))
    return out


def ledger_for(person, session, offset=0.):
    return [Opportunity(f"{person}-g", person, session, .5 + offset, 1.0 + offset, (("G",), ("NQ",)), "test"),
            Opportunity(f"{person}-nq", person, session, 1.05 + offset, 1.35 + offset, (("NQ",), ("NQ",)), "test")]


def test_label_ticks_excludes_unknown_ambiguous_and_out_of_scope():
    packets = recorded_packets()
    # fixture ticks sit at 0.5, 0.6, ..., 1.3 s
    ops = [Opportunity("p1-g", "p1", "s1", .5, .8, (("G",), ("NQ",)), "test"),
           Opportunity("p1-nq", "p1", "s1", .85, 1.0, (("NQ",), ("NQ",)), "test"),
           Opportunity("p1-unk", "p1", "s1", 1.05, 1.15, (None, ("NQ",)), "test"),
           Opportunity("p1-amb", "p1", "s1", 1.2, 1.3, (("G", "R"), ("NQ",)), "test")]
    examples, counts = label_ticks(packets, ops)
    assert counts["ticks"] == 9
    assert counts["labelled"] == len(examples) == 9 - counts["outside_any_interval"] - counts["unknown_reference"]
    assert counts["partial_reference"] >= 1 and counts["ambiguous_alternatives"] >= 1
    labels = {e.label for e in examples}
    assert pair_index("G", "NQ") in labels and any(isinstance(y, tuple) and len(y)==7 for y in labels)
    assert all(e.bundle.source_kinds == ("recorded",) for e in examples)
    assert all(e.provenance.startswith("tick_inside_interval") for e in examples)


def test_overlapping_intervals_are_not_labelled():
    packets = recorded_packets()
    ops = [Opportunity("a", "p1", "s1", .5, 1.0, (("G",), ("NQ",)), "t"),
           Opportunity("b", "p1", "s1", .8, 1.2, (("NQ",), ("NQ",)), "t")]
    examples, counts = label_ticks(packets, ops)
    assert counts["overlapping_intervals"] >= 1
    assert all(not (.8 <= e.decision_s < 1.0) for e in examples)


@pytest.mark.parametrize("factory", [LinearJoint, MLPJoint, HypothesisPoolMLP])
def test_fit_recorded_refuses_synthetic_and_records_provenance(factory):
    examples, _ = label_ticks(recorded_packets(), ledger_for("p1", "s1"))
    model = factory()
    before = mean_log_loss(model, examples)
    fit_recorded(model, examples, split_id="test-split", steps=40)
    assert mean_log_loss(model, examples) < before
    assert model.fit_provenance.startswith("recorded_ledger_labels|participants=p1|n=")
    assert "split=test-split" in model.fit_provenance
    synthetic_examples = [replace(e, bundle=replace(e.bundle, source_kinds=("synthetic",))) for e in examples]
    with pytest.raises(ValueError):
        fit_recorded(factory(), synthetic_examples, split_id="x")
    with pytest.raises(ValueError):
        fit_recorded(SimpleRules(), examples, split_id="x")  # rules have no parameters


def test_leave_one_person_out_keeps_all_sessions_of_a_person_together():
    ex = []
    for person, off in (("p1", 0.), ("p2", 10.), ("p3", 20.)):
        for session, soff in (("s1", 0.), ("s2", 5.)):
            e, _ = label_ticks(recorded_packets(person, session, off + soff), ledger_for(person, session, off + soff))
            ex.extend(e)
    splits = list(leave_one_person_out(tuple(ex)))
    assert [held for held, _, _ in splits] == ["p1", "p2", "p3"]
    for held, train, test in splits:
        assert {e.participant_id for e in test} == {held}
        assert held not in {e.participant_id for e in train}
        assert {e.session_id for e in test} == {"s1", "s2"}
        inner_train, dev = development_split(train)
        assert {e.participant_id for e in dev}.isdisjoint({e.participant_id for e in inner_train})
        assert held not in {e.participant_id for e in dev}
    with pytest.raises(ValueError):
        development_split(tuple(e for e in ex if e.participant_id == "p1"))


def test_recorded_fit_changes_replay_provenance():
    examples, _ = label_ticks(recorded_packets(), ledger_for("p1", "s1"))
    model = fit_recorded(LinearJoint(), examples, split_id="lopo:p1", steps=10)
    rows = Replay(model=model).run(recorded_packets())
    assert rows[0]["model_fit"].startswith("recorded_ledger_labels")
    assert rows[0]["model"] == "relational_mixture"


def test_participant_level_accounting_and_paired_contrast():
    rows = {p: Replay().run(recorded_packets(p, "s1", off)) for p, off in (("p1", 0.), ("p2", 10.))}
    ops = {p: ledger_for(p, "s1", off) for p, off in (("p1", 0.), ("p2", 10.))}
    outcomes = [o for p in rows for o in evaluate_opportunities(rows[p], ops[p])]
    per = per_participant_success(outcomes)
    assert set(per) == {"p1", "p2"}
    for c in per.values():
        assert c["evaluable_request_hands"] == 1 and c["no_request_hands"] == 3
        assert c["success_fraction"] in (0., 1.)
    exp = unwanted_exposure(outcomes, [o for p in ops for o in ops[p]])
    assert exp["p1"]["no_request_hand_intervals"] == 3 and abs(exp["p1"]["exposure_hand_s"] - 1.1) < 1e-9
    other = {p: dict(c, success_fraction=0.) for p, c in per.items()}
    contrast = paired_contrast(per, other)
    assert contrast["n_people"] == 2 and all(d is not None for d in contrast["per_person"].values())
    assert "no interval" in contrast["scope"]
    empty = paired_contrast(per, {"p9": {"success_fraction": 1.}})
    assert empty["n_people"] == 0 and empty["mean_difference"] is None
