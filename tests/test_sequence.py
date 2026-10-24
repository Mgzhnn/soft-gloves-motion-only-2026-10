"""Causal sequence readouts: shapes, causality, matched removal pairs, training path. Skipped without torch."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dataclasses import replace
from sg_offline.contracts import Source
from sg_offline.evaluation import Opportunity
from sg_offline.fixtures import scenarios
from sg_offline.replay import Replay
from sg_offline.sequence import KINDS, SequenceReadout, build_windows, sequence_examples
from sg_offline.training import bundles_for


def recorded_packets(person="p1", session="s1", offset=0.):
    out = []
    for packet in scenarios():
        def restamp(pt):
            return replace(pt, stamp=replace(pt.stamp, source=Source.RECORDED, sample_id=f"{person}:{pt.stamp.sample_id}",
                                             acquired_s=pt.stamp.acquired_s + offset, available_s=pt.stamp.available_s + offset))
        out.append(replace(packet, packet_id=f"{person}-{packet.packet_id}", decision_s=packet.decision_s + offset,
                           hands=tuple(restamp(p) for p in packet.hands), objects=tuple(restamp(p) for p in packet.objects),
                           telemetry=(), participant_id=person, session_id=session))
    return out


def synthetic_windows(window=5):
    return [w for _, w in build_windows(bundles_for(scenarios()), window)]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("retain", [True, False])
def test_predict_is_normalised_and_causal(kind, retain):
    model = SequenceReadout(kind, retain_hypotheses=retain, window=5, hidden=8)
    wins = synthetic_windows(5)
    p = model.predict_window(wins[3])
    assert p.shape == (49,) and abs(p.sum() - 1) < 1e-6 and (p >= 0).all()
    # causality: the prediction at tick 3 must not depend on ticks after 3
    stream = [b for _, b in bundles_for(scenarios())]
    p_prefix = model.predict_window(tuple(stream[:4]))
    assert np.allclose(p_prefix, p)
    moved = [replace(k, hands=tuple(replace(h, xyz=None if h.xyz is None else (h.xyz[0] + .3, h.xyz[1], h.xyz[2])) for h in k.hands))
             for k in scenarios()[4:]]
    altered = [w for _, w in build_windows(bundles_for(scenarios()[:4] + moved), 5)]
    assert not np.allclose(model.predict_window(altered[-1]), model.predict_window(wins[-1]))   # the change is visible later
    assert np.allclose(model.predict_window(altered[3]), p)                                     # but not at tick 3
    # stateful path equals the stateless window path tick by tick
    model.reset()
    for i, b in enumerate(stream[:4]):
        q = model.predict(b)
    assert np.allclose(q, p, atol=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_matched_pair_and_parameter_counts(kind):
    a = SequenceReadout(kind, retain_hypotheses=True, window=5, hidden=8)
    b = SequenceReadout(kind, retain_hypotheses=False, window=5, hidden=8)
    assert a.n_parameters() == b.n_parameters() > 0
    assert a.name.endswith("_set") and b.name.endswith("_pooled")


@pytest.mark.parametrize("kind", KINDS)
def test_synthetic_fit_reduces_loss_and_refuses_recorded(kind):
    model = SequenceReadout(kind, retain_hypotheses=True, window=5, hidden=8)
    wins = synthetic_windows(5)[:4]
    labels = [0, 0, 7, 7]
    def loss():
        return float(np.mean([-np.log(model.predict_window(w)[y]) for w, y in zip(wins, labels)]))
    before = loss()
    model.fit_synthetic(wins, labels, steps=60)
    assert loss() < before
    assert model.fit_provenance == "synthetic_only_no_reference_validation"
    rec = [w for _, w in build_windows(bundles_for(recorded_packets()), 5)][:4]
    with pytest.raises(ValueError):
        SequenceReadout(kind, window=5, hidden=8).fit_synthetic(rec, labels, steps=1)


def test_recorded_windows_and_fit_provenance():
    packets = recorded_packets()
    ops = [Opportunity("g", "p1", "s1", .5, .8, (("G",), ("NQ",)), "t"),
           Opportunity("nq", "p1", "s1", .85, 1.3, (("NQ",), ("NQ",)), "t")]
    wins, counts = sequence_examples(packets, ops, window=4)
    assert counts["labelled"] == len(wins) == 7
    assert all(len(w.bundles) <= 4 and w.bundles[-1].source_kinds == ("recorded",) for w in wins)
    model = SequenceReadout("gru", window=4, hidden=8).fit_recorded(wins, split_id="lopo:p1", steps=20)
    assert model.fit_provenance.startswith("recorded_ledger_labels|participants=p1|n=7|split=lopo:p1")
    assert "params=" in model.fit_provenance
    synthetic = [w for _, w in build_windows(bundles_for(scenarios()), 4)]
    from sg_offline.sequence import Window
    fake = [Window(w, 0, "p", "s", f"k{i}", i * .1, "o", "x") for i, w in enumerate(synthetic)]
    with pytest.raises(ValueError):
        SequenceReadout("gru", window=4, hidden=8).fit_recorded(fake, split_id="x", steps=1)


def test_sequence_model_runs_inside_replay_and_is_named():
    model = SequenceReadout("transformer", window=6, hidden=8)
    rows = Replay(model=model).run(recorded_packets())
    assert len(rows) == 9 and rows[0]["model"] == "seq_transformer_set" and rows[0]["model_fit"] == "untrained"
    assert all(abs(sum(r["joint_scores"]) - 1) < 1e-6 for r in rows)

@pytest.mark.parametrize('kind', KINDS)
def test_replay_resets_sequence_history_between_sessions(kind):
    from sg_offline.replay import canonical
    model=SequenceReadout(kind,window=5,hidden=8)
    replay=Replay(model=model)
    assert canonical(replay.run(scenarios()))==canonical(replay.run(scenarios()))


def test_capacity_preserves_nineteen_hypotheses_and_rejects_truncation():
    from sg_offline.sequence import _pad_set
    bundle=list(bundles_for(scenarios()))[7][1]
    f,w=_pad_set([bundle],19)
    assert np.allclose(w[0],bundle.weights)
    with pytest.raises(ValueError,match='truncation'):_pad_set([bundle],8)


@pytest.mark.parametrize('kind', KINDS)
def test_short_window_is_independent_of_batch_padding(kind):
    model=SequenceReadout(kind,window=8,hidden=8)
    wins=synthetic_windows(8)
    f,w,n=model._tensors([wins[0],wins[-1]])
    with torch.no_grad():p=model._forward(f,w,n)[0].numpy()
    assert np.allclose(p,model.predict_window(wins[0]),atol=1e-6)


def test_temporal_cnn_can_use_the_oldest_tick_in_full_window():
    model=SequenceReadout('tcn',window=15,hidden=8)
    b=list(bundles_for(scenarios()))[1][1]
    f,w,n=model._tensors([[b]*15]);f.requires_grad_()
    model._forward(f,w,n)[0,0].backward()
    assert model.receptive_field>=15 and f.grad[0,0].abs().sum()>0


@pytest.mark.parametrize('kind', KINDS)
def test_partial_reference_sequence_training_is_finite(kind):
    model=SequenceReadout(kind,window=5,hidden=8)
    wins=synthetic_windows(5)[:4];labels=[tuple(i*7+1 for i in range(7))]*4
    before=-np.log(model.predict_window(wins[-1]).reshape(7,7)[:,1].sum())
    model._fit(wins,labels,steps=20)
    after=-np.log(model.predict_window(wins[-1]).reshape(7,7)[:,1].sum())
    assert after<before and np.isfinite(after)
