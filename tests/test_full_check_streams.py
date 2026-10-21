"""Stream-contract regressions, synthetic offline software; participant N=0."""
from dataclasses import replace

import pytest

from sg_offline.contracts import Packet, Point, Source, Stamp
from sg_offline.replay import Config, Replay
from sg_offline.training import bundles_for


def packet(t, d, clock="one", packet_id=None, sample=None):
    def stamp(name):
        return Stamp(sample or name + str(t), t, t, Source.SYNTHETIC, clock)
    return Packet(packet_id or str(t), t,
                  (Point("left", (0., 0., 0.), stamp("left"), calibration_id="cal"),
                   Point("right", None, stamp("right"), reason="out_of_frame", calibration_id="cal")),
                  (Point("cup", (d, 0., 0.), stamp("cup"), calibration_id="cal"),), clock=clock)


@pytest.mark.parametrize("route", ["replay", "training"])
def test_a_session_cannot_change_clock_domains(route):
    packets = [packet(1., .10), packet(1.1, .09, "other_domain")]
    with pytest.raises(ValueError, match="clock"):
        Replay().run(packets) if route == "replay" else list(bundles_for(packets))


@pytest.mark.parametrize("fault", ["packet", "sample"])
def test_training_refuses_streams_that_replay_rejects(fault):
    kw = {"packet_id": "duplicate"} if fault == "packet" else {"sample": "reused"}
    packets = [packet(1., .1, **kw), packet(1.1, .09, **kw)]
    with pytest.raises(ValueError):
        Replay().run(packets)
    with pytest.raises(ValueError):
        list(bundles_for(packets))


@pytest.mark.parametrize("kind", ["gru", "transformer", "tcn"])
def test_recorded_sequence_replay_requires_the_training_reset_gap(kind):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from sg_offline.evaluation import Opportunity
    from sg_offline.sequence import SequenceReadout, sequence_examples
    packets = []
    for t, d in [(1., .1), (1.1, .09), (1.2, .08)]:
        p = packet(t, d)
        def recorded(point):
            return replace(point, stamp=replace(point.stamp, source=Source.RECORDED))
        packets.append(replace(p, hands=tuple(map(recorded, p.hands)), objects=tuple(map(recorded, p.objects))))
    op = Opportunity("nq", packets[0].participant_id, packets[0].session_id, 0., 2.,
                     (("NQ",), ("NQ",)), "software fixture")
    config = Config(max_velocity_gap_s=.05)
    examples, _ = sequence_examples(packets, [op], 3, config)
    model = SequenceReadout(kind, window=3, hidden=4, seed=9)
    model.fit_recorded(examples, "software-test", steps=1)
    assert len(Replay(config, model).run(packets)) == 3
    with pytest.raises(ValueError, match="gap"):
        Replay(Config(max_velocity_gap_s=.25), model).run(packets)
