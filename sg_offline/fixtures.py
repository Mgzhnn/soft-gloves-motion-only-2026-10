"""Author-designed geometric scenarios; not recorded glove observations or measurements."""
from .contracts import Stamp, Source, Point, RawTension, Packet


def make_packet(index, left, right, objects=(("container", (0., 0., 1.)),), raw=(None, None),
                stale_left=False, previous_commands=("NQ", "NQ"), previous_command_times=(None, None)):
    t = index * .1 + .5
    def point(identity, xyz, stale=False):
        stamp = Stamp(f"p{index}:{identity}", t-.4 if stale else t, t, Source.SYNTHETIC)
        return Point(identity, xyz, stamp, sigma_m=.002 if xyz is not None else None,
                     reason="synthetic_occlusion" if xyz is None else "",
                     frame="synthetic_fixed", frame_is_fixed=True)
    hands = (point("left", left, stale_left), point("right", right))
    obj = tuple(point(k, xyz) for k, xyz in objects)
    telemetry = tuple(RawTension(side, "fixture_actuator_wire_A", value,
                                Stamp(f"t{index}:{side}", t, t, Source.SYNTHETIC),
                                reason="synthetic_missing_channel" if value is None else "")
                      for side, value in zip(("left", "right"), raw))
    return Packet(f"fixture-{index}", t, hands, obj, telemetry,
                  previous_commands=previous_commands, previous_command_times_s=previous_command_times)


def scenarios():
    return [
        make_packet(0, (-.2, 0., 1.), (.5, 0., 1.)),
        make_packet(1, (-.1, 0., 1.), (.5, 0., 1.)),  # asynchronous pre-contact approach
        make_packet(2, (-.06, 0., 1.), (.4, 0., 1.), raw=(1., None)),
        make_packet(3, (-.05, 0., 1.), (.2, 0., 1.), raw=(1., 1.)),
        make_packet(4, None, (.1, 0., 1.), raw=(1., 1.)),  # one hand missing, other usable
        make_packet(5, (-.03, 0., 1.), (.08, 0., 1.), raw=(1., 1.), stale_left=True),
        make_packet(6, (.04, 0., 1.), (-.05, 0., 1.), raw=(None, None)),  # crossing, named identities retained
        make_packet(7, (.08, 0., 1.), (-.15, 0., 1.),
                    objects=(("container", (0., 0., 1.)), ("candidate_part", (.12, 0., 1.))),
                    raw=(.5, .5)),  # a second observation is NOT a confirmed split
        make_packet(8, (.08, 0., 1.), (-.15, 0., 1.), raw=(1.5, 1.5)),  # load alone, no new motion
    ]
