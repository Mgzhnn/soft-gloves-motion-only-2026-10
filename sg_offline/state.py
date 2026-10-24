"""Causal motion and conditional calibration. None of these map commands to force."""
from dataclasses import dataclass
from math import sqrt
import numpy as np
from .contracts import Packet, Point, LoadEstimate, SIDES, Source


@dataclass(frozen=True)
class Motion:
    point: Point
    velocity_mps: tuple[float, float, float] | None
    age_s: float
    velocity_reason: str


class CausalKinematics:
    def __init__(self, max_gap_s=0.25):
        self.previous: dict[str, Point] = {}
        self.max_gap_s = max_gap_s

    def update(self, packet: Packet) -> dict[str, Motion]:
        result = {}
        for point in packet.hands + packet.objects:
            prior = self.previous.get(point.identity)
            v = None
            reason = "no_valid_past_pair"
            if point.xyz is not None and prior is not None and prior.xyz is not None:
                dt = point.stamp.acquired_s - prior.stamp.acquired_s
                if dt <= 0:
                    reason = "nonadvancing_acquisition"
                elif point.frame != prior.frame:
                    reason = "frame_changed"
                elif not point.frame_is_fixed or not prior.frame_is_fixed:
                    reason = "moving_or_unverified_reference_frame"
                elif point.calibration_id != prior.calibration_id:
                    reason = "calibration_changed"
                elif point.definition != prior.definition:
                    reason = "point_definition_changed"
                elif dt > self.max_gap_s:
                    reason = "gap_exceeded"
                else:
                    v = tuple(float(x) for x in (np.asarray(point.xyz) - prior.xyz) / dt)
                    reason = "past_secant_not_instantaneous_velocity"
            result[point.identity] = Motion(point, v, packet.decision_s - point.stamp.acquired_s, reason)
            # A delayed old observation may be logged but cannot overwrite current history.
            if prior is None or point.stamp.acquired_s > prior.stamp.acquired_s:
                self.previous[point.identity] = point
        return result


@dataclass(frozen=True)
class Relative:
    hand: str
    object_id: str
    distance_m: float | None
    rate_mps: float | None  # d(distance)/dt over the past pair; positive when separating
    age_s: float
    reason: str
    source: str = ""


class RelativeGeometry:
    """Same-time hand-object distance and its past rate without any fixed frame.

    The Euclidean distance between two points observed at the same instant in the same
    calibrated camera frame is invariant to rigid camera motion, so its rate over a bounded
    gap is admissible even while directional velocity must be withheld.

    ``window=1`` (default) is the past secant over the last admissible pair, bit-for-bit the
    original behaviour. ``window=K>1`` fits a causal least-squares slope over up to K past pairs
    plus the current one, all chained within ``max_gap_s`` and under one calibration/point
    definition; it exists because the one-frame secant is noise-dominated at 15-30 fps relative
    to the -4 mm/s rule threshold. The window is a batch setting chosen
    once on development data (``replay.Config.rate_window``), never per clip."""

    def __init__(self, max_gap_s=0.25, window=1):
        if type(window) is not int or window < 1:
            raise ValueError("rate window must be a positive integer number of past pairs")
        # key -> chain of (distance, acquired_s, calibrations, definitions), oldest first, newest last
        self.previous: dict[tuple[str, str], list[tuple[float, float, tuple[str, str], tuple[str, str]]]] = {}
        self.max_gap_s = max_gap_s
        self.window = window

    def update(self, packet: Packet) -> dict[tuple[str, str], Relative]:
        result = {}
        for hand in packet.hands:
            for obj in packet.objects:
                key = (hand.identity, obj.identity)
                age = packet.decision_s - hand.stamp.acquired_s
                if hand.xyz is None or obj.xyz is None:
                    result[key] = Relative(*key, None, None, age, "point_unavailable"); continue
                if hand.frame != obj.frame or hand.stamp.clock != obj.stamp.clock:
                    result[key] = Relative(*key, None, None, age, "frame_or_clock_mismatch"); continue
                if hand.stamp.acquired_s != obj.stamp.acquired_s:
                    result[key] = Relative(*key, None, None, age, "asynchronous_points"); continue
                d = distance(hand.xyz, obj.xyz)
                t = hand.stamp.acquired_s
                defs = (hand.definition, obj.definition)
                cals = (hand.calibration_id, obj.calibration_id)
                chain = self.previous.get(key, [])
                rate, reason, chained = None, "no_valid_past_pair", False
                if chain:
                    d0, t0, cals0, defs0 = chain[-1]
                    dt = t - t0
                    if dt <= 0:
                        reason = "nonadvancing_acquisition"
                    elif cals != cals0:
                        reason = "calibration_changed"
                    elif defs != defs0:
                        reason = "point_definition_changed"
                    elif dt > self.max_gap_s:
                        reason = "gap_exceeded"
                    else:
                        chained = True
                        points = (chain + [(d, t, cals, defs)])[-(self.window + 1):]
                        if len(points) == 2:
                            rate, reason = (d - d0) / dt, "past_secant_distance_rate_frame_invariant"
                        else:
                            ts = np.array([q[1] for q in points]); ds = np.array([q[0] for q in points])
                            tc = ts - ts.mean()
                            rate = float(np.dot(tc, ds - ds.mean()) / np.dot(tc, tc))
                            reason = f"past_least_squares_distance_rate_frame_invariant_{len(points)}_points"
                result[key] = Relative(*key, d, rate, age, reason, "same_time_distance_rate" if rate is not None else "")
                if not chain or t > chain[-1][1]:
                    # A delayed old observation may be logged but cannot overwrite current history.
                    entry = (d, t, cals, defs)
                    self.previous[key] = (chain + [entry])[-(self.window + 1):] if chained else [entry]
        return result


class UnavailableKinetics:
    def estimate(self, packet):
        return tuple(LoadEstimate(s, None, None, None, None, None,
                                 "V0_channel_and_reference_unverified") for s in SIDES)


class DisabledKinetics:
    def estimate(self, packet):
        return tuple(LoadEstimate(side, None, None, None, None, None,
                                 "kinetics_deferred_by_motion_only_configuration")
                     for side in SIDES)


@dataclass(frozen=True)
class Calibration:
    calibration_id: str
    sensor_site: str
    gain_N_per_V: float
    offset_V: float
    sigma_N: float
    range_V: tuple[float, float]
    side: str
    synthetic: bool = True

    def __post_init__(self):
        vals = (self.gain_N_per_V, self.offset_V, self.sigma_N, *self.range_V)
        if not self.calibration_id or not self.sensor_site or not np.isfinite(vals).all():
            raise ValueError("incomplete calibration")
        if self.gain_N_per_V <= 0 or self.sigma_N < 0 or self.range_V[0] >= self.range_V[1]:
            raise ValueError("invalid calibration domain")
        if self.side not in SIDES:
            raise ValueError("calibration must be hand-bound")
        if self.synthetic is not True:
            raise ValueError("real calibration admission is not implemented; use unavailable kinetics")


class CalibratedTension:
    """Locally tested calibration arithmetic; no real calibration is supplied."""
    def __init__(self, calibrations: dict[str, Calibration], max_age_s=0.25):
        if any(side != c.side for side, c in calibrations.items()):
            raise ValueError("calibration side mismatch")
        self.calibrations = calibrations
        self.max_age_s = max_age_s

    def estimate(self, packet):
        rows = {r.side: r for r in packet.telemetry}
        results = []
        for side in SIDES:
            c = self.calibrations.get(side)
            if c is None:
                results.append(LoadEstimate(side, None, None, None, None, None, "hand_calibration_unavailable"))
                continue
            r = rows.get(side)
            reason = "channel_unavailable"
            value = None
            if r is not None:
                if c.synthetic != (r.stamp.source == Source.SYNTHETIC):
                    reason = "calibration_source_mismatch"
                elif r.sensor_site != c.sensor_site or r.units != "V":
                    reason = "site_or_unit_mismatch"
                elif packet.decision_s - r.stamp.acquired_s > self.max_age_s:
                    reason = "stale_channel"
                elif r.raw_value is None:
                    reason = r.reason
                elif not c.range_V[0] <= r.raw_value <= c.range_V[1]:
                    reason = "outside_calibration_range"
                elif (r.raw_value - c.offset_V) * c.gain_N_per_V < 0:
                    reason = "negative_tension_outside_model"
                else:
                    value = c.gain_N_per_V * (r.raw_value - c.offset_V)
                    reason = "synthetic_calibration_only" if c.synthetic else "unvalidated_calibration_mapping"
            results.append(LoadEstimate(side, value, c.sigma_N if value is not None else None,
                                        r.stamp if r else None, c.calibration_id,
                                        c.sensor_site, reason))
        return tuple(results)


def distance(a, b):
    return sqrt(sum((x-y)**2 for x, y in zip(a, b)))
