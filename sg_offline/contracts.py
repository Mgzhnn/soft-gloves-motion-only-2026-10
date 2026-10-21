"""Versioned, immutable observation contracts; references deliberately live elsewhere."""
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Protocol

SCHEMA = "sg-offline/3"
SIDES = ("left", "right")
ACTIONS = ("G", "FP", "AR", "NT", "HP", "R")
OUTCOMES = ("NQ",) + ACTIONS  # NQ is no NEW request, not a physical action.


class Source(str, Enum):
    SYNTHETIC = "synthetic"
    RECORDED = "recorded"
    MEASURED = "measured"


@dataclass(frozen=True)
class Stamp:
    sample_id: str
    acquired_s: float
    available_s: float
    source: Source
    clock: str = "session"

    def __post_init__(self):
        if not self.sample_id or not self.clock:
            raise ValueError("sample identity and clock required")
        if not all(isfinite(t) and t >= 0 for t in (self.acquired_s, self.available_s)):
            raise ValueError("finite nonnegative timestamps required")
        if self.acquired_s > self.available_s:
            raise ValueError("acquisition cannot follow availability")
        if not isinstance(self.source, Source):
            raise ValueError("explicit source kind required")


@dataclass(frozen=True)
class Point:
    identity: str
    xyz: tuple[float, float, float] | None
    stamp: Stamp
    frame: str = "camera_optical"
    units: str = "m"
    sigma_m: float | None = None
    reason: str = ""
    definition: str = "functional_point"
    frame_is_fixed: bool = False
    calibration_id: str = ""
    transform_id: str = ""

    def __post_init__(self):
        if type(self.frame_is_fixed) is not bool:
            raise ValueError("fixed-frame declaration must be boolean")
        if self.units != "m" or not self.frame or not self.identity:
            raise ValueError("point requires identity, metric metres and named frame")
        if self.xyz is not None and (len(self.xyz) != 3 or not all(isfinite(x) for x in self.xyz)):
            raise ValueError("point must be finite xyz or explicitly unavailable")
        if self.xyz is not None and not isinstance(self.xyz, tuple):
            raise ValueError("point coordinates must be immutable tuples")
        if self.xyz is None and not self.reason:
            raise ValueError("missing point requires reason")
        if self.sigma_m is not None and (not isfinite(self.sigma_m) or self.sigma_m < 0):
            raise ValueError("invalid uncertainty")


@dataclass(frozen=True)
class RawTension:
    side: str
    sensor_site: str
    raw_value: float | None
    stamp: Stamp
    units: str = "V"
    reason: str = ""

    def __post_init__(self):
        if self.side not in SIDES or not self.sensor_site:
            raise ValueError("hand and sensor site required")
        if self.raw_value is not None and not isfinite(self.raw_value):
            raise ValueError("nonfinite raw channel")
        if self.raw_value is None and not self.reason:
            raise ValueError("missing channel requires reason")


@dataclass(frozen=True)
class LoadEstimate:
    side: str
    value_N: float | None
    sigma_N: float | None
    stamp: Stamp | None
    calibration_id: str | None
    sensor_site: str | None
    reason: str
    quantity: str = "specified_transmission_wire_axial_tension"
    units: str = "N"
    frame: str = "sensor_axial_positive_tension"

    def __post_init__(self):
        if self.side not in SIDES or self.units != "N":
            raise ValueError("invalid load contract")
        if self.quantity != "specified_transmission_wire_axial_tension" or self.frame != "sensor_axial_positive_tension":
            raise ValueError("quantity/site-frame change requires a different contract version")
        if self.value_N is None:
            if not self.reason:
                raise ValueError("unavailable load requires reason")
        elif (not isfinite(self.value_N) or self.value_N < 0 or self.stamp is None
              or not self.calibration_id or not self.sensor_site or self.sigma_N is None
              or not isfinite(self.sigma_N) or self.sigma_N < 0):
            raise ValueError("valid load requires value, uncertainty, site and calibration provenance")


@dataclass(frozen=True)
class Packet:
    packet_id: str
    decision_s: float
    hands: tuple[Point, Point]
    objects: tuple[Point, ...] = ()
    telemetry: tuple[RawTension, ...] = ()
    previous_commands: tuple[str, str] = ("NQ", "NQ")
    previous_command_times_s: tuple[float | None, float | None] = (None, None)
    schema: str = SCHEMA
    clock: str = "session"
    session_id: str = "synthetic-session"
    participant_id: str = "synthetic-person"

    def __post_init__(self):
        if self.schema != SCHEMA or not self.packet_id or not self.session_id or not self.participant_id or not isfinite(self.decision_s) or self.decision_s < 0:
            raise ValueError("invalid packet metadata")
        if any(not isinstance(x, tuple) for x in (self.hands, self.objects, self.telemetry, self.previous_commands, self.previous_command_times_s)):
            raise ValueError("packet collections must be immutable tuples")
        if len(self.hands) != 2 or tuple(p.identity for p in self.hands) != SIDES:
            raise ValueError("explicit ordered left/right observations required, including missing")
        if len(self.previous_commands) != 2 or any(c not in OUTCOMES for c in self.previous_commands):
            raise ValueError("invalid past-command alphabet")
        if len(self.previous_command_times_s) != 2:
            raise ValueError("two past-command timestamps required")
        for command, t in zip(self.previous_commands, self.previous_command_times_s):
            if t is not None and (not isfinite(t) or t < 0 or t >= self.decision_s):
                raise ValueError("command history must strictly precede decision")
            if command != "NQ" and t is None:
                raise ValueError("issued past command needs timestamp")
        if len({p.identity for p in self.objects}) != len(self.objects):
            raise ValueError("duplicate object identity")
        if any(p.identity in SIDES for p in self.objects):
            raise ValueError("object identity collides with reserved hand namespace")
        if len({p.side for p in self.telemetry}) != len(self.telemetry):
            raise ValueError("duplicate tendon channel per hand")
        records = self.hands + self.objects + self.telemetry
        for r in records:
            if r.stamp.clock != self.clock or r.stamp.available_s > self.decision_s:
                raise ValueError("future/unmapped-clock input")
        frames = {p.frame for p in self.hands + self.objects if p.xyz is not None}
        if len(frames) > 1:
            raise ValueError("frame transform required before geometry")


class KineticsProvider(Protocol):
    def estimate(self, packet: Packet) -> tuple[LoadEstimate, LoadEstimate]: ...


class ObjectProvider(Protocol):
    def observe(self, rgb, stamp: Stamp) -> tuple[Point, ...]: ...


class AppearanceProvider(Protocol):
    scheme: str
    def describe(self, rgb, mask): ...


class JointInference(Protocol):
    name: str
    def predict(self, bundle): ...
