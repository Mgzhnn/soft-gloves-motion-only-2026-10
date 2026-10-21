"""Evaluation-side contact-sensor reference.

Fingertip contact sensors (FSR402-style) on both index fingers of the UNPOWERED gloves give an independent onset
reference for the main collection, as the reference precedence study did (local reference, N = 7): the sensor's
transition from zero to non-zero (grasp, press) or a change of at least 10 % against the previous sample (release =
a 10 % drop; squeeze = a 10 % rise) marks the physical event. These samples are a REFERENCE like a ledger: nothing in
``inputs.py``, the packets, the readouts or the supervisor reads them (the runtime stays vision-only), and the file
format is a side record on the recording's clock.

``contact_events`` applies the reference rule; ``compare_onsets`` measures how far annotated onsets and model
proposals lie from the sensor events (median and 90th percentile of the signed differences), which is what the
0.1 s tolerance needs as a justification. Synthetic tests only until a sensor channel exists.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np

from .contracts import SIDES

CONTACT_SCHEMA = "sg-contact-reference/1"
EVENT_KINDS = ("onset", "rise", "drop", "offset")


@dataclass(frozen=True)
class ContactSample:
    side: str
    t_s: float          # on the recording's clock (the same clock as the packets' acquired_s)
    value: float        # raw sensor reading (units recorded in the file header; zero = no contact)
    sensor_site: str = "index_fingertip"

    def __post_init__(self):
        if self.side not in SIDES or not isfinite(self.t_s) or not isfinite(self.value) or self.value < 0 or not self.sensor_site:
            raise ValueError("contact sample needs a side, finite time, nonnegative finite value and a sensor site")


@dataclass(frozen=True)
class ContactEvent:
    side: str
    t_s: float
    kind: str           # onset: 0 -> non-zero; rise / drop: >= `change` relative change; offset: non-zero -> 0
    value_before: float
    value_after: float


def read_contact_file(path) -> tuple[dict, tuple[ContactSample, ...]]:
    """JSON: {"schema", "participant_id", "session_id", "clock", "units", "sensor", "samples": [[side, t_s, value], ...]}."""
    data = json.loads(Path(path).read_text())
    required = {"schema", "participant_id", "session_id", "clock", "units", "sensor", "samples"}
    if not isinstance(data, dict) or set(data) - {"note"} != required or data["schema"] != CONTACT_SCHEMA:
        raise ValueError("unknown contact reference schema or fields")
    samples = tuple(ContactSample(side, float(t), float(v), data["sensor"]) for side, t, v in data["samples"])
    for side in SIDES:
        ts = [s.t_s for s in samples if s.side == side]
        if any(b <= a for a, b in zip(ts, ts[1:])):
            raise ValueError(f"{side}: contact samples must be strictly time-ordered")
    return {k: data[k] for k in required - {"samples"}}, samples


def contact_events(samples, change=0.10, zero=0.0) -> tuple[ContactEvent, ...]:
    """The reference event rule per hand: 0 -> non-zero is an onset; a relative change of at least ``change`` against
    the previous sample is a rise or a drop; non-zero -> 0 is an offset. ``zero`` is the value at or below which the
    sensor counts as untouched (a declared setting, recorded with the result)."""
    if not 0 < change < 1 or zero < 0:
        raise ValueError("change must be in (0, 1) and zero nonnegative")
    out = []
    for side in SIDES:
        prev = None
        for s in sorted((s for s in samples if s.side == side), key=lambda s: s.t_s):
            if prev is not None:
                was_zero, is_zero = prev.value <= zero, s.value <= zero
                if was_zero and not is_zero:
                    out.append(ContactEvent(side, s.t_s, "onset", prev.value, s.value))
                elif not was_zero and is_zero:
                    out.append(ContactEvent(side, s.t_s, "offset", prev.value, s.value))
                elif not was_zero and not is_zero and abs(s.value - prev.value) >= change * prev.value:
                    out.append(ContactEvent(side, s.t_s, "rise" if s.value > prev.value else "drop", prev.value, s.value))
            prev = s
    return tuple(sorted(out, key=lambda e: (e.t_s, e.side)))


def compare_onsets(reference_times, candidate_times, max_pair_s=2.0):
    """Pair each reference event with the nearest candidate time within ``max_pair_s`` (one to one, in time order) and
    report the signed differences candidate - reference (negative = the candidate precedes the sensor event): count,
    median, 90th percentile of the absolute value, and the unpaired counts. No interval, no test."""
    if max_pair_s <= 0:
        raise ValueError("max_pair_s must be positive")
    ref = sorted(float(t) for t in reference_times); cand = sorted(float(t) for t in candidate_times)
    used = set(); diffs = []
    for r in ref:
        best = None
        for j, c in enumerate(cand):
            if j in used or abs(c - r) > max_pair_s:
                continue
            if best is None or abs(c - r) < abs(cand[best] - r):
                best = j
        if best is not None:
            used.add(best); diffs.append(cand[best] - r)
    arr = np.asarray(diffs, dtype=float)
    return dict(paired=len(diffs), reference_unpaired=len(ref) - len(diffs), candidate_unpaired=len(cand) - len(diffs),
                median_difference_s=(float(np.median(arr)) if len(arr) else None),
                p90_absolute_difference_s=(float(np.percentile(np.abs(arr), 90)) if len(arr) else None),
                differences_s=diffs, max_pair_s=max_pair_s,
                scope="signed candidate - sensor-event times, paired one to one within max_pair_s; descriptive, no interval or test")
