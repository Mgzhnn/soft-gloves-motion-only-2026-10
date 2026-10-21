"""Participant-level accounting for the M1 endpoint. People are the units; no interval is invented.

Inputs are the per-opportunity outcomes from ``evaluation.evaluate_opportunities`` (one arm) and the
in-scope opportunities themselves. Outputs are per-person fractions and exposure-based event counts,
plus paired per-person differences between two arms. A 95% paired t interval is reported only from
``MIN_PEOPLE_FOR_INTERVAL`` people upward (experiment_matrix.md: "wide intervals are inconclusive");
below that the field says why it is absent. p-values are deliberately absent; the frozen analysis plan
decides the estimator once the pilot exists.
"""
from __future__ import annotations

from collections import defaultdict
from math import sqrt

SUCCESS = ("compatible_first",)
REQUEST_RESULTS = ("compatible_first", "wrong_first", "wrong_timing_first", "missed_including_withheld")
MIN_PEOPLE_FOR_INTERVAL = 5

# Two-sided 97.5% quantiles of Student's t by degrees of freedom (standard table); >120 uses the normal 1.960,
# which is narrower than the exact value by at most 1.0 % (at 121 degrees of freedom).
_T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
          11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
          20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
          29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t_975(df):
    if df < 1:
        raise ValueError("degrees of freedom must be at least 1")
    if df in _T_975:
        return _T_975[df]
    if df > 120:
        return 1.960
    return _T_975[max(k for k in _T_975 if k <= df)]  # conservative: the next lower tabulated df


def paired_interval(differences, min_people=MIN_PEOPLE_FOR_INTERVAL):
    """95% paired t interval of the mean per-person difference, or an explicit reason for its absence."""
    if min_people < 2:
        raise ValueError("an interval needs at least 2 people (n - 1 degrees of freedom)")
    values = [float(d) for d in differences if d is not None]
    n = len(values)
    if n < min_people:
        return dict(mean=None, lower=None, upper=None, n_people=n,
                    reason=f"fewer than {min_people} people with a paired difference; no interval is reported")
    mean = sum(values) / n
    sd = sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = t_975(n - 1) * sd / sqrt(n)
    return dict(mean=mean, lower=mean - half, upper=mean + half, sd_of_differences=sd, n_people=n,
                method="paired Student t, two-sided 95%, df = n_people - 1; people are the units; no multiplicity adjustment")


def _by_person(outcomes):
    groups = defaultdict(list)
    for o in outcomes:
        groups[o["participant_id"]].append(o)
    return groups


def per_participant_success(outcomes):
    """Per person: acceptable first decisions over reference-evaluable requested hand-opportunities.

    Denominators are kept separate: requested hand-opportunities with a known reference (evaluable),
    unknown-reference hand-opportunities (not scored), no-request hand-intervals (scored under exposure).
    ``wrong_timing_first`` is split into early (the first proposal, or the proposal run it belongs to,
    began before the acceptable window) and late (after it).
    """
    rows = {}
    for person, group in _by_person(outcomes).items():
        c = dict(evaluable_request_hands=0, compatible_first=0, wrong_first=0, wrong_timing_first=0,
                 wrong_timing_first_early=0, wrong_timing_first_late=0,
                 missed_including_withheld=0, unknown_reference_hands=0, no_request_hands=0, unwanted=0,
                 unknown_reference_hands_by_reason=defaultdict(int), request_continuation_rows=0,
                 missed_with_withheld_request_ticks=0, missed_model_never_requested=0, missed_no_decision_ticks=0,
                 missed_window_output_ticks=defaultdict(int))
        for o in group:
            for hand in o["hands"]:
                r = hand["result"]
                if r == "unknown_reference":
                    c["unknown_reference_hands"] += 1
                    c["unknown_reference_hands_by_reason"][hand.get("reference_reason", "unknown")] += 1
                elif r in ("unwanted", "correct_no_new_request"):
                    c["no_request_hands"] += 1
                    c["unwanted"] += int(r == "unwanted")
                elif r == "continuation_of_request":
                    c["request_continuation_rows"] += 1  # already scored once in its first row
                elif r in REQUEST_RESULTS:
                    c["evaluable_request_hands"] += 1
                    c[r] += 1
                    if r == "wrong_timing_first":
                        first = hand.get("first_proposal")
                        start = hand.get("acceptable_window_s", (None, None))[0]
                        early = hand.get("early_by_s", 0.) > 0. or (first is not None and start is not None and first[0] < start)
                        c["wrong_timing_first_early" if early else "wrong_timing_first_late"] += 1
                    if r == "missed_including_withheld":
                        window = hand.get("window_outputs", {})
                        if not window:
                            c["missed_no_decision_ticks"] += 1  # no replayed tick inside the window at all
                        elif any(k.startswith("withheld_request:") for k in window):
                            c["missed_with_withheld_request_ticks"] += 1  # the model asked; the supervisor withheld
                        else:
                            c["missed_model_never_requested"] += 1
                        for k, v in window.items():
                            c["missed_window_output_ticks"][k] += v
        c["unknown_reference_hands_by_reason"] = dict(c["unknown_reference_hands_by_reason"])
        c["missed_window_output_ticks"] = dict(c["missed_window_output_ticks"])
        c["success_fraction"] = (c["compatible_first"] / c["evaluable_request_hands"]
                                 if c["evaluable_request_hands"] else None)
        rows[person] = c
    return rows


def unwanted_exposure(outcomes, opportunities):
    """Proposal burden on labelled no-request hand-intervals, in two forms.

    * ticks: ALL proposed ticks per labelled no-request hand-minute (cadence-dependent; invariant to
      interval tiling; the 23 Sep metric, kept for comparability).
    * onsets: proposal RUNS that begin inside the interval per hand-minute (a run already active when the
      interval opens is not a new onset). For sustained runs this does not scale with the tick rate and is the
      closer stand-in for "unwanted activations"; a run ends at the first tick that is not a proposal, so on
      traces that flicker at the tick scale onsets follow the tick count (correction 5 Oct 2026: on the first
      real rule traces 174 of 202 runs were one or two ticks long). It is still offline replay, not an online
      accident rate.
    Also per interval: time from interval start to the first unwanted proposal, or None when censored; an
    interval is one timeline row of that hand, and a run still active when the row opens counts at its first
    tick. Exposure is LABELLED time: no-request hand-intervals that contain no decision tick at all (a truncated
    or late-started trace) are counted in ``no_request_hand_intervals_without_decision_ticks`` and
    ``exposure_hand_s_without_decision_ticks`` so that a diluted rate cannot pass unnoticed (5 Oct 2026).
    The trace cadence must be identical across arms. First-proposal success is separate.
    """
    exposure = defaultdict(float); intervals = defaultdict(int); events = defaultdict(int); onsets = defaultdict(int)
    first_times = defaultdict(list); with_unwanted = defaultdict(int)
    uncovered = defaultdict(int); uncovered_s = defaultdict(float)
    seen = defaultdict(list); spans = {}
    for op in opportunities:
        spans[(op.participant_id, op.session_id, op.opportunity_id)] = op.end_s - op.start_s
        for i, allowed in enumerate(op.acceptable):
            if allowed == ("NQ",):
                key = (op.participant_id, op.session_id, i)
                if any(max(a, op.start_s) < min(b, op.end_s) for a, b in seen[key]):
                    raise ValueError("overlapping no-request exposure")
                seen[key].append((op.start_s, op.end_s))
                exposure[op.participant_id] += op.end_s - op.start_s
                intervals[op.participant_id] += 1
    for outcome in outcomes:
        for hand in outcome["hands"]:
            if hand["result"] in ("unwanted", "correct_no_new_request"):
                if "proposal_count" not in hand:
                    raise ValueError("proposal counts required; old interval event metric is unsupported")
                person = outcome["participant_id"]
                if hand.get("eligible_decision_ticks") == 0:
                    uncovered[person] += 1
                    uncovered_s[person] += spans.get((person, outcome["session_id"], outcome["opportunity_id"]), 0.)
                events[person] += hand["proposal_count"]
                onsets[person] += hand.get("proposal_onsets", 0)
                start = hand.get("acceptable_window_s", (None, None))[0]
                first = hand.get("first_proposal")
                if first is not None and start is not None:
                    with_unwanted[person] += 1
                    first_times[person].append(first[0] - start)
                else:
                    first_times[person].append(None)  # censored: no unwanted proposal in this interval
    return {p: dict(no_request_hand_intervals=intervals[p], exposure_hand_s=exposure[p],
                    unwanted_proposals=events[p],
                    proposals_per_hand_minute=(events[p]*60/exposure[p] if exposure[p] else None),
                    unwanted_onsets=onsets[p],
                    onsets_per_hand_minute=(onsets[p]*60/exposure[p] if exposure[p] else None),
                    intervals_with_unwanted=with_unwanted[p],
                    time_to_first_unwanted_s=first_times[p],
                    no_request_hand_intervals_without_decision_ticks=uncovered[p],
                    exposure_hand_s_without_decision_ticks=uncovered_s[p],
                    scope="ticks: all proposal ticks per labelled no-request hand-minute (cadence-dependent); "
                          "onsets: proposal runs starting inside the interval per hand-minute; neither is an "
                          "independent command event or accident rate")
            for p in sorted(set(exposure) | set(events))}


def paired_contrast(per_person_a, per_person_b, field="success_fraction", min_people=MIN_PEOPLE_FOR_INTERVAL):
    """Per-person differences A − B on a common person set; an interval only from ``min_people`` upward."""
    people = sorted(set(per_person_a) & set(per_person_b))
    diffs = {}
    for p in people:
        a, b = per_person_a[p].get(field), per_person_b[p].get(field)
        diffs[p] = None if a is None or b is None else a - b
    known = [d for d in diffs.values() if d is not None]
    interval = paired_interval(known, min_people)
    return dict(per_person=diffs, n_people=len(known),
                mean_difference=(sum(known) / len(known) if known else None),
                min_difference=(min(known) if known else None), max_difference=(max(known) if known else None),
                interval_95=interval,
                scope=("participant-level paired differences with a 95% paired t interval; no test"
                       if interval["mean"] is not None else
                       "participant-level paired differences; no interval or test with fewer people than the frozen plan requires"))
