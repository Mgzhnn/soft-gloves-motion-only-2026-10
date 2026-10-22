"""Evaluation-only references. Runtime packet/model/replay never accepts this type."""
from dataclasses import dataclass
from collections import Counter
from math import isfinite
from .contracts import OUTCOMES

REFERENCE_REASONS = ("known", "unknown", "excluded", "unsupported_action")
BOUNDARY_TOLERANCE_S = 1e-6  # shared with import_requests: rows/windows closer than this are contiguous


@dataclass(frozen=True)
class Reference:
    packet_id: str
    opportunity_id: str
    participant_id: str
    session_id: str
    outcomes: tuple[str | None, str | None]
    provenance: str

    def __post_init__(self):
        if not all((self.packet_id, self.opportunity_id, self.participant_id, self.session_id, self.provenance)):
            raise ValueError("reference identity and provenance required")
        if len(self.outcomes) != 2 or any(x is not None and x not in OUTCOMES for x in self.outcomes):
            raise ValueError("unknown reference is None, not no-request")


def _ordered_rows(rows):
    rows = tuple(rows)
    previous = {}; seen = set()
    for row in rows:
        key = (row["inputs"]["participant_id"], row["inputs"]["session_id"])
        ident = key + (row["packet_id"],)
        t = row["decision_s"]
        if not all(key) or not isfinite(t) or t <= previous.get(key, -1.) or ident in seen:
            raise ValueError("ordered unique first-issued trace required per participant/session")
        previous[key] = t; seen.add(ident)
    return rows


def summarize(rows, references=()):
    """Per-hand decision-tick accounting; not the planned opportunity-level clinical endpoint."""
    references = tuple(references)
    rows = _ordered_rows(rows)
    ref = {(r.participant_id, r.session_id, r.packet_id): r for r in references}
    if len(ref) != len(references):
        raise ValueError("duplicate reference packet")
    counts = Counter(); matched = set()
    for row in rows:
        key = (row["inputs"]["participant_id"], row["inputs"]["session_id"], row["packet_id"])
        if key in ref:
            matched.add(key)
        for i, output in enumerate(row["outputs"]):
            counts["all_hand_decision_ticks"] += 1
            counts[output["status"]] += 1
            target = ref[key].outcomes[i] if key in ref else None
            if target is None:
                counts["unknown_reference_ticks"] += 1
                continue
            counts["known_reference_ticks"] += 1
            issued = output["status"] == "proposed_simulation_only"
            if target == "NQ" and issued:
                counts["unwanted_proposal_ticks"] += 1
            elif target != "NQ" and not issued:
                counts["missed_or_withheld_request_ticks"] += 1
            elif target != "NQ" and issued and output["raw_outcome"] != target:
                counts["wrong_action_ticks"] += 1
    return {"scope": "offline_software_accounting_not_scientific_event_endpoint", "counts": dict(counts),
            "unmatched_reference_records": len(set(ref) - matched)}


@dataclass(frozen=True)
class Opportunity:
    opportunity_id: str
    participant_id: str
    session_id: str
    start_s: float
    end_s: float
    acceptable: tuple[tuple[str, ...] | None, tuple[str, ...] | None]
    provenance: str
    hand_windows_s: tuple[tuple[float, float], tuple[float, float]] | None = None
    acceptable_joint_pairs: tuple[tuple[str, str], ...] | None = None
    # One physical request of one hand may span several rows (e.g. the other hand's state changes).
    # Rows sharing a hand's request id are scored as ONE first-proposal opportunity.
    request_ids: tuple[str | None, str | None] | None = None
    # Why an unscored hand is unscored: unknown, excluded, or unsupported_action (outside admitted actions).
    reference_reasons: tuple[str, str] | None = None

    def __post_init__(self):
        # The evaluator, the statistics and the ledger summary compare ``acceptable`` with the tuple ("NQ",). An
        # opportunity rebuilt from JSON carries lists; unnormalised, its no-request hand was scored as a missed
        # request and its exposure vanished (architecture review, 5 October 2026). Lists become tuples here.
        if isinstance(self.acceptable, (list, tuple)):
            object.__setattr__(self, "acceptable", tuple(a if a is None or isinstance(a, str) else tuple(a) for a in self.acceptable))
        if self.hand_windows_s is not None:
            object.__setattr__(self, "hand_windows_s", tuple(tuple(w) for w in self.hand_windows_s))
        if self.acceptable_joint_pairs is not None:
            object.__setattr__(self, "acceptable_joint_pairs", tuple(tuple(p) for p in self.acceptable_joint_pairs))
        for name in ("request_ids", "reference_reasons"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, tuple(getattr(self, name)))
        if not all((self.opportunity_id, self.participant_id, self.session_id, self.provenance)):
            raise ValueError("opportunity identity and provenance required")
        if not all(isfinite(t) for t in (self.start_s, self.end_s)) or not 0 <= self.start_s <= self.end_s or len(self.acceptable) != 2:
            raise ValueError("valid interval and two hand reference sets required")
        for allowed in self.acceptable:
            if allowed is not None and (not allowed or any(x not in OUTCOMES for x in allowed)
                                        or ("NQ" in allowed and len(allowed) > 1)):
                raise ValueError("request alternatives or no-request; ambiguity remains unknown")
        if self.hand_windows_s is not None:
            if len(self.hand_windows_s) != 2 or any(not all(isfinite(t) for t in w) or
                    not self.start_s <= w[0] <= w[1] <= self.end_s for w in self.hand_windows_s):
                raise ValueError("hand windows must be finite nested intervals")
        if self.acceptable_joint_pairs is not None:
            if not self.acceptable_joint_pairs or any(a is None for a in self.acceptable):
                raise ValueError("joint acceptable set requires both hand references")
            if any(len(pair) != 2 or any(v not in self.acceptable[i] for i, v in enumerate(pair))
                   for pair in self.acceptable_joint_pairs):
                raise ValueError("joint outcomes must agree with per-hand allowed sets")
        if self.request_ids is not None:
            if len(self.request_ids) != 2 or any(r is not None and (not isinstance(r, str) or not r) for r in self.request_ids):
                raise ValueError("request ids must be two nonempty strings or None")
            if any(r is not None and (a is None or a == ("NQ",)) for r, a in zip(self.request_ids, self.acceptable)):
                raise ValueError("a request id is allowed only for a hand with a known request")
        if self.reference_reasons is not None:
            if len(self.reference_reasons) != 2 or any(r not in REFERENCE_REASONS for r in self.reference_reasons):
                raise ValueError("unknown reference reason")
            if any((r == "known") != (a is not None) for r, a in zip(self.reference_reasons, self.acceptable)):
                raise ValueError("reference reason must be known exactly when the hand is scored")


def _window_outputs(rows, i, start, end):
    """Per-hand output statuses inside a scored window, so misses can be split by reason.

    A withheld tick is a withheld *request* only when the model's own choice was not NQ; a
    low-confidence NQ is also withheld by the supervisor, but the model never asked for anything.
    """
    counts = Counter()
    for r in rows:
        if start <= r["decision_s"] < end:
            out = r["outputs"][i]
            if out["status"] == "withheld":
                kind = "withheld_request:" if out.get("raw_outcome", "NQ") != "NQ" else "withheld_no_request:"
                counts[kind + out.get("reason", "")] += 1
            else:
                counts[out["status"]] += 1
    return dict(counts)


def _hand_ticks(rows, i):
    """Time-ordered (decision_s, proposed?, raw outcome) for one hand."""
    return [(r["decision_s"], r["outputs"][i]["status"] == "proposed_simulation_only", r["outputs"][i].get("raw_outcome", "NQ"))
            for r in rows]


def proposal_runs(ticks):
    """Maximal sequences of consecutive proposed ticks with one outcome: [{start_s, end_s, outcome, ticks}]."""
    runs, current = [], None
    for t, proposed, outcome in ticks:
        if proposed and current is not None and current["outcome"] == outcome:
            current["end_s"] = t; current["ticks"] += 1
        elif proposed:
            current = dict(start_s=t, end_s=t, outcome=outcome, ticks=1); runs.append(current)
        else:
            current = None
    return runs


def _tolerated_early_ticks(ticks, request_s, early_limit_s, events=False):
    """Ticks before ``request_s`` of a proposal run that began at most ``early_limit_s`` before the hand's next request
    window and is still running when it opens. That request is credited with the run, so a no-request row must not
    also charge it as unwanted. With ``events`` (accepted commands, one tick each) it is the first command inside the
    tolerance instead, run or not."""
    if events:
        early = [t for t, proposed, _ in ticks if proposed and t < request_s and request_s - t <= early_limit_s]
        return set(early[:1])
    run = []
    for t, proposed, outcome in ticks:
        if proposed and run and run[-1][1] == outcome:
            run.append((t, outcome))
        elif proposed:
            run = [(t, outcome)]
        else:
            run = []
        if t >= request_s:
            break
    if not run or run[-1][0] < request_s or run[0][0] >= request_s or request_s - run[0][0] > early_limit_s:
        return set()
    return {t for t, _ in run if t < request_s}


def _score_hand(allowed, eligible, i, start, end, session_rows=(), early_limit_s=0., rearm_s=None, next_request_s=None,
                events=False):
    """Score one hand over one acceptable window [start, end).

    ``first_proposal`` is the first proposed tick inside the eligible rows, as before. New (2 Oct 2026):
    if the hand was ALREADY proposing that outcome without interruption when the window opened, the
    proposal did not begin here; ``proposal_run_start_s`` walks back through the session's earlier
    ticks to where the run began and ``early_by_s`` = start − run start. A request is
    ``wrong_timing_first`` when early_by_s exceeds the declared ``early_limit_s`` (default 0: strict).
    A first proposal inside the row but before the hand's own window is early by ``start`` minus its time,
    under the same tolerance (5 Oct 2026). A run counts as uninterrupted only tick by tick: one tick that is
    not a proposal ends it, so on traces that flicker at the tick scale this check sees only the last burst
    (``proposals_since_rearm_before_window`` counts the earlier ones).
    Before this, a request preceded by 3 s of continuous proposing still scored ``compatible_first``
    (independent review, 28 Sep 2026). Also reported: proposal runs and onsets inside the window (runs
    that begin inside it) and the number of proposals since the hand's last re-arm point.
    In a no-request row, the ticks of a run the hand's next request (starting at ``next_request_s``) is credited with
    under ``early_limit_s`` are not proposals of this row (``tolerated_early_ticks`` counts them).
    ``events``: the ticks are accepted commands, single events rather than runs, so a command at most ``early_limit_s``
    before the window is the request's first event even though nothing is still running when the window opens.
    """
    ticks = _hand_ticks(eligible, i)
    tolerated = set()
    if allowed == ("NQ",) and early_limit_s > 0 and next_request_s is not None and session_rows:
        session = _hand_ticks(session_rows, i)
        if events:  # the request credits the first command after its re-arm point, which this row starts after
            session = [tk for tk in session if tk[0] >= start]
        tolerated = _tolerated_early_ticks(session, next_request_s, early_limit_s, events)
        ticks = [(t, p and t not in tolerated, o) for t, p, o in ticks]
    issued = [(t, o) for t, p, o in ticks if p]
    first = issued[0] if issued else None
    runs = proposal_runs(ticks)
    history = [tk for tk in _hand_ticks(session_rows, i) if tk[0] < start] if session_rows else []
    run_start, early_by, active = (first[0] if first else None), 0., False
    if first is not None and history and ticks[0][1] and ticks[0][0] == first[0]:
        for t, proposed, outcome in reversed(history):
            if proposed and outcome == first[1]:
                run_start, active = t, True
            else:
                break
        early_by = max(0., start - run_start)
    if first is not None and first[0] < start:
        # The hand's window starts inside its row and the first proposal came before it: early by that much. Before
        # 5 October 2026 this case failed the request whatever the declared tolerance, while the same situation
        # labelled as two rows was admitted within it. With the default tolerance (0) the result is unchanged.
        early_by = max(early_by, start - first[0])
    if events and allowed not in (None, ("NQ",)) and early_limit_s > 0:
        early = [(t, o) for t, p, o in history if p and start - t <= early_limit_s and (rearm_s is None or t >= rearm_s)]
        if early and (first is None or early[0][0] < first[0]):
            first = early[0]; issued = [first] + [x for x in issued if x[0] != first[0]]
            run_start, early_by = first[0], start - first[0]
    since_rearm = sum(1 for t, p, _ in history if p and (rearm_s is None or t >= rearm_s))
    if allowed is None:
        result = "unknown_reference"
    elif allowed == ("NQ",):
        result = "unwanted" if first else "correct_no_new_request"
    elif first is None:
        result = "missed_including_withheld"
    elif early_by > early_limit_s or first[0] >= end:
        result = "wrong_timing_first"
    else:
        result = "compatible_first" if first[1] in allowed else "wrong_first"
    return dict(first_proposal=first, result=result,
                proposal_count=len(issued), repeated_proposals=max(0, len(issued)-1),
                proposals_in_window=sum(start <= t < end for t, _ in issued),
                eligible_decision_ticks=len(eligible), accepted_actions=allowed,
                acceptable_window_s=(start, end),
                proposal_runs=len(runs), proposal_onsets=len(runs) - int(active),
                run_active_at_window_start=active, proposal_run_start_s=run_start, early_by_s=early_by,
                early_limit_s=early_limit_s, proposals_since_rearm_before_window=since_rearm, rearm_s=rearm_s,
                tolerated_early_ticks=sum(1 for t, _, _ in _hand_ticks(eligible, i) if t in tolerated))


def _request_groups(opportunities):
    """(participant, session, hand, request id) -> time-ordered opportunities of that one request."""
    groups = {}
    for op in opportunities:
        for i, rid in enumerate(op.request_ids or (None, None)):
            if rid is not None:
                groups.setdefault((op.participant_id, op.session_id, i, rid), []).append(op)
    for key, ops in groups.items():
        ops.sort(key=lambda o: o.start_s)
        i = key[2]
        if len({frozenset(o.acceptable[i]) for o in ops}) != 1:  # "G|FP" and "FP|G" are the same set
            raise ValueError(f"request {key[3]!r} changes its acceptable actions across rows")
        windows = [o.hand_windows_s[i] if o.hand_windows_s else (o.start_s, o.end_s) for o in ops]
        if any(abs(a[1] - b[0]) > BOUNDARY_TOLERANCE_S for a, b in zip(windows, windows[1:])):
            raise ValueError(f"request {key[3]!r} spans rows whose hand windows are not contiguous")
    return groups


def _request_window_ends(opportunities):
    """(participant, session, hand) -> ends of every known request window of that hand (re-arm points)."""
    ends = {}
    for op in opportunities:
        for i, allowed in enumerate(op.acceptable):
            if allowed is not None and allowed != ("NQ",):
                window = op.hand_windows_s[i] if op.hand_windows_s else (op.start_s, op.end_s)
                ends.setdefault((op.participant_id, op.session_id, i), []).append(window[1])
    return ends


def _request_window_starts(opportunities):
    """(participant, session, hand) -> starts of every known request window of that hand."""
    starts = {}
    for op in opportunities:
        for i, allowed in enumerate(op.acceptable):
            if allowed is not None and allowed != ("NQ",):
                window = op.hand_windows_s[i] if op.hand_windows_s else (op.start_s, op.end_s)
                starts.setdefault((op.participant_id, op.session_id, i), []).append(window[0])
    return starts


def _next_request(starts, key, start):
    later = [s for s in starts.get(key, []) if s > start]
    return min(later) if later else None


def _rearm_point(ends, key, start):
    prior = [e for e in ends.get(key, []) if e <= start + BOUNDARY_TOLERANCE_S]
    return max(prior) if prior else None


def evaluate_opportunities(rows, opportunities, early_limit_s=0.):
    """First actually proposed recommendation per hand-request in independently defined intervals.

    A hand request normally equals one opportunity row. When rows share a hand's request id, that
    hand is scored once over the union of the rows (first proposal, repeated proposals, timing);
    the later rows carry ``continuation_of_request`` for that hand and are not counted again.
    ``early_limit_s`` is the declared tolerance for a proposal run that began before the acceptable
    window (0 = strict); it is an analysis-plan setting, never tuned on test results.
    This code can check synthetic event accounting. Valid clinical intervals/labels and
    onset-error analysis remain data/protocol dependencies, not supplied by the evaluator.
    """
    if not isfinite(early_limit_s) or early_limit_s < 0:
        raise ValueError("early_limit_s must be a finite nonnegative number of seconds")
    opportunities = tuple(opportunities)
    rows = _ordered_rows(rows)
    keys = [(o.participant_id, o.session_id, o.opportunity_id) for o in opportunities]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate independent opportunity reference")
    groups = _request_groups(opportunities)
    ends = _request_window_ends(opportunities)
    starts = _request_window_starts(opportunities)
    events = any(r.get("endpoint") == "accepted_command_secondary" for r in rows)  # rows from fsm.command_rows
    by_session = {}
    for r in rows:
        by_session.setdefault((r["inputs"]["participant_id"], r["inputs"]["session_id"]), []).append(r)

    def rows_between(op_first, op_last):
        return [r for r in by_session.get((op_first.participant_id, op_first.session_id), [])
                if op_first.start_s <= r["decision_s"] < op_last.end_s]

    outcomes = []
    for op in opportunities:
        eligible = rows_between(op, op)
        session_rows = by_session.get((op.participant_id, op.session_id), [])
        evaluated = []; spans_rows = False
        for i, allowed in enumerate(op.acceptable):
            rid = (op.request_ids or (None, None))[i]
            group = groups.get((op.participant_id, op.session_id, i, rid)) if rid is not None else None
            if group is not None and len(group) > 1:
                spans_rows = True
                first_op, last_op = group[0], group[-1]
                if op is not first_op:
                    window = op.hand_windows_s[i] if op.hand_windows_s else (op.start_s, op.end_s)
                    evaluated.append(dict(first_proposal=None, result="continuation_of_request", request_id=rid,
                                          proposal_count=0, repeated_proposals=0,
                                          eligible_decision_ticks=len(eligible), accepted_actions=allowed,
                                          acceptable_window_s=window, scored_in_opportunity=first_op.opportunity_id))
                    continue
                start = first_op.hand_windows_s[i][0] if first_op.hand_windows_s else first_op.start_s
                end = last_op.hand_windows_s[i][1] if last_op.hand_windows_s else last_op.end_s
                group_rows = rows_between(first_op, last_op)
                rearm = _rearm_point(ends, (op.participant_id, op.session_id, i), start)
                hand = _score_hand(allowed, group_rows, i, start, end, session_rows, early_limit_s, rearm, events=events)
                hand.update(request_id=rid, grouped_opportunities=[o.opportunity_id for o in group])
            else:
                start, end = op.hand_windows_s[i] if op.hand_windows_s else (op.start_s, op.end_s)
                group_rows = eligible
                rearm = _rearm_point(ends, (op.participant_id, op.session_id, i), start)
                hand = _score_hand(allowed, eligible, i, start, end, session_rows, early_limit_s, rearm,
                                   _next_request(starts, (op.participant_id, op.session_id, i), start), events)
                if rid is not None:
                    hand["request_id"] = rid
            if allowed is None:
                hand["reference_reason"] = (op.reference_reasons or ("unknown", "unknown"))[i]
            elif hand["result"] == "missed_including_withheld":
                hand["window_outputs"] = _window_outputs(group_rows, i, start, end)
            evaluated.append(hand)
        if spans_rows:
            joint_ok = None  # a hand's result belongs to several rows; a per-row joint verdict would mix units
        else:
            joint_ok = (all(e["result"] in ("compatible_first", "correct_no_new_request") for e in evaluated)
                        if all(a is not None for a in op.acceptable) else None)
            if joint_ok and op.acceptable_joint_pairs is not None:
                pair = tuple(e["first_proposal"][1] if e["first_proposal"] else "NQ" for e in evaluated)
                joint_ok = pair in op.acceptable_joint_pairs
        outcomes.append(dict(opportunity_id=op.opportunity_id, participant_id=op.participant_id,
                             session_id=op.session_id, hands=evaluated,
                             fully_referenceable=all(a is not None for a in op.acceptable),
                             joint_compatible=joint_ok, early_limit_s=early_limit_s))
    return outcomes


def timeline_accounting(rows, opportunities):
    """Count every output, including uncovered/unknown time; gaps are never invented as NQ."""
    counts = Counter()
    for row in _ordered_rows(rows):
        hits = [o for o in opportunities if (o.participant_id, o.session_id) ==
                (row["inputs"]["participant_id"], row["inputs"]["session_id"])
                and o.start_s <= row["decision_s"] < o.end_s]
        for i, out in enumerate(row["outputs"]):
            counts["all_hand_ticks"] += 1
            category = "unlabelled" if not hits else "overlapping" if len(hits) > 1 else (
                "unknown_or_excluded" if hits[0].acceptable[i] is None else
                "no_request" if hits[0].acceptable[i] == ("NQ",) else "request")
            if category == "request" and hits[0].hand_windows_s is not None:
                start, end = hits[0].hand_windows_s[i]
                if not start <= row["decision_s"] < end:
                    category = "outside_request_window"
            counts[category + "_hand_ticks"] += 1
            if out["status"] == "proposed_simulation_only":
                counts[category + "_proposals"] += 1
    return dict(counts)
