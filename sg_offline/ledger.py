"""Human-written reference ledgers (M0): landmark labels and opportunity/reference intervals.

Evaluation-side only. Nothing here is a runtime input: packets, models and the replay never
read a ledger (``inputs.py`` rejects label fields by design). Two file formats:

* ``sg-landmark-ledger/1`` — JSONL, one row per (frame, hand side, functional point) with an
  explicit visibility state; pixel coordinates exist only for ``visible`` rows.
* ``sg-opportunity-ledger/1`` — JSON, one file per participant/session, listing independently
  defined intervals (requests, no-request intervals, out-of-scope requests) in session-relative
  seconds, converted here into the trace clock through the recording's first-frame availability
  stamp so that ``evaluation.evaluate_opportunities`` can join them on ``decision_s``.

The loaders validate structure and vocabulary. They do not establish that a label is correct,
that an interval is clinically meaningful, or that annotators agreed; those are M0 measurements.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from .contracts import ACTIONS, OUTCOMES, SIDES
from .evaluation import Opportunity

LANDMARK_SCHEMA = "sg-landmark-ledger/1"
OPPORTUNITY_SCHEMA = "sg-opportunity-ledger/1"
FUNCTIONAL_POINTS = ("thumb_mcp", "index_mcp", "trapezium")
LANDMARK_STATES = ("visible", "occluded", "out_of_frame", "ambiguous")
MARKER_PASSES = ("marker_on", "marker_off", "unmarked")
OPPORTUNITY_KINDS = ("request", "no_request_interval", "out_of_scope_request")
ENTRY_STATES = ("hand_open", "hand_closed", "unknown")
REFERENCE_STATUS = ("known", "unknown", "excluded")
# Who wrote a timeline row: the named annotator; an agent draft the
# annotator accepted unchanged; an agent draft whose boundaries the annotator changed. Absent = not recorded
# (ledgers written before the field existed).
ROW_AUTHORS = ("annotator", "agent_draft_accepted", "agent_draft_edited")


@dataclass(frozen=True)
class LandmarkLabel:
    frame_id: str
    participant_id: str
    session_id: str
    side: str
    point: str
    state: str
    u: float | None
    v: float | None
    annotator: str
    marker_pass: str
    note: str = ""
    box: tuple[float, float, float, float] | None = None
    glove_condition: str = "unspecified"

    def __post_init__(self):
        if not all((self.frame_id, self.participant_id, self.session_id, self.annotator)):
            raise ValueError("landmark label needs frame, participant, session and annotator identity")
        if self.side not in SIDES or self.point not in (*FUNCTIONAL_POINTS, "hand_box"):
            raise ValueError("side must be left/right and point one of the three functional points")
        if self.state not in LANDMARK_STATES or self.marker_pass not in MARKER_PASSES:
            raise ValueError("unknown visibility state or marker pass")
        if self.point == "hand_box":
            if self.u is not None or self.v is not None:
                raise ValueError("hand boxes use box coordinates")
            if self.state == "visible":
                if self.box is None or len(self.box) != 4 or not all(isfinite(v) and v >= 0 for v in self.box) or self.box[0] >= self.box[2] or self.box[1] >= self.box[3]:
                    raise ValueError("visible hand needs ordered xyxy box")
            elif self.box is not None:
                raise ValueError("non-visible box must be unavailable")
            return
        if self.box is not None:
            raise ValueError("boxes belong to a hand_box row")
        has_pixel = self.u is not None or self.v is not None
        if self.state == "visible":
            if self.u is None or self.v is None or not all(isfinite(x) and x >= 0 for x in (self.u, self.v)):
                raise ValueError("visible label requires finite nonnegative pixel coordinates")
        elif has_pixel:
            raise ValueError("only visible labels carry pixel coordinates")

    def key(self):
        return (self.participant_id, self.session_id, self.frame_id, self.side, self.point)

    def to_row(self):
        return dict(schema=LANDMARK_SCHEMA, frame_id=self.frame_id, participant_id=self.participant_id,
                    session_id=self.session_id, side=self.side, point=self.point, state=self.state,
                    u=self.u, v=self.v, annotator=self.annotator, marker_pass=self.marker_pass, note=self.note, box=self.box, glove_condition=self.glove_condition)


def label_from_row(row: dict) -> LandmarkLabel:
    if row.get("schema") != LANDMARK_SCHEMA:
        raise ValueError("unknown landmark ledger schema")
    fields = {k: row.get(k) for k in ("frame_id", "participant_id", "session_id", "side", "point",
                                       "state", "u", "v", "annotator", "marker_pass")}
    return LandmarkLabel(note=row.get("note", ""), box=None if row.get("box") is None else tuple(row["box"]),
                         glove_condition=row.get("glove_condition", "unspecified"), **fields)


def read_landmark_ledger(path) -> tuple[LandmarkLabel, ...]:
    """Read JSONL; the last row for a (participant, session, frame, side, point) key wins."""
    labels = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            label = label_from_row(json.loads(line))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        if labels and (label.annotator, label.marker_pass) != (next(iter(labels.values())).annotator, next(iter(labels.values())).marker_pass):
            raise ValueError("use a separate label file for each annotator and marker pass")
        labels[label.key()] = label
    return tuple(labels.values())


def write_landmark_ledger(path, labels) -> None:
    labels = tuple(labels)
    if len({(l.annotator, l.marker_pass) for l in labels}) > 1:
        raise ValueError("use a separate label file for each annotator and marker pass")
    rows = [json.dumps(l.to_row(), sort_keys=True) for l in labels]
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    # Write a sibling temporary file, then replace: a crash mid-write never truncates the labels.
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def landmark_coverage(labels, image_size: tuple[int, int] | None = None) -> dict:
    """Counts by side/point/state, plus out-of-image pixels if an image size is given (width, height)."""
    counts: dict = {}
    out_of_image = 0
    for l in labels:
        counts.setdefault(l.side, {}).setdefault(l.point, {}).setdefault(l.state, 0)
        counts[l.side][l.point][l.state] += 1
        if image_size and l.state == "visible" and l.point != "hand_box" and not (l.u < image_size[0] and l.v < image_size[1]):
            out_of_image += 1
    return {"by_side_point_state": counts, "visible_pixels_outside_image": out_of_image,
            "scope": "label bookkeeping only; agreement, error and coverage are M0 measurements"}


# ----------------------------------------------------------------------------------------------
# Opportunity / reference ledger
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LedgerEntry:
    """One independently defined interval, before conversion into ``Opportunity``."""
    opportunity_id: str
    kind: str
    start_s: float
    end_s: float
    acceptable: tuple[tuple[str, ...] | None, tuple[str, ...] | None]
    provenance: str
    annotator: str
    prefix_only: bool
    entry_state: tuple[str, str]
    onset_uncertainty_s: float | None
    hand_windows_s: tuple[tuple[float, float], tuple[float, float]] | None
    acceptable_joint_pairs: tuple[tuple[str, str], ...] | None
    note: str = ""
    request_ids: tuple[str | None, str | None] | None = None
    reference_status: tuple[str, str] | None = None
    row_author: str | None = None


def session_origin_s(recording_manifest) -> tuple[float, str]:
    """(availability stamp of the first frame, clock name) — the session-relative zero."""
    data = json.loads(Path(recording_manifest).read_text(encoding="utf-8"))
    frames = data.get("frames") or []
    if not frames:
        raise ValueError("recording manifest has no frames")
    first = frames[0]["rgb_stamp"]
    depth = frames[0]["depth_stamp"]
    if first["clock"] != depth["clock"]:
        raise ValueError("recording streams disagree on clock")
    return float(max(first["available_s"], depth["available_s"])), first["clock"]


def frame_time_rows(records) -> list[dict]:
    """Frame index -> timestamps and session-relative seconds, for annotators and the request CSV.

    The origin is the availability stamp of the first converted frame, exactly as ``session_origin_s``
    defines it, so ``session_relative_*`` values can be written straight into ``import_requests.py``
    timelines. ``session_relative_acquired_s`` is when the camera saw the frame; the decision tick made
    from that frame comes later by its availability delay plus processing time (2 Oct 2026).
    """
    records = list(records)
    if not records:
        return []
    first = records[0]
    if first["rgb_stamp"]["clock"] != first["depth_stamp"]["clock"]:
        raise ValueError("recording streams disagree on clock")
    origin = max(first["rgb_stamp"]["available_s"], first["depth_stamp"]["available_s"])
    rows = []
    for k, r in enumerate(records):
        available = max(r["rgb_stamp"]["available_s"], r["depth_stamp"]["available_s"])
        rows.append(dict(index=k, frame_id=r["frame_id"], acquired_s=r["rgb_stamp"]["acquired_s"], available_s=available,
                         session_relative_acquired_s=r["rgb_stamp"]["acquired_s"] - origin,
                         session_relative_available_s=available - origin))
    return rows


def _acceptable(value, kind):
    if value is None:
        return None
    if not isinstance(value, list) or not value or any(v not in OUTCOMES for v in value):
        raise ValueError("acceptable must be null or a nonempty list of outcomes")
    allowed = tuple(value)
    if "NQ" in allowed and len(allowed) > 1:
        raise ValueError("no-request cannot be mixed with request alternatives")
    if kind == "no_request_interval" and allowed != ("NQ",):
        raise ValueError("a no-request interval must reference NQ for every referenced hand")
    if kind in ("request", "out_of_scope_request") and allowed == ("NQ",):
        pass  # the other hand may legitimately have no new request
    return allowed


def read_opportunity_ledger(path) -> dict:
    """Validate an ``sg-opportunity-ledger/1`` file and return its parsed content."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"schema", "participant_id", "session_id", "clock", "opportunities"}
    if not isinstance(data, dict) or set(data) - {"note"} != required or data["schema"] != OPPORTUNITY_SCHEMA:
        raise ValueError("unknown opportunity ledger schema or fields")
    if not all(isinstance(data[k], str) and data[k] for k in ("participant_id", "session_id")):
        raise ValueError("participant and session identity required")
    clock = data["clock"]
    if not isinstance(clock, dict) or clock.get("kind") not in ("session_relative", "trace") \
            or not isinstance(clock.get("name"), str) or not clock["name"]:
        raise ValueError("clock must declare kind session_relative|trace and the trace clock name")
    entries = []
    seen = set()
    for i, raw in enumerate(data["opportunities"]):
        try:
            entry = _entry(raw)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"opportunity #{i}: {exc}") from exc
        if entry.opportunity_id in seen:
            raise ValueError(f"duplicate opportunity_id {entry.opportunity_id}")
        seen.add(entry.opportunity_id)
        entries.append(entry)
    if len({e.annotator for e in entries}) > 1 or len({e.prefix_only for e in entries}) > 1:
        raise ValueError("separate opportunity files required per annotator and viewing pass")
    return dict(participant_id=data["participant_id"], session_id=data["session_id"], clock=clock,
                entries=tuple(entries), note=data.get("note", ""))


def _entry(raw: dict) -> LedgerEntry:
    kind = raw["kind"]
    if kind not in OPPORTUNITY_KINDS:
        raise ValueError("unknown opportunity kind")
    start, end = float(raw["start_s"]), float(raw["end_s"])
    if not (isfinite(start) and isfinite(end)) or not 0 <= start <= end:
        raise ValueError("interval must be finite, nonnegative and ordered")
    acc = raw["acceptable"]
    if not isinstance(acc, dict) or set(acc) != set(SIDES):
        raise ValueError("acceptable must map left and right")
    acceptable = (_acceptable(acc["left"], kind), _acceptable(acc["right"], kind))
    if kind == "out_of_scope_request" and not any(a and set(a) - {"NQ"} for a in acceptable):
        raise ValueError("out-of-scope request must name at least one requested action")
    entry_state = raw.get("entry_state", {"left": "unknown", "right": "unknown"})
    if set(entry_state) != set(SIDES) or any(entry_state[s] not in ENTRY_STATES for s in SIDES):
        raise ValueError("entry_state must map left/right to hand_open|hand_closed|unknown")
    onset = raw.get("onset_uncertainty_s")
    if onset is not None and (not isfinite(float(onset)) or float(onset) < 0):
        raise ValueError("onset uncertainty must be a nonnegative number of seconds")
    windows = raw.get("hand_windows_s")
    if windows is not None:
        windows = tuple(tuple(float(t) for t in windows[s]) for s in SIDES)
    pairs = raw.get("acceptable_joint_pairs")
    if pairs is not None:
        pairs = tuple(tuple(p) for p in pairs)
    prefix_only = raw["prefix_only"]
    if type(prefix_only) is not bool:
        raise ValueError("prefix_only must be true or false; retrospective labels are not prefix-only")
    if not raw.get("provenance") or not raw.get("annotator"):
        raise ValueError("provenance and annotator required")
    request_ids = raw.get("request_ids")
    if request_ids is not None:
        if not isinstance(request_ids, dict) or set(request_ids) != set(SIDES):
            raise ValueError("request_ids must map left and right")
        request_ids = tuple(request_ids[s] for s in SIDES)
        for rid, allowed in zip(request_ids, acceptable):
            if rid is not None and (not isinstance(rid, str) or not rid):
                raise ValueError("a request id must be a nonempty string or null")
            if rid is not None and (allowed is None or allowed == ("NQ",)):
                raise ValueError("a request id is allowed only for a hand with a known request")
    row_author = raw.get("row_author")
    if row_author is not None and row_author not in ROW_AUTHORS:
        raise ValueError(f"row_author must be one of {ROW_AUTHORS}, got {row_author!r}")
    status = raw.get("reference_status")
    if status is not None:
        if not isinstance(status, dict) or set(status) != set(SIDES) or any(status[s] not in REFERENCE_STATUS for s in SIDES):
            raise ValueError("reference_status must map left/right to known|unknown|excluded")
        status = tuple(status[s] for s in SIDES)
        if any((st == "known") != (allowed is not None) for st, allowed in zip(status, acceptable)):
            raise ValueError("reference_status must be known exactly when the hand has an acceptable set")
    return LedgerEntry(str(raw["opportunity_id"]), kind, start, end, acceptable, str(raw["provenance"]),
                       str(raw["annotator"]), prefix_only, (entry_state["left"], entry_state["right"]),
                       None if onset is None else float(onset), windows, pairs, str(raw.get("note", "")),
                       request_ids, status, row_author)


def to_opportunities(ledger: dict, origin_s: float | None = None, admitted_actions=("G",)):
    """Convert ledger entries into ``Opportunity`` objects on the trace clock.

    Returns ``(in_scope, out_of_scope)``. Unsupported hands become unknown in the scored
    copy; the original is retained in out_of_scope. Thus a left release cannot discard a
    right grasp. The two collections can contain different versions of the same interval;
    never score both together. Unsupported actions are never relabelled no-request.
    """
    clock = ledger["clock"]
    if clock["kind"] == "session_relative":
        if origin_s is None or not isfinite(origin_s):
            raise ValueError("session-relative ledger needs the recording's first-frame availability origin")
        shift = origin_s
    else:
        shift = 0.
    in_scope, out_of_scope = [], []
    for e in ledger["entries"]:
        windows = None if e.hand_windows_s is None else tuple((w[0] + shift, w[1] + shift) for w in e.hand_windows_s)
        allowed = tuple(None if a is not None and any(v != "NQ" and v not in admitted_actions for v in a)
                        else a for a in e.acceptable)
        # Scope is applied per hand: release on the left must not discard a right grasp.
        pairs = e.acceptable_joint_pairs if all(a is not None for a in allowed) else None
        provenance = f"{e.provenance}|annotator={e.annotator}|prefix_only={e.prefix_only}|kind={e.kind}"
        if e.row_author is not None:  # only when recorded, so older ledgers keep their provenance strings
            provenance += f"|row_author={e.row_author}"
        status = e.reference_status or tuple("unknown" if a is None else "known" for a in e.acceptable)
        reasons = tuple(st if original is None else "unsupported_action" if a is None else "known"
                        for st, original, a in zip(status, e.acceptable, allowed))
        ids = e.request_ids or (None, None)
        scored_ids = tuple(None if a is None else rid for rid, a in zip(ids, allowed))
        op = Opportunity(e.opportunity_id, ledger["participant_id"], ledger["session_id"],
                         e.start_s + shift, e.end_s + shift, allowed, provenance,
                         hand_windows_s=windows, acceptable_joint_pairs=pairs,
                         request_ids=scored_ids if any(scored_ids) else None, reference_reasons=reasons)
        # Keep fully unknown/excluded intervals too, so coverage is not mislabeled as a gap.
        in_scope.append(op)
        if allowed != e.acceptable or e.kind == "out_of_scope_request":
            out_of_scope.append(Opportunity(e.opportunity_id, ledger["participant_id"], ledger["session_id"],
                e.start_s + shift, e.end_s + shift, e.acceptable, provenance,
                hand_windows_s=windows, acceptable_joint_pairs=e.acceptable_joint_pairs,
                # Ids only for the hands masked out of the scored copy; the other hand is scored there.
                request_ids=tuple(rid if a is None else None for rid, a in zip(ids, allowed))
                if any(rid is not None and a is None for rid, a in zip(ids, allowed)) else None,
                reference_reasons=tuple(status)))
    return tuple(in_scope), tuple(out_of_scope)


def ledger_summary(ledger: dict) -> dict:
    counts = {k: 0 for k in OPPORTUNITY_KINDS}
    prefix_only = 0
    exposure_s = 0.
    for e in ledger["entries"]:
        counts[e.kind] += 1
        prefix_only += int(e.prefix_only)
        if e.kind == "no_request_interval":
            exposure_s += e.end_s - e.start_s
    hand_exposure = sum((e.end_s-e.start_s) for e in ledger["entries"] for a in e.acceptable if a == ("NQ",))
    return dict(counts=counts, prefix_only_entries=prefix_only, no_request_exposure_s=exposure_s,
                no_request_exposure_hand_s=hand_exposure,
                scope="ledger bookkeeping; exposure is labelled time at risk, not an observed event rate")
