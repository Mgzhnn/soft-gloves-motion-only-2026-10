"""Validate explicit per-hand timeline CSV annotations into a session opportunity ledger.

Required columns: start_s,end_s,left,right. Values: NQ, G (or another named action),
UNKNOWN, EXCLUDED; alternatives use |. Optional: left_start_s,left_end_s,right_start_s,
right_end_s,left_request_id,right_request_id,note,row_author. Any other column is an error (a typo must
not silently disappear). row_author records who wrote each row (annotator, agent_draft_accepted,
agent_draft_edited); when the column is present every row needs one, when it is absent the output is unchanged. Intervals partition [0,duration) without gaps. Exclusions require a note.

One hand request may cross several rows (for example when the other hand starts holding).
The same request continues automatically while that hand keeps the same label and its window
stays contiguous; it is then scored once, not once per row. Give an explicit
left_request_id/right_request_id to split back-to-back requests with the same label.
Rows where both hands are NQ are written as no-request intervals. UNKNOWN and EXCLUDED stay
distinct in the ledger (reference_status); neither ever becomes NQ.
This importer does not claim to enforce prefix-only viewing; its labels are retrospective.
"""
import argparse, csv, json
from math import isfinite
from pathlib import Path
from sg_offline.ledger import OPPORTUNITY_SCHEMA, ROW_AUTHORS, read_opportunity_ledger, to_opportunities
from sg_offline.evaluation import BOUNDARY_TOLERANCE_S as TOL


REQUIRED_COLUMNS = {"start_s", "end_s", "left", "right"}
OPTIONAL_COLUMNS = {"left_start_s", "left_end_s", "right_start_s", "right_end_s",
                    "left_request_id", "right_request_id", "note", "row_author"}
UNSCORED = ("UNKNOWN", "EXCLUDED")


def _is_request(value):
    return value not in ("NQ",) + UNSCORED


def convert(rows, participant, session, annotator, clock, duration):
    if not isfinite(duration) or duration <= 0:
        raise ValueError("positive finite duration required")
    entries = []; previous = 0.
    last = {side: dict(label=None, end=None, rid=None) for side in ("left", "right")}
    used = {side: set() for side in ("left", "right")}; counter = {side: 0 for side in ("left", "right")}
    for i, row in enumerate(rows):
        if None in row:
            raise ValueError(f"row {i}: more cells than header columns")
        columns = set(row)
        if REQUIRED_COLUMNS - columns:
            raise ValueError(f"missing required columns: {sorted(REQUIRED_COLUMNS - columns)}")
        if columns - REQUIRED_COLUMNS - OPTIONAL_COLUMNS:
            raise ValueError(f"unknown timeline columns: {sorted(columns - REQUIRED_COLUMNS - OPTIONAL_COLUMNS)}")
        start, end = float(row["start_s"]), float(row["end_s"])
        if abs(start-previous) > TOL:
            raise ValueError("rows must partition the timeline without overlap or gaps; label unknown time explicitly")
        # A start within the tolerance of the previous end is written as exactly that end: the scorer compares
        # interval bounds exactly, and rows 0-2.0000005 / 2.0-4 would be imported and then refused there.
        declared_start, start = start, previous
        if end <= start:
            raise ValueError("rows must partition the timeline without overlap or gaps; label unknown time explicitly")
        labels = {side: (row[side] or "").strip() for side in ("left", "right")}
        acceptable = {side: None if labels[side] in UNSCORED else labels[side].split("|") for side in ("left", "right")}
        status = {side: "unknown" if labels[side] == "UNKNOWN" else "excluded" if labels[side] == "EXCLUDED" else "known"
                  for side in ("left", "right")}
        if "EXCLUDED" in labels.values() and not row.get("note"):
            raise ValueError("per-hand exclusions need a reason in note")
        windows = {}
        for side in ("left", "right"):
            if (row.get(side+"_start_s") or row.get(side+"_end_s")) and not _is_request(labels[side]):
                # A window on a no-request or unscored hand has no meaning and gave a negative first-unwanted time.
                raise ValueError(f"row {i}: {side} hand window given for a hand without a known request")
            windows[side] = [float(row.get(side+"_start_s") or start), float(row.get(side+"_end_s") or end)]
            if abs(windows[side][0] - declared_start) <= TOL:
                windows[side][0] = start  # a window written at the row's declared start follows the row
        request_ids = {}
        for side in ("left", "right"):
            explicit = (row.get(side+"_request_id") or "").strip()
            key = tuple(sorted(labels[side].split("|"))) if _is_request(labels[side]) else None
            if key is None:
                if explicit:
                    raise ValueError(f"row {i}: {side}_request_id given for a hand without a known request")
                rid = None
            elif explicit:
                if explicit.startswith("auto:"):
                    raise ValueError("explicit request ids must not start with 'auto:'")
                rid = explicit
            elif last[side]["rid"] and last[side]["label"] == key and abs(last[side]["end"] - windows[side][0]) <= TOL:
                rid = last[side]["rid"]  # the same request continues across a row boundary
            else:
                counter[side] += 1; rid = f"auto:{side}:{counter[side]:04d}"
            if rid is not None and rid != last[side]["rid"]:
                if rid in used[side]:
                    raise ValueError(f"row {i}: {side} request id {rid!r} reused after a different request")
                used[side].add(rid)
            if rid is not None and rid == last[side]["rid"] and (last[side]["label"] != key
                                                                  or abs(last[side]["end"] - windows[side][0]) > TOL):
                raise ValueError(f"row {i}: {side} request {rid!r} must keep its label and a contiguous window")
            request_ids[side] = rid
            last[side] = dict(label=key, end=windows[side][1], rid=rid)
        kind = "no_request_interval" if labels["left"] == "NQ" and labels["right"] == "NQ" else "request"
        entry = dict(opportunity_id=f"interval-{i:04d}", kind=kind, start_s=start, end_s=end,
            acceptable=acceptable, hand_windows_s=windows, reference_status=status,
            provenance="human_timeline_csv_retrospective",
            annotator=annotator, prefix_only=False, note=row.get("note", "") or "")
        if any(request_ids.values()):
            entry["request_ids"] = request_ids
        if "row_author" in columns:  # who wrote this row; absent column = output unchanged
            author = (row["row_author"] or "").strip()
            if author not in ROW_AUTHORS:
                raise ValueError(f"row {i}: row_author must be one of {ROW_AUTHORS}, got {author!r}")
            entry["row_author"] = author
        entries.append(entry)
        previous = end
    if not entries or abs(previous-duration) > TOL:
        raise ValueError("timeline must end at the declared recording duration")
    return dict(schema=OPPORTUNITY_SCHEMA,participant_id=participant,session_id=session,
                clock=dict(kind="session_relative",name=clock),opportunities=entries)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv",type=Path);ap.add_argument("output",type=Path)
    for name in ("participant","session","annotator","clock"):
        ap.add_argument("--"+name,required=True)
    ap.add_argument("--duration-s",type=float,required=True)
    a=ap.parse_args(argv)
    if a.output.exists(): ap.error("use a fresh file for each annotator/pass")
    # utf-8-sig: a spreadsheet's "CSV UTF-8" export begins with a byte-order mark, which otherwise ends up in the first
    # header name and was reported as "missing required columns: ['start_s']".
    with a.csv.open(encoding="utf-8-sig",newline="") as f: data=convert(list(csv.DictReader(f)),a.participant,a.session,a.annotator,a.clock,a.duration_s)
    # Validate before publishing the result; use the same entry validation as the reader.
    from sg_offline.ledger import _entry
    from sg_offline.evaluation import Opportunity
    for row in data["opportunities"]:
        e=_entry(row)
        Opportunity(e.opportunity_id,a.participant,a.session,e.start_s,e.end_s,e.acceptable,e.provenance,e.hand_windows_s)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,indent=2)+"\n")
    ledger=read_opportunity_ledger(a.output); inside,outside=to_opportunities(ledger,0.)
    print(json.dumps(dict(intervals=len(inside),intervals_with_unsupported_hands=len(outside),label_mode="retrospective")))

if __name__=="__main__": main()
