"""Score one session's replay trace against an independently annotated request ledger.

Inputs: the recording manifest (session origin and clock), a perception run folder (its ``trace.jsonl``) and the
opportunity ledger written by ``import_requests.py``. Output: one JSON report with the ledger summary, one row per
scored request hand (first proposal, run start, early-by, proposals inside the window), the per-person success
accounting, the unwanted-proposal exposure of the labelled no-request time, the timeline accounting and provenance
(hashes of the inputs, the replay configuration and implementation version read from the trace). One session of one
person: no comparison, no training, no interval. ``run_comparison.py`` remains the multi-person route.
usage: score_requests.py RECORDING_JSON RUN_DIR LEDGER_JSON OUT_JSON [--early-limit-s 0] [--scope TEXT]
                         [--admitted-actions G[,R,...]] [--endpoint proposals|accepted_commands]
``--admitted-actions`` names the actions scored as requests (default G, the primary endpoint; an action not
named stays unsupported and its hand unscored, never no-request). ``--endpoint accepted_commands`` scores the run's
``commands.jsonl`` (written by ``replay_packets.py --fsm``) instead of the proposals: the accepted-command SECONDARY endpoint.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

from sg_offline.contracts import ACTIONS
from sg_offline.evaluation import evaluate_opportunities, timeline_accounting
from sg_offline.fsm import Command, command_rows
from sg_offline.ledger import ledger_summary, read_opportunity_ledger, session_origin_s, to_opportunities
from sg_offline.stats import per_participant_success, unwanted_exposure
from sg_offline.recordings import RGBDRecording

SIDES = ("left", "right")
SCHEMA = "sg-step3-request-scoring/1"
ENDPOINTS = ("proposals", "accepted_commands")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def request_rows(outcomes, origin):
    """One row per scored request hand (continuation rows excluded), times in session-relative seconds."""
    rows = []
    for o in outcomes:
        for i, h in enumerate(o["hands"]):
            actions = list(h.get("accepted_actions") or [])
            if not h.get("request_id") or h.get("result") == "continuation_of_request" or actions == ["NQ"]:
                continue
            window = h.get("acceptable_window_s")
            rows.append(dict(side=SIDES[i], request_id=h["request_id"], opportunity_id=o["opportunity_id"],
                             window_start_s=round(window[0] - origin, 3) if window else None, window_end_s=round(window[1] - origin, 3) if window else None,
                             accepted_actions=actions, result=h["result"],
                             first_proposal_s=round(h["first_proposal"][0] - origin, 3) if h.get("first_proposal") else None,
                             proposal_run_start_s=round(h["proposal_run_start_s"] - origin, 3) if h.get("proposal_run_start_s") is not None else None,
                             early_by_s=h.get("early_by_s"), run_active_at_window_start=h.get("run_active_at_window_start"),
                             proposals_in_window=h.get("proposals_in_window"), proposal_runs=h.get("proposal_runs"),
                             proposals_since_rearm_before_window=h.get("proposals_since_rearm_before_window"),
                             eligible_decision_ticks=h.get("eligible_decision_ticks")))
    return rows


def validate_score_inputs(manifest, ledger, rows, admitted_actions=("G",)):
    """Bind every scored row to one recording and ledger before interpreting relative times."""
    recording = RGBDRecording(manifest)
    origin, clock = session_origin_s(manifest)
    if not rows:
        raise ValueError("empty trace")
    identity = (recording.participant_id, recording.session_id)
    if (ledger["participant_id"], ledger["session_id"]) != identity:
        raise ValueError("ledger identity differs from the recording")
    if ledger["clock"]["name"] != clock:
        raise ValueError("ledger clock differs from the recording clock")
    for row in rows:
        inputs = row["inputs"]
        if (inputs["participant_id"], inputs["session_id"]) != identity or inputs["clock"] != clock:
            raise ValueError("trace identity or clock differs from the recording")
        if tuple(o.get("side") for o in row["outputs"]) != SIDES:
            raise ValueError("trace outputs must be in left, right side order")
        if any(o.get("status") not in ("withheld", "no_new_request", "proposed_simulation_only") for o in row["outputs"]):
            raise ValueError("unknown trace output status")
        if any(row.get(k) != rows[0].get(k) for k in ("configuration", "model", "implementation_version")):
            raise ValueError("trace configuration, model and implementation must be uniform")
    inside, outside = to_opportunities(ledger, origin, admitted_actions)
    accounting = timeline_accounting(rows, inside)
    if inside and accounting.get("unlabelled_hand_ticks", 0) == accounting.get("all_hand_ticks", 0):
        raise ValueError("no decision tick falls inside any labelled interval; check recording origin and clock")
    return origin, clock, inside, outside


def score(manifest, run_dir, ledger_path, early_limit_s=0., scope="", admitted_actions=("G",), endpoint="proposals"):
    manifest, run_dir, ledger_path = Path(manifest), Path(run_dir), Path(ledger_path)
    if endpoint not in ENDPOINTS:
        raise ValueError(f"endpoint must be one of {ENDPOINTS}")
    admitted_actions = tuple(admitted_actions)
    if not admitted_actions or any(a not in ACTIONS for a in admitted_actions):
        raise ValueError(f"admitted actions must be a nonempty subset of {ACTIONS}")
    trace = run_dir / "trace.jsonl"
    rows = [json.loads(l) for l in trace.read_text().splitlines() if l.strip()]
    ledger = read_opportunity_ledger(ledger_path)
    origin, clock, inside, outside = validate_score_inputs(manifest, ledger, rows, admitted_actions)
    commands_path = run_dir / "commands.jsonl"
    if endpoint == "accepted_commands":
        if not commands_path.exists():
            raise FileNotFoundError(f"{commands_path} is missing: run replay_packets.py --fsm first")
        commands = [Command(**json.loads(l)) for l in commands_path.read_text().splitlines() if l.strip()]
        rows = command_rows(rows, commands)
    outcomes = evaluate_opportunities(rows, inside, early_limit_s=early_limit_s)
    provenance_path = run_dir / "provenance.json"
    report = dict(schema=SCHEMA, scope=scope, made_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                  manifest=str(manifest.resolve()), run_dir=str(run_dir.resolve()), ledger=str(ledger_path.resolve()),
                  session_origin_s=origin, clock=clock, early_limit_s=early_limit_s, ledger_summary=ledger_summary(ledger),
                  admitted_actions=list(admitted_actions), endpoint=endpoint,
                  commands_sha256=sha256_file(commands_path) if endpoint == "accepted_commands" else None,
                  trace_rows=len(rows), opportunities_in_scope=len(inside), opportunities_out_of_scope=len(outside),
                  requests=request_rows(outcomes, origin), per_participant_success=per_participant_success(outcomes),
                  unwanted_exposure=unwanted_exposure(outcomes, inside), timeline_accounting=timeline_accounting(rows, inside),
                  provenance=dict(trace_sha256=sha256_file(trace), ledger_sha256=sha256_file(ledger_path), manifest_sha256=sha256_file(manifest),
                                  run_provenance=str(provenance_path.resolve()) if provenance_path.exists() else None,
                                  run_provenance_sha256=sha256_file(provenance_path) if provenance_path.exists() else None,
                                  replay_configuration=rows[0].get("configuration"), implementation_version=rows[0].get("implementation_version"),
                                  model=rows[0].get("model")),
                  limits="counts of software behaviour against one annotator's timeline on one session; not accuracy; "
                         "the burden is per labelled no-request hand-minute and depends on the decision cadence")
    return report


def summary_line(report):
    pp = next(iter(report["per_participant_success"].values()), {})
    ex = next(iter(report["unwanted_exposure"].values()), {})
    return (f"requests {pp.get('evaluable_request_hands', 0)}: compatible first {pp.get('compatible_first', 0)}, early {pp.get('wrong_timing_first_early', 0)}, "
            f"late {pp.get('wrong_timing_first_late', 0)}, wrong action {pp.get('wrong_first', 0)}, missed {pp.get('missed_including_withheld', 0)} | "
            f"no-request {ex.get('exposure_hand_s', 0.):.1f} hand-s: {ex.get('unwanted_onsets', 0)} unwanted onsets = {ex.get('onsets_per_hand_minute', 0.):.1f} per hand-minute")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", type=Path); ap.add_argument("run_dir", type=Path); ap.add_argument("ledger", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--early-limit-s", type=float, default=0.); ap.add_argument("--scope", default="")
    ap.add_argument("--admitted-actions", default="G", help="comma-separated actions scored as requests (default G)")
    ap.add_argument("--endpoint", default="proposals", choices=ENDPOINTS)
    a = ap.parse_args(argv)
    if a.early_limit_s < 0:
        ap.error("the early tolerance must be nonnegative")
    if a.out.exists():
        raise FileExistsError(f"report already exists: {a.out}")
    admitted = tuple(x.strip() for x in a.admitted_actions.split(",") if x.strip())
    report = score(a.manifest, a.run_dir, a.ledger, a.early_limit_s, a.scope, admitted, a.endpoint)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(summary_line(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
