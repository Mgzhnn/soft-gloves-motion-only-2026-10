"""Choose ``Config.rate_window`` once on the development clip and freeze it.

Replays the stored packets of a deterministic perception run with each candidate window, scores every trace against
the annotated ledger and applies a selection rule fixed BEFORE any number is seen:
  eligible windows: every scored request hand receives at least one proposal tick inside its acceptable window on the
  primary run (a window that silences the baseline is not selectable: the silent-point rule of 2 October);
  choice: the eligible window with the fewest unwanted proposal onsets per no-request hand-minute on the primary run;
  ties -> the smaller window.
A second run (another object provider) is reported as a check only. The strict early tolerance (0 s) is used for the
choice; the sensitivity to the other tolerances is reported for the pending tolerance decision, not used. The replay
at the window the stored run used must reproduce the stored per-tick statuses, which is checked and recorded.
usage: choose_rate_window.py RECORDING_JSON LEDGER_JSON OUT_STEM --primary-run DIR [--check-run DIR] [--windows 1,2,3,4,6,8] [--tolerances 0,0.1,0.2]
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from score_requests import request_rows, validate_score_inputs
from sg_offline.evaluation import evaluate_opportunities
from sg_offline.inputs import packets_from_jsonl
from sg_offline.ledger import read_opportunity_ledger, session_origin_s, to_opportunities
from sg_offline.replay import Config, Replay
from sg_offline.stats import per_participant_success, unwanted_exposure

SCHEMA = "sg-rate-window-choice/1"
RULE = ("eligible: every scored request hand gets at least one proposal tick inside its window on the primary run; "
        "choice: the eligible window with the fewest unwanted proposal onsets per no-request hand-minute on the primary run; ties -> the smaller window; "
        "strict early tolerance 0 s for the choice")


def _score(rows, inside, origin, early):
    outcomes = evaluate_opportunities(rows, inside, early_limit_s=early)
    pp = next(iter(per_participant_success(outcomes).values())); ex = next(iter(unwanted_exposure(outcomes, inside).values()))
    return dict(compatible_first=pp["compatible_first"], wrong_timing_first_early=pp.get("wrong_timing_first_early", 0),
                wrong_timing_first_late=pp.get("wrong_timing_first_late", 0), evaluable=pp["evaluable_request_hands"],
                unwanted_ticks=ex["unwanted_proposals"], unwanted_onsets=ex["unwanted_onsets"], onsets_per_hand_minute=round(ex["onsets_per_hand_minute"], 1),
                ticks_per_hand_minute=round(ex["proposals_per_hand_minute"], 1), intervals_with_unwanted=ex["intervals_with_unwanted"],
                no_request_hand_intervals=ex["no_request_hand_intervals"], requests=request_rows(outcomes, origin))


def choose(manifest, ledger_path, primary_run, check_run=None, windows=(1, 2, 3, 4, 6, 8), tolerances=(0., .1, .2)):
    manifest, ledger_path = Path(manifest), Path(ledger_path)
    origin, clock = session_origin_s(manifest)
    ledger = read_opportunity_ledger(ledger_path)
    if ledger["clock"]["name"] != clock:
        raise ValueError("ledger clock differs from the recording clock")
    inside, _ = to_opportunities(ledger, origin)
    result = dict(schema=SCHEMA, rule=RULE, windows=list(windows), tolerances=list(tolerances), ledger=str(ledger_path.resolve()), manifest=str(manifest.resolve()), runs={})
    for name, run in (("primary", primary_run), ("check", check_run)):
        if run is None:
            continue
        run = Path(run)
        packets = tuple(packets_from_jsonl(run / "packets.jsonl"))
        stored = [json.loads(l) for l in (run / "trace.jsonl").read_text().splitlines() if l.strip()]
        origin, clock, inside, _ = validate_score_inputs(manifest, ledger, stored)
        if stored[0].get("model") != "motion_independent":
            raise ValueError("rate-window selection supports the recorded motion_independent rules baseline only")
        source_config = Config(**stored[0]["configuration"])
        stored_window = source_config.rate_window
        entry = dict(run_dir=str(run.resolve()), stored_rate_window=stored_window, by_window={})
        for k in sorted(set(windows) | {stored_window}):
            rows = Replay(replace(source_config, rate_window=k)).run(packets)
            validate_score_inputs(manifest, ledger, rows)
            if k == stored_window:
                entry["stored_window_reproduces_trace_statuses"] = len(rows) == len(stored) and all(
                    r["packet_id"] == s["packet_id"] and r["decision_s"] == s["decision_s"] and
                    [(o["side"], o["status"], o["reason"], o["raw_outcome"]) for o in r["outputs"]] ==
                    [(o["side"], o["status"], o["reason"], o["raw_outcome"]) for o in s["outputs"]] for r, s in zip(rows, stored))
                if not entry["stored_window_reproduces_trace_statuses"]:
                    raise ValueError("stored-window replay does not reproduce the source trace; no window may be selected")
            if k in windows:
                entry["by_window"][str(k)] = {str(t): _score(rows, inside, origin, t) for t in tolerances}
        result["runs"][name] = entry
    primary = result["runs"]["primary"]["by_window"]
    eligible = [k for k in windows if primary[str(k)]["0.0"]["requests"] and all((r["proposals_in_window"] or 0) >= 1 for r in primary[str(k)]["0.0"]["requests"])]
    result["eligible_windows"] = eligible
    result["chosen_rate_window"] = min(eligible, key=lambda k: (primary[str(k)]["0.0"]["onsets_per_hand_minute"], k)) if eligible else None
    result["chosen_is_at_the_edge_of_the_candidate_set"] = result["chosen_rate_window"] in (min(windows), max(windows)) if eligible else None
    return result


def markdown(result):
    lines = ["| run | K | compatible first | early first | evaluable | proposals inside each window | unwanted onsets / hand-min | unwanted ticks / hand-min | no-request intervals with unwanted |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, entry in result["runs"].items():
        for k in result["windows"]:
            s = entry["by_window"][str(k)]["0.0"]
            lines.append(f"| {name} | {k} | {s['compatible_first']} | {s['wrong_timing_first_early']} | {s['evaluable']} | {' / '.join(str(r['proposals_in_window']) for r in s['requests']) or '-'} | "
                         f"{s['onsets_per_hand_minute']} | {s['ticks_per_hand_minute']} | {s['intervals_with_unwanted']} of {s['no_request_hand_intervals']} |")
    k = result["chosen_rate_window"]
    lines += ["", f"Rule: {result['rule']}.", f"Eligible windows: {result['eligible_windows']}. Chosen: K = {k}" + (" (at the edge of the candidate set)" if result.get("chosen_is_at_the_edge_of_the_candidate_set") else "") + "."]
    if k is not None:
        by = result["runs"]["primary"]["by_window"][str(k)]
        lines.append("Sensitivity to the early tolerance at the chosen window (primary run): " + "; ".join(f"{t} s -> compatible first {by[str(t)]['compatible_first']} of {by[str(t)]['evaluable']}" for t in result["tolerances"]) + ".")
    for name, entry in result["runs"].items():
        lines.append(f"Replay at the stored window ({entry['stored_rate_window']}) reproduces the stored trace statuses, {name} run: {entry.get('stored_window_reproduces_trace_statuses')}.")
    lines.append("Scope: counts of software behaviour on one session against one annotator's timeline; not accuracy.")
    return "\n".join(lines)


def _ints(text):
    return tuple(int(v) for v in text.split(","))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", type=Path); ap.add_argument("ledger", type=Path); ap.add_argument("out_stem", type=Path)
    ap.add_argument("--primary-run", type=Path, required=True); ap.add_argument("--check-run", type=Path)
    ap.add_argument("--windows", default="1,2,3,4,6,8"); ap.add_argument("--tolerances", default="0,0.1,0.2")
    a = ap.parse_args(argv)
    windows = _ints(a.windows); tolerances = tuple(float(v) for v in a.tolerances.split(","))
    if any(k < 1 for k in windows) or len(set(windows)) != len(windows) or any(t < 0 for t in tolerances) or 0. not in tolerances:
        ap.error("windows must be distinct positive integers and the tolerances must include 0")
    for suffix in (".json", ".md"):
        if Path(str(a.out_stem) + suffix).exists():
            raise FileExistsError(f"report already exists: {a.out_stem}{suffix}")
    result = choose(a.manifest, a.ledger, a.primary_run, a.check_run, windows, tolerances)
    a.out_stem.parent.mkdir(parents=True, exist_ok=True)
    Path(str(a.out_stem) + ".json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    text = markdown(result); Path(str(a.out_stem) + ".md").write_text(text + "\n"); print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
