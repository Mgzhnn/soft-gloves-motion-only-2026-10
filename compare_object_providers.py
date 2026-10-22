"""Removal test for object perception: compare run_perception outputs made with different
object providers on IDENTICAL frames (same recording, same --decimate, --queue-policy fixed_latency).

For every run directory: object availability per identity, unavailability reasons, 3-D positions. Across runs, an
object is paired with the SAME object of the other run: by identity name when both runs use the same names, and
otherwise (the plane provider's generic identities) one-to-one by nearest 3-D position within a gate that must be
smaller than the spacing of distinct objects (default 0.05 m; the recorded cylinders stand 0.09-0.10 m apart). The
position agreement and the downstream replay outputs (proposal ticks per hand, first proposal time) are reported.
Pairing by nearest-within-0.15 m whatever the name would pair a lone blue cylinder of one
run with the cream cylinder of the other. Runs of different recordings, and run folders written before the
diagnostics carried ``frame_index`` and ``object_xyz``, are refused. This is software behaviour on one recording, not
accuracy: no reference object positions exist.

usage: compare_object_providers.py REPORT_OUT.json RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from sg_offline.appearance import global_assignment

DEFAULT_MAX_MATCH_M = 0.05  # position pairing only; below the spacing of distinct objects, above same-object disagreement


def load_run(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text())
    diag = [json.loads(l) for l in (run_dir / "perception_diagnostics.jsonl").read_text().splitlines() if l.strip()]
    rows = [json.loads(l) for l in (run_dir / "trace.jsonl").read_text().splitlines() if l.strip()]
    provider = summary.get("object_provider", {}).get("kind", "unknown")
    absent = [key for key in ("frame_index", "object_xyz") if any(key not in d for d in diag)]
    if absent:
        # Joined by row number, such a folder would be reported with zero objects.
        raise ValueError(f"{run_dir}: perception_diagnostics.jsonl carries no {' / '.join(absent)} (a run written before the diagnostics carried them); "
                         "it cannot be compared, rerun perception for it")
    frames = {d["frame_index"]: d for d in diag}
    identities = sorted({o for d in diag for o in d.get("objects", {})})
    availability = {o: sum(1 for d in diag if d.get("object_xyz", {}).get(o) is not None) for o in identities}
    reasons = {o: Counter(d["objects"][o] for d in diag if o in d.get("objects", {}) and d.get("object_xyz", {}).get(o) is None) for o in identities}
    # session-relative seconds when the run wrote frame_times.csv (origin = first frame availability); else trace seconds
    origin = 0.
    times = run_dir / "frame_times.csv"
    if times.exists():
        import csv
        with open(times, newline="") as f:
            first_row = next(csv.DictReader(f), None)
        if first_row and "available_s" in first_row:
            origin = float(first_row["available_s"])
    proposals = Counter(); first = {}
    for r in rows:
        for out in r["outputs"]:
            if out["status"] == "proposed_simulation_only":
                proposals[out["side"]] += 1
                first.setdefault(out["side"], round(r["decision_s"] - origin, 3))
    return dict(run_dir=str(run_dir), provider=provider, frames=len(diag), identities=identities, availability=availability,
                frames_with_any_object=sum(1 for d in diag if any(v is not None for v in d.get("object_xyz", {}).values())),
                frames_with_two_objects=sum(1 for d in diag if sum(v is not None for v in d.get("object_xyz", {}).values()) >= 2),
                unavailable_reasons={o: dict(c) for o, c in reasons.items()}, proposal_ticks=dict(proposals), first_proposal_s=first,
                replay_counts=summary.get("evaluation", {}).get("counts", {}), frame_table=frames, time_origin_s=origin,
                manifest_sha256=summary.get("manifest_sha256"),
                object_processing_s_median=summary.get("object_processing_s_median"), queue_policy=summary.get("queue_policy"),
                fixed_latency_s=summary.get("fixed_latency_s"), decimate=summary.get("decimate"))


def _pairs_by_position(pa: dict, pb: dict, gate_m: float) -> list:
    """One-to-one (name_a, name_b, distance) pairs minimising the total distance; an object stays unpaired when its
    best partner is not closer than ``gate_m``."""
    names_a, names_b = sorted(pa), sorted(pb)
    if not names_a or not names_b:
        return []
    costs = [[float(np.linalg.norm(pa[x] - pb[y])) for y in names_b] for x in names_a]
    assignment, _ = global_assignment(costs, unmatched_cost=gate_m)  # exact; at most four objects per run and frame
    return [(names_a[i], names_b[j], costs[i][j]) for i, j in enumerate(assignment) if j is not None]


def agreement(a: dict, b: dict, max_match_m=DEFAULT_MAX_MATCH_M) -> dict:
    """Per frame, pair each available object of ``a`` with the same object of ``b``; report distances and coverage.

    Same identity names in both runs: paired by name, every such pair is reported whatever its distance. Different
    names: one-to-one by nearest position within ``max_match_m``."""
    shared = sorted(set(a["frame_table"]) & set(b["frame_table"]))
    by_name = bool(a["identities"]) and set(a["identities"]) == set(b["identities"])
    distances = []; only_a = only_b = both = 0
    for f in shared:
        pa = {o: np.asarray(v) for o, v in a["frame_table"][f].get("object_xyz", {}).items() if v is not None}
        pb = {o: np.asarray(v) for o, v in b["frame_table"][f].get("object_xyz", {}).items() if v is not None}
        if by_name:
            pairs = [(o, o, float(np.linalg.norm(pa[o] - pb[o]))) for o in sorted(pa) if o in pb]
        else:
            pairs = _pairs_by_position(pa, pb, max_match_m)
        distances.extend(d for _, _, d in pairs); both += len(pairs)
        only_a += len(pa) - len(pairs); only_b += len(pb) - len(pairs)
    return dict(shared_frames=len(shared), matched_object_frames=both, only_in_first=only_a, only_in_second=only_b,
                distance_m=dict(median=float(np.median(distances)) if distances else None, p90=float(np.percentile(distances, 90)) if distances else None,
                                max=float(max(distances)) if distances else None),
                pairing="by_identity_name" if by_name else "by_nearest_position", max_match_m=None if by_name else max_match_m)


def compare(run_dirs, max_match_m=DEFAULT_MAX_MATCH_M) -> dict:
    runs = [load_run(Path(d)) for d in run_dirs]
    if len({r["manifest_sha256"] for r in runs if r["manifest_sha256"]}) > 1:
        raise ValueError("the runs come from different recordings (manifest_sha256 differs); compare runs of one recording")
    frame_sets = [set(r["frame_table"]) for r in runs]
    identical = all(s == frame_sets[0] for s in frame_sets)
    kinds = [r["provider"] for r in runs]
    for r in runs:  # two runs of one provider kind keep separate rows and pairs (they used to overwrite each other)
        r["label"] = r["provider"] if kinds.count(r["provider"]) == 1 else f"{r['provider']}:{Path(r['run_dir']).name}"
    pairs = {}
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            pairs[f"{runs[i]['label']}_vs_{runs[j]['label']}"] = agreement(runs[i], runs[j], max_match_m)
    report = dict(schema="sg-object-provider-comparison/1", identical_frame_sets=identical,
                  identical_timing_settings=len({(r["queue_policy"], r["decimate"], r["fixed_latency_s"]) for r in runs}) == 1,
                  runs=[{k: v for k, v in r.items() if k != "frame_table"} for r in runs], pairwise_agreement=pairs,
                  scope="software behaviour on identical frames of one recording; no reference object positions, so agreement is consistency, not accuracy; "
                        "downstream proposals are from the rules baseline with unknown references")
    return report


def markdown(report: dict) -> str:
    lines = ["| provider | frames | objects available (per identity) | frames with 2 objects | proposals L / R | first proposal L / R (s) | object ms |", "|---|---|---|---|---|---|---|"]
    for r in report["runs"]:
        av = ", ".join(f"{o}: {n}" for o, n in r["availability"].items()) or "-"
        pl = r["proposal_ticks"]; fp = r["first_proposal_s"]
        ms = round(1000 * r["object_processing_s_median"], 1) if r.get("object_processing_s_median") is not None else "-"
        lines.append(f"| {r.get('label', r['provider'])} | {r['frames']} | {av} | {r['frames_with_two_objects']} | {pl.get('left', 0)} / {pl.get('right', 0)} | {fp.get('left', '-')} / {fp.get('right', '-')} | {ms} |")
    lines += ["", "Pairwise position agreement (same object in both runs: by identity name, or by nearest position within the stated gate):", ""]
    for name, ag in report["pairwise_agreement"].items():
        d = ag["distance_m"]
        how = "by name" if ag.get("pairing") == "by_identity_name" else f"by position within {ag.get('max_match_m')} m"
        lines.append(f"- {name} ({how}): matched {ag['matched_object_frames']} object-frames; only in first {ag['only_in_first']}, only in second {ag['only_in_second']}; "
                     f"distance median {d['median'] if d['median'] is None else round(d['median'], 3)} m, p90 {d['p90'] if d['p90'] is None else round(d['p90'], 3)} m")
    lines += ["", f"Identical frame sets: {report['identical_frame_sets']}. {report['scope']}"]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("report", type=Path); ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--max-match-m", type=float, default=DEFAULT_MAX_MATCH_M,
                    help="gate for pairing generic identities by position; keep it below the spacing of distinct objects")
    a = ap.parse_args(argv)
    report = compare(a.runs, a.max_match_m)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=1) + "\n")
    a.report.with_suffix(".md").write_text(markdown(report) + "\n")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
