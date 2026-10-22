"""One person's recorded-label fitting and Replay diagnostic (participant N = 1): the training smoke run.

Fits every learned arm on the TRAINING session(s) with the ledger labels (``training.label_ticks`` + ``fit_recorded``),
keeps the pass with the lowest development log loss and chooses the reporting threshold on the DEVELOPMENT session(s)
under the declared unwanted-proposal budget, with the same two functions a comparison fold uses
(``experiment.fit_learned_arm`` and ``experiment.development_selection``). It then replays each fitted arm on the
training and development packets at its selected threshold (``Replay``: no perception, no GPU, no new data) and scores
the traces against the ledgers with ``score_requests.score``, strictly and at the declared early tolerance.

One operator means: no comparison, no contrast, no interval, no test session. ``experiment.compare`` and
``run_comparison.py`` remain the multi-person route; their guard is untouched, and this tool refuses more than one
person. Sessions named ``untouched`` in the specification are never loaded. In-sample figures (a fitted arm replayed on
its own training clip) are descriptive only and are labelled as such.
usage: fit_diagnostic.py SPEC_JSON OUT_DIR [--manifest-csv PATH]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from run_perception import DEFAULT_MANIFEST_CSV, append_manifest_row, git_state, implementation_hashes
from score_requests import score as score_trace, summary_line
from sg_offline.contracts import OUTCOMES
from sg_offline.evaluation import summarize
from sg_offline.experiment import ARMS, FLOOR_ARMS, THRESHOLD_GRID, _examples, development_selection, factory, fit_learned_arm, parameter_count
from sg_offline.inputs import packets_from_jsonl
from sg_offline.ledger import ledger_summary, read_opportunity_ledger, session_origin_s, to_opportunities
from sg_offline.recordings import RGBDRecording
from sg_offline.replay import Config, Replay, write_trace
from sg_offline.scene import read_scenes
from sg_offline.training import pair_index

SCHEMA = "sg-fit-diagnostic/1"
SCOPE = ("one person's recorded-label fitting and Replay diagnostic (participant N = 1, one annotator); software and "
         "feasibility counts, not accuracy; no comparison, no contrast, no interval, no test session")
THRESHOLDS = THRESHOLD_GRID
SPEC_KEYS = frozenset(("scope", "rate_window", "decision_rule", "training", "development", "untouched", "arms", "seeds",
                       "max_unwanted_proposals_per_hand_minute", "budget_status", "burden_unit", "early_limit_s", "thresholds",
                       "steps", "patience", "lr", "regularization", "window",
                       "feature_layout", "admitted_actions", "posterior_forgetting"))  # layout v1|v2 and the actions scored as requests (default ["G"])
SESSION_KEYS = frozenset(("run", "manifest", "ledger", "scene"))  # scene: scene.jsonl for feature layout v2; defaults to RUN/scene.jsonl when it exists
_PAIR_NAMES = {pair_index(l, r): f"{l}|{r}" for l in OUTCOMES for r in OUTCOMES}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _identity(session):
    return (session["packets"][0].participant_id, session["packets"][0].session_id)


def label_histogram(examples):
    """How many labelled ticks carry each outcome pair; marginal labels (one hand unknown) are counted apart."""
    hist = {}
    for e in examples:
        key = _PAIR_NAMES.get(e.label, f"pair{e.label}") if isinstance(e.label, int) else f"marginal({len(e.label)} alternatives)"
        hist[key] = hist.get(key, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: -kv[1]))


def diagnose(training, development, arms, seeds=(0,), steps=2000, lr=.01, regularization=.001, thresholds=THRESHOLDS,
             budget=None, burden_unit="onsets", early_limit_s=0., window=15, config=Config(), synthetic_check=False, patience=None):
    """Fit and select every arm on already-loaded sessions (``dict(packets=..., opportunities=...)``).

    Returns ``(records, models)``: one record and one fitted model per arm and seed. The fitting and the development
    selection are the comparison fold's own functions, so a diagnostic record equals the corresponding fold's fields.
    """
    if not training or not development:
        raise ValueError("the diagnostic needs at least one training session and one development session")
    ids = [_identity(s) for s in training + development]
    if len(set(ids)) != len(ids):
        raise ValueError("a session may appear once, in one role: the development session must differ from the training sessions")
    for s in training + development:
        if not s["packets"]:
            raise ValueError("empty packet stream")
        if any((p.participant_id, p.session_id) != _identity(s) for p in s["packets"]):
            raise ValueError("mixed session packets")
        if any((o.participant_id, o.session_id) != _identity(s) for o in s["opportunities"]):
            raise ValueError("ledger identity mismatch")
    people = sorted({p for p, _ in ids})
    if len(people) != 1:
        raise ValueError("the diagnostic is for one person's sessions; a multi-person set belongs to experiment.compare (person-held-out)")
    if any(a not in ARMS for a in arms):
        raise ValueError("unknown model arm")
    if steps < 1 or lr <= 0 or regularization < 0:
        raise ValueError("invalid fit settings")
    if not thresholds or any(not 0 < t <= 1 for t in thresholds):
        raise ValueError("invalid threshold grid")
    if budget is None or budget < 0:
        raise ValueError("declare a development unwanted-proposal budget before fitting")
    if early_limit_s < 0:
        raise ValueError("the early tolerance must be nonnegative")
    if patience is not None and (type(patience) is not int or patience < 1):
        raise ValueError("patience must be a positive integer or None")
    split_id = "diagnostic:train=" + ",".join(s for _, s in map(_identity, training)) + "|dev=" + ",".join(s for _, s in map(_identity, development))
    records = []; models = []
    for name in arms:
        for seed in (seeds if name not in FLOOR_ARMS else seeds[:1]):
            model = factory(name, seed, window, config.feature_layout)   # layout v2 readouts carry the cue block
            examples, accounting = _examples(model, training, window, config)
            if not examples:
                raise ValueError("no usable training labels")
            dev_examples, dev_accounting = _examples(model, development, window, config)
            before = after = None; stopping = None
            if name not in FLOOR_ARMS:
                before, after, stopping = fit_learned_arm(model, examples, dev_examples, steps, lr, regularization, split_id, synthetic_check, patience)
            selected, dev_curve, feasible, reason = development_selection(model, development, config, thresholds, budget, burden_unit, early_limit_s)
            records.append(dict(arm=name, seed=seed, parameter_count=parameter_count(model),
                                training_sessions=[_identity(s) for s in training], development_sessions=[_identity(s) for s in development],
                                training_examples=len(examples), training_label_histogram=label_histogram(examples), training_label_accounting=accounting,
                                development_examples=len(dev_examples), development_label_histogram=label_histogram(dev_examples),
                                development_label_accounting=dev_accounting,
                                steps=steps if name not in FLOOR_ARMS else 0, patience=patience, learning_rate=lr, regularization=regularization,
                                loss_before=before, loss_after=after, fitting_improved=None if after is None else after < before,
                                early_stopping=stopping, fitting_diagnostics=getattr(model, "fit_diagnostics", None),
                                threshold_selection="development", reporting_threshold=selected, development_budget_feasible=feasible,
                                development_selection_reason=reason, development_budget_unwanted_proposals_per_hand_minute=budget,
                                development_budget_unit=burden_unit, development_curve=dev_curve,
                                fit_provenance=getattr(model, "fit_provenance", "unvalidated_rules")))
            models.append(model)
    return records, models


def load_session(entry, base, role, config, admitted_actions=("G",)):
    """Packets, ledger and recording manifest of one session, bound to each other and to the diagnostic configuration."""
    run_dir = (base / entry["run"]).resolve(); manifest = (base / entry["manifest"]).resolve(); ledger_path = (base / entry["ledger"]).resolve()
    scene_path = (base / entry["scene"]).resolve() if entry.get("scene") else (run_dir / "scene.jsonl" if (run_dir / "scene.jsonl").is_file() else None)
    if scene_path is not None and not scene_path.is_file():
        raise ValueError(f"{role}: scene file {scene_path} is missing")
    scenes = read_scenes(scene_path) if scene_path is not None else None
    if config.feature_layout != "v1" and scenes is None:
        raise ValueError(f"{role}: feature layout {config.feature_layout} needs a scene file (session key 'scene' or RUN/scene.jsonl)")
    packets_path = run_dir / "packets.jsonl"
    if not packets_path.is_file():
        raise FileNotFoundError(f"{packets_path} is missing")
    packets = tuple(packets_from_jsonl(packets_path))
    if not packets:
        raise ValueError("empty packet stream")
    recording = RGBDRecording(manifest)
    origin, clock = session_origin_s(manifest)
    ledger = read_opportunity_ledger(ledger_path)
    identity = (recording.participant_id, recording.session_id)
    if (ledger["participant_id"], ledger["session_id"]) != identity or (packets[0].participant_id, packets[0].session_id) != identity:
        raise ValueError(f"{role}: ledger, packets and recording manifest must name the same participant and session")
    if ledger["clock"]["name"] != clock or packets[0].clock != clock:
        raise ValueError(f"{role}: ledger, packets and recording manifest must share one clock")
    inside, outside = to_opportunities(ledger, origin, admitted_actions)
    summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").is_file() else {}
    provenance = json.loads((run_dir / "provenance.json").read_text()) if (run_dir / "provenance.json").is_file() else {}
    source_config = (provenance.get("settings") or {}).get("replay_configuration") or summary.get("replay_configuration")
    if source_config:
        differing = sorted(k for k, v in asdict(config).items() if k not in ("rate_window", "feature_layout", "plane_window", "reversal_deadband_mps", "posterior_forgetting") and source_config.get(k, v) != v)
        if differing:
            raise ValueError(f"{role}: the diagnostic replays the source run's configuration except the rate window; differing: {differing}")
    provider = summary.get("object_provider") if isinstance(summary.get("object_provider"), dict) else {"kind": summary.get("object_provider")}
    source = dict(role=role, run=str(run_dir), packets=str(packets_path), packets_sha256=sha256_file(packets_path),
                  manifest=str(manifest), manifest_sha256=sha256_file(manifest), ledger=str(ledger_path), ledger_sha256=sha256_file(ledger_path),
                  participant_id=identity[0], session_id=identity[1], clock=clock, origin_s=origin, frames=len(packets),
                  ledger_summary=ledger_summary(ledger), opportunities_in_scope=len(inside), opportunities_out_of_scope=len(outside),
                  perception_checkpoints_sha256=provenance.get("checkpoints_sha256"),
                  source_settings={k: summary.get(k) for k in ("queue_policy", "fixed_latency_s", "decimate", "start", "max_frames")},
                  object_provider=provider, source_replay_configuration=source_config,
                  source_provenance_sha256=sha256_file(run_dir / "provenance.json") if (run_dir / "provenance.json").is_file() else None,
                  scene=None if scene_path is None else dict(path=str(scene_path), sha256=sha256_file(scene_path)), admitted_actions=list(admitted_actions))
    return dict(packets=packets, opportunities=inside, manifest=manifest, ledger=ledger_path, run_dir=run_dir, scenes=scenes, admitted_actions=tuple(admitted_actions)), source


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serialisable: {type(o).__name__}")


def _replay_and_score(model, record, session, source, out_dir, config, early_limit_s, command, implementation, git):
    """Replay one fitted arm on one session at its reporting threshold, write the run folder and score it twice."""
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    folder = out_dir / f"{record['arm']}_seed{record['seed']}" / source["session_id"]
    folder.mkdir(parents=True, exist_ok=False)
    replay_config = replace(config, proposal_score_threshold=record["reporting_threshold"])
    rows = Replay(replay_config, model, scenes=session.get("scenes")).run(session["packets"])
    write_trace(folder / "trace.jsonl", rows)
    if (session["run_dir"] / "frame_times.csv").is_file():
        shutil.copyfile(session["run_dir"] / "frame_times.csv", folder / "frame_times.csv")
    role_note = "in-sample: the arm was fitted on this clip (descriptive only)" if source["role"] == "training" else "development: the pass and threshold were chosen on this clip"
    (folder / "summary.json").write_text(json.dumps(dict(kind="diagnostic_replay_of_fitted_arm", role=source["role"], role_note=role_note, arm=record["arm"], seed=record["seed"],
        frames=len(rows), replay_configuration=asdict(replay_config), source_run=source["run"], source_packets_sha256=source["packets_sha256"],
        evaluation=summarize(rows), scope=SCOPE), indent=2, default=_json_default) + "\n")
    (folder / "provenance.json").write_text(json.dumps(dict(started_utc=started, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        command=command, role=source["role"], arm=record["arm"], seed=record["seed"], fit_provenance=record["fit_provenance"],
        reporting_threshold=record["reporting_threshold"], source_run=source["run"], source_packets_sha256=source["packets_sha256"],
        source_provenance_sha256=source["source_provenance_sha256"], replay_configuration=asdict(replay_config), implementation_sha256=implementation, git=git,
        participant_id=source["participant_id"], session_id=source["session_id"],
        scope="diagnostic replay provenance; the perception provenance is the source run's provenance.json"), indent=2, default=_json_default) + "\n")
    scores = {}
    for label, limit in (("strict", 0.), (f"tol{early_limit_s:g}", early_limit_s)):
        report = score_trace(session["manifest"], folder, session["ledger"], limit, scope=f"{SCOPE}; {role_note}", admitted_actions=session.get("admitted_actions", ("G",)))
        (folder / f"scoring_{label}.json").write_text(json.dumps(report, indent=1, default=_json_default) + "\n")
        scores[label] = dict(early_limit_s=limit, summary=summary_line(report), requests=report["requests"],
                             per_participant_success=report["per_participant_success"], unwanted_exposure=report["unwanted_exposure"],
                             timeline_accounting=report["timeline_accounting"])
    return dict(role=source["role"], role_note=role_note, trace_dir=str(folder), trace_sha256=sha256_file(folder / "trace.jsonl"),
                reporting_threshold=record["reporting_threshold"], frames=len(rows), scores=scores)


def _cell(scores, label):
    pp = next(iter(scores[label]["per_participant_success"].values()), {}); ex = next(iter(scores[label]["unwanted_exposure"].values()), {})
    return f"{pp.get('compatible_first', 0)} of {pp.get('evaluable_request_hands', 0)}", f"{ex.get('onsets_per_hand_minute', 0.):.1f}"


def markdown_report(report):
    cfg = report["configuration"]; tol = f"tol{cfg['early_limit_s']:g}"
    train_ids = [s["session_id"] for s in report["sessions"] if s["role"] == "training"]; dev_ids = [s["session_id"] for s in report["sessions"] if s["role"] == "development"]
    lines = [f"# Fitting diagnostic — {report['made_utc'][:10]}", "",
             f"**Scope.** {report['scope']}. Training session(s): {', '.join(train_ids)}; development session(s): {', '.join(dev_ids)}; "
             f"untouched (never loaded): {', '.join(report['untouched']) or 'none'}. Participant(s): {', '.join(report['participants'])}.", "",
             f"**Settings (declared before fitting).** rate window {cfg['rate_window']}; supervisor rule {cfg['decision_rule']}; up to {cfg['steps']} passes with patience {cfg['patience']}, learning rate {cfg['lr']}, regularisation {cfg['regularization']}; "
             f"threshold grid {cfg['thresholds']}; development budget {cfg['max_unwanted_proposals_per_hand_minute']} unwanted {cfg['burden_unit']} per no-request hand-minute ({cfg['budget_status']}); "
             f"early tolerance {cfg['early_limit_s']} s; seeds {cfg['seeds']}.", "",
             "Compatible first proposals are counted at the declared tolerance with the strict count in brackets; burden = unwanted proposal onsets per labelled no-request hand-minute. "
             "The training-clip columns are in-sample (descriptive only). Counts of software behaviour against one annotator's timeline; not accuracy.", "",
             "| arm | parameters | training ticks (labels) | log loss before → after | best pass (stop) | threshold (selection) | development: compatible first | development: onsets/min | training (in-sample): compatible first | training (in-sample): onsets/min |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for a in report["arms"]:
        es = a.get("early_stopping") or {}
        best = f"{es['best_pass']} of {es.get('passes_run', es['max_passes'])} run ({es.get('stopped_by', 'all passes')}; cap {es['max_passes']})" if es.get("best_pass") is not None else ("—" if a["steps"] == 0 else "all")
        loss = "—" if a["loss_before"] is None else f"{a['loss_before']:.3f} → {a['loss_after']:.3f}"
        status = "feasible" if a["development_budget_feasible"] else ("silent" if a["development_selection_reason"].startswith("silent") else "infeasible")
        hist = ", ".join(f"{k} {v}" for k, v in a["training_label_histogram"].items())
        dev = [r for r in a["replays"] if r["role"] == "development"]; tr = [r for r in a["replays"] if r["role"] == "training"]
        def cells(rs):
            if not rs:
                return "—", "—"
            c = [_cell(r["scores"], tol) for r in rs]; s = [_cell(r["scores"], "strict") for r in rs]
            return "; ".join(f"{ct[0]} (strict {st[0]})" for ct, st in zip(c, s)), "; ".join(ct[1] for ct in c)
        dc, db = cells(dev); tc, tb = cells(tr)
        lines.append(f"| {a['arm']} | {a['parameter_count']} | {a['training_examples']} ({hist}) | {loss} | {best} | {a['reporting_threshold']} ({status}) | {dc} | {db} | {tc} | {tb} |")
    lines += ["", "**Selection reasons.**"] + [f"- {a['arm']}: {a['development_selection_reason']}" for a in report["arms"]]
    lines += ["", f"**Limits.** {report['limits']}", ""]
    return "\n".join(lines)


def check_specification_keys(spec):
    """Refuse keys this tool does not read. A misspelt or nested declared setting
    (``early_limit``, a run_comparison-style ``config`` block, ``threshold_grid``) used to fall back silently to the code
    default, which the record then reported as if it had been declared."""
    if not isinstance(spec, dict):
        raise ValueError("the specification must be a JSON object")
    unknown = sorted(set(spec) - SPEC_KEYS)
    if unknown:
        raise ValueError(f"unknown specification key(s) {unknown}; this tool reads only {sorted(SPEC_KEYS)}")
    for role in ("training", "development"):
        for entry in spec.get(role, []):
            extra = sorted(set(entry) - SESSION_KEYS) if isinstance(entry, dict) else ["<not an object>"]
            if extra:
                raise ValueError(f"unknown {role} session key(s) {extra}; a session entry names only {sorted(SESSION_KEYS)}")


def run_diagnostic(spec_path, out_dir, manifest_csv=DEFAULT_MANIFEST_CSV, command=""):
    spec_path, out_dir = Path(spec_path), Path(out_dir)
    spec = json.loads(spec_path.read_text())
    if "synthetic_check" in spec:
        raise ValueError("synthetic check is available only in the test API")
    check_specification_keys(spec)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("output directory must be empty")
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rate_window = int(spec["rate_window"])
    if rate_window < 1:
        raise ValueError("rate window must be a positive integer")
    config = Config(rate_window=rate_window, **({"decision_rule": spec["decision_rule"]} if "decision_rule" in spec else {}),
                    **({"feature_layout": spec["feature_layout"]} if "feature_layout" in spec else {}),
                    **({"posterior_forgetting": float(spec["posterior_forgetting"])} if "posterior_forgetting" in spec else {}))
    admitted_actions = tuple(spec.get("admitted_actions", ("G",)))
    untouched = list(spec.get("untouched", []))
    base = spec_path.parent
    sessions = []; sources = []
    for role in ("training", "development"):
        for entry in spec.get(role, []):
            session, source = load_session(entry, base, role, config, **({"admitted_actions": admitted_actions} if "admitted_actions" in spec else {}))
            if source["session_id"] in untouched:
                raise ValueError(f"session {source['session_id']} is declared untouched and cannot be loaded")
            sessions.append(session); sources.append(source)
    checkpoints = {json.dumps(s["perception_checkpoints_sha256"], sort_keys=True) for s in sources}
    frozen = ("queue_policy", "fixed_latency_s", "decimate")   # start and max_frames describe the clip, not the route
    settings = {json.dumps(dict({k: s["source_settings"].get(k) for k in frozen}, provider=s["object_provider"].get("kind")), sort_keys=True) for s in sources}
    if len(checkpoints) != 1 or len(settings) != 1:
        raise ValueError("every session must come from the same perception checkpoints and the same frozen run settings (policy, latency, decimation, provider kind)")
    training = [s for s, src in zip(sessions, sources) if src["role"] == "training"]; development = [s for s, src in zip(sessions, sources) if src["role"] == "development"]
    arms = tuple(spec["arms"]); seeds = tuple(spec.get("seeds", (0,)))
    budget = spec["max_unwanted_proposals_per_hand_minute"]; burden_unit = spec.get("burden_unit", "onsets"); early = float(spec.get("early_limit_s", 0.))
    thresholds = tuple(spec.get("thresholds", THRESHOLDS)); steps = int(spec.get("steps", 2000)); lr = float(spec.get("lr", .01)); reg = float(spec.get("regularization", .001)); window = int(spec.get("window", 15))
    patience = spec.get("patience"); patience = None if patience is None else int(patience)   # default rule: all passes, best development pass kept
    records, models = diagnose(training, development, arms, seeds, steps, lr, reg, thresholds, budget, burden_unit, early, window, config, patience=patience)
    out_dir.mkdir(parents=True, exist_ok=True)
    implementation = implementation_hashes(); git = git_state()
    for record, model in zip(records, models):
        record["replays"] = [_replay_and_score(model, record, session, source, out_dir, config, early, command, implementation, git)
                             for session, source in zip(sessions, sources)]
    report = dict(schema=SCHEMA, scope=SCOPE, made_utc=started, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), command=command,
                  specification=str(spec_path.resolve()), specification_sha256=sha256_file(spec_path), declared_scope=spec.get("scope", ""),
                  participants=sorted({s["participant_id"] for s in sources}), sessions=sources, untouched=untouched,
                  configuration=dict(replay_configuration=asdict(config), rate_window=rate_window, decision_rule=config.decision_rule, window=window, steps=steps, patience=patience, lr=lr, regularization=reg,
                                     thresholds=list(thresholds), seeds=list(seeds), arms=list(arms), selection="development",
                                     max_unwanted_proposals_per_hand_minute=budget, burden_unit=burden_unit, budget_status=spec.get("budget_status", "undeclared"),
                                     early_limit_s=early,
                                     training="up to `steps` passes over the training examples for every learned arm; the pass with the lowest development log loss is kept; fitting stops after `patience` passes without a new best (None = all passes)",
                                     silent_points="a development operating point that never proposes is never selected and is reported as silent",
                                     rule_arms="hand-authored rules are never fitted; they decide identically at every grid threshold (descriptive floor)"),
                  perception=dict(checkpoints_sha256=sources[0]["perception_checkpoints_sha256"], identical_across_sessions=True,
                                  source_settings=sources[0]["source_settings"], object_provider_kind=sources[0]["object_provider"].get("kind")),
                  arms=records, implementation_sha256=implementation, git=git,
                  limits=("one operator, one annotator, two or three short clips: the figures are software behaviour counts under a declared protocol, "
                          "not accuracy and not evidence that any arm generalises to another person; the person-held-out comparison needs independent people"))
    report["registered_in_manifest_csv"] = append_manifest_row(manifest_csv, dict(
        run_id=f"{out_dir.resolve().name}@{started}", date_utc=started, git_commit=git.get("commit") or "", dirty_worktree=git.get("dirty_tracked_or_untracked", ""),
        command=command, config_path=str(spec_path.resolve()), config_sha256=report["specification_sha256"], seed=",".join(map(str, seeds)),
        dataset_id=";".join(f"{s['participant_id']}/{s['session_id']}" for s in sources), dataset_version=";".join(s["packets_sha256"][:12] for s in sources),
        split_manifest_sha256="", environment_snapshot="", hardware="cpu fit + replay", output_path=str(out_dir.resolve()), status="completed",
        notes=f"N=1 fitting diagnostic; train={[s['session_id'] for s in sources if s['role']=='training']} dev={[s['session_id'] for s in sources if s['role']=='development']} rate_window={rate_window}"))
    (out_dir / "fit_diagnostic.json").write_text(json.dumps(report, indent=1, default=_json_default, allow_nan=False) + "\n")
    (out_dir / "fit_diagnostic_report.md").write_text(markdown_report(report))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spec", type=Path); ap.add_argument("out_dir", type=Path)
    ap.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    a = ap.parse_args(argv)
    command = " ".join(sys.argv if argv is None else ["fit_diagnostic.py", *map(str, argv)])
    report = run_diagnostic(a.spec, a.out_dir, a.manifest_csv, command)
    tol = f"tol{report['configuration']['early_limit_s']:g}"
    for arm in report["arms"]:
        es = arm.get("early_stopping") or {}
        print(f"{arm['arm']}: {arm['training_examples']} training ticks; loss {arm['loss_before']} -> {arm['loss_after']}; best pass {es.get('best_pass')}; "
              f"threshold {arm['reporting_threshold']} ({'feasible' if arm['development_budget_feasible'] else 'not feasible'})")
        for r in arm["replays"]:
            print(f"   {r['role']:<11} {r['scores'][tol]['summary']}")
    print(json.dumps(dict(report=str(Path(a.out_dir) / "fit_diagnostic.json"), registered_in_manifest_csv=report["registered_in_manifest_csv"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
