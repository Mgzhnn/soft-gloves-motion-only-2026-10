"""Replay the stored packets of a perception run under a replay configuration: no perception, no GPU, no new data.

Reads ``RUN_DIR/packets.jsonl`` (the validated ``sg-offline/3`` packets a perception run wrote), runs the unchanged
``Replay`` with the given ``Config`` and writes a run folder with ``trace.jsonl``, ``summary.json`` (evaluation counts,
replay configuration, source run and hashes), ``provenance.json`` and a copy of the source ``frame_times.csv``. The
packets are not copied; the provenance names the source file and its hash. With the configuration the source run used
the trace reproduces it tick for tick (the replay is deterministic), which is how a batch setting chosen on the
development clip (``rate_window``, 7 October 2026) is applied to runs that already exist.
usage: replay_packets.py RUN_DIR OUT_DIR [--rate-window K] [--manifest-csv PATH]
                         [--feature-layout v1|v2 --scene SCENE_JSONL] [--model rules|rules_actions] [--decision-rule RULE]
                         [--fsm] [--fsm-config JSON]
16 Oct 2026: ``--feature-layout v2`` with ``--scene`` replays with the per-action cue block (scene records from the run or
from ``build_scene.py``); ``--model rules_actions`` is the per-action descriptive floor; ``--fsm`` additionally runs the
simulated glove state machine over the trace and writes ``commands.jsonl``, ``fsm_states.jsonl`` and ``fsm_summary.json``.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from run_perception import DEFAULT_MANIFEST_CSV, append_manifest_row, git_state, implementation_hashes
from sg_offline.evaluation import summarize
from sg_offline.fsm import FSMConfig, commands_record, simulate_commands
from sg_offline.inference import ActionRules, SimpleRules
from sg_offline.inputs import packets_from_jsonl
from sg_offline.replay import Config, Replay, write_trace
from sg_offline.scene import read_scenes


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


MODELS = {"rules": SimpleRules, "rules_actions": ActionRules}


def replay_run(run_dir, out_dir, config: Config, manifest_csv=DEFAULT_MANIFEST_CSV, command="", scene_path=None, model="rules",
               fsm_config: FSMConfig | None = None) -> dict:
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    packets_path = run_dir / "packets.jsonl"
    if not packets_path.is_file():
        raise FileNotFoundError(f"{packets_path} is missing")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("output directory must be empty")
    if model not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}")
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    packets = tuple(packets_from_jsonl(packets_path))
    if not packets:
        raise ValueError("empty packet stream")
    source_summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    scenes = None
    if scene_path is None and (run_dir / "scene.jsonl").exists():
        scene_path = run_dir / "scene.jsonl"
    if scene_path is not None:
        scenes = read_scenes(scene_path)
        missing = [p.packet_id for p in packets if p.packet_id not in scenes]
        if missing:
            raise ValueError(f"{len(missing)} packets have no scene record (first: {missing[0]})")
    rows = Replay(config, MODELS[model](), scenes=scenes).run(packets)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_trace(out_dir / "trace.jsonl", rows)
    fsm_summary = None
    if fsm_config is not None:
        commands, states, fsm_summary = simulate_commands(rows, fsm_config)
        (out_dir / "commands.jsonl").write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in commands_record(commands)))
        (out_dir / "fsm_states.jsonl").write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in states))
        (out_dir / "fsm_summary.json").write_text(json.dumps(fsm_summary, indent=2, default=str) + "\n")
    if (run_dir / "frame_times.csv").exists():
        shutil.copyfile(run_dir / "frame_times.csv", out_dir / "frame_times.csv")
    summary = dict(kind="replay_of_stored_packets", source_run=str(run_dir.resolve()), source_packets_sha256=sha256_file(packets_path),
                   frames=len(packets), replay_configuration=asdict(config), model=model,
                   scene=None if scene_path is None else dict(path=str(Path(scene_path).resolve()), sha256=sha256_file(scene_path)),
                   fsm=None if fsm_summary is None else {k: fsm_summary[k] for k in ("left", "right", "config", "scope")},
                   source_settings={k: source_summary.get(k) for k in ("queue_policy", "fixed_latency_s", "decimate", "object_provider") if k in source_summary},
                   evaluation=summarize(rows), scope="replay of stored packets under a declared replay configuration; no perception was run")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    provenance = dict(started_utc=started, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), command=command,
                      source_run=str(run_dir.resolve()), source_packets_sha256=summary["source_packets_sha256"],
                      source_provenance_sha256=sha256_file(run_dir / "provenance.json") if (run_dir / "provenance.json").exists() else None,
                      replay_configuration=asdict(config), implementation_sha256=implementation_hashes(), git=git_state(),
                      participant_id=packets[0].participant_id, session_id=packets[0].session_id,
                      scope="replay provenance; the perception provenance is the source run's provenance.json")
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    summary["registered_in_manifest_csv"] = append_manifest_row(manifest_csv, dict(
        run_id=f"{out_dir.resolve().name}@{started}", date_utc=started, git_commit=provenance["git"].get("commit") or "",
        dirty_worktree=provenance["git"].get("dirty_tracked_or_untracked", ""), command=command, config_path=str(packets_path.resolve()),
        config_sha256=summary["source_packets_sha256"], seed="", dataset_id=f"{packets[0].participant_id}/{packets[0].session_id}",
        dataset_version=summary["source_packets_sha256"][:12], split_manifest_sha256="", environment_snapshot="", hardware="cpu replay",
        output_path=str(out_dir.resolve()), status="completed", notes=f"replay of {run_dir.name} rate_window={config.rate_window} layout={config.feature_layout} model={model} fsm={fsm_config is not None}"))
    return summary


def main(argv=None) -> int:
    import sys
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path); ap.add_argument("out_dir", type=Path)
    ap.add_argument("--rate-window", type=int, default=1); ap.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    ap.add_argument("--feature-layout", default="v1", choices=("v1", "v2", "v3")); ap.add_argument("--scene", type=Path)
    ap.add_argument("--model", default="rules", choices=sorted(MODELS)); ap.add_argument("--decision-rule", default="best_action_threshold")
    ap.add_argument("--fsm", action="store_true", help="also simulate the glove state machine over the trace (secondary endpoint)")
    ap.add_argument("--fsm-config", type=Path, help="JSON with FSMConfig fields (defaults otherwise)")
    ap.add_argument("--posterior-forgetting", type=float, default=0., help="temporal recursion on the joint scores, 0 = off (plan setting, 16 Oct 2026)")
    a = ap.parse_args(argv)
    if a.rate_window < 1:
        ap.error("rate window must be a positive integer")
    if a.feature_layout == "v2" and a.scene is None and not (a.run_dir / "scene.jsonl").exists():
        ap.error("--feature-layout v2 needs --scene (or a scene.jsonl inside the run)")
    if a.fsm_config is not None and not a.fsm:
        ap.error("--fsm-config needs --fsm")
    fsm = (FSMConfig(**{k: v for k, v in json.loads(a.fsm_config.read_text()).items() if not k.startswith("_")}) if a.fsm_config else FSMConfig()) if a.fsm else None  # "_note" keys are comments
    command = " ".join(sys.argv if argv is None else ["replay_packets.py", *map(str, argv)])
    summary = replay_run(a.run_dir, a.out_dir, Config(rate_window=a.rate_window, feature_layout=a.feature_layout, decision_rule=a.decision_rule, posterior_forgetting=a.posterior_forgetting),
                         a.manifest_csv, command, scene_path=a.scene, model=a.model, fsm_config=fsm)
    print(json.dumps({k: summary[k] for k in ("frames", "replay_configuration", "evaluation", "registered_in_manifest_csv")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
