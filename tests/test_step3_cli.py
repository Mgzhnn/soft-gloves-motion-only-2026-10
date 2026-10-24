"""The scoring command-line tools (score_requests.py, choose_rate_window.py) on the synthetic end-to-end clip; participant N=0."""
import csv
import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

import choose_rate_window  # noqa: E402
import import_requests  # noqa: E402
import score_requests  # noqa: E402
from sg_offline.ledger import session_origin_s  # noqa: E402
from test_end_to_end_step3 import FIXED, clip, read_jsonl, run  # noqa: E402,F401  (clip is a fixture)


def timeline(tmp_path, manifest, out):
    """A two-row timeline: rest, then the right hand's request from 0.05 s before its first proposal to the end; the left hand is never observed."""
    origin, clock = session_origin_s(manifest)
    rows = read_jsonl(out / "trace.jsonl")
    times = list(csv.DictReader(open(out / "frame_times.csv")))
    first = next(r for r in rows if r["outputs"][1]["status"] == "proposed_simulation_only")
    frame = next(t for t in times if t["frame_id"] == first["packet_id"])
    onset = round(float(frame["session_relative_acquired_s"]) - .05, 3)
    duration = round(float(times[-1]["session_relative_available_s"]) + .2, 2)
    path = tmp_path / "timeline.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["start_s", "end_s", "left", "right", "note"])
        w.writerow(["0", onset, "NQ", "NQ", ""]); w.writerow([onset, duration, "EXCLUDED", "G", "left never observed"])
    ledger = tmp_path / "requests.json"
    import_requests.main([str(path), str(ledger), "--participant", "operator", "--session", "s", "--annotator", "ann", "--clock", clock, "--duration-s", str(duration)])  # returns None
    assert ledger.exists()
    return ledger, first["decision_s"] - origin


def test_score_requests_cli_reports_the_request_outcome_and_refuses_a_foreign_ledger(tmp_path, monkeypatch, clip):
    manifest, seg = clip
    out = run(tmp_path, monkeypatch, "run", *FIXED, "--object-provider", "masks", "--masks", str(seg))
    ledger, first_rel = timeline(tmp_path, manifest, out)
    report = tmp_path / "score.json"
    assert score_requests.main([str(manifest), str(out), str(ledger), str(report), "--scope", "synthetic"]) == 0
    d = json.loads(report.read_text())
    assert d["schema"] == "sg-step3-request-scoring/1" and d["scope"] == "synthetic" and d["early_limit_s"] == 0.
    assert d["per_participant_success"]["operator"]["compatible_first"] == 1 and d["per_participant_success"]["operator"]["evaluable_request_hands"] == 1
    (req,) = d["requests"]
    assert req["side"] == "right" and req["result"] == "compatible_first" and req["accepted_actions"] == ["G"]
    assert req["first_proposal_s"] == pytest.approx(first_rel, abs=1e-6) and req["proposals_in_window"] == 3 and req["early_by_s"] == 0.
    assert d["provenance"]["replay_configuration"]["rate_window"] == 1 and len(d["provenance"]["trace_sha256"]) == 64
    assert d["unwanted_exposure"]["operator"]["unwanted_proposals"] == 0 and d["timeline_accounting"]["all_hand_ticks"] == 2 * d["trace_rows"]
    # an annotated onset after the run began turns the same proposals into an early first command
    assert score_requests.main([str(manifest), str(out), str(ledger), str(tmp_path / "s2.json"), "--early-limit-s", "0.3"]) == 0
    foreign = tmp_path / "foreign.json"
    foreign.write_text(ledger.read_text().replace('"d435_global_time_epoch_s"', '"another_clock"'))
    with pytest.raises(ValueError, match="clock"):
        score_requests.main([str(manifest), str(out), str(foreign), str(tmp_path / "x.json")])


def test_choose_rate_window_cli_applies_the_declared_rule_and_checks_the_stored_window(tmp_path, monkeypatch, clip):
    manifest, seg = clip
    out = run(tmp_path, monkeypatch, "run", *FIXED, "--object-provider", "masks", "--masks", str(seg))
    ledger, _ = timeline(tmp_path, manifest, out)
    stem = tmp_path / "rate_window"
    assert choose_rate_window.main([str(manifest), str(ledger), str(stem), "--primary-run", str(out), "--check-run", str(out), "--windows", "1,2,4", "--tolerances", "0,0.1"]) == 0
    d = json.loads(Path(str(stem) + ".json").read_text())
    assert d["windows"] == [1, 2, 4] and d["tolerances"] == [0., .1]
    assert d["runs"]["primary"]["stored_rate_window"] == 1 and d["runs"]["primary"]["stored_window_reproduces_trace_statuses"] is True
    by = d["runs"]["primary"]["by_window"]
    assert all(len(by[str(k)]["0.0"]["requests"]) == 1 for k in (1, 2, 4))
    assert d["eligible_windows"] and d["chosen_rate_window"] in d["eligible_windows"]
    best = min(d["eligible_windows"], key=lambda k: (by[str(k)]["0.0"]["onsets_per_hand_minute"], k))
    assert d["chosen_rate_window"] == best and "Rule:" in Path(str(stem) + ".md").read_text()
    with pytest.raises(SystemExit):
        choose_rate_window.main([str(manifest), str(ledger), str(stem), "--primary-run", str(out), "--windows", "0,1"])


def test_run_perception_applies_and_records_the_rate_window(tmp_path, monkeypatch, clip):
    manifest, seg = clip
    out = run(tmp_path, monkeypatch, "run4", *FIXED, "--rate-window", "4", "--object-provider", "masks", "--masks", str(seg))
    rows = read_jsonl(out / "trace.jsonl")
    assert rows[0]["configuration"]["rate_window"] == 4
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["settings"]["replay_configuration"]["rate_window"] == 4


def test_replay_packets_reproduces_the_stored_run_and_applies_another_window(tmp_path, monkeypatch, clip):
    import replay_packets
    manifest, seg = clip
    out = run(tmp_path, monkeypatch, "run", *FIXED, "--object-provider", "masks", "--masks", str(seg))
    absent = tmp_path / "absent_registry.csv"
    monkeypatch.setattr(replay_packets, "git_state", lambda *a, **k: dict(commit=None, note="not queried in tests"))
    assert replay_packets.main([str(out), str(tmp_path / "re1"), "--rate-window", "1", "--manifest-csv", str(absent)]) == 0
    stored = read_jsonl(out / "trace.jsonl"); again = read_jsonl(tmp_path / "re1" / "trace.jsonl")
    assert len(again) == len(stored) and all([o["status"] for o in r["outputs"]] == [o["status"] for o in s["outputs"]] for r, s in zip(again, stored))
    summary = json.loads((tmp_path / "re1" / "summary.json").read_text())
    assert summary["kind"] == "replay_of_stored_packets" and summary["replay_configuration"]["rate_window"] == 1 and summary["frames"] == len(stored)
    assert summary["evaluation"]["counts"] == json.loads((out / "summary.json").read_text())["evaluation"]["counts"]
    assert replay_packets.main([str(out), str(tmp_path / "re4"), "--rate-window", "4", "--manifest-csv", str(absent)]) == 0
    four = read_jsonl(tmp_path / "re4" / "trace.jsonl")
    assert four[0]["configuration"]["rate_window"] == 4 and (tmp_path / "re4" / "frame_times.csv").read_text() == (out / "frame_times.csv").read_text()
    provenance = json.loads((tmp_path / "re4" / "provenance.json").read_text())
    assert provenance["source_packets_sha256"] == summary["source_packets_sha256"] and provenance["replay_configuration"]["rate_window"] == 4
    with pytest.raises(FileExistsError):
        replay_packets.main([str(out), str(tmp_path / "re4"), "--manifest-csv", str(absent)])
