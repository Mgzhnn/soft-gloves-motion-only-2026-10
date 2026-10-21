"""Scoring hand-over regressions; synthetic offline software, participant N=0."""
import json
from dataclasses import replace

import pytest

import choose_rate_window
import score_requests
from sg_offline.evaluation import Opportunity, evaluate_opportunities
from sg_offline.inputs import packets_from_jsonl
from sg_offline.replay import Config, Replay, write_trace
from test_end_to_end_step3 import FIXED, clip, read_jsonl, run
from test_step3_cli import timeline


def setup_run(tmp_path, monkeypatch, clip):
    manifest, seg = clip
    out = run(tmp_path, monkeypatch, "run", *FIXED, "--object-provider", "masks", "--masks", str(seg))
    ledger, _ = timeline(tmp_path, manifest, out)
    return manifest, out, ledger


@pytest.mark.parametrize("fault", ["manifest_identity", "manifest_origin", "later_clock", "later_identity", "side_order", "configuration"])
def test_scorer_rejects_inconsistent_recording_trace_join(tmp_path, monkeypatch, clip, fault):
    manifest, out, ledger = setup_run(tmp_path, monkeypatch, clip)
    if fault.startswith("manifest"):
        data = json.loads(manifest.read_text())
        if fault == "manifest_identity":
            data["session_id"] = "foreign"
        else:
            for frame in data["frames"]:
                for stream in ("rgb_stamp", "depth_stamp"):
                    for key in ("acquired_s", "available_s"):
                        frame[stream][key] += 100
        manifest.write_text(json.dumps(data))
    else:
        rows = read_jsonl(out / "trace.jsonl")
        if fault == "later_clock": rows[1]["inputs"]["clock"] = "foreign"
        elif fault == "later_identity": rows[1]["inputs"]["session_id"] = "foreign"
        elif fault == "side_order": rows[1]["outputs"].reverse()
        else: rows[1]["configuration"]["rate_window"] = 8
        write_trace(out / "trace.jsonl", rows)
    with pytest.raises(ValueError):
        score_requests.score(manifest, out, ledger)


def test_window_selection_preserves_the_source_configuration(tmp_path, monkeypatch, clip):
    manifest, out, ledger = setup_run(tmp_path, monkeypatch, clip)
    packets = tuple(packets_from_jsonl(out / "packets.jsonl"))
    write_trace(out / "trace.jsonl", Replay(Config(proposal_score_threshold=.99)).run(packets))
    result = choose_rate_window.choose(manifest, ledger, out, windows=(1, 2), tolerances=(0.,))
    assert result["runs"]["primary"]["stored_window_reproduces_trace_statuses"] is True
    assert result["chosen_rate_window"] is None
    assert result["eligible_windows"] == []


def test_window_selection_stops_when_the_stored_trace_does_not_reproduce(tmp_path, monkeypatch, clip):
    manifest, out, ledger = setup_run(tmp_path, monkeypatch, clip)
    rows = read_jsonl(out / "trace.jsonl")
    rows[0]["outputs"][0]["status"] = "proposed_simulation_only"
    write_trace(out / "trace.jsonl", rows)
    with pytest.raises(ValueError, match="reproduc"):
        choose_rate_window.choose(manifest, ledger, out, windows=(1,), tolerances=(0.,))


@pytest.mark.parametrize("destination", ["score", "choice_json", "choice_md"])
def test_scoring_tools_preserve_existing_reports(tmp_path, monkeypatch, clip, destination):
    manifest, out, ledger = setup_run(tmp_path, monkeypatch, clip)
    stem = tmp_path / "report"
    existing = stem.with_suffix(".md" if destination == "choice_md" else ".json")
    existing.write_text("original record\n")
    with pytest.raises(FileExistsError):
        if destination == "score":
            score_requests.main(list(map(str, [manifest, out, ledger, existing])))
        else:
            choose_rate_window.main([str(manifest), str(ledger), str(stem), "--primary-run", str(out), "--windows", "1"])
    assert existing.read_text() == "original record\n"


def test_printed_summary_reports_actual_misses_and_wrong_actions():
    report = {"per_participant_success": {"p": {"evaluable_request_hands": 2, "wrong_first": 1, "missed_including_withheld": 1}},
              "unwanted_exposure": {}}
    line = score_requests.summary_line(report)
    assert "missed 1" in line and "wrong action 1" in line
    assert "wrong object" not in line


def test_in_window_count_excludes_proposals_elsewhere_in_the_row():
    op = Opportunity("q", "p", "s", 0., 2., (("G",), ("NQ",)), "synthetic",
                     hand_windows_s=((1., 2.), (0., 2.)), request_ids=("g", None))
    rows = [{"inputs": {"participant_id": "p", "session_id": "s"}, "packet_id": str(k), "decision_s": t,
             "outputs": [{"status": status, "raw_outcome": "G"}, {"status": "no_new_request", "raw_outcome": "NQ"}]}
            for k, (t, status) in enumerate([(.5, "proposed_simulation_only"), (1.5, "no_new_request")])]
    outcomes = evaluate_opportunities(rows, [op])
    assert outcomes[0]["hands"][0]["result"] == "wrong_timing_first"
    assert outcomes[0]["hands"][0]["proposal_count"] == 1
    assert score_requests.request_rows(outcomes, 0.)[0]["proposals_in_window"] == 0


def test_request_report_keeps_the_recorded_early_run_caveat():
    op = Opportunity("q", "p", "s", 1., 2., (("G",), ("NQ",)), "synthetic", request_ids=("g", None))
    rows = [{"inputs": {"participant_id": "p", "session_id": "s"}, "packet_id": str(k), "decision_s": t,
             "outputs": [{"status": status, "raw_outcome": "G"}, {"status": "no_new_request", "raw_outcome": "NQ"}]}
            for k, (t, status) in enumerate([(.5, "proposed_simulation_only"), (.9, "no_new_request"),
                                              (1.1, "proposed_simulation_only")])]
    req = score_requests.request_rows(evaluate_opportunities(rows, [op]), 0.)[0]
    assert req["result"] == "compatible_first"
    assert req["proposals_since_rearm_before_window"] == 1


def test_invalid_source_summary_does_not_publish_partial_replay(tmp_path):
    from dataclasses import asdict
    from replay_packets import replay_run
    from sg_offline.fixtures import scenarios
    source = tmp_path / "source"
    source.mkdir()
    (source / "packets.jsonl").write_text("".join(json.dumps(asdict(p)) + "\n" for p in scenarios()[:3]))
    (source / "summary.json").write_text("{truncated")
    destination = tmp_path / "replay"
    with pytest.raises(json.JSONDecodeError):
        replay_run(source, destination, Config(), tmp_path / "absent_registry.csv")
    assert not destination.exists() or not list(destination.iterdir())


def test_registry_rejects_missing_identity_columns_before_append(tmp_path):
    from run_perception import append_manifest_row
    registry = tmp_path / "registry.csv"
    original = "runid,output,status\n"
    registry.write_text(original)
    with pytest.raises(ValueError, match="header"):
        append_manifest_row(registry, {"run_id": "probe", "output_path": "/tmp/probe", "status": "completed"})
    assert registry.read_text() == original
