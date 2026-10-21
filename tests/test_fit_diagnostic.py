"""The one-person fitting diagnostic (fit_diagnostic.py, 16 October 2026): it fits and selects with the comparison fold's own
functions, refuses more than one person and never reports a contrast. Synthetic, participant N=0."""
import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

import fit_diagnostic
from fit_diagnostic import SCOPE, diagnose
from sg_offline.evaluation import Opportunity
from sg_offline.experiment import compare
from test_review_implementation import people_sessions  # noqa: F401


def one_person_two_sessions():
    """p1's session s plus a second p1 session s2 (p2's packets re-identified: a software check, not data) and p3 untouched."""
    p1, p2, p3 = people_sessions()
    s2 = dict(packets=[replace(p, participant_id="p1", session_id="s2") for p in p2["packets"]],
              opportunities=[Opportunity(o.opportunity_id, "p1", "s2", o.start_s, o.end_s, o.acceptable, o.provenance) for o in p2["opportunities"]])
    return p1, s2, p2, p3


ARMS = ("rules", "mi_learned", "kt_r")
GRID = (.3, .6, .9)


def test_diagnostic_fits_and_selects_exactly_like_the_comparison_fold_with_the_same_split():
    s1, s2, _, p3 = one_person_two_sessions()
    # compare's fold for held-out p3: one training person with two sessions -> the last session is the development split
    folds = compare([s1, s2, p3], arms=ARMS, steps=12, selection="development", max_unwanted_proposals_per_hand_minute=1e9,
                    thresholds=GRID, synthetic_check=True, outer_test_people=("p3",))["folds"]
    records, models = diagnose([s1], [s2], ARMS, steps=12, thresholds=GRID, budget=1e9, synthetic_check=True)
    assert [f["development_sessions"] for f in folds] == [[("p1", "s2")]] * len(folds) and [f["train_sessions"] for f in folds] == [[("p1", "s")]] * len(folds)
    assert len(records) == len(models) == len(folds) == len(ARMS)
    for fold, record in zip(folds, records):
        assert record["arm"] == fold["arm"] and record["training_sessions"] == fold["train_sessions"] and record["development_sessions"] == fold["development_sessions"]
        for key in ("loss_before", "loss_after", "early_stopping", "reporting_threshold", "development_selection_reason",
                    "development_budget_feasible", "development_budget_unit", "parameter_count", "steps"):
            assert record[key] == fold[key], key
        assert [(p["threshold"], p["feasible"], p["success"]) for p in record["development_curve"]] == \
               [(p["threshold"], p["feasible"], p["success"]) for p in fold["development_curve"]]
    learned = [r for r in records if r["arm"] != "rules"]
    assert all(r["early_stopping"]["criterion"] == "development log loss" and len(r["early_stopping"]["development_loss_curve"]) == 12 for r in learned)
    assert all(r["fit_provenance"] == "synthetic_experiment_software_check_only" for r in learned) and records[0]["steps"] == 0
    assert "N = 1" in SCOPE and not any("contrast" in k for r in records for k in r)


def test_GUARD_diagnostic_is_one_person_one_role_per_session_and_a_declared_budget():
    s1, s2, p2, _ = one_person_two_sessions()
    with pytest.raises(ValueError, match="one person"):
        diagnose([s1], [p2], ARMS, steps=2, budget=1e9, synthetic_check=True)
    with pytest.raises(ValueError, match="one role"):
        diagnose([s1], [s1], ARMS, steps=2, budget=1e9, synthetic_check=True)
    with pytest.raises(ValueError, match="at least one"):
        diagnose([s1], [], ARMS, steps=2, budget=1e9, synthetic_check=True)
    with pytest.raises(ValueError, match="budget"):
        diagnose([s1], [s2], ARMS, steps=2, synthetic_check=True)
    with pytest.raises(ValueError, match="unknown model arm"):
        diagnose([s1], [s2], ("kt_r", "external_method"), steps=2, budget=1e9, synthetic_check=True)


cv2 = pytest.importorskip("cv2")
import import_requests  # noqa: E402
from sg_offline.ledger import session_origin_s  # noqa: E402
from test_end_to_end_step3 import FIXED, clip, make_recording, read_jsonl, run  # noqa: E402,F401  (clip is a fixture)


def second_session(tmp_path, session="s2"):
    folder = tmp_path / "rec2"; folder.mkdir()
    manifest, _ = make_recording(folder)
    data = json.loads(manifest.read_text()); data["session_id"] = session
    manifest.write_text(json.dumps(data))
    return manifest


def ledger_for(tmp_path, manifest, out, session):
    """Rest, then the right hand's request from 0.05 s before its first proposal to the end; the left hand is never observed."""
    origin, clock = session_origin_s(manifest)
    rows = read_jsonl(out / "trace.jsonl"); times = list(csv.DictReader(open(out / "frame_times.csv")))
    first = next(r for r in rows if r["outputs"][1]["status"] == "proposed_simulation_only")
    frame = next(t for t in times if t["frame_id"] == first["packet_id"])
    onset = round(float(frame["session_relative_acquired_s"]) - .05, 3); duration = round(float(times[-1]["session_relative_available_s"]) + .2, 2)
    path = tmp_path / f"timeline_{session}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["start_s", "end_s", "left", "right", "note"])
        w.writerow(["0", onset, "NQ", "NQ", ""]); w.writerow([onset, duration, "EXCLUDED", "G", "left never observed"])
    ledger = tmp_path / f"requests_{session}.json"
    import_requests.main([str(path), str(ledger), "--participant", "operator", "--session", session, "--annotator", "ann", "--clock", clock, "--duration-s", str(duration)])
    return ledger


def test_fit_diagnostic_cli_fits_replays_scores_and_refuses_an_untouched_session(tmp_path, monkeypatch, clip):
    manifest1, _ = clip
    manifest2 = second_session(tmp_path)
    (tmp_path / "plane_settings.json").write_text(json.dumps(dict(min_area_px=50, max_range_m=5.)))
    plane = ("--object-provider", "plane", "--plane-settings", str(tmp_path / "plane_settings.json"))
    run1 = run(tmp_path, monkeypatch, "run_s1", *FIXED, *plane)
    run2 = run(tmp_path, monkeypatch, "run_s2", *FIXED, *plane, manifest=manifest2)
    ledger1 = ledger_for(tmp_path, manifest1, run1, "s"); ledger2 = ledger_for(tmp_path, manifest2, run2, "s2")
    monkeypatch.setattr(fit_diagnostic, "git_state", lambda *a, **k: dict(commit=None, note="not queried in tests"))
    spec = dict(scope="synthetic software check", rate_window=1, arms=["rules", "mi_learned", "kt_r"], seeds=[0], steps=5, lr=.01, regularization=.001,
                thresholds=[.3, .6, .9], max_unwanted_proposals_per_hand_minute=1e9, budget_status="test", burden_unit="onsets", early_limit_s=.1,
                training=[dict(run=str(run1), manifest=str(manifest1), ledger=str(ledger1))],
                development=[dict(run=str(run2), manifest=str(manifest2), ledger=str(ledger2))], untouched=["s3"])
    spec_path = tmp_path / "spec.json"; spec_path.write_text(json.dumps(spec))
    out = tmp_path / "diag"; absent = tmp_path / "absent_registry.csv"
    assert fit_diagnostic.main([str(spec_path), str(out), "--manifest-csv", str(absent)]) == 0
    report = json.loads((out / "fit_diagnostic.json").read_text())
    assert report["schema"] == "sg-fit-diagnostic/1" and report["participants"] == ["operator"] and report["untouched"] == ["s3"]
    assert [s["role"] for s in report["sessions"]] == ["training", "development"] and [s["session_id"] for s in report["sessions"]] == ["s", "s2"]
    assert report["perception"]["identical_across_sessions"] is True and report["registered_in_manifest_csv"] is False and not absent.exists()
    assert [a["arm"] for a in report["arms"]] == ["rules", "mi_learned", "kt_r"]
    for arm in report["arms"]:
        assert arm["training_examples"] > 0 and [r["role"] for r in arm["replays"]] == ["training", "development"]
        for r in arm["replays"]:
            folder = Path(r["trace_dir"])
            assert folder == out / f"{arm['arm']}_seed0" / folder.name and folder.name in ("s", "s2")
            assert all((folder / f).is_file() for f in ("trace.jsonl", "provenance.json", "summary.json", "scoring_strict.json", "scoring_tol0.1.json"))
            rows = read_jsonl(folder / "trace.jsonl")
            assert rows[0]["configuration"]["proposal_score_threshold"] == arm["reporting_threshold"] and rows[0]["inputs"]["session_id"] == folder.name
            assert set(r["scores"]) == {"strict", "tol0.1"} and r["scores"]["tol0.1"]["early_limit_s"] == .1
            assert json.loads((folder / "scoring_tol0.1.json").read_text())["early_limit_s"] == .1
    learned = [a for a in report["arms"] if a["arm"] != "rules"]
    assert all(a["fit_provenance"].startswith("recorded_ledger_labels|participants=operator|") and a["early_stopping"]["max_passes"] == 5 for a in learned)
    assert report["arms"][0]["steps"] == 0 and report["arms"][0]["fit_provenance"] == "unvalidated_rules"
    text = (out / "fit_diagnostic_report.md").read_text()
    assert "| rules |" in text and "in-sample" in text and "N = 1" in text
    with pytest.raises(FileExistsError):
        fit_diagnostic.main([str(spec_path), str(out), "--manifest-csv", str(absent)])
    # a session declared untouched cannot be loaded in any role; the synthetic switch is not reachable from a specification
    bad = dict(spec, untouched=["s2"]); (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="untouched"):
        fit_diagnostic.main([str(tmp_path / "bad.json"), str(tmp_path / "diag_bad"), "--manifest-csv", str(absent)])
    (tmp_path / "syn.json").write_text(json.dumps(dict(spec, synthetic_check=True)))
    with pytest.raises(ValueError, match="synthetic"):
        fit_diagnostic.main([str(tmp_path / "syn.json"), str(tmp_path / "diag_syn"), "--manifest-csv", str(absent)])


def test_diagnostic_runs_under_layout_v2_with_release_admitted(tmp_path, monkeypatch, clip):
    """16 Oct 2026 (later): a specification may name feature_layout v2 (the run's scene.jsonl is used) and admitted_actions;
    the readouts are built for the layout, the replay sees the scenes and the scoring admits the named actions. Before the
    fix the v2 fit raised 'model feature layout v1 differs from the bundle layout v2' and the scoring admitted G only."""
    manifest1, _ = clip
    manifest2 = second_session(tmp_path)
    (tmp_path / "plane_settings.json").write_text(json.dumps(dict(min_area_px=50, max_range_m=5.)))
    plane = ("--object-provider", "plane", "--plane-settings", str(tmp_path / "plane_settings.json"))
    run1 = run(tmp_path, monkeypatch, "run_s1", *FIXED, *plane)
    run2 = run(tmp_path, monkeypatch, "run_s2", *FIXED, *plane, manifest=manifest2)
    assert (run1 / "scene.jsonl").is_file() and (run2 / "scene.jsonl").is_file()          # written by run_perception since 16 Oct 2026
    ledger1 = ledger_for(tmp_path, manifest1, run1, "s"); ledger2 = ledger_for(tmp_path, manifest2, run2, "s2")
    monkeypatch.setattr(fit_diagnostic, "git_state", lambda *a, **k: dict(commit=None, note="not queried in tests"))
    spec = dict(scope="synthetic software check", rate_window=1, arms=["rules_actions", "mi_learned", "kt_r"], seeds=[0], steps=3, lr=.01, regularization=.001,
                thresholds=[.3, .9], max_unwanted_proposals_per_hand_minute=1e9, budget_status="test", burden_unit="onsets", early_limit_s=.1,
                feature_layout="v2", admitted_actions=["G", "R"],
                training=[dict(run=str(run1), manifest=str(manifest1), ledger=str(ledger1))],
                development=[dict(run=str(run2), manifest=str(manifest2), ledger=str(ledger2))], untouched=["s3"])
    spec_path = tmp_path / "spec_v2.json"; spec_path.write_text(json.dumps(spec))
    out = tmp_path / "diag_v2"; absent = tmp_path / "absent_registry.csv"
    assert fit_diagnostic.main([str(spec_path), str(out), "--manifest-csv", str(absent)]) == 0
    report = json.loads((out / "fit_diagnostic.json").read_text())
    assert [a["arm"] for a in report["arms"]] == ["rules_actions", "mi_learned", "kt_r"] and report["arms"][0]["steps"] == 0
    assert all(s["scene"] is not None and s["admitted_actions"] == ["G", "R"] for s in report["sessions"])
    for arm in report["arms"]:
        for r in arm["replays"]:
            rows = read_jsonl(Path(r["trace_dir"]) / "trace.jsonl")
            assert rows[0]["configuration"]["feature_layout"] == "v2" and "scene" in rows[0]
            assert json.loads((Path(r["trace_dir"]) / "scoring_tol0.1.json").read_text())["admitted_actions"] == ["G", "R"]
    assert report["arms"][2]["parameter_count"] == 49 * 70                                   # kt_r sized for layout v2
    with pytest.raises(ValueError, match="scene"):
        bad = dict(spec, training=[dict(run=str(run1), manifest=str(manifest1), ledger=str(ledger1), scene=str(tmp_path / "missing.jsonl"))])
        (tmp_path / "bad.json").write_text(json.dumps(bad)); fit_diagnostic.main([str(tmp_path / "bad.json"), str(tmp_path / "d_bad"), "--manifest-csv", str(absent)])
