"""Coverage pins for the step-3 tools whose 7 October 2026 mutants still survived both suites when rerun
on 16 October 2026: S1 (score_requests.score ignores the declared early tolerance), C3 (choose_rate_window drops the
eligibility rule), C4 (the chooser scores every tolerance strictly) and C5 (the chooser maximises the burden). Tests
only; no behaviour changed. Synthetic end-to-end clip, participant N = 0.
"""
import csv
import json

import pytest

cv2 = pytest.importorskip("cv2")

import choose_rate_window  # noqa: E402
import import_requests  # noqa: E402
import score_requests  # noqa: E402
from sg_offline.ledger import session_origin_s  # noqa: E402
from test_end_to_end_step3 import FIXED, clip, read_jsonl, run  # noqa: E402,F401  (clip is a fixture)
from test_step3_cli import timeline  # noqa: E402


def _masks_run(tmp_path, monkeypatch, clip):
    manifest, seg = clip
    return manifest, run(tmp_path, monkeypatch, "run", *FIXED, "--object-provider", "masks", "--masks", str(seg))


def _late_onset_ledger(tmp_path, manifest, out, after_first_s):
    """Rest, then the right hand's request from ``after_first_s`` after its first proposal (decision time) to the end:
    the run that answers the request starts that much before the annotated onset. The left hand is never observed."""
    origin, clock = session_origin_s(manifest)
    rows = read_jsonl(out / "trace.jsonl"); times = list(csv.DictReader(open(out / "frame_times.csv")))
    first = next(r for r in rows if r["outputs"][1]["status"] == "proposed_simulation_only")
    onset = round(first["decision_s"] - origin + after_first_s, 3); duration = round(float(times[-1]["session_relative_available_s"]) + .2, 2)
    path = tmp_path / "late_timeline.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["start_s", "end_s", "left", "right", "note"])
        w.writerow(["0", onset, "NQ", "NQ", ""]); w.writerow([onset, duration, "EXCLUDED", "G", "left never observed"])
    ledger = tmp_path / "late_requests.json"
    import_requests.main([str(path), str(ledger), "--participant", "operator", "--session", "s", "--annotator", "ann", "--clock", clock, "--duration-s", str(duration)])
    return ledger, onset - (first["decision_s"] - origin)


def test_the_scorer_applies_the_declared_early_tolerance(tmp_path, monkeypatch, clip):
    manifest, out = _masks_run(tmp_path, monkeypatch, clip)
    ledger, early = _late_onset_ledger(tmp_path, manifest, out, .04)
    strict = score_requests.score(manifest, out, ledger)
    (req,) = strict["requests"]
    assert req["result"] == "wrong_timing_first" and req["early_by_s"] == pytest.approx(early, abs=1e-6) and 0 < early < .1
    assert req["proposals_in_window"] >= 1 and strict["per_participant_success"]["operator"]["compatible_first"] == 0
    tolerant = score_requests.score(manifest, out, ledger, early_limit_s=.1)
    (req,) = tolerant["requests"]
    assert tolerant["early_limit_s"] == .1 and req["result"] == "compatible_first" and req["early_by_s"] == pytest.approx(early, abs=1e-6)
    assert tolerant["per_participant_success"]["operator"]["compatible_first"] == 1
    # the command line passes the tolerance through to the written report
    report = tmp_path / "tol.json"
    assert score_requests.main([str(manifest), str(out), str(ledger), str(report), "--early-limit-s", "0.1"]) == 0
    d = json.loads(report.read_text())
    assert d["early_limit_s"] == .1 and d["requests"][0]["result"] == "compatible_first"
    # a tolerance below the early margin still fails the request
    assert score_requests.score(manifest, out, ledger, early_limit_s=early / 2)["requests"][0]["result"] == "wrong_timing_first"


# Designed per-window scores: rate window -> (proposal ticks inside each request hand's window, unwanted onsets per
# no-request hand-minute), at the strict tolerance; the 0.1 s entry makes every window eligible with no unwanted onset,
# so a chooser reading anything but the strict entry selects differently.
DESIGNS = [
    ({2: ([0], 0.), 4: ([2], 3.), 6: ([1], 5.)}, [4, 6], 4),            # the silent window is not eligible; then the fewest onsets
    ({2: ([1], 3.), 4: ([1], 3.), 6: ([1], 3.)}, [2, 4, 6], 2),         # ties -> the smaller window
    ({2: ([], 0.), 4: ([3, None], 1.), 6: ([1, 1], 2.)}, [6], 6),       # no scored request, or one hand without a tick: not eligible
    ({2: ([0], 0.), 4: ([], 0.), 6: ([None], 0.)}, [], None),           # nothing eligible: no window is chosen
]


@pytest.mark.parametrize("design, eligible, chosen", DESIGNS)
def test_the_rate_window_rule_is_eligibility_then_fewest_strict_onsets(tmp_path, monkeypatch, clip, design, eligible, chosen):
    manifest, out = _masks_run(tmp_path, monkeypatch, clip)
    ledger, _ = timeline(tmp_path, manifest, out)
    seen = []

    def designed_score(rows, inside, origin, early):
        k = rows[0]["configuration"]["rate_window"]; seen.append((k, early))
        ticks, onsets = design[k] if early == 0. else ([1], 0.)
        return dict(marker=(k, early), onsets_per_hand_minute=onsets, requests=[dict(proposals_in_window=n) for n in ticks])

    monkeypatch.setattr(choose_rate_window, "_score", designed_score)
    result = choose_rate_window.choose(manifest, ledger, out, windows=(2, 4, 6), tolerances=(0., .1))
    assert result["runs"]["primary"]["stored_rate_window"] == 1 and result["runs"]["primary"]["stored_window_reproduces_trace_statuses"] is True
    assert sorted(seen) == [(k, t) for k in (2, 4, 6) for t in (0., .1)]                # every window at every declared tolerance
    by = result["runs"]["primary"]["by_window"]
    assert all(by[str(k)][str(t)]["marker"] == (k, t) for k in (2, 4, 6) for t in (0., .1))
    assert result["eligible_windows"] == eligible and result["chosen_rate_window"] == chosen
    assert result["chosen_is_at_the_edge_of_the_candidate_set"] == (None if chosen is None else chosen in (2, 6))
