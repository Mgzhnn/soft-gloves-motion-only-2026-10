"""Regressions in fit_diagnostic.py and label_frames_web.py (software, participant N = 0).

1. fit_diagnostic.py read only the specification keys it knew and silently ignored the rest, so a misspelt declared
   setting (``early_limit`` for ``early_limit_s``), the run_comparison-style ``config`` block or ``threshold_grid`` ran
   with the code defaults (strict tolerance, best_action_threshold, default grid) and recorded them as if declared.
2. label_frames_web.py offered ``--pass unmarked|marked``; the ledger accepts ``unmarked``, ``marker_on`` and
   ``marker_off`` (sg_offline.ledger.MARKER_PASSES), so every label of a ``marked`` session was refused and the marker
   passes of the annotation protocol could not be labelled on the server.
"""
import json

import pytest

import fit_diagnostic
from sg_offline.ledger import MARKER_PASSES, read_landmark_ledger


def _spec(**extra):
    return dict(scope="synthetic key check", rate_window=8, arms=["rules"], max_unwanted_proposals_per_hand_minute=10, early_limit_s=.1,
                training=[dict(run="run_1", manifest="m1.json", ledger="l1.json")],
                development=[dict(run="run_2", manifest="m2.json", ledger="l2.json")], **extra)


@pytest.mark.parametrize("spec, named", [
    (_spec(early_limit=.1), "early_limit"),
    (_spec(config={"decision_rule": "marginal_argmax"}), "config"),
    (_spec(threshold_grid=[.3, .6]), "threshold_grid"),
    (dict(_spec(), training=[dict(run="run_1", manifest="m1.json", ledger="l1.json", origin_s=0.)]), "origin_s"),
])
def test_fit_diagnostic_refuses_unknown_specification_keys(tmp_path, spec, named):
    path = tmp_path / "spec.json"; path.write_text(json.dumps(spec))
    out = tmp_path / "out"
    with pytest.raises(ValueError, match=named):
        fit_diagnostic.run_diagnostic(path, out, tmp_path / "absent_registry.csv")
    assert not out.exists() and not (tmp_path / "absent_registry.csv").exists()


def test_label_frames_web_offers_the_ledger_marker_passes(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    import label_frames_web
    from test_end_to_end_step3 import make_recording
    (tmp_path / "rec").mkdir()
    manifest, _ = make_recording(tmp_path / "rec")
    captured = {}

    class _Stop:
        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    def fake_serve(session, indices, *args, **kwargs):
        captured["session"] = session
        return _Stop()

    monkeypatch.setattr(label_frames_web, "serve", fake_serve)
    for marker_pass in MARKER_PASSES:
        ledger = tmp_path / f"labels_{marker_pass}.jsonl"
        assert label_frames_web.main([str(manifest), str(ledger), "--annotator", "ann", "--pass", marker_pass, "--frames", "0"]) == 0
        session = captured.pop("session")
        session.set_label(session.recording.records[0]["frame_id"], "left", "hand_box", "out_of_frame")
        assert [l.marker_pass for l in read_landmark_ledger(ledger)] == [marker_pass]
    with pytest.raises(SystemExit):   # a pass the ledger cannot store is refused before any frame is served
        label_frames_web.main([str(manifest), str(tmp_path / "labels_marked.jsonl"), "--annotator", "ann", "--pass", "marked", "--frames", "0"])
    assert "session" not in captured
