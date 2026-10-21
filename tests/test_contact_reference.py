"""Contact-sensor onset reference. Synthetic only, participant N = 0."""
import json

import pytest

from sg_offline.contact import CONTACT_SCHEMA, ContactSample, compare_onsets, contact_events, read_contact_file


def _series(side, pairs):
    return [ContactSample(side, t, v) for t, v in pairs]


def test_event_rule_marks_onset_rise_drop_and_offset():
    samples = _series("right", [(0.0, 0), (0.1, 0), (0.2, 120), (0.3, 125), (0.4, 140), (0.5, 139), (0.6, 110), (0.7, 0)])
    events = contact_events(samples, change=.10)
    assert [(e.kind, e.t_s) for e in events] == [("onset", .2), ("rise", .4), ("drop", .6), ("offset", .7)]
    assert all(e.side == "right" for e in events)
    with pytest.raises(ValueError):
        contact_events(samples, change=1.5)
    with pytest.raises(ValueError, match="nonnegative"):
        ContactSample("left", 0., -1.)


def test_file_format_is_a_side_record_on_the_recording_clock(tmp_path):
    data = dict(schema=CONTACT_SCHEMA, participant_id="p01", session_id="s01", clock="d435_global_time_epoch_s", units="mV",
                sensor="index_fingertip", samples=[["left", 1.0, 0], ["left", 1.1, 90], ["right", 1.0, 0], ["right", 1.05, 0]])
    path = tmp_path / "contact.json"; path.write_text(json.dumps(data))
    header, samples = read_contact_file(path)
    assert header["clock"] == "d435_global_time_epoch_s" and len(samples) == 4 and contact_events(samples)[0].t_s == 1.1
    data["samples"].append(["right", 1.02, 5]); path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="time-ordered"):
        read_contact_file(path)
    data["schema"] = "other"; path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema"):
        read_contact_file(path)


def test_onset_comparison_pairs_nearest_within_window_and_reports_signed_differences():
    sensor = [10.0, 20.0, 30.0]
    annotated = [9.4, 19.7, 31.0, 45.0]          # intention precedes contact by 0.6 and 0.3 s; one late; one unpaired
    r = compare_onsets(sensor, annotated, max_pair_s=2.0)
    assert r["paired"] == 3 and r["candidate_unpaired"] == 1 and r["reference_unpaired"] == 0
    assert r["median_difference_s"] == pytest.approx(-.3) and r["p90_absolute_difference_s"] == pytest.approx(.92, abs=1e-9)   # |diffs| sorted 0.3, 0.6, 1.0; linear 90th percentile
    assert compare_onsets([], [1.0])["median_difference_s"] is None
    with pytest.raises(ValueError):
        compare_onsets(sensor, annotated, max_pair_s=0)
