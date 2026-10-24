"""Provider comparison report. Synthetic run directories, N=0."""
import json

import pytest

import compare_object_providers as cmp


def write_run(tmp_path, name, provider, objects_by_frame, proposals_by_frame):
    """objects_by_frame: {frame_index: {identity: xyz or None}}; proposals_by_frame: {frame_index: set of sides proposing}."""
    d = tmp_path / name; d.mkdir()
    diag = [dict(frame_index=f, frame_id=f"s/f{f}", hands={}, objects={o: ("" if v is not None else "unavailable") for o, v in objs.items()},
                 object_xyz={o: (list(v) if v is not None else None) for o, v in objs.items()}) for f, objs in sorted(objects_by_frame.items())]
    (d / "perception_diagnostics.jsonl").write_text("".join(json.dumps(x) + "\n" for x in diag))
    rows = [dict(packet_id=f"p{f}", decision_s=f / 10, inputs=dict(participant_id="p", session_id="s"),
                 outputs=[dict(side=s, raw_outcome="G" if s in proposals_by_frame.get(f, set()) else "NQ",
                               status="proposed_simulation_only" if s in proposals_by_frame.get(f, set()) else "no_new_request", reason="")
                          for s in ("left", "right")]) for f in sorted(objects_by_frame)]
    (d / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "summary.json").write_text(json.dumps(dict(object_provider=dict(kind=provider), evaluation=dict(counts={}), object_processing_s_median=.01,
                                                     queue_policy="fixed_latency", decimate=2)))
    return d


def test_comparison_matches_objects_by_position_and_reports_availability_and_proposals(tmp_path):
    frames = range(0, 10)
    colour = write_run(tmp_path, "c", "colour", {f: {"blue": (0.1, 0., .6), "cream": ((0.3, 0., .6) if f < 5 else None)} for f in frames}, {3: {"left"}})
    masks = write_run(tmp_path, "m", "masks", {f: {"blue_cylinder": (0.11, 0., .61), "cream_cylinder": (0.3, 0.01, .6)} for f in frames}, {3: {"left"}, 4: {"left"}})
    plane = write_run(tmp_path, "p", "plane", {f: {"object_1": (0.1, 0., .6)} for f in frames}, {})
    report = cmp.compare([colour, masks, plane])
    assert report["identical_frame_sets"] is True
    by = {r["provider"]: r for r in report["runs"]}
    assert by["colour"]["availability"] == {"blue": 10, "cream": 5} and by["masks"]["frames_with_two_objects"] == 10 and by["plane"]["frames_with_two_objects"] == 0
    assert by["colour"]["unavailable_reasons"]["cream"] == {"unavailable": 5}
    assert by["masks"]["proposal_ticks"] == {"left": 2} and by["masks"]["first_proposal_s"] == {"left": .3} and by["plane"]["proposal_ticks"] == {}
    ag = report["pairwise_agreement"]["colour_vs_masks"]
    assert ag["matched_object_frames"] == 15 and ag["only_in_first"] == 0 and ag["only_in_second"] == 5          # cream missing from colour in 5 frames
    assert ag["distance_m"]["median"] == pytest.approx(.01414, abs=1e-4)      # 10 blue pairs at 14.1 mm, 5 cream pairs at 10 mm
    assert report["pairwise_agreement"]["colour_vs_plane"]["only_in_first"] == 5
    text = cmp.markdown(report)
    assert "| masks |" in text and "colour_vs_masks" in text and "not accuracy" in text
    out = tmp_path / "rep.json"
    assert cmp.main([str(out), str(colour), str(masks)]) == 0 and out.exists() and out.with_suffix(".md").exists()


def test_an_object_is_never_paired_with_a_different_object_of_the_other_run(tmp_path):
    """Each object of the first run used to be paired with the nearest unused object of
    the second within 0.15 m, whatever its name. With the recorded cylinders 0.09-0.10 m apart, a frame in which one
    run saw only the blue and the other only the cream cylinder was recorded as a 0.1 m "position difference" (16
    such pairs in the stored trial_2 record). Runs that use the same names are now paired by name."""
    frames = range(10)
    a = write_run(tmp_path, "a", "colour", {f: {"blue": (0., 0., .6), "cream": None} for f in frames}, {})
    b = write_run(tmp_path, "b", "masks", {f: {"blue": None, "cream": (.1, 0., .6)} for f in frames}, {})
    ag = cmp.compare([a, b])["pairwise_agreement"]["colour_vs_masks"]
    assert (ag["matched_object_frames"], ag["only_in_first"], ag["only_in_second"]) == (0, 10, 10)      # before: 10 matched at 0.10 m
    assert ag["distance_m"]["median"] is None and ag["pairing"] == "by_identity_name"
    # same names, both objects in one run and one in the other: the true pair is kept, the other is not invented
    c = write_run(tmp_path, "c", "colour", {f: {"blue": (0., 0., .6), "cream": (.1, 0., .6)} for f in frames}, {})
    d = write_run(tmp_path, "d", "masks", {f: {"blue": None, "cream": (.1, .005, .6)} for f in frames}, {})
    ag = cmp.compare([c, d])["pairwise_agreement"]["colour_vs_masks"]
    assert (ag["matched_object_frames"], ag["only_in_first"], ag["only_in_second"]) == (10, 10, 0)
    assert ag["distance_m"]["max"] == pytest.approx(.005)                                                  # before: 0.10 m (blue paired with cream)


def test_generic_identities_are_paired_by_position_one_to_one_within_an_object_sized_gate(tmp_path):
    """The plane provider's identities are generic, so only position can pair them. The gate must be smaller than
    the spacing of distinct objects: with 0.15 m a lone cluster was paired with the other cylinder 0.1 m away."""
    frames = range(10)
    named = write_run(tmp_path, "m", "masks", {f: {"blue": (0., 0., .6), "cream": (None if f < 5 else (.1, 0., .6))} for f in frames}, {})
    generic = write_run(tmp_path, "p", "plane", {f: {"object_1": (.1, .004, .6), "object_2": (None if f < 5 else (0., .003, .6))} for f in frames}, {})
    ag = cmp.compare([named, generic])["pairwise_agreement"]["masks_vs_plane"]
    # frames 0-4: masks sees only blue (x=0), plane only a cluster on the cream cylinder (x=0.1): different objects, no pair
    # frames 5-9: both see both; object_1 pairs with cream and object_2 with blue although the names say nothing
    assert (ag["matched_object_frames"], ag["only_in_first"], ag["only_in_second"]) == (10, 5, 5)          # before: 15 matched, max 0.10 m
    assert ag["distance_m"]["max"] == pytest.approx(.004)
    assert ag["pairing"] == "by_nearest_position" and ag["max_match_m"] == .05


def test_two_runs_of_one_provider_kind_keep_separate_rows_and_pairs(tmp_path):
    """Pairs were keyed "<kind>_vs_<kind>": comparing colour, old masks and re-segmented masks in one call kept one of
    the two colour-versus-masks results and showed two indistinguishable "masks" rows."""
    frames = range(4)
    colour = write_run(tmp_path, "run_colour", "colour", {f: {"blue": (0., 0., .6)} for f in frames}, {})
    old = write_run(tmp_path, "run_masks_old", "masks", {f: {"blue": (0., .01, .6)} for f in frames}, {})
    new = write_run(tmp_path, "run_masks_new", "masks", {f: {"blue": (0., .02, .6)} for f in frames}, {})
    report = cmp.compare([colour, old, new])
    assert len(report["pairwise_agreement"]) == 3                                                          # before: 2
    assert report["pairwise_agreement"]["colour_vs_masks:run_masks_old"]["distance_m"]["median"] == pytest.approx(.01)
    assert report["pairwise_agreement"]["colour_vs_masks:run_masks_new"]["distance_m"]["median"] == pytest.approx(.02)
    assert [r["label"] for r in report["runs"]] == ["colour", "masks:run_masks_old", "masks:run_masks_new"]
    assert "| masks:run_masks_old |" in cmp.markdown(report)


def test_runs_that_cannot_be_compared_are_refused_not_reported_as_empty(tmp_path):
    """A run folder written by an older version has no `frame_index`/`object_xyz` in its diagnostics; it was joined by row
    number and reported with zero objects. Two runs of different recordings were compared without a word."""
    frames = range(4)
    good = write_run(tmp_path, "good", "colour", {f: {"blue": (0., 0., .6)} for f in frames}, {})
    old = write_run(tmp_path, "old", "colour", {f: {"blue": (0., 0., .6)} for f in frames}, {})
    rows = [json.loads(l) for l in (old / "perception_diagnostics.jsonl").read_text().splitlines()]
    (old / "perception_diagnostics.jsonl").write_text("".join(json.dumps({k: v for k, v in r.items() if k not in ("frame_index", "object_xyz")}) + "\n" for r in rows))
    with pytest.raises(ValueError, match="frame_index"):
        cmp.compare([good, old])
    other = write_run(tmp_path, "other", "masks", {f: {"blue": (0., 0., .6)} for f in frames}, {})
    for folder, sha in ((good, "a" * 64), (other, "b" * 64)):
        summary = json.loads((folder / "summary.json").read_text()); summary["manifest_sha256"] = sha
        (folder / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="different recordings"):
        cmp.compare([good, other])
