"""build_scene.py refuses a manifest that is not the run's own and reuses the run's plane settings. Synthetic, participant N=0."""
import json

import pytest

pytest.importorskip("cv2")

import build_scene  # noqa: E402
from test_end_to_end_step3 import FIXED, clip, read_jsonl, run  # noqa: E402,F401  (clip is a fixture)


def _plane_run(tmp_path, monkeypatch):
    (tmp_path / "plane_settings.json").write_text(json.dumps(dict(min_area_px=50, max_range_m=5., inlier_m=.02)))
    return run(tmp_path, monkeypatch, "run_plane", *FIXED, "--object-provider", "plane", "--plane-settings", str(tmp_path / "plane_settings.json"))


def test_the_rebuilt_scene_uses_the_runs_plane_settings(tmp_path, monkeypatch, clip, capsys):
    manifest, _ = clip
    out = _plane_run(tmp_path, monkeypatch)
    assert build_scene.main([str(out), str(manifest), str(tmp_path / "scene")]) == 0
    provenance = json.loads((tmp_path / "scene" / "scene_provenance.json").read_text())
    assert provenance["plane_settings"]["inlier_m"] == .02 and provenance["plane_settings"]["min_area_px"] == 50
    assert len(read_jsonl(tmp_path / "scene" / "scene.jsonl")) == len(read_jsonl(out / "packets.jsonl"))


def test_a_manifest_from_another_recording_is_refused(tmp_path, monkeypatch, clip):
    manifest, _ = clip
    out = _plane_run(tmp_path, monkeypatch)
    data = json.loads(manifest.read_text())
    other = dict(data, participant_id="someone_else")
    (tmp_path / "other.json").write_text(json.dumps(other))
    with pytest.raises(SystemExit):                                   # the hash differs from the run's provenance
        build_scene.main([str(out), str(tmp_path / "other.json"), str(tmp_path / "scene_other")])
    # without the run's provenance the content checks still catch it
    (out / "provenance.json").unlink()
    with pytest.raises(ValueError, match="participants or sessions"):
        list(build_scene.build_scenes(out, tmp_path / "other.json"))
    def later(stamp):
        return dict(stamp, acquired_s=stamp["acquired_s"] + 500., available_s=stamp["available_s"] + 500.)
    shifted = dict(data, frames=[dict(f, depth_stamp=later(f["depth_stamp"]), rgb_stamp=later(f["rgb_stamp"])) for f in data["frames"]])
    (manifest.parent / "shifted.json").write_text(json.dumps(shifted))
    with pytest.raises(ValueError, match="not made from frame"):
        list(build_scene.build_scenes(out, manifest.parent / "shifted.json"))
