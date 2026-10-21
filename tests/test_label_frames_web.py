"""The browser labelling tool writes the same ledger as the window tool (synthetic clip, participant N=0)."""
import json
import threading
import urllib.request

import pytest

cv2 = pytest.importorskip("cv2")

import label_frames_web  # noqa: E402
from label_frames import Session  # noqa: E402
from sg_offline.ledger import read_landmark_ledger  # noqa: E402
from sg_offline.recordings import RGBDRecording  # noqa: E402
from test_end_to_end_step3 import clip  # noqa: E402,F401  (fixture)


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return r.status


def test_browser_tool_labels_boxes_points_and_states_into_the_ledger(tmp_path, clip):
    manifest, _ = clip
    ledger = tmp_path / "labels.jsonl"
    session = Session(RGBDRecording(manifest), ledger, "ann", "unmarked", glove_condition="plain-gloves")
    server = label_frames_web.serve(session, [0, 8, 16], port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        assert b"label_frames" in urllib.request.urlopen(base + "/").read()
        state = json.loads(urllib.request.urlopen(base + "/state?pos=1").read())
        assert state == dict(pos=1, n=3, index=8, t=pytest.approx(state["t"]), labels=0) and state["t"] > 0
        png = urllib.request.urlopen(base + "/frame/1").read()
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        assert _post(base + "/label", dict(pos=1, side="right", point="hand_box", state="visible", box=[30, 58, 46, 74])) == 200
        assert _post(base + "/label", dict(pos=1, side="right", point="thumb_mcp", state="visible", u=38, v=62)) == 200
        assert _post(base + "/label", dict(pos=1, side="left", point="hand_box", state="out_of_frame")) == 200
        labels = {(l.frame_id, l.side, l.point): l for l in read_landmark_ledger(ledger)}
        assert labels[("s/frame-0008", "right", "hand_box")].box == (30., 58., 46., 74.) and labels[("s/frame-0008", "right", "hand_box")].glove_condition == "plain-gloves"
        assert (labels[("s/frame-0008", "right", "thumb_mcp")].u, labels[("s/frame-0008", "right", "thumb_mcp")].v) == (38., 62.)
        assert labels[("s/frame-0008", "left", "hand_box")].state == "out_of_frame" and labels[("s/frame-0008", "left", "hand_box")].annotator == "ann"
        assert json.loads(urllib.request.urlopen(base + "/state?pos=1").read())["labels"] == 3
        assert _post(base + "/delete", dict(pos=1, side="right", point="thumb_mcp")) == 200
        assert ("s/frame-0008", "right", "thumb_mcp") not in {(l.frame_id, l.side, l.point) for l in read_landmark_ledger(ledger)}
        with pytest.raises(urllib.error.HTTPError):
            _post(base + "/label", dict(pos=1, side="right", point="index_mcp", state="nonsense"))
    finally:
        server.shutdown(); server.server_close()
    with pytest.raises(ValueError):   # another annotator cannot continue this file
        Session(RGBDRecording(manifest), ledger, "someone_else", "unmarked")
