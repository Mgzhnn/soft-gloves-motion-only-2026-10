"""Click-to-label tool for the three functional points per hand on converted RGB-D frames.

Writes an ``sg-landmark-ledger/1`` JSONL file (see ``sg_offline/ledger.py``). Labels are human
references for M0 (tracker coverage, identity and error); they never enter a runtime packet.

Usage (GUI):
    python label_frames.py data/<clip>_sg/recording.json labels.jsonl --annotator <name> --pass unmarked --every 5

Keys inside the window:
    b + drag     draw a hand box (independent of point visibility)
    click        place the current point (state -> visible)
    1 / 2 / 3    select point: thumb_mcp / index_mcp / trapezium
    l / r        select left / right hand
    o / x / a    mark current point occluded / out_of_frame / ambiguous (clears any pixel)
    n / p        next / previous frame        (labels are written to disk on every change)
    z            delete the current point's label on this frame
    q            quit

Non-GUI helpers (``render_labels``, ``frame_indices``) are unit-tested; the GUI loop is not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sg_offline.ledger import (FUNCTIONAL_POINTS, LandmarkLabel, MARKER_PASSES, frame_time_rows, read_landmark_ledger,
                               session_origin_s, write_landmark_ledger)
from sg_offline.recordings import RGBDRecording

COLOURS = {"left": (0, 200, 255), "right": (255, 120, 0)}
MARKS = {"thumb_mcp": "T", "index_mcp": "I", "trapezium": "Z"}


def frame_indices(n_frames: int, every: int, explicit: list[int] | None) -> list[int]:
    if explicit:
        bad = [i for i in explicit if not 0 <= i < n_frames]
        if bad:
            raise ValueError(f"frame indices outside recording: {bad}")
        return sorted(set(explicit))
    if every < 1:
        raise ValueError("--every must be >= 1")
    return list(range(0, n_frames, every))


def render_labels(rgb: np.ndarray, labels, frame_id: str, header: str = "") -> np.ndarray:
    """Draw the labels of one frame on a BGR copy of ``rgb``; pure function, used by the GUI and tests."""
    import cv2
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    y = 18
    if header:
        cv2.putText(bgr, header, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1); y += 16
    for l in labels:
        if l.frame_id != frame_id:
            continue
        colour = COLOURS[l.side]
        if l.point == "hand_box" and l.box is not None:
            cv2.rectangle(bgr, tuple(int(v) for v in l.box[:2]), tuple(int(v) for v in l.box[2:]), colour, 2)
            continue
        if l.state == "visible":
            cv2.circle(bgr, (int(round(l.u)), int(round(l.v))), 4, colour, -1)
            cv2.putText(bgr, f"{l.side[0].upper()}{MARKS[l.point]}", (int(round(l.u)) + 5, int(round(l.v)) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
        else:
            cv2.putText(bgr, f"{l.side} {l.point}: {l.state}", (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1); y += 14
    cv2.putText(bgr, "human reference labels; not runtime inputs", (6, bgr.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    return bgr


class Session:
    def __init__(self, recording: RGBDRecording, out_path: Path, annotator: str, marker_pass: str, glove_condition="unspecified"):
        self.recording = recording
        self.frames = recording  # lazy indexed frames; never load a whole long RGB-D clip into RAM
        self.glove_condition = glove_condition
        self.out_path = out_path
        self.annotator = annotator
        self.marker_pass = marker_pass
        self.labels = {l.key(): l for l in read_landmark_ledger(out_path)} if out_path.exists() else {}
        # Session-relative zero = availability of the first frame (ledger.session_origin_s), so the time shown
        # with each frame is the number an annotator writes into the request timeline CSV (2 Oct 2026).
        self.origin_s, self.clock = session_origin_s(recording.path)

        if any((l.annotator, l.marker_pass) != (annotator, marker_pass) for l in self.labels.values()):
            raise ValueError("existing file belongs to another annotator/pass; use a fresh independent file")

    def time_of(self, index: int) -> float:
        """Session-relative acquisition time of frame ``index`` in seconds (no array is loaded)."""
        return float(self.recording.records[index]["rgb_stamp"]["acquired_s"]) - self.origin_s

    def frame_times(self):
        return frame_time_rows(self.recording.records)

    def set_label(self, frame_id, side, point, state, u=None, v=None, box=None):
        label = LandmarkLabel(frame_id, self.recording.participant_id, self.recording.session_id, side, point,
                              state, u, v, self.annotator, self.marker_pass, box=box, glove_condition=self.glove_condition)
        self.labels[label.key()] = label
        self.flush()

    def delete(self, frame_id, side, point):
        self.labels.pop((self.recording.participant_id, self.recording.session_id, frame_id, side, point), None)
        self.flush()

    def flush(self):
        write_landmark_ledger(self.out_path, self.labels.values())


def run_gui(session: Session, indices: list[int]) -> None:  # pragma: no cover - interactive
    import cv2
    state = {"side": "right", "point": FUNCTIONAL_POINTS[0], "pos": 0}
    window = "label_frames (q quits)"

    def on_mouse(event, x, y, flags, _):
        frame = session.frames[indices[state["pos"]]]
        if state["point"] == "hand_box":
            if event == cv2.EVENT_LBUTTONDOWN:
                state["corner"] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and "corner" in state:
                x0, y0 = state.pop("corner")
                if x0 != x and y0 != y:
                    session.set_label(frame.frame_id, state["side"], "hand_box", "visible",
                                      box=(min(x0,x), min(y0,y), max(x0,x), max(y0,y)))
        elif event == cv2.EVENT_LBUTTONDOWN:
            session.set_label(frame.frame_id, state["side"], state["point"], "visible", float(x), float(y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        frame = session.frames[indices[state["pos"]]]
        header = (f"frame {indices[state['pos']]} ({state['pos'] + 1}/{len(indices)})  t={session.time_of(indices[state['pos']]):.3f}s  "
                  f"hand={state['side']}  point={state['point']}  pass={session.marker_pass}")
        cv2.imshow(window, render_labels(frame.rgb, session.labels.values(), frame.frame_id, header))
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        elif k in (ord("1"), ord("2"), ord("3")):
            state["point"] = FUNCTIONAL_POINTS[k - ord("1")]
        elif k == ord("b"):
            state["point"] = "hand_box"
        elif k == ord("l"):
            state["side"] = "left"
        elif k == ord("r"):
            state["side"] = "right"
        elif k in (ord("o"), ord("x"), ord("a")):
            st = {ord("o"): "occluded", ord("x"): "out_of_frame", ord("a"): "ambiguous"}[k]
            session.set_label(frame.frame_id, state["side"], state["point"], st)
        elif k == ord("z"):
            session.delete(frame.frame_id, state["side"], state["point"])
        elif k == ord("n"):
            state["pos"] = min(state["pos"] + 1, len(indices) - 1)
        elif k == ord("p"):
            state["pos"] = max(state["pos"] - 1, 0)
    cv2.destroyAllWindows()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", type=Path, help="sg-rgbd-recording/1 recording.json")
    ap.add_argument("ledger", type=Path, help="output JSONL (appended to / rewritten if it exists)")
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--pass", dest="marker_pass", choices=MARKER_PASSES, required=True)
    ap.add_argument("--glove-condition", required=True, help="physical glove/marker condition, e.g. Schiek-unmarked")
    ap.add_argument("--every", type=int, default=5, help="label every Nth frame")
    ap.add_argument("--frames", type=int, nargs="*", help="explicit frame indices instead of --every")
    ap.add_argument("--render", type=Path, help="no GUI: write PNGs of the selected frames with existing labels here")
    ap.add_argument("--frame-times", type=Path, help="no GUI: write a CSV of frame index -> session-relative seconds and exit")
    a = ap.parse_args(argv)
    session = Session(RGBDRecording(a.manifest), a.ledger, a.annotator, a.marker_pass, a.glove_condition)
    indices = frame_indices(len(session.frames), a.every, a.frames)
    if a.frame_times:
        import csv
        rows = session.frame_times()
        a.frame_times.parent.mkdir(parents=True, exist_ok=True)
        with open(a.frame_times, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        print(json.dumps({"frame_times": str(a.frame_times), "frames": len(rows), "clock": session.clock, "origin_s": session.origin_s}))
        return 0
    if a.render:
        import cv2
        a.render.mkdir(parents=True, exist_ok=True)
        for i in indices:
            frame = session.frames[i]
            cv2.imwrite(str(a.render / f"frame_{i:04d}.png"),
                        render_labels(frame.rgb, session.labels.values(), frame.frame_id, f"frame {i}  t={session.time_of(i):.3f}s"))
        print(json.dumps({"rendered": len(indices), "labels_on_disk": len(session.labels)}))
        return 0
    run_gui(session, indices)
    print(json.dumps({"labels_on_disk": len(session.labels), "ledger": str(a.ledger)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
