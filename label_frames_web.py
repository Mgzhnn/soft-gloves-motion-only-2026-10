"""Browser version of label_frames.py for a server without a display.

Same ledger, same ``Session`` and the same drawing as the window tool: the page shows one frame at its native size
with the existing labels drawn, and the keys and mouse actions are those of the window tool (b + drag = hand box;
1 / 2 / 3 = functional point by click; l / r = hand; o / x / a = occluded / out of frame / ambiguous; z = delete;
n / p = next / previous). The server binds 127.0.0.1 only; reach it from another machine through an SSH tunnel
(``ssh -L 8765:localhost:8765 <server>``) and open http://localhost:8765. Single annotator per file, as before.
usage: label_frames_web.py RECORDING_JSON LEDGER_JSONL --annotator NAME --pass unmarked|marker_on|marker_off --glove-condition TEXT [--every N | --frames i j ...] [--port 8765]
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from label_frames import Session, render_labels
from sg_offline.ledger import FUNCTIONAL_POINTS, LANDMARK_STATES, MARKER_PASSES
from sg_offline.recordings import RGBDRecording

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>label_frames (browser)</title>
<style>body{font-family:sans-serif;background:#222;color:#ddd;margin:12px}#hdr{margin:6px 0;font-size:14px}#img{cursor:crosshair;display:block}
#help{font-size:13px;color:#aaa;max-width:900px}b{color:#fff}kbd{background:#444;padding:1px 5px;border-radius:3px}</style></head><body>
<div id="hdr">loading</div><img id="img" draggable="false">
<div id="help"><kbd>b</kbd> then drag = hand box &nbsp; <kbd>1</kbd>/<kbd>2</kbd>/<kbd>3</kbd> then click = thumb MCP / index MCP / trapezium &nbsp;
<kbd>l</kbd>/<kbd>r</kbd> = left/right hand &nbsp; <kbd>o</kbd>/<kbd>x</kbd>/<kbd>a</kbd> = occluded / out of frame / ambiguous (for the current hand+point) &nbsp;
<kbd>z</kbd> = delete current label &nbsp; <kbd>n</kbd>/<kbd>p</kbd> or arrows = next/previous frame. Every action is saved at once.</div>
<script>
let pos=0,side="right",point="hand_box",down=null,n=0;
const img=document.getElementById("img"),hdr=document.getElementById("hdr");
async function refresh(){const r=await fetch("/state?pos="+pos);const s=await r.json();n=s.n;pos=s.pos;
 hdr.innerHTML=`<b>frame ${s.index}</b> (${s.pos+1}/${s.n}) &nbsp; t=${s.t.toFixed(3)} s &nbsp; hand=<b>${side}</b> &nbsp; mode=<b>${point}</b> &nbsp; labels on this frame: ${s.labels}`;
 img.src="/frame/"+pos+"?"+Date.now();}
async function post(url,body){await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});await refresh();}
img.addEventListener("mousedown",e=>{if(point==="hand_box")down=[e.offsetX,e.offsetY];else post("/label",{pos,side,point,state:"visible",u:e.offsetX,v:e.offsetY});});
img.addEventListener("mouseup",e=>{if(point==="hand_box"&&down){const b=[Math.min(down[0],e.offsetX),Math.min(down[1],e.offsetY),Math.max(down[0],e.offsetX),Math.max(down[1],e.offsetY)];down=null;if(b[2]>b[0]&&b[3]>b[1])post("/label",{pos,side,point:"hand_box",state:"visible",box:b});}});
document.addEventListener("keydown",e=>{const k=e.key;
 if(k==="n"||k==="ArrowRight"){pos=Math.min(pos+1,n-1);refresh();}else if(k==="p"||k==="ArrowLeft"){pos=Math.max(pos-1,0);refresh();}
 else if(k==="l"){side="left";refresh();}else if(k==="r"){side="right";refresh();}else if(k==="b"){point="hand_box";refresh();}
 else if(k==="1"||k==="2"||k==="3"){point=["thumb_mcp","index_mcp","trapezium"][+k-1];refresh();}
 else if(k==="o"||k==="x"||k==="a"){post("/label",{pos,side,point,state:{o:"occluded",x:"out_of_frame",a:"ambiguous"}[k]});}
 else if(k==="z"){post("/delete",{pos,side,point});}});
refresh();
</script></body></html>"""


def make_handler(session: Session, indices: list[int]):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body, ctype):
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if u.path == "/state":
                pos = min(max(int(parse_qs(u.query).get("pos", ["0"])[0]), 0), len(indices) - 1)
                frame_id = session.recording.records[indices[pos]]["frame_id"]
                with lock:
                    count = sum(1 for l in session.labels.values() if l.frame_id == frame_id)
                return self._send(200, json.dumps(dict(pos=pos, n=len(indices), index=indices[pos], t=session.time_of(indices[pos]), labels=count)).encode(), "application/json")
            if u.path.startswith("/frame/"):
                import cv2
                pos = min(max(int(u.path.split("/")[2]), 0), len(indices) - 1)
                frame = session.frames[indices[pos]]
                with lock:
                    bgr = render_labels(frame.rgb, list(session.labels.values()), frame.frame_id, f"frame {indices[pos]}  t={session.time_of(indices[pos]):.3f}s")
                ok, png = cv2.imencode(".png", bgr)
                return self._send(200, png.tobytes(), "image/png")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            u = urlparse(self.path)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0")) or 0) or b"{}")
            pos = min(max(int(body.get("pos", 0)), 0), len(indices) - 1)
            frame_id = session.recording.records[indices[pos]]["frame_id"]
            side, point = body.get("side"), body.get("point")
            try:
                with lock:
                    if u.path == "/label":
                        state = body.get("state", "visible")
                        if state not in LANDMARK_STATES:
                            raise ValueError("unknown state")
                        if point == "hand_box":
                            box = body.get("box")
                            session.set_label(frame_id, side, "hand_box", state, box=tuple(float(v) for v in box) if box else None)
                        else:
                            session.set_label(frame_id, side, point, state, float(body["u"]) if state == "visible" else None, float(body["v"]) if state == "visible" else None)
                    elif u.path == "/delete":
                        session.delete(frame_id, side, point)
                    else:
                        return self._send(404, b"not found", "text/plain")
            except (ValueError, KeyError, TypeError) as exc:
                return self._send(400, str(exc).encode(), "text/plain")
            self._send(200, b"ok", "text/plain")

    return Handler


def serve(session: Session, indices: list[int], host="127.0.0.1", port=8765):
    server = ThreadingHTTPServer((host, port), make_handler(session, indices))
    return server


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", type=Path); ap.add_argument("ledger", type=Path)
    ap.add_argument("--annotator", required=True); ap.add_argument("--pass", dest="marker_pass", required=True, choices=MARKER_PASSES)   # the ledger's passes
    ap.add_argument("--glove-condition", default="unspecified")
    ap.add_argument("--every", type=int, default=16); ap.add_argument("--frames", type=int, nargs="*")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    recording = RGBDRecording(a.manifest)
    indices = sorted(set(a.frames)) if a.frames else list(range(0, len(recording), a.every))
    if any(i < 0 or i >= len(recording) for i in indices) or not indices:
        ap.error("frame indices outside the recording")
    session = Session(recording, a.ledger, a.annotator, a.marker_pass, a.glove_condition)
    server = serve(session, indices, a.host, a.port)
    print(f"label_frames_web: {len(indices)} frames of {recording.session_id}; ledger {a.ledger}; open http://{a.host}:{a.port} (Ctrl+C stops)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
