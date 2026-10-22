"""Zero-shot bare-hand models on glove frames: WiLoR and HaMeR, same boxes, same frames.

Purpose: measure how the published bare-hand models behave on the V0-base
glove frames *before* any fine-tuning. Outputs per frame: detector boxes with left/right class and
confidence, and per box the 21 projected 2-D joints from each model. A summary counts frames with
zero / one / two hands found, left-right agreement between the detector class and image position,
and (when a landmark ledger is given) the pixel distance between the model's MCP joints and the
human labels. Nothing here is accuracy in the reference's sense; it is the software baseline the
fine-tuned tracker must beat.

Usage:
    python zero_shot_hands.py <frames_dir> <out_dir> --wilor <wilor_repo> [--hamer <hamer_repo>]
        [--ledger labels.jsonl] [--every 1] [--conf 0.3]

The MANO right-hand model must be present where each repo expects it (WiLoR: mano_data/MANO_RIGHT.pkl;
HaMeR: _DATA/data/mano/MANO_RIGHT.pkl). Models are loaded lazily so a missing one is reported, not fatal.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# MANO joint order used by both models (21 joints): 0 wrist, 1-4 thumb (1 = CMC/trapezium-ish, 2 = thumb MCP),
# 5-8 index (5 = index MCP), 9-12 middle, 13-16 ring, 17-20 pinky.
JOINT_NAMES = ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "index_mcp", "index_pip", "index_dip",
               "index_tip", "middle_mcp", "middle_pip", "middle_dip", "middle_tip", "ring_mcp", "ring_pip",
               "ring_dip", "ring_tip", "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"]
FUNCTIONAL_FROM_MANO = {"thumb_mcp": 2, "index_mcp": 5, "trapezium": 1}


def project(points3d, cam_t, focal, img_wh):
    pts = points3d + cam_t[None, :]
    x = focal * pts[:, 0] / pts[:, 2] + img_wh[0] / 2.
    y = focal * pts[:, 1] / pts[:, 2] + img_wh[1] / 2.
    return np.stack([x, y], axis=1)


class WilorRunner:
    name = "wilor"

    def __init__(self, repo: Path, device):
        import torch
        sys.path.insert(0, str(repo))
        cwd = Path.cwd()
        import os
        os.chdir(repo)  # the repo resolves mano_data/ and pretrained_models/ relative to cwd
        try:
            from wilor.models import load_wilor
            from wilor.datasets.vitdet_dataset import ViTDetDataset
            from wilor.utils import recursive_to
            from wilor.utils.renderer import cam_crop_to_full
            from ultralytics import YOLO
            self.model, self.cfg = load_wilor(checkpoint_path="./pretrained_models/wilor_final.ckpt",
                                              cfg_path="./pretrained_models/model_config.yaml")
            self.detector = YOLO("./pretrained_models/detector.pt").to(device)
        finally:
            os.chdir(cwd)
        self.ViTDetDataset, self.recursive_to, self.cam_crop_to_full = ViTDetDataset, recursive_to, cam_crop_to_full
        self.torch = torch
        self.device = device
        self.model = self.model.to(device).eval()

    def detect(self, bgr, conf):
        det = self.detector(bgr, conf=conf, verbose=False)[0]
        boxes, right, confs = [], [], []
        for d in det:
            b = d.boxes.data.cpu().numpy().squeeze()
            boxes.append(b[:4].tolist()); right.append(int(d.boxes.cls.cpu().item())); confs.append(float(d.boxes.conf.cpu().item()))
        return boxes, right, confs

    def estimate(self, bgr, boxes, right):
        torch = self.torch
        ds = self.ViTDetDataset(self.cfg, bgr, np.array(boxes), np.array(right), rescale_factor=2.0)
        dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
        results = []
        for batch in dl:
            batch = self.recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)
            mult = (2 * batch["right"] - 1)
            pred_cam = out["pred_cam"]; pred_cam[:, 1] = mult * pred_cam[:, 1]
            img_size = batch["img_size"].float()
            focal = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * img_size.max()
            cam_t = self.cam_crop_to_full(pred_cam, batch["box_center"].float(), batch["box_size"].float(),
                                          img_size, focal).detach().cpu().numpy()
            for n in range(batch["img"].shape[0]):
                joints = out["pred_keypoints_3d"][n].detach().cpu().numpy().copy()
                is_right = int(batch["right"][n].cpu().item())
                joints[:, 0] = (2 * is_right - 1) * joints[:, 0]
                results.append(dict(right=is_right, joints2d=project(joints, cam_t[n], float(focal), img_size[n].cpu().numpy()).tolist()))
        return results


class HamerRunner:
    name = "hamer"

    def __init__(self, repo: Path, device):
        import torch, os
        sys.path.insert(0, str(repo))
        cwd = Path.cwd(); os.chdir(repo)
        try:
            from hamer.models import load_hamer, DEFAULT_CHECKPOINT
            from hamer.datasets.vitdet_dataset import ViTDetDataset
            from hamer.utils import recursive_to
            from hamer.utils.renderer import cam_crop_to_full
            self.model, self.cfg = load_hamer(DEFAULT_CHECKPOINT)
        finally:
            os.chdir(cwd)
        self.ViTDetDataset, self.recursive_to, self.cam_crop_to_full = ViTDetDataset, recursive_to, cam_crop_to_full
        self.torch = torch; self.device = device
        self.model = self.model.to(device).eval()

    estimate = WilorRunner.estimate  # identical crop/projection pipeline; boxes come from the WiLoR detector


def assign_by_position(boxes):
    """Image-position rule for two boxes: the box further left in the image is called 'image_left'."""
    if len(boxes) != 2:
        return None
    cx = [(b[0] + b[2]) / 2 for b in boxes]
    return ["image_left", "image_right"] if cx[0] < cx[1] else ["image_right", "image_left"]


def load_labels(path):
    if not path:
        return {}
    labels = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("state") == "visible":
                labels.setdefault(r["frame_id"], {}).setdefault(r["side"], {})[r["point"]] = (r["u"], r["v"])
    return labels


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("frames_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--wilor", type=Path, required=True)
    ap.add_argument("--hamer", type=Path)
    ap.add_argument("--ledger", type=Path)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--overlay", action="store_true")
    a = ap.parse_args(argv)
    if a.ledger:
        ap.error("use run_perception.py plus evaluate_perception.py for exact-key, box-matched reference evaluation")
    import functools, torch
    # Both checkpoints predate torch 2.6's weights_only default; their sha256 are recorded in the run log.
    torch.load = functools.partial(torch.load, weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    runners = {}
    errors = {}
    for name, cls, repo in (("wilor", WilorRunner, a.wilor), ("hamer", HamerRunner, a.hamer)):
        if repo is None:
            continue
        try:
            runners[name] = cls(repo.resolve(), device)
        except Exception as exc:  # report, do not hide
            errors[name] = f"{type(exc).__name__}: {exc}"
    if "wilor" not in runners:
        print(json.dumps({"fatal": "WiLoR (detector) unavailable", "errors": errors}, indent=1)); return 1
    frames = sorted(p for p in a.frames_dir.glob("*.png"))[:: max(1, a.every)]
    rows = []
    counts = {"frames": 0, "hands_found": {0: 0, 1: 0, 2: 0, "more": 0}, "two_hand_class_matches_position": 0,
              "two_hand_frames": 0, "time_s": {}}
    err_px = {m: {p: [] for p in FUNCTIONAL_FROM_MANO} for m in runners}
    for path in frames:
        bgr = cv2.imread(str(path)); counts["frames"] += 1
        t0 = time.perf_counter(); boxes, right, confs = runners["wilor"].detect(bgr, a.conf); t_det = time.perf_counter() - t0
        k = len(boxes); counts["hands_found"][k if k <= 2 else "more"] += 1
        row = dict(frame=path.name, boxes=boxes, right_class=right, conf=confs, detect_s=t_det, models={})
        if k == 2:
            counts["two_hand_frames"] += 1
            pos = assign_by_position(boxes)
            # detector class 1 = right hand; from a head camera the right hand is usually on the image right
            if all((r == 1) == (p == "image_right") for r, p in zip(right, pos)):
                counts["two_hand_class_matches_position"] += 1
        for name, runner in runners.items():
            if not boxes:
                row["models"][name] = []; continue
            t0 = time.perf_counter(); est = runner.estimate(bgr, boxes, right); dt = time.perf_counter() - t0
            counts["time_s"].setdefault(name, []).append(dt)
            row["models"][name] = est
        rows.append(row)
        if a.overlay:
            img = bgr.copy()
            for b, r, c in zip(boxes, right, confs):
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0) if r else (0, 165, 255), 1)
                cv2.putText(img, f"{'R' if r else 'L'} {c:.2f}", (int(b[0]), int(b[1]) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            colours = {"wilor": (255, 0, 255), "hamer": (255, 255, 0)}
            for name, est in row["models"].items():
                for e in est:
                    for j in FUNCTIONAL_FROM_MANO.values():
                        u, v = e["joints2d"][j]; cv2.circle(img, (int(u), int(v)), 3, colours.get(name, (255, 255, 255)), -1)
            cv2.imwrite(str(a.out_dir / f"overlay_{path.stem}.png"), img)
    (a.out_dir / "detections.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summary = dict(scope="zero-shot bare-hand models on glove frames; software baseline, not accuracy; N=1 operator, no participant",
                   frames_dir=str(a.frames_dir), conf=a.conf, models=list(runners), model_errors=errors, counts=counts,
                   median_infer_s={m: float(np.median(v)) for m, v in counts["time_s"].items()},
                   label_error_px={m: {p: (dict(n=len(v), median=float(np.median(v)), p90=float(np.percentile(v, 90))) if v else None)
                                       for p, v in d.items()} for m, d in err_px.items()})
    summary["counts"]["time_s"] = {m: len(v) for m, v in counts["time_s"].items()}
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
