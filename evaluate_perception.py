"""Exact frame-key, box-matched detector/landmark evaluation; never feeds references into runtime.

Tracker counters come in two groups, each summing to its denominator: for a side with a visible reference box
(``reference_sides``): correct_identity, on_other_reference_hand, not_on_any_reference_hand, withheld_or_missed; for
a side whose reference is out of frame (``reference_out_of_frame_sides``): correctly_empty_out_of_frame,
out_of_frame_side_emitted_on_other_reference_hand, emitted_while_reference_out_of_frame. Labelled frames the run
never processed (decimated or dropped) are only counted in ``reference_frames_without_predictions``: choose the
labelled frames among the processed ones. The report names its input files with their hashes; an existing report
is never overwritten.
"""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from sg_offline.ledger import read_landmark_ledger, FUNCTIONAL_POINTS
from sg_offline.perception import match_boxes, box_iou

MATCH_MIN_IOU = .3  # same threshold for raw-detector matching and tracker-output scoring


def _tracker_side(lab, side, out, tracker, errors):
    """Score one runtime output side against the human box of the same anatomical side."""
    ref = lab[(side, "hand_box")]
    other = lab[("right" if side == "left" else "left", "hand_box")]
    if ref.box is None:
        tracker["reference_out_of_frame_sides"] += 1
        if out is None:
            tracker["correctly_empty_out_of_frame"] += 1
        elif other.box is not None and box_iou(out["box"], other.box) >= MATCH_MIN_IOU:
            # The visible hand emitted under the absent side's name. Its own counter: this side is not
            # in ``reference_sides``, so sharing ``on_other_reference_hand`` broke both groups' sums.
            tracker["out_of_frame_side_emitted_on_other_reference_hand"] += 1
        else:
            tracker["emitted_while_reference_out_of_frame"] += 1
        return
    tracker["reference_sides"] += 1
    if out is None:
        tracker["withheld_or_missed"] += 1
        return
    same = box_iou(out["box"], ref.box)
    across = box_iou(out["box"], other.box) if other.box is not None else 0.
    if same >= MATCH_MIN_IOU and same >= across:
        tracker["correct_identity"] += 1
        for point in FUNCTIONAL_POINTS:
            label = lab.get((side, point)); pred = out["points"].get(point)
            if label is not None and label.state == "visible" and pred is not None:
                errors[point].append(float(np.linalg.norm(np.array(pred)-[label.u, label.v])))
    elif across >= MATCH_MIN_IOU:
        tracker["on_other_reference_hand"] += 1  # e.g. the right glove emitted as the left hand
    else:
        tracker["not_on_any_reference_hand"] += 1  # e.g. a bystander or background box


def evaluate(rows, labels):
    grouped={};counts=Counter();errors={p:[] for p in FUNCTIONAL_POINTS};seen=set()
    tracker=Counter();tracker_errors={p:[] for p in FUNCTIONAL_POINTS};tracker_rows=0
    for label in labels:
        grouped.setdefault((label.participant_id,label.session_id,label.frame_id),{})[(label.side,label.point)]=label
    for row in rows:
        key=tuple(row[k] for k in ("participant_id","session_id","frame_id"))
        if key in seen: raise ValueError("duplicate prediction frame")
        seen.add(key); lab=grouped.get(key)
        if lab is None: counts["unlabelled_prediction_frames"]+=1;continue
        if any((s,"hand_box") not in lab or lab[(s,"hand_box")].state not in ("visible","out_of_frame") for s in ("left","right")):
            counts["incomplete_box_reference_frames"]+=1;continue
        refs=[lab[(s,"hand_box")] for s in ("left","right") if lab[(s,"hand_box")].box is not None]
        predictions=row["detections"]
        pairs=match_boxes([d["box"] for d in predictions],[r.box for r in refs],minimum_iou=MATCH_MIN_IOU)
        if "assigned" in row:
            tracker_rows+=1
            for side in ("left","right"):
                _tracker_side(lab,side,row["assigned"].get(side),tracker,tracker_errors)
        counts["labelled_frames"]+=1;counts["reference_hands"]+=len(refs)
        counts["matched_hands"]+=len(pairs);counts["missed_hands"]+=len(refs)-len(pairs)
        counts["unmatched_detections"]+=len(predictions)-len(pairs)
        for i,j in pairs:
            counts["handedness_correct"]+=int(predictions[i]["side"]==refs[j].side)
            for point in FUNCTIONAL_POINTS:
                ref=lab.get((refs[j].side,point));pred=predictions[i]["points"].get(point)
                if ref is not None and ref.state=="visible":
                    counts["visible_matched_reference_points"]+=1
                    if pred is not None: errors[point].append(float(np.linalg.norm(np.array(pred)-[ref.u,ref.v])))
    counts["reference_frames_without_predictions"]=len(set(grouped)-seen)
    summary=lambda errs:{p:dict(n=len(e),median=float(np.median(e)) if e else None) for p,e in errs.items()}
    if tracker_rows:
        tracker_result=dict(frames=tracker_rows,counts=dict(tracker),point_error_px=summary(tracker_errors),
                            scope="runtime left/right outputs after duplicate suppression and identity continuity, scored "
                                  "against the same-side human box; point error only where identity was correct")
    elif any("assigned" in row for row in rows):
        tracker_result=dict(status="tracker assignments logged, but no labelled frame with complete hand boxes to score them")
    else:
        tracker_result=dict(status="tracker assignments not logged in these rows (older run_perception output)")
    return dict(counts=dict(counts),point_error_px=summary(errors),tracker=tracker_result,
                scope="raw detector: one-to-one box match; handedness scored separately; point errors conditional on matched detection; missing frames reported")


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("detections",type=Path);ap.add_argument("labels",type=Path)
    ap.add_argument("output",type=Path);a=ap.parse_args(argv)
    if a.output.exists(): ap.error("use a fresh output file; an existing evaluation is never overwritten")
    rows=[json.loads(l) for l in a.detections.read_text().splitlines() if l.strip()]
    result=evaluate(rows,read_landmark_ledger(a.labels))
    result["inputs"]=dict(detections=str(a.detections.resolve()),detections_sha256=hashlib.sha256(a.detections.read_bytes()).hexdigest(),
                          labels=str(a.labels.resolve()),labels_sha256=hashlib.sha256(a.labels.read_bytes()).hexdigest(),
                          prediction_rows=len(rows),match_min_iou=MATCH_MIN_IOU)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
if __name__=="__main__": main()
