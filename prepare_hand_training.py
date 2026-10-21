"""Export the same box/three-point labels for detector adaptation and a sparse pose comparator.

Input JSON: {"sessions":[{"manifest":"recording.json","ledger":"annotator.jsonl",
"split":"train|val|test"}, ...]}. Test people cannot appear in fitting/development.
No random frame splitting. Training is optional and requires an existing local checkpoint or YAML.
"""
import argparse, hashlib, json
from pathlib import Path
from sg_offline.ledger import FUNCTIONAL_POINTS, read_landmark_ledger
from sg_offline.recordings import RGBDRecording


def export(config, output, task="detect"):
    import cv2
    sessions=[];people={"train":set(),"val":set(),"test":set()};seen=set()
    for entry in config["sessions"]:
        rec=RGBDRecording(entry["manifest"]);split=entry["split"]
        if split not in people: raise ValueError("unknown split")
        key=(rec.participant_id,rec.session_id)
        if key in seen: raise ValueError("session repeated across dataset entries")
        seen.add(key);people[split].add(rec.participant_id)
        labels=read_landmark_ledger(entry["ledger"])
        if any((l.participant_id,l.session_id)!=key for l in labels): raise ValueError("ledger/recording identity mismatch")
        sessions.append((entry,rec,labels))
    if people["test"] & (people["train"]|people["val"]): raise ValueError("held-out people leaked into perception training/development")
    if not people["train"] or not people["val"]: raise ValueError("whole training and validation sessions required")
    if output.exists() and any(output.iterdir()): raise ValueError("dataset output must be empty")
    # Validate every labelled frame before writing anything, so a bad label never leaves a partial dataset.
    # A hand box marked occluded/ambiguous is a legal label but has no YOLO box: that frame is skipped and counted.
    planned=[];counts={s:0 for s in people};skipped={s:0 for s in people};provenance=[]
    for entry,rec,labels in sessions:
        split=entry["split"];grouped={}
        for l in labels: grouped.setdefault(l.frame_id,{})[(l.side,l.point)]=l
        for index,record in enumerate(rec.records):
            lab=grouped.get(record["frame_id"])
            if lab is None: continue
            if any((side,"hand_box") not in lab for side in ("left","right")):
                raise ValueError("annotate each hand box or explicitly mark it out of frame before export")
            if any(lab[(side,"hand_box")].state not in ("visible","out_of_frame") for side in ("left","right")):
                skipped[split]+=1;continue
            frame=rec[index];height,width=frame.rgb.shape[:2];lines=[]
            for side_index,side in enumerate(("left","right")):
                box=lab[(side,"hand_box")].box
                if box is None: continue
                x0,y0,x1,y1=box
                if not 0<=x0<x1<=width or not 0<=y0<y1<=height: raise ValueError("box outside image")
                row=[side_index,(x0+x1)/(2*width),(y0+y1)/(2*height),(x1-x0)/width,(y1-y0)/height]
                if task=="pose":
                    for name in FUNCTIONAL_POINTS:
                        label=lab.get((side,name))
                        if label is None: raise ValueError("pose export requires all three point visibility labels")
                        if label.state=="visible":
                            if not 0<=label.u<width or not 0<=label.v<height: raise ValueError("point outside image")
                            row.extend([label.u/width,label.v/height,2])
                        else: row.extend([0,0,0])
                lines.append(" ".join(str(v) for v in row))
            name=hashlib.sha256(f"{rec.participant_id}|{rec.session_id}|{frame.frame_id}".encode()).hexdigest()[:20]
            planned.append((split,rec,index,name,lines));counts[split]+=1
        provenance.append(dict(**entry,participant_id=rec.participant_id,session_id=rec.session_id,
            manifest_sha256=hashlib.sha256(Path(entry["manifest"]).read_bytes()).hexdigest(),
            label_sha256=hashlib.sha256(Path(entry["ledger"]).read_bytes()).hexdigest()))
    if not counts["train"] or not counts["val"]: raise ValueError("no fully labelled training/validation frames")
    output.mkdir(parents=True,exist_ok=True)
    for split,rec,index,name,lines in planned:
        frame=rec[index]
        image_path=output/"images"/split/(name+".png");label_path=output/"labels"/split/(name+".txt")
        image_path.parent.mkdir(parents=True,exist_ok=True);label_path.parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(image_path),cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR));label_path.write_text("\n".join(lines)+"\n")
    # JSON is valid YAML and avoids adding a new writer dependency.
    data=dict(path=str(output.resolve()),train="images/train",val="images/val",names={0:"left",1:"right"})
    if counts["test"]:data["test"]="images/test"
    # No flip_idx: without it, Ultralytics itself turns horizontal flips off for keypoint datasets.
    # A flip would turn a left hand into an image of a right hand while keeping the "left" class.
    if task=="pose":data.update(kpt_shape=[3,3])
    (output/"dataset.yaml").write_text(json.dumps(data,indent=2)+"\n")
    gloves={split:sorted({l.glove_condition for entry,rec,labels in sessions if entry["split"]==split for l in labels})
            for split in ("train","val","test")}
    report=dict(task=task,frames=counts,skipped_frames_hand_box_occluded_or_ambiguous=skipped,sources=provenance,
                glove_conditions=gloves,
                # Allowed (a single operator), but model selection on a training person is optimistic.
                train_val_people_overlap=sorted(people["train"]&people["val"]),
                augmentation="horizontal/vertical flips disabled: handedness classes must not be corrupted",
                training_warning="train through this script (fliplr=0, flipud=0); a separate Ultralytics detect run "
                                 "uses its default horizontal flip and would swap left/right hand classes")
    (output/"provenance.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("config",type=Path);ap.add_argument("output",type=Path)
    ap.add_argument("--task",choices=("detect","pose"),default="detect")
    ap.add_argument("--train-model",type=Path,help="optional existing local .pt or model YAML; no implicit download")
    ap.add_argument("--image-size",type=int,default=640);ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--epochs",type=int,default=100);ap.add_argument("--device",default="0");ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args(argv);config=json.loads(a.config.read_text())
    for entry in config["sessions"]:
        for field in ("manifest","ledger"):entry[field]=str((a.config.parent/entry[field]).resolve())
    report=export(config,a.output,a.task)
    if a.train_model:
        if not a.train_model.is_file():ap.error("training model must exist locally")
        from sg_offline.local_models import load_yolo, train_yolo
        model=load_yolo(a.train_model,task=a.task)
        train_yolo(model,data=str(a.output/"dataset.yaml"),epochs=a.epochs,seed=a.seed,device=a.device,
                    imgsz=a.image_size,batch=a.batch,
                    project=str(a.output/"training"),name="fit",fliplr=0.,flipud=0.,workers=0,cache=False,amp=False)
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
