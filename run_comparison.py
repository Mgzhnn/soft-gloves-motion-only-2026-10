"""Run a reproducible offline comparison from packets and independently annotated ledgers.

For externally fixed perception, use {"sessions":[...], ...settings}.
For adapted perception, use {"folds":[{"held_out_person":"p1","sessions":[...]}, ...], ...settings}.
Each session names packets, ledger, origin_s and perception_provenance. Adapted folds must use
observations produced without fitting OR selecting perception on the held-out person.
"""
import argparse,hashlib,json
from pathlib import Path
from sg_offline.inputs import packets_from_jsonl
from sg_offline.ledger import read_opportunity_ledger,to_opportunities
from sg_offline.experiment import compare, contrasts_from_folds
from sg_offline.replay import Config
from sg_offline.scene import read_scenes
from sg_offline.stats import MIN_PEOPLE_FOR_INTERVAL


def load_sessions(entries,base,held=None,admitted_actions=("G",),feature_layout="v1"):
    """16 Oct 2026: a session may name ``scene`` (scene.jsonl, needed for feature layout v2); ``admitted_actions`` are the
    actions scored as requests (default G); both are recorded in the sources."""
    sessions=[];sources=[];models=set()
    for row in entries:
        packet_path=(base/row["packets"]).resolve();ledger_path=(base/row["ledger"]).resolve()
        packets=tuple(packets_from_jsonl(packet_path));ledger=read_opportunity_ledger(ledger_path)
        scene_path=(base/row["scene"]).resolve() if row.get("scene") else None
        scenes=read_scenes(scene_path) if scene_path is not None else None
        if feature_layout!="v1" and scenes is None:raise ValueError("feature layout "+feature_layout+" needs a scene file per session (key 'scene')")
        if not packets or ledger["clock"]["name"]!=packets[0].clock:raise ValueError("ledger/packet clock mismatch")
        perception=dict(row.get("perception_provenance",{}))
        if not perception.get("model_id") or not perception.get("checkpoint_sha256"):
            raise ValueError("perception model identity and checkpoint hash required")
        # A declared hash is only a claim; when the checkpoint file is named, check it.
        if perception.get("checkpoint_path"):
            actual=hashlib.sha256((base/perception["checkpoint_path"]).resolve().read_bytes()).hexdigest()
            if actual!=perception["checkpoint_sha256"]:
                raise ValueError("declared perception checkpoint_sha256 does not match the checkpoint file")
            perception["checkpoint_hash_verified"]=True
        else:perception["checkpoint_hash_verified"]=False
        models.add((perception["model_id"],perception["checkpoint_sha256"]))
        if perception.get("fit")=="adapted":
            if held is None:raise ValueError("adapted perception needs explicit outer folds")
            if "training_people" not in perception or "development_people" not in perception:
                raise ValueError("list perception fitting and development people")
            if not all(isinstance(perception[k],list) and all(isinstance(p,str) for p in perception[k]) for k in ("training_people","development_people")):
                raise ValueError("perception training_people and development_people must be lists of person ids")  # set("p1") is {"p","1"}
            if held in set(perception["training_people"])|set(perception["development_people"]):
                raise ValueError("outer test person leaked into perception fitting or selection")
        elif perception.get("fit")!="fixed_external":raise ValueError("unknown perception fit provenance")
        ops,out=to_opportunities(ledger,row.get("origin_s"),admitted_actions)
        sessions.append(dict(packets=packets,opportunities=ops,scenes=scenes))
        sources.append(dict(packets=str(packet_path),ledger=str(ledger_path),origin_s=row.get("origin_s"),
            packets_sha256=hashlib.sha256(packet_path.read_bytes()).hexdigest(),
            ledger_sha256=hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
            scene=None if scene_path is None else dict(path=str(scene_path),sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest()),
            admitted_actions=list(admitted_actions),unsupported_intervals=len(out),perception=perception))
    if len(models)!=1:raise ValueError("one identical perception checkpoint per outer fold required")
    return sessions,sources


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("config",type=Path);ap.add_argument("output",type=Path);a=ap.parse_args(argv)
    if a.output.exists():raise ValueError("use a fresh output path to preserve comparison provenance")
    config=json.loads(a.config.read_text())
    if config.pop("synthetic_check",False):raise ValueError("synthetic check is available only in the test API")
    # The replay configuration (for example {"rate_window": 4}) is a JSON object in the file; compare() needs a Config.
    # Unknown or invalid settings are refused by Config itself.
    if "config" in config:config["config"]=Config(**config["config"])
    admitted=tuple(config.pop("admitted_actions",("G",)));layout=config["config"].feature_layout if "config" in config else "v1"
    if "folds" in config:
        declarations=config.pop("folds");reports=[];seen=set()
        if "sessions" in config:raise ValueError("choose sessions or explicit folds")
        for fold in declarations:
            held=fold["held_out_person"]
            if held in seen:raise ValueError("repeated outer test person")
            seen.add(held);sessions,sources=load_sessions(fold["sessions"],a.config.parent,held,admitted,layout)
            report=compare(sessions,outer_test_people=(held,),**config);report["sources"]=sources;reports.append(report)
        result=dict(schema="sg-fold-specific-perception-comparison/2",reports=reports,
                    paired_contrasts=contrasts_from_folds([f for report in reports for f in report["folds"]],
                                                          config.get("seeds",(0,)),config.get("contrast","primary_contrast"),
                                                          # the declared minimum also governs the combined contrasts
                                                          config.get("min_people_for_interval",MIN_PEOPLE_FOR_INTERVAL)))
    else:
        sessions,sources=load_sessions(config.pop("sessions"),a.config.parent,None,admitted,layout)
        result=compare(sessions,**config);result["sources"]=sources
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(report=str(a.output),schema=result["schema"])))
if __name__=="__main__":main()
