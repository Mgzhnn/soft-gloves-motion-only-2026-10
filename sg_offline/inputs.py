"""Small JSONL observation reader. No label or future-frame field is admitted."""
import json
from pathlib import Path
from .contracts import Stamp, Source, Point, RawTension, Packet


def stamp(data):
    data = dict(data); data["source"] = Source(data["source"])
    return Stamp(**data)


def point(data):
    data = dict(data); data["stamp"] = stamp(data["stamp"])
    data["xyz"] = tuple(data["xyz"]) if data["xyz"] is not None else None
    return Point(**data)


def packets_from_jsonl(path):
    for line in Path(path).read_text().splitlines():
        data = json.loads(line)
        data["hands"] = tuple(point(p) for p in data["hands"])
        data["objects"] = tuple(point(p) for p in data.get("objects", []))
        telemetry = []
        for raw in data.get("telemetry", []):
            raw["stamp"] = stamp(raw["stamp"]); telemetry.append(RawTension(**raw))
        data["telemetry"] = tuple(telemetry)
        for key in ("previous_commands", "previous_command_times_s"):
            if key in data:
                data[key] = tuple(data[key])
        yield Packet(**data)  # Unknown fields, including references, fail closed.
