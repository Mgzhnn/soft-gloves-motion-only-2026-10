"""Scene anchor and hand landmarks: the observations the five non-grasp actions need (16 October 2026).

The packet contract ``sg-offline/3`` is unchanged: a scene record is a SIDE record keyed by ``packet_id`` (``scene.jsonl``
next to ``packets.jsonl``), so every stored packet, trace and record stays valid and a replay without scenes is the
replay of before. A scene carries, per decision tick:

* ``SupportPlane`` — the table plane fitted from this frame's registered depth (seeded RANSAC, ``tableplane.py``), as a unit
  normal and offset in the frame's optical frame. Height above a world-fixed plane is the same number whatever the
  camera does, so heights, height rates (vertical velocity), in-plane distances and swept angles about the plane normal
  are camera-invariant WITHOUT a camera pose. Measured on 16 October 2026 on the development clip trial_2: hand
  height at rest varies by 1-2 mm (sd) while the head turns through 9 degrees of tilt; the fit costs 9 ms per frame on CPU.
* ``HandLandmarks`` — the three functional points per hand in 3-D (thumb MCP, index MCP, trapezium proxy), when the
  tracker lifted them; they give the index direction and the thumb-index bisector (grasp,
  reference) as same-frame vectors, again pose-free.

``PlaneGeometry`` is the causal tracker over these records (the plane counterpart of ``state.RelativeGeometry``): per
identity the height and its least-squares rate over a bounded window; per hand-object pair the in-plane distance and the
signed angle swept about the normal; per hand the count of vertical-direction reversals in the window; per
candidate object the nearest other object's in-plane offset and height difference. Nothing here is a label,
an intention or a command; the readouts and the supervisor consume these numbers under feature layout ``v2``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from math import isfinite, atan2
from pathlib import Path

import numpy as np

from .contracts import Packet, Point, SIDES

SCENE_SCHEMA = "sg-scene/1"
LANDMARK_NAMES = ("thumb_mcp", "index_mcp", "trapezium")
# Optional further joints from the 21-joint tracker: the thumb and index fingertips give
# the hand aperture, the direct kinematic signature of closing (grasp) and opening (release). Absent in records made before
# this date; the aperture cue then reads "unavailable", never zero-as-measured.
EXTRA_LANDMARK_NAMES = ("thumb_tip", "index_tip")


@dataclass(frozen=True)
class SupportPlane:
    """``height(p) = normal . p + offset``, positive toward the camera; None-valued planes carry a reason."""
    normal: tuple[float, float, float] | None
    offset: float | None
    frame: str
    calibration_id: str
    acquired_s: float
    reason: str = ""
    inlier_fraction: float | None = None
    rmse_m: float | None = None
    tilt_deg: float | None = None

    def __post_init__(self):
        if not self.frame or not isfinite(self.acquired_s):
            raise ValueError("plane needs a frame and a finite acquisition time")
        if self.normal is None or self.offset is None:
            if not self.reason:
                raise ValueError("an unavailable plane needs a reason")
            if self.normal is not None or self.offset is not None:
                raise ValueError("normal and offset are available together or not at all")
            return
        n = np.asarray(self.normal, dtype=float)
        if n.shape != (3,) or not np.isfinite(n).all() or not isfinite(self.offset) or abs(np.linalg.norm(n) - 1.) > 1e-6:
            raise ValueError("plane normal must be a finite unit 3-vector with a finite offset")
        object.__setattr__(self, "normal", tuple(float(v) for v in n))

    @property
    def available(self):
        return self.normal is not None

    def height(self, xyz):
        return float(np.asarray(xyz, dtype=float) @ np.asarray(self.normal) + self.offset)

    def in_plane(self, vector):
        n = np.asarray(self.normal); v = np.asarray(vector, dtype=float)
        return v - (v @ n) * n


@dataclass(frozen=True)
class HandLandmarks:
    side: str
    points: dict  # name -> (x, y, z) or None
    frame: str
    calibration_id: str
    acquired_s: float
    definition: str = "three_functional_points_lifted"
    extra: dict = field(default_factory=dict)  # EXTRA_LANDMARK_NAMES -> (x, y, z) or None; empty when the record predates them

    def __post_init__(self):
        if self.side not in SIDES or set(self.points) != set(LANDMARK_NAMES) or not self.frame or not isfinite(self.acquired_s):
            raise ValueError("landmarks need a side, the three named points, a frame and a finite time")
        if not set(self.extra) <= set(EXTRA_LANDMARK_NAMES):
            raise ValueError(f"extra landmarks must be among {EXTRA_LANDMARK_NAMES}")
        for attribute in ("points", "extra"):
            fixed = {}
            for name, xyz in getattr(self, attribute).items():
                if xyz is None:
                    fixed[name] = None
                    continue
                v = tuple(float(x) for x in xyz)
                if len(v) != 3 or not all(isfinite(x) for x in v):
                    raise ValueError("landmark must be finite xyz or None")
                fixed[name] = v
            object.__setattr__(self, attribute, fixed)

    @property
    def complete(self):
        return all(v is not None for v in self.points.values())

    @property
    def aperture_m(self):
        """Thumb-tip to index-tip distance in metres, or None when either tip is absent (a same-frame length: pose-free)."""
        a, b = self.extra.get("thumb_tip"), self.extra.get("index_tip")
        if a is None or b is None:
            return None
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    @property
    def index_curl(self):
        """Index-tip to index-MCP distance over index-MCP to trapezium distance: near or above 1 when extended, smaller when flexed."""
        tip, mcp, base = self.extra.get("index_tip"), self.points.get("index_mcp"), self.points.get("trapezium")
        if tip is None or mcp is None or base is None:
            return None
        d = float(np.linalg.norm(np.asarray(mcp) - np.asarray(base)))
        return None if d < 1e-6 else float(np.linalg.norm(np.asarray(tip) - np.asarray(mcp))) / d


@dataclass(frozen=True)
class Scene:
    packet_id: str
    plane: SupportPlane | None = None
    landmarks: tuple[HandLandmarks, ...] = ()
    schema: str = SCENE_SCHEMA
    source: str = ""

    def __post_init__(self):
        if self.schema != SCENE_SCHEMA or not self.packet_id:
            raise ValueError("unknown scene schema or missing packet identity")
        sides = [l.side for l in self.landmarks]
        if len(set(sides)) != len(sides):
            raise ValueError("one landmark record per hand")

    def landmarks_for(self, side):
        for l in self.landmarks:
            if l.side == side:
                return l
        return None


def scene_to_row(scene: Scene) -> dict:
    return dict(schema=scene.schema, packet_id=scene.packet_id, source=scene.source,
                plane=None if scene.plane is None else asdict(scene.plane),
                landmarks=[asdict(l) for l in scene.landmarks])


def scene_from_row(row: dict) -> Scene:
    plane = None if row.get("plane") is None else SupportPlane(**{**row["plane"], "normal": None if row["plane"]["normal"] is None else tuple(row["plane"]["normal"])})
    landmarks = tuple(HandLandmarks(**{**l, "extra": l.get("extra") or {}}) for l in row.get("landmarks", []))
    return Scene(row["packet_id"], plane, landmarks, row.get("schema", SCENE_SCHEMA), row.get("source", ""))


def write_scenes(path, scenes) -> int:
    rows = [json.dumps(scene_to_row(s), sort_keys=True, separators=(",", ":")) for s in scenes]
    Path(path).write_text("".join(r + "\n" for r in rows))
    return len(rows)


def read_scenes(path) -> dict:
    """``packet_id -> Scene``; a scene file is consumed by the replay only through this reader."""
    out = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            scene = scene_from_row(json.loads(line))
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        if scene.packet_id in out:
            raise ValueError(f"{path}:{number}: duplicate scene for packet {scene.packet_id}")
        out[scene.packet_id] = scene
    return out


def plane_from_fit(plane, frame, calibration_id, acquired_s) -> SupportPlane:
    """``tableplane.Plane`` (or None) -> ``SupportPlane``."""
    if plane is None:
        return SupportPlane(None, None, frame, calibration_id, acquired_s, reason="table_plane_not_found")
    return SupportPlane(tuple(plane.normal), float(plane.offset), frame, calibration_id, acquired_s,
                        inlier_fraction=float(plane.inlier_fraction), rmse_m=float(plane.rmse_m), tilt_deg=float(plane.tilt_deg))


# ----------------------------------------------------------------------------------------------
# Causal geometry on the plane anchor
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class HeightTrack:
    identity: str
    height_m: float | None
    rate_mps: float | None   # d(height)/dt over the chained past window; positive = rising
    reason: str


@dataclass(frozen=True)
class PairTrack:
    hand: str
    object_id: str
    inplane_distance_m: float | None
    sweep_rate_rad_s: float | None   # signed rate of the angle between the in-plane hand-object vector and a same-frame reference
    sweep_window_rad: float | None   # sum of |angle change| over the window
    reason: str
    reference: str = ""              # the reference vector's end point: the other hand, else the nearest other object


@dataclass(frozen=True)
class OverTrack:
    """A candidate object relative to the nearest OTHER object: in-plane offset and signed height difference."""
    object_id: str
    other_id: str | None
    inplane_distance_m: float | None
    height_difference_m: float | None  # height(object) - height(other); positive = candidate above the other
    reason: str


@dataclass(frozen=True)
class ApertureTrack:
    side: str
    aperture_m: float | None
    rate_mps: float | None    # d(aperture)/dt over the chained window; negative = closing
    index_curl: float | None
    reason: str


@dataclass(frozen=True)
class SceneCues:
    """Everything ``inference.encode`` reads under layouts v2 and v3 for one tick."""
    plane_available: bool
    heights: dict            # identity -> HeightTrack
    pairs: dict              # (hand, object) -> PairTrack
    reversals: dict          # hand -> int (vertical-direction reversals in the window) or None
    over: dict               # object -> OverTrack
    landmarks: dict          # side -> HandLandmarks or None
    reason: str = ""
    apertures: dict = field(default_factory=dict)  # side -> ApertureTrack (layout v3; independent of the plane)


def _chain_rate(chain, window):
    """Least-squares slope over the last ``window+1`` (value, time) points, or a secant for two."""
    pts = chain[-(window + 1):]
    if len(pts) < 2:
        return None
    if len(pts) == 2:
        (v0, t0, _), (v1, t1, _) = pts
        return (v1 - v0) / (t1 - t0)
    ts = np.array([q[1] for q in pts]); vs = np.array([q[0] for q in pts])
    tc = ts - ts.mean()
    return float(np.dot(tc, vs - vs.mean()) / np.dot(tc, tc))


class PlaneGeometry:
    """Causal, bounded-gap tracker of plane-anchored quantities; one instance per session (like ``RelativeGeometry``)."""

    def __init__(self, max_gap_s=0.25, window=8, reversal_window=8, reversal_deadband_mps=0.1):
        """``reversal_deadband_mps``: a vertical secant counts toward a direction reversal only above this speed; the
        hand's height noise at rest (1-4 mm per tick at 15 Hz, measured 16 Oct 2026) gives secants of a few cm/s that
        flip sign every tick, so a dead band well above it is needed (declared, untuned; a shake moves at 0.3-1 m/s)."""
        if type(window) is not int or window < 1 or type(reversal_window) is not int or reversal_window < 1:
            raise ValueError("windows must be positive integers")
        if not isfinite(reversal_deadband_mps) or reversal_deadband_mps < 0:
            raise ValueError("reversal dead band must be a finite nonnegative speed")
        self.max_gap_s, self.window, self.reversal_window, self.deadband = max_gap_s, window, reversal_window, reversal_deadband_mps
        self.heights: dict[str, list] = {}          # identity -> [(height, t, calibration)]
        self.vectors: dict[tuple, list] = {}        # (hand, object) -> [(in-plane vector, t, calibration)]
        self.rates: dict[str, list] = {}            # hand -> [(secant height rate, t)]
        self.apertures: dict[str, list] = {}        # hand -> [(aperture, t, calibration)]

    def _chain(self, store, key, value, t, cal):
        chain = store.get(key, [])
        if chain and (t <= chain[-1][1]):
            return chain  # a delayed old observation cannot overwrite current history
        if chain and (t - chain[-1][1] > self.max_gap_s or cal != chain[-1][2]):
            chain = []
        chain = (chain + [(value, t, cal)])[-(max(self.window, self.reversal_window) + 2):]
        store[key] = chain
        return chain

    def _aperture(self, packet, landmarks):
        """Aperture (thumb tip to index tip) and its causal rate per hand; a same-frame length, so it needs no plane."""
        out = {}
        for hand in packet.hands:
            lm = landmarks.get(hand.identity)
            aperture = None if lm is None else lm.aperture_m
            if aperture is None or lm.acquired_s != hand.stamp.acquired_s or lm.frame != hand.frame:
                out[hand.identity] = ApertureTrack(hand.identity, None, None, None, "fingertips_unavailable"); continue
            chain = self._chain(self.apertures, hand.identity, aperture, hand.stamp.acquired_s, lm.calibration_id)
            out[hand.identity] = ApertureTrack(hand.identity, aperture, _chain_rate(chain, self.window), lm.index_curl, "fingertip_aperture")
        return out

    def update(self, packet: Packet, scene: Scene | None) -> SceneCues:
        landmarks = {side: (scene.landmarks_for(side) if scene is not None else None) for side in SIDES}
        apertures = self._aperture(packet, landmarks)
        plane = scene.plane if scene is not None else None
        if plane is None or not plane.available:
            reason = "no_scene_record" if scene is None else (plane.reason if plane is not None else "no_plane_record")
            # Chains are not extended: a tick without a plane breaks the causal window, as a gap would.
            return SceneCues(False, {}, {}, {s: None for s in SIDES}, {}, landmarks, reason, apertures)
        records = [p for p in packet.hands + packet.objects]
        heights = {}
        for p in records:
            if p.xyz is None or p.frame != plane.frame or p.stamp.acquired_s != plane.acquired_s or p.calibration_id != plane.calibration_id:
                heights[p.identity] = HeightTrack(p.identity, None, None, "point_unavailable_or_not_same_frame_time")
                continue
            h = plane.height(p.xyz)
            chain = self._chain(self.heights, p.identity, h, p.stamp.acquired_s, p.calibration_id)
            rate = _chain_rate(chain, self.window)
            heights[p.identity] = HeightTrack(p.identity, h, rate, "plane_anchored_height" + ("" if rate is not None else "_no_past_pair"))
        reversals = {}
        for hand in packet.hands:
            track = heights.get(hand.identity)
            if track is None or track.height_m is None:
                reversals[hand.identity] = None
                continue
            chain = self.heights[hand.identity]
            if len(chain) >= 2 and chain[-1][1] > chain[-2][1]:
                secant = (chain[-1][0] - chain[-2][0]) / (chain[-1][1] - chain[-2][1])
                self.rates[hand.identity] = (self.rates.get(hand.identity, []) + [(secant, chain[-1][1])])[-(self.reversal_window + 1):]
            rates = [r for r, _ in self.rates.get(hand.identity, []) if abs(r) > self.deadband]
            reversals[hand.identity] = sum(1 for a, b in zip(rates, rates[1:]) if a * b < 0)
        pairs = {}
        for hand in packet.hands:
            for obj in packet.objects:
                key = (hand.identity, obj.identity)
                if heights[hand.identity].height_m is None or heights[obj.identity].height_m is None:
                    pairs[key] = PairTrack(*key, None, None, None, "point_unavailable"); continue
                v = plane.in_plane(np.asarray(hand.xyz) - np.asarray(obj.xyz))
                d = float(np.linalg.norm(v))
                # The angle of v about the normal is NOT comparable across ticks when the camera rotates about the normal
                # (each tick's vector lives in its own camera frame). The angle between v and a SAME-FRAME in-plane reference
                # vector from the object is: the reference is the other hand when observed, else the nearest other object.
                others = [p for p in packet.hands if p.identity != hand.identity and heights.get(p.identity) is not None and heights[p.identity].height_m is not None]
                if not others:
                    others = sorted([p for p in packet.objects if p.identity != obj.identity and heights.get(p.identity) is not None and heights[p.identity].height_m is not None],
                                    key=lambda p: (float(np.linalg.norm(plane.in_plane(np.asarray(p.xyz) - np.asarray(obj.xyz)))), p.identity))
                rate, window_sum, reference = None, None, ""
                if others:
                    ref = plane.in_plane(np.asarray(others[0].xyz) - np.asarray(obj.xyz))
                    reference = others[0].identity
                    if np.linalg.norm(v) > 1e-6 and np.linalg.norm(ref) > 1e-6:
                        angle = atan2(float(np.cross(ref, v) @ np.asarray(plane.normal)), float(ref @ v))
                        chain = self._chain(self.vectors, key, (angle, reference), hand.stamp.acquired_s, hand.calibration_id)
                        steps = []
                        for (a, ta, _), (b, tb, _) in zip(chain, chain[1:]):
                            if a[1] == b[1] and tb > ta:  # same reference on both ticks
                                delta = (b[0] - a[0] + np.pi) % (2 * np.pi) - np.pi
                                steps.append((float(delta), tb - ta))
                        if steps:
                            rate = steps[-1][0] / steps[-1][1]
                            window_sum = float(sum(abs(a) for a, _ in steps[-self.window:]))
                pairs[key] = PairTrack(*key, d, rate, window_sum, "plane_anchored_pair" + ("" if reference else "_no_inplane_reference"), reference)
        over = {}
        objects = [o for o in packet.objects if heights.get(o.identity) is not None and heights[o.identity].height_m is not None]
        for o in objects:
            others = [q for q in objects if q.identity != o.identity]
            if not others:
                over[o.identity] = OverTrack(o.identity, None, None, None, "no_other_object"); continue
            q = min(others, key=lambda q: (float(np.linalg.norm(plane.in_plane(np.asarray(o.xyz) - np.asarray(q.xyz)))), q.identity))
            over[o.identity] = OverTrack(o.identity, q.identity, float(np.linalg.norm(plane.in_plane(np.asarray(o.xyz) - np.asarray(q.xyz)))),
                                         heights[o.identity].height_m - heights[q.identity].height_m, "nearest_other_object")
        return SceneCues(True, heights, pairs, reversals, over, landmarks, "", apertures)


def landmark_alignments(landmarks: HandLandmarks | None, hand_xyz, object_xyz):
    """(index alignment, grasp-bisector alignment, valid): cosines between a same-frame hand direction and hand->object.

    Index direction = index MCP - trapezium; grasp direction = mean of the thumb and index vectors from
    the trapezium. Both are frame-invariant because every vector lives in one frame at one time.
    """
    if landmarks is None or not landmarks.complete or hand_xyz is None or object_xyz is None:
        return 0., 0., False
    p = landmarks.points
    base = np.asarray(p["trapezium"]); idx = np.asarray(p["index_mcp"]) - base; thumb = np.asarray(p["thumb_mcp"]) - base
    to_object = np.asarray(object_xyz) - np.asarray(hand_xyz)
    grasp = (idx + thumb) / 2.
    def cosine(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 1e-6 and nb > 1e-6 else 0.
    return cosine(idx, to_object), cosine(grasp, to_object), True
